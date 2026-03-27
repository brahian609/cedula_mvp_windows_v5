import io
import os
import re
import csv
import zipfile
import tempfile
import shutil
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Dict

import cv2
import fitz
import numpy as np
import pandas as pd
import pytesseract
import streamlit as st
from PIL import Image

DEFAULT_TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
DEFAULT_TESSDATA_DIR = r"C:\Program Files\Tesseract-OCR\tessdata"
RE_NUMBER_DOTTED = re.compile(r"\b\d{1,3}(?:\.\d{3}){1,3}\b")
RE_NUMBER_PLAIN = re.compile(r"\b\d{6,12}\b")
RE_DIGIT_BLOCK = re.compile(r"(?:\d[\s\.\-]{0,2}){8,14}\d")


@dataclass
class DetectionResult:
    original_name: str
    final_name: str
    status: str
    detected_number: str
    confidence: float
    reason: str


def configure_tesseract(tesseract_cmd: str, tessdata_dir: str) -> Tuple[bool, str]:
    try:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        os.environ["TESSDATA_PREFIX"] = tessdata_dir
        langs = pytesseract.get_languages(config=f'--tessdata-dir "{tessdata_dir}"')
        if "spa" not in langs:
            return False, f"No se encontró el idioma 'spa' en {tessdata_dir}."
        version = str(pytesseract.get_tesseract_version())
        return True, f"Tesseract OK. Versión: {version}. Idiomas: {', '.join(langs)}"
    except Exception as e:
        return False, str(e)


def render_first_page(pdf_bytes: bytes, zoom: float = 2.0) -> np.ndarray:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[0]
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def render_page(doc: fitz.Document, page_index: int, zoom: float = 1.8) -> np.ndarray:
    page = doc[page_index]
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1).reshape(-1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    max_w = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    max_h = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    dst = np.array([[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]], dtype="float32")
    m = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, m, (max_w, max_h))
    if warped.shape[0] > warped.shape[1]:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    return warped


def detect_card_in_region(region_bgr: np.ndarray, min_area_ratio: float = 0.012) -> Optional[np.ndarray]:
    gray = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY)
    # Detectar todo lo que no sea fondo blanco. Esta estrategia funciona mejor
    # para los PDFs reales que contienen la cédula sobre una hoja casi vacía.
    mask = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)[1]
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = gray.shape
    region_area = h * w
    best = None
    best_area = 0.0

    for c in contours:
        area = cv2.contourArea(c)
        if area < region_area * min_area_ratio:
            continue

        rect = cv2.minAreaRect(c)
        (cx, cy), (rw, rh), angle = rect
        if rw <= 0 or rh <= 0:
            continue
        aspect = max(rw, rh) / max(1.0, min(rw, rh))
        if not (1.35 <= aspect <= 2.20):
            continue

        box = cv2.boxPoints(rect).astype("float32")
        warped = four_point_transform(region_bgr, box)
        hh, ww = warped.shape[:2]
        if hh < 120 or ww < 180:
            continue
        if area > best_area:
            best_area = area
            best = warped

    return best


def detect_cards(page_bgr: np.ndarray) -> List[np.ndarray]:
    # Los PDFs reales vienen con la cédula al revés, así que giramos la página
    # completa una sola vez y evitamos OSD / múltiples rotaciones.
    page_bgr = cv2.rotate(page_bgr, cv2.ROTATE_180)
    h, w = page_bgr.shape[:2]
    regions = [page_bgr]
    regions.extend(
        [
            page_bgr[0 : int(h * 0.60), :],
            page_bgr[int(h * 0.40) : h, :],
            page_bgr[:, 0 : int(w * 0.70)],
            page_bgr[:, int(w * 0.30) : w],
        ]
    )
    cards: List[np.ndarray] = []
    for idx, region in enumerate(regions):
        min_area_ratio = 0.005 if idx == 0 else 0.010
        card = detect_card_in_region(region, min_area_ratio=min_area_ratio)
        if card is not None:
            cards.append(card)
    if not cards:
        for region in regions:
            card = detect_card_in_region(region, min_area_ratio=0.0035)
            if card is not None:
                cards.append(card)
    if cards:
        return cards

    gray = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2GRAY)
    mask = cv2.threshold(gray, 242, 255, cv2.THRESH_BINARY_INV)[1]
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8), iterations=2)
    mask = cv2.dilate(mask, np.ones((11, 11), np.uint8), iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    page_area = page_bgr.shape[0] * page_bgr.shape[1]
    fallback_cards: List[Tuple[float, np.ndarray]] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < page_area * 0.004:
            continue
        x, y, cw, ch = cv2.boundingRect(c)
        aspect = cw / max(1.0, float(ch))
        if not (1.20 <= aspect <= 2.30):
            continue
        crop = page_bgr[y : y + ch, x : x + cw]
        if crop.size == 0:
            continue
        fallback_cards.append((area, crop))

    fallback_cards.sort(key=lambda t: t[0], reverse=True)
    return [c for _, c in fallback_cards[:4]]


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.upper()
    replacements = {
        "Á": "A", "É": "E", "Í": "I", "Ó": "O", "Ú": "U",
        "\n": " ", "\r": " ", "|": " ", ",": "."
    }
    for a, b in replacements.items():
        text = text.replace(a, b)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_plain_digits(raw: str) -> Optional[str]:
    digits = re.sub(r"\D", "", raw)
    if len(digits) == 11:
        # En RUT puede aparecer NIT + DV pegado; priorizamos el NIT base.
        digits = digits[:10]
    if 8 <= len(digits) <= 10:
        return digits
    return None


def has_dian_markers(text: str) -> bool:
    t = normalize_text(text)
    t_compact = t.replace(" ", "")
    has_dian = "DIAN" in t_compact
    has_rut_form = "FORMULARIO DEL REGISTRO UNICO TRIBUTARIO" in t or "REGISTRO UNICO TRIBUTARIO" in t
    if not has_dian and not has_rut_form:
        return False
    return (
        "NIT" in t
        or "REGISTRO UNICO TRIBUTARIO" in t
        or "IDENTIFICACION TRIBUTARIA" in t
    )


def extract_nit_candidates_from_text(text: str) -> List[str]:
    t = normalize_text(text)
    if not has_dian_markers(t):
        return []

    candidates: List[str] = []
    patterns = [
        r"NIT\)?[^0-9]{0,35}((?:\d[\s\.\-]{0,2}){8,12}\d)",
        r"IDENTIFICACION TRIBUTARIA[^0-9]{0,45}((?:\d[\s\.\-]{0,2}){8,12}\d)",
        r"CEDULA DE CIUDADANIA[^0-9]{0,30}(\d{8,11})",
        r"NUMERO DE IDENTI[^0-9]{0,35}(\d{8,11})",
    ]
    for pat in patterns:
        for m in re.finditer(pat, t):
            digit_candidate = normalize_plain_digits(m.group(1))
            if digit_candidate:
                candidates.append(digit_candidate)

    if not candidates:
        # Fallback: buscar secuencias de digitos cerca del bloque NIT.
        idx = t.find("NIT")
        if idx >= 0:
            window = t[max(0, idx - 40) : idx + 180]
            for m in RE_DIGIT_BLOCK.findall(window):
                digit_candidate = normalize_plain_digits(m)
                if digit_candidate:
                    candidates.append(digit_candidate)

    unique: List[str] = []
    seen = set()
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


def score_nit_candidate(candidate: str, occurrences: int) -> float:
    score = 0.72 + min(0.10, 0.03 * max(0, occurrences - 1))
    if len(candidate) == 10:
        score += 0.12
    if not candidate.startswith("0"):
        score += 0.06
    return min(score, 0.97)


def extract_number_from_dian(pdf_bytes: bytes, tessdata_dir: str) -> Tuple[Optional[str], float, str]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total = len(doc)
    priority = [1, 2]  # paginas 2 y 3 (indice base 0)
    page_order = [p for p in priority if p < total]
    page_order.extend([i for i in range(total) if i not in page_order])

    best_num: Optional[str] = None
    best_conf = 0.0
    best_reason = "No se detecto DIAN/NIT"

    for page_idx in page_order:
        is_priority_page = page_idx in priority
        page_bgr = render_page(doc, page_idx, zoom=2.0 if is_priority_page else 1.3)
        local_candidates: List[str] = []
        for oriented in (cv2.rotate(page_bgr, cv2.ROTATE_180), page_bgr):
            gray = cv2.cvtColor(oriented, cv2.COLOR_BGR2GRAY)
            if is_priority_page:
                pre = cv2.resize(gray, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
                pre = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(pre)
                quick = ocr_text(pre, tessdata_dir=tessdata_dir, psm=6, whitelist=None, timeout=12)
            else:
                quick = ocr_text(gray, tessdata_dir=tessdata_dir, psm=6, whitelist=None, timeout=1)
            local_candidates.extend(extract_nit_candidates_from_text(quick))
            if local_candidates:
                break

            if not local_candidates and has_dian_markers(quick):
                up = cv2.resize(gray, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(up)
                dense = ocr_text(clahe, tessdata_dir=tessdata_dir, psm=11, whitelist=None, timeout=4 if is_priority_page else 3)
                local_candidates.extend(extract_nit_candidates_from_text(dense))
                if local_candidates:
                    break

        if not local_candidates:
            continue

        counts = Counter(local_candidates)
        page_best = ""
        page_score = -1.0
        page_freq = 0
        for cand, freq in counts.items():
            cand_score = score_nit_candidate(cand, freq)
            if cand_score > page_score:
                page_best = cand
                page_score = cand_score
                page_freq = freq

        if page_score > best_conf:
            best_num = page_best
            best_conf = page_score
            best_reason = f"NIT DIAN detectado en pagina {page_idx + 1} ({page_freq} vez/veces)"

        if best_conf >= 0.90:
            break

    return best_num, best_conf, best_reason


def build_variants(gray: np.ndarray) -> Dict[str, np.ndarray]:
    up = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    denoise = cv2.bilateralFilter(up, 7, 60, 60)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(denoise)
    otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    blur = cv2.GaussianBlur(clahe, (3, 3), 0)
    adapt = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 8)
    return {
        "gray": up,
        "clahe": clahe,
        "otsu": otsu,
        "adapt": adapt,
    }


def ocr_text(image: np.ndarray, tessdata_dir: str, psm: int = 6, whitelist: Optional[str] = None, timeout: int = 8) -> str:
    cfg = f"--oem 3 --psm {psm}"
    if whitelist:
        cfg += f' -c tessedit_char_whitelist={whitelist}'
    try:
        return pytesseract.image_to_string(image, lang="spa+eng", config=cfg, timeout=timeout)
    except Exception:
        return ""


def get_number_rois(card_bgr: np.ndarray) -> List[np.ndarray]:
    h, w = card_bgr.shape[:2]
    # Regiones enfocadas en el area superior izquierda del frente.
    boxes = [
        (0.00, 0.00, 0.72, 0.40),
    ]
    rois: List[np.ndarray] = []
    for x1, y1, x2, y2 in boxes:
        xa, ya, xb, yb = int(w * x1), int(h * y1), int(w * x2), int(h * y2)
        crop = card_bgr[ya:yb, xa:xb]
        if crop.size:
            rois.append(crop)
    return rois


def normalize_candidate_number(raw: str) -> Optional[str]:
    text = normalize_text(raw)
    text = (
        text.replace("O", "0")
        .replace("I", "1")
        .replace("L", "1")
        .replace("S", "5")
        .replace("B", "8")
    )
    text = text.replace(" ", "")
    # Preferir formato colombiano con puntos de miles/millones.
    m = RE_NUMBER_DOTTED.search(text)
    if m:
        value = m.group(0)
        digits = value.replace(".", "")
        if 6 <= len(digits) <= 10:
            return value
    m = RE_NUMBER_PLAIN.search(text)
    if m:
        digits = m.group(0)
        if 6 <= len(digits) <= 10:
            return digits
    return None


def extract_number_from_front(card_bgr: np.ndarray, tessdata_dir: str) -> Tuple[Optional[str], float, str]:
    rois = get_number_rois(card_bgr)
    readings: List[str] = []

    for roi in rois:
        for rotated in (roi, cv2.rotate(roi, cv2.ROTATE_180)):
            gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
            variants = build_variants(gray)
            for variant_name in ("otsu",):
                variant = variants[variant_name]
                text = ocr_text(
                    variant,
                    tessdata_dir=tessdata_dir,
                    psm=7,
                    whitelist="0123456789.",
                    timeout=1,
                )
                candidate = normalize_candidate_number(text)
                if candidate:
                    readings.append(candidate)
                    continue
                broad_text = ocr_text(
                    variant,
                    tessdata_dir=tessdata_dir,
                    psm=7,
                    whitelist=None,
                    timeout=1,
                )
                broad_candidate = normalize_candidate_number(broad_text)
                if broad_candidate:
                    readings.append(broad_candidate)

    if not readings:
        gray = cv2.cvtColor(card_bgr, cv2.COLOR_BGR2GRAY)
        variants = build_variants(gray)
        for variant_name in ("otsu",):
            text = ocr_text(
                variants[variant_name],
                tessdata_dir=tessdata_dir,
                psm=11,
                whitelist=None,
                timeout=1,
            )
            candidate = normalize_candidate_number(text)
            if candidate:
                readings.append(candidate)
        for rotated in (card_bgr, cv2.rotate(card_bgr, cv2.ROTATE_180)):
            gray_full = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
            full = cv2.resize(gray_full, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
            text_full = ocr_text(
                full,
                tessdata_dir=tessdata_dir,
                psm=11,
                whitelist=None,
                timeout=8,
            )
            m_line = re.findall(r"M[-\s]?(\d{8,10})", normalize_text(text_full))
            for m in m_line:
                candidate = normalize_candidate_number(m)
                if candidate:
                    readings.append(candidate)
            for m in re.findall(r"\d{6,10}", text_full.replace(",", ".")):
                candidate = normalize_candidate_number(m)
                if candidate:
                    readings.append(candidate)

    if not readings:
        return None, 0.0, "No se detecto numero con formato valido"

    counter = Counter(readings)
    best = ""
    freq = 0
    best_score = -1.0
    for candidate, count in counter.items():
        digits = candidate.replace(".", "")
        score = float(count)
        if "." in candidate:
            score += 0.70
        if len(digits) >= 8:
            score += 0.40
        if len(digits) == 10:
            score += 0.40
        if digits and not digits.startswith("0"):
            score += 0.20
        if score > best_score:
            best_score = score
            best = candidate
            freq = count
    digits = best.replace(".", "")
    confidence = 0.70
    if "." in best:
        confidence += 0.12
    if 6 <= len(digits) <= 10:
        confidence += 0.06
    if freq >= 2:
        confidence += 0.10
    if freq >= 3:
        confidence += 0.04

    reason = f"Numero detectado {freq} vez/veces"
    return best, min(confidence, 0.99), reason


def analyze_pdf(pdf_bytes: bytes, original_name: str, tessdata_dir: str) -> DetectionResult:
    try:
        dian_number, dian_conf, dian_reason = extract_number_from_dian(pdf_bytes, tessdata_dir=tessdata_dir)
        if dian_number and dian_conf >= 0.86:
            ext = os.path.splitext(original_name)[1].lower() or ".pdf"
            return DetectionResult(
                original_name,
                f"{dian_number}{ext}",
                "OK",
                dian_number,
                dian_conf,
                f"Renombrado usando DIAN/NIT. {dian_reason}",
            )

        page = render_first_page(pdf_bytes)
        cards = detect_cards(page)

        best_number: Optional[str] = None
        best_conf = 0.0
        best_reason = "No se detecto numero con formato valido"

        if dian_number:
            best_number = dian_number
            best_conf = dian_conf
            best_reason = f"DIAN/NIT encontrado con confianza media. {dian_reason}"

        for card in cards[:4]:
            number, confidence, reason = extract_number_from_front(card, tessdata_dir=tessdata_dir)
            if number and confidence > best_conf:
                best_number = number
                best_conf = confidence
                best_reason = reason
            if best_conf >= 0.92:
                break

        if best_conf < 0.88:
            # Fallback final sobre la primera pagina completa para casos con
            # tarjetas muy pequenas o ruido fuerte.
            page_rot = cv2.rotate(page, cv2.ROTATE_180)
            gray_page = cv2.cvtColor(page_rot, cv2.COLOR_BGR2GRAY)
            gray_page = cv2.resize(gray_page, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
            text_page = ocr_text(gray_page, tessdata_dir=tessdata_dir, psm=11, whitelist=None, timeout=8)
            dotted = [normalize_candidate_number(x) for x in RE_NUMBER_DOTTED.findall(text_page.replace(",", "."))]
            dotted = [x for x in dotted if x]
            if dotted:
                page_counter = Counter(dotted)
                page_best, page_freq = page_counter.most_common(1)[0]
                page_conf = min(0.90 + min(0.06, page_freq * 0.02), 0.96)
                if page_conf > best_conf:
                    best_number = page_best
                    best_conf = page_conf
                    best_reason = f"Numero detectado en OCR de pagina completa ({page_freq} vez/veces)"

        if not best_number:
            return DetectionResult(original_name, f"PENDIENTE_{original_name}", "Pendiente", "", 0.0, best_reason)

        if best_conf < 0.88:
            return DetectionResult(
                original_name,
                f"PENDIENTE_{original_name}",
                "Pendiente",
                best_number,
                best_conf,
                f"Numero encontrado pero con confianza insuficiente. {best_reason}",
            )

        ext = os.path.splitext(original_name)[1].lower() or ".pdf"
        return DetectionResult(
            original_name,
            f"{best_number}{ext}",
            "OK",
            best_number,
            best_conf,
            f"Renombrado con alta confianza. {best_reason}",
        )
    except Exception as e:
        return DetectionResult(original_name, f"PENDIENTE_{original_name}", "Pendiente", "", 0.0, str(e))


def unique_name(name: str, used: set) -> str:
    if name not in used:
        used.add(name)
        return name
    base, ext = os.path.splitext(name)
    i = 2
    while f"{base}_{i}{ext}" in used:
        i += 1
    final = f"{base}_{i}{ext}"
    used.add(final)
    return final


def build_output_zip(uploaded_files, results: List[DetectionResult]) -> bytes:
    used_names = set()
    mapping = {r.original_name: r for r in results}

    temp_dir = tempfile.mkdtemp(prefix="cedulas_")
    try:
        out_pdf_dir = os.path.join(temp_dir, "pdfs")
        os.makedirs(out_pdf_dir, exist_ok=True)

        csv_path = os.path.join(temp_dir, "resultado.csv")
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["original_name", "final_name", "status", "detected_number", "confidence", "reason"])
            writer.writeheader()
            for result in results:
                writer.writerow(asdict(result))

        for uf in uploaded_files:
            result = mapping[uf.name]
            final_name = unique_name(result.final_name, used_names)
            with open(os.path.join(out_pdf_dir, final_name), "wb") as f:
                f.write(uf.getvalue())

        zip_bytes = io.BytesIO()
        with zipfile.ZipFile(zip_bytes, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(csv_path, arcname="resultado.csv")
            for file_name in os.listdir(out_pdf_dir):
                zf.write(os.path.join(out_pdf_dir, file_name), arcname=f"pdfs/{file_name}")
        zip_bytes.seek(0)
        return zip_bytes.getvalue()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def process_files(uploaded_files, tessdata_dir: str, use_parallel: bool) -> List[DetectionResult]:
    if not use_parallel or len(uploaded_files) <= 1:
        return [analyze_pdf(uf.getvalue(), uf.name, tessdata_dir=tessdata_dir) for uf in uploaded_files]

    results: List[DetectionResult] = []
    max_workers = min(4, max(1, os.cpu_count() or 1), len(uploaded_files))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(analyze_pdf, uf.getvalue(), uf.name, tessdata_dir): uf.name
            for uf in uploaded_files
        }
        progress = st.progress(0.0)
        done = 0
        for future in as_completed(futures):
            results.append(future.result())
            done += 1
            progress.progress(done / len(futures))
    results.sort(key=lambda r: r.original_name.lower())
    return results


st.set_page_config(page_title="Renombrar PDFs por cédula", layout="wide")
st.title("Renombrar PDFs por número de cédula colombiana")
st.caption("Version rapida y enfocada: prioriza DIAN + campo NIT (paginas 2, 3 u otras), y usa cedula en primera pagina como fallback. Si no hay alta confianza, marca PENDIENTE.")

with st.sidebar:
    st.subheader("Configuración Tesseract")
    tesseract_cmd = st.text_input("Ruta tesseract.exe", value=DEFAULT_TESSERACT_CMD)
    tessdata_dir = st.text_input("Ruta tessdata", value=DEFAULT_TESSDATA_DIR)
    use_parallel = st.checkbox("Procesar en paralelo", value=True)

    if st.button("Validar Tesseract"):
        ok, msg = configure_tesseract(tesseract_cmd, tessdata_dir)
        if ok:
            st.success(msg)
        else:
            st.error(msg)

ok, msg = configure_tesseract(tesseract_cmd, tessdata_dir)
if not ok:
    st.error(msg)
    st.stop()

uploaded_files = st.file_uploader("Selecciona uno o varios PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files and st.button("Procesar lote"):
    with st.spinner("Procesando PDFs..."):
        if use_parallel and len(uploaded_files) > 1:
            results = process_files(uploaded_files, tessdata_dir=tessdata_dir, use_parallel=True)
        else:
            progress = st.progress(0.0)
            results = []
            for i, uf in enumerate(uploaded_files, start=1):
                results.append(analyze_pdf(uf.getvalue(), uf.name, tessdata_dir=tessdata_dir))
                progress.progress(i / len(uploaded_files))

    df = pd.DataFrame([asdict(r) for r in results])
    st.dataframe(df, use_container_width=True)

    zip_bytes = build_output_zip(uploaded_files, results)
    st.download_button(
        "Descargar ZIP procesado",
        data=zip_bytes,
        file_name="cedulas_renombradas.zip",
        mime="application/zip",
    )

    ok_count = sum(1 for r in results if r.status == "OK")
    pending_count = len(results) - ok_count
    st.info(f"Procesados: {len(results)} | OK: {ok_count} | Pendiente: {pending_count}")





