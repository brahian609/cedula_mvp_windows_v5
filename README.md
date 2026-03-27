# MVP Windows V5 - Renombrar PDFs por número de cédula colombiana

Esta versión está optimizada para tu caso real:

- la **cédula siempre está en la primera página**
- la cédula viene **rotada 180°**
- el PDF puede contener **otros documentos escaneados**, pero el proceso se enfoca en detectar la cédula
- el objetivo es leer **solo el número de cédula** y renombrar el PDF
- si no hay alta confianza, el archivo se marca como **PENDIENTE**

## Qué cambió en esta versión

- procesa solo la **primera página**
- gira la página **180° una sola vez**
- divide la página en dos zonas para detectar más rápido las tarjetas
- intenta identificar el **frente** de la cédula
- hace OCR únicamente sobre la **zona del número**
- prioriza formatos como:
  - `1.113.520.688`
  - `11.358.456`
- si la confianza es baja, deja el PDF como:
  - `PENDIENTE_nombreoriginal.pdf`

## Requisitos

1. Python 3.11+
2. Tesseract OCR para Windows
3. Idioma español de Tesseract (`spa.traineddata`)

## Instalación rápida

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Rutas esperadas por defecto

- `C:\Program Files\Tesseract-OCR\tesseract.exe`
- `C:\Program Files\Tesseract-OCR\tessdata\spa.traineddata`

Si las tuyas son distintas, cámbialas en la barra lateral de la app.

## Uso

1. Ejecuta la app
2. Valida Tesseract en la barra lateral
3. Carga varios PDFs
4. Pulsa **Procesar lote**
5. Descarga el ZIP con:
   - PDFs renombrados
   - `resultado.csv`

## Salida

- Si detecta con alta confianza:
  - `1.113.520.688.pdf`
- Si no está seguro:
  - `PENDIENTE_20260319220425523.pdf`

## Recomendación

Si al probar con tus lotes reales ves que todavía algunos archivos quedan en `PENDIENTE`, comparte 2 o 3 ejemplos adicionales del mismo formato para seguir afinando las regiones del número.
