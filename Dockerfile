# Palma OCR: PDF → text (PaddleOCR / Tesseract). Run in container.
FROM python:3.11-slim

# Tesseract (optional fallback when Paddle not used)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-swe \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Optional: PaddleOCR (better quality, heavier). Uncomment to use.
# RUN pip install --no-cache-dir paddlepaddle paddleocr

COPY engine.py polish.py ./

# Default: create dirs and run OCR, then polish
RUN mkdir -p input_pdfs output_text

ENV PALMA_DPI=2 \
    PALMA_FAST=1 \
    PALMA_PREPROCESS=mixed

# Run OCR then polish. Override with: docker run ... palma python polish.py
CMD ["sh", "-c", "python engine.py && python polish.py"]
