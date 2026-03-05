# Running Palma OCR in Docker

## Build the image

From the project root (where `Dockerfile` is):

```bash
docker build -t palma .
```

**With PaddleOCR** (better quality, larger image): edit `Dockerfile`, uncomment the line  
`RUN pip install --no-cache-dir paddlepaddle paddleocr`, then rebuild.

---

## Run OCR on your PDFs

Mount your `input_pdfs` and `output_text` folders so the container reads PDFs from your machine and writes results back.

**Linux / macOS (Git Bash):**
```bash
docker run --rm -v "$(pwd)/input_pdfs:/app/input_pdfs" -v "$(pwd)/output_text:/app/output_text" palma
```

**Windows (PowerShell):**
```powershell
docker run --rm -v "${PWD}/input_pdfs:/app/input_pdfs" -v "${PWD}/output_text:/app/output_text" palma
```

**Windows (CMD):**
```cmd
docker run --rm -v "%cd%/input_pdfs:/app/input_pdfs" -v "%cd%/output_text:/app/output_text" palma
```

- Put PDFs in `input_pdfs/` before running.
- Text files appear in `output_text/` after the run.
- `--rm` removes the container when it exits.

---

## Optional environment variables

Pass with `-e`:

```bash
docker run --rm \
  -v "$(pwd)/input_pdfs:/app/input_pdfs" \
  -v "$(pwd)/output_text:/app/output_text" \
  -e PALMA_DPI=4 \
  -e PALMA_OCR=tesseract \
  -e PALMA_FAST=1 \
  palma
```

Examples:
- `PALMA_DPI=4` – higher resolution (slower).
- `PALMA_OCR=tesseract` – use Tesseract instead of Paddle (if Paddle is installed).
- `PALMA_FAST=0` – try multiple variants per page (slower, sometimes better).

---

## Run only OCR or only polish

**OCR only (no polish):**
```bash
docker run --rm -v "$(pwd)/input_pdfs:/app/input_pdfs" -v "$(pwd)/output_text:/app/output_text" palma python engine.py
```

**Polish only** (format existing `.txt` in `output_text`):
```bash
docker run --rm -v "$(pwd)/output_text:/app/output_text" palma python polish.py output_text
```

(Use `output_text` as the first argument so `polish.py` reads/writes that folder.)
