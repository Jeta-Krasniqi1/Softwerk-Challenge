# Palma OCR: Cold Case Archive Digitization

This repository contains a robust, offline OCR pipeline designed for the Softwerk x LNUAIS Challenge. It is optimized for historical documents containing noise, handwriting, and faded text.

## Prerequisites

- Docker installed on your machine.

## Build Instructions

Build the Docker image from the project root:

```bash
docker build -t palma .
```

**Note:** To enable higher-quality PaddleOCR, ensure the relevant lines are uncommented in the Dockerfile before building.

## How to Run (The Jury Test)

The application is designed to process all PDFs in a local folder and save results to an output folder.

### 1. Prepare your folders

Ensure you have an `input_pdfs` folder containing your documents and an empty `output_text` folder.

### 2. Run the Container

**Linux / macOS:**

```bash
docker run --rm -v "$(pwd)/input_pdfs:/app/input_pdfs" -v "$(pwd)/output_text:/app/output_text" palma
```

**Windows (PowerShell):**

```powershell
docker run --rm -v "${PWD}/input_pdfs:/app/input_pdfs" -v "${PWD}/output_text:/app/output_text" palma
```

## Features & Configuration

The pipeline automatically performs the following steps:

1. **Deskewing & Cleaning:** Fixes tilted scans and removes noise.
2. **Stroke Removal:** Strips underlines to improve character recognition.
3. **Polish Pass:** Automatically formats the final `.txt` files with consistent page headers.

## Environment Variables

You can tune the performance using `-e`:

| Variable       | Example   | Description                                                                 |
|----------------|-----------|-----------------------------------------------------------------------------|
| `PALMA_OCR`    | `paddle`  | Use PaddleOCR instead of Tesseract.                                         |
| `PALMA_DPI`    | `3`       | Set higher rendering resolution.                                           |
| `PALMA_FAST`   | `0`       | Enable multiple pre-processing passes for maximum accuracy (slower).       |

Example:

```bash
docker run --rm -e PALMA_OCR=paddle -e PALMA_DPI=3 -e PALMA_FAST=0 -v "$(pwd)/input_pdfs:/app/input_pdfs" -v "$(pwd)/output_text:/app/output_text" palma
```
