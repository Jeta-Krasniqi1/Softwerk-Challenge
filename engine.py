"""
PDF → preprocess → OCR → text.
Uses Tesseract by default (reliable). PaddleOCR 3.x optional (can hit oneDNN bugs on some CPUs).
"""
import os
import gc
import tempfile
import warnings
import shutil

warnings.filterwarnings("ignore")
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "1"

import cv2
import numpy as np
import fitz
import re

# --- OCR: PaddleOCR first (better on complex layouts/handwriting), Tesseract fallback ---
_ocr_engine = None
_ocr_backend = None


def _init_tesseract():
    try:
        import pytesseract

        # On Windows, Tesseract is often installed but not on PATH.
        # Allow explicit override, then try common install locations.
        cmd = os.environ.get("TESSERACT_CMD", "").strip()
        if cmd:
            pytesseract.pytesseract.tesseract_cmd = cmd
        else:
            which = shutil.which("tesseract")
            if which:
                pytesseract.pytesseract.tesseract_cmd = which
            else:
                common = [
                    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                    os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
                ]
                for p in common:
                    if p and os.path.exists(p):
                        pytesseract.pytesseract.tesseract_cmd = p
                        break

        pytesseract.get_tesseract_version()
        return pytesseract, "tesseract"
    except Exception:
        return None, None


def _init_paddle():
    try:
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(lang="en", use_textline_orientation=True)
        # Smoke test: run on tiny image to catch oneDNN/runtime errors
        tiny = np.ones((50, 200, 3), dtype=np.uint8) * 255
        # NOTE: On Windows, NamedTemporaryFile keeps the file handle open (locked).
        # PaddleOCR reads the file path, so we must ensure the file is closed first.
        fd, tmp_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        try:
            cv2.imwrite(tmp_path, tiny)
            # PaddleOCR v2 uses ocr.ocr(); some newer builds expose predict(). Support both.
            if hasattr(ocr, "predict"):
                ocr.predict(tmp_path)
            else:
                ocr.ocr(tmp_path, cls=True)
        except Exception:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            return None, None
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
        return ocr, "paddle"
    except Exception as e:
        print("PaddleOCR not used:", e)
        return None, None


def _get_ocr():
    global _ocr_engine, _ocr_backend
    if _ocr_engine is not None:
        return _ocr_engine, _ocr_backend
    # PALMA_OCR=tesseract → use Tesseract first (faster on laptop). Else Paddle first.
    prefer = (os.environ.get("PALMA_OCR", "").strip() or "").lower()
    if prefer == "tesseract":
        _ocr_engine, _ocr_backend = _init_tesseract()
        if _ocr_engine is None:
            _ocr_engine, _ocr_backend = _init_paddle()
    else:
        _ocr_engine, _ocr_backend = _init_paddle()
        if _ocr_engine is None:
            _ocr_engine, _ocr_backend = _init_tesseract()
    if _ocr_engine is None:
        print("No OCR backend. Install Tesseract (https://github.com/UB-Mannheim/tesseract/wiki) then: pip install pytesseract")
        return None, None
    print("OCR backend:", _ocr_backend)
    return _ocr_engine, _ocr_backend


def _page_to_bgr(page, dpi_scale=2):
    """PDF page -> BGR numpy array."""
    mat = fitz.Matrix(dpi_scale, dpi_scale)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    h, w, n = pix.height, pix.width, pix.n
    arr = np.frombuffer(pix.samples, dtype=np.uint8)
    if n == 4:
        arr = arr.reshape(h, w, 4)[:, :, :3].copy()
    else:
        arr = arr.reshape(h, w, 3).copy()
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _score_text(text: str) -> float:
    """
    Heuristic OCR quality score.

    Higher is better. Designed to distinguish "real document text" from
    symbol soup / empty / extremely short outputs.
    """
    if not text:
        return 0.0
    t = text.strip()
    if not t:
        return 0.0

    # Basic counts
    alpha = sum(ch.isalnum() for ch in t)
    spaces = sum(ch.isspace() for ch in t)

    # Tokens that look like words (>= 2 letters, including accented letters)
    tokens = re.split(r"\s+", t)
    word_like = sum(1 for tok in tokens if sum(ch.isalpha() for ch in tok) >= 2)

    # "Weird" symbols (not alnum/space/common punctuation)
    common_punct = set(".,;:!?-()[]{}'\"/\\@#%&+*=<>_")
    weird = sum(
        1
        for ch in t
        if not (ch.isalnum() or ch.isspace() or ch in common_punct)
    )

    # Penalize long runs of non-alnum (often OCR garbage)
    non_alnum_runs = len(re.findall(r"[^0-9A-Za-zÅÄÖåäöÉÈÊËéèêëÀÂÇÎÏÔÛÜàâçîïôûü]{4,}", t))

    # Combine. (Coefficients are intentionally simple and stable.)
    score = (
        1.0 * alpha
        + 6.0 * word_like
        + 0.2 * spaces
        - 2.0 * weird
        - 15.0 * non_alnum_runs
    )
    # Guard against negative scores.
    return float(max(0.0, score))


def _deskew_bgr(bgr):
    """Attempt to deskew a page image. Returns original on failure/low skew."""
    try:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        coords = np.column_stack(np.where(bw > 0))
        if coords.shape[0] < 500:
            return bgr
        # coords are (row, col) = (y, x) → swap to (x, y) for minAreaRect.
        pts = coords[:, ::-1].astype(np.float32)
        angle = cv2.minAreaRect(pts)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        # Be a bit more aggressive: deskew even for small but consistent tilts.
        if abs(angle) < 0.1:
            return bgr
        h, w = bgr.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        return cv2.warpAffine(
            bgr,
            M,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
    except Exception:
        return bgr


def _preprocess_heavy_handwriting(bgr, remove_lines=True):
    """
    Stronger preprocessing for noisy / low-contrast / handwritten pages.
    Tries to preserve strokes while suppressing background texture.
    """
    # Deskew early: helps thresholding and OCR layout assumptions.
    bgr = _deskew_bgr(bgr)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # If the page is overall very light (faint pencil), gently boost darker strokes.
    mean_val = float(gray.mean())
    if mean_val > 165:
        # Gamma < 1.0 darkens mid-tones a bit and helps reveal light ink.
        gamma = 0.8
        inv_gamma = 1.0 / gamma
        table = np.array(
            [(i / 255.0) ** inv_gamma * 255.0 for i in range(256)],
            dtype=np.uint8,
        )
        gray = cv2.LUT(gray, table)

    # Optionally strip big solid black/redaction boxes before enhancing strokes.
    redact_env = os.environ.get("PALMA_REDACTION_MASK", "1").strip() not in (
        "0",
        "false",
        "False",
        "no",
        "NO",
    )
    if redact_env:
        box_mask = _mask_black_boxes(gray)
        if box_mask is not None:
            gray = cv2.inpaint(gray, box_mask, 3, cv2.INPAINT_TELEA)

    # Background/illumination normalization via black-hat (enhance dark strokes).
    h, w = gray.shape[:2]
    k = max(19, min(h, w) // 30)
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    enhanced = cv2.add(gray, blackhat)

    # Local contrast boost.
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(enhanced)

    # Denoise while keeping edges (handwriting strokes).
    enhanced = cv2.bilateralFilter(enhanced, d=7, sigmaColor=55, sigmaSpace=55)
    enhanced = cv2.medianBlur(enhanced, 3)

    # Optional underline/long-stroke removal.
    remove_lines_env = os.environ.get("PALMA_REMOVE_LINES", "1").strip() not in (
        "0",
        "false",
        "False",
        "no",
        "NO",
    )
    if remove_lines and remove_lines_env:
        enhanced = _remove_long_strokes(enhanced)

    # Adaptive threshold with larger window for textured/noisy paper.
    bw = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35,
        12,
    )

    # Clean small speckles; keep thin handwriting strokes.
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8), iterations=1)

    return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)


def _remove_long_strokes(gray):
    """
    Remove long pen/pencil strokes (e.g., underlines/strikethroughs).

    Strategy:
    - Detect long near-horizontal strokes via morphology on a binarized image.
    - Also detect very long near-horizontal segments via HoughLinesP (handles slightly slanted strokes).
    - Inpaint only the detected stroke pixels (keeps characters intact in most cases).
    """
    h, w = gray.shape[:2]
    if h < 50 or w < 200:
        return gray

    # Build a mask for dark strokes.
    bw = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15
    )

    # Morphology: isolate long horizontal structures (underlines, strikethroughs).
    k = max(30, w // 30)  # scale kernel with page width
    horiz = cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1))
    horiz_mask = cv2.morphologyEx(bw, cv2.MORPH_OPEN, horiz, iterations=1)

    # Hough: catch slightly slanted long strokes.
    edges = cv2.Canny(gray, 60, 180)
    line_mask = np.zeros_like(gray)
    min_len = int(w * 0.35)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=120, minLineLength=min_len, maxLineGap=18
    )
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0:
                continue
            angle = abs(np.degrees(np.arctan2(dy, dx)))
            if angle <= 18:  # near-horizontal
                cv2.line(line_mask, (x1, y1), (x2, y2), 255, 2)

    # Combine and slightly expand the mask so we remove the full stroke thickness.
    mask = cv2.bitwise_or(horiz_mask, line_mask)
    if cv2.countNonZero(mask) == 0:
        return gray
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

    # Inpaint the stroke pixels back to background.
    return cv2.inpaint(gray, mask, 3, cv2.INPAINT_TELEA)


def _mask_black_boxes(gray, min_area_ratio=0.01):
    """
    Detect large, very dark rectangular regions (typical redaction/black boxes).
    Returns a binary mask (uint8 0/255) or None if nothing significant is found.
    """
    h, w = gray.shape[:2]
    if h < 80 or w < 80:
        return None

    # Threshold darker content; we only care about big contiguous blobs.
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Close gaps so large filled areas become single blobs.
    kx = max(15, w // 80)
    ky = max(15, h // 80)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    page_area = float(h * w)
    mask = np.zeros_like(gray, dtype=np.uint8)
    found = 0
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        area = float(cw * ch)
        if area < min_area_ratio * page_area:
            continue
        # Check that the region is really very dark.
        roi = gray[y : y + ch, x : x + cw]
        if roi.size == 0:
            continue
        if roi.mean() > 70:
            continue
        cv2.rectangle(mask, (x, y), (x + cw, y + ch), 255, thickness=-1)
        found += 1

    if found == 0 or cv2.countNonZero(mask) == 0:
        return None

    # Slightly dilate so we fully cover the box edges.
    mask = cv2.dilate(
        mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1
    )
    return mask


def _preprocess(bgr, mode="mixed", remove_lines=True):
    """
    mode: "print" = strong binarization for clean print
          "mixed" = adaptive threshold for mixed content
          "light" = contrast only for fragile/artistic
    """
    # Optional deskew for lightly rotated pages (helps long typed letters).
    deskew_env = os.environ.get("PALMA_DESKEW", "1").strip() not in (
        "0",
        "false",
        "False",
        "no",
        "NO",
    )
    if deskew_env:
        bgr = _deskew_bgr(bgr)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # If the page is globally very bright and low-contrast, slightly darken strokes.
    mean_val = float(gray.mean())
    if mean_val > 165:
        gamma = 0.85
        inv_gamma = 1.0 / gamma
        table = np.array(
            [(i / 255.0) ** inv_gamma * 255.0 for i in range(256)],
            dtype=np.uint8,
        )
        gray = cv2.LUT(gray, table)

    # Optionally remove very large solid black/redaction boxes so they don't
    # confuse OCR (we inpaint them back to background).
    redact_env = os.environ.get("PALMA_REDACTION_MASK", "1").strip() not in (
        "0",
        "false",
        "False",
        "no",
        "NO",
    )
    if redact_env:
        box_mask = _mask_black_boxes(gray)
        if box_mask is not None:
            gray = cv2.inpaint(gray, box_mask, 3, cv2.INPAINT_TELEA)

    # Lift faint pencil/low-contrast text before line removal / binarization.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Remove underlines / long pencil strokes (toggleable).
    remove_lines_env = os.environ.get("PALMA_REMOVE_LINES", "1").strip() not in (
        "0",
        "false",
        "False",
        "no",
        "NO",
    )
    if remove_lines and remove_lines_env:
        gray = _remove_long_strokes(gray)

    if mode == "print":
        gray = cv2.fastNlMeansDenoising(gray, h=8)
        _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((2, 2), np.uint8)
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    elif mode == "mixed":
        gray = cv2.fastNlMeansDenoising(gray, h=4)
        gray = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 8
        )
    else:
        gray = cv2.fastNlMeansDenoising(gray, h=3)
        # Keep as grayscale (no hard threshold) for fragile content.
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _extract_text_paddle(ocr, result):
    """Parse PaddleOCR 2.x / 3.x result into plain text."""
    if result is None:
        return ""

    def _rows_from_boxes_texts(boxes, texts):
        """Group PaddleOCR boxes + texts into reading-order rows (for tables/forms)."""
        cells = []
        heights = []
        for box, txt in zip(boxes, texts):
            if txt is None:
                continue
            t = str(txt).strip()
            if not t:
                continue
            if _score_text(t) < 5.0:
                continue
            # box: 4 points [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
            try:
                pts = np.array(box, dtype=float)
                xs = pts[:, 0]
                ys = pts[:, 1]
                x_min = float(xs.min())
                y_min = float(ys.min())
                y_max = float(ys.max())
            except Exception:
                # Fallback: no geometry, treat as a single row.
                cells.append({"cy": 0.0, "x": 0.0, "txt": t})
                continue
            cy = (y_min + y_max) / 2.0
            h = max(1.0, y_max - y_min)
            heights.append(h)
            cells.append({"cy": cy, "x": x_min, "txt": t})

        if not cells:
            return []

        # Sort cells top-to-bottom, then left-to-right.
        cells.sort(key=lambda c: (c["cy"], c["x"]))

        # Estimate typical text height to decide row grouping tolerance.
        if heights:
            median_h = float(np.median(heights))
        else:
            median_h = 20.0
        band = max(10.0, median_h * 0.7)

        rows = []
        current_row = []
        current_y = None
        for c in cells:
            if current_y is None or abs(c["cy"] - current_y) <= band:
                current_row.append(c)
                # Update running average y for the row.
                if current_y is None:
                    current_y = c["cy"]
                else:
                    current_y = (current_y * (len(current_row) - 1) + c["cy"]) / len(
                        current_row
                    )
            else:
                # Finalize previous row.
                current_row.sort(key=lambda x: x["x"])
                rows.append(current_row)
                current_row = [c]
                current_y = c["cy"]
        if current_row:
            current_row.sort(key=lambda x: x["x"])
            rows.append(current_row)

        # Build text rows. Use " | " as column separator to keep table feeling.
        lines = []
        for row in rows:
            parts = [cell["txt"] for cell in row]
            if not parts:
                continue
            lines.append(" | ".join(parts))
        return lines

    # 2.x: result = [ [ [box], (text, conf) ], ... ]
    if isinstance(result, list) and len(result) > 0:
        page = result
        # Some variants wrap one page: [page_lines]
        if (
            len(result) == 1
            and isinstance(result[0], list)
            and result
            and result[0]
            and isinstance(result[0][0], (list, tuple))
        ):
            # If it looks like [[box, (text, conf)], ...], keep as-is; otherwise unwrap.
            looks_like_line = (
                len(result[0]) >= 2 and isinstance(result[0][1], (list, tuple))
            )
            if not looks_like_line:
                page = result[0]
        if page is None:
            return ""
        if isinstance(page, list):
            # Try to use box geometry if available to respect table / column layout.
            boxes = []
            texts = []
            has_boxes = True
            for line in page:
                if not line or len(line) < 2:
                    continue
                box = line[0]
                part = line[1]
                if isinstance(part, (list, tuple)) and len(part) > 0:
                    txt = str(part[0]).strip()
                elif part is not None:
                    txt = str(part).strip()
                else:
                    continue
                if not txt:
                    continue
                if not isinstance(box, (list, tuple)):
                    has_boxes = False
                boxes.append(box)
                texts.append(txt)
            if has_boxes and boxes:
                lines = _rows_from_boxes_texts(boxes, texts)
                if lines:
                    return "\n".join(lines)
            # Fallback: simple line-by-line text, still filtering garbage.
            clean_lines = []
            for txt in texts:
                if _score_text(txt) < 5.0:
                    continue
                clean_lines.append(txt)
            return "\n".join(clean_lines)
        if isinstance(page, dict):
            # 3.x may return dict with geometry + text.
            texts = page.get("rec_texts") or page.get("texts") or page.get(
                "text_line", []
            )
            boxes = (
                page.get("rec_boxes")
                or page.get("boxes")
                or page.get("polygons")
                or []
            )
            if isinstance(texts, list) and isinstance(boxes, list) and len(texts) == len(
                boxes
            ):
                lines = _rows_from_boxes_texts(boxes, texts)
                if lines:
                    return "\n".join(lines)
            if isinstance(texts, list):
                clean_lines = []
                for t in texts:
                    txt = str(t).strip()
                    if not txt:
                        continue
                    if _score_text(txt) < 5.0:
                        continue
                    clean_lines.append(txt)
                return "\n".join(clean_lines)
            if isinstance(texts, str):
                return texts
    return ""


def _run_ocr(bgr_image, ocr_engine, backend, tesseract_config_override=None):
    """Run OCR on BGR image; return plain text."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        path = f.name
    try:
        if not cv2.imwrite(path, bgr_image):
            return ""
        if backend == "paddle":
            if hasattr(ocr_engine, "predict"):
                result = ocr_engine.predict(path)
            else:
                result = ocr_engine.ocr(path, cls=True)
            return _extract_text_paddle(ocr_engine, result)
        if backend == "tesseract":
            config = (tesseract_config_override or "").strip()
            if not config:
                config = os.environ.get("PALMA_TESSERACT_CONFIG", "").strip()
            if not config:
                # psm 6 = assume a uniform block of text; tends to work well for documents.
                config = "--oem 1 --psm 6 -c preserve_interword_spaces=1"
            return ocr_engine.image_to_string(path, lang="eng", config=config).strip()
        return ""
    except Exception as e:
        print("OCR error:", e)
        return ""
    finally:
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass


def _best_ocr_text(bgr_image, ocr_engine, backend):
    """
    Run OCR and, when using Tesseract, try a couple of PSMs to improve
    results on noisy/handwritten pages. Returns best-scoring text.
    """
    if backend != "tesseract":
        # For Paddle we don't do PSM sweeps; caller should pass in any
        # alternative preprocessed variants and pick based on _score_text.
        return _run_ocr(bgr_image, ocr_engine, backend)

    # Keep the default env-config as the first attempt (preserves current behavior).
    candidates = [
        ("", _run_ocr(bgr_image, ocr_engine, backend)),
    ]

    psms_env = os.environ.get("PALMA_FALLBACK_PSMS", "6,11,4").strip()
    psms = []
    for part in psms_env.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            psms.append(int(part))
        except ValueError:
            continue

    # Only add alternates if user didn't already fix psm in env config.
    base_cfg = os.environ.get("PALMA_TESSERACT_CONFIG", "").strip()
    base_has_psm = "--psm" in base_cfg
    if not base_has_psm:
        for psm in psms:
            cfg = f"--oem 1 --psm {psm} -c preserve_interword_spaces=1"
            candidates.append((cfg, _run_ocr(bgr_image, ocr_engine, backend, tesseract_config_override=cfg)))

    best_text = ""
    best_score = -1.0
    for _, txt in candidates:
        s = _score_text(txt or "")
        if s > best_score:
            best_score = s
            best_text = txt or ""
    return best_text.strip()


def _clean_page_text(text):
    """
    Light cleanup of raw OCR text: normalize whitespace, collapse excess newlines.
    """
    if not text or not text.strip():
        return text.strip() if text else ""
    lines = [line.rstrip() for line in text.splitlines()]
    out = []
    prev_blank = False
    for line in lines:
        is_blank = not line.strip()
        if is_blank:
            if not prev_blank:
                out.append("")
            prev_blank = True
        else:
            out.append(line)
            prev_blank = False
    # Trim leading/trailing blank lines
    while out and not out[0].strip():
        out.pop(0)
    while out and not out[-1].strip():
        out.pop()
    return "\n".join(out)


def _extract_pdf_text_layer(page):
    """Get text from PDF embedded layer (no OCR). Returns None if empty."""
    try:
        t = page.get_text()
        return t.strip() if t and t.strip() else None
    except Exception:
        return None


def process_folder(input_folder, output_folder, preprocess_mode="mixed"):
    ocr_engine, backend = _get_ocr()
    use_ocr = ocr_engine is not None
    if not use_ocr:
        print("No OCR backend. Install Tesseract (https://github.com/UB-Mannheim/tesseract/wiki) then: pip install pytesseract")
        print("Falling back to PDF text layer only (works only if PDF has selectable text).")

    os.makedirs(output_folder, exist_ok=True)
    pdf_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print("No PDFs in", input_folder)
        return

    for file_name in pdf_files:
        pdf_path = os.path.join(input_folder, file_name)
        print("\n[Processing]", file_name)
        pages_text = []
        try:
            doc = fitz.open(pdf_path)
            for i in range(len(doc)):
                print("  Page %d/%d" % (i + 1, len(doc)), end="\r")
                page = doc.load_page(i)
                text = None
                if use_ocr:
                    dpi = float(os.environ.get("PALMA_DPI", "2").strip() or "2")
                    img = _page_to_bgr(page, dpi_scale=dpi)

                    if backend == "paddle":
                        # PALMA_FAST=1 (default): one OCR per page for speed. PALMA_FAST=0: try 3 variants, pick best.
                        fast = os.environ.get("PALMA_FAST", "1").strip().lower() not in ("0", "false", "no")
                        if fast:
                            # Single run: light contrast (CLAHE) only. Good balance and much faster.
                            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                            gray2 = clahe.apply(gray)
                            v = cv2.cvtColor(gray2, cv2.COLOR_GRAY2BGR)
                            text = _run_ocr(v, ocr_engine, backend).strip()
                            if not text or sum(ch.isalnum() for ch in text) < 15:
                                text = _run_ocr(img, ocr_engine, backend).strip()
                        else:
                            variants = []
                            variants.append(img)
                            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                            gray2 = clahe.apply(gray)
                            v2 = cv2.cvtColor(gray2, cv2.COLOR_GRAY2BGR)
                            variants.append(v2)
                            v3 = _deskew_bgr(v2)
                            if v3 is not None:
                                variants.append(v3)
                            best_text = ""
                            best_score = -1.0
                            for v in variants:
                                cand = _run_ocr(v, ocr_engine, backend)
                                s = _score_text(cand or "")
                                if s > best_score:
                                    best_score = s
                                    best_text = cand or ""
                            text = best_text.strip()
                            if not text or sum(ch.isalnum() for ch in text) < 15:
                                cand = _run_ocr(img, ocr_engine, backend)
                                if _score_text(cand or "") > best_score:
                                    text = (cand or "").strip()
                            del variants
                    else:
                        # Tesseract path keeps the more aggressive preprocessing and fallbacks.
                        cleaned = _preprocess(img, mode=preprocess_mode, remove_lines=True)
                        text = _best_ocr_text(cleaned, ocr_engine, backend)

                        # If line-removal result looks too short, retry without removing lines
                        # and keep whichever seems to contain more actual text.
                        if text is not None:
                            t = text.strip()
                            alpha = sum(ch.isalnum() for ch in t)
                            if alpha < 25:
                                cleaned2 = _preprocess(img, mode=preprocess_mode, remove_lines=False)
                                text2 = _best_ocr_text(cleaned2, ocr_engine, backend)
                                t2 = (text2 or "").strip()
                                alpha2 = sum(ch.isalnum() for ch in t2)
                                if alpha2 > alpha:
                                    text = text2
                                del cleaned2

                        if not text or not text.strip():
                            text = _best_ocr_text(img, ocr_engine, backend)

                        # Optional quality-scored heavy fallback for very hard pages
                        # (only really useful with Tesseract).
                        fallback_on = os.environ.get("PALMA_FALLBACK", "0").strip() not in (
                            "0",
                            "false",
                            "False",
                            "no",
                            "NO",
                        )
                        if fallback_on:
                            base_score = _score_text(text or "")
                            base_alpha = sum(ch.isalnum() for ch in (text or ""))
                            if base_alpha < 35 or base_score < 95:
                                best_text = text or ""
                                best_score = base_score

                                # Try heavier preprocess variants on same DPI.
                                variants = [
                                    _preprocess_heavy_handwriting(img, remove_lines=True),
                                    _preprocess_heavy_handwriting(img, remove_lines=False),
                                ]
                                # Try higher DPI render (only when needed).
                                try:
                                    fb_dpi = float(os.environ.get("PALMA_FALLBACK_DPI", "3").strip() or "3")
                                except ValueError:
                                    fb_dpi = 3.0
                                if fb_dpi > dpi + 0.1:
                                    try:
                                        img_hi = _page_to_bgr(page, dpi_scale=fb_dpi)
                                        variants.extend(
                                            [
                                                _preprocess(img_hi, mode=preprocess_mode, remove_lines=True),
                                                _preprocess_heavy_handwriting(img_hi, remove_lines=True),
                                                _preprocess_heavy_handwriting(img_hi, remove_lines=False),
                                            ]
                                        )
                                    except Exception:
                                        img_hi = None
                                    finally:
                                        if img_hi is not None:
                                            del img_hi

                                for v in variants:
                                    cand = _best_ocr_text(v, ocr_engine, backend)
                                    s = _score_text(cand or "")
                                    if s > best_score:
                                        best_score = s
                                        best_text = cand or ""
                                    del v
                                text = best_text

                        del img
                if not text or not text.strip():
                    text = _extract_pdf_text_layer(page)
                if not text or not text.strip():
                    text = "[Text unreadable or page blank]"
                text = _clean_page_text(text)
                pages_text.append("--- PAGE %d ---\n%s\n" % (i + 1, text))
                gc.collect()
            doc.close()

            out_name = file_name.replace(".pdf", ".txt").replace(".PDF", ".txt")
            out_path = os.path.join(output_folder, out_name)
            with open(out_path, "w", encoding="utf-8") as f:
                f.writelines(pages_text)
            print("\n[OK] Wrote", out_name)
        except Exception as e:
            print("\n[Error]", file_name, ":", e)


if __name__ == "__main__":
    IN = "input_pdfs"
    OUT = "output_text"
    MODE = os.environ.get("PALMA_PREPROCESS", "mixed")

    if not os.path.exists(IN):
        os.makedirs(IN)
        print("Put PDFs in '%s' and run again." % IN)
    else:
        process_folder(IN, OUT, preprocess_mode=MODE)
