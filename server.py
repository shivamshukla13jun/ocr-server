"""
Lightweight PaddleOCR HTTP service for manga/webtoon pages.

Endpoints:
  POST /ocr         (multipart "file") -> { success, lines, raw }
  POST /preprocess   (multipart "file") -> cleaned PNG image bytes
  GET  /health       -> { ok, engine }
"""

import io
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, Response

app = FastAPI()
_engine = None


def get_engine():
    """Lazy-load PaddleOCR with the lightweight PP-OCR mobile models."""
    global _engine
    if _engine is not None:
        return _engine
    from paddleocr import PaddleOCR

    try:
        # PaddleOCR 3.x — mobile (lightweight) detection + recognition models
        _engine = PaddleOCR(
            lang="en",
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="PP-OCRv5_mobile_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device="cpu",
        )
    except TypeError:
        # PaddleOCR 2.x fallback signature
        _engine = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    return _engine


def _items_from_result(result) -> list[dict]:
    """Normalise paddleocr 2.x/3.x output into [{text, score, x0, x1, y0, y1}]."""
    items: list[dict] = []
    if not result:
        return items

    # 3.x: predict() -> list of dict-like results with rec_texts/rec_scores/rec_polys
    for res in result:
        try:
            keys = res.keys() if hasattr(res, "keys") else []
        except Exception:
            keys = []

        if "rec_texts" in keys:
            texts = res["rec_texts"]
            scores = res.get("rec_scores", [1.0] * len(texts))
            polys = res.get("rec_polys", res.get("dt_polys", res.get("rec_boxes")))
            for poly, text, score in zip(polys, texts, scores):
                pts = np.asarray(poly, dtype=float).reshape(-1, 2)
                items.append({
                    "text": str(text), "score": float(score),
                    "x0": float(pts[:, 0].min()), "x1": float(pts[:, 0].max()),
                    "y0": float(pts[:, 1].min()), "y1": float(pts[:, 1].max()),
                })
        elif isinstance(res, list):
            # 2.x: ocr() -> [[ [box], (text, score) ], ...]
            for line in res:
                if not line or len(line) < 2:
                    continue
                box, (text, score) = line[0], line[1]
                pts = np.asarray(box, dtype=float).reshape(-1, 2)
                items.append({
                    "text": str(text), "score": float(score),
                    "x0": float(pts[:, 0].min()), "x1": float(pts[:, 0].max()),
                    "y0": float(pts[:, 1].min()), "y1": float(pts[:, 1].max()),
                })
    return items


def group_bubbles(items: list[dict]) -> list[str]:
    """Merge text lines that sit directly under each other with horizontal
    overlap into speech bubbles, then order bubbles top-to-bottom
    (left-to-right when side by side)."""
    items.sort(key=lambda l: (l["y0"], l["x0"]))
    bubbles: list[dict] = []
    for l in items:
        h = max(l["y1"] - l["y0"], 1.0)
        target = None
        for b in bubbles:
            gap = l["y0"] - b["y1"]
            overlap = min(b["x1"], l["x1"]) - max(b["x0"], l["x0"])
            if gap < h * 1.2 and gap > -h * 0.8 and \
               overlap > min(b["x1"] - b["x0"], l["x1"] - l["x0"]) * 0.25:
                target = b
                break
        if target:
            target["lines"].append(l)
            target["x0"] = min(target["x0"], l["x0"])
            target["x1"] = max(target["x1"], l["x1"])
            target["y1"] = max(target["y1"], l["y1"])
        else:
            bubbles.append({"lines": [l], "x0": l["x0"], "x1": l["x1"],
                            "y0": l["y0"], "y1": l["y1"]})
    # Reading order: sort by y-row (30px granularity), then x within the row
    bubbles.sort(key=lambda b: (round(b["y0"] / 30.0), b["x0"]))
    out = []
    for b in bubbles:
        b["lines"].sort(key=lambda l: (l["y0"], l["x0"]))
        out.append(" ".join(l["text"] for l in b["lines"]))
    return out


@app.get("/health")
def health():
    return {"ok": True, "engine": "paddleocr"}


# ---- Image preprocessing (clean manga pages before OCR) ---- #

def _auto_level(img: np.ndarray) -> np.ndarray:
    """CLAHE adaptive contrast enhancement on the L channel."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _detect_watermark_mask(gray: np.ndarray) -> np.ndarray:
    """Detect semi-transparent watermark text overlaid on the image.
    Returns a mask of watermark pixels to inpaint."""
    h, w = gray.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    # Common watermark zones: bottom 8%, top 5%, right-edge strip
    zones = [
        (int(h * 0.92), h, 0, w),        # bottom band
        (0, int(h * 0.05), 0, w),          # top band
        (0, h, int(w * 0.88), w),          # right strip
    ]
    for y1, y2, x1, x2 in zones:
        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        # Watermarks are typically lighter text with consistent luminance
        mean_val = float(np.mean(roi))
        std_val = float(np.std(roi))
        # Low-contrast, near-white regions in these zones are likely watermarks
        if std_val < 35 and mean_val > 180:
            mask[y1:y2, x1:x2] = 255
            continue
        # Edge-detect in the zone to find overlaid text strokes
        edges = cv2.Canny(roi, 50, 150)
        edge_density = float(np.count_nonzero(edges)) / max(1, edges.size)
        # Sparse edges in a mostly-uniform zone → likely watermark
        if 0.005 < edge_density < 0.08 and std_val < 50:
            mask[y1:y2, x1:x2] = 255

    return mask


def _clean_borders(img: np.ndarray) -> np.ndarray:
    """Fill solid-colour borders with white (common in scanlation crops)."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result = img.copy()

    # Check each edge: if a thin strip is near-uniform, fill it white
    for strip, y1, y2, x1, x2 in [
        ("top",    0,             min(12, h // 40), 0, w),
        ("bottom", max(0, h - min(12, h // 40)), h, 0, w),
        ("left",   0, h,         0,             min(12, w // 40)),
        ("right",  0, h,         max(0, w - min(12, w // 40)), w),
    ]:
        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        if float(np.std(roi)) < 15:
            result[y1:y2, x1:x2] = 255

    return result


def _sharpen_text(img: np.ndarray) -> np.ndarray:
    """Mild unsharp-mask to sharpen text edges for OCR."""
    blurred = cv2.GaussianBlur(img, (0, 0), 2.0)
    return cv2.addWeighted(img, 1.3, blurred, -0.3, 0)


def preprocess_image(img: np.ndarray) -> np.ndarray:
    """Full manga page cleaning pipeline."""
    # 1. Denoise (bilateral preserves edges better than fastNlMeans for manga)
    denoised = cv2.bilateralFilter(img, 9, 75, 75)

    # 2. Auto-level contrast
    enhanced = _auto_level(denoised)

    # 3. Clean borders
    cleaned = _clean_borders(enhanced)

    # 4. Detect & inpaint watermarks
    gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
    wm_mask = _detect_watermark_mask(gray)
    if np.count_nonzero(wm_mask) > 0:
        cleaned = cv2.inpaint(cleaned, wm_mask, 7, cv2.INPAINT_TELEA)

    # 5. Sharpen text
    final = _sharpen_text(cleaned)

    return final


@app.post("/preprocess")
async def preprocess(file: UploadFile = File(...)):
    """Clean a manga page: denoise, enhance contrast, remove watermarks, sharpen text.
    Returns the processed image as PNG bytes."""
    data = await file.read()
    if not data:
        return JSONResponse({"success": False, "error": "empty file"}, status_code=400)

    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"success": False, "error": "could not decode image"}, status_code=400)

    try:
        result = preprocess_image(img)
        _, buf = cv2.imencode(".png", result)
        return Response(content=buf.tobytes(), media_type="image/png")
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        return JSONResponse({"success": False, "error": "empty file"}, status_code=400)

    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"success": False, "error": "could not decode image"}, status_code=400)

    try:
        engine = get_engine()
        result = engine.predict(img) if hasattr(engine, "predict") else engine.ocr(img)
        items = _items_from_result(result)
        # Drop low-confidence fragments — mostly art/speed lines misread as text
        items = [i for i in items if i["score"] >= 0.45]
        lines = group_bubbles(items)
        return {"success": True, "lines": lines, "raw": "\n".join(lines)}
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)
