"""
Lightweight PaddleOCR HTTP service for manga/webtoon pages.

Endpoints:
  POST /ocr          (multipart "file") -> { success, lines, raw }
  POST /preprocess    (multipart "file") -> cleaned PNG image bytes
  POST /translate     (JSON { text, target? }) -> { success, text }
  GET  /health        -> { ok, engine }
"""

import os
import threading
import time
import traceback
import numpy as np
import cv2
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

_engine = None
_engine_lock = threading.Lock()

# PADDLE_OCR_MODEL=server (default, accurate) | mobile (fast, low RAM)
_MODEL_TIER = os.environ.get("PADDLE_OCR_MODEL", "server").strip().lower()
# Minimum recognition confidence to keep a text line
_MIN_SCORE = float(os.environ.get("OCR_MIN_SCORE", "0.5"))
# Strip height for splitting tall webtoon pages
_STRIP_HEIGHT = int(os.environ.get("OCR_STRIP_HEIGHT", "2200"))
_STRIP_OVERLAP = int(os.environ.get("OCR_STRIP_OVERLAP", "160"))


def get_engine():
    """Lazy-load PaddleOCR. Defaults to the accurate server models;
    set PADDLE_OCR_MODEL=mobile for the lightweight models on weak CPUs."""
    global _engine
    if _engine is not None:
        return _engine
    with _engine_lock:
        if _engine is not None:
            return _engine
        from paddleocr import PaddleOCR

        tier = _MODEL_TIER
        try:
            det = "PP-OCRv5_server_det" if tier == "server" else "PP-OCRv5_mobile_det"
            rec = "PP-OCRv5_server_rec" if tier == "server" else "PP-OCRv5_mobile_rec"
            try:
                _engine = PaddleOCR(
                    lang="en",
                    text_detection_model_name=det,
                    text_recognition_model_name=rec,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    device="cpu",
                )
            except Exception as e:
                # Server models unavailable → fall back to mobile
                print(f"[ocr] {det}/{rec} init failed, trying mobile models: {e}", flush=True)
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
        except Exception:
            print("[ocr] PaddleOCR engine init failed:", flush=True)
            traceback.print_exc()
            raise
        print(f"[ocr] PaddleOCR engine ready (tier={tier})", flush=True)
    return _engine


def _warmup_engine():
    """Load PaddleOCR at startup — downloads the models on first run.
    Retries so a transient download failure doesn't leave every request 500ing."""
    for attempt in range(1, 4):
        try:
            get_engine()
            return
        except ImportError:
            # A failed import leaves paddlex half-initialized — retrying in the
            # same process only hits "PDX has already been initialized". Bail.
            print("[ocr] engine warmup failed (import error — fix the env, no retry):", flush=True)
            traceback.print_exc()
            break
        except Exception:
            print(f"[ocr] engine warmup attempt {attempt}/3 failed:", flush=True)
            traceback.print_exc()
            if attempt < 3:
                time.sleep(10)
    print("[ocr] engine warmup gave up — /ocr will retry init on demand", flush=True)


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    # Warm up PaddleOCR in a background thread so model downloads and init
    # errors surface at startup instead of on the first real request.
    threading.Thread(target=_warmup_engine, daemon=True).start()
    yield


app = FastAPI(lifespan=_lifespan)


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
            if polys is None:
                continue
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
    return {"ok": True, "engine": "paddleocr", "canTranslate": True}


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


def _ocr_prep(img: np.ndarray) -> np.ndarray:
    """Lightweight preparation applied inside /ocr before detection:
    border cleanup + adaptive contrast + text sharpening. (The heavier
    denoise/watermark-inpaint pipeline stays in /preprocess for pages the
    user explicitly wants cleaned and saved.)"""
    img = _clean_borders(img)
    img = _auto_level(img)
    img = _sharpen_text(img)
    return img


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
        traceback.print_exc()
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


# ---- Translation (free, no API key) ---- #

class TranslateRequest(BaseModel):
    text: str
    target: str = "hi"  # default: English -> Hindi
    source: str = "en"


# MyMemoryTranslator needs full locale codes, not short ISO codes
_MYMEMORY_LANG = {
    "en": "en-GB", "hi": "hi-IN", "es": "es-ES", "fr": "fr-FR",
    "de": "de-DE", "ja": "ja-JP", "ko": "ko-KR", "zh": "zh-CN",
    "ar": "ar-SA", "pt": "pt-PT", "ru": "ru-RU", "it": "it-IT",
    "nl": "nl-NL", "tr": "tr-TR", "vi": "vi-VN", "th": "th-TH",
    "bn": "bn-IN", "ta": "ta-IN", "te": "te-IN", "mr": "mr-IN",
    "gu": "gu-IN", "kn": "kn-IN", "ml": "ml-IN", "pa": "pa-IN",
    "ur": "ur-PK",
}


def _mymemory_code(code: str) -> str:
    return _MYMEMORY_LANG.get(code, code)


def _chunk_text(t: str, max_len: int) -> list[str]:
    """Split at sentence boundaries so translators get coherent chunks.
    GoogleTranslator degrades on >1200-char blobs; MyMemory caps at ~450."""
    if len(t) <= max_len:
        return [t]
    chunks: list[str] = []
    while t:
        if len(t) <= max_len:
            chunks.append(t)
            break
        idx = -1
        for sep in ('. ', '! ', '? ', '… ', '" ', "' "):
            idx = max(idx, t.rfind(sep, 0, max_len))
        if idx <= 0:
            idx = t.rfind(' ', 0, max_len)
        if idx <= 0:
            idx = max_len
        else:
            idx += 1
        chunks.append(t[:idx].strip())
        t = t[idx:].strip()
    return [c for c in chunks if c]


@app.post("/translate")
async def translate(body: TranslateRequest):
    """Translate text using deep-translator (free, no API key).
    GoogleTranslator first (sentence-chunked), then MyMemoryTranslator."""
    import time

    text = body.text.strip()
    if not text:
        return JSONResponse({"success": False, "error": "empty text"}, status_code=400)

    errors = []

    # Try GoogleTranslator first — better EN->HI quality, chunked per ~1200 chars
    try:
        from deep_translator import GoogleTranslator
        tr = GoogleTranslator(source=body.source, target=body.target)
        parts: list[str] = []
        for chunk in _chunk_text(text, 1200):
            part = tr.translate(chunk)
            parts.append(part or "")
            time.sleep(0.25)  # gentle pacing to avoid rate limits
        translated = " ".join(p for p in parts if p).strip()
        if translated:
            return {"success": True, "text": translated}
    except Exception as e:
        errors.append(f"Google: {e}")

    # Fallback: MyMemoryTranslator (free, needs full locale codes, 500-char limit)
    try:
        from deep_translator import MyMemoryTranslator
        src = _mymemory_code(body.source)
        tgt = _mymemory_code(body.target)
        tr = MyMemoryTranslator(source=src, target=tgt)
        parts = []
        for chunk in _chunk_text(text, 450):
            parts.append(tr.translate(chunk) or "")
            time.sleep(0.3)
        translated = " ".join(p for p in parts if p).strip()
        if translated:
            return {"success": True, "text": translated}
    except Exception as e:
        errors.append(f"MyMemory: {e}")

    return JSONResponse(
        {"success": False, "error": f"All translators failed: {'; '.join(errors)}"},
        status_code=500,
    )


def _run_ocr(engine, img: np.ndarray) -> list[dict]:
    result = engine.predict(img) if hasattr(engine, "predict") else engine.ocr(img)
    return _items_from_result(result)


def _ocr_strips(engine, img: np.ndarray) -> list[dict]:
    """Run OCR on the image, splitting very tall webtoon pages into
    overlapping horizontal strips — PaddleOCR's detector loses accuracy on
    extremely tall inputs, and overlapping strips avoid cutting bubbles."""
    h, _w = img.shape[:2]
    if h <= _STRIP_HEIGHT:
        return _run_ocr(engine, img)
    items: list[dict] = []
    y = 0
    while y < h:
        y2 = min(y + _STRIP_HEIGHT, h)
        for it in _run_ocr(engine, np.ascontiguousarray(img[y:y2])):
            it["y0"] += y
            it["y1"] += y
            items.append(it)
        if y2 >= h:
            break
        y = y2 - _STRIP_OVERLAP
    return items


def _dedupe_items(items: list[dict]) -> list[dict]:
    """Remove duplicate detections produced by strip overlap or the
    inverted second pass. Same/nested text whose box centres are close
    is considered a duplicate — keep the higher-confidence one."""
    out: list[dict] = []
    for it in sorted(items, key=lambda i: -i["score"]):
        t = " ".join(str(it["text"]).lower().split())
        dup = False
        for kept in out:
            kt = " ".join(str(kept["text"]).lower().split())
            if not t or (t != kt and t not in kt and kt not in t):
                continue
            if abs((it["y0"] + it["y1"]) / 2 - (kept["y0"] + kept["y1"]) / 2) < 30 and \
               abs((it["x0"] + it["x1"]) / 2 - (kept["x0"] + kept["x1"]) / 2) < 60:
                dup = True
                break
        if not dup:
            out.append(it)
    return out


@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        return JSONResponse({"success": False, "error": "empty file"}, status_code=400)

    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"success": False, "error": "could not decode image"}, status_code=400)

    try:
        # Upscale narrow pages — recognition accuracy collapses below ~900px wide
        h, w = img.shape[:2]
        if w < 900:
            scale = 900.0 / w
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        prepared = _ocr_prep(img)
        engine = get_engine()
        items = _ocr_strips(engine, prepared)

        # Dark pages (night scenes, black speech bubbles with white text) —
        # run a second pass on the inverted image to catch them
        gray = cv2.cvtColor(prepared, cv2.COLOR_BGR2GRAY)
        if float(np.mean(gray)) < 110:
            items += _ocr_strips(engine, cv2.bitwise_not(prepared))

        items = _dedupe_items(items)
        items = [i for i in items if i["score"] >= _MIN_SCORE]
        lines = group_bubbles(items)
        return {"success": True, "lines": lines, "raw": "\n".join(lines)}
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)
