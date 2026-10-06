# PaddleOCR Server

Lightweight, self-hosted OCR + translation sidecar for manga/webtoon pages.
Runs the **PP-OCRv5 mobile** models (detection + recognition) on CPU — no GPU,
no API keys, no rate limits. Hindi translation uses `deep-translator` (free
Google Translate / MyMemory fallback).

The app talks to it over HTTP:

- `GET  /health` → `{ "ok": true, "engine": "paddleocr", "canTranslate": true }`
- `POST /ocr` (multipart field `file` = page image) →
  `{ "success": true, "lines": ["bubble 1", "bubble 2", ...], "raw": "..." }`
- `POST /preprocess` (multipart field `file` = page image) →
  cleaned PNG image bytes (denoise, contrast, watermark removal, text sharpening)
- `POST /translate` (JSON `{ text, target?, source? }`) →
  `{ "success": true, "text": "translated text" }`

Detections are grouped into speech bubbles in reading order (top→bottom,
left→right), which is what the Studio OCR pipeline stores as narration.

## Endpoints

| Method | Path           | Body                            | Returns                        |
|--------|----------------|---------------------------------|--------------------------------|
| GET    | `/health`      | —                               | `{ ok, engine, canTranslate }` |
| POST   | `/ocr`         | multipart `file` (png/jpg/webp) | `{ success, lines, raw }`      |
| POST   | `/preprocess`  | multipart `file` (png/jpg/webp) | cleaned PNG bytes (image/png)  |
| POST   | `/translate`   | JSON `{ text, target, source }` | `{ success, text }`            |

## Run with Docker (recommended)

From the repo root:

```bash
docker compose up -d paddleocr
```

- Listens on **http://localhost:5004**
- First start downloads the PP-OCRv5 mobile models (~20 MB) into the
  `paddle_models` volume — subsequent starts are instant.
- Point the app at it with `PADDLEOCR_URL=http://localhost:5004` in `.env`.

## Run without Docker

Requires **Python 3.10–3.12** (PaddlePaddle wheels are published per minor
version; 3.11 is safest).

```powershell
cd ocr-server
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt

uvicorn server:app --host 0.0.0.0 --port 5004
```

> Run the pip install as a **single command** — `pip install -r requirements.txt`
> reads `requirements.txt` in this folder. (Line-continuation `\` doesn't work
> in PowerShell, which is why pasting the multi-line command fails.)

> Linux note: if `cv2` fails to import, install the system libs it needs:
> `sudo apt-get install libglib2.0-0 libgl1 libgomp1`

Then set `PADDLEOCR_URL=http://localhost:5004` in `.env` and restart `npm run dev`.

## Quick test

```bash
# health
curl http://localhost:5004/health

# OCR an image
curl -X POST http://localhost:5004/ocr -F "file=@page.png"

# Preprocess (clean) an image — returns PNG bytes
curl -X POST http://localhost:5004/preprocess -F "file=@page.png" --output cleaned.png

# Translate English to Hindi
curl -X POST http://localhost:5004/translate -H "Content-Type: application/json" \
  -d '{"text": "The hero stood tall.", "target": "hi", "source": "en"}'
```

Expected OCR response: `{"success":true,"lines":[ ...bubble texts... ]}`
Expected translate response: `{"success":true,"text":"नायक लंबा खड़ा था।"}`

## Notes

- **First OCR call is slow** (models load lazily on the first `/ocr` request,
  not on `/health`). Subsequent calls are fast.
- CPU-only by design; a tall webtoon page takes a few seconds. To use a GPU,
  change `device="cpu"` to `device="gpu"` in `server.py` and install the
  CUDA build of `paddlepaddle-gpu`.
- Models are the **mobile** variants for speed. If accuracy matters more
  than speed, swap `PP-OCRv5_mobile_det/rec` for `PP-OCRv5_server_det/rec`
  in `get_engine()` — the download is bigger but one-time.
- `lang="en"` — switch to `lang="ch"`/`"japan"` etc. for other scripts.
- Translation tries Google Translate first (free API), falls back to
  MyMemory if Google rate-limits. No API key required for either.
