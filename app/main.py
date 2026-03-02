import os
import io
import time
import json
import zipfile
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

import onnxruntime as ort
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# =========================================================
# Paths / Config
# =========================================================
BASE_DIR = Path(__file__).resolve().parents[1]   # .../app
MODEL_DIR = BASE_DIR / "models"

MODEL_ONNX_NAME = "ser_w2v2_3cls.onnx"
MODEL_DATA_NAME = "ser_w2v2_3cls.onnx.data"
LABEL_MAP_NAME = "label_map.json"

MODEL_ONNX_PATH = MODEL_DIR / MODEL_ONNX_NAME
MODEL_DATA_PATH = MODEL_DIR / MODEL_DATA_NAME
LABEL_MAP_PATH = MODEL_DIR / LABEL_MAP_NAME

# Set in Railway Variables:
# MODEL_ZIP_URL = https://github.com/.../releases/download/v0.1/models_bundle.zip
MODEL_ZIP_URL = os.environ.get("MODEL_ZIP_URL")

# Labels fallback (if label_map.json missing)
DEFAULT_LABELS = ["NEG", "NEU", "POS"]

# Lazy session globals
_sess = None
_sess_err = None
_labels = None

# =========================================================
# FastAPI
# =========================================================
app = FastAPI(title="SER (ONNX) API - Railway Demo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # demo friendly; lock down later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# Utilities: model download + load
# =========================================================
def _download_and_unzip_model():
    """
    Download models_bundle.zip from MODEL_ZIP_URL and extract into MODEL_DIR.
    Handles the common case where zip contains either:
      - files directly, or
      - a top-level folder containing the files
    """
    if not MODEL_ZIP_URL:
        raise RuntimeError("MODEL_ZIP_URL is not set. Please set Railway Variables -> MODEL_ZIP_URL")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"⬇️ Downloading model zip from: {MODEL_ZIP_URL}")
    data = urllib.request.urlopen(MODEL_ZIP_URL, timeout=180).read()

    with zipfile.ZipFile(io.BytesIO(data)) as z:
        z.extractall(MODEL_DIR)

    # If extracted into a nested folder, try to locate and move files up
    if not (MODEL_ONNX_PATH.exists() and MODEL_DATA_PATH.exists()):
        # Search for onnx inside extracted tree
        found_onnx = next(MODEL_DIR.rglob(MODEL_ONNX_NAME), None)
        found_data = next(MODEL_DIR.rglob(MODEL_DATA_NAME), None)

        if found_onnx and found_data:
            # Move them to MODEL_DIR root
            (MODEL_DIR / MODEL_ONNX_NAME).write_bytes(found_onnx.read_bytes())
            (MODEL_DIR / MODEL_DATA_NAME).write_bytes(found_data.read_bytes())

        found_label = next(MODEL_DIR.rglob(LABEL_MAP_NAME), None)
        if found_label:
            (MODEL_DIR / LABEL_MAP_NAME).write_bytes(found_label.read_bytes())

    if not MODEL_ONNX_PATH.exists():
        raise RuntimeError(f"ONNX file not found after unzip: {MODEL_ONNX_PATH}")
    if not MODEL_DATA_PATH.exists():
        raise RuntimeError(f"ONNX external data file not found after unzip: {MODEL_DATA_PATH}")

    # sanity sizes (helps catch pointer files)
    onnx_size = MODEL_ONNX_PATH.stat().st_size
    data_size = MODEL_DATA_PATH.stat().st_size
    print(f"✅ Model files ready. onnx={onnx_size} bytes, data={data_size} bytes")


def ensure_model_files():
    """
    Ensure ONNX + external data exist locally.
    """
    if MODEL_ONNX_PATH.exists() and MODEL_DATA_PATH.exists():
        return
    _download_and_unzip_model()


def load_labels():
    global _labels
    if _labels is not None:
        return _labels

    if LABEL_MAP_PATH.exists():
        try:
            obj = json.loads(LABEL_MAP_PATH.read_text(encoding="utf-8"))
            # support both {"0":"NEG",...} or {"labels":["NEG","NEU","POS"]}
            if isinstance(obj, dict) and "labels" in obj and isinstance(obj["labels"], list):
                _labels = obj["labels"]
            elif isinstance(obj, dict):
                # assume numeric keys
                keys = sorted(obj.keys(), key=lambda x: int(x))
                _labels = [obj[k] for k in keys]
            else:
                _labels = DEFAULT_LABELS
        except Exception:
            _labels = DEFAULT_LABELS
    else:
        _labels = DEFAULT_LABELS

    return _labels


def get_sess():
    """
    Lazy-load ONNX Runtime session.
    Never run at import time -> avoids CRASH loops.
    """
    global _sess, _sess_err
    if _sess is not None:
        return _sess
    if _sess_err is not None:
        raise RuntimeError(_sess_err)

    try:
        ensure_model_files()
        # CPU only (Railway)
        _sess = ort.InferenceSession(str(MODEL_ONNX_PATH), providers=["CPUExecutionProvider"])
        print("✅ ONNX Runtime session initialized")
        return _sess
    except Exception as e:
        _sess_err = str(e)
        raise


# =========================================================
# Audio preprocessing (match training as much as possible)
# =========================================================
def load_wav_mono_16k_max4s(path: str) -> np.ndarray:
    """
    Returns shape [1, T] float32, mono, 16kHz, max 4 seconds (64000 samples).
    """
    y, sr = sf.read(path, dtype="float32")
    if y.ndim == 2:
        y = y.mean(axis=1)

    # resample to 16k if needed
    if sr != 16000:
        y = resample_poly(y, 16000, sr).astype(np.float32)

    # clip to max 4s
    y = y[: 16000 * 4]
    if y.size == 0:
        # avoid empty
        y = np.zeros((16000,), dtype=np.float32)

    return y.astype(np.float32)[None, :]


def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


# =========================================================
# Routes
# =========================================================
@app.get("/")
def root():
    return {"status": "ok", "service": "SER (ONNX)"}


@app.get("/health")
def health():
    """
    Always fast. Does NOT load model.
    """
    return {
        "ok": True,
        "model_present": MODEL_ONNX_PATH.exists() and MODEL_DATA_PATH.exists(),
        "model_loaded": _sess is not None,
        "error": _sess_err,
    }


@app.post("/predict_emotion")
async def predict_emotion(file: UploadFile = File(...)):
    """
    Upload wav audio -> emotion prediction (NEG/NEU/POS)
    """
    t0 = time.time()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        x = load_wav_mono_16k_max4s(tmp_path)  # [1,T] float32

        sess = get_sess()
        labels = load_labels()

        # Most common input name is "input_values" (as exported)
        inp_name = sess.get_inputs()[0].name
        out_names = [o.name for o in sess.get_outputs()]

        outputs = sess.run(out_names, {inp_name: x})
        logits = outputs[0]  # expected [1,3]

        if logits.ndim == 2:
            logits = logits[0]
        probs = softmax_np(logits.astype(np.float32))

        pred_id = int(np.argmax(probs))
        emotion = labels[pred_id] if pred_id < len(labels) else DEFAULT_LABELS[pred_id]
        conf = float(probs[pred_id])

        return {
            "emotion": emotion,
            "confidence": round(conf, 3),
            "probs": {labels[i]: round(float(probs[i]), 3) for i in range(min(3, len(labels)))},
            "latency_sec": round(time.time() - t0, 3),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass