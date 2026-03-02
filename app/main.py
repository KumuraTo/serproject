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
# Config
# =========================================================
BASE_DIR = Path(__file__).resolve().parents[1]   # .../app
MODEL_DIR = BASE_DIR / "models"

MODEL_ONNX_NAME = "ser_w2v2_3cls.onnx"
MODEL_DATA_NAME = "ser_w2v2_3cls.onnx.data"
LABEL_MAP_NAME = "label_map.json"

MODEL_ONNX_PATH = MODEL_DIR / MODEL_ONNX_NAME
MODEL_DATA_PATH = MODEL_DIR / MODEL_DATA_NAME
LABEL_MAP_PATH = MODEL_DIR / LABEL_MAP_NAME

# Railway Variables:
# MODEL_ZIP_URL = https://github.com/.../releases/download/v0.1/models_bundle.zip
MODEL_ZIP_URL = os.environ.get("MODEL_ZIP_URL")

# Demo mode: set DEMO_MODE=1 in Railway Variables to always return stub response
DEMO_MODE = os.environ.get("DEMO_MODE", "0") == "1"

DEFAULT_LABELS = ["NEG", "NEU", "POS"]

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
# Model download / load
# =========================================================
def _download_and_unzip_model():
    if not MODEL_ZIP_URL:
        raise RuntimeError("MODEL_ZIP_URL is not set. Please set it in Railway Variables.")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"⬇️ Downloading model zip from: {MODEL_ZIP_URL}")
    data = urllib.request.urlopen(MODEL_ZIP_URL, timeout=180).read()

    with zipfile.ZipFile(io.BytesIO(data)) as z:
        z.extractall(MODEL_DIR)

    # If extracted into nested folder(s), locate and copy files to MODEL_DIR root
    if not (MODEL_ONNX_PATH.exists() and MODEL_DATA_PATH.exists()):
        found_onnx = next(MODEL_DIR.rglob(MODEL_ONNX_NAME), None)
        found_data = next(MODEL_DIR.rglob(MODEL_DATA_NAME), None)
        found_label = next(MODEL_DIR.rglob(LABEL_MAP_NAME), None)

        if found_onnx:
            (MODEL_DIR / MODEL_ONNX_NAME).write_bytes(found_onnx.read_bytes())
        if found_data:
            (MODEL_DIR / MODEL_DATA_NAME).write_bytes(found_data.read_bytes())
        if found_label:
            (MODEL_DIR / LABEL_MAP_NAME).write_bytes(found_label.read_bytes())

    if not MODEL_ONNX_PATH.exists():
        raise RuntimeError(f"ONNX file not found after unzip: {MODEL_ONNX_PATH}")
    if not MODEL_DATA_PATH.exists():
        raise RuntimeError(f"ONNX external data file not found after unzip: {MODEL_DATA_PATH}")

    onnx_size = MODEL_ONNX_PATH.stat().st_size
    data_size = MODEL_DATA_PATH.stat().st_size
    print(f"✅ Model files ready. onnx={onnx_size} bytes, data={data_size} bytes")

    # sanity check: your external data should be huge (hundreds of MB)
    if data_size < 10_000_000:
        raise RuntimeError(
            f"Bad model download: onnx={onnx_size} bytes, data={data_size} bytes. "
            f"Check MODEL_ZIP_URL (may be HTML/redirect), or re-upload release asset."
        )


def ensure_model_files():
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
            if isinstance(obj, dict) and "labels" in obj and isinstance(obj["labels"], list):
                _labels = obj["labels"]
            elif isinstance(obj, dict):
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
    global _sess, _sess_err
    if _sess is not None:
        return _sess
    if _sess_err is not None:
        raise RuntimeError(_sess_err)

    try:
        ensure_model_files()
        _sess = ort.InferenceSession(str(MODEL_ONNX_PATH), providers=["CPUExecutionProvider"])
        print("✅ ONNX Runtime session initialized")
        return _sess
    except Exception as e:
        _sess_err = str(e)
        raise


# =========================================================
# Audio preprocessing
# =========================================================
def load_wav_mono_16k_max4s(path: str) -> np.ndarray:
    y, sr = sf.read(path, dtype="float32")
    if y.ndim == 2:
        y = y.mean(axis=1)

    if sr != 16000:
        y = resample_poly(y, 16000, sr).astype(np.float32)

    y = y[: 16000 * 4]
    if y.size == 0:
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
    return {"status": "ok", "service": "SER (ONNX)", "demo_mode": DEMO_MODE}


@app.get("/health")
def health():
    return {
        "ok": True,
        "demo_mode": DEMO_MODE,
        "model_present": MODEL_ONNX_PATH.exists() and MODEL_DATA_PATH.exists(),
        "model_loaded": _sess is not None,
        "error": _sess_err,
    }


@app.post("/predict_emotion")
async def predict_emotion(file: UploadFile = File(...)):
    """
    Upload wav -> emotion prediction.
    Demo mode: always returns stub to guarantee success for presentation.
    """
    t0 = time.time()

    # ✅ DEMO MODE: do not touch model at all
    if DEMO_MODE:
        return {
            "emotion": "NEU",
            "confidence": 0.999,
            "probs": {"NEG": 0.001, "NEU": 0.999, "POS": 0.0},
            "latency_sec": round(time.time() - t0, 3),
            "demo_mode": True,
        }

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        x = load_wav_mono_16k_max4s(tmp_path)  # [1, T]

        sess = get_sess()
        labels = load_labels()

        inp_name = sess.get_inputs()[0].name
        out_names = [o.name for o in sess.get_outputs()]
        outputs = sess.run(out_names, {inp_name: x})

        logits = outputs[0]
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
            "demo_mode": False,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass