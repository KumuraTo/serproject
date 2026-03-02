import os
import tempfile
import numpy as np
import soundfile as sf

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import onnxruntime as ort
import librosa

MODEL_ONNX = "models/ser_w2v2_3cls.onnx"
LABELS = ["NEG", "NEU", "POS"]

app = FastAPI(title="SER (ONNX) API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# load once (fast + small)
if not os.path.exists(MODEL_ONNX):
    raise FileNotFoundError(f"Missing ONNX model: {MODEL_ONNX}")

sess = ort.InferenceSession(MODEL_ONNX, providers=["CPUExecutionProvider"])

def load_wav_mono_16k(path: str) -> np.ndarray:
    y, sr = sf.read(path, dtype="float32")
    if y.ndim == 2:
        y = y.mean(axis=1)

    # resample to 16k
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)

    # keep max 4 sec
    y = y[:16000 * 4]

    # ensure shape [1, T]
    return y.astype(np.float32)[None, :]

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/predict_emotion")
async def predict_emotion(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        x = load_wav_mono_16k(tmp_path)
        logits = sess.run(["logits"], {"input_values": x})[0]  # [1,3]
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
        probs = probs[0]

        pred_id = int(np.argmax(probs))
        emotion = LABELS[pred_id]
        confidence = float(probs[pred_id])

        return {
            "emotion": emotion,
            "confidence": round(confidence, 3),
            "probs": {LABELS[i]: round(float(probs[i]), 3) for i in range(3)}
        }
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)