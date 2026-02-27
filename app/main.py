import os
import time
import tempfile
import threading
import traceback

import numpy as np
import soundfile as sf

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from openai import OpenAI


# =========================
# Basic config
# =========================
# Fly 上沒有 GPU，先固定 CPU，避免一開始就碰 torch.cuda
DEVICE = "cpu"
MODEL_PATH = "models/ser_w2v2_3cls.pt"
LABELS = ["NEG", "NEU", "POS"]

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# =========================
# Prompts
# =========================
NEG_PROMPT = """
你是一隻溫柔、耐心的 AI 陪伴寵物。
使用者現在情緒偏低，請先表達理解與關心，
避免說教、避免否定，回覆要短、溫暖、可被長者理解。
"""

NEU_PROMPT = """
你是一隻自然、友善的 AI 陪伴寵物。
使用者情緒平穩，請以正常聊天方式回應，
語氣清楚自然，不要過度關心。
"""

POS_PROMPT = """
你是一隻活潑、正向的 AI 陪伴寵物。
使用者心情不錯，請給予回饋與鼓勵，
語氣可以稍微開心、親近一點，但仍要簡單好懂。
"""


# =========================
# FastAPI app
# =========================
app = FastAPI(title="SER + STT + Pet Chat API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # production 建議限制
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 狀態（讓服務先起來，再背景載模型）
app.state.ready = False
app.state.load_error = None
app.state.ser_model = None
app.state.torch = None
app.state.torchaudio = None


# =========================
# Background loader
# =========================
def _load_ser_model_in_background():
    try:
        print("🔄 Importing torch/torchaudio/transformers ...", flush=True)
        import torch
        import torch.nn as nn
        import torchaudio
        from transformers import Wav2Vec2Model

        print("✅ torch imported", flush=True)

        class W2V2SER(nn.Module):
            def __init__(self):
                super().__init__()
                # 這行可能會下載模型（若 cache 不在），第一次會很慢
                self.encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
                hidden = self.encoder.config.hidden_size
                self.classifier = nn.Sequential(
                    nn.Linear(hidden, hidden),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden, 3)
                )

            def forward(self, x, attention_mask=None):
                out = self.encoder(input_values=x, attention_mask=attention_mask)
                hs = out.last_hidden_state
                pooled = hs.mean(dim=1)
                return self.classifier(pooled)

        print("🔄 Loading SER checkpoint ...", flush=True)
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Missing model file: {MODEL_PATH} (check .dockerignore!)")

        ckpt = torch.load(MODEL_PATH, map_location="cpu")
        ser_model = W2V2SER().to("cpu")
        ser_model.load_state_dict(ckpt["state_dict"])
        ser_model.eval()

        # 存到 state（之後 predict 使用）
        app.state.torch = torch
        app.state.torchaudio = torchaudio
        app.state.ser_model = ser_model
        app.state.ready = True

        print("✅ SER model loaded successfully", flush=True)

    except Exception as e:
        app.state.ready = False
        app.state.load_error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        print("❌ Model load failed:\n" + app.state.load_error, flush=True)


@app.on_event("startup")
def startup():
    # 不阻塞 uvicorn listen：用 thread 背景載入
    t = threading.Thread(target=_load_ser_model_in_background, daemon=True)
    t.start()


# =========================
# Utilities
# =========================
def load_wav_mono_16k_for_ser(path: str):
    """Load wav for SER. Keep max 4 seconds."""
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim == 2:
        wav = wav.mean(axis=1)

    wav = wav[:16000 * 4]

    # torch/torchaudio 都要等 model ready 才能用
    torch = app.state.torch
    torchaudio = app.state.torchaudio

    wav = torch.from_numpy(wav)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)

    return wav.unsqueeze(0)


def pick_system_prompt(emotion: str) -> str:
    if emotion == "NEG":
        return NEG_PROMPT.strip()
    if emotion == "POS":
        return POS_PROMPT.strip()
    return NEU_PROMPT.strip()


def stt_transcribe(wav_path: str) -> str:
    if not OPENAI_API_KEY or client is None:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    with open(wav_path, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=f
        )
    return transcript.text.strip()


def pet_chat(system_prompt: str, user_text: str) -> str:
    if not OPENAI_API_KEY or client is None:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text}
        ],
        temperature=0.7
    )
    return resp.choices[0].message.content.strip()


# =========================
# Routes
# =========================
@app.get("/health")
def health():
    # 你用這個判斷是否載入完成
    return {
        "ok": True,
        "ready": app.state.ready,
        "error": app.state.load_error
    }


@app.get("/")
def root():
    return {"status": "API is running", "device": DEVICE, "ready": app.state.ready}


@app.post("/predict_emotion")
async def predict_emotion(file: UploadFile = File(...)):
    if not app.state.ready:
        raise HTTPException(status_code=503, detail="Model is loading. Try again later.")

    torch = app.state.torch
    ser_model = app.state.ser_model

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        wav = load_wav_mono_16k_for_ser(tmp_path).to("cpu")
        with torch.no_grad():
            logits = ser_model(wav)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

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


@app.post("/chat_audio")
async def chat_audio(file: UploadFile = File(...)):
    if not app.state.ready:
        raise HTTPException(status_code=503, detail="Model is loading. Try again later.")

    start_t = time.time()
    torch = app.state.torch
    ser_model = app.state.ser_model

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        transcript = stt_transcribe(tmp_path)  # 需要 OPENAI_API_KEY
        if not transcript:
            transcript = "（聽不清楚）"

        wav = load_wav_mono_16k_for_ser(tmp_path).to("cpu")
        with torch.no_grad():
            logits = ser_model(wav)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        pred_id = int(np.argmax(probs))
        emotion = LABELS[pred_id]
        confidence = float(probs[pred_id])

        system_prompt = pick_system_prompt(emotion)
        pet_reply = pet_chat(system_prompt, transcript)

        elapsed = round(time.time() - start_t, 3)
        return {
            "transcript": transcript,
            "emotion": emotion,
            "confidence": round(confidence, 3),
            "pet_reply": pet_reply,
            "latency_sec": elapsed
        }
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)