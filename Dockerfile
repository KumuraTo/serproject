FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TRANSFORMERS_CACHE=/tmp/hf \
    HF_HOME=/tmp/hf

WORKDIR /app

# (可選) torchaudio/soundfile 常需要的系統庫，先裝最小集合
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY models ./models

# Railway 用 $PORT
CMD ["sh","-lc","python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]