import os
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchaudio
import soundfile as sf

from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.metrics import classification_report, confusion_matrix

# =====================
# Label setup
# =====================
LABELS = ["NEG", "NEU", "POS"]
label2id = {k: i for i, k in enumerate(LABELS)}
id2label = {i: k for k, i in label2id.items()}


# =====================
# Audio loader (FAST, CPU friendly)
# =====================
def load_wav_mono_16k(path: str) -> torch.Tensor:
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    wav = torch.from_numpy(wav)

    # 🔑 限制最長 4 秒（加速關鍵）
    max_len = 16000 * 4
    wav = wav[:max_len]

    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)

    return wav


# =====================
# Dataset
# =====================
class SERDataset(Dataset):
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        r = self.df.iloc[idx]
        wav = load_wav_mono_16k(r["path"])
        y = label2id[str(r["label"])]
        return wav, y


# =====================
# Collate
# =====================
class Collate:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        wavs, ys = zip(*batch)

        inputs = self.processor(
            [w.numpy() for w in wavs],
            sampling_rate=16000,
            padding=True,
            return_tensors="pt"
        )

        input_values = inputs.input_values
        attention_mask = torch.ones_like(input_values, dtype=torch.long)

        labels = torch.tensor(ys, dtype=torch.long)
        return input_values, attention_mask, labels


# =====================
# Model
# =====================
class W2V2SER(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

        for p in self.encoder.parameters():
            p.requires_grad = False  # freeze encoder

        hidden = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 3)
        )

    def forward(self, input_values, attention_mask=None):
        out = self.encoder(input_values=input_values, attention_mask=attention_mask)
        hs = out.last_hidden_state
        pooled = hs.mean(dim=1)  # ✅ 穩定
        return self.classifier(pooled)


# =====================
# Main
# =====================
def main():
    os.makedirs("models", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # 🔑 Demo 設定（務實）
    EPOCHS = 1
    BATCH_SIZE = 1
    LR = 2e-4

    model = W2V2SER().to(device)
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

    train_ds = SERDataset("data/train.csv")
    valid_ds = SERDataset("data/valid.csv")

    # 🔑 只取部分資料（專題 demo 完全夠）
    train_ds.df = train_ds.df.sample(n=min(1000, len(train_ds.df)), random_state=42)
    valid_ds.df = valid_ds.df.sample(n=min(200, len(valid_ds.df)), random_state=42)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=Collate(processor),
        num_workers=0
    )

    valid_loader = DataLoader(
        valid_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=Collate(processor),
        num_workers=0
    )

    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    # =====================
    # Training
    # =====================
    for epoch in range(1, EPOCHS + 1):
        model.train()
        start_time = time.time()
        last_log = time.time()

        for step, (x, attn, y) in enumerate(train_loader):
            x, attn, y = x.to(device), attn.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x, attn)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            # 🔑 每 30 秒印一次（你一定看得到）
            if time.time() - last_log > 30:
                elapsed = int(time.time() - start_time)
                print(f"[Epoch {epoch}] step {step}/{len(train_loader)} | elapsed {elapsed}s")
                last_log = time.time()

        print(f"Epoch {epoch} finished.")

    # =====================
    # Save model
    # =====================
    torch.save({
        "state_dict": model.state_dict(),
        "label2id": label2id,
        "id2label": id2label
    }, "models/ser_w2v2_3cls.pt")

    with open("models/label_map.json", "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, ensure_ascii=False, indent=2)

    print("✅ Model saved to models/ser_w2v2_3cls.pt")


if __name__ == "__main__":
    main()
