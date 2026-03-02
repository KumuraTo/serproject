import os
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

MODEL_PATH = "models/ser_w2v2_3cls.pt"
OUT_PATH = "models/ser_w2v2_3cls.onnx"

class W2V2SER(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        hidden = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 3)
        )

    def forward(self, x):
        out = self.encoder(input_values=x)
        hs = out.last_hidden_state
        pooled = hs.mean(dim=1)
        return self.classifier(pooled)

def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(MODEL_PATH)

    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    model = W2V2SER().eval()
    model.load_state_dict(ckpt["state_dict"], strict=True)

    # dummy input: [1, T] (4 sec @16k = 64000)
    dummy = torch.randn(1, 64000, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy,
        OUT_PATH,
        input_names=["input_values"],
        output_names=["logits"],
        dynamic_axes={"input_values": {1: "T"}},
        opset_version=17
    )

    print("Exported:", OUT_PATH)

if __name__ == "__main__":
    main()