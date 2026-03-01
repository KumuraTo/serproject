import argparse
from pathlib import Path
import pandas as pd
import random

# 你的資料夾情緒名稱是 "Angry Happy Neutral Sad Surprise"
MAP_3CLS = {
    "happy": "POS",
    "surprise": "POS",
    "neutral": "NEU",
    "angry": "NEG",
    "sad": "NEG",
}

def split_by_speaker(df, seed=42, train=0.8, valid=0.1, test=0.1):
    assert abs(train + valid + test - 1.0) < 1e-6
    speakers = sorted(df["speaker"].unique().tolist())
    random.seed(seed)
    random.shuffle(speakers)

    n = len(speakers)
    n_train = max(1, int(round(n * train)))
    n_valid = max(1, int(round(n * valid)))
    if n_train + n_valid >= n:
        n_valid = max(1, n - n_train - 1)

    train_spk = set(speakers[:n_train])
    valid_spk = set(speakers[n_train:n_train + n_valid])
    test_spk = set(speakers[n_train + n_valid:])

    return (
        df[df["speaker"].isin(train_spk)].reset_index(drop=True),
        df[df["speaker"].isin(valid_spk)].reset_index(drop=True),
        df[df["speaker"].isin(test_spk)].reset_index(drop=True),
        train_spk, valid_spk, test_spk
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset_root",
        required=True,
        help="指到 'Emotion Speech Dataset' 這個資料夾"
    )
    ap.add_argument("--out_dir", default="data", help="輸出 train/valid/test.csv 的資料夾")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--valid_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)
    args = ap.parse_args()

    root = Path(args.dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"找不到資料夾：{root}")

    rows = []
    # 掃描 0001~0020
    for spk_dir in sorted(root.iterdir()):
        if not spk_dir.is_dir():
            continue
        speaker = spk_dir.name  # "0001"..."0020"

        # 掃描 Angry/Happy/Neutral/Sad/Surprise
        for emo_dir in sorted(spk_dir.iterdir()):
            if not emo_dir.is_dir():
                continue
            emo = emo_dir.name.strip().lower()  # "angry"...
            if emo not in MAP_3CLS:
                continue
            label = MAP_3CLS[emo]

            for wav_path in sorted(emo_dir.rglob("*.wav")):
                rows.append({
                    "path": str(wav_path.resolve()),
                    "label": label,
                    "speaker": speaker,
                    "emotion": emo,
                })

    if not rows:
        raise RuntimeError(
            "沒有掃到任何 wav 檔！請確認你 --dataset_root 是否指到 'Emotion Speech Dataset' 資料夾。"
        )

    df = pd.DataFrame(rows)
    print("總檔案數:", len(df))
    print("speaker 數:", df["speaker"].nunique())
    print("原始情緒分布:\n", df["emotion"].value_counts())
    print("三分類分布:\n", df["label"].value_counts())

    train_df, valid_df, test_df, train_spk, valid_spk, test_spk = split_by_speaker(
        df,
        seed=args.seed,
        train=args.train_ratio,
        valid=args.valid_ratio,
        test=args.test_ratio,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df[["path", "label"]].to_csv(out_dir / "train.csv", index=False, encoding="utf-8")
    valid_df[["path", "label"]].to_csv(out_dir / "valid.csv", index=False, encoding="utf-8")
    test_df[["path", "label"]].to_csv(out_dir / "test.csv", index=False, encoding="utf-8")

    # 也輸出 split 資訊，寫報告很好用
    info = {
        "train_speakers": sorted(list(train_spk)),
        "valid_speakers": sorted(list(valid_spk)),
        "test_speakers": sorted(list(test_spk)),
        "counts": {"train": len(train_df), "valid": len(valid_df), "test": len(test_df)}
    }
    import json
    with open(out_dir / "split_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print("\n✅ 已輸出：")
    print(" -", (out_dir / "train.csv").resolve())
    print(" -", (out_dir / "valid.csv").resolve())
    print(" -", (out_dir / "test.csv").resolve())
    print(" -", (out_dir / "split_info.json").resolve())

if __name__ == "__main__":
    main()
