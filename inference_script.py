import argparse
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm

from dataloader import DeepfakeDataset
from model import load_inference_model

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — Matches train.py setup
# ─────────────────────────────────────────────────────────────────────────────
IMAGE_SIZE  = 224
BATCH_SIZE  = 64
NUM_WORKERS = 4
THRESHOLD   = 0.5
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ARCHS       = ["efficientnet_b4", "convnext_tiny", "swin_t"]   # ensemble members (probabilities are averaged)


# ─────────────────────────────────────────────────────────────────────────────
# MODELS — Ensemble of DeepfakeDetectionModel from model.py
# ─────────────────────────────────────────────────────────────────────────────
def load_models(weights_path: str) -> dict:
    """weights_path is either
       - a folder with {arch}_weights.pt (merged) or {arch}_best.pt (training checkpoint) per architecture, or
       - a single .pt file {arch: merged state_dict} (written with --save_merged)."""
    if os.path.isdir(weights_path):
        state_dicts = {}
        for arch in ARCHS:
            for name in (f"{arch}_weights.pt", f"{arch}_best.pt"):
                path = os.path.join(weights_path, name)
                if os.path.exists(path):
                    state_dicts[arch] = torch.load(path, map_location="cpu", weights_only=True)
                    break
    else:
        state_dicts = torch.load(weights_path, map_location="cpu", weights_only=True)
        if not set(state_dicts) <= set(ARCHS):
            raise ValueError(f"{weights_path} is not an ensemble file ({{arch: state_dict}} for {ARCHS})")
    if not state_dicts:
        raise FileNotFoundError(f"No model weights found in {weights_path}")

    models = {}
    for arch, state_dict in state_dicts.items():
        model = load_inference_model(arch, state_dict).to(DEVICE)
        if DEVICE.type == "cuda":
            model = model.to(memory_format=torch.channels_last)
        models[arch] = model
    print(f"[INFO] Loaded {list(models)} from {weights_path} on {DEVICE}")
    return models


# ─────────────────────────────────────────────────────────────────────────────
# TRANSFORMS — Matches train.py normalization
# ─────────────────────────────────────────────────────────────────────────────
def get_inference_transforms():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE LOOP — Batched, FP16 on GPU, averages the ensemble's probabilities
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(models: dict, test_dir: str) -> pd.DataFrame:
    test_dataset = DeepfakeDataset(
        test_dir,
        isTrain=False,
        csv_path="",   # every image in test_dir, no CSV needed
        transform=get_inference_transforms(),
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )

    all_ids = test_dataset.labels["image_id"].tolist()
    all_confidences = []

    print(f"[INFO] Running inference on {len(test_dataset)} images (batch_size={BATCH_SIZE})...")

    for batch_imgs, _ in tqdm(test_loader, desc="Running inference"):
        batch_imgs = batch_imgs.to(DEVICE, non_blocking=True)
        if DEVICE.type == "cuda":
            batch_imgs = batch_imgs.to(memory_format=torch.channels_last)

        probs = torch.zeros(batch_imgs.size(0), device=DEVICE)
        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=(DEVICE.type == "cuda")):
            for model in models.values():
                probs += torch.sigmoid(model(batch_imgs).float().squeeze(-1))
        all_confidences.extend((probs / len(models)).cpu().numpy().tolist())

    confidences = np.clip(np.asarray(all_confidences), 0.0, 1.0)
    return pd.DataFrame({
        "image_id":   all_ids,
        "label":      (confidences >= THRESHOLD).astype(int),
        "confidence": confidences,
    })


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    global BATCH_SIZE, THRESHOLD, NUM_WORKERS
    parser = argparse.ArgumentParser(description="Deepfake ML Hackathon — Inference Script")
    parser.add_argument("--test_dir", required=True, help="Path to test images directory")
    parser.add_argument("--weights",  required=True, help="Ensemble weights file, or folder with one file per model")
    parser.add_argument("--output",   default="predictions.csv", help="Output CSV path")
    parser.add_argument("--batch_size",  type=int, default=BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--threshold",   type=float, default=THRESHOLD)
    parser.add_argument("--save_merged", default=None, help="Also save the loaded ensemble as one weights file")
    args = parser.parse_args()

    BATCH_SIZE  = args.batch_size
    NUM_WORKERS = args.num_workers
    THRESHOLD   = args.threshold

    models = load_models(args.weights)
    if args.save_merged:
        torch.save({arch: {k: v.cpu() for k, v in m.state_dict().items()} for arch, m in models.items()}, args.save_merged)
        print(f"[INFO] Saved ensemble weights to {args.save_merged}")

    predictions = run_inference(models, args.test_dir)

    assert "image_id"   in predictions.columns, "Missing image_id column"
    assert "label"      in predictions.columns, "Missing label column"
    assert "confidence" in predictions.columns, "Missing confidence column"
    assert predictions["image_id"].is_unique,   "Duplicate image ids"
    assert predictions["confidence"].between(0, 1).all(), "Confidence values must be in [0, 1]"
    assert predictions["label"].isin([0, 1]).all(),       "Labels must be 0 or 1"

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    predictions.to_csv(args.output, index=False)
    print(f"[DONE] Predictions saved to {args.output}")
    print(f"  Total images: {len(predictions)}")
    print(f"  Predicted FAKE: {predictions['label'].sum()} ({predictions['label'].mean()*100:.1f}%)")
    print(f"  Predicted REAL: {(predictions['label']==0).sum()}")
    print(f"  Mean confidence: {predictions['confidence'].mean():.4f}")


if __name__ == "__main__":
    main()
