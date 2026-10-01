"""
train.py — Stage 1, skeletal-graph stream: ST-GCN++ (13-joint skeleton).

Config
──────
  backbone : ST-GCN++ (9 blocks, MSTCN, 256-dim features)
  joints   : 13 (COCO-17 → 13, mid-hip centred)
  frames   : 100 (uniform temporal sampling)
  batch    : 128 default, pass --batch_size to override
  epochs   : 150
  LR       : 0.1 @ batch=32, scales linearly (SGD+Nesterov+CosineAnnealing per-epoch)
  grad_clip: 1.0,  AMP: False
"""

import os, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader
from sklearn.metrics import recall_score
from tqdm import tqdm

from model import STGCNPlusPlus
from utils import (
    set_seed,
    SkeletonGraphDataset,
    save_ckpt, load_ckpt,
)


# ════════════════════════════════════════════════════════════════════════════
# Config
# ════════════════════════════════════════════════════════════════════════════

NUM_CLASSES_BY_DATASET = {"gym99": 99, "gym288": 288, "diving48": 48}

parser = argparse.ArgumentParser(description="Stage 1 — skeletal-graph stream: ST-GCN++")
parser.add_argument("--dataset",    required=True, choices=NUM_CLASSES_BY_DATASET)
parser.add_argument("--pose_pkl",   required=True)
parser.add_argument("--output_dir", required=True)
parser.add_argument("--batch_size", type=int, default=128)
args = parser.parse_args()

POSE_PKL  = args.pose_pkl
SAVE_ROOT = args.output_dir

NUM_CLASSES  = NUM_CLASSES_BY_DATASET[args.dataset]
BATCH_SIZE   = args.batch_size
NUM_EPOCHS   = 150
NUM_FRAMES   = 100
NUM_JOINTS   = 13
NUM_WORKERS  = 4

BASE_LR      = 0.1
BASE_BATCH   = 32
LR           = BASE_LR * (BATCH_SIZE / BASE_BATCH)
MOMENTUM     = 0.9
WEIGHT_DECAY = 5e-4
GRAD_CLIP    = 1.0

EXP_NAME      = f"{os.path.basename(os.path.normpath(SAVE_ROOT))}_{args.dataset}"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(42)


# ════════════════════════════════════════════════════════════════════════════
# Train / Val — one epoch
# ════════════════════════════════════════════════════════════════════════════

def run_epoch(model, loader, criterion, optimizer, phase, epoch):
    if phase == "train":
        model.train()
    else:
        model.eval()

    total_loss = 0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc=f"  {phase.capitalize()} Ep{epoch}/{NUM_EPOCHS}", leave=False)
    for data in pbar:
        data = data.to(device)
        with torch.set_grad_enabled(phase == "train"):
            out  = model(data)
            loss = criterion(out, data.y.view(-1))
            if phase == "train":
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

        total_loss += loss.item()
        preds = out.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(data.y.cpu().numpy())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    acc = float(np.mean(np.array(all_preds) == np.array(all_labels)))
    mca = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, acc, mca


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def train():
    weights_dir     = os.path.join(SAVE_ROOT, "weights")
    best_model_path = os.path.join(SAVE_ROOT, "best_model.pth")
    last_ckpt_path  = os.path.join(weights_dir, "last_checkpoint.pth")
    os.makedirs(weights_dir, exist_ok=True)

    print("[Data] Loading datasets...")
    train_dataset = SkeletonGraphDataset(POSE_PKL, split_type="train", num_frame=NUM_FRAMES)
    val_dataset   = SkeletonGraphDataset(POSE_PKL, split_type="val",   num_frame=NUM_FRAMES)
    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                               shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
    val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                               shuffle=False, num_workers=NUM_WORKERS)
    print(f"  Train: {len(train_dataset)}  Val: {len(val_dataset)}")

    model = STGCNPlusPlus(
        num_classes=NUM_CLASSES, in_channels=2,
        num_point=NUM_JOINTS, num_frame=NUM_FRAMES,
    ).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM,
                                weight_decay=WEIGHT_DECAY, nesterov=True)
    criterion = nn.CrossEntropyLoss()
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    start_epoch = best_mca = 0.0
    if os.path.isfile(last_ckpt_path):
        start_epoch, best_mca = load_ckpt(last_ckpt_path, model, optimizer, scheduler, device)
        start_epoch += 1
        print(f"[Resume] epoch {int(start_epoch)}  best_mca={best_mca:.4f}")


    results_csv = os.path.join(weights_dir, "results.csv")
    if not os.path.isfile(results_csv):
        pd.DataFrame(columns=["epoch", "train_loss", "train_acc", "train_mca",
                               "val_loss", "val_acc", "val_mca", "best_mca", "lr"]
                    ).to_csv(results_csv, index=False)

    print(f"[{EXP_NAME}]  LR={LR:.4f} (batch={BATCH_SIZE})")

    try:
        for epoch in range(int(start_epoch), NUM_EPOCHS):
            print(f"\n{'='*60}\n  {EXP_NAME}  Epoch {epoch+1}/{NUM_EPOCHS}\n{'='*60}")

            tr_loss, tr_acc, tr_mca = run_epoch(
                model, train_loader, criterion, optimizer, "train", epoch + 1
            )
            val_loss, val_acc, val_mca = run_epoch(
                model, val_loader, criterion, optimizer, "val", epoch + 1
            )
            scheduler.step()

            is_best = val_mca > best_mca
            cur_lr  = optimizer.param_groups[0]["lr"]
            metrics = {"train_loss": tr_loss, "val_loss": val_loss, "val_mca": val_mca}

            if is_best:
                best_mca = val_mca
            # save last AFTER updating best_mca so a resumed run keeps the true best
            save_ckpt(last_ckpt_path, epoch, model, optimizer, scheduler, best_mca, metrics)
            if is_best:
                save_ckpt(best_model_path, epoch, model, optimizer, scheduler, best_mca, metrics)

            print(f"  Train  loss={tr_loss:.4f}  acc={tr_acc:.4f}  mca={tr_mca:.4f}")
            print(f"  Val    loss={val_loss:.4f}  acc={val_acc:.4f}  mca={val_mca:.4f}"
                  f"{'  ★ best ★' if is_best else ''}")

            pd.DataFrame([{
                "epoch": epoch + 1,
                "train_loss": round(tr_loss, 4), "train_acc": round(tr_acc, 4),
                "train_mca": round(tr_mca, 4),
                "val_loss": round(val_loss, 4),  "val_acc": round(val_acc, 4),
                "val_mca": round(val_mca, 4),    "best_mca": round(best_mca, 4),
                "lr": round(cur_lr, 8),
            }]).to_csv(results_csv, mode="a", header=False, index=False)

    except KeyboardInterrupt:
        print("\nTraining interrupted.")

    print(f"\n[{EXP_NAME}] Done.  Best val MCA = {best_mca:.4f}")


if __name__ == "__main__":
    train()
