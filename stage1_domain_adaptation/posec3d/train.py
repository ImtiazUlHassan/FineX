"""
train.py — Stage 1, pose-heatmap stream: PoseC3D (SlowOnly-R50) (joint + limb heatmaps).

Config
──────
  backbone : ResNet3dSlowOnly-R50 (scratch, pyskl-exact)
  stream   : joint+limb (34 channels, GPU-rendered Gaussian heatmaps)
  frames   : 48 (pyskl UniformSampleFrames — random within-bin per epoch)
  heatmap  : 56×56 train / 64×64 val
  batch    : 32 default, pass --batch_size to override
  epochs   : 45  ×  repeat=5
  LR       : 0.4 @ batch=256, scales linearly (SGD+Nesterov+CosineAnnealing per-step)
  grad_clip: 40,  AMP: True
  TTA      : 4 clips × double (orig+flip) = 8 volumes → softmax average
"""

import os, pickle, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from model import SlowOnlyHeatmap
from utils import (
    set_seed,
    sample_tta, extract_keypoints,
    PoseHeatmapDataset, RepeatDataset,
    save_ckpt, load_ckpt,
)


# ════════════════════════════════════════════════════════════════════════════
# Config
# ════════════════════════════════════════════════════════════════════════════

NUM_CLASSES_BY_DATASET = {"gym99": 99, "gym288": 288, "diving48": 48}

parser = argparse.ArgumentParser(description="Stage 1 — pose-heatmap stream: PoseC3D (SlowOnly-R50)")
parser.add_argument("--dataset",    required=True, choices=NUM_CLASSES_BY_DATASET)
parser.add_argument("--pose_pkl",   required=True)
parser.add_argument("--output_dir", required=True)
parser.add_argument("--batch_size", type=int, default=32)
args = parser.parse_args()

POSE_PKL  = args.pose_pkl
SAVE_ROOT = args.output_dir

NUM_CLASSES    = NUM_CLASSES_BY_DATASET[args.dataset]
BATCH_SIZE     = args.batch_size
NUM_EPOCHS     = 45
REPEAT_TIMES   = 5
NUM_VID_FRAMES = 48
NUM_JOINTS     = 17
HEATMAP_TRAIN  = 56
HEATMAP_TEST   = 64
SIGMA          = 2.0
WITH_LIMB      = True
TTA_CLIPS      = 4
FLIP_PROB      = 0.5
NUM_WORKERS    = 8

BASE_LR      = 0.4
BASE_BATCH   = 256
LR           = BASE_LR * (BATCH_SIZE / BASE_BATCH)
MOMENTUM     = 0.9
WEIGHT_DECAY = 3e-4
GRAD_CLIP    = 40.0
USE_AMP      = True

EXP_NAME      = f"{os.path.basename(os.path.normpath(SAVE_ROOT))}_{args.dataset}"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(42)


# ════════════════════════════════════════════════════════════════════════════
# Val — TTA_CLIPS clips × double (orig+flip), softmax-averaged
# ════════════════════════════════════════════════════════════════════════════

def val_with_tta(model, annotations, video_ids, labels_list):
    model.eval()
    cc = torch.zeros(NUM_CLASSES, device=device)
    ct = torch.zeros(NUM_CLASSES, device=device)
    correct = total = 0

    with torch.no_grad():
        pbar = tqdm(zip(video_ids, labels_list), total=len(video_ids),
                    desc=f"  Val TTA×{2*TTA_CLIPS}", leave=False)
        for vid_id, label in pbar:
            item = annotations.get(vid_id)
            if item is not None:
                kp_raw   = item['keypoint']
                T        = max((kp_raw.shape[1] if kp_raw.ndim == 4 else kp_raw.shape[0]), 1)
                all_inds = sample_tta(T, NUM_VID_FRAMES, TTA_CLIPS)
            else:
                all_inds = [np.zeros(NUM_VID_FRAMES, dtype=np.int64)] * TTA_CLIPS

            vols = []
            for inds in all_inds:
                if item is not None:
                    kp_orig = extract_keypoints(item, inds, augment_train=False, do_flip=False)
                    kp_flip = extract_keypoints(item, inds, augment_train=False, do_flip=True)
                else:
                    kp_orig = np.zeros((NUM_VID_FRAMES, NUM_JOINTS, 2), dtype=np.float32)
                    kp_flip = kp_orig.copy()
                vols.append(torch.from_numpy(kp_orig).float())
                vols.append(torch.from_numpy(kp_flip).float())

            clips  = torch.stack(vols).to(device)
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                logits = model(clips, H=HEATMAP_TEST, W=HEATMAP_TEST)
            probs = F.softmax(logits, dim=1).mean(dim=0)
            pred  = probs.argmax().item()

            correct += int(pred == label); total += 1
            if pred == label: cc[label] += 1
            ct[label] += 1

    acc  = 100 * correct / max(total, 1)
    mask = ct > 0
    mca  = (cc[mask] / ct[mask]).mean().item() * 100 if mask.any() else 0.0
    return acc, mca


# ════════════════════════════════════════════════════════════════════════════
# Train — one epoch
# ════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, epoch):
    model.train()
    sum_loss = total = correct = 0
    cc = torch.zeros(NUM_CLASSES, device=device)
    ct = torch.zeros(NUM_CLASSES, device=device)

    pbar = tqdm(loader, desc=f"  Ep{epoch+1}/{NUM_EPOCHS}", leave=False)
    for kp, labels in pbar:
        kp, labels = kp.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=USE_AMP):
            logits = model(kp)
            loss   = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        sum_loss += loss.item() * labels.size(0)
        total    += labels.size(0)
        _, pred   = torch.max(logits, 1)
        correct  += (pred == labels).sum().item()
        for t, p in zip(labels, pred):
            if t == p: cc[t] += 1
            ct[t] += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}",
                         acc=f"{100*correct/max(total,1):.1f}%",
                         lr=f"{optimizer.param_groups[0]['lr']:.5f}")

    avg_loss = sum_loss / max(total, 1)
    acc      = 100 * correct / max(total, 1)
    mask     = ct > 0
    mca      = (cc[mask] / ct[mask]).mean().item() * 100 if mask.any() else 0.0
    return avg_loss, acc, mca


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def train():
    weights_dir     = os.path.join(SAVE_ROOT, "weights")
    best_model_path = os.path.join(SAVE_ROOT, "best_model.pth")
    last_ckpt_path  = os.path.join(weights_dir, "last_checkpoint.pth")
    os.makedirs(weights_dir, exist_ok=True)

    print("[Data] Loading pose annotations...")
    with open(POSE_PKL, 'rb') as f:
        data = pickle.load(f)

    ann_list    = data['annotations']
    annotations = {a['frame_dir']: a for a in ann_list}

    val_ids    = [fd for fd in data['split']['val'] if fd in annotations]
    val_labels = [int(annotations[fd]['label']) for fd in val_ids]

    train_base   = PoseHeatmapDataset(POSE_PKL, 'train', NUM_VID_FRAMES, FLIP_PROB)
    train_ds     = RepeatDataset(train_base, REPEAT_TIMES)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)

    total_steps = NUM_EPOCHS * len(train_loader)
    print(f"[{EXP_NAME}]  Train×{REPEAT_TIMES}: {len(train_ds)}  Val: {len(val_ids)}  "
          f"Batches/ep: {len(train_loader)} | LR={LR:.4f} (batch={BATCH_SIZE})")

    model     = SlowOnlyHeatmap(
        num_classes=NUM_CLASSES, J=NUM_JOINTS,
        H=HEATMAP_TRAIN, W=HEATMAP_TRAIN,
        sigma=SIGMA, with_limb=WITH_LIMB,
    ).to(device)
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM,
                          weight_decay=WEIGHT_DECAY, nesterov=True)
    criterion = nn.CrossEntropyLoss()
    scaler    = GradScaler("cuda", enabled=USE_AMP)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=0)

    start_epoch = best_mca = 0.0
    if os.path.isfile(last_ckpt_path):
        start_epoch, best_mca = load_ckpt(last_ckpt_path, model, optimizer, scheduler, device)
        start_epoch += 1
        print(f"[Resume] epoch {int(start_epoch)}  best_mca={best_mca:.2f}%")


    results_csv = os.path.join(weights_dir, "results.csv")
    if not os.path.isfile(results_csv):
        pd.DataFrame(columns=["epoch", "train_loss", "train_acc", "train_mca",
                               "val_acc_tta", "val_mca_tta", "best_mca", "lr"]
                    ).to_csv(results_csv, index=False)

    for epoch in range(int(start_epoch), NUM_EPOCHS):
        print(f"\n{'='*60}\n  {EXP_NAME}  Epoch {epoch+1}/{NUM_EPOCHS}\n{'='*60}")

        train_loss, train_acc, train_mca = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, epoch
        )
        print(f"  Train  loss={train_loss:.4f}  acc={train_acc:.2f}%  mca={train_mca:.2f}%")

        val_acc, val_mca = val_with_tta(model, annotations, val_ids, val_labels)
        is_best = val_mca > best_mca
        print(f"  Val    acc={val_acc:.2f}%  mca={val_mca:.2f}%"
              f"{'  ★ best ★' if is_best else ''}")

        cur_lr  = optimizer.param_groups[0]["lr"]
        metrics = {"train_loss": train_loss, "val_mca": val_mca}
        if is_best:
            best_mca = val_mca
        # save last AFTER updating best_mca so a resumed run keeps the true best
        save_ckpt(last_ckpt_path, epoch, model, optimizer, scheduler, best_mca, metrics)
        if is_best:
            save_ckpt(best_model_path, epoch, model, optimizer, scheduler, best_mca, metrics)
            print(f"  Saved best model  mca={best_mca:.2f}%")

        pd.DataFrame([{
            "epoch": epoch + 1, "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 2), "train_mca": round(train_mca, 2),
            "val_acc_tta": round(val_acc, 2), "val_mca_tta": round(val_mca, 2),
            "best_mca": round(best_mca, 2), "lr": round(cur_lr, 8),
        }]).to_csv(results_csv, mode="a", header=False, index=False)

    print(f"\n[{EXP_NAME}] Done.  Best val MCA = {best_mca:.2f}%")


if __name__ == "__main__":
    train()
