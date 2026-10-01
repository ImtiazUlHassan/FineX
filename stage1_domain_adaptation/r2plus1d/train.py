"""
train.py — Stage 1, RGB stream: R(2+1)D-34 (IG-65M) fine-tuning.

Config
──────
  backbone : R2+1D-34 IG65M (full fine-tune, ~21M params)
  frames   : 48 (pyskl UniformSampleFrames — random within-bin per epoch)
  spatial  : 112×112 (RandomCrop train / CenterCrop val)
  batch    : 8 default, pass --batch_size to override
  epochs   : 40
  LR       : 0.01 @ batch=32, scales linearly (SGD+Nesterov+CosineAnnealing per-step)
  grad_clip: 40,  AMP: True
  TTA      : 1 clip × double (orig+flip) = 2 volumes → softmax average
"""

import os, random, argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from model import R2Plus1DRGB
from utils import (
    set_seed, make_loader,
    sample_train, sample_tta,
    transform_train, transform_val, flip_video,
    save_ckpt, load_ckpt,
)


# ════════════════════════════════════════════════════════════════════════════
# Config
# ════════════════════════════════════════════════════════════════════════════

NUM_CLASSES_BY_DATASET = {"gym99": 99, "gym288": 288, "diving48": 48}

parser = argparse.ArgumentParser(description="Stage 1 — RGB stream: R(2+1)D-34 (IG-65M)")
parser.add_argument("--dataset",         required=True, choices=NUM_CLASSES_BY_DATASET)
parser.add_argument("--csv_train",       required=True)
parser.add_argument("--csv_val",         required=True)
parser.add_argument("--video_dir_train", required=True, help="videos at <dir>/<Class>/<FileName>")
parser.add_argument("--video_dir_val",   required=True)
parser.add_argument("--output_dir",      required=True)
parser.add_argument("--batch_size",      type=int, default=8)
args = parser.parse_args()

CSV_TRAIN   = args.csv_train
CSV_VAL     = args.csv_val
VIDEO_TRAIN = args.video_dir_train
VIDEO_VAL   = args.video_dir_val
SAVE_ROOT   = args.output_dir

NUM_CLASSES    = NUM_CLASSES_BY_DATASET[args.dataset]
BATCH_SIZE     = args.batch_size
NUM_EPOCHS     = 40
NUM_VID_FRAMES = 48
FLIP_PROB      = 0.5
TTA_CLIPS      = 1

BASE_LR      = 0.01
BASE_BATCH   = 32
LR           = BASE_LR * (BATCH_SIZE / BASE_BATCH)
MOMENTUM     = 0.9
WEIGHT_DECAY = 1e-4
GRAD_CLIP    = 40.0
USE_AMP      = True

EXP_NAME      = f"{os.path.basename(os.path.normpath(SAVE_ROOT))}_{args.dataset}"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(42)


# ════════════════════════════════════════════════════════════════════════════
# Val — TTA_CLIPS clips × double (orig+flip), softmax-averaged
# ════════════════════════════════════════════════════════════════════════════

def val_with_tta(model):
    model.eval()
    val_loader, totalframes_val, boundaries_val = make_loader(CSV_VAL, VIDEO_VAL, shuffle=False)

    cc = torch.zeros(NUM_CLASSES, device=device)
    ct = torch.zeros(NUM_CLASSES, device=device)
    correct = total = processed = 0
    current_chunks = []

    pbar = tqdm(total=len(boundaries_val), desc=f"  Val TTA×{2*TTA_CLIPS}", leave=False)

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            chunk    = data[0]["videos"][0]
            label_id = int(data[0]["labels"])
            current_chunks.append(chunk)

            if (i + 1) == boundaries_val[processed]:
                T_full   = totalframes_val[processed]
                full_vid = torch.cat(current_chunks, dim=0)[:T_full]

                batch_v = []
                for inds in sample_tta(T_full, NUM_VID_FRAMES, TTA_CLIPS):
                    v_orig = transform_val(full_vid[inds])
                    batch_v.extend([v_orig, flip_video(v_orig)])

                all_probs = []
                for sb in range(0, len(batch_v), 4):
                    vids = torch.stack(batch_v[sb:sb+4]).to(device)
                    with torch.amp.autocast("cuda", enabled=USE_AMP):
                        logits = model(vids)
                    all_probs.append(F.softmax(logits, dim=1))

                pred = torch.cat(all_probs, dim=0).mean(0).argmax().item()
                correct += int(pred == label_id); total += 1
                if pred == label_id: cc[label_id] += 1
                ct[label_id] += 1

                current_chunks = []
                processed += 1
                pbar.update(1)

    pbar.close()
    acc  = 100 * correct / total if total > 0 else 0.0
    mask = ct > 0
    mca  = (cc[mask] / ct[mask]).mean().item() * 100 if mask.any() else 0.0
    return acc, mca


# ════════════════════════════════════════════════════════════════════════════
# Train — one epoch
# ════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, criterion, optimizer, scheduler, scaler, epoch):
    model.train()
    train_loader, totalframes_train, boundaries_train = make_loader(
        CSV_TRAIN, VIDEO_TRAIN, shuffle=True
    )

    sum_loss = correct = total = 0
    cc = torch.zeros(NUM_CLASSES, device=device)
    ct = torch.zeros(NUM_CLASSES, device=device)
    current_chunks         = []
    batch_v, batch_lbl     = [], []
    processed = batchcount = 0

    pbar = tqdm(total=len(boundaries_train), desc=f"  Ep{epoch+1}/{NUM_EPOCHS}", leave=False)

    for i, data in enumerate(train_loader):
        chunk    = data[0]["videos"][0]
        label_id = int(data[0]["labels"])
        current_chunks.append(chunk)

        if (i + 1) == boundaries_train[processed]:
            T_full   = totalframes_train[processed]
            full_vid = torch.cat(current_chunks, dim=0)[:T_full]

            video = transform_train(full_vid[sample_train(T_full, NUM_VID_FRAMES)])
            if random.random() < FLIP_PROB:
                video = flip_video(video)

            batch_v.append(video)
            batch_lbl.append(label_id)
            batchcount += 1

            if batchcount == BATCH_SIZE:
                videos = torch.stack(batch_v).to(device)
                labels = torch.tensor(batch_lbl, device=device)

                optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=USE_AMP):
                    logits = model(videos)
                    loss   = criterion(logits, labels)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                sum_loss += loss.item() * labels.size(0)
                _, pred   = torch.max(logits, 1)
                total    += labels.size(0)
                correct  += (pred == labels).sum().item()
                for t, p in zip(labels, pred):
                    if t == p: cc[t] += 1
                    ct[t] += 1

                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    acc=f"{100*correct/max(total,1):.1f}%",
                    lr=f"{optimizer.param_groups[0]['lr']:.6f}",
                )
                batch_v, batch_lbl, batchcount = [], [], 0

            current_chunks = []
            processed += 1
            pbar.update(1)

    pbar.close()
    avg_loss = sum_loss / max(total, 1)
    acc      = 100 * correct / max(total, 1)
    mask_c   = ct > 0
    mca      = (cc[mask_c] / ct[mask_c]).mean().item() * 100 if mask_c.any() else 0.0
    return avg_loss, acc, mca


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def train():
    weights_dir     = os.path.join(SAVE_ROOT, "weights")
    best_model_path = os.path.join(SAVE_ROOT, "best_model.pth")
    last_ckpt_path  = os.path.join(weights_dir, "last_checkpoint.pth")
    os.makedirs(weights_dir, exist_ok=True)

    total_videos = len(pd.read_csv(CSV_TRAIN))
    steps_per_ep = total_videos // BATCH_SIZE
    total_steps  = NUM_EPOCHS * steps_per_ep
    print(f"[{EXP_NAME}]  {total_videos} videos | {steps_per_ep} steps/ep | "
          f"{NUM_EPOCHS} epochs | LR={LR:.4f} (batch={BATCH_SIZE})")

    model     = R2Plus1DRGB(num_classes=NUM_CLASSES, dropout=0.5).to(device)
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM,
                          weight_decay=WEIGHT_DECAY, nesterov=True)
    criterion = nn.CrossEntropyLoss()
    scaler    = GradScaler("cuda", enabled=USE_AMP)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)

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
            model, criterion, optimizer, scheduler, scaler, epoch
        )
        print(f"  Train  loss={train_loss:.4f}  acc={train_acc:.2f}%  mca={train_mca:.2f}%")

        val_acc, val_mca = val_with_tta(model)
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
            "epoch": epoch+1, "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 2), "train_mca": round(train_mca, 2),
            "val_acc_tta": round(val_acc, 2), "val_mca_tta": round(val_mca, 2),
            "best_mca": round(best_mca, 2), "lr": round(cur_lr, 8),
        }]).to_csv(results_csv, mode="a", header=False, index=False)

    print(f"\n[{EXP_NAME}] Done.  Best val MCA = {best_mca:.2f}%")


if __name__ == "__main__":
    train()
