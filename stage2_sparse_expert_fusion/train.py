"""
train.py — Stage 2: train the FineX fusion module on cached Stage 1 features.

Loss (Eq. 8): label-smoothed CE (eps = 0.1) + lambda_lb * L_lb (Eq. 9, lambda_lb = 0.1).
Adam, base lr 3e-4 scaled by batch/256, weight decay 1e-3, cosine schedule,
gradient clipping 1.0, up to 80 epochs with early stopping on val Top-1.
"""
import os, random, argparse
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import recall_score

from dataset import make_loaders, NUM_CLASSES_BY_DATASET
from model import PairwiseCrossAttnMoE, load_balance_loss

EPOCHS       = 80
BASE_LR      = 3e-4
WEIGHT_DECAY = 1e-3
EARLY_STOP   = 6
LB_COEFF     = 0.1   # lambda_lb in Eq. (8)
LABEL_SMOOTH = 0.1   # eps in Eq. (8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval_split(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for fr, fs, fg, labels in loader:
            fr, fs, fg = fr.to(device), fs.to(device), fg.to(device)
            out, _, _, _ = model(fr, fs, fg)
            all_preds.extend(out.argmax(1).cpu().tolist())
            all_labels.extend(labels.tolist())
    acc = 100.0 * np.mean(np.array(all_preds) == np.array(all_labels))
    mca = recall_score(all_labels, all_preds, average="macro", zero_division=0) * 100
    return acc, mca


def main():
    parser = argparse.ArgumentParser(description="FineX Stage 2 — cross-modal sparse expert fusion")
    parser.add_argument("--dataset",    required=True, choices=NUM_CLASSES_BY_DATASET)
    parser.add_argument("--feat_dir",   required=True, help="folder with train_logits.npz / val_logits.npz")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    print(f"Seed: {args.seed}")

    num_classes = NUM_CLASSES_BY_DATASET[args.dataset]
    train_loader, val_loader = make_loaders(args.feat_dir, num_classes, args.batch_size)
    model = PairwiseCrossAttnMoE(num_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {n_params/1e6:.2f}M")

    lr = BASE_LR * (args.batch_size / 256)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

    os.makedirs(args.output_dir, exist_ok=True)
    best_path = os.path.join(args.output_dir, "best_model.pth")
    best_acc = 0.0; best_mca = 0.0; no_improve = 0

    for epoch in range(EPOCHS):
        model.train()
        all_preds, all_labels = [], []
        for fr, fs, fg, labels in train_loader:
            fr, fs, fg, labels = fr.to(device), fs.to(device), fg.to(device), labels.to(device)
            optimizer.zero_grad()
            out, rl, _, _ = model(fr, fs, fg)
            loss = criterion(out, labels) + LB_COEFF * load_balance_loss(rl)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            all_preds.extend(out.argmax(1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

        tr_mca = recall_score(all_labels, all_preds, average="macro", zero_division=0) * 100
        val_acc, val_mca = eval_split(model, val_loader)
        scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc; best_mca = val_mca; no_improve = 0
            torch.save(model.state_dict(), best_path)
        else:
            no_improve += 1

        print(f"Ep{epoch+1:02d}: train_mca={tr_mca:.2f}% val={val_acc:.2f}% mca={val_mca:.2f}% "
              f"best={best_acc:.2f}% patience={no_improve}/{EARLY_STOP}")
        if no_improve >= EARLY_STOP:
            print("Early stop."); break

    print(f"\nFinal — Top-1: {best_acc:.2f}%  MCA: {best_mca:.2f}%  ({best_path})")


if __name__ == "__main__":
    main()
