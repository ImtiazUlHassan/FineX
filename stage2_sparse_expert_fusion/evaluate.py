"""
evaluate.py — Top-1 and Mean Class Accuracy (MCA) of a trained FineX fusion model.
"""
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import recall_score

from dataset import MoEDataset, NUM_CLASSES_BY_DATASET
from model import PairwiseCrossAttnMoE


def main():
    parser = argparse.ArgumentParser(description="Evaluate FineX Stage 2")
    parser.add_argument("--dataset",  required=True, choices=NUM_CLASSES_BY_DATASET)
    parser.add_argument("--feat_dir", required=True, help="folder with val_logits.npz")
    parser.add_argument("--weights",  required=True, help="best_model.pth from train.py")
    parser.add_argument("--split",    default="val")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PairwiseCrossAttnMoE(NUM_CLASSES_BY_DATASET[args.dataset]).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    loader = DataLoader(MoEDataset(args.feat_dir, args.split), batch_size=256, shuffle=False)
    preds, labels = [], []
    with torch.no_grad():
        for fr, fs, fg, y in loader:
            out, _, _, _ = model(fr.to(device), fs.to(device), fg.to(device))
            preds.extend(out.argmax(1).cpu().tolist())
            labels.extend(y.tolist())

    top1 = 100.0 * np.mean(np.array(preds) == np.array(labels))
    mca  = 100.0 * recall_score(labels, preds, average="macro", zero_division=0)
    print(f"Dataset : {args.dataset}  ({args.split}, N={len(labels)})")
    print(f"Top-1   : {top1:.2f}%")
    print(f"MCA     : {mca:.2f}%")


if __name__ == "__main__":
    main()
