"""
dataset.py — loader for the cached Stage 1 features (train_logits.npz / val_logits.npz).
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

NUM_CLASSES_BY_DATASET = {"gym99": 99, "gym288": 288, "diving48": 48}


class MoEDataset(Dataset):
    def __init__(self, feat_dir, split):
        npz = np.load(os.path.join(feat_dir, f"{split}_logits.npz"))
        self.feat_r21d  = torch.from_numpy(npz["feat_r21d"].astype(np.float32))
        self.feat_slow  = torch.from_numpy(npz["feat_slow"].astype(np.float32))
        self.feat_stgcn = torch.from_numpy(npz["feat_stgcn"].astype(np.float32))
        self.labels     = torch.from_numpy(npz["labels"].astype(np.int64))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.feat_r21d[i], self.feat_slow[i], self.feat_stgcn[i], self.labels[i]


def make_loaders(feat_dir, num_classes, batch_size=64):
    """Class-balanced (inverse-frequency) sampling for train; sequential val."""
    train_ds = MoEDataset(feat_dir, "train")
    val_ds   = MoEDataset(feat_dir, "val")
    labels_np    = train_ds.labels.numpy()
    class_counts = np.bincount(labels_np, minlength=num_classes).astype(np.float32)
    class_counts = np.where(class_counts == 0, 1, class_counts)
    sample_w     = 1.0 / class_counts[labels_np]
    sampler = WeightedRandomSampler(torch.from_numpy(sample_w), len(train_ds), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,   num_workers=4, pin_memory=True)
    return train_loader, val_loader


