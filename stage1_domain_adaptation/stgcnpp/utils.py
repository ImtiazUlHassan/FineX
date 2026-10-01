"""
utils.py — dataset, seed, and checkpoint helpers.
"""

import pickle, random
import numpy as np
import torch
from torch_geometric.data import Data, Dataset


# ════════════════════════════════════════════════════════════════════════════
# Seed
# ════════════════════════════════════════════════════════════════════════════

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ════════════════════════════════════════════════════════════════════════════
# Dataset
# ════════════════════════════════════════════════════════════════════════════

class SkeletonGraphDataset(Dataset):
    # 13-Joint Mapping: COCO-17 → 13
    # 0:Nose, 1:L-Shldr, 2:R-Shldr, 3:L-Elb, 4:R-Elb, 5:L-Wrst, 6:R-Wrst,
    # 7:L-Hip, 8:R-Hip, 9:L-Knee, 10:R-Knee, 11:L-Ank, 12:R-Ank
    KEEP_INDICES = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

    def __init__(self, pkl_path, split_type='train', num_frame=100):
        super().__init__()
        self.num_frame = num_frame

        with open(pkl_path, 'rb') as f:
            full_data = pickle.load(f)

        split_ids = set(full_data['split'][split_type])
        self.annotations = [
            ann for ann in full_data['annotations']
            if ann['frame_dir'] in split_ids
        ]

    def len(self):
        return len(self.annotations)

    def get(self, idx):
        item  = self.annotations[idx]
        kp    = item['keypoint'][0]   # (T, 17, 2)
        img_h, img_w = item['img_shape']

        # Temporal uniform sampling
        T_src = kp.shape[0]
        if T_src > 0:
            inds = np.linspace(0, T_src - 1, self.num_frame).astype(int)
            kp   = kp[inds]
        else:
            kp = np.zeros((self.num_frame, 17, 2))

        # Filter to 13 joints: (T, 13, 2)
        kp_13 = kp[:, self.KEEP_INDICES, :2]
        nodes = torch.from_numpy(kp_13).float()

        # Global normalisation: pixel → [0, 1]
        nodes[:, :, 0] /= max(img_w, 1)
        nodes[:, :, 1] /= max(img_h, 1)

        # Local normalisation: centre on mid-hip (indices 7=L-Hip, 8=R-Hip)
        mid_hip = (nodes[:, 7, :] + nodes[:, 8, :]) / 2
        nodes   = nodes - mid_hip.unsqueeze(1)

        # Flatten for PyG: (T*13, 2)
        x = nodes.reshape(-1, 2)
        y = torch.tensor([item['label']], dtype=torch.long)
        return Data(x=x, y=y)


# ════════════════════════════════════════════════════════════════════════════
# Checkpoint helpers
# ════════════════════════════════════════════════════════════════════════════

def save_ckpt(path, epoch, model, optimizer, scheduler, best_mca, metrics):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_mca": best_mca,
        **metrics,
    }, path)


def load_ckpt(path, model, optimizer, scheduler, device):
    ck = torch.load(path, map_location=device)
    model.load_state_dict(ck["model_state_dict"])
    optimizer.load_state_dict(ck["optimizer_state_dict"])
    scheduler.load_state_dict(ck["scheduler_state_dict"])
    return ck["epoch"], ck.get("best_mca", 0.0)
