"""
utils.py — sampling, pose augmentation, dataset, and checkpoint helpers.
"""

import pickle, random
import numpy as np
import torch
from torch.utils.data import Dataset


# ════════════════════════════════════════════════════════════════════════════
# Seed
# ════════════════════════════════════════════════════════════════════════════

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark     = True


# ════════════════════════════════════════════════════════════════════════════
# Sampling  (pyskl UniformSampleFrames)
# ════════════════════════════════════════════════════════════════════════════

def _sample_one(num_frames: int, clip_len: int, rng) -> np.ndarray:
    if num_frames < clip_len:
        start = rng.randint(0, max(num_frames, 1))
        inds  = np.arange(start, start + clip_len) % num_frames
    elif num_frames < 2 * clip_len:
        basic  = np.arange(clip_len)
        picks  = rng.choice(clip_len + 1, num_frames - clip_len, replace=False)
        offset = np.zeros(clip_len + 1, dtype=np.int64)
        offset[picks] = 1
        inds   = basic + np.cumsum(offset)[:-1]
    else:
        bids   = np.array([i * num_frames // clip_len for i in range(clip_len + 1)])
        bst    = bids[:clip_len]
        bsize  = np.diff(bids)
        offset = np.array([rng.randint(0, max(b, 1)) for b in bsize])
        inds   = bst + offset
    return np.mod(inds, num_frames).astype(np.int64)


def sample_train(num_frames: int, clip_len: int) -> np.ndarray:
    return _sample_one(num_frames, clip_len, np.random.RandomState())


def sample_val(num_frames: int, clip_len: int, seed: int = 255) -> np.ndarray:
    return _sample_one(num_frames, clip_len, np.random.RandomState(seed))


def sample_tta(num_frames: int, clip_len: int, num_clips: int = 10, seed: int = 255) -> list:
    rng = np.random.RandomState(seed)
    return [_sample_one(num_frames, clip_len, rng) for _ in range(num_clips)]


# ════════════════════════════════════════════════════════════════════════════
# Spatial augmentation helpers
# (equivalent to pyskl PoseCompact + RandomResizedCrop / center crop)
# ════════════════════════════════════════════════════════════════════════════

_AREA_RANGE = (0.56, 1.0)
_PADDING    = 0.25

LEFT_KP  = [1, 3,  5,  7,  9, 11, 13, 15]
RIGHT_KP = [2, 4,  6,  8, 10, 12, 14, 16]


def _pose_compact_bbox(nodes):
    """nodes : (T, J, 2) float32 in [0, 1] → (x0, y0, size) square crop."""
    x_min, x_max = float(nodes[:, :, 0].min()), float(nodes[:, :, 0].max())
    y_min, y_max = float(nodes[:, :, 1].min()), float(nodes[:, :, 1].max())
    cx   = (x_min + x_max) / 2
    cy   = (y_min + y_max) / 2
    half = max(x_max - x_min, y_max - y_min) / 2 * (1 + _PADDING)
    half = min(half, 0.5)
    x0   = float(np.clip(cx - half, 0.0, 1.0 - 2 * half)) if 2 * half <= 1 else 0.0
    y0   = float(np.clip(cy - half, 0.0, 1.0 - 2 * half)) if 2 * half <= 1 else 0.0
    return x0, y0, min(2 * half, 1.0)


def pose_compact_train(nodes):
    """Train aug: PoseCompact + RandomResizedCrop(area=0.56-1.0). nodes: (T,J,2)→(T,J,2)."""
    x0_c, y0_c, size_c = _pose_compact_bbox(nodes)
    scale   = np.random.uniform(np.sqrt(_AREA_RANGE[0]), np.sqrt(_AREA_RANGE[1]))
    crop    = size_c * scale
    max_off = size_c - crop
    off_x   = np.random.uniform(0, max(max_off, 0))
    off_y   = np.random.uniform(0, max(max_off, 0))
    cx0, cy0 = x0_c + off_x, y0_c + off_y
    out = nodes.copy()
    out[:, :, 0] = (nodes[:, :, 0] - cx0) / (crop + 1e-8)
    out[:, :, 1] = (nodes[:, :, 1] - cy0) / (crop + 1e-8)
    return np.clip(out, 0.0, 1.0)


def pose_compact_val(nodes):
    """Val/Test: PoseCompact only (center, no jitter). nodes: (T,J,2)→(T,J,2)."""
    x0, y0, size = _pose_compact_bbox(nodes)
    out = nodes.copy()
    out[:, :, 0] = (nodes[:, :, 0] - x0) / (size + 1e-8)
    out[:, :, 1] = (nodes[:, :, 1] - y0) / (size + 1e-8)
    return np.clip(out, 0.0, 1.0)


def flip_keypoints(nodes):
    """Horizontal flip: x → 1−x, swap left ↔ right joints. nodes: (T,J,2)→(T,J,2)."""
    out = nodes.copy()
    out[:, :, 0] = 1.0 - out[:, :, 0]
    out[:, LEFT_KP, :], out[:, RIGHT_KP, :] = \
        out[:, RIGHT_KP, :].copy(), out[:, LEFT_KP, :].copy()
    return out


def extract_keypoints(item, frame_inds, augment_train=False, do_flip=False):
    """
    Returns (T, J, 2) float32 in [0,1] after spatial aug.
    augment_train=True  → PoseCompact + RandomResizedCrop  (train)
    augment_train=False → PoseCompact center only           (val/test)
    do_flip             → horizontal flip                   (double TTA)
    """
    kp = item["keypoint"]
    if kp.ndim == 4:
        kp = kp[0]
    T_src = max(kp.shape[0], 1)
    inds  = np.clip(frame_inds, 0, T_src - 1)
    kp    = kp[inds].astype(np.float32) if kp.shape[0] > 0 \
            else np.zeros((len(inds), 17, 2), dtype=np.float32)
    nodes = kp[:, :17, :2].copy()
    h, w  = item["img_shape"]
    nodes[:, :, 0] /= max(w, 1)
    nodes[:, :, 1] /= max(h, 1)
    nodes = np.clip(nodes, 0.0, 1.0)
    nodes = pose_compact_train(nodes) if augment_train else pose_compact_val(nodes)
    if do_flip:
        nodes = flip_keypoints(nodes)
    return nodes


# ════════════════════════════════════════════════════════════════════════════
# Dataset
# PKL format: annotations is a list of dicts with 'frame_dir', 'keypoint',
#             'img_shape', 'label' keys.
# ════════════════════════════════════════════════════════════════════════════

class PoseHeatmapDataset(Dataset):
    def __init__(self, pkl_path, split_type='train', clip_len=48, flip_prob=0.5):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        ann_dict  = {a['frame_dir']: a for a in data['annotations']}
        split_ids = set(data['split'][split_type])
        self.samples   = [(fd, ann_dict[fd]) for fd in split_ids if fd in ann_dict]
        self.clip_len  = clip_len
        self.is_train  = (split_type == 'train')
        self.flip_prob = flip_prob if self.is_train else 0.0
        print(f"  [{split_type}] {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        _, item    = self.samples[idx]
        label      = int(item['label'])
        kp_raw     = item['keypoint']
        T          = max((kp_raw.shape[1] if kp_raw.ndim == 4 else kp_raw.shape[0]), 1)
        frame_inds = sample_train(T, self.clip_len) if self.is_train \
                     else sample_val(T, self.clip_len)
        do_flip    = self.is_train and (random.random() < self.flip_prob)
        nodes      = extract_keypoints(item, frame_inds,
                                       augment_train=self.is_train, do_flip=do_flip)
        return torch.from_numpy(nodes).float(), label


class RepeatDataset(Dataset):
    def __init__(self, ds, times):
        self.ds    = ds
        self.times = times

    def __len__(self):          return len(self.ds) * self.times
    def __getitem__(self, idx): return self.ds[idx % len(self.ds)]


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
