"""
utils.py — DALI pipeline, sampling, transforms, and checkpoint helpers.
"""

import os, random
import numpy as np
import torch
import pandas as pd
from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.fn as fn
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from torchvision.transforms._transforms_video import NormalizeVideo, CenterCropVideo


SEQ_LEN     = 40
NUM_WORKERS = 4
IMG_SIZE    = 112
MEAN        = [0.45, 0.45, 0.45]
STD         = [0.225, 0.225, 0.225]

_normalize   = NormalizeVideo(MEAN, STD)
_center_crop = CenterCropVideo(IMG_SIZE)


# ════════════════════════════════════════════════════════════════════════════
# Seed
# ════════════════════════════════════════════════════════════════════════════

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
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
# DALI pipeline
# ════════════════════════════════════════════════════════════════════════════

@pipeline_def
def video_pipe(filenames, labels):
    videos, lbs = fn.readers.video_resize(
        device="gpu",
        filenames=filenames,
        labels=labels,
        sequence_length=SEQ_LEN,
        shard_id=0, num_shards=1,
        random_shuffle=False,
        pad_sequences=True,
        enable_frame_num=False,
        name="Reader",
        resize_shorter=128,
    )
    info = fn.get_property(videos, key="source_info")
    info = fn.pad(info)
    return videos / 255.0, lbs, info


def decode_filename(info_field) -> str:
    return "".join(chr(i) for i in info_field)


def load_video_metadata(csv_path: str, video_root: str, shuffle: bool):
    df = pd.read_csv(csv_path)
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)
    file_paths  = df.apply(
        lambda r: os.path.join(video_root, str(r["Class"]), r["FileName"]), axis=1
    ).tolist()
    labels      = df["ClassEncoded"].tolist()
    totalframes = df["TotalFrames"].tolist()
    boundaries  = list(np.cumsum([int(np.ceil(tf / SEQ_LEN)) for tf in totalframes]))
    return file_paths, labels, totalframes, boundaries


def make_loader(csv_path: str, video_root: str, shuffle: bool):
    fps, lbs, tfs, bnds = load_video_metadata(csv_path, video_root, shuffle)
    pipe   = video_pipe(batch_size=1, num_threads=NUM_WORKERS, device_id=0,
                        filenames=fps, labels=lbs)
    loader = DALIGenericIterator([pipe], ["videos", "labels", "info"], reader_name="Reader")
    return loader, tfs, bnds


# ════════════════════════════════════════════════════════════════════════════
# Video transforms
# ════════════════════════════════════════════════════════════════════════════

def transform_train(frames: torch.Tensor) -> torch.Tensor:
    """(T,H,W,C) GPU float → (3,T,112,112) normalised, random crop."""
    x = frames.permute(3, 0, 1, 2)
    _, _, H, W = x.shape
    i = random.randint(0, max(H - IMG_SIZE, 0))
    j = random.randint(0, max(W - IMG_SIZE, 0))
    return _normalize(x[:, :, i:i+IMG_SIZE, j:j+IMG_SIZE])


def transform_val(frames: torch.Tensor) -> torch.Tensor:
    """(T,H,W,C) GPU float → (3,T,112,112) normalised, center crop."""
    return _normalize(_center_crop(frames.permute(3, 0, 1, 2)))


def flip_video(x: torch.Tensor) -> torch.Tensor:
    return torch.flip(x, dims=[-1])


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
