"""
extract_features.py — cache Stage 1 features for Stage 2 fusion.
════════════════════════════════════════════════════════════════════════════════
Runs the three frozen Stage 1 experts in eval mode on the train and val splits and
stores the input to each expert's final Linear layer (its penultimate feature).

R(2+1)D       : clips from the split CSV (row order defines the output order)
PoseC3D       : poses from the .pkl split keys (data['split']['train'/'val'])
ST-GCN++      : poses from the .pkl split keys

Outputs (--output_dir):
  train_logits.npz  — feat_r21d (N,512), feat_slow (N,512), feat_stgcn (N,256),
                       logit_r21d (N,C), logit_slow (N,C), logit_stgcn (N,C),
                       labels (N,)
  val_logits.npz    — same schema
"""

import os
import argparse
import pickle
import importlib.util

import numpy as np
import pandas as pd
import torch
from scipy.stats import mode
from sklearn.metrics import recall_score

from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.fn as fn
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from torchvision.transforms._transforms_video import NormalizeVideo, CenterCropVideo
from tqdm import tqdm


# ════════════════════════════════════════════════════════════════════════════
# Path setup — load each expert's model.py by path (all named model.py)
# ════════════════════════════════════════════════════════════════════════════

_HERE    = os.path.dirname(os.path.abspath(__file__))
_EXPERTS = os.path.dirname(_HERE)   # stage1_domain_adaptation/


def _load_module(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_r21d_mod  = _load_module(os.path.join(_EXPERTS, "r2plus1d", "model.py"), "model_r21d")
_slow_mod  = _load_module(os.path.join(_EXPERTS, "posec3d",  "model.py"), "model_slow")
_stgcn_mod = _load_module(os.path.join(_EXPERTS, "stgcnpp",  "model.py"), "model_stgcn")

R2Plus1DRGB     = _r21d_mod.R2Plus1DRGB
SlowOnlyHeatmap = _slow_mod.SlowOnlyHeatmap
STGCNPlusPlus   = _stgcn_mod.STGCNPlusPlus


# ════════════════════════════════════════════════════════════════════════════
# Config
# ════════════════════════════════════════════════════════════════════════════

NUM_CLASSES_BY_DATASET = {"gym99": 99, "gym288": 288, "diving48": 48}

parser = argparse.ArgumentParser(description="Extract and cache Stage 1 features")
parser.add_argument("--dataset",         required=True, choices=NUM_CLASSES_BY_DATASET)
parser.add_argument("--csv_train",       required=True)
parser.add_argument("--csv_val",         required=True)
parser.add_argument("--video_dir_train", required=True)
parser.add_argument("--video_dir_val",   required=True)
parser.add_argument("--pose_pkl",        required=True)
parser.add_argument("--r2plus1d_ckpt",   required=True)
parser.add_argument("--posec3d_ckpt",    required=True)
parser.add_argument("--stgcnpp_ckpt",    required=True)
parser.add_argument("--output_dir",      required=True)
args = parser.parse_args()

# R(2+1)D uses CSV-based splits
CSV_TRAIN   = args.csv_train
CSV_VAL     = args.csv_val
VIDEO_TRAIN = args.video_dir_train
VIDEO_VAL   = args.video_dir_val

# PoseC3D + ST-GCN++ use the .pkl split keys
POSE_PKL    = args.pose_pkl

R21D_CKPT     = args.r2plus1d_ckpt
SLOWONLY_CKPT = args.posec3d_ckpt
STGCN_CKPT    = args.stgcnpp_ckpt

FEAT_DIR      = args.output_dir

NUM_CLASSES     = NUM_CLASSES_BY_DATASET[args.dataset]
NUM_VID_FRAMES  = 48
NUM_POSE_FRAMES = 100
NUM_JOINTS      = 17
IMG_SIZE        = 112
SEQ_LEN         = 40
HEATMAP_SIZE    = 64    # SlowOnly val: 64×64

MEAN = [0.45, 0.45, 0.45]
STD  = [0.225, 0.225, 0.225]

KEEP_INDICES = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]  # COCO-17 → 13

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(FEAT_DIR, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════
# Sampling  (pyskl UniformSampleFrames — deterministic val)
# ════════════════════════════════════════════════════════════════════════════

def _sample_one(num_frames, clip_len, rng):
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
        bst    = bids[:clip_len]; bsize = np.diff(bids)
        offset = np.array([rng.randint(0, max(b, 1)) for b in bsize])
        inds   = bst + offset
    return np.mod(inds, num_frames).astype(np.int64)


def sample_val(num_frames, clip_len, seed=255):
    return _sample_one(num_frames, clip_len, np.random.RandomState(seed))


# ════════════════════════════════════════════════════════════════════════════
# DALI pipeline  (identical to r2plus1/utils.py)
# ════════════════════════════════════════════════════════════════════════════

@pipeline_def
def video_pipe(filenames, labels):
    videos, lbs = fn.readers.video_resize(
        device="gpu", filenames=filenames, labels=labels,
        sequence_length=SEQ_LEN, shard_id=0, num_shards=1,
        random_shuffle=False, pad_sequences=True,
        enable_frame_num=False, name="Reader", resize_shorter=128,
    )
    info = fn.get_property(videos, key="source_info")
    info = fn.pad(info)
    return videos / 255.0, lbs, info


_normalize   = NormalizeVideo(MEAN, STD)
_center_crop = CenterCropVideo(IMG_SIZE)


def transform_val(frames):
    """(T,H,W,C) GPU float → (3,T,112,112) normalised, center crop."""
    return _normalize(_center_crop(frames.permute(3, 0, 1, 2)))


def load_video_metadata(csv_path, video_root):
    df = pd.read_csv(csv_path)
    file_paths  = df.apply(
        lambda r: os.path.join(video_root, str(r["Class"]), r["FileName"]), axis=1
    ).tolist()
    vid_ids     = df["FileName"].apply(lambda f: os.path.splitext(f)[0]).tolist()
    labels      = df["ClassEncoded"].tolist()
    totalframes = df["TotalFrames"].tolist()
    boundaries  = list(np.cumsum([int(np.ceil(tf / SEQ_LEN)) for tf in totalframes]))
    return file_paths, vid_ids, labels, totalframes, boundaries


# ════════════════════════════════════════════════════════════════════════════
# Pose helpers
# ════════════════════════════════════════════════════════════════════════════

_PADDING = 0.25


def _pose_compact_bbox(nodes):
    x_min, x_max = float(nodes[:, :, 0].min()), float(nodes[:, :, 0].max())
    y_min, y_max = float(nodes[:, :, 1].min()), float(nodes[:, :, 1].max())
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    half = max(x_max - x_min, y_max - y_min) / 2 * (1 + _PADDING)
    half = min(half, 0.5)
    x0 = float(np.clip(cx - half, 0.0, 1.0 - 2*half)) if 2*half <= 1 else 0.0
    y0 = float(np.clip(cy - half, 0.0, 1.0 - 2*half)) if 2*half <= 1 else 0.0
    return x0, y0, min(2*half, 1.0)


def extract_kp_slowonly(item, frame_inds):
    """Val preprocessing matching posec3d/utils.PoseHeatmapDataset (augment_train=False)."""
    kp = item["keypoint"]
    if kp.ndim == 4: kp = kp[0]
    T_src = max(kp.shape[0], 1)
    inds  = np.clip(frame_inds, 0, T_src - 1)
    kp    = kp[inds, :17, :2].copy().astype(np.float32)
    h, w  = item["img_shape"]
    kp[:, :, 0] /= max(w, 1)
    kp[:, :, 1] /= max(h, 1)
    kp = np.clip(kp, 0.0, 1.0)
    x0, y0, size = _pose_compact_bbox(kp)
    kp[:, :, 0] = (kp[:, :, 0] - x0) / (size + 1e-8)
    kp[:, :, 1] = (kp[:, :, 1] - y0) / (size + 1e-8)
    return np.clip(kp, 0.0, 1.0)


class MockData:
    """Minimal stand-in for a PyG Data batch (single sample)."""
    def __init__(self, x, num_graphs=1):
        self.x = x
        self.num_graphs = num_graphs


def process_pose_stgcn(item, num_frame=100):
    """Matching stgcnpp/utils.SkeletonGraphDataset.get(): item['keypoint'][0], linspace, mid-hip centre."""
    kp = item["keypoint"]
    if kp.ndim == 4: kp = kp[0]
    if kp.shape[0] > 0:
        inds = np.linspace(0, kp.shape[0] - 1, num_frame).astype(int)
        kp   = kp[inds]
    else:
        kp = np.zeros((num_frame, 17, 2))
    nodes = torch.from_numpy(kp[:, KEEP_INDICES, :2].astype(np.float32))
    h, w  = item["img_shape"]
    nodes[:, :, 0] /= max(w, 1)
    nodes[:, :, 1] /= max(h, 1)
    mid_hip = (nodes[:, 7, :] + nodes[:, 8, :]) / 2
    nodes  -= mid_hip.unsqueeze(1)
    return nodes.reshape(-1, 2)


# ════════════════════════════════════════════════════════════════════════════
# Model loading
# ════════════════════════════════════════════════════════════════════════════

def load_r21d():
    print("[R2+1D] Loading...")
    model = R2Plus1DRGB(num_classes=NUM_CLASSES, dropout=0.5)
    ck    = torch.load(R21D_CKPT, map_location="cpu")
    model.load_state_dict(ck.get("model_state_dict", ck))
    for p in model.parameters():
        p.requires_grad = False
    model = model.to(device).eval()
    print(f"  {R21D_CKPT}")
    return model


def load_slowonly():
    print("[SlowOnly] Loading...")
    model = SlowOnlyHeatmap(num_classes=NUM_CLASSES, J=NUM_JOINTS,
                             H=HEATMAP_SIZE, W=HEATMAP_SIZE, sigma=2.0, with_limb=True)
    ck    = torch.load(SLOWONLY_CKPT, map_location="cpu")
    model.load_state_dict(ck.get("model_state_dict", ck))
    for p in model.parameters():
        p.requires_grad = False
    model = model.to(device).eval()
    print(f"  {SLOWONLY_CKPT}")
    return model


def load_stgcn():
    print("[STGCN++] Loading...")
    model = STGCNPlusPlus(num_classes=NUM_CLASSES, in_channels=2,
                          num_point=13, num_frame=NUM_POSE_FRAMES)
    ck    = torch.load(STGCN_CKPT, map_location="cpu")
    model.load_state_dict(ck.get("model_state_dict", ck))
    for p in model.parameters():
        p.requires_grad = False
    model = model.to(device).eval()
    print(f"  {STGCN_CKPT}")
    return model


# ════════════════════════════════════════════════════════════════════════════
# Feature hook  (captures input to final Linear — the feature vector)
# ════════════════════════════════════════════════════════════════════════════

def register_feat_hook(linear_layer):
    store = []
    def _hook(m, inp):
        store.append(inp[0].detach().cpu())
    handle = linear_layer.register_forward_pre_hook(_hook)
    return handle, store


# ════════════════════════════════════════════════════════════════════════════
# Extraction: R2+1D  (CSV + DALI, single deterministic clip)
# ════════════════════════════════════════════════════════════════════════════

def extract_r21d(csv_path, video_root, model, split_name):
    print(f"\n[R2+1D] Extracting — {split_name}")
    fps, vid_ids, labels, totalframes, boundaries = load_video_metadata(csv_path, video_root)

    pipe   = video_pipe(batch_size=1, num_threads=4, device_id=0,
                        filenames=fps, labels=labels)
    loader = DALIGenericIterator([pipe], ["videos", "labels", "info"], reader_name="Reader")

    handle, feat_store = register_feat_hook(model.classifier[-1])

    feat_dict  = {}
    logit_dict = {}
    current_chunks = []
    processed = 0

    with torch.no_grad():
        for i, data in enumerate(tqdm(loader, desc=f"  {split_name}", leave=False)):
            chunk = data[0]["videos"][0]
            current_chunks.append(chunk)

            if (i + 1) == boundaries[processed]:
                T_full   = totalframes[processed]
                full_vid = torch.cat(current_chunks, dim=0)[:T_full]
                inds     = sample_val(T_full, NUM_VID_FRAMES)
                video    = transform_val(full_vid[inds]).unsqueeze(0).to(device)

                feat_store.clear()
                logits = model(video)

                vid_id = vid_ids[processed]
                feat_dict[vid_id]  = feat_store[0].squeeze().numpy().astype(np.float16)
                logit_dict[vid_id] = logits.cpu().squeeze().numpy().astype(np.float32)

                current_chunks = []
                processed += 1

    handle.remove()
    print(f"  Extracted {len(feat_dict)} videos")
    return feat_dict, logit_dict, vid_ids, labels


# ════════════════════════════════════════════════════════════════════════════
# Extraction: SlowOnly  (PKL split keys — no CSV)
# ════════════════════════════════════════════════════════════════════════════

def extract_slowonly(ann_dict, split_ids, model, split_name):
    print(f"\n[SlowOnly] Extracting — {split_name}")
    samples = [(fd, ann_dict[fd]) for fd in split_ids if fd in ann_dict]
    vid_ids = [s[0] for s in samples]
    labels  = [int(s[1]["label"]) for s in samples]

    handle, feat_store = register_feat_hook(model.classifier[-1])

    feat_dict  = {}
    logit_dict = {}

    with torch.no_grad():
        for vid_id, item in tqdm(samples, desc=f"  {split_name}", leave=False):
            kp_r = item["keypoint"]
            T    = max((kp_r.shape[1] if kp_r.ndim == 4 else kp_r.shape[0]), 1)
            inds = sample_val(T, NUM_VID_FRAMES)
            kp   = extract_kp_slowonly(item, inds)

            kp_t = torch.from_numpy(kp).unsqueeze(0).to(device)  # (1, T, 17, 2)
            feat_store.clear()
            logits = model(kp_t, H=HEATMAP_SIZE, W=HEATMAP_SIZE)

            feat_dict[vid_id]  = feat_store[0].squeeze().numpy().astype(np.float16)
            logit_dict[vid_id] = logits.cpu().squeeze().numpy().astype(np.float32)

    handle.remove()
    print(f"  Extracted {len(feat_dict)} videos")
    return feat_dict, logit_dict, vid_ids, labels


# ════════════════════════════════════════════════════════════════════════════
# Extraction: STGCN++  (PKL split keys — no CSV)
# ════════════════════════════════════════════════════════════════════════════

def extract_stgcn(ann_dict, split_ids, model, split_name):
    print(f"\n[STGCN++] Extracting — {split_name}")
    samples = [(fd, ann_dict[fd]) for fd in split_ids if fd in ann_dict]

    handle, feat_store = register_feat_hook(model.fc)

    feat_dict  = {}
    logit_dict = {}

    with torch.no_grad():
        for vid_id, item in tqdm(samples, desc=f"  {split_name}", leave=False):
            kp_node = process_pose_stgcn(item, NUM_POSE_FRAMES).to(device)

            feat_store.clear()
            logits = model(MockData(kp_node, num_graphs=1))

            feat_dict[vid_id]  = feat_store[0].squeeze().numpy().astype(np.float16)
            logit_dict[vid_id] = logits.cpu().squeeze().numpy().astype(np.float32)

    handle.remove()
    print(f"  Extracted {len(feat_dict)} videos")
    return feat_dict, logit_dict


# ════════════════════════════════════════════════════════════════════════════
# Merge and save
# R2+1D CSV order is canonical — slowonly/stgcn results looked up by vid_id
# ════════════════════════════════════════════════════════════════════════════

def run_extraction(csv_path, video_root, split_type, split_name,
                   model_r21d, model_slow, model_stgcn, ann_dict, pkl_split_ids):
    out_path = os.path.join(FEAT_DIR, f"{split_name}_logits.npz")
    if os.path.exists(out_path):
        print(f"[Cache] {split_name} already exists — skipping ({out_path})")
        return

    print(f"\n{'='*60}\n  EXTRACTION — {split_name}\n{'='*60}")

    f_r21d,  l_r21d,  vid_ids, labels = extract_r21d(
        csv_path, video_root, model_r21d, split_name)
    f_slow,  l_slow,  _, _ = extract_slowonly(
        ann_dict, pkl_split_ids, model_slow, split_name)
    f_stgcn, l_stgcn       = extract_stgcn(
        ann_dict, pkl_split_ids, model_stgcn, split_name)

    N = len(vid_ids)
    feat_r21d   = np.zeros((N, 512),         dtype=np.float16)
    feat_slow   = np.zeros((N, 512),         dtype=np.float16)
    feat_stgcn  = np.zeros((N, 256),         dtype=np.float16)
    logit_r21d  = np.zeros((N, NUM_CLASSES), dtype=np.float32)
    logit_slow  = np.zeros((N, NUM_CLASSES), dtype=np.float32)
    logit_stgcn = np.zeros((N, NUM_CLASSES), dtype=np.float32)

    for i, vid_id in enumerate(vid_ids):
        if vid_id in f_r21d:
            feat_r21d[i]   = f_r21d[vid_id]
            logit_r21d[i]  = l_r21d[vid_id]
        if vid_id in f_slow:
            feat_slow[i]   = f_slow[vid_id]
            logit_slow[i]  = l_slow[vid_id]
        if vid_id in f_stgcn:
            feat_stgcn[i]  = f_stgcn[vid_id]
            logit_stgcn[i] = l_stgcn[vid_id]

    np.savez_compressed(
        out_path,
        feat_r21d=feat_r21d, feat_slow=feat_slow, feat_stgcn=feat_stgcn,
        logit_r21d=logit_r21d, logit_slow=logit_slow, logit_stgcn=logit_stgcn,
        labels=np.array(labels, dtype=np.int64),
    )

    r21d_cov  = sum(1 for v in vid_ids if v in f_r21d)
    slow_cov  = sum(1 for v in vid_ids if v in f_slow)
    stgcn_cov = sum(1 for v in vid_ids if v in f_stgcn)
    print(f"\n[Cache] Saved {split_name} → {out_path}  (N={N})")
    print(f"  Coverage: R21D={r21d_cov}/{N}  SlowOnly={slow_cov}/{N}  STGCN={stgcn_cov}/{N}")


# ════════════════════════════════════════════════════════════════════════════
# Expert evaluation
# ════════════════════════════════════════════════════════════════════════════

def _mca(gt, pred):
    return recall_score(gt, pred, average="macro", zero_division=0) * 100

def _acc(gt, pred):
    return (np.array(gt) == np.array(pred)).mean() * 100


def evaluate_experts(split_name="val"):
    path = os.path.join(FEAT_DIR, f"{split_name}_logits.npz")
    if not os.path.exists(path):
        print(f"[Eval] Cache not found for {split_name} — skipping.")
        return

    npz      = np.load(path)
    gt       = npz["labels"]
    l_r21d   = npz["logit_r21d"]
    l_slow   = npz["logit_slow"]
    l_stgcn  = npz["logit_stgcn"]

    pred_r21d  = l_r21d.argmax(axis=1)
    pred_slow  = l_slow.argmax(axis=1)
    pred_stgcn = l_stgcn.argmax(axis=1)
    pred_avg   = (l_r21d + l_slow + l_stgcn).argmax(axis=1)
    votes      = np.stack([pred_r21d, pred_slow, pred_stgcn], axis=1)
    pred_maj   = mode(votes, axis=1)[0].flatten()
    pred_oracle = np.where(pred_r21d == gt, pred_r21d,
                  np.where(pred_slow  == gt, pred_slow, pred_stgcn))

    results = {
        "R2+1D (RGB)"     : (pred_r21d,   _acc(gt, pred_r21d),   _mca(gt, pred_r21d)),
        "SlowOnly (pose)" : (pred_slow,   _acc(gt, pred_slow),   _mca(gt, pred_slow)),
        "STGCN++"         : (pred_stgcn,  _acc(gt, pred_stgcn),  _mca(gt, pred_stgcn)),
        "SimpleAverage"   : (pred_avg,    _acc(gt, pred_avg),    _mca(gt, pred_avg)),
        "MajorityVote"    : (pred_maj,    _acc(gt, pred_maj),    _mca(gt, pred_maj)),
        "Oracle"          : (pred_oracle, _acc(gt, pred_oracle), _mca(gt, pred_oracle)),
    }

    N = len(gt)
    print(f"\n{'='*58}")
    print(f"  Expert Evaluation — {split_name}  (N={N}  |  {NUM_CLASSES} classes)")
    print(f"{'='*58}")
    print(f"  {'Method':<22} {'Acc':>8} {'MCA':>8}")
    print(f"  {'-'*40}")
    for name, (_, acc, mca) in results.items():
        print(f"  {name:<22} {acc:>7.2f}% {mca:>7.2f}%")
    print(f"{'='*58}\n")

    r_ok = pred_r21d  == gt
    p_ok = pred_slow  == gt
    s_ok = pred_stgcn == gt
    print("  Agreement breakdown:")
    print(f"    All 3 correct          : {(r_ok&p_ok&s_ok).sum():5d} ({(r_ok&p_ok&s_ok).mean()*100:.1f}%)")
    print(f"    All 3 wrong            : {(~r_ok&~p_ok&~s_ok).sum():5d} ({(~r_ok&~p_ok&~s_ok).mean()*100:.1f}%)")
    print(f"    Only RGB correct       : {(r_ok&~p_ok&~s_ok).sum():5d} ({(r_ok&~p_ok&~s_ok).mean()*100:.1f}%)")
    print(f"    Only Pose correct      : {(~r_ok&p_ok&~s_ok).sum():5d} ({(~r_ok&p_ok&~s_ok).mean()*100:.1f}%)")
    print(f"    Only STGCN correct     : {(~r_ok&~p_ok&s_ok).sum():5d} ({(~r_ok&~p_ok&s_ok).mean()*100:.1f}%)")
    print(f"    Pose+STGCN, not RGB    : {(~r_ok&p_ok&s_ok).sum():5d} ({(~r_ok&p_ok&s_ok).mean()*100:.1f}%)")
    print(f"    RGB+STGCN, not Pose    : {(r_ok&~p_ok&s_ok).sum():5d} ({(r_ok&~p_ok&s_ok).mean()*100:.1f}%)")
    print(f"    RGB+Pose, not STGCN    : {(r_ok&p_ok&~s_ok).sum():5d} ({(r_ok&p_ok&~s_ok).mean()*100:.1f}%)")
    print(f"    Exploitable gap        : {((r_ok|p_ok|s_ok)&~(r_ok&p_ok&s_ok)).mean()*100:.1f}%")

    classes = np.unique(gt)
    best_counts = {"R2+1D": 0, "SlowOnly": 0, "STGCN++": 0, "tie": 0}
    for c in classes:
        m = gt == c
        ra = r_ok[m].mean(); pa = p_ok[m].mean(); sa = s_ok[m].mean()
        best = max(ra, pa, sa)
        winners = (ra == best) + (pa == best) + (sa == best)
        if winners > 1:           best_counts["tie"] += 1
        elif ra == best:          best_counts["R2+1D"] += 1
        elif pa == best:          best_counts["SlowOnly"] += 1
        else:                     best_counts["STGCN++"] += 1

    print(f"\n  Per-class best expert ({len(classes)} classes):")
    for k, v in best_counts.items():
        print(f"    {k}: {v} classes")


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"Device: {device}")

    print("\n[Data] Loading pose PKL...")
    with open(POSE_PKL, "rb") as f:
        pose_data = pickle.load(f)
    ann_dict = {a["frame_dir"]: a for a in pose_data["annotations"]}
    train_ids = set(pose_data["split"]["train"])
    val_ids   = set(pose_data["split"]["val"])
    print(f"  Annotations: {len(ann_dict)}  |  train_ids: {len(train_ids)}  val_ids: {len(val_ids)}")

    model_r21d  = load_r21d()
    model_slow  = load_slowonly()
    model_stgcn = load_stgcn()

    run_extraction(CSV_TRAIN, VIDEO_TRAIN, "train", "train",
                   model_r21d, model_slow, model_stgcn, ann_dict, train_ids)
    run_extraction(CSV_VAL,   VIDEO_VAL,   "val",   "val",
                   model_r21d, model_slow, model_stgcn, ann_dict, val_ids)

    del model_r21d, model_slow, model_stgcn
    torch.cuda.empty_cache()

    evaluate_experts("val")
    evaluate_experts("train")

    print("\nDone. Run Stage 2 (stage2_sparse_expert_fusion/train.py) next.")
