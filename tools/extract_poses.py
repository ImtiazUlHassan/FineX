"""
Extract 2D poses with D-FINE (person detection) + ViTPose++-Large (17 COCO keypoints).
Reads video lists from CSV files (FileName, Class, TotalFrames, ClassEncoded).
Saves results in the same pkl schema as diving48_hrnet.pkl.

Only the single largest detected person per frame is kept (TOP_K = 1).

Supports resume: already-processed frame_dirs are skipped.
"""
import os
import csv
import argparse
import pickle
import traceback
import numpy as np
import torch
from PIL import Image
from decord import VideoReader, cpu
from tqdm import tqdm

# Monkey-patch: bug in transformers VitPoseImageProcessor — missing imports
from numpy.linalg import inv as _inv
from scipy.ndimage import affine_transform as _affine_transform
import transformers.models.vitpose.image_processing_vitpose as _vp_ip
_vp_ip.inv = _inv
_vp_ip.affine_transform = _affine_transform

from transformers import (
    AutoProcessor,
    DFineForObjectDetection,
    VitPoseForPoseEstimation,
    VitPoseImageProcessor,
)

# ── Config ────────────────────────────────────────────────────────────────────
DET_MODEL   = 'ustc-community/dfine-xlarge-coco'
POSE_MODEL  = 'usyd-community/vitpose-plus-large'
DET_THRESH  = 0.4
TOP_K       = 1     # keep only the single largest person box
PERSON_CLS  = 0     # COCO class index for 'person'
DATASET_IDX = 0     # ViTPose-plus expert (0 = COCO)
DET_BATCH   = 16    # frames per D-FINE batch
POSE_BATCH  = 32    # crops per ViTPose batch
NUM_KP      = 17    # COCO keypoints


def build_models(device):
    print('Loading D-FINE xlarge ...')
    det_proc  = AutoProcessor.from_pretrained(DET_MODEL, use_fast=False)
    det_model = DFineForObjectDetection.from_pretrained(DET_MODEL).to(device).eval()

    print('Loading ViTPose-plus-large ...')
    pose_proc  = VitPoseImageProcessor.from_pretrained(POSE_MODEL)
    pose_model = VitPoseForPoseEstimation.from_pretrained(POSE_MODEL).to(device).eval()

    ds_idx = torch.tensor([DATASET_IDX], device=device)
    print('Models ready.\n')
    return det_proc, det_model, pose_proc, pose_model, ds_idx


# ── Detection ─────────────────────────────────────────────────────────────────

def detect_batch(pil_frames, det_proc, det_model, device):
    """D-FINE on a batch of PIL frames. Returns list[list[xyxy]], top-1 by area."""
    inputs = det_proc(images=pil_frames, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = det_model(**inputs)

    sizes = [(img.height, img.width) for img in pil_frames]
    batch_results = det_proc.post_process_object_detection(
        out, target_sizes=sizes, threshold=DET_THRESH
    )

    frame_boxes = []
    for res in batch_results:
        persons = []
        for score, label, box in zip(res['scores'], res['labels'], res['boxes']):
            if int(label) == PERSON_CLS:
                b = box.tolist()
                area = (b[2] - b[0]) * (b[3] - b[1])
                persons.append((area, b))
        persons.sort(reverse=True)
        frame_boxes.append([b for (_, b) in persons[:TOP_K]])  # largest 1
    return frame_boxes


# ── Pose ──────────────────────────────────────────────────────────────────────

def run_vitpose_batch(images, xywh_boxes_list, pose_proc, pose_model, ds_idx, device):
    """
    images          : list of PIL images (one per crop)
    xywh_boxes_list : list of [[x,y,w,h]]  (one single-box list per image)
    Returns list of (kp [17,2], score [17]) CPU tensors.
    """
    inputs = pose_proc(images=images, boxes=xywh_boxes_list, return_tensors='pt')
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
              for k, v in inputs.items()}
    with torch.no_grad():
        out = pose_model(**inputs, dataset_index=ds_idx)
    results = pose_proc.post_process_pose_estimation(out, boxes=xywh_boxes_list)
    return [(r[0]['keypoints'].cpu(), r[0]['scores'].cpu()) for r in results]


def estimate_poses_for_video(pil_frames, all_frame_boxes,
                             pose_proc, pose_model, ds_idx, device):
    T = len(pil_frames)
    kp_out    = np.zeros((TOP_K, T, NUM_KP, 2), dtype=np.float16)  # (1,T,17,2)
    score_out = np.zeros((TOP_K, T, NUM_KP),    dtype=np.float16)  # (1,T,17)

    crop_jobs = []
    for fi, (img, boxes) in enumerate(zip(pil_frames, all_frame_boxes)):
        for pi, xyxy in enumerate(boxes):
            xywh = [xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]]
            crop_jobs.append((fi, pi, img, xywh))

    if not crop_jobs:
        return kp_out, score_out

    pose_steps = range(0, len(crop_jobs), POSE_BATCH)
    for start in tqdm(pose_steps, desc='  ViTPose', leave=False, unit='batch'):
        batch = crop_jobs[start: start + POSE_BATCH]
        imgs  = [j[2] for j in batch]
        boxes = [[j[3]] for j in batch]
        poses = run_vitpose_batch(imgs, boxes, pose_proc, pose_model, ds_idx, device)
        for (fi, pi, _, _), (kp, sc) in zip(batch, poses):
            kp_out[pi, fi]    = kp.numpy().astype(np.float16)
            score_out[pi, fi] = sc.numpy().astype(np.float16)

    return kp_out, score_out


# ── Per-video ─────────────────────────────────────────────────────────────────

def process_video(video_path, label, det_proc, det_model,
                  pose_proc, pose_model, ds_idx, device):
    frame_dir = os.path.splitext(os.path.basename(video_path))[0]

    vr           = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    frames_np    = vr.get_batch(list(range(total_frames))).asnumpy()  # (T,H,W,3) RGB
    H, W         = frames_np.shape[1], frames_np.shape[2]
    pil_frames   = [Image.fromarray(frames_np[i]) for i in range(total_frames)]

    all_frame_boxes = []
    det_steps = range(0, total_frames, DET_BATCH)
    for start in tqdm(det_steps, desc='  D-FINE ', leave=False, unit='batch'):
        all_frame_boxes.extend(
            detect_batch(pil_frames[start:start + DET_BATCH],
                         det_proc, det_model, device)
        )

    keypoints, kp_scores = estimate_poses_for_video(
        pil_frames, all_frame_boxes, pose_proc, pose_model, ds_idx, device
    )

    return {
        'frame_dir'     : frame_dir,
        'label'         : label,
        'img_shape'     : (H, W),
        'original_shape': (H, W),
        'total_frames'  : total_frames,
        'num_person_raw': max((len(b) for b in all_frame_boxes), default=0),
        'keypoint'      : keypoints,    # float16 (1, T, 17, 2)
        'keypoint_score': kp_scores,    # float16 (1, T, 17)
    }


# ── CSV loader ────────────────────────────────────────────────────────────────

def collect_videos_from_csv(csv_path, video_dir, split_name):
    """
    Returns list of (video_path, label, split_name) for every row in the CSV
    whose video file exists on disk.
    CSV columns: FileName, Class, TotalFrames, ClassEncoded
    Videos are stored under video_dir/{ClassEncoded}/FileName
    """
    videos = []
    missing = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname  = row['FileName']
            label  = int(row['ClassEncoded'])
            vpath  = os.path.join(video_dir, str(label), fname)
            if os.path.isfile(vpath):
                videos.append((vpath, label, split_name))
            else:
                missing.append(fname)
    if missing:
        print(f'  WARNING: {len(missing)} files from {os.path.basename(csv_path)} not found on disk.')
    return videos


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def load_checkpoint(output_pkl):
    if not os.path.exists(output_pkl):
        return [], set()
    with open(output_pkl, 'rb') as f:
        data = pickle.load(f)
    annotations = data.get('annotations', [])
    done = {a['frame_dir'] for a in annotations}
    print(f'Resuming: {len(done)} videos already done.')
    return annotations, done


def save_checkpoint(output_pkl, split, annotations):
    os.makedirs(os.path.dirname(os.path.abspath(output_pkl)), exist_ok=True)
    with open(output_pkl, 'wb') as f:
        pickle.dump({'split': split, 'annotations': annotations}, f)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="D-FINE + ViTPose++ pose extraction to a PySKL-style .pkl")
    parser.add_argument("--csv_train",       required=True)
    parser.add_argument("--csv_val",         required=True)
    parser.add_argument("--video_dir_train", required=True, help="videos at <dir>/<ClassEncoded>/<FileName>")
    parser.add_argument("--video_dir_val",   required=True)
    parser.add_argument("--output_pkl",      required=True)
    args = parser.parse_args()
    CSV_TRAIN, CSV_VAL = args.csv_train, args.csv_val
    VIDEO_DIR_TRAIN, VIDEO_DIR_VAL = args.video_dir_train, args.video_dir_val
    OUTPUT_PKL = args.output_pkl

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'GPU   : {torch.cuda.get_device_name(0) if device.type == "cuda" else "N/A"}')
    print(f'CUDA  : {torch.version.cuda}\n')

    det_proc, det_model, pose_proc, pose_model, ds_idx = build_models(device)

    print('Collecting videos from CSVs ...')
    train_videos = collect_videos_from_csv(CSV_TRAIN, VIDEO_DIR_TRAIN, 'train')
    val_videos   = collect_videos_from_csv(CSV_VAL,   VIDEO_DIR_VAL,   'val')
    all_videos   = train_videos + val_videos

    split = {
        'train': [os.path.splitext(os.path.basename(v))[0] for v, _, _ in train_videos],
        'val'  : [os.path.splitext(os.path.basename(v))[0] for v, _, _ in val_videos],
    }
    print(f'  train={len(train_videos)}  val={len(val_videos)}  total={len(all_videos)}\n')

    annotations, done = load_checkpoint(OUTPUT_PKL)

    failed = []
    pbar = tqdm(all_videos, total=len(all_videos), desc='Videos', unit='vid')
    for vpath, label, split_name in pbar:
        fname     = os.path.basename(vpath)
        frame_dir = os.path.splitext(fname)[0]

        if frame_dir in done:
            pbar.set_postfix_str(f'SKIP {frame_dir}')
            continue

        pbar.set_postfix_str(f'{split_name}/{fname}')
        try:
            ann = process_video(
                vpath, label, det_proc, det_model,
                pose_proc, pose_model, ds_idx, device
            )
            annotations.append(ann)
            done.add(frame_dir)
            pbar.set_postfix_str(f'done {frame_dir} frames={ann["total_frames"]}')

            # Save after every video so progress is never lost
            save_checkpoint(OUTPUT_PKL, split, annotations)

        except RuntimeError as e:
            # CUDA errors can leave the device in a bad state — exit immediately
            # so the user can rerun and resume from the checkpoint.
            print(f'  CUDA ERROR on {fname}: {e}')
            print('  Progress saved. Fix the CUDA issue and rerun to resume.')
            save_checkpoint(OUTPUT_PKL, split, annotations)
            raise SystemExit(1)

        except Exception as e:
            print(f'  ERROR on {fname}: {e}')
            traceback.print_exc()
            failed.append(vpath)
            continue

    save_checkpoint(OUTPUT_PKL, split, annotations)
    print(f'\nDone. {len(annotations)} annotations saved → {OUTPUT_PKL}')
    if failed:
        print(f'Skipped {len(failed)} videos due to errors:')
        for f in failed:
            print(f'  {f}')


if __name__ == '__main__':
    main()