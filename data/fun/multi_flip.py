#!/usr/bin/env python3
"""
Flip and rename VSL hand-landmark .npy files across classes, and copy base files.

Reads all .npy files under `data/{class_name}/{base}_{idx}.npy`,
finds highest existing idx for each base name, flips each file (with flip_x_only),
writes to `data_flip/{class_name}/{base}_{new_idx}.npy` (new_idx = max_idx + sequence order),
and additionally copies any base file(s) without _<idx> suffix from IN_ROOT to OUT_ROOT.
Logs progress and skips invalid files. Supports multiprocessing.
"""
import os
import re
import sys
import shutil
import logging
from multiprocessing import Pool, cpu_count
from typing import Tuple
import numpy as np

# ---------- CONFIG ----------
IN_ROOT     = "/workspace/data/augment_norm_data"
OUT_ROOT    = "/workspace/data/augment_norm_data"
LOG_FILE    = "/workspace/data/log/process_flip_100.log"
NUM_WORKERS = 32
IMG_WIDTH   = None  # set if coordinates are absolute pixel indices
# ----------------------------
# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Landmark pairs to swap in pose33
LR_PAIRS_POSE = (
    (1, 4), (2, 5), (3, 6),
    (7, 8),
    (9, 10),
    (11,12),(13,14),(15,16),
    (17,20),(18,21),(19,22),
    (23,24),(25,26),(27,28),
    (29,30),(31,32)
)

# compile pattern for files with suffix
pattern = re.compile(r"^(.+?)_(\d+)\.npy$")


def flip_x_only(
    left_hand: np.ndarray,
    right_hand: np.ndarray,
    pose33: np.ndarray,
    img_width: int | None = IMG_WIDTH
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lh = np.asarray(left_hand, dtype=np.float32)
    rh = np.asarray(right_hand, dtype=np.float32)
    p  = np.asarray(pose33, dtype=np.float32)
    if img_width is None:
        lh[:,0], rh[:,0], p[:,0] = 1.0-lh[:,0], 1.0-rh[:,0], 1.0-p[:,0]
    else:
        lh[:,0] = img_width-1 - lh[:,0]
        rh[:,0] = img_width-1 - rh[:,0]
        p[:,0]  = img_width-1 - p[:,0]
    # swap hands
    new_lh, new_rh = rh.copy(), lh.copy()
    # swap pose landmarks
    for l, r in LR_PAIRS_POSE:
        p[[l, r], :] = p[[r, l], :]
    return new_lh, new_rh, p


def process_pair(args: Tuple[str, str]):
    src, dst = args
    try:
        data = np.load(src, allow_pickle=True)
    except Exception as e:
        logging.error(f"Failed to load {src}: {e}")
        return
    out = []
    for lh, rh, pose in data:
        lh2, rh2, p2 = flip_x_only(lh, rh, pose)
        out.append([lh2, rh2, p2])
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        np.save(dst, np.asarray(out, dtype=object))
        logging.info(f"Saved flipped: {dst}")
    except Exception as e:
        logging.error(f"Failed to save {dst}: {e}")

def flip_job(src: str, dst: str):
    """Load src .npy, flip and swap, save to dst."""
    try:
        data = np.load(src, allow_pickle=True)
    except Exception as e:
        logging.error(f"Failed to load {src}: {e}")
        return
    out = []
    for lh, rh, pose in data:
        lh2, rh2, p2 = flip_x_only(lh, rh, pose)
        out.append([lh2, rh2, p2])
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        np.save(dst, np.asarray(out, dtype=object))
        logging.info(f"Flipped and saved: {dst}")
    except Exception as e:
        logging.error(f"Failed to save {dst}: {e}")

def main():
    # # Step 1: Copy ALL .npy files from IN_ROOT to OUT_ROOT
    # copied = 0
    # for root, _, files in os.walk(IN_ROOT):
    #     for fname in files:
    #         if not fname.lower().endswith('.npy'):
    #             continue
    #         src = os.path.join(root, fname)
    #         rel = os.path.relpath(src, IN_ROOT)
    #         dst = os.path.join(OUT_ROOT, rel)
    #         os.makedirs(os.path.dirname(dst), exist_ok=True)
    #         try:
    #             shutil.copy2(src, dst)
    #             copied += 1
    #         except Exception as e:
    #             logging.error(f"Copy failed {src} -> {dst}: {e}")
    # logging.info(f"Copied {copied} files from IN_ROOT to OUT_ROOT.")

    # Step 2: Prepare flip jobs for files in IN_ROOT with suffix
    groups = {}
    for root, _, files in os.walk(IN_ROOT):
        rel_dir = os.path.relpath(root, IN_ROOT)
        for fname in files:
            m = pattern.match(fname)
            if not m:
                continue
            base, idx = m.group(1), int(m.group(2))
            groups.setdefault((rel_dir, base), []).append(idx)

    jobs = []
    for (rel_dir, base), idxs in groups.items():
        idxs_sorted = sorted(idxs)
        max_idx = idxs_sorted[-1]
        for offset, idx in enumerate(idxs_sorted, start=1):
            new_idx = max_idx + offset
            src = os.path.join(IN_ROOT, rel_dir, f"{base}_{idx}.npy")
            dst = os.path.join(OUT_ROOT, rel_dir, f"{base}_{new_idx}.npy")
            jobs.append((src, dst))

    if not jobs:
        logging.info("No files to flip.")
        return
    logging.info(f"Starting {len(jobs)} flip jobs with {NUM_WORKERS} workers.")

    # Step 3: Run flips in parallel
    with Pool(NUM_WORKERS) as pool:
        for src,dst in jobs:
            pool.apply_async(flip_job, args=(src,dst))
        pool.close()
        pool.join()

if __name__ == '__main__':
    main()