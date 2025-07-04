import os
import sys
import logging
from multiprocessing import Pool, cpu_count
import numpy as np

# ---------- CONFIG ----------
IN_ROOT     = "/workspace/data/cre_npy"
OUT_ROOT    = "/workspace/data/norm_npy"
TH_BOTH     = 10/24   # ≥% frames with both hands → treat as two-hand
TH_ONE      = 0.70   # ≥% frames with one hand → dominant hand
MIN_PTS     = 3     # ≥ landmark count ≠0 to consider hand present
LOG_FILE    = "/workspace/data/log/norm_process.log"
# Use all but one CPU core, at least one
NUM_WORKERS = 32
# ----------------------------

# Configure logging to file and console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)


def process_file(rel_path):
    in_path = os.path.join(IN_ROOT, rel_path)
    out_path = os.path.join(OUT_ROOT, rel_path)
    try:
        seq = np.load(in_path, allow_pickle=True)
    except Exception as e:
        logging.error(f"Failed to load {rel_path}: {e}")
        return

    # Validate shape and non-empty
    if seq.ndim != 2 or seq.shape[1] != 3:
        logging.warning(f"Skipping {rel_path}: unexpected shape {seq.shape}")
        return
    T = seq.shape[0]
    if T == 0:
        logging.warning(f"Skipping {rel_path}: empty sequence")
        return

    # Build zero-frame dynamically
    sample = np.asarray(seq[0, 0])  # shape = (num_landmarks, coord_dims)
    num_landmarks, coord_dims = sample.shape
    zeros = np.zeros((num_landmarks, coord_dims), dtype=sample.dtype)

    # Determine presence masks
    def present(frames):
        mask = []
        for joints in frames:
            arr = np.asarray(joints)
            mask.append((arr != 0).any(axis=1).sum() >= MIN_PTS)
        return np.array(mask, dtype=bool)

    L = present(seq[:, 0])
    R = present(seq[:, 1])
    B = L & R
    nL, nR, nB = L.sum(), R.sum(), B.sum()

    # Decide mode and impute
    mode = 'skip'
    if nB >= TH_BOTH * T:
        mode = 'both'
        # no further imputation for both-hand case

    elif nL >= TH_ONE * T:
        mode = 'left'
        idx = R & ~L
        for i in np.where(idx)[0]:
            seq[i, 0] = np.asarray(seq[i, 1]).copy()
            seq[i, 1] = zeros.copy()

    elif nR >= TH_ONE * T:
        mode = 'right'
        idx = L & ~R
        for i in np.where(idx)[0]:
            seq[i, 1] = np.asarray(seq[i, 0]).copy()
            seq[i, 0] = zeros.copy()

    # Save output
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    try:
        np.save(out_path, seq)
    except Exception as e:
        logging.error(f"Failed to save {rel_path}: {e}")
        return

    # Log results
    summary = f"{rel_path} MODE={mode} nL={nL}/{T} nR={nR}/{T} nB={nB}/{T}"
    if mode == 'skip':
        logging.warning("SKIP " + summary)
    elif mode in ('left', 'right'):
        logging.warning("ONE-HAND " + summary)
    else:
        logging.info("OK " + summary)


def main():
    # Collect all .npy files under IN_ROOT
    files = []
    for root, _, filenames in os.walk(IN_ROOT):
        for fname in filenames:
            if fname.lower().endswith('.npy'):
                full = os.path.join(root, fname)
                rel = os.path.relpath(full, IN_ROOT)
                files.append(rel)

    if not files:
        logging.error(f"No .npy files found under {IN_ROOT}")
        sys.exit(1)
        
    logging.info(f"Spawning {NUM_WORKERS} worker processes")
    # Process in parallel
    with Pool(NUM_WORKERS) as pool:
        pool.map(process_file, files)


if __name__ == '__main__':
    main()
