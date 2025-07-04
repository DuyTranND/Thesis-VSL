import os
import glob
import json
import random
import numpy as np
import h5py
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import re
from collections import defaultdict

# parameters
T_TARGET, V_JOINTS, C_COORD = 24, 61, 3
BODY_KEEP = list(range(17)) + [23, 24]
H5_COMP = ("gzip", 4)
RANDOM_SEED = 42

def make_splits(file_list):
    """
    Chia train/val/test riêng cho từng lớp (từ),
    đảm bảo các file cùng nhóm (cls, base, gid) luôn chung split.
    Trả về 3 list index đã sort: train_idx, val_idx, test_idx.
    """
    random.seed(RANDOM_SEED)

    # 1) Tạo nhóm theo (cls, base, gid)
    pattern = re.compile(r"^(.+?)_(\d+)\.npy$")
    groups = defaultdict(list)
    for idx, path in enumerate(file_list):
        cls = os.path.basename(os.path.dirname(path))
        fn = os.path.basename(path)
        m = pattern.match(fn)
        if not m:
            continue
        base, num = m.group(1), int(m.group(2))
        gid = num % 100
        groups[(cls, base, gid)].append(idx)

    # 2) Hàm đánh độ khó
    def difficulty(gid):
        if 0 <= gid < 25:   return "easy"
        if 25 <= gid < 75:  return "normal"
        if 75 <= gid < 100: return "hard"
        raise ValueError(gid)

    # 3) Cấu hình số nhóm cho mỗi độ khó và split
    counts = {
        "easy":   {"train": 15, "val": 5, "test": 5},
        "normal": {"train": 30, "val": 10, "test": 10},
        "hard":   {"train": 15, "val": 5, "test": 5},
    }

    # 4) Tập hợp từng class
    classes = sorted({cls for cls, _, _ in groups.keys()})

    train_idxs, val_idxs, test_idxs = set(), set(), set()

    # 5) Chia split cho mỗi class
    for cls in classes:
        # Lấy tất cả key nhóm của class này
        cls_keys = [key for key in groups if key[0] == cls]

        # Phân nhóm theo độ khó
        by_diff = {"easy": [], "normal": [], "hard": []}
        for key in cls_keys:
            d = difficulty(key[2])
            by_diff[d].append(key)

        # Với mỗi độ khó, shuffle rồi slice theo counts
        for diff, key_list in by_diff.items():
            random.shuffle(key_list)
            start = 0
            # train
            n_train = min(counts[diff]["train"], len(key_list) - start)
            for key in key_list[start:start + n_train]:
                train_idxs.update(groups[key])
            start += n_train
            # val
            n_val = min(counts[diff]["val"], len(key_list) - start)
            for key in key_list[start:start + n_val]:
                val_idxs.update(groups[key])
            start += n_val
            # test
            n_test = min(counts[diff]["test"], len(key_list) - start)
            for key in key_list[start:start + n_test]:
                test_idxs.update(groups[key])
            # không cần tăng start thêm cho test

    return sorted(train_idxs), sorted(val_idxs), sorted(test_idxs)

def obj_to_61_trim(obj_arr, dtype=np.float32):
    """Convert object-array (T_raw,3) → (T_raw,61,3)."""
    T_raw = len(obj_arr)
    out = np.empty((T_raw, V_JOINTS, 3), dtype=dtype)
    for t, (lh, rh, body) in enumerate(obj_arr):
        out[t, 0:21]  = lh
        out[t, 21:42] = rh
        out[t, 42:61] = np.asarray(body, dtype=dtype)[BODY_KEEP]
    return out

def pad_or_cut(seq, target_T=T_TARGET):
    """Center-crop or repeat-pad seq to exactly target_T frames."""
    T_raw, V, C = seq.shape
    if T_raw >= target_T:
        start = (T_raw - target_T) // 2
        return seq[start:start + target_T]
    pad = np.repeat(seq[-1][None], target_T - T_raw, axis=0)
    return np.concatenate([seq, pad], axis=0)

def process_clip(args):
    """
    Worker function cho multiprocessing.
    args: tuple (idx, path, class_id)
    Trả về: (idx, processed_seq, class_id, abs_path).
    """
    idx, path, class_id = args
    arr = np.load(path, allow_pickle=True)
    if len(arr) == 0:
        raise ValueError(f"{path} contains no frames")
    seq = obj_to_61_trim(arr)
    seq = pad_or_cut(seq)
    return idx, seq, class_id, os.path.abspath(path)

def build_hdf5(root_dir, h5_out, workers=None):
    """
    Build HDF5 dataset:
    - Thu thập .npy
    - Chia train/val/test per-class với make_splits
    """
    if workers is None:
        workers = cpu_count()

    # 1) Thu thập
    all_files = sorted(glob.glob(os.path.join(root_dir, "*", "*.npy")))
    file_list = all_files[:]  # Dùng hết tất cả
    if not file_list:
        raise RuntimeError(f"No .npy files under {root_dir}")

    # 2) Chia split
    train_idx, val_idx, test_idx = make_splits(file_list)
    N = len(file_list)
    print(f"Total clips: {N}")
    print(f"Splits → train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")

    # 3) Class mapping
    classes = sorted({os.path.basename(os.path.dirname(p)) for p in file_list})
    class2id = {c:i for i,c in enumerate(classes)}

    # 4) Tạo tasks
    tasks = [
        (i, file_list[i], class2id[os.path.basename(os.path.dirname(file_list[i]))])
        for i in range(N)
    ]

    # 5) Ghi HDF5
    with h5py.File(h5_out, "w") as f:
        data_ds = f.create_dataset(
            "data", (N, T_TARGET, V_JOINTS, C_COORD),
            dtype="float32", chunks=(1, T_TARGET, V_JOINTS, C_COORD),
            compression=H5_COMP[0], compression_opts=H5_COMP[1], shuffle=True
        )
        label_ds = f.create_dataset("label", (N,), dtype="int32")
        fname_ds = f.create_dataset("filenames", (N,), dtype=h5py.string_dtype("utf-8"))

        with Pool(workers) as pool:
            for idx_in_task, seq, class_id, fpath in tqdm(
                    pool.imap_unordered(process_clip, tasks),
                    total=N, desc="Processing"):
                data_ds[idx_in_task]  = seq
                label_ds[idx_in_task] = class_id
                fname_ds[idx_in_task] = fpath

        # 6) Lưu metadata
        f.attrs.create(
            "class_mapping",
            json.dumps(class2id, ensure_ascii=False),
            dtype=h5py.string_dtype("utf-8")
        )
        grp = f.create_group("splits")
        grp.create_dataset("train_idx", data=np.array(train_idx, dtype="int32"))
        grp.create_dataset("val_idx",   data=np.array(val_idx,   dtype="int32"))
        grp.create_dataset("test_idx",  data=np.array(test_idx,  dtype="int32"))
        f.create_dataset("class_names", data=np.array(classes, dtype=h5py.string_dtype("utf-8")))

    print("HDF5 build complete.")

if __name__ == "__main__":
    build_hdf5(
        '/workspace/data/augment_norm_data_100',
        '/workspace/data/log/hdf5/data_100.h5',
        workers=56
    )
