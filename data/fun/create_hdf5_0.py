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

import glob
import os
import random

RANDOM_SEED = 42 # Giả sử biến này đã được định nghĩa ở đâu đó

def make_splits(data_root, train_val_ratio=0.8, test_suffix='_0.npy'):
    """
    Chia dữ liệu thành các tập Train/Val/Test.
    - Test set: Bao gồm tất cả các file có tên kết thúc bằng `test_suffix`.
    - Train/Val set: Phần còn lại được chia ngẫu nhiên theo `train_val_ratio`.
    """
    print("="*50)
    # Dòng print này đã được sửa để hiển thị đúng logic đang chạy
    print(f"Bắt đầu chia dữ liệu (Test set là các file kết thúc bằng '{test_suffix}')...")
    print(f"Đang quét các file trong: {data_root}")

    file_list = sorted(glob.glob(os.path.join(data_root, '**', '*.npy'), recursive=True))
    if not file_list:
        print(f"Cảnh báo: Không tìm thấy file nào trong '{data_root}'.")
        return [], [], [], []

    random.seed(RANDOM_SEED)

    # Logic chính xác để tìm file test dựa trên hậu tố
    test_files = [p for p in file_list if os.path.basename(p).endswith(test_suffix)]
    other_files = [p for p in file_list if not os.path.basename(p).endswith(test_suffix)]

    random.shuffle(other_files)
    
    n_others = len(other_files)
    split_point = int(n_others * train_val_ratio)

    train_files = other_files[:split_point]
    val_files = other_files[split_point:]

    path_to_idx = {path: i for i, path in enumerate(file_list)}

    train_idx = sorted([path_to_idx[p] for p in train_files])
    val_idx = sorted([path_to_idx[p] for p in val_files])
    test_idx = sorted([path_to_idx[p] for p in test_files])

    print("Hoàn thành chia dữ liệu:")
    print(f"- Tổng số file: {len(file_list)}")
    print(f"- Train ({len(train_idx)} mẫu)")
    print(f"- Val   ({len(val_idx)} mẫu)")
    print(f"- Test  ({len(test_idx)} mẫu)")
    print("="*50)

    return train_idx, val_idx, test_idx, file_list

def obj_to_61_trim(obj_arr, dtype=np.float32):
    """Convert object-array (T_raw,3) → (T_raw,61,3)."""
    T_raw = len(obj_arr)
    out = np.empty((T_raw, V_JOINTS, 3), dtype=dtype)
    for t, (lh, rh, body) in enumerate(obj_arr):
        out[t, 0:21]  = lh
        out[t, 21:42] = rh
        out[t, 42:61] = np.asarray(body, dtype=dtype)[BODY_KEEP]
    return out

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

    return idx, seq, class_id, os.path.abspath(path)

def build_hdf5(root_dir, h5_out, workers=None):
    """
    Build HDF5 dataset:
    - Giao cho `make_splits` quét file và chia train/val/test.
    - Xử lý và ghi dữ liệu vào file HDF5.
    """
    if workers is None:
        workers = cpu_count()

    # 1 & 2) Quét file và chia split (Giao hoàn toàn cho make_splits)
    # make_splits giờ sẽ trả về 4 giá trị, bao gồm cả file_list nó đã quét được
    train_idx, val_idx, test_idx, file_list = make_splits(root_dir)

    # Kiểm tra xem make_splits có tìm thấy file nào không
    if not file_list:
        print("Hàm build_hdf5 dừng lại vì không có file nào được tìm thấy.")
        return # Thoát khỏi hàm

    N = len(file_list)
    print(f"\nTổng số clip sau khi lọc: {N}")
    print(f"Splits → train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")

    # 3) Class mapping (Không đổi)
    classes = sorted({os.path.basename(os.path.dirname(p)) for p in file_list})
    class2id = {c:i for i,c in enumerate(classes)}

    # 4) Tạo tasks (Không đổi)
    tasks = [
        (i, file_list[i], class2id[os.path.basename(os.path.dirname(file_list[i]))])
        for i in range(N)
    ]

    # 5) Ghi HDF5 (Không đổi)
    with h5py.File(h5_out, "w") as f:
        # Code tạo dataset và xử lý đa luồng giữ nguyên...
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

        # 6) Lưu metadata (Không đổi)
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
        "/workspace/data/augment_norm_data",
        '/workspace/data/log/hdf5/cre_visl.h5',
        workers=32
    )
