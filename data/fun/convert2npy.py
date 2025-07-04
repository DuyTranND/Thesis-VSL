import os
import h5py
import math
import numpy as np
import pickle
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def rotation_matrix(axis, theta):
    if np.abs(axis).sum() < 1e-6 or np.abs(theta) < 1e-6:
        return np.eye(3)
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def angle_between(v1, v2):
    if np.abs(v1).sum() < 1e-6 or np.abs(v2).sum() < 1e-6:
        return 0
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def pre_normalization(data): # Bỏ progress_bar
    N, C, T, V, M = data.shape
    s = np.transpose(data, [0, 4, 2, 3, 1])  # N, C, T, V, M  to  N, M, T, V, C
    l_shoulder, r_shoulder = 53, 54
    l_hip, r_hip = 59, 60
    center_joint = 42  # Nose
    xaxis_joints = [r_shoulder, l_shoulder]

    # Bỏ logging: logging.info('Pad the null frames with the previous frames')
    items = tqdm(s, dynamic_ncols=True)
    for i_s, skeleton in enumerate(items):
        if skeleton.sum() == 0:
            # Bỏ logging: logging.info('Sample {:d} has no skeleton'.format(i_s))
            pass # Giữ pass hoặc xóa dòng này nếu không cần gì ở đây
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            if person[0].sum() == 0:
                index = (person.sum(-1).sum(-1) != 0)
                tmp = person[index].copy()
                person *= 0
                person[:len(tmp)] = tmp
            for i_f, frame in enumerate(person):
                if frame.sum() == 0:
                    if person[i_f:].sum() == 0:
                        rest = len(person) - i_f
                        num = int(np.ceil(rest / i_f))
                        pad = np.concatenate([person[0:i_f] for _ in range(num)], 0)[:rest]
                        s[i_s, i_p, i_f:] = pad
                        break

    # Bỏ logging: logging.info('Sub the center joint #42')
    items = tqdm(s, dynamic_ncols=True)
    for i_s, skeleton in enumerate(items):
        if skeleton.sum() == 0:
            continue
        main_body_center = skeleton[0][:, center_joint:center_joint+1, :].copy()
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            mask = (person.sum(-1) != 0).reshape(T, V, 1)
            s[i_s, i_p] = (s[i_s, i_p] - main_body_center) * mask

    data = np.transpose(s, [0, 4, 2, 3, 1])
    return data

def process_split(args):
    h5_path, out_dir, split_key = args
    C, T_max, V_max, M_max = 3, 20, 61, 1
    split_name = split_key.replace('_idx', '')

    data_list, label_list, class_list = [], [], []

    # Open file and process inside context manager
    with h5py.File(h5_path, 'r') as f:
        data_raw = f['data'] # Shape: (N_total, T, V, C)
        labels_ds = f['label']
        raw_names = f['class_names'][()]
        indices = f['splits'][split_key][()]

        class_names_all = [n.decode() if isinstance(n, bytes) else str(n) for n in raw_names]

        # Read all data for the current split at once
        current_split_data_raw = data_raw[indices]

        data_for_preprocessing = np.transpose(current_split_data_raw, (0, 3, 1, 2))
        # Add a new axis for M (number of persons), which is 1 in this case
        data_for_preprocessing = data_for_preprocessing[:, :, :, :, np.newaxis] # Resulting shape: (N_split, C, T, V, M=1)

        # Apply pre-normalization
        preprocessed_data = pre_normalization(data_for_preprocessing)

        # Iterate through preprocessed samples to collect them
        # (This loop is still necessary as preprocessed_data is an array, not a list of arrays yet)
        for i_sample in tqdm(range(preprocessed_data.shape[0]), desc=f"Collecting {split_name} samples", unit="sample", leave=False):
            data_list.append(preprocessed_data[i_sample])
            
            # Retrieve label and class name using original index from the h5 file
            original_idx = indices[i_sample]
            lbl = int(labels_ds[original_idx])
            label_list.append(lbl)
            class_list.append(class_names_all[lbl] if 0 <= lbl < len(class_names_all) else 'Unknown')

    split_data = np.stack(data_list, axis=0)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"{split_name}_data.npy"), split_data)
    with open(os.path.join(out_dir, f"{split_name}_label.pkl"), 'wb') as lf:
        pickle.dump(label_list, lf)
    with open(os.path.join(out_dir, f"{split_name}_class_name.pkl"), 'wb') as cf:
        pickle.dump(class_list, cf)

    return split_name

if __name__ == '__main__':
    # Save global class names
    h5_file = '/workspace/data/log/hdf5/cre_visl.h5'
    output_dir = '/workspace/data/npy_splits_new'
    
    with h5py.File(h5_file, 'r') as f:
        raw_names = f['class_names'][()]
    class_names_all = [n.decode() if isinstance(n, bytes) else str(n) for n in raw_names]
    np.save(os.path.join(output_dir, 'class_names.npy'), np.array(class_names_all, dtype=object))
    print(f"Saved global class_names.npy (count={len(class_names_all)})")

    # Prepare tasks and run
    split_keys = ['train_idx', 'val_idx', 'test_idx']
    tasks = [(h5_file, output_dir, key) for key in split_keys]
    with Pool(processes=32) as pool:
        for split_name in tqdm(pool.imap_unordered(process_split, tasks),
                                total=len(tasks), desc="Splits", unit="split"):
            print(f"Completed split: {split_name}")

    print("All splits converted successfully.")

