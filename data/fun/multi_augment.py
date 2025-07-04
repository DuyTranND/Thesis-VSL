# run_augmentation.py (version 5 - Separate Input/Output Dirs)
import os
import random
import numpy as np
import logging
import argparse
import shutil
import multiprocessing
import queue
from logging.handlers import QueueHandler, QueueListener
from tqdm import tqdm
from graphs_ import Graph

EPSILON_BODY = 0.005
EPSILON_HAND_FINGER = 0.001
EPSILON_EYE = 0.001

RIGHT_HAND_IDX, LEFT_HAND_IDX, POSE_IDX = 0, 1, 2
NUM_POSE_LANDMARKS = 33
NUM_HAND_LANDMARKS = 21

RANGES = {
    'Nhẹ': {
        'torso':    [(0.98,1.02,)],
        'arm':      [(0.98,1.02,)],
        'finger':   [(0.99,1.01,)],
    },
    'Bình thường': {
        'torso':    [(0.97,0.98),(1.02,1.03)],
        'arm':      [(0.97,0.98),(1.02,1.03)],
        'finger':   [(0.98,0.99),(1.01,1.02)],
    },
    'Khó': {
        'torso':    [(0.95,0.97),(1.03,1.05)],
        'arm':      [(0.95,0.97),(1.03,1.05)],
        'finger':   [(0.97,0.97),(1.02,1.03)],
    },
}

class Pose:
    NOSE = 0
    LEFT_EYE_INNER, LEFT_EYE, LEFT_EYE_OUTER = 1, 2, 3
    RIGHT_EYE_INNER, RIGHT_EYE, RIGHT_EYE_OUTER = 4, 5, 6
    LEFT_EAR, RIGHT_EAR = 7, 8
    MOUTH_LEFT, MOUTH_RIGHT = 9, 10
    LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
    LEFT_ELBOW, RIGHT_ELBOW = 13, 14
    LEFT_WRIST, RIGHT_WRIST = 15, 16
    LEFT_HIP, RIGHT_HIP = 23, 24

def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def create_noise_point_2d(point, epsilon):
    point = np.asarray(point, dtype=np.float32)
    if point.shape != (3,): return [0.0, 0.0, 0.0]
    px, py, pz = point
    ex = random.uniform(-epsilon, epsilon)
    ey = random.uniform(-epsilon, epsilon)
    return [round(px + ex, 3), round(py + ey, 3), 0.0]

def create_point_by_k(ref_A, ref_B, new_A, k):
    ref_A, ref_B, new_A = np.asarray(ref_A, dtype=np.float32), np.asarray(ref_B, dtype=np.float32), np.asarray(new_A, dtype=np.float32)
    vec = ref_B - ref_A
    new_B = new_A + k * vec
    return [round(new_B[0], 3), round(new_B[1], 3), 0.0]

def create_next_point(ref0_A, ref0_B, reft_A, reft_B, new0_A, new0_B, newt_A):
    ref0_A, ref0_B, reft_A, reft_B, new0_A, new0_B, newt_A = \
        np.asarray(ref0_A), np.asarray(ref0_B), np.asarray(reft_A), np.asarray(reft_B), \
        np.asarray(new0_A), np.asarray(new0_B), np.asarray(newt_A)
    dist_ref0 = distance(ref0_A, ref0_B)
    if dist_ref0 < 1e-6: k = 1.0
    else:
        dist_new0 = distance(new0_A, new0_B)
        k = dist_new0 / dist_ref0
    vec_reft = reft_B - reft_A
    newt_B = newt_A + k * vec_reft
    return [round(newt_B[0], 3), round(newt_B[1], 3), 0.0]

def generate_hand(ref0_hand, reft_hand, ref0_wrist, reft_wrist, new0_hand, new0_wrist, newt_wrist, mode: str, k_finger: float):
    new_hand_result = [[0.0, 0.0, 0.0] for _ in range(NUM_HAND_LANDMARKS)]
    new_hand_result[0] = np.asarray(newt_wrist).tolist()
    for i in range(1, NUM_HAND_LANDMARKS):
        is_finger_base = i in [1, 5, 9, 13, 17]
        ref0_parent = ref0_wrist if is_finger_base else ref0_hand[i-1]
        if mode == 'initial':
            newt_parent = newt_wrist if is_finger_base else new_hand_result[i-1]
            new_point = create_point_by_k(ref0_parent, ref0_hand[i], newt_parent, k_finger)
        else:
            new0_parent = new0_wrist if is_finger_base else new0_hand[i-1]
            reft_parent = reft_wrist if is_finger_base else reft_hand[i-1]
            newt_parent = newt_wrist if is_finger_base else new_hand_result[i-1]
            new_point = create_next_point(ref0_parent, ref0_hand[i], reft_parent, reft_hand[i], new0_parent, new0_hand[i], newt_parent)
        new_hand_result[i] = create_noise_point_2d(new_point, EPSILON_HAND_FINGER)
    return new_hand_result

def _finalize_augmented_frame(all_landmarks, graph_map):
    final_right_hand = np.copy(all_landmarks[RIGHT_HAND_IDX*NUM_HAND_LANDMARKS : (RIGHT_HAND_IDX+1)*NUM_HAND_LANDMARKS]).tolist()
    final_left_hand = np.copy(all_landmarks[LEFT_HAND_IDX*NUM_HAND_LANDMARKS : (LEFT_HAND_IDX+1)*NUM_HAND_LANDMARKS]).tolist()
    pose_33_map = {k: v for k, v in graph_map.items() if k >= NUM_HAND_LANDMARKS * 2}
    final_pose = [[0.0, 0.0, 0.0] for _ in range(NUM_POSE_LANDMARKS)]
    for graph_idx, pose_idx in pose_33_map.items():
        if pose_idx < len(final_pose):
            final_pose[pose_idx] = all_landmarks[graph_idx]
    return [final_right_hand, final_left_hand, final_pose]

def pick_scale(rng_list):
    """Cho danh sách 1 hoặc 2 interval, chọn ngẫu nhiên interval rồi uniform."""
    # Với 2 interval, p=0.5 cho mỗi nhánh
    if len(rng_list) == 2 and random.random() < 0.5:
        lo, hi = rng_list[0]
    else:
        lo, hi = rng_list[-1]  # nếu chỉ 1 interval hoặc rơi nhánh else
    return random.uniform(lo, hi)

# --- HÀM generate_augmented_frame0 ĐÃ SỬA LỖI ---
def generate_augmented_frame0(ref0_data, graph_obj, difficulty: str):
    cfg = RANGES[difficulty]
    # Sinh scale
    k_torso  = pick_scale(cfg['torso'])
    k_arm    = k_torso * pick_scale(cfg['arm'])
    k_finger = k_arm   * pick_scale(cfg['finger'])

    logging.info(f"Mức độ: {difficulty} -> Hệ số ngẫu nhiên: k_torso={k_torso:.4f}, k_arm={k_arm:.4f}, k_finger={k_finger:.4f}")
    
    new0_all_landmarks = [[0.0, 0.0, 0.0] for _ in range(graph_obj.num_node)]
    pose_33_to_graph_61_map = {
        Pose.NOSE: 42, Pose.LEFT_EYE_INNER: 43, Pose.LEFT_EYE: 44, Pose.LEFT_EYE_OUTER: 45,
        Pose.RIGHT_EYE_INNER: 46, Pose.RIGHT_EYE: 47, Pose.RIGHT_EYE_OUTER: 48,
        Pose.LEFT_EAR: 49, Pose.RIGHT_EAR: 50, Pose.MOUTH_LEFT: 51, Pose.MOUTH_RIGHT: 52,
        Pose.LEFT_SHOULDER: 53, Pose.RIGHT_SHOULDER: 54, Pose.LEFT_ELBOW: 55, Pose.RIGHT_ELBOW: 56,
        Pose.LEFT_WRIST: 57, Pose.RIGHT_WRIST: 58, Pose.LEFT_HIP: 59, Pose.RIGHT_HIP: 60,
    }
    graph_61_to_pose_33_map = {v: k for k, v in pose_33_to_graph_61_map.items()}
    for pose_idx, graph_idx in pose_33_to_graph_61_map.items():
        if pose_idx in [Pose.NOSE, Pose.LEFT_WRIST, Pose.RIGHT_WRIST]:
             if pose_idx < len(ref0_data['pose']):
                epsilon = EPSILON_EYE if pose_idx == Pose.NOSE else EPSILON_BODY
                new0_all_landmarks[graph_idx] = create_noise_point_2d(ref0_data['pose'][pose_idx], epsilon)
    arm_edges = {(53, 55), (55, 57), (54, 56), (56, 58)}
    for _ in range(2): 
        for parent_graph_idx, child_graph_idx in graph_obj.directed_edges:
            if (0 <= child_graph_idx <= 41): continue
            if new0_all_landmarks[child_graph_idx] == [0.0, 0.0, 0.0] and new0_all_landmarks[parent_graph_idx] != [0.0, 0.0, 0.0]:
                ref0_parent_idx = graph_61_to_pose_33_map.get(parent_graph_idx)
                ref0_child_idx = graph_61_to_pose_33_map.get(child_graph_idx)
                if ref0_parent_idx is None or ref0_child_idx is None or ref0_child_idx >= len(ref0_data['pose']): continue
                current_k = k_arm if (parent_graph_idx, child_graph_idx) in arm_edges else k_torso
                new_point = create_point_by_k(
                    ref0_data['pose'][ref0_parent_idx], ref0_data['pose'][ref0_child_idx], 
                    new0_all_landmarks[parent_graph_idx], current_k
                )
                epsilon = EPSILON_EYE if child_graph_idx in range(43, 53) else EPSILON_BODY
                new0_all_landmarks[child_graph_idx] = create_noise_point_2d(new_point, epsilon)
    
    # ===== PHẦN ĐÃ SỬA LỖI LOGIC =====

    # --- Xử lý tay trái ---
    ref0_left_wrist_from_pose = np.asarray(ref0_data['pose'][Pose.LEFT_WRIST])
    consistent_ref0_left_hand = np.asarray(ref0_data['left_hand']) - np.asarray(ref0_data['left_hand'][0]) + ref0_left_wrist_from_pose
    
    new0_left_wrist = new0_all_landmarks[57]
    
    aug_left_hand = generate_hand(
        consistent_ref0_left_hand, None, 
        consistent_ref0_left_hand[0],  # << SỬA LỖI: Dùng cổ tay từ chính dữ liệu đã được làm nhất quán
        None, 
        None, new0_left_wrist, new0_left_wrist, 
        'initial', k_finger
    )
    for i in range(NUM_HAND_LANDMARKS): 
        new0_all_landmarks[LEFT_HAND_IDX*NUM_HAND_LANDMARKS + i] = aug_left_hand[i]

    # --- Xử lý tay phải ---
    ref0_right_wrist_from_pose = np.asarray(ref0_data['pose'][Pose.RIGHT_WRIST])
    consistent_ref0_right_hand = np.asarray(ref0_data['right_hand']) - np.asarray(ref0_data['right_hand'][0]) + ref0_right_wrist_from_pose
    
    new0_right_wrist = new0_all_landmarks[58]
    
    aug_right_hand = generate_hand(
        consistent_ref0_right_hand, None, 
        consistent_ref0_right_hand[0], # << SỬA LỖI: Dùng cổ tay từ chính dữ liệu đã được làm nhất quán
        None, 
        None, new0_right_wrist, new0_right_wrist, 
        'initial', k_finger
    )
    for i in range(NUM_HAND_LANDMARKS): 
        new0_all_landmarks[RIGHT_HAND_IDX*NUM_HAND_LANDMARKS + i] = aug_right_hand[i]
        
    # =================================================================
    augmented_frame_0 = _finalize_augmented_frame(new0_all_landmarks, graph_61_to_pose_33_map)
    return augmented_frame_0, k_finger



def generate_next_frame(ref0_data, reft_data, new0_frame, graph_obj, k_finger):
    # --- Phần 1: Tái tạo Pose (Giữ nguyên) ---
    newt_all_landmarks = [[0.0, 0.0, 0.0] for _ in range(graph_obj.num_node)]
    new0_right_hand, new0_left_hand, new0_pose = new0_frame
    pose_33_to_graph_61_map = {
        Pose.NOSE: 42, Pose.LEFT_EYE_INNER: 43, Pose.LEFT_EYE: 44, Pose.LEFT_EYE_OUTER: 45,
        Pose.RIGHT_EYE_INNER: 46, Pose.RIGHT_EYE: 47, Pose.RIGHT_EYE_OUTER: 48,
        Pose.LEFT_EAR: 49, Pose.RIGHT_EAR: 50, Pose.MOUTH_LEFT: 51, Pose.MOUTH_RIGHT: 52,
        Pose.LEFT_SHOULDER: 53, Pose.RIGHT_SHOULDER: 54, Pose.LEFT_ELBOW: 55, Pose.RIGHT_ELBOW: 56,
        Pose.LEFT_WRIST: 57, Pose.RIGHT_WRIST: 58, Pose.LEFT_HIP: 59, Pose.RIGHT_HIP: 60,
    }
    graph_61_to_pose_33_map = {v: k for k, v in pose_33_to_graph_61_map.items()}
    for pose_idx, graph_idx in pose_33_to_graph_61_map.items():
        if pose_idx in [Pose.NOSE, Pose.LEFT_WRIST, Pose.RIGHT_WRIST]:
             if pose_idx < len(reft_data['pose']):
                epsilon = EPSILON_EYE if pose_idx == Pose.NOSE else EPSILON_BODY
                newt_all_landmarks[graph_idx] = create_noise_point_2d(reft_data['pose'][pose_idx], epsilon)
    for _ in range(2):
        for parent_graph_idx, child_graph_idx in graph_obj.directed_edges:
            if (0 <= child_graph_idx <= 41): continue
            if newt_all_landmarks[child_graph_idx] == [0.0, 0.0, 0.0] and newt_all_landmarks[parent_graph_idx] != [0.0, 0.0, 0.0]:
                ref0_parent_idx = graph_61_to_pose_33_map.get(parent_graph_idx)
                ref0_child_idx = graph_61_to_pose_33_map.get(child_graph_idx)
                if ref0_parent_idx is None or ref0_child_idx is None or ref0_child_idx >= len(ref0_data['pose']): continue
                new_point = create_next_point(
                    ref0_data['pose'][ref0_parent_idx], ref0_data['pose'][ref0_child_idx],
                    reft_data['pose'][ref0_parent_idx], reft_data['pose'][ref0_child_idx],
                    new0_pose[ref0_parent_idx], new0_pose[ref0_child_idx],
                    newt_all_landmarks[parent_graph_idx]
                )
                epsilon = EPSILON_EYE if child_graph_idx in range(43, 53) else EPSILON_BODY
                newt_all_landmarks[child_graph_idx] = create_noise_point_2d(new_point, epsilon)

    # --- Phần 2: Tái tạo Hands ---
    # --- Xử lý tay trái ---
    if np.allclose(reft_data['left_hand'], ref0_data['left_hand']):
        newt_left_wrist = np.asarray(newt_all_landmarks[57])
        new0_left_wrist = np.asarray(new0_left_hand[0])
        translation_vec_left = newt_left_wrist - new0_left_wrist
        aug_left_hand = (np.asarray(new0_left_hand) + translation_vec_left).tolist()
    else:
        ref0_left_wrist_from_pose = np.asarray(ref0_data['pose'][Pose.LEFT_WRIST])
        consistent_ref0_left_hand = np.asarray(ref0_data['left_hand']) - np.asarray(ref0_data['left_hand'][0]) + ref0_left_wrist_from_pose
        reft_left_wrist_from_pose = np.asarray(reft_data['pose'][Pose.LEFT_WRIST])
        consistent_reft_left_hand = np.asarray(reft_data['left_hand']) - np.asarray(reft_data['left_hand'][0]) + reft_left_wrist_from_pose
        newt_left_wrist = newt_all_landmarks[57]
        new0_left_wrist = new0_left_hand[0]
        aug_left_hand = generate_hand(
            consistent_ref0_left_hand, consistent_reft_left_hand, consistent_ref0_left_hand[0],
            consistent_reft_left_hand[0], new0_left_hand, new0_left_wrist,
            newt_left_wrist, 'sequential', k_finger
        )
    for i in range(NUM_HAND_LANDMARKS):
        newt_all_landmarks[LEFT_HAND_IDX * NUM_HAND_LANDMARKS + i] = aug_left_hand[i]

    # --- Xử lý tay phải ---
    if np.allclose(reft_data['right_hand'], ref0_data['right_hand']):
        newt_right_wrist = np.asarray(newt_all_landmarks[58])
        new0_right_wrist = np.asarray(new0_right_hand[0])
        
        # SỬA LỖI Ở ĐÂY: Dùng new0_right_wrist
        translation_vec_right = newt_right_wrist - new0_right_wrist
        aug_right_hand = (np.asarray(new0_right_hand) + translation_vec_right).tolist()
    else:
        ref0_right_wrist_from_pose = np.asarray(ref0_data['pose'][Pose.RIGHT_WRIST])
        consistent_ref0_right_hand = np.asarray(ref0_data['right_hand']) - np.asarray(ref0_data['right_hand'][0]) + ref0_right_wrist_from_pose
        reft_right_wrist_from_pose = np.asarray(reft_data['pose'][Pose.RIGHT_WRIST])
        consistent_reft_right_hand = np.asarray(reft_data['right_hand']) - np.asarray(reft_data['right_hand'][0]) + reft_right_wrist_from_pose
        newt_right_wrist = newt_all_landmarks[58]
        new0_right_wrist = new0_right_hand[0]
        aug_right_hand = generate_hand(
            consistent_ref0_right_hand, consistent_reft_right_hand, consistent_ref0_right_hand[0],
            consistent_reft_right_hand[0], new0_right_hand, new0_right_wrist,
            newt_right_wrist, 'sequential', k_finger
        )
    for i in range(NUM_HAND_LANDMARKS):
        newt_all_landmarks[RIGHT_HAND_IDX * NUM_HAND_LANDMARKS + i] = aug_right_hand[i]

    return _finalize_augmented_frame(newt_all_landmarks, graph_61_to_pose_33_map)

# =============================================================================
# --- PHẦN 2: CÁC HÀM TĂNG CƯỜNG (Tái cấu trúc cho multiprocessing) ---
# =============================================================================

graph_obj = None

def init_worker(log_queue):
    queue_handler = QueueHandler(log_queue)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(queue_handler)

    global graph_obj
    graph_obj = Graph(dataset='mediapipe61')

# --- HÀM ĐÃ ĐƯỢC CẬP NHẬT: SỬ DỤNG CẢ HAI TEMPLATE BÀN TAY ---
def init_worker(log_queue):
    queue_handler = QueueHandler(log_queue); root_logger = logging.getLogger(); root_logger.setLevel(logging.INFO); root_logger.addHandler(queue_handler)
    global graph_obj; graph_obj = Graph(dataset='mediapipe61')

def augment_single_sequence(input_path, output_path, difficulty, seed, template_left, template_right):
    random.seed(seed); np.random.seed(seed)
    original_data = np.load(input_path, allow_pickle=True)
    if original_data.ndim == 3 and original_data.shape[0] == 1: original_data = np.squeeze(original_data, axis=0)
    num_frames = original_data.shape[0]

    # --- BƯỚC LÀM SẠCH DỮ LIỆU (Phần này của bạn đã đúng) ---
    for t in range(num_frames):
        pose_data = np.asarray(original_data[t, POSE_IDX], dtype=np.float32).reshape(-1, 3)
        
        # Xử lý cho tay trái
        if np.all(np.asarray(original_data[t, LEFT_HAND_IDX]) == 0):
            anchor_pos = pose_data[Pose.LEFT_WRIST]
            if np.any(anchor_pos != 0):
                original_data[t, LEFT_HAND_IDX] = (template_left + anchor_pos).tolist() if t == 0 else original_data[t-1, LEFT_HAND_IDX]

        # Xử lý cho tay phải
        if np.all(np.asarray(original_data[t, RIGHT_HAND_IDX]) == 0):
            anchor_pos = pose_data[Pose.RIGHT_WRIST]
            if np.any(anchor_pos != 0):
                original_data[t, RIGHT_HAND_IDX] = (template_right + anchor_pos).tolist() if t == 0 else original_data[t-1, RIGHT_HAND_IDX]

    # =============================================================================
    # === BẮT ĐẦU PHẦN SỬA LỖI LOGIC TUẦN TỰ ===
    # =============================================================================

    # 1. Khởi tạo trạng thái ban đầu từ frame 0 đã được làm sạch
    prev_ref_data = {
        'pose': np.asarray(original_data[0, POSE_IDX], dtype=np.float32).reshape(-1, 3),
        'left_hand': np.asarray(original_data[0, LEFT_HAND_IDX], dtype=np.float32).reshape(-1, 3),
        'right_hand': np.asarray(original_data[0, RIGHT_HAND_IDX], dtype=np.float32).reshape(-1, 3)
    }
    
    # 2. Tạo frame tăng cường đầu tiên
    all_augmented_frames = []
    prev_aug_frame, k_finger = generate_augmented_frame0(prev_ref_data, graph_obj, difficulty)
    all_augmented_frames.append(prev_aug_frame)

    # 3. Vòng lặp tuần tự cho các frame còn lại
    for t in range(1, num_frames):
        # Lấy dữ liệu gốc của frame HIỆN TẠI
        current_ref_data = {
            'pose': np.asarray(original_data[t, POSE_IDX], dtype=np.float32).reshape(-1, 3),
            'left_hand': np.asarray(original_data[t, LEFT_HAND_IDX], dtype=np.float32).reshape(-1, 3),
            'right_hand': np.asarray(original_data[t, RIGHT_HAND_IDX], dtype=np.float32).reshape(-1, 3)
        }

        # Tạo frame tăng cường HIỆN TẠI dựa trên trạng thái của frame TRƯỚC ĐÓ
        current_aug_frame = generate_next_frame(
            prev_ref_data,       # Dữ liệu gốc của frame t-1
            current_ref_data,    # Dữ liệu gốc của frame t
            prev_aug_frame,      # Dữ liệu TĂNG CƯỜNG của frame t-1
            graph_obj, 
            k_finger
        )
        all_augmented_frames.append(current_aug_frame)
        
        # CẬP NHẬT TRẠNG THÁI: Frame hiện tại sẽ trở thành frame "trước đó" cho vòng lặp tiếp theo
        prev_ref_data = current_ref_data
        prev_aug_frame = current_aug_frame
        
    np.save(output_path, np.array(all_augmented_frames, dtype=object))

def process_file_task(args):
    file_path, target_output_dir, difficulty_distribution, template_left, template_right = args
    try:
        filename_no_ext = os.path.splitext(os.path.basename(file_path))[0]
        shutil.copy2(file_path, os.path.join(target_output_dir, f"{filename_no_ext}_0.npy"))
        for i in range(1, 50):
            seed = i; difficulty = difficulty_distribution[i-1]
            output_path = os.path.join(target_output_dir, f"{filename_no_ext}_{i}.npy")
            augment_single_sequence(file_path, output_path, difficulty, seed, template_left, template_right)
    except Exception as e:
        logging.error(f"Lỗi khi xử lý {file_path}: {e}", exc_info=True)
# =============================================================================
# --- PHẦN 3: HÀM ĐIỀU PHỐI CHÍNH VÀ THỰC THI (Cập nhật lớn) ---
# =============================================================================

def setup_main_logging(log_queue):
    log_formatter = logging.Formatter("%(asctime)s [%(processName)-12s] [%(levelname)s]  %(message)s")
    file_handler = logging.FileHandler("/workspace/data/log/augmentation_run_100.log", mode='w')
    file_handler.setFormatter(log_formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    listener = QueueListener(log_queue, file_handler, console_handler, respect_handler_level=True)
    return listener

def run_augmentations_for_folder(input_dir, output_dir, num_processes, template_path):
    log_queue = multiprocessing.Manager().Queue(); listener = setup_main_logging(log_queue); listener.start()
    main_logger = logging.getLogger(); main_logger.addHandler(QueueHandler(log_queue)); main_logger.setLevel(logging.INFO)
    main_logger.info(f"Bắt đầu với {num_processes} luồng.")

    try:
        template_data_raw = np.load(template_path)
        if template_data_raw.shape != (NUM_HAND_LANDMARKS * 2, 3):
            raise ValueError(f"Shape của template file phải là (42, 3), nhận được {template_data_raw.shape}")
        # TÁCH TEMPLATE ĐÚNG
        template_left = template_data_raw[0:NUM_HAND_LANDMARKS]
        template_right = template_data_raw[NUM_HAND_LANDMARKS:]
        main_logger.info("Tải và tách template thành công.")
    except Exception as e:
        main_logger.error(f"Lỗi nghiêm trọng khi tải template: {e}"); listener.stop(); return

    source_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(input_dir) for f in fn if f.endswith('.npy')]
    main_logger.info(f"Tìm thấy {len(source_files)} file gốc.")
    
    tasks = []
    difficulty_dist = ['Nhẹ'] * 12 + ['Bình thường'] * 25 + ['Khó'] * 12
    for path in source_files:
        relative_path = os.path.relpath(os.path.dirname(path), input_dir)
        target_output_dir = os.path.join(output_dir, relative_path)
        os.makedirs(target_output_dir, exist_ok=True)
        tasks.append((path, target_output_dir, difficulty_dist, template_left, template_right))

    with multiprocessing.Pool(processes=num_processes, initializer=init_worker, initargs=(log_queue,)) as pool:
        with tqdm(total=len(tasks), desc="Đang tăng cường dữ liệu") as pbar:
            for _ in pool.imap_unordered(process_file_task, tasks):
                pbar.update(1)

    main_logger.info("===== HOÀN TẤT TOÀN BỘ QUÁ TRÌNH. ====="); listener.stop()


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(description="Script tăng cường dữ liệu cho các file .npy, hỗ trợ multi-processing.")
    parser.add_argument(
        '--input_dir', default = '/workspace/data/norm_npy/',
        type=str, 
        help='Đường dẫn đến thư mục gốc chứa dữ liệu cần xử lý (ví dụ: data/).'
    )
    parser.add_argument(
        '--output_dir', default = '/workspace/data/augment_norm_data/',
        type=str,
        help='Đường dẫn đến thư mục riêng để lưu kết quả (ví dụ: augment/).'
    )
    parser.add_argument(
        '--processes',
        type=int,
        default=32,
        help='Số lượng process để sử dụng. Mặc định là 16.'
    )
    parser.add_argument(
        '--template_hand_path',
        type=str,
        default='/workspace/data/log/hand/template_both_hands.npy',
        help='Đường dẫn đến file .npy chứa template bàn tay.'
    )
    args = parser.parse_args()

    run_augmentations_for_folder(args.input_dir, args.output_dir, args.processes, args.template_hand_path)