import os
import cv2
import yaml
import torch
import argparse
import numpy as np
import mediapipe as mp
import pandas as pd
import json
import time
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.append('/workspace') 
from src import model as model_module
from src import utils as U
from src.dataset.graphs import Graph

class MP4Inference:
    def __init__(self, config_path, model_path, vocab_path):
        """Initialize the inference pipeline"""
        self.config = self.load_config(config_path)
        self.vocabulary = self.load_vocabulary(vocab_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize connections
        graph = Graph('mediapipe61')
        self.conn = graph.connect_joint
        
        # Initialize inputs
        self.inputs = self.config.get('dataset_args').get('mediapipe61').get("inputs")

        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5
        )
        self.mp_pose = mp.solutions.pose.Pose(
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5
        )

        # Load model
        self.model = self.load_model(model_path)

    def load_config(self, config_path):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return config

    def load_vocabulary(self, vocab_path):
        """Tải vocabulary từ file .npy"""
        vocab_array = np.load(vocab_path, allow_pickle=True)
        return vocab_array.tolist()
    
    def _create_graph_and_parts(self, dataset_name='mediapipe61-custom', max_hop=10, dilation=1):
        """
        Creates the graph adjacency matrix and parts using the Graph class.
        """
        graph = Graph(dataset=dataset_name, max_hop=max_hop, dilation=dilation)
        return graph.A, graph.parts

    def load_model(self, model_path):
        """Load the trained EfficientGCN model"""
        model_args = self.config['model_args']
        model_type = self.config['model_type']
        data_shape = [3, 6, 24, 61, 1]
        num_class = len(self.vocabulary)
        A, parts = self._create_graph_and_parts(
            dataset_name=self.config.get('dataset', 'mediapipe61-custom'),
            max_hop=self.config.get('max_hop', 10),
            dilation=self.config.get('dilation', 1)
        )
        kwargs = {
            'data_shape': data_shape,
            'num_class': num_class,
            'A': torch.Tensor(A),
            'parts': parts,
        }
        model = model_module.create(model_type, **model_args, **kwargs)
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            if 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
            else:
                model.load_state_dict(checkpoint)
            print(f"Loaded model from: {model_path}")
        else:
            print(f"Warning: Model file not found: {model_path}")
            print("Using randomly initialized model")
        model.to(self.device)
        model.eval()
        return model

    def extract_landmarks(self, img):
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        left_hand = [[0, 0, 0]] * 21
        right_hand = [[0, 0, 0]] * 21
        pose = [[0, 0, 0]] * 33
        hands_results = self.mp_hands.process(image_rgb)
        if hands_results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
                if idx < 2:
                    hand_label = hands_results.multi_handedness[idx].classification[0].label
                    landmarks = [[lm.x, lm.y, 0] for lm in hand_landmarks.landmark]
                    if hand_label == "Left": left_hand = landmarks
                    elif hand_label == "Right": right_hand = landmarks
        pose_results = self.mp_pose.process(image_rgb)
        if pose_results.pose_landmarks:
            pose = [[round(lm.x, 3), round(lm.y, 3), 0] for lm in pose_results.pose_landmarks.landmark]
            for i in range(17, 33):
                if i not in [23, 24]: pose[i] = [0, 0, 0]
        return left_hand, right_hand, pose

    def extract_video_frames(self, video_path, target_frames=24):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened(): raise ValueError(f"Cannot open video: {video_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_data = []
        if total_frames > 0:
            step = max(1, total_frames // target_frames)
            for frame_idx in range(0, total_frames, step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                success, frame = cap.read()
                if not success: break
                resized_frame = cv2.resize(frame, (640, 480))
                landmarks = self.extract_landmarks(resized_frame)
                frames_data.append(landmarks)
        cap.release()
        return frames_data

    def select_representative_frames(self, frames_data, n_clusters=24):
        from sklearn.cluster import MiniBatchKMeans
        from scipy.spatial.distance import cdist
        def flatten_landmarks(lh, rh, p): return np.concatenate([np.array(lh).flatten(), np.array(rh).flatten(), np.array(p).flatten()])
        def has_valid_hands(lh, rh): return not np.all(np.array(lh) == 0) or not np.all(np.array(rh) == 0)
        valid_frames = [flatten_landmarks(*d) for d in frames_data if has_valid_hands(d[0], d[1])]
        valid_indices = [i for i, d in enumerate(frames_data) if has_valid_hands(d[0], d[1])]
        if not valid_frames: return [([[0,0,0]]*21, [[0,0,0]]*21, [[0,0,0]]*33)] * n_clusters
        if len(valid_frames) < n_clusters:
            original_frames = [frames_data[i] for i in valid_indices]
            while len(original_frames) < n_clusters: original_frames.append(original_frames[-1])
            return original_frames
        X = np.array(valid_frames)
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=1, batch_size=min(len(X), 256))
        labels = kmeans.fit_predict(X)
        representative_frames = []
        for i in range(n_clusters):
            cluster_indices = np.where(labels == i)[0]
            if len(cluster_indices) == 0: continue
            cluster_members = X[cluster_indices]
            cluster_center = kmeans.cluster_centers_[i]
            distances = cdist(cluster_members, [cluster_center], 'euclidean')
            closest_idx_in_cluster = np.argmin(distances)
            original_idx = valid_indices[cluster_indices[closest_idx_in_cluster]]
            representative_frames.append((frames_data[original_idx], original_idx))
        representative_frames.sort(key=lambda x: x[1])
        sorted_frames = [frame for frame, _ in representative_frames]
        while len(sorted_frames) < n_clusters: sorted_frames.append(sorted_frames[-1])
        return sorted_frames[:n_clusters]

    def convert_to_61_joints(self, frames_data):
        BODY_KEEP = list(range(17)) + [23, 24]
        T, V, C = len(frames_data), 61, 3
        output = np.zeros((T, V, C), dtype=np.float32)
        for t, (left_hand, right_hand, pose) in enumerate(frames_data):
            output[t, 0:21] = np.array(left_hand, dtype=np.float32)
            output[t, 21:42] = np.array(right_hand, dtype=np.float32)
            output[t, 42:61] = np.array(pose, dtype=np.float32)[BODY_KEEP]
        return output

    def pre_normalization(self, data):
        N, C, T, V, M = data.shape
        s = np.transpose(data, [0, 4, 2, 3, 1])
        center_joint = 42
        for i_s, skeleton in enumerate(s):
            if skeleton.sum() == 0: continue
            main_body_center = skeleton[0][:, center_joint:center_joint+1, :].copy()
            for i_p, person in enumerate(skeleton):
                if person.sum() == 0: continue
                mask = (person.sum(-1) != 0).reshape(T, V, 1)
                s[i_s, i_p] = (s[i_s, i_p] - main_body_center) * mask
        return np.transpose(s, [0, 4, 2, 3, 1])

    def multi_input(self, data):
        C, T, V, M = data.shape
        joint, velocity, bone = (np.zeros((C*2, T, V, M), dtype=data.dtype) for _ in range(3))
        joint[:C, :, :, :] = data
        base_joint_idx = 42
        for i in range(V): joint[C:, :, i, :] = data[:, :, i, :] - data[:, :, base_joint_idx, :]
        velocity[:C, :T - 1, :, :] = data[:, 1:, :, :] - data[:, :T - 1, :, :]
        velocity[C:, :T - 2, :, :] = data[:, 2:, :, :] - data[:, :T - 2, :, :]
        if self.conn is not None:
            for i, parent_idx in enumerate(self.conn):
                if i < V and parent_idx < V: bone[:C, :, i, :] = data[:, :, i, :] - data[:, :, parent_idx, :]
        return joint, velocity, bone

    def preprocess_video(self, video_path):
        frames_data = self.extract_video_frames(video_path, target_frames=256)
        if not frames_data:
            raise ValueError(f"No frames could be extracted from {video_path}")
        representative_frames = self.select_representative_frames(frames_data, n_clusters=24)
        sequence = self.convert_to_61_joints(representative_frames)
        data = sequence.transpose(2, 0, 1)[np.newaxis, :, :, :, np.newaxis]
        data_normalized = self.pre_normalization(data)[0]
        joint, velocity, bone = self.multi_input(data_normalized)
        data_new = []
        if 'J' in self.inputs: data_new.append(joint)
        if 'V' in self.inputs: data_new.append(velocity)
        if 'B' in self.inputs: data_new.append(bone)
        return np.stack(data_new, axis=0)[np.newaxis, ...]

    def predict(self, video_path, top_k=5):
        input_data = self.preprocess_video(video_path)
        input_tensor = torch.from_numpy(input_data).float().to(self.device)
        with torch.no_grad():
            output, _ = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
        top_probs, top_indices = torch.topk(probabilities, top_k, dim=1)
        top_probs = top_probs.cpu().numpy()[0]
        top_indices = top_indices.cpu().numpy()[0]
        predictions = []
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            word = self.vocabulary[idx] if idx < len(self.vocabulary) else f"Unknown_{idx}"
            predictions.append({'rank': i + 1, 'word': word, 'confidence': float(prob)})
        return predictions

    def __del__(self):
        if hasattr(self, 'mp_hands'): self.mp_hands.close()
        if hasattr(self, 'mp_pose'): self.mp_pose.close()

def load_and_filter_videos(csv_path: Path) -> pd.DataFrame:
    """Đọc file CSV và lọc các video hợp lệ."""
    print(f"Đang đọc và lọc file CSV từ: {csv_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    processed_df = df[df['status'] == 'PROCESSED_SUCCESSFULLY'].copy()
    print(f"Đã tìm thấy {len(df)} dòng, trong đó có {len(processed_df)} video hợp lệ.")
    return processed_df

# HÀM process_video_task KHÔNG CÒN CẦN THIẾT VÀ ĐÃ ĐƯỢC XÓA

def run_evaluation(model_name: str, model_config: dict, videos_df: pd.DataFrame, video_dir: Path, output_dir: Path):
    """
    Chạy đánh giá cho một mô hình cụ thể trên danh sách video bằng logic đơn luồng.
    """
    print("\n" + "="*80)
    print(f"BẮT ĐẦU ĐÁNH GIÁ CHO MODEL: {model_name}")
    print(f" - Config: {model_config['config_path']}")
    print(f" - Model: {model_config['model_path']}")
    print("="*80)

    # --- THAY ĐỔI QUAN TRỌNG: Khởi tạo model CHỈ MỘT LẦN ---
    try:
        print("Đang khởi tạo model...")
        inference = MP4Inference(
            config_path=model_config['config_path'],
            model_path=model_config['model_path'],
            vocab_path=model_config['vocab_path']
        )
        print("Khởi tạo model thành công.")
    except Exception as e:
        print(f"Lỗi nghiêm trọng khi khởi tạo model {model_name}: {e}")
        return # Bỏ qua model này nếu không thể khởi tạo

    tasks = []
    for _, row in videos_df.iterrows():
        video_path = video_dir / row['video_filename']
        if video_path.exists():
            tasks.append((video_path, row['label']))
        else:
            print(f"Cảnh báo: Bỏ qua file không tồn tại - {video_path}")
            
    results = []
    # --- THAY ĐỔI: Sử dụng vòng lặp for đơn giản thay vì ProcessPoolExecutor ---
    for video_path, ground_truth in tqdm(tasks, desc=f"Processing videos for {model_name}"):
        try:
            start_time = time.time()
            # Gọi hàm predict trên đối tượng inference đã được tạo
            predictions = inference.predict(video_path, top_k=5)
            duration = time.time() - start_time
            
            # Thêm kết quả thành công
            results.append({
                "video_file": video_path.name,
                "ground_truth": ground_truth,
                "top_1_prediction": predictions[0]['word'] if predictions else None,
                "top_5_predictions": [p['word'] for p in predictions],
                "predictions_full": predictions,
                "processing_time": duration,
                "status": "success"
            })
        except Exception as e:
            # Ghi nhận lỗi cho video cụ thể
            results.append({
                "video_file": video_path.name,
                "ground_truth": ground_truth,
                "status": "error",
                "error_message": str(e)
            })

    # Giải phóng tài nguyên model trước khi chuyển sang model tiếp theo
    del inference

    # --- Xử lý kết quả (giữ nguyên) ---
    successful_results = [r for r in results if r['status'] == 'success']
    failed_results = [r for r in results if r['status'] == 'error']
    
    if not successful_results:
        print("Không có video nào được xử lý thành công.")
        return

    # Tính toán các chỉ số
    total_time = sum(r['processing_time'] for r in successful_results)
    avg_time = total_time / len(successful_results)
    
    top1_correct = sum(1 for r in successful_results if r['ground_truth'] == r['top_1_prediction'])
    top5_correct = sum(1 for r in successful_results if r['ground_truth'] in r['top_5_predictions'])
    
    total_successful = len(successful_results)
    top1_accuracy = top1_correct / total_successful if total_successful > 0 else 0
    top5_accuracy = top5_correct / total_successful if total_successful > 0 else 0

    # Chuẩn bị file JSON output
    output_data = {
        "model_name": model_name,
        "summary": {
            "total_videos_processed": total_successful,
            "total_videos_failed": len(failed_results),
            "average_processing_time_per_video": f"{avg_time:.4f} seconds",
            "top_1_accuracy": f"{top1_accuracy:.2%}",
            "top_5_accuracy": f"{top5_accuracy:.2%}",
        },
        "detailed_results": successful_results,
        "failed_videos": failed_results
    }
    
    # In kết quả ra màn hình
    print("\n--- KẾT QUẢ TÓM TẮT ---")
    print(f"Tổng số video xử lý thành công: {total_successful}")
    print(f"Tổng số video xử lý thất bại: {len(failed_results)}")
    print(f"Thời gian xử lý trung bình: {avg_time:.4f} giây/video")
    print(f"Top-1 Accuracy: {top1_accuracy:.2%} ({top1_correct}/{total_successful})")
    print(f"Top-5 Accuracy: {top5_accuracy:.2%} ({top5_correct}/{total_successful})")

    # Lưu kết quả vào file JSON
    output_path = output_dir / f"{model_name}_evaluation_results.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
        
    print(f"\nĐã lưu kết quả chi tiết vào: {output_path}")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Đánh giá và so sánh các mô hình nhận dạng ngôn ngữ ký hiệu.')
    parser.add_argument('--csv_file', type=str, required=True, help='Đường dẫn tới file CSV chứa thông tin video.')
    parser.add_argument('--video_dir', type=str, required=True, help='Đường dẫn tới thư mục chứa các file video MP4.')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results', help='Thư mục để lưu file JSON kết quả.')
    args = parser.parse_args()

    csv_path = Path(args.csv_file)
    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)

    # --- KHAI BÁO CÁC MÔ HÌNH CẦN SO SÁNH TẠI ĐÂY ---
    MODELS_TO_EVALUATE = {
        "model_A_2002_train100_test100": {
            "config_path": "/workspace/workdir/2002_100_EfficientGCN-B0_mediapipe61/2025-07-03 17-51-22/config.yaml",
            "model_path": "/workspace/workdir/2002_100_EfficientGCN-B0_mediapipe61/2025-07-03 17-51-22/2002_100_EfficientGCN-B0_mediapipe61.pth.tar",
            "vocab_path": "/workspace/data/npy_100/transformed/mediapipe61/class_names.npy"
        },
        "model_B_2002_trainstrong_test100": {
            "config_path": "/workspace/workdir/2002_EfficientGCN-B0_mediapipe61/2025-07-03 15-50-16/config.yaml",
            "model_path": "/workspace/workdir/2002_EfficientGCN-B0_mediapipe61/2025-07-03 15-50-16/2002_EfficientGCN-B0_mediapipe61.pth.tar",
            "vocab_path": "/workspace/npy_converted_full/transformed/mediapipe61/class_names.npy"
        }
    }
    
    MODELS_NEXT = {
        "model_stgcn_trainsplitnew_testsplitnew": {
            "config_path": "/workspace/workdir/2002_EfficientGCN-B0_mediapipe61/2025-07-03 15-50-16/config.yaml",
            "model_path": "/workspace/workdir/2002_EfficientGCN-B0_mediapipe61/2025-07-03 15-50-16/2002_EfficientGCN-B0_mediapipe61.pth.tar",
            "vocab_path": "/workspace/npy_converted_full/transformed/mediapipe61/class_names.npy"
        },
    }
    
    try:
        # 1. Đọc và lọc dữ liệu từ CSV (chỉ làm một lần)
        videos_df = load_and_filter_videos(csv_path)

        videos_df = videos_df.head(500)  # Giới hạn số lượng video để kiểm tra nhanh

        # 2. Chạy đánh giá cho từng mô hình
        for model_name, model_config in MODELS_TO_EVALUATE.items():
            run_evaluation(
                model_name=model_name,
                model_config=model_config,
                videos_df=videos_df,
                video_dir=video_dir,
                output_dir=output_dir
            )
            
    except FileNotFoundError as e:
        print(f"Lỗi: {e}")
    except Exception as e:
        print(f"Một lỗi không mong muốn đã xảy ra: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()