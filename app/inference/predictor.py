import os
import cv2
import yaml
import torch
import argparse
import numpy as np
import mediapipe as mp
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import cdist

# Import các thành phần mô hình và tiện ích từ code đúng
# Đảm bảo các đường dẫn này chính xác trong môi trường của bạn
import sys
sys.path.append('/workspace/src') # Giả sử code của bạn nằm trong /workspace/src
from src import model as model_module
from src.dataset.graphs import Graph

class Predictor:
    def __init__(self, config_path, checkpoint_path, vocabulary_path):
        """
        Khởi tạo pipeline dự đoán, kết hợp logic từ MP4Inference.
        """
        print("Initializing predictor...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 1. Tải cấu hình và từ vựng
        self.config = self.load_config(config_path)
        self.vocabulary = self.load_vocabulary(vocabulary_path)
        self.num_class = len(self.vocabulary)
        
        # 2. Khởi tạo Graph để lấy ma trận kề và các kết nối
        graph_cfg = self.config.get('dataset_args').get('mediapipe61')
        graph = Graph(
            dataset=self.config.get('dataset', 'mediapipe61-custom'),
            max_hop=self.config.get('max_hop', 10)
        )
        self.conn = graph.connect_joint # Kết nối xương cho việc tính toán 'bone'
        self.A = torch.Tensor(graph.A)
        self.parts = graph.parts
        
        # 3. Lấy các luồng đầu vào từ config (J, V, B)
        self.inputs = graph_cfg.get("inputs")

        # 4. Khởi tạo MediaPipe (giống như code đúng)
        self.init_mediapipe()

        # 5. Tải mô hình đã được huấn luyện
        self.model = self.load_model(checkpoint_path)
        print(f"Predictor initialized successfully on {self.device}.")

    def load_config(self, config_path):
        """Tải cấu hình từ file YAML."""
        with open(config_path, 'r') as f:
            return yaml.load(f, Loader=yaml.FullLoader)

    def load_vocabulary(self, vocab_path):
        """Tải từ vựng từ file .npy (giống code đúng)."""
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")
        vocab_array = np.load(vocab_path, allow_pickle=True)
        return vocab_array.tolist()
        
    def init_mediapipe(self):
        """Khởi tạo các model MediaPipe cho tay và tư thế."""
        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5
        )
        self.mp_pose = mp.solutions.pose.Pose(
            model_complexity=2, enable_segmentation=True, min_detection_confidence=0.5
        )

    def load_model(self, model_path):
        """Tải mô hình EfficientGCN đã được huấn luyện."""
        model_args = self.config['model_args']
        model_type = self.config['model_type']

        # Xác định shape dữ liệu đầu vào từ config
        dataset_cfg = self.config['dataset_args']['mediapipe61']
        data_shape = [len(dataset_cfg['inputs']), 6, dataset_cfg['num_frame'], 61, 1]
        
        kwargs = {
            'data_shape': data_shape,
            'num_class': self.num_class,
            'A': self.A,
            'parts': self.parts,
        }

        model = model_module.create(model_type, **model_args, **kwargs)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Checkpoint not found at: {model_path}")
            
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
        
        print(f"Loaded model from: {model_path}")
        model.to(self.device)
        model.eval()
        return model

    def extract_landmarks(self, img):
        """Trích xuất keypoints từ một frame ảnh."""
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        left_hand = [[0, 0, 0]] * 21
        right_hand = [[0, 0, 0]] * 21
        pose = [[0, 0, 0]] * 33

        hands_results = self.mp_hands.process(image_rgb)
        if hands_results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
                hand_label = hands_results.multi_handedness[idx].classification[0].label
                landmarks = [[lm.x, lm.y, 0] for lm in hand_landmarks.landmark]
                if hand_label == "Left":
                    left_hand = landmarks
                elif hand_label == "Right":
                    right_hand = landmarks

        pose_results = self.mp_pose.process(image_rgb)
        if pose_results.pose_landmarks:
            pose = [[round(lm.x, 3), round(lm.y, 3), 0] for lm in pose_results.pose_landmarks.landmark]
            for i in range(17, 33):
                if i not in [23, 24]:
                    pose[i] = [0, 0, 0]

        return left_hand, right_hand, pose

    def extract_video_frames(self, video_path, target_frames=256):
        """Trích xuất các frame từ video và lấy landmarks."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_data = []
        step = max(1, total_frames // target_frames) if total_frames > target_frames else 1

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
        """Lựa chọn các frame đại diện bằng K-means clustering."""
        def flatten_landmarks(lh, rh, p):
            return np.concatenate([np.array(lh).flatten(), np.array(rh).flatten(), np.array(p).flatten()])

        valid_frames = [flatten_landmarks(*d) for d in frames_data if not np.all(np.array(d[0]) == 0) or not np.all(np.array(d[1]) == 0)]
        valid_indices = [i for i, d in enumerate(frames_data) if not np.all(np.array(d[0]) == 0) or not np.all(np.array(d[1]) == 0)]

        if len(valid_frames) < n_clusters:
            # Nếu không đủ frame, lặp lại frame cuối cùng
            while len(frames_data) < n_clusters:
                frames_data.append(frames_data[-1] if frames_data else ([[0,0,0]]*21, [[0,0,0]]*21, [[0,0,0]]*33))
            return frames_data[:n_clusters]

        X = np.array(valid_frames)
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=1, batch_size=min(len(X), 256))
        labels = kmeans.fit_predict(X)

        representative_frames = []
        for i in range(n_clusters):
            cluster_indices = np.where(labels == i)[0]
            if len(cluster_indices) == 0: continue
            
            cluster_center = kmeans.cluster_centers_[i]
            distances = cdist(X[cluster_indices], [cluster_center], 'euclidean')
            closest_idx_in_cluster = np.argmin(distances)
            original_idx = valid_indices[cluster_indices[closest_idx_in_cluster]]
            representative_frames.append((frames_data[original_idx], original_idx))

        representative_frames.sort(key=lambda x: x[1])
        sorted_frames = [frame for frame, _ in representative_frames]

        while len(sorted_frames) < n_clusters:
            sorted_frames.append(sorted_frames[-1])
        return sorted_frames

    def convert_to_61_joints(self, frames_data):
        """Chuyển đổi sang định dạng 61 khớp."""
        BODY_KEEP = list(range(17)) + [23, 24]
        T, V, C = len(frames_data), 61, 3
        output = np.zeros((T, V, C), dtype=np.float32)

        for t, (left_hand, right_hand, pose) in enumerate(frames_data):
            output[t, 0:21] = np.array(left_hand, dtype=np.float32)
            output[t, 21:42] = np.array(right_hand, dtype=np.float32)
            output[t, 42:61] = np.array(pose, dtype=np.float32)[BODY_KEEP]
        return output

    def pre_normalization(self, data):
        """Áp dụng pre-normalization."""
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
        """Tạo các luồng dữ liệu J, V, B."""
        C, T, V, M = data.shape
        joint = np.zeros((C * 2, T, V, M), dtype=data.dtype)
        velocity = np.zeros((C * 2, T, V, M), dtype=data.dtype)
        bone = np.zeros((C * 2, T, V, M), dtype=data.dtype)
        
        base_joint_idx = 42 
        joint[:C, :, :, :] = data
        for i in range(V):
            joint[C:, :, i, :] = data[:, :, i, :] - data[:, :, base_joint_idx, :]

        velocity[:C, :T - 1, :, :] = data[:, 1:, :, :] - data[:, :T - 1, :, :]
        velocity[C:, :T - 2, :, :] = data[:, 2:, :, :] - data[:, :T - 2, :, :]
        
        for i, parent_idx in enumerate(self.conn):
            if i < V and parent_idx < V:
                bone[:C, :, i, :] = data[:, :, i, :] - data[:, :, parent_idx, :]
        bone_length = np.sqrt(np.sum(bone[:C] ** 2, axis=0, keepdims=True)) + 1e-5
        for i in range(C):
            bone[C + i] = np.arccos(np.clip(bone[i] / bone_length[0], -1.0, 1.0))
            
        return joint, velocity, bone
        
    def preprocess_video(self, video_path):
        """Quy trình tiền xử lý video hoàn chỉnh."""
        num_frame_config = self.config['dataset_args']['mediapipe61']['num_frame']
        
        # 1. Trích xuất tất cả các frame và landmarks
        frames_data = self.extract_video_frames(video_path)
        
        # 2. Chọn các frame đại diện
        representative_frames = self.select_representative_frames(frames_data, n_clusters=num_frame_config)
        
        # 3. Chuyển đổi sang định dạng 61 khớp
        sequence = self.convert_to_61_joints(representative_frames)
        
        # 4. Chuẩn bị dữ liệu cho mô hình
        data = sequence.transpose(2, 0, 1)  # (C, T, V)
        data = data[:, :, :, np.newaxis]  # (C, T, V, M)
        
        # 5. Pre-normalization
        data_normalized = self.pre_normalization(data[np.newaxis, ...])[0]
        
        # 6. Tạo multi-stream inputs
        joint, velocity, bone = self.multi_input(data_normalized)
        
        # 7. Chọn và xếp chồng các luồng theo config
        data_new = []
        if 'J' in self.inputs: data_new.append(joint)
        if 'V' in self.inputs: data_new.append(velocity)
        if 'B' in self.inputs: data_new.append(bone)
        data_final = np.stack(data_new, axis=0)
        
        # 8. Thêm batch dimension
        return data_final[np.newaxis, ...]

    def predict(self, video_path, top_k=5):
        """
        Thực hiện dự đoán trên một video.
        """
        print(f"\nProcessing video: {video_path}")
        
        # Áp dụng toàn bộ pipeline tiền xử lý
        input_data = self.preprocess_video(video_path)
        input_tensor = torch.from_numpy(input_data).float().to(self.device)

        # Dự đoán
        with torch.no_grad():
            output, _ = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)

        # Lấy top K kết quả
        top_probs, top_indices = torch.topk(probabilities, top_k, dim=1)
        top_probs = top_probs.cpu().numpy()[0]
        top_indices = top_indices.cpu().numpy()[0]

        # Định dạng kết quả
        predictions = []
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            word = self.vocabulary[idx]
            predictions.append({
                'rank': i + 1,
                'word': word,
                'confidence': float(prob)
            })
        return predictions
            
    def __del__(self):
        """Dọn dẹp tài nguyên MediaPipe."""
        if hasattr(self, 'mp_hands'): self.mp_hands.close()
        if hasattr(self, 'mp_pose'): self.mp_pose.close()

predictor = None

def get_predictor():
    """
    Hàm khởi tạo singleton Predictor với các đường dẫn từ code đúng.
    """
    global predictor
    if predictor is None:
        # Cập nhật các đường dẫn này cho chính xác
        # Ví dụ lấy từ code `main` của bạn
        config_path = '/workspace/configs/2002.yaml'
        vocab_path = '/workspace/npy_converted_full/transformed/mediapipe61/class_names.npy'
        checkpoint_path = "/workspace/workdir/2002_EfficientGCN-B0_mediapipe61/2025-07-03 15-50-16/2002_EfficientGCN-B0_mediapipe61.pth.tar"

        # Kiểm tra sự tồn tại của các file
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found for predictor: {config_path}")
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Vocabulary file not found for predictor: {vocab_path}")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found for predictor: {checkpoint_path}")

        predictor = Predictor(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            vocabulary_path=vocab_path
        )
    return predictor

def predict(video_path, top_k=5):
    """
    Hàm tiện ích để gọi dự đoán từ instance singleton.
    """
    p = get_predictor()
    return p.predict(video_path, top_k=top_k)

# --- VÍ DỤ SỬ DỤNG ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Vietnamese Sign Language Video Inference (Corrected)')
    parser.add_argument('--video', '-v', required=True, help='Path to input MP4 video')
    parser.add_argument('--top_k', '-k', type=int, default=5, help='Number of top predictions to show')
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
    else:
        # Sử dụng hàm predict đã được định nghĩa
        predictions = predict(args.video, top_k=args.top_k)
        
        if predictions:
            print("\n" + "=" * 50)
            print("PREDICTION RESULTS")
            print("=" * 50)
            for pred in predictions:
                print(f"{pred['rank']}. {pred['word']} (confidence: {pred['confidence']:.4f})")
            print("\nTop prediction:", predictions[0]['word'])