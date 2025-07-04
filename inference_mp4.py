import os
import cv2
import yaml
import torch
import argparse
import numpy as np
import mediapipe as mp
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import cdist

# Import model components
import sys
sys.path.append('/workspace/src')
from src import model as model_module
from src import utils as U
from src.dataset.graphs import Graph  # Import the Graph class

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
        
        Args:
            dataset_name (str): The name of the dataset for the graph.
            max_hop (int): The maximum hop distance for the graph adjacency.
            dilation (int): The dilation for the graph adjacency.
            
        Returns:
            tuple: A tuple containing the adjacency matrix (A) and parts.
        """
        graph = Graph(dataset=dataset_name, max_hop=max_hop, dilation=dilation)
        return graph.A, graph.parts

    def load_model(self, model_path):
        """Load the trained EfficientGCN model"""
        # Model parameters from config
        model_args = self.config['model_args']
        model_type = self.config['model_type']

        # Create model architecture
        data_shape = [3, 6, 24, 61, 1]
        num_class = len(self.vocabulary)

        # Create adjacency matrix and parts using the Graph class
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

        # Load checkpoint
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
        """Extract MediaPipe landmarks from image"""
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        left_hand = [[0, 0, 0]] * 21
        right_hand = [[0, 0, 0]] * 21
        pose = [[0, 0, 0]] * 33

        # Process hands
        hands_results = self.mp_hands.process(image_rgb)
        if hands_results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
                if idx < 2:
                    hand_label = hands_results.multi_handedness[idx].classification[0].label
                    landmarks = [[lm.x, lm.y, 0] for lm in hand_landmarks.landmark]
                    if hand_label == "Left":
                        left_hand = landmarks
                    elif hand_label == "Right":
                        right_hand = landmarks

        # Process pose
        pose_results = self.mp_pose.process(image_rgb)
        if pose_results.pose_landmarks:
            pose = [[round(lm.x, 3), round(lm.y, 3), 0] for lm in pose_results.pose_landmarks.landmark]
            # Zero out certain joints as in original code
            for i in range(17, 33):
                if i not in [23, 24]:
                    pose[i] = [0, 0, 0]

        return left_hand, right_hand, pose

    def extract_video_frames(self, video_path, target_frames=24):
        """Extract frames from video and get landmarks"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < target_frames:
            print(f"Warning: Video has only {total_frames} frames, expected at least {target_frames}")

        frames_data = []
        step = max(1, total_frames // target_frames) if total_frames > target_frames else 1

        frame_indices = range(0, total_frames, step)
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, frame = cap.read()
            if not success:
                break

            # Resize frame for faster processing
            resized_frame = cv2.resize(frame, (640, 480))
            landmarks = self.extract_landmarks(resized_frame)
            frames_data.append(landmarks)

        cap.release()
        return frames_data

    def select_representative_frames(self, frames_data, n_clusters=24):
        """Select representative frames using K-means clustering"""
        def flatten_landmarks(left_hand, right_hand, pose):
            return np.concatenate([
                np.array(left_hand).flatten(),
                np.array(right_hand).flatten(),
                np.array(pose).flatten()
            ])

        def has_valid_hands(left_hand, right_hand):
            left_detected = not np.all(np.array(left_hand) == 0)
            right_detected = not np.all(np.array(right_hand) == 0)
            return left_detected or right_detected

        # Filter frames with valid hands
        valid_frames = []
        valid_indices = []

        for i, (left_hand, right_hand, pose) in enumerate(frames_data):
            if has_valid_hands(left_hand, right_hand):
                flattened = flatten_landmarks(left_hand, right_hand, pose)
                valid_frames.append(flattened)
                valid_indices.append(i)

        if len(valid_frames) < n_clusters:
            print(f"Warning: Only {len(valid_frames)} valid frames found, using all available")
            # If not enough frames, repeat the last frame
            while len(valid_frames) < n_clusters:
                if valid_frames:
                    valid_frames.append(valid_frames[-1])
                    valid_indices.append(valid_indices[-1])
                else:
                    # No valid frames at all, create dummy frame
                    dummy_frame = np.zeros(21*3 + 21*3 + 33*3)
                    valid_frames.append(dummy_frame)
                    valid_indices.append(0)

        X = np.array(valid_frames)

        # Perform K-means clustering
        kmeans = MiniBatchKMeans(
            n_clusters=min(n_clusters, len(X)),
            random_state=42,
            n_init=1,
            batch_size=min(len(X), 256)
        )

        labels = kmeans.fit_predict(X)

        # Select representative frames
        representative_frames = []
        for i in range(min(n_clusters, len(np.unique(labels)))):
            cluster_indices = np.where(labels == i)[0]
            if len(cluster_indices) == 0:
                continue

            cluster_members = X[cluster_indices]
            cluster_center = kmeans.cluster_centers_[i]

            distances = cdist(cluster_members, [cluster_center], 'euclidean')
            closest_idx = np.argmin(distances)
            original_idx = valid_indices[cluster_indices[closest_idx]]

            representative_frames.append((frames_data[original_idx], original_idx))

        # Sort by original frame index
        representative_frames.sort(key=lambda x: x[1])
        sorted_frames = [frame for frame, _ in representative_frames]

        # Ensure we have exactly n_clusters frames
        while len(sorted_frames) < n_clusters:
            sorted_frames.append(sorted_frames[-1] if sorted_frames else
                                 ([[0,0,0]]*21, [[0,0,0]]*21, [[0,0,0]]*33))

        return sorted_frames[:n_clusters]

    def convert_to_61_joints(self, frames_data):
        """Convert MediaPipe format to 61-joint format"""
        BODY_KEEP = list(range(17)) + [23, 24]  # Keep specific pose joints

        T = len(frames_data)
        V = 61  # Total joints
        C = 3   # Coordinates (x, y, z)

        output = np.zeros((T, V, C), dtype=np.float32)

        for t, (left_hand, right_hand, pose) in enumerate(frames_data):
            # Left hand: joints 0-20
            output[t, 0:21] = np.array(left_hand, dtype=np.float32)
            # Right hand: joints 21-41
            output[t, 21:42] = np.array(right_hand, dtype=np.float32)
            # Pose: joints 42-60 (select specific joints)
            pose_array = np.array(pose, dtype=np.float32)
            output[t, 42:61] = pose_array[BODY_KEEP]

        return output

    def pad_or_cut_sequence(self, seq, target_T=24):
        """Pad or cut sequence to target length (from config)"""
        T_raw, V, C = seq.shape
        if T_raw >= target_T:
            # Center crop
            start = (T_raw - target_T) // 2
            return seq[start:start + target_T]
        else:
            # Repeat padding
            pad = np.repeat(seq[-1:], target_T - T_raw, axis=0)
            return np.concatenate([seq, pad], axis=0)

    def pre_normalization(self, data):
        """Apply pre-normalization as in training pipeline"""
        # Input shape: (N, C, T, V, M) where M=1
        N, C, T, V, M = data.shape
        s = np.transpose(data, [0, 4, 2, 3, 1])  # N, M, T, V, C

        center_joint = 42  # Nose joint

        # Subtract center joint (nose) from all joints
        for i_s, skeleton in enumerate(s):
            if skeleton.sum() == 0:
                continue
            main_body_center = skeleton[0][:, center_joint:center_joint+1, :].copy()
            for i_p, person in enumerate(skeleton):
                if person.sum() == 0:
                    continue
                mask = (person.sum(-1) != 0).reshape(T, V, 1)
                s[i_s, i_p] = (s[i_s, i_p] - main_body_center) * mask

        data = np.transpose(s, [0, 4, 2, 3, 1])  # Back to N, C, T, V, M
        return data

    def multi_input(self, data):
        """
        Tạo ra nhiều luồng dữ liệu (joint, velocity, bone) từ dữ liệu khớp ban đầu.
        Hàm này được chuyển thành một phương thức của class.
        """
        # Data đầu vào phải có shape (C, T, V, M)
        C, T, V, M = data.shape
        
        joint = np.zeros((C * 2, T, V, M), dtype=data.dtype)
        velocity = np.zeros((C * 2, T, V, M), dtype=data.dtype)
        bone = np.zeros((C * 2, T, V, M), dtype=data.dtype)

        # 1. Joint-based input (J)
        # Giữ lại tọa độ gốc
        joint[:C, :, :, :] = data
        
        # Tạo tọa độ tương đối so với khớp trung tâm
        # **QUAN TRỌNG**: Đặt `base_joint_idx` thành chỉ mục khớp trung tâm của bạn.
        # Ví dụ: khớp số 42 ("Nose") hoặc khớp hông.
        base_joint_idx = 42 
        for i in range(V):
            joint[C:, :, i, :] = data[:, :, i, :] - data[:, :, base_joint_idx, :]

        # 2. Velocity-based input (V) - Vận tốc
        # Vận tốc 1: delta(t+1) - delta(t)
        velocity[:C, :T - 1, :, :] = data[:, 1:, :, :] - data[:, :T - 1, :, :]
        # Vận tốc 2: delta(t+2) - delta(t)
        velocity[C:, :T - 2, :, :] = data[:, 2:, :, :] - data[:, :T - 2, :, :]
        
        # 3. Bone-based input (B) - Xương
        # **QUAN TRỌNG**: `self.conn` phải được định nghĩa đúng với sơ đồ 61 khớp.
        if self.conn is not None:
            for i, parent_idx in enumerate(self.conn):
                if i < V and parent_idx < V: # Đảm bảo chỉ mục hợp lệ
                    bone[:C, :, i, :] = data[:, :, i, :] - data[:, :, parent_idx, :]
            
            # Tính toán đặc trưng thứ hai cho bone (ví dụ: góc)
            bone_length = np.sqrt(np.sum(bone[:C, :, :, :] ** 2, axis=0, keepdims=True)) + 1e-5
            for i in range(C):
                bone[C + i, :, :, :] = np.arccos(np.clip(bone[i, :, :, :] / bone_length[0, :, :, :], -1.0, 1.0))

        return joint, velocity, bone


    def preprocess_video(self, video_path):
        """Complete preprocessing pipeline"""
        print("Extracting frames and landmarks...")
        frames_data = self.extract_video_frames(video_path, target_frames=256)

        print(f"Selecting representative frames from {len(frames_data)} total frames...")
        representative_frames = self.select_representative_frames(frames_data, n_clusters=24)

        print("Converting to 61-joint format...")
        sequence = self.convert_to_61_joints(representative_frames)

        # print("Padding/cutting to target length...")
        # sequence = self.pad_or_cut_sequence(sequence, target_T=24)

        # 1. Reshape dữ liệu về dạng (C, T, V, M)
        # Đầu vào `sequence` có shape (T, V, C) -> (24, 61, 3)
        data = sequence.transpose(2, 0, 1)  # Shape -> (C, T, V) -> (3, 24, 61)
        data = data[:, :, :, np.newaxis]   # Shape -> (C, T, V, M) -> (3, 24, 61, 1)

        # 2. Áp dụng pre-normalization (nếu có)
        # Lưu ý: self.pre_normalization cần xử lý input shape (1, C, T, V, M)
        # và trả về shape tương tự.
        print("Applying pre-normalization...")
        data_normalized = self.pre_normalization(data[np.newaxis, ...])[0]

        # 3. Áp dụng multi_input để tạo các luồng dữ liệu
        print("Generating multi-stream inputs (Joint, Velocity, Bone)...")
        joint, velocity, bone = self.multi_input(data_normalized)
        
        # 4. Chọn và xếp chồng các luồng theo self.inputs
        data_new = []
        if 'J' in self.inputs:
            data_new.append(joint)
        if 'V' in self.inputs:
            data_new.append(velocity)
        if 'B' in self.inputs:
            data_new.append(bone)
        
        # Xếp chồng các luồng lại với nhau trên một trục mới (axis=0)
        # Shape sẽ là (num_inputs, C*2, T, V, M), ví dụ (3, 6, 24, 61, 1)
        data_final  = np.stack(data_new, axis=0)
        
        # Thêm một chiều mới để phù hợp với đầu vào của mô hình
        data_final = data_final[np.newaxis, ...]  # Shape -> (1, num_inputs, C*2, T, V, M)

        return data_final

    def predict(self, video_path, top_k=5):
        """Run inference on video"""
        print(f"Processing video: {video_path}")

        # Preprocess video
        input_data = self.preprocess_video(video_path)

        # Convert to tensor
        input_tensor = torch.from_numpy(input_data).float().to(self.device)

        print("Running inference...")
        with torch.no_grad():
            output, _ = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)

        # Get top predictions
        top_probs, top_indices = torch.topk(probabilities, top_k, dim=1)
        top_probs = top_probs.cpu().numpy()[0]
        top_indices = top_indices.cpu().numpy()[0]

        # Map to vocabulary
        predictions = []
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            word = self.vocabulary[idx] if idx < len(self.vocabulary) else f"Unknown_{idx}"
            predictions.append({
                'rank': i + 1,
                'word': word,
                'confidence': float(prob),
                'index': int(idx)
            })

        return predictions

    def __del__(self):
        """Clean up MediaPipe resources"""
        if hasattr(self, 'mp_hands'):
            self.mp_hands.close()
        if hasattr(self, 'mp_pose'):
            self.mp_pose.close()


def main():
    parser = argparse.ArgumentParser(description='Vietnamese Sign Language Video Inference')
    parser.add_argument('--video', '-v', required=True, help='Path to input MP4 video')
    parser.add_argument('--config', '-c', default='2002', help='Config name (without .yaml)')
    parser.add_argument('--top_k', '-k', type=int, default=5, help='Number of top predictions to show')

    args = parser.parse_args()

    # Paths
    config_path = f'/workspace/configs/{args.config}.yaml'
    vocab_path = '/workspace/npy_stronger/transformed/mediapipe61/class_names.npy'

    # Try to find the trained model
    model_path = f'/workspace/workdir/{args.config}_EfficientGCN-B0_mediapipe61'
    if os.path.exists(model_path):
        # Look for the best model file
        subdirs = [d for d in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, d))]
        if subdirs:
            # Use the first subdirectory (or implement selection logic)
            model_file = os.path.join(model_path, subdirs[0], f'{args.config}_EfficientGCN-B0_mediapipe61.pth.tar')
            if not os.path.exists(model_file):
                # Try alternative naming
                model_file = os.path.join(model_path, 'checkpoint.pth.tar')
        else:
            model_file = os.path.join(model_path, 'checkpoint.pth.tar')
    else:
        # Fallback to pretrained models directory
        model_file = f'/workspace/pretrained_models/{args.config}_EfficientGCN-B0_mediapipe61.pth.tar'

    model_file = "/workspace/workdir/2002_EfficientGCN-B0_mediapipe61/2025-07-03 14-38-34/2002_EfficientGCN-B0_mediapipe61.pth.tar"

    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        return

    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        return

    if not os.path.exists(vocab_path):
        print(f"Error: Vocabulary file not found: {vocab_path}")
        return

    print("=" * 50)
    print("VSL Inference Pipeline")
    print("=" * 50)
    print(f"Video: {args.video}")
    print(f"Config: {config_path}")
    print(f"Model: {model_file}")
    print(f"Vocabulary: {vocab_path}")
    print()

    try:
        # Initialize inference pipeline
        inference = MP4Inference(config_path, model_file, vocab_path)

        # Run prediction
        predictions = inference.predict(args.video, top_k=args.top_k)

        # Display results
        print("\n" + "=" * 50)
        print("PREDICTION RESULTS")
        print("=" * 50)

        for pred in predictions:
            print(f"{pred['rank']}. {pred['word']} "
                  f"(confidence: {pred['confidence']:.4f}, "
                  f"index: {pred['index']})")

        print("\nTop prediction:", predictions[0]['word'])

    except Exception as e:
        print(f"Error during inference: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()