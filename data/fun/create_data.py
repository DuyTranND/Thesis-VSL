import os
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import cdist
from tqdm import tqdm
import time
import logging

# --- OPTIMIZATION PARAMETERS ---
# 1. Reduce the number of frames to process per video. This is the BIGGEST speed-up.
TARGET_FRAMES_PER_VIDEO = 24
# 2. Use a lower-complexity pose model. 0=fast, 1=medium, 2=slow/accurate.
POSE_MODEL_COMPLEXITY = 2

# --- FIXED PARAMETERS ---
VIDEOS_FOLDER = "/workspace/data/filtered_videos"
OUTPUT_FOLDER = "/workspace/data/cre_npy"
METADATA_FILE = "/workspace/data/vsl_metadata.csv"
FINAL_STATUS_LOG = os.path.join("/workspace/data/log", "processing_log_single_thread_v2.csv")
DETAILED_PROCESSING_LOG = os.path.join("/workspace/data/log", "processing_details_v2.log")
N_CLUSTERS = 24
MIN_TOTAL_FRAMES = 24


def extract_landmarks(img, hands_model, pose_model):
    """
    Extracts hand and pose landmarks from a single image using the provided MediaPipe models.

    Args:
        img (np.ndarray): The input image in BGR format.
        hands_model: The initialized MediaPipe Hands model.
        pose_model: The initialized MediaPipe Pose model.

    Returns:
        Tuple[list, list, list]: A tuple containing landmarks for the left hand, right hand, and pose.
    """
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    left_hand = [[0, 0, 0]] * 21
    right_hand = [[0, 0, 0]] * 21
    pose = [[0, 0, 0]] * 33

    # Process for hand landmarks
    hands_results = hands_model.process(image_rgb)
    if hands_results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
            # The API can detect more than 2 hands, so we limit it.
            if idx < 2:
                hand_label = hands_results.multi_handedness[idx].classification[0].label
                landmarks = [[lm.x, lm.y, 0] for lm in hand_landmarks.landmark]
                if hand_label == "Left":
                    left_hand = landmarks
                elif hand_label == "Right":
                    right_hand = landmarks

    pose_results = pose_model.process(image_rgb)
    if pose_results.pose_landmarks:
        pose = [
            [round(lm.x, 3), round(lm.y, 3), 0]
            for lm in pose_results.pose_landmarks.landmark
        ]
        for i in range(17, 33):
            if i not in [23, 24]:
                pose[i] = [0, 0, 0]

    return left_hand, right_hand, pose


def sample_and_extract_landmarks(
    video_path, target_frames, hands_model, pose_model, min_total_frames=20
):
    """
    Opens a video, samples frames evenly, and extracts landmarks for each sampled frame.

    Args:
        video_path (str): Path to the video file.
        target_frames (int): The target number of frames to sample from the video.
        hands_model: The initialized MediaPipe Hands model.
        pose_model: The initialized MediaPipe Pose model.
        min_total_frames (int): The minimum number of frames a video must have to be processed.

    Returns:
        Tuple[str, list]: A status string and a list of extracted landmark data.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Could not open video file: {video_path}")
        return "ERROR_CANNOT_OPEN", []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < min_total_frames:
        cap.release()
        logging.warning(
            f"Video {os.path.basename(video_path)} has only {total_frames} frames, skipping."
        )
        return "ERROR_INSUFFICIENT_TOTAL_FRAMES", []

    frames_data = []
    # Calculate which frames to sample
    # step = max(1, total_frames // target_frames)
    step = 1
    frame_indices_to_process = range(0, total_frames, step)

    for frame_idx in frame_indices_to_process:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = cap.read()
        if not success:
            break

        # Resizing the frame before processing is a key optimization
        resized_frame = cv2.resize(frame, (640, 480))
        landmarks = extract_landmarks(resized_frame, hands_model, pose_model)
        frames_data.append(landmarks)

    cap.release()
    return "SUCCESS", frames_data


def flatten_landmarks(left_hand, right_hand, pose):
    """Flattens landmarks for all body parts into a single 1D NumPy array."""
    return np.concatenate(
        [
            np.array(left_hand).flatten(),
            np.array(right_hand).flatten(),
            np.array(pose).flatten(),
        ]
    )


def has_valid_hands(left_hand, right_hand):
    """Checks if at least one hand was detected in the frame."""
    left_detected = not np.all(np.array(left_hand) == 0)
    right_detected = not np.all(np.array(right_hand) == 0)
    return left_detected or right_detected


def select_representative_frames_strict(frames_data, n_clusters=24):
    """
    Selects a fixed number of representative frames using KMeans clustering.
    This version is strict: it only processes frames where both hands are visible.
    """
    valid_frames = []
    valid_indices = []

    for i, (left_hand, right_hand, pose) in enumerate(frames_data):
        if has_valid_hands(left_hand, right_hand):
            flattened = flatten_landmarks(left_hand, right_hand, pose)
            valid_frames.append(flattened)
            valid_indices.append(i)

    if len(valid_frames) < n_clusters:
        return []

    X = np.array(valid_frames)
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=1,
        batch_size=min(len(X), 256),  # Use a batch size appropriate for the data
    )

    try:
        labels = kmeans.fit_predict(X)
    except ValueError as e:
        logging.error(f"KMeans failed: {e}. Number of valid frames: {len(X)}")
        return []

    # Ensure we got the number of clusters we asked for
    if len(np.unique(labels)) < n_clusters:
        return []

    representative_frames = []
    for i in range(n_clusters):
        cluster_member_indices = np.where(labels == i)[0]
        if len(cluster_member_indices) == 0:
            continue

        cluster_members = X[cluster_member_indices]
        cluster_center = kmeans.cluster_centers_[i]

        # Find the frame in the cluster that is closest to the cluster's center
        distances_to_center = cdist(cluster_members, [cluster_center], "euclidean")
        closest_member_idx_in_cluster = np.argmin(distances_to_center)
        # Map this back to the index in the original 'valid_frames' list
        closest_member_idx_in_X = cluster_member_indices[closest_member_idx_in_cluster]
        # And map that back to the index in the *original* 'frames_data' list
        original_frame_idx = valid_indices[closest_member_idx_in_X]

        representative_frames.append((frames_data[original_frame_idx], original_frame_idx))

    # Only return a result if we found exactly the right number of frames
    representative_frames.sort(key=lambda x: x[1])

    # 3. Trích xuất chỉ dữ liệu frame từ danh sách đã sắp xếp
    sorted_frames = [frame for frame, index in representative_frames]

    return sorted_frames


def process_and_save(video_path, output_dir, label, hands_model, pose_model):
    """
    Main processing pipeline for a single video.
    """
    video_filename = os.path.basename(video_path)

    status, frames_data = sample_and_extract_landmarks(
        video_path, TARGET_FRAMES_PER_VIDEO, hands_model, pose_model, MIN_TOTAL_FRAMES
    )
    if status != "SUCCESS":
        return video_filename, status, label

    representative_frames = select_representative_frames_strict(frames_data, N_CLUSTERS)
    if not representative_frames:
        return video_filename, "ERROR_INSUFFICIENT_REPRESENTATIVE_FRAMES", label

    npy_filename = os.path.splitext(video_filename)[0] + ".npy"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, npy_filename)
    # Ensure the array has a consistent object type for saving
    np.save(output_path, np.array(representative_frames, dtype=object))

    return video_filename, "PROCESSED_SUCCESSFULLY", label


if __name__ == "__main__":

    # --- CHECKPOINTING PARAMETER ---
    # Save the status log to disk after processing this many videos.
    # This prevents data loss if the script crashes during a long run.
    SAVE_CHECKPOINT_EVERY_N_TASKS = 8

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        filename=DETAILED_PROCESSING_LOG,
        filemode="w",
    )

    print("--- Video Processing Script Started (Single-Threaded with Checkpoints) ---")

    # Initialize MediaPipe models ONCE
    print("Initializing MediaPipe models...")
    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5
    )
    mp_pose = mp.solutions.pose.Pose(
        model_complexity=POSE_MODEL_COMPLEXITY,
        enable_segmentation=True,
        min_detection_confidence=0.5,
    )

    try:
        metadata_df = pd.read_csv(METADATA_FILE, encoding="utf-8")
        metadata_df.set_index("id", inplace=True)

        video_files = [
            f
            for f in os.listdir(VIDEOS_FOLDER)
            if f.lower().endswith((".mp4", ".mov", ".avi"))
        ]

        tasks = []
        for video_filename in video_files:
            video_id = os.path.splitext(video_filename)[0]
            if video_id in metadata_df.index:
                label = metadata_df.loc[video_id, "label"]
                tasks.append(
                    {
                        "video_path": os.path.join(VIDEOS_FOLDER, video_filename),
                        "output_dir": os.path.join(OUTPUT_FOLDER, str(label)),
                        "label": label,
                    }
                )

        print(f"Found {len(tasks)} videos to process.")
        print(f"Detailed logs are being written to: {DETAILED_PROCESSING_LOG}")
        print(f"Status log will be saved to: {FINAL_STATUS_LOG}")
        print(
            f"A checkpoint will be saved every {SAVE_CHECKPOINT_EVERY_N_TASKS} videos.\n"
        )

        # --- CHECKPOINTING SETUP ---
        # 1. Create/overwrite the log file with just the header at the start.
        header_df = pd.DataFrame(columns=["video_filename", "status", "label"])
        header_df.to_csv(FINAL_STATUS_LOG, index=False, encoding="utf-8")

        batch_results = []
        total_processed_count = 0
        start_time = time.time()

        # Main processing loop with a TQDM progress bar
        for i, task in enumerate(tqdm(tasks, desc="Processing Videos", ncols=100)):
            result = process_and_save(
                task["video_path"],
                task["output_dir"],
                task["label"],
                hands_model=mp_hands,
                pose_model=mp_pose,
            )
            batch_results.append(result)
            total_processed_count += 1

            # 2. Check if it's time to save a checkpoint
            if total_processed_count % SAVE_CHECKPOINT_EVERY_N_TASKS == 0:
                print(
                    f"\nCheckpointing... Saving status for {len(batch_results)} videos."
                )
                log_df_batch = pd.DataFrame(
                    batch_results, columns=["video_filename", "status", "label"]
                )
                # Append to the CSV file without writing the header again
                log_df_batch.to_csv(
                    FINAL_STATUS_LOG,
                    mode="a",
                    header=False,
                    index=False,
                    encoding="utf-8",
                )
                # Reset the batch list
                batch_results = []

        # --- FINAL SAVE ---
        # 3. After the loop, save any remaining results that didn't form a full batch
        if batch_results:
            print(f"\nSaving final batch of {len(batch_results)} results.")
            log_df_final = pd.DataFrame(
                batch_results, columns=["video_filename", "status", "label"]
            )
            log_df_final.to_csv(
                FINAL_STATUS_LOG, mode="a", header=False, index=False, encoding="utf-8"
            )

        end_time = time.time()
        total_time = end_time - start_time

        print("-" * 50)
        print(f"\nProcessing finished. Total time: {total_time:.2f} seconds.")
        if tasks:
            print(f"Average time per video: {total_time / len(tasks):.2f} seconds.")

        # 4. Read the complete log file from disk for final statistics
        # This is safer as it reflects what was actually saved.
        if os.path.exists(FINAL_STATUS_LOG):
            print("\n--- Final Processing Statistics (from log file) ---")
            final_log_df = pd.read_csv(FINAL_STATUS_LOG)
            print(final_log_df["status"].value_counts())
        else:
            print("No results were logged.")

    finally:
        # IMPORTANT: Clean up MediaPipe resources
        print("\nClosing MediaPipe models.")
        mp_hands.close()
        mp_pose.close()
        print("--- Script Finished ---")
