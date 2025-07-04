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
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- OPTIMIZATION PARAMETERS ---
# 1. Set the number of threads to use for processing. A good starting point is the number of CPU cores.
MAX_WORKERS = 1 # Default to 4 if cpu_count() is None
# 2. Reduce the number of frames to process per video. This is the BIGGEST speed-up.
TARGET_FRAMES_PER_VIDEO = 24
# 3. Use a lower-complexity pose model. 0=fast, 1=medium, 2=slow/accurate.
POSE_MODEL_COMPLEXITY = 2

# --- FIXED PARAMETERS ---
VIDEOS_FOLDER = "/workspace/data/filtered_videos"
OUTPUT_FOLDER = "/workspace/data/cre_npy"
METADATA_FILE = "/workspace/data/vsl_metadata.csv"
# Changed log file name to reflect multithreading
FINAL_STATUS_LOG = os.path.join('/workspace/data', "processing_log_multi_thread.csv")
DETAILED_PROCESSING_LOG = os.path.join('/workspace/data', "processing_details_multi_thread.log")
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
                if hand_label == 'Left':
                    left_hand = landmarks
                elif hand_label == 'Right':
                    right_hand = landmarks

    # Process for pose landmarks
    pose_results = pose_model.process(image_rgb)
    if pose_results.pose_landmarks:
        pose_landmarks_list = [[lm.x, lm.y, 0] for lm in pose_results.pose_landmarks.landmark]
        # We only care about upper body, so zero out lower body landmarks
        # This keeps the array structure consistent while discarding unnecessary data.
        for i in range(23, 33):
            pose_landmarks_list[i] = [0, 0, 0]
        pose = pose_landmarks_list

    return left_hand, right_hand, pose

def sample_and_extract_landmarks(video_path, target_frames, hands_model, pose_model, min_total_frames=20):
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
        return 'ERROR_CANNOT_OPEN', []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < min_total_frames:
        cap.release()
        logging.warning(f"Video {os.path.basename(video_path)} has only {total_frames} frames, skipping.")
        return 'ERROR_INSUFFICIENT_TOTAL_FRAMES', []

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
    return 'SUCCESS', frames_data

def flatten_landmarks(left_hand, right_hand, pose):
    """Flattens landmarks for all body parts into a single 1D NumPy array."""
    return np.concatenate([
        np.array(left_hand).flatten(),
        np.array(right_hand).flatten(),
        np.array(pose).flatten()
    ])

def has_valid_hands(left_hand, right_hand):
    """Checks if at least one hand was detected in the frame."""
    left_detected = not np.all(np.array(left_hand) == 0)
    right_detected = not np.all(np.array(right_hand) == 0)
    return left_detected or right_detected

def select_representative_frames_strict(frames_data, n_clusters=20):
    """
    Selects a fixed number of representative frames using KMeans clustering.
    This version is strict: it only processes frames where at least one hand is visible.
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
        batch_size=min(len(X), 256) # Use a batch size appropriate for the data
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
        distances_to_center = cdist(cluster_members, [cluster_center], 'euclidean')
        closest_member_idx_in_cluster = np.argmin(distances_to_center)
        # Map this back to the index in the original 'valid_frames' list
        closest_member_idx_in_X = cluster_member_indices[closest_member_idx_in_cluster]
        # And map that back to the index in the *original* 'frames_data' list
        original_frame_idx = valid_indices[closest_member_idx_in_X]

        representative_frames.append(frames_data[original_frame_idx])

    # Only return a result if we found exactly the right number of frames
    return representative_frames if len(representative_frames) == n_clusters else []

def process_and_save(video_path, output_dir, label, hands_model, pose_model):
    """
    Main processing pipeline for a single video.
    This function is designed to be thread-safe.
    """
    video_filename = os.path.basename(video_path)

    status, frames_data = sample_and_extract_landmarks(
        video_path, TARGET_FRAMES_PER_VIDEO, hands_model, pose_model, MIN_TOTAL_FRAMES
    )
    if status != 'SUCCESS':
        return video_filename, status, label

    representative_frames = select_representative_frames_strict(frames_data, N_CLUSTERS)
    if not representative_frames:
        return video_filename, 'ERROR_INSUFFICIENT_REPRESENTATIVE_FRAMES', label

    npy_filename = os.path.splitext(video_filename)[0] + '.npy'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, npy_filename)
    # Ensure the array has a consistent object type for saving
    np.save(output_path, np.array(representative_frames, dtype=object))

    return video_filename, 'PROCESSED_SUCCESSFULLY', label

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s', # Added threadName to log format
        filename=DETAILED_PROCESSING_LOG,
        filemode='w'
    )

    print("--- Video Processing Script Started (Multi-Threaded) ---")
    print(f"Using a maximum of {MAX_WORKERS} worker threads.")

    # Initialize MediaPipe models ONCE in the main thread
    print("Initializing MediaPipe models...")
    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5
    )
    mp_pose = mp.solutions.pose.Pose(
        model_complexity=POSE_MODEL_COMPLEXITY,
        enable_segmentation=True,
        min_detection_confidence=0.5
    )

    try:
        metadata_df = pd.read_csv(METADATA_FILE, encoding='utf-8')
        metadata_df.set_index('id', inplace=True)

        video_files = [f for f in os.listdir(VIDEOS_FOLDER) if f.lower().endswith(('.mp4', '.mov', '.avi'))]

        tasks = []
        for video_filename in video_files:
            video_id = os.path.splitext(video_filename)[0]
            if video_id in metadata_df.index:
                label = metadata_df.loc[video_id, 'label']
                tasks.append({
                    "video_path": os.path.join(VIDEOS_FOLDER, video_filename),
                    "output_dir": os.path.join(OUTPUT_FOLDER, str(label)),
                    "label": label
                })

        print(f"Found {len(tasks)} videos to process.")
        print(f"Detailed logs are being written to: {DETAILED_PROCESSING_LOG}\n")

        results = []
        start_time = time.time()

        # --- MULTITHREADING IMPLEMENTATION ---
        # Use ThreadPoolExecutor to manage a pool of threads
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all tasks to the executor. We pass the shared MediaPipe models to each task.
            # This returns a dictionary mapping each future to its corresponding task.
            future_to_task = {
                executor.submit(
                    process_and_save,
                    task["video_path"],
                    task["output_dir"],
                    task["label"],
                    mp_hands, # Pass the initialized model
                    mp_pose   # Pass the initialized model
                ): task for task in tasks
            }

            # Use tqdm to create a progress bar. as_completed() yields futures as they finish.
            for future in tqdm(as_completed(future_to_task), total=len(tasks), desc="Processing Videos", ncols=100):
                try:
                    # Get the result from the completed future
                    result = future.result()
                    results.append(result)
                except Exception as exc:
                    # Log any exceptions that occurred during the task execution
                    task = future_to_task[future]
                    video_path = task['video_path']
                    logging.error(f"Video {os.path.basename(video_path)} generated an exception: {exc}")
                    results.append((os.path.basename(video_path), f'ERROR_EXCEPTION_{exc.__class__.__name__}', task['label']))


        end_time = time.time()
        total_time = end_time - start_time

        print("-" * 40)
        print(f"\nProcessing finished. Total time: {total_time:.2f} seconds.")
        if tasks:
            print(f"Average time per video: {total_time / len(tasks):.2f} seconds.")

        if results:
            print("\nWriting final status summary...")
            log_df = pd.DataFrame(results, columns=['video_filename', 'status', 'label'])
            log_df.to_csv(FINAL_STATUS_LOG, index=False, encoding='utf-8')
            print(f"Final status log saved to: {FINAL_STATUS_LOG}")

            print("\n--- Processing Statistics ---")
            print(log_df['status'].value_counts())
        else:
            print("No results to log.")

    finally:
        # IMPORTANT: Clean up MediaPipe resources
        print("\nClosing MediaPipe models.")
        mp_hands.close()
        mp_pose.close()