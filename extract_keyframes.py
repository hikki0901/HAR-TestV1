import os
import cv2
import numpy as np
from findframe import Frame

# Input and output directories
UCF11_FOLDER = "./UCF11"
KEYFRAMES_FOLDER = "./UCF11_keyframes"

# Create the keyframes folder if it doesn't exist
if not os.path.exists(KEYFRAMES_FOLDER):
    os.makedirs(KEYFRAMES_FOLDER)

def process_video(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    curr_frame = None
    prev_frame = None
    frame_diffs = []
    frames = []
    success, frame = cap.read()
    i = 0
    FRAME = Frame(0, 0)

    # Read frames and compute frame differences
    while success:
        luv = cv2.cvtColor(frame, cv2.COLOR_BGR2LUV)
        curr_frame = luv

        if curr_frame is not None and prev_frame is not None:
            diff = cv2.absdiff(curr_frame, prev_frame)
            diff_sum = np.sum(diff)
            diff_sum_mean = diff_sum / (diff.shape[0] * diff.shape[1])
            frame_diffs.append(diff_sum_mean)
            frame_obj = Frame(i, diff_sum_mean)
            frames.append(frame_obj)
        elif curr_frame is not None and prev_frame is None:
            diff_sum_mean = 0
            frame_diffs.append(diff_sum_mean)
            frame_obj = Frame(i, diff_sum_mean)
            frames.append(frame_obj)

        prev_frame = curr_frame
        i += 1
        success, frame = cap.read()

    cap.release()

    # Detect possible keyframes
    possible_frames, start_id_spot_old, end_id_spot_old = FRAME.find_possible_frame(frames)

    # Optimize the possible keyframes
    optimized_frames, start_id_spot, end_id_spot = FRAME.optimize_frame(possible_frames, frames)

    # Save the keyframes
    cap = cv2.VideoCapture(video_path)
    for frame in optimized_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame.id)
        success, keyframe = cap.read()
        if success:
            output_path = os.path.join(output_folder, f"frame_{frame.id}.jpg")
            cv2.imwrite(output_path, keyframe)
    cap.release()

# Iterate through the UCF11 dataset
for category in os.listdir(UCF11_FOLDER):
    category_path = os.path.join(UCF11_FOLDER, category)
    if not os.path.isdir(category_path):
        continue

    for subfolder in os.listdir(category_path):
        subfolder_path = os.path.join(category_path, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        # Create output directory for the subfolder
        subfolder_output_path = os.path.join(KEYFRAMES_FOLDER, category, subfolder)
        if not os.path.exists(subfolder_output_path):
            os.makedirs(subfolder_output_path)

        for file in os.listdir(subfolder_path):
            if file.endswith(".mpg"):  # Adjust extension if needed
                video_path = os.path.join(subfolder_path, file)

                # Process the video
                process_video(video_path, subfolder_output_path)

print("Keyframe extraction complete!")
