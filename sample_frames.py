import os
import shutil

# Paths
input_folder = "UCF11_keyframes"
output_folder = "UCF11_samples"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Traverse through the folder structure
for category in os.listdir(input_folder):
    category_path = os.path.join(input_folder, category)
    if os.path.isdir(category_path):
        for video in os.listdir(category_path):
            video_path = os.path.join(category_path, video)
            if os.path.isdir(video_path):
                keyframes = os.listdir(video_path)
                keyframes.sort()  # Ensure frames are sorted
                if keyframes:  # Check if there are any keyframes
                    selected_frame = keyframes[0]  # Select the first frame
                    selected_frame_path = os.path.join(video_path, selected_frame)

                    # Create the same folder structure in the output folder
                    video_output_path = os.path.join(output_folder, category, video)
                    os.makedirs(video_output_path, exist_ok=True)

                    # Copy the selected frame to the respective folder in the output
                    shutil.copy(selected_frame_path, os.path.join(video_output_path, selected_frame))

print("Frames have been successfully copied to the UCF11_samples folder.")