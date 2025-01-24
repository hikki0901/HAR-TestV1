import os
import shutil
import random

# Define paths
original_dataset_dir = "UCF11_samples"  # Change this if your dataset is elsewhere
new_dataset_dir = "dataset"  # New structured dataset

# Train, validation, and test split ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Ensure the new dataset directory exists
os.makedirs(new_dataset_dir, exist_ok=True)

for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(new_dataset_dir, split), exist_ok=True)

# Loop through each category (e.g., basketball, biking)
for category in os.listdir(original_dataset_dir):
    category_path = os.path.join(original_dataset_dir, category)

    if not os.path.isdir(category_path):
        continue  # Skip non-directory files

    # Gather all image file paths from different video subfolders
    image_files = []
    for subfolder in os.listdir(category_path):
        subfolder_path = os.path.join(category_path, subfolder)
        if os.path.isdir(subfolder_path):
            for img in os.listdir(subfolder_path):
                if img.endswith(('.jpg', '.png')):  # Ensure it's an image file
                    image_files.append(os.path.join(subfolder_path, img))

    # Shuffle images randomly
    random.shuffle(image_files)

    # Split into train, validation, and test
    total_images = len(image_files)
    train_end = int(total_images * train_ratio)
    val_end = train_end + int(total_images * val_ratio)

    train_images = image_files[:train_end]
    val_images = image_files[train_end:val_end]
    test_images = image_files[val_end:]

    # Function to copy files into respective directories
    def copy_files(image_list, split):
        split_category_dir = os.path.join(new_dataset_dir, split, category)
        os.makedirs(split_category_dir, exist_ok=True)

        for img_path in image_list:
            filename = os.path.basename(img_path)
            dest_path = os.path.join(split_category_dir, filename)
            shutil.copy(img_path, dest_path)

    # Copy files into new structure
    copy_files(train_images, "train")
    copy_files(val_images, "val")
    copy_files(test_images, "test")

    print(f"Processed {category}: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test")

print("Dataset successfully reorganized!")
