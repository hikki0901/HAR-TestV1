
import os
import sys

# Get the absolute path of "Recognize_Anything-Tag2Text"
ram_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "Recognize_Anything-Tag2Text"))

# Add it to sys.path
sys.path.append(ram_path)

import argparse
from PIL import Image
import torch

# Now import from "ram" inside "Recognize_Anything-Tag2Text"
from ram.models import ram
from ram import inference_ram as inference
from ram import get_transform

def process_images(image_dir, output_file, model, transform, device):
    with open(output_file, 'w') as f:
        for root, _, files in os.walk(image_dir):
            for file_name in files:
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    action_name = os.path.basename(os.path.dirname(root))  # Get action name
                    image_path = os.path.join(root, file_name)
                    image = Image.open(image_path).convert('RGB')
                    image_tensor = transform(image).unsqueeze(0).to(device)

                    # Forward pass through the model
                    with torch.no_grad():
                        res = inference(image_tensor, model)

                    # Write results to the file
                    f.write(f'Action: {action_name}\n')
                    f.write(f'Image: {file_name}\n')
                    f.write(f'Tags: {res[0]}\n')
                    f.write('-' * 40 + '\n')  # Separator for readability

def main():
    # **Direct paths** (no need for command-line arguments)
    image_dir = "UCF11_samples"   # Hardcoded input folder
    output_file = "image-features.txt"  # Hardcoded output file

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = get_transform(image_size=384)

    # Initialize the RAM model
    model = ram(pretrained='RAM/ram_swin_large_14m.pth',
                image_size=384,
                vit='swin_l')
    model.eval()
    model.to(device)

    # Process images and save results
    process_images(image_dir, output_file, model, transform, device)
    print(f"Processing complete. Features saved to {output_file}")

if __name__ == "__main__":
    main()