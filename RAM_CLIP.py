import os
import sys
import argparse
from PIL import Image
import torch
import clip
import numpy as np

# Get the absolute path of "Recognize_Anything-Tag2Text"
ram_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "Recognize_Anything-Tag2Text"))
sys.path.append(ram_path)

from ram.models import ram
from ram import inference_ram as inference
from ram import get_transform

def process_images(image_dir, output_file, ram_model, clip_model, preprocess, transform, device):
    with open(output_file, 'w') as f:
        for root, _, files in os.walk(image_dir):
            action_name = os.path.basename(root)  # Get action name from the correct folder level
            for file_name in files:
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(root, file_name)
                    image = Image.open(image_path).convert('RGB')

                    # Transform image for RAM model
                    image_tensor = transform(image).unsqueeze(0).to(device)
                    
                    # Extract tags from RAM
                    with torch.no_grad():
                        res = inference(image_tensor, ram_model)
                    ram_tags = res[0]  # Extracted textual tags
                    
                    # Extract CLIP image features
                    clip_image = preprocess(image).unsqueeze(0).to(device)
                    with torch.no_grad():
                        clip_image_features = clip_model.encode_image(clip_image)
                    
                    # Convert RAM tags to CLIP text embeddings
                    text_inputs = clip.tokenize(ram_tags).to(device)
                    with torch.no_grad():
                        clip_text_features = clip_model.encode_text(text_inputs)
                    
                    # Average text embeddings if multiple tags exist
                    clip_text_features = clip_text_features.mean(dim=0, keepdim=True)
                    
                    # Combine CLIP image and text embeddings
                    combined_features = torch.cat((clip_image_features, clip_text_features), dim=1).cpu().numpy()
                    
                    # Save to file
                    np.savetxt(f, combined_features, delimiter=',', header=f'Action: {action_name}, Image: {file_name}', comments='')
                    
                    print(f'Processed {file_name}')

def main():
    dataset_dirs = {
        "train": "dataset/train",
        "test": "dataset/test",
        "val": "dataset/val"
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = get_transform(image_size=384)

    # Load RAM model
    ram_model = ram(pretrained='RAM/ram_swin_large_14m.pth', image_size=384, vit='swin_l')
    ram_model.eval().to(device)
    
    # Load CLIP model and preprocessing function
    clip_model, preprocess = clip.load("ViT-L/14", device=device)
    clip_model.eval()
    
    # Process images and extract features for each dataset split
    for split, image_dir in dataset_dirs.items():
        output_file = f"combined_features_{split}.txt"  # Output file per dataset split
        process_images(image_dir, output_file, ram_model, clip_model, preprocess, transform, device)
        print(f"Processing complete for {split}. Features saved to {output_file}")

if __name__ == "__main__":
    main()