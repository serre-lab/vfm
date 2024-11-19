import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
from captum.attr import IntegratedGradients, Occlusion

from src.data import OrigPlank
from src.models import ResNet50VAEClassifier, ResNet50Classifier
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as F
import cv2

# Seed for reproducibility
torch.manual_seed(42)

# Directory for saving Integrated Gradients overlayed images
output_dir = "./attributions/integrated_gradients"
os.makedirs(output_dir, exist_ok=True)

def compute_and_save_ig(model, dataloader, device='cuda', modelname="resnet50vae_np_cls", explainer = "ig"):
    model.eval()
    
    # Initialize Integrated Gradients
    if explainer == "ig":
        exp = IntegratedGradients(model)
    else:
        exp = Occlusion(model)
    
    progress_bar = tqdm(dataloader)
    count = 0
    for images, labels in progress_bar:
        images, labels = images.to(device).float(), labels.to(device)
        
        # Compute attributions using Integrated Gradients
        if explainer == "ig":
            attributions = exp.attribute(images, target=labels, n_steps=50, internal_batch_size = 16)
        else:
            attributions = exp.attribute(images, target=labels, sliding_window_shapes=(1,30,40))

        # Convert images and attributions for overlaying and saving
        images_np = images.permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.uint8)
        attributions = attributions.permute(0, 2, 3, 1).detach().cpu().numpy()

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        correct_indices = (predicted == labels).cpu().numpy()
        
        for i, (img, attr) in enumerate(zip(images_np, attributions)):
            # Normalize and colorize the attribution map
            attr_map = np.abs(attr).mean(axis=-1)  # Aggregate across color channels
            attr_map = cv2.normalize(attr_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            attr_map_colored = cv2.applyColorMap(attr_map, cv2.COLORMAP_MAGMA)
            
            # Overlay attribution map on the original image
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            alpha = 0.5
            overlayed_image = cv2.addWeighted(img_bgr, 1 - alpha, attr_map_colored, alpha, 0)
            
            # Convert back to RGB and save the overlayed image
            overlayed_image_rgb = cv2.cvtColor(overlayed_image, cv2.COLOR_BGR2RGB)
            filename = f"img_{count}_label_{labels[i].item()}_correct_{correct_indices[i]}.png"
            output_path = os.path.join(output_dir, modelname, filename)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.imsave(output_path, overlayed_image_rgb)
            
            count += 1

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Dataset and DataLoader setup
    dataset_dir = "data/dataset/"
    dataset = OrigPlank(path=dataset_dir, transform=None)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    _, test_dataset = random_split(dataset, [train_size, test_size])
    test_dataloader = DataLoader(test_dataset, batch_size=1)

    # Load model and pretrained weights
    modelname = "resnet50_np_cls"
    model = ResNet50Classifier(num_classes=2).to(device)
    load_path = os.path.join("pretrained_models", modelname, "trial_3_best_model.pth") 
    model.load_state_dict(torch.load(load_path, map_location=device))

    # Run Integrated Gradients overlay and save results
    compute_and_save_ig(model, test_dataloader, device, modelname, "occlusion")

if __name__ == "__main__":
    main()
