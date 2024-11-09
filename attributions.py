import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
# from captum.attr import IntegratedGradients
from xplique.wrappers import TorchWrapper
from xplique.attributions import Saliency
from xplique.metrics import Deletion
# import wandb
import os

from src.data import OrigPlank
from src.models import ResNet50Classifier, ResNet50VAEClassifier
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as F
import cv2

torch.manual_seed(42)
np.random.seed(42)

root_dir = "./attributions"

def test(model, dataloader, device='cuda', modelname = "resnet50vae_np_cls"):

    progress_bar = tqdm(dataloader)
    count = 0

    explainer = Saliency(model)
   
    for images, labels in progress_bar:
        images, labels = images.float(), labels
        explanations = explainer(images, labels)
        # outputs, _ = model(images.to(device))
        # _, predicted = torch.max(outputs, 1)
        # correct_indices = (predicted == labels).numpy()
        count = 0
        for i, img in enumerate(images):
            filename = "img_" + str(count) + "_basket" + str(labels[i]+1) + ".png"
            img = np.array(img).astype(np.uint8)
            plt.imshow(img)
            plt.imshow(explanations[i].numpy() * 255., cmap = "jet", alpha = 0.5)
            path = os.path.join(root_dir, modelname, filename)
            plt.savefig(path)
            plt.show()
            
        count += 1

    return

def main():


    device='cuda' if torch.cuda.is_available() else 'cpu'

    dataset_dir = "data/dataset/"
    dataset = OrigPlank(path=dataset_dir, transform=None)

    train_size = int(0.8 * len(dataset))
    test_size  = len(dataset) - train_size
    _, test_dataset = random_split(dataset, [train_size, test_size])

    test_dataloader = DataLoader(test_dataset, batch_size=1)


    modelname = "resnet50_np_cls"
    model = ResNet50Classifier(num_classes=2).to("cuda:0")

    load_path = os.path.join("pretrained_models", modelname, "trial_4_best_model.pth") 
    model.load_state_dict(torch.load(load_path, map_location = "cuda:0"))
    model.eval()
    wrapped_model = TorchWrapper(model, device)

    test(wrapped_model, test_dataloader, device, modelname)

    
if __name__ == "__main__":
    main()
    
    
