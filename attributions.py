import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
# import wandb
import os

from src.data import OrigPlank
from src.models import ResNet50Classifier
import matplotlib.pyplot as plt
import numpy as np

    
def test(model, dataloader, device='cuda'):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader)
    count = 0
    with torch.no_grad():
        for images, labels, filename in progress_bar:
            images, labels = images.to(device).float(), labels.to(device)
            outputs, feature_maps = model(images)
            images = images.permute(0,2,3,1).detach().cpu().numpy().astype(np.uint8)
            import ipdb;ipdb.set_trace()
            # feature_maps = feature_maps.permute(0,2,3,1)
            # feature_maps = feature_maps.detach().cpu().numpy()

            _, predicted = torch.max(outputs, 1)
            correct_indices = (predicted == labels)
           
            for i, img in enumerate(images):
                filename = str(count) + "_" + str(labels[i]) + "_" + str(correct_indices[i]) + ".png"
                plt.imshow(img)
                print(img.shape)
                plt.savefig('trial.png')
    return

def main():

    device='cuda' if torch.cuda.is_available() else 'cpu'

    dataset_dir = "data/dataset/"
    dataset = OrigPlank(path=dataset_dir, transform=None)

    train_size = int(0.8 * len(dataset))
    test_size  = len(dataset) - train_size
    _, test_dataset = random_split(dataset, [train_size, test_size])

    test_dataloader = DataLoader(test_dataset, batch_size=32)


    modelname = "resnet50_np_cls"
    model = ResNet50Classifier(num_classes=2).to("cuda:0")

    load_path = os.path.join("pretrained_models", modelname, "trial_4_best_model.pth")
    model.load_state_dict(torch.load(load_path, map_location = "cuda:0"))


    test(model, test_dataloader, device)

    
if __name__ == "__main__":
    main()
    
    
