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
from src.models import ResNet50VAEClassifier
from src.utils import kld

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group("nccl", rank = rank, world_size = world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

# Training function with a progress bar
def train_one_epoch(model, dataloader, criterion, optimizer, epoch, epochs=10, device='cuda', rank = 0):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", position=rank)
    
    for images, labels in progress_bar:
        images, labels = images.to(device).float(), labels.to(device)
        optimizer.zero_grad()
        outputs, mu, logvar = model(images)

        loss = criterion(outputs, labels) + kld(mu, logvar)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        batch_loss = loss.item() / labels.size(0)
        batch_accuracy = 100 * correct / total

        # if rank == 0:
            # wandb.log({"train_loss": batch_loss, "train_accuracy": batch_accuracy, "epoch": epoch + 1})
        
        progress_bar.set_postfix({"Train Loss": batch_loss, "Train Accuracy": batch_accuracy})
    
    avg_loss = running_loss / len(dataloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

    
def test(model, dataloader, criterion, epoch, epochs, device='cuda', rank = 0):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", position=rank)


    with torch.no_grad():
        for images, labels in progress_bar:
            images, labels = images.to(device).float(), labels.to(device)
            outputs, mu, logvar = model(images)
            loss = criterion(outputs, labels) + kld(mu, logvar)
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            batch_loss = loss.item() / labels.size(0)
            batch_accuracy = 100 * correct / total

            # if rank == 0:
            #     wandb.log({"test_loss": batch_loss, "test_accuracy": batch_accuracy})
            
            progress_bar.set_postfix({"Test Loss": batch_loss, "Test Accuracy": batch_accuracy})


    avg_loss = running_loss / len(dataloader)
    accuracy = 100 * correct / total
    if rank == 0:
        print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy

def main(rank, world_size):
    setup(rank, world_size)

    device='cuda' if torch.cuda.is_available() else 'cpu'

    epochs = 10
    dataset_dir = "data/dataset/"
    dataset = OrigPlank(path=dataset_dir, transform=None)

    train_size = int(0.8 * len(dataset))
    test_size  = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_dataloader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler)
    test_dataloader = DataLoader(test_dataset, batch_size=32, sampler=test_sampler)

    modelname = "resnet50vae_np_cls"
    model = ResNet50VAEClassifier(latent_dim=128, num_classes=2).to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_accuracy = 0.0
    save_path = os.path.join("pretrained_models", modelname)
    os.makedirs(save_path, exist_ok = True)
    trial_number = str(len(os.listdir(save_path)))
    modelname = modelname + "_" + trial_number

    # if rank == 0:
    #     wandb.init(project="vfm", entity='gaga13', name = modelname)
    #     wandb.watch(model, criterion, log="all", log_freq=10)


    # Training the model
    for e in range(epochs):

        train_loss, train_acc = train_one_epoch(model, train_dataloader, criterion, optimizer, e, epochs=10, device = device, rank = rank)

        if rank == 0:
            print(f"Epoch {e+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
            test_loss, test_acc = test(model, test_dataloader, criterion, e, epochs, device, rank = rank)
            
            if test_acc > best_accuracy:
                best_accuracy = test_acc
                torch.save(model.state_dict(), os.path.join(save_path, "best_model.pth"))
                print(f"Best model saved with Test Accuracy: {best_accuracy:.2f}%")
    
    # if rank == 0:
    #     wandb.finish()
    cleanup()

    
if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(main, args = (world_size,), nprocs = world_size, join = True)
    
    
