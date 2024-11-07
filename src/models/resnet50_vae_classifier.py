import torch
import torch.nn as nn
from timm.models._registry import register_model
import torchvision

# Custom Model with ResNet50 Encoder and VAE Reparameterization
class ResNet50VAEClassifier(nn.Module):
    def __init__(self, latent_dim=128, num_classes=10):
        super(ResNet50VAEClassifier, self).__init__()
        # Load ResNet50 and remove fully connected layer
        self.encoder = torchvision.models.resnet.resnet50(weights=None)
        self.encoder.fc = nn.Identity()  # Removing FC layer for feature extraction
        self.fc_mu = nn.Linear(2048, latent_dim)  # Learned mean
        self.fc_logvar = nn.Linear(2048, latent_dim)  # Learned log variance
        self.fc_decision = nn.Linear(latent_dim, num_classes)  # Decision layer

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        features = self.encoder(x)  # Extract features
        mu = self.fc_mu(features)  # Mean
        logvar = self.fc_logvar(features)  # Log variance
        z = self.reparameterize(mu, logvar)  # New latent vector
        return self.fc_decision(z), mu, logvar  # Output logits and latent parameters
    
# __all__ = []


# @register_model
# def resnet50vaeclassifier(pretrained=False, **kwargs):
#     model = ResNet50VAEClassifier(n_classes=1)
#     return model