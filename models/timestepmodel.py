"""
timestepmdoel.py
Defines the Conditional Convolutional Variational Autoencoder (ConditionalConvVAE) model
for forecasting future climate states based on historical data.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

class ConditionalConvVAE(nn.Module):
    def __init__(self, latent_dim=512):
        super(ConditionalConvVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=(3, 4), stride=2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(16, 32, kernel_size=(3,3), stride=2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(32, 64, kernel_size=(4,4), stride=2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(64, 128, kernel_size=(4,3), stride=2),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(128 * 10 * 4, latent_dim)
        self.fc_logvar = nn.Linear(128 * 10 * 4, latent_dim)
        self.fc_decode = nn.Linear(latent_dim + (128 * 10 * 4), 128 * 10 * 4)
        
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, 4, 10)),
            nn.ConvTranspose2d(128, 64, kernel_size=(4,3), stride=2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.ConvTranspose2d(64, 32, kernel_size=(4,4), stride=2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.ConvTranspose2d(32, 16, kernel_size=(3,3), stride=2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.ConvTranspose2d(16, 2, kernel_size=(3,4), stride=2)
        )
    
    def encode(self, x):
        """
        Encodes input data into latent space.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, latitudes, longitude).
        
        Returns:
            mu (torch.Tensor): Mean of latent space distribution.
            logvar (torch.Tensor): Log variance of latent space distribution.
            h (torch.Tensor): Encoded feature representation.
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar, h
    
    def reparameterize(self, mu, logvar):
        """
        Performs the reparameterization trick to sample from latent space.
        
        Args:
            mu (torch.Tensor): Mean of latent space distribution.
            logvar (torch.Tensor): Log variance of latent space distribution.
        
        Returns:
            z (torch.Tensor): Sampled latent vector.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, h):
        """
        Decodes latent space representation back into original data space.
        
        Args:
            z (torch.Tensor): Latent vector.
            h (torch.Tensor): Encoded feature representation.
        
        Returns:
            torch.Tensor: Reconstructed future climate state.
        """
        z = torch.cat([z, h], dim=1)
        h = self.fc_decode(z)
        return self.decoder(h)
    
    def forward(self, x):
        """
        Forward pass through the Conditional VAE.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            reconstructed_x (torch.Tensor): Reconstructed next time step.
            mu (torch.Tensor): Mean of latent distribution.
            logvar (torch.Tensor): Log variance of latent distribution.
        """
        mu, logvar, h = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, h), mu, logvar
    
def loss_function(reconstructed_xt1, xt1, mu, logvar):
    """
    Computes the loss for the Conditional Variational Autoencoder.
    
    Args:
        reconstructed_xt1 (torch.Tensor): Reconstructed next time step.
        xt1 (torch.Tensor): Ground truth next time step.
        mu (torch.Tensor): Mean of latent distribution.
        logvar (torch.Tensor): Log variance of latent distribution.
    
    Returns:
        torch.Tensor: Total loss (Reconstruction loss + KL divergence).
    """
    BCE = nn.functional.mse_loss(reconstructed_xt1, xt1, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD