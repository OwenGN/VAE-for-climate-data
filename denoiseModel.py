"""
denoisemodel.py
Defines the Convolutional Variational Autoencoder (ConvVAE) model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvVAE(nn.Module):
    """
    Convolutional Variational Autoencoder (ConvVAE) for imputing missing climate data.
    """

    def __init__(self, latent_dim=512):
        super(ConvVAE, self).__init__()

        # Encoder
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

        # Latent space
        self.fc_mu = nn.Linear(128 * 10 * 4, latent_dim)
        self.fc_logvar = nn.Linear(128 * 10 * 4, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 128 * 10 * 4)

        # Decoder
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
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

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

    def decode(self, z):
        """
        Decodes latent space representation back into original data space.

        Args:
            z (torch.Tensor): Latent vector.

        Returns:
            torch.Tensor: Reconstructed data.
        """
        h = self.fc_decode(z)
        return self.decoder(h)

    def forward(self, x):
        """
        Forward pass through the VAE.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            reconstructed_x (torch.Tensor): Reconstructed input.
            mu (torch.Tensor): Mean of latent distribution.
            logvar (torch.Tensor): Log variance of latent distribution.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_function(reconstructed_x, x, mu, logvar):
    """
    Computes the loss for the Variational Autoencoder.

    Args:
        reconstructed_x (torch.Tensor): Reconstructed data.
        x (torch.Tensor): Original input data.
        mu (torch.Tensor): Mean of latent distribution.
        logvar (torch.Tensor): Log variance of latent distribution.

    Returns:
        torch.Tensor: Total loss (Reconstruction loss + KL divergence).
    """
    BCE = F.mse_loss(reconstructed_x, x, reduction='sum')  # Reconstruction loss
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())  # KL divergence
    return BCE + KLD