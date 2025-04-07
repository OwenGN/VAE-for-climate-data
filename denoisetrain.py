"""
train.py
Training script for the Convolutional Variational Autoencoder (ConvVAE).
"""

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from dataset import load_netcdf, normalize_data, corrupt_data
from denoisemodel import ConvVAE, loss_function

batch_size = 32
epochs = 100
learning_rate = 1e-3
device = torch.device("mps" if torch.mps.is_available() else "cpu")
file_path = "./data/z1979.nc"
corruption_percentage = 30

X = load_netcdf(file_path)
X_corrupted = corrupt_data(X,corruption_percentage)
X = normalize_data(X)
X_corrupted = normalize_data(X_corrupted)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
X_corrupted_tensor = torch.tensor(X_corrupted, dtype=torch.float32)

# Create DataLoader
dataset = TensorDataset(X_tensor, X_corrupted_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model, optimizer
model = ConvVAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in dataloader:
        x, x_corrupted = batch
        x, x_corrupted = x.to(device), x_corrupted.to(device)

        optimizer.zero_grad()
        reconstructed_x, mu, logvar = model(x_corrupted)
        loss = loss_function(reconstructed_x, x, mu, logvar)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataloader)}")

# Save trained model
torch.save(model.state_dict(), "vae_model.pth")
print("Model saved as vae_model.pth")


# Visualization
model.eval()
with torch.no_grad():
    sample = X_corrupted_tensor[:1].to(device)  # Take one sample for visualization
    reconstructed_sample, _, _ = model(sample)

# Plot original vs reconstructed
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(X[0, 1, :, :], cmap="viridis")
plt.title("Original Data")
plt.subplot(1, 2, 2)
plt.imshow(reconstructed_sample[0, 1, :, :].cpu().numpy(), cmap="viridis")
plt.title("Reconstructed Data")
plt.show()
