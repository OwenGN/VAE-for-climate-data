"""
timesteptrain.py
Training script for the Convolutional Variational Autoencoder (ConvVAE).
"""

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from dataset import load_netcdf, normalize_data, corrupt_data
from timestepmodel import ConditionalConvVAE, loss_function

batch_size = 32
epochs = 100
learning_rate = 1e-3
device = torch.device("mps" if torch.mps.is_available() else "cpu")
file_path = "./data/sample.nc"

X = load_netcdf(file_path)
Xt1 = X[1:]
X = X[:-1]
X = normalize_data(X)
Xt1 = normalize_data(Xt1)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
Xt1_tensor = torch.tensor(Xt1, dtype=torch.float32)

# Create DataLoader
dataset = TensorDataset(X_tensor, Xt1_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model, optimizer
model = ConditionalConvVAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in dataloader:
        x, xt1 = batch
        x, xt1 = x.to(device), xt1.to(device)

        optimizer.zero_grad()
        reconstructed_xt1, mu, logvar = model(x)
        loss = loss_function(reconstructed_xt1, xt1, mu, logvar)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataloader)}")

# Save trained model
torch.save(model.state_dict(), "conditional_vae_model.pth")
print("Model saved as conditional_vae_model.pth")


# Visualization
model.eval()
with torch.no_grad():
    sample = Xt1_tensor[:1].to(device)  # Take one sample for visualization
    reconstructed_sample, _, _ = model(sample)

# Plot original vs reconstructed
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(Xt1[0, 1, :, :], cmap="viridis")
plt.title("Original Data")
plt.subplot(1, 2, 2)
plt.imshow(reconstructed_sample[0, 1, :, :].cpu().numpy(), cmap="viridis")
plt.title("Reconstructed Data")
plt.show()
