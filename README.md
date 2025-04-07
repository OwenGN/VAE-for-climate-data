# Climate Data Reconstruction and Forecasting with Convolutional VAEs

This repository contains two deep learning models for working with gridded climate data:

- **ConvVAE** for imputing missing values in spatial climate data.
- **ConditionalConvVAE** for forecasting future time steps conditioned on historical input.

Both models use convolutional encoders and decoders to capture spatiotemporal dependencies in netCDF climate datasets.

_____________________________________________________________________________________

## 📁 Project Structure

├── dataset.py # Functions to load and preprocess netCDF data 
├── models
  └── denoisemodel.py # Defines denoisemodel for imputation
  └── timestepmodel.py # Defines timestepmodel for forecasting 
├── denoisetrain.py # Training script for denoisemodel 
├── timesteptrain.py# Training script for timestepmodel 
├── README.md # This file



# requirements
- Python
- PyTorch
- NumPy
- netCDF4
- matplotlib

# Train the model
python3 denoisetrain.py #for imputation model

python3 timesteptrain.py #for forecasting model
