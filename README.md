# VAE for Climate Data Imputation

This repository implements a Convolutional Variational Autoencoder (ConvVAE) for imputing missing climate data from netCDF files.

🚀 Features
Loads netCDF climate data and preprocesses it.

Corrupts data artificially to train the model on missing values.

Implements a ConvVAE for reconstructing the missing climate data.

Provides a training script for model optimization.

# Repository Structure
VAE-for-climate-data/
│── dataset.py         # Functions to load and preprocess netCDF data
│── denoiseModel.py    # Defines the ConvVAE model
│── train.py           # Training script
│── README.md          # Documentation

# requirements
pip3 install torch netCDF4 numpy matplotlib

# Train the model
python3 train.py
