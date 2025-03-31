"""
dataset.py
Functions to load and preprocess netCDF climate data.
"""

import os
import numpy as np
import netCDF4 as nc

def load_netcdf(file_path):
    """
    Load netCDF data from the given file path.

    Args:
        file_path (str): Path to the netCDF file.

    Returns:
        dataset (np.ndarray): Loaded netCDF dataset made to a numpy file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    dataset = nc.Dataset(file_path, 'r')
    print("The variables are as follows:",dataset.variables.keys())
    dataset = np.array(dataset.variables['z'][:])
    return dataset

def normalize_data(data):
    """
    Normalize the data using mean and standard deviation.

    Args:
        data (np.ndarray): Raw data.

    Returns:
        np.ndarray: Normalized data.
    """
    return (data - np.mean(data)) / np.std(data)

def corrupt_data(data, perc_corruption=30):
    """
    Corrupt the data by setting random grid points to zero.
    The grid points are on the latitude, logitude grid 
    and are selected for each timestep for each channel.
    Works for data in the shape: time, level/channel, latitude, longitude.

    Args:
        data (np.ndarray): The input data.
        perc_corruption (int): Percentage of elements to corrupt per grid.

    Returns:
        np.ndarray: Corrupted data.
    """
    corrupted_data = data.copy()
    for t in range(data.shape[0]):  # Iterate over time steps
        for c in range(data.shape[1]):  # Iterate over levels
            lat_indices, lon_indices = np.unravel_index(
                np.random.choice(data.shape[2] * data.shape[3], (perc_corruption * data.shape[2] * data.shape[3]) // 100, replace=False),
                (data.shape[2], data.shape[3])
            )
            corrupted_data[t, c, lat_indices, lon_indices] = 0
    return corrupted_data
