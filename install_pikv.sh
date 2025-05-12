#!/bin/bash

# Exit on error
set -e

echo "Installing PiKV environment..."

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Conda is not installed. Please install Miniconda or Anaconda first."
    exit 1
fi

# Create conda environment
echo "Creating conda environment 'pikv'..."
conda create -n pikv python=3.11 -y

# Activate environment
echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate pikv

# Install PyTorch with CUDA support
echo "Installing PyTorch..."
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install other dependencies
echo "Installing other dependencies..."
pip install -r requirements.txt

# Install additional dependencies
echo "Installing additional dependencies..."
pip install transformers>=4.30.0
pip install datasets>=2.12.0
pip install tqdm>=4.65.0
pip install wandb>=0.15.0
pip install scipy>=1.10.0
pip install numpy<2.0.0  # Fix for NumPy 2.x compatibility issue

# Install PiKV package in development mode
echo "Installing PiKV package..."
pip install -e .

echo "Installation completed successfully!"
echo "To activate the environment, run: conda activate pikv" 