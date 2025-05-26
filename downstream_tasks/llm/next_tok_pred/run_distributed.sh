#!/bin/bash

# Multi-GPU Distributed Training Script for PiKV
# This script uses torchrun to launch distributed training

# Set the number of GPUs to use
NGPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Detected $NGPUS GPUs"

# Check if we have at least 2 GPUs
if [ $NGPUS -lt 2 ]; then
    echo "Warning: Only $NGPUS GPU(s) detected. Distributed training works best with 2+ GPUs."
    echo "Running with available GPUs..."
fi

# Set training parameters
EPOCHS=${1:-10}
SAVE_EVERY=${2:-5}
MODEL_TYPE=${3:-pikv}

echo "Starting distributed training with:"
echo "  - GPUs: $NGPUS"
echo "  - Epochs: $EPOCHS"
echo "  - Save every: $SAVE_EVERY epochs"
echo "  - Model type: $MODEL_TYPE"

# Create data directory if it doesn't exist
mkdir -p data

# Check if training data exists
if [ ! -f "data/train.txt" ]; then
    echo "Creating training data..."
    echo "The quick brown fox jumps over the lazy dog. This is a simple test sentence for training our language model." > data/train.txt
    echo "Machine learning is a fascinating field that combines statistics, computer science, and domain expertise." >> data/train.txt
    echo "Natural language processing enables computers to understand and generate human language." >> data/train.txt
    echo "Deep learning models like transformers have revolutionized the field of artificial intelligence." >> data/train.txt
    echo "The attention mechanism allows models to focus on relevant parts of the input sequence." >> data/train.txt
    echo "Large language models can generate coherent and contextually appropriate text." >> data/train.txt
    echo "Training neural networks requires careful tuning of hyperparameters and optimization algorithms." >> data/train.txt
    echo "The transformer architecture has become the foundation for many state-of-the-art language models." >> data/train.txt
    echo "Self-attention mechanisms enable models to capture long-range dependencies in sequences." >> data/train.txt
    echo "Pre-training on large corpora followed by fine-tuning has become a standard approach in NLP." >> data/train.txt
fi

# Launch distributed training using torchrun
echo "Launching distributed training..."
torchrun \
    --nproc_per_node=$NGPUS \
    --master_port=29500 \
    train_distributed.py \
    --epochs $EPOCHS \
    --save_every $SAVE_EVERY \
    --model_type $MODEL_TYPE

echo "Distributed training completed!"

# Check if checkpoints were created
if [ -d "checkpoints" ]; then
    echo "Checkpoints saved in:"
    ls -la checkpoints/
else
    echo "No checkpoints found."
fi 