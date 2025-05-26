#!/usr/bin/env python3
"""
Multi-GPU Distributed Training Script for PiKV
Usage: 
  Method 1 (Recommended): torchrun --nproc_per_node=4 train_distributed.py --epochs 10
  Method 2: python train_distributed.py --epochs 10 --use_spawn
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer
from tqdm import tqdm
import math
from typing import Tuple

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from core.pipeline.distributed.pikv_moe import DistributedPiKVMoE
from core.single.normal_moe import StandardMoE
from core.single.config import config

class PiKVLanguageModel(nn.Module):
    """
    Language model wrapper for DistributedPiKVMoE
    """
    def __init__(self, rank=4, alpha=1.0):
        super(PiKVLanguageModel, self).__init__()
        
        # Add embedding layer to handle token IDs
        self.embedding = nn.Embedding(config['vocab_size'], config['hidden_size'])
        
        # Use the distributed PiKV MoE
        self.pikv_moe = DistributedPiKVMoE(
            hidden_size=config['hidden_size'],
            num_experts=config['num_experts'],
            expert_size=config['hidden_size'] * 2,  # Expert size
            num_heads=config['num_heads'],
            top_k=2,
            compression_ratio=0.5
        )
        
        # Projection to vocabulary size
        self.vocab_proj = nn.Linear(config['hidden_size'], config['vocab_size'])
    
    def forward(self, x):
        # Handle both token IDs (2D) and embeddings (3D) as input
        if len(x.shape) == 2:  # Token IDs: [batch_size, seq_len]
            x = self.embedding(x)  # Convert to embeddings: [batch_size, seq_len, hidden_size]
        elif len(x.shape) == 3:  # Already embeddings: [batch_size, seq_len, hidden_size]
            pass
        else:
            raise ValueError(f"Input must be 2D (token IDs) or 3D (embeddings), got shape: {x.shape}")
        
        # Forward through PiKV MoE
        output, lb_loss = self.pikv_moe(x)
        
        # Project to vocabulary size
        logits = self.vocab_proj(output)
        
        return logits, lb_loss

class TextDataset:
    def __init__(self, file_path, tokenizer, max_length=32):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load and tokenize text
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Tokenize text with truncation
        tokens = tokenizer.encode(text, max_length=10000, truncation=True)  # Limit initial tokenization
        
        # Create sequences with fixed length
        self.sequences = []
        for i in range(0, len(tokens) - max_length, max_length):
            sequence = tokens[i:i + max_length]
            if len(sequence) == max_length:
                # Split into input and target
                input_sequence = sequence[:-1]  # All tokens except last
                target_sequence = sequence[1:]  # All tokens except first
                self.sequences.append((input_sequence, target_sequence))
        
        print(f"Created {len(self.sequences)} sequences of length {max_length-1}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        input_sequence, target_sequence = self.sequences[idx]
        return torch.tensor(input_sequence), torch.tensor(target_sequence)

def ddp_setup():
    """
    Initialize the distributed process group for torchrun
    """
    # torchrun sets these environment variables automatically
    init_process_group(backend="nccl")
    
    # Get rank and world size from environment
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    # Set the GPU device for this process
    torch.cuda.set_device(local_rank)
    
    return rank, local_rank, world_size

def ddp_setup_manual(rank: int, world_size: int):
    """
    Initialize the distributed process group manually (for mp.spawn)
    
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    
    # Set the GPU device for this process
    torch.cuda.set_device(rank)
    
    # Initialize the process group
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

class DistributedTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
        model_type: str = "pikv"
    ):
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.model_type = model_type
        
        # Wrap model with DDP
        self.model = DDP(model, device_ids=[gpu_id])
        
    def _run_batch(self, source, targets):
        """Run a single training batch"""
        self.optimizer.zero_grad()
        
        # Forward pass
        if hasattr(self.model.module, 'forward'):
            # For PiKVLanguageModel
            if isinstance(self.model.module, PiKVLanguageModel):
                output, lb_loss = self.model(source)
                # Calculate main loss
                criterion = nn.CrossEntropyLoss()
                main_loss = criterion(
                    output.reshape(-1, output.size(-1)),
                    targets.reshape(-1)
                )
                loss = main_loss + lb_loss
            else:
                # For standard models
                output = self.model(source)
                criterion = nn.CrossEntropyLoss()
                loss = criterion(
                    output.reshape(-1, output.size(-1)),
                    targets.reshape(-1)
                )
        else:
            raise ValueError(f"Model {type(self.model.module)} does not have forward method")
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def _run_epoch(self, epoch):
        """Run a single training epoch"""
        batch_size = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {batch_size} | Steps: {len(self.train_data)}")
        
        # Set epoch for distributed sampler
        if isinstance(self.train_data.sampler, DistributedSampler):
            self.train_data.sampler.set_epoch(epoch)
        
        total_loss = 0
        num_batches = 0
        
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            
            loss = self._run_batch(source, targets)
            total_loss += loss
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Average Loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def _save_checkpoint(self, epoch):
        """Save model checkpoint (only from rank 0)"""
        if self.gpu_id == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.module.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }
            
            os.makedirs('checkpoints', exist_ok=True)
            checkpoint_path = f"checkpoints/distributed_{self.model_type}_epoch_{epoch}.pt"
            torch.save(checkpoint, checkpoint_path)
            print(f"[GPU{self.gpu_id}] Checkpoint saved: {checkpoint_path}")
    
    def train(self, max_epochs: int):
        """Main training loop"""
        self.model.train()
        
        for epoch in range(max_epochs):
            avg_loss = self._run_epoch(epoch)
            
            # Save checkpoint periodically
            if epoch % self.save_every == 0:
                self._save_checkpoint(epoch)
        
        # Save final checkpoint
        self._save_checkpoint(max_epochs - 1)

def load_train_objs(model_type: str = "pikv"):
    """Load training objects (dataset, model, optimizer)"""
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # Create dataset
    dataset = TextDataset('data/train.txt', tokenizer, max_length=32)
    
    # Create model
    if model_type == "pikv":
        model = PiKVLanguageModel(rank=4, alpha=1.0)
    else:
        model = StandardMoE()
    
    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    return dataset, model, optimizer

def prepare_dataloader(dataset, batch_size: int):
    """Prepare distributed dataloader"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,  # Don't shuffle when using DistributedSampler
        sampler=DistributedSampler(dataset)
    )

def main_torchrun(total_epochs: int, save_every: int, model_type: str):
    """Main training function for torchrun"""
    # Setup distributed training
    rank, local_rank, world_size = ddp_setup()
    
    print(f"[Rank {rank}] Starting training on GPU {local_rank}")
    
    # Load training objects
    dataset, model, optimizer = load_train_objs(model_type)
    
    # Prepare dataloader
    train_data = prepare_dataloader(dataset, batch_size=4)
    
    # Create trainer
    trainer = DistributedTrainer(
        model=model,
        train_data=train_data,
        optimizer=optimizer,
        gpu_id=local_rank,
        save_every=save_every,
        model_type=model_type
    )
    
    # Start training
    trainer.train(total_epochs)
    
    # Cleanup
    destroy_process_group()

def main_spawn(rank: int, world_size: int, total_epochs: int, save_every: int, model_type: str):
    """Main training function for mp.spawn"""
    # Setup distributed training
    ddp_setup_manual(rank, world_size)
    
    # Load training objects
    dataset, model, optimizer = load_train_objs(model_type)
    
    # Prepare dataloader
    train_data = prepare_dataloader(dataset, batch_size=4)
    
    # Create trainer
    trainer = DistributedTrainer(
        model=model,
        train_data=train_data,
        optimizer=optimizer,
        gpu_id=rank,
        save_every=save_every,
        model_type=model_type
    )
    
    # Start training
    trainer.train(total_epochs)
    
    # Cleanup
    destroy_process_group()

def spawn_processes(fn, args, nprocs):
    """Spawn multiple processes for distributed training"""
    import multiprocessing as mp
    processes = []
    
    for rank in range(nprocs):
        p = mp.Process(target=fn, args=(rank,) + args)
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()

def run_distributed_training(args):
    """Launch distributed training"""
    # Check if we're running under torchrun
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        print("Running under torchrun...")
        main_torchrun(args.epochs, args.save_every, args.model_type)
    elif args.use_spawn:
        print("Using mp.spawn for distributed training...")
        world_size = torch.cuda.device_count()
        
        if world_size < 2:
            print("Warning: Only 1 GPU detected. Distributed training requires at least 2 GPUs.")
            print("Running on single GPU instead...")
            # Fall back to single GPU training
            main_spawn(0, 1, args.epochs, args.save_every, args.model_type)
        else:
            print(f"Starting distributed training on {world_size} GPUs")
            spawn_processes(
                main_spawn,
                (world_size, args.epochs, args.save_every, args.model_type),
                world_size
            )
    else:
        print("Error: Please use torchrun or --use_spawn flag")
        print("Recommended usage: torchrun --nproc_per_node=4 train_distributed.py --epochs 10")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Distributed PiKV Training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--save_every', type=int, default=5, help='Save checkpoint every N epochs')
    parser.add_argument('--model_type', type=str, choices=['pikv', 'standard'], 
                       default='pikv', help='Model type to train')
    parser.add_argument('--use_spawn', action='store_true', 
                       help='Use mp.spawn instead of torchrun (not recommended)')
    
    args = parser.parse_args()
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. This script requires GPU support.")
        sys.exit(1)
    
    print(f"Available GPUs: {torch.cuda.device_count()}")
    print(f"Training {args.model_type} model for {args.epochs} epochs")
    
    run_distributed_training(args) 