import torch

# Model Configuration for PiKV
config = {
    'num_experts': 4,  # Number of experts in MoE
    'hidden_size': 256,  # Hidden size for each expert
    'num_heads': 8,  # Number of attention heads
    'kv_cache_size': 128,  # Cache size for each expert
    'epochs': 10,  # Number of epochs for training
    'batch_size': 32,  # Batch size
    'learning_rate': 1e-3,  # Learning rate
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'pyramidal_cache': True,  # Use pyramidal cache allocation strategy
    'cache_decrement': 10,  # Cache size decrement for each layer
    'num_layers': 5,  # Number of layers in the model
}
