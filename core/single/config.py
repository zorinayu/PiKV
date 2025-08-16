"""
PiKV Configuration
Essential configuration parameters for the system
"""

config = {
    # Model dimensions
    'hidden_size': 1024,
    'vocab_size': 50257,
    'num_layers': 12,
    'num_heads': 16,
    
    # MoE configuration
    'num_experts': 8,
    'top_k': 2,
    'expert_size': 2048,
    
    # Cache configuration
    'kv_cache_size': 4096,
    'cache_decrement': 256,
    'num_layers': 12,
    
    # Training configuration
    'dropout': 0.1,
    'learning_rate': 1e-4,
    'weight_decay': 0.01,
    
    # Compression configuration
    'compression_ratio': 0.5,
    'importance_threshold': 0.5,
    
    # LoRA configuration
    'lora_rank': 16,
    'lora_alpha': 32.0,
    
    # Distillation configuration
    'distillation_temperature': 4.0,
    'distillation_alpha': 0.7
}
