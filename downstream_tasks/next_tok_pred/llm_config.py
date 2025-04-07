import torch

llm_config = {
    'vocab_size': 50257,  # GPT-2 vocabulary size
    'max_seq_length': 1024,
    'hidden_size': 768,
    'num_experts': 8,
    'num_heads': 12,
    'kv_cache_size': 512,
    'batch_size': 32,
    'learning_rate': 1e-4,
    'epochs': 10,
    'warmup_steps': 1000,
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'pyramidal_cache': True,
    'cache_decrement': 32,
    'num_layers': 12,
    'dropout': 0.1,
    'expert_capacity': 1.0,  # Capacity factor for load balancing
    'use_mixed_precision': True,
    'gradient_accumulation_steps': 4,
    'eval_steps': 100,
    'save_steps': 1000,
    'max_grad_norm': 1.0,
    'weight_decay': 0.01,
    'lr_schedule': 'cosine',  # 'linear' or 'cosine'
    'use_flash_attention': True,  # Use Flash Attention if available
} 