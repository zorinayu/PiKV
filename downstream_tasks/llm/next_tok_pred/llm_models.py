import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer
from llm_config import llm_config as config
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class PiKVLLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embedding = nn.Embedding(config['vocab_size'], config['hidden_size'])
        self.position_encoding = PositionalEncoding(config['hidden_size'])
        
        # Transformer layers with PiKV
        self.layers = nn.ModuleList([
            PiKVTransformerLayer(
                hidden_size=config['hidden_size'],
                num_heads=config['num_heads'],
                num_experts=config['num_experts'],
                dropout=config['dropout'],
                layer_id=i
            ) for i in range(config['num_layers'])
        ])
        
        # Output layer
        self.ln_f = nn.LayerNorm(config['hidden_size'])
        self.head = nn.Linear(config['hidden_size'], config['vocab_size'], bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(self, input_ids, attention_mask=None):
        B, T = input_ids.shape
        
        # Get embeddings
        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_encoding(token_embeddings)
        x = position_embeddings
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, attention_mask)
            
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits

class PiKVTransformerLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, num_experts, dropout, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        
        # Self-attention
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout)
        self.ln_1 = nn.LayerNorm(hidden_size)
        
        # PiKV MoE
        self.moe = PiKVMoE(
            hidden_size=hidden_size,
            num_experts=num_experts,
            dropout=dropout,
            layer_id=layer_id
        )
        self.ln_2 = nn.LayerNorm(hidden_size)
        
    def forward(self, x, attention_mask=None):
        # Self-attention
        attn_output, _ = self.attention(x, x, x, attn_mask=attention_mask)
        x = self.ln_1(x + attn_output)
        
        # PiKV MoE
        moe_output = self.moe(x)
        x = self.ln_2(x + moe_output)
        
        return x

class PiKVMoE(nn.Module):
    def __init__(self, hidden_size, num_experts, dropout, layer_id):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.layer_id = layer_id
        
        # Calculate cache size for this layer using pyramidal allocation
        base_cache_size = config['kv_cache_size']
        cache_decrement = config['cache_decrement']
        self.cache_size = base_cache_size - (layer_id * cache_decrement)
        
        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size)
            ) for _ in range(num_experts)
        ])
        
        # Router
        self.router = nn.Linear(hidden_size, num_experts)
        
        # KV Cache
        self.register_buffer('kv_cache', torch.zeros(
            num_experts, self.cache_size, hidden_size
        ))
        self.register_buffer('cache_ptr', torch.zeros(1, dtype=torch.long))
        
    def forward(self, x):
        B, T, C = x.shape
        
        # Get routing probabilities
        router_logits = self.router(x)  # [B, T, num_experts]
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Get top-k experts
        top_k = 2
        top_k_probs, top_k_indices = torch.topk(router_probs, top_k, dim=-1)
        
        # Initialize output tensor
        output = torch.zeros_like(x)
        
        # Process each expert
        for i in range(self.num_experts):
            # Find tokens routed to this expert
            mask = (top_k_indices == i).any(dim=-1)  # [B, T]
            
            if mask.any():
                # Get tokens for this expert
                expert_tokens = x[mask]  # [num_tokens, C]
                
                # Process through expert
                expert_output = self.experts[i](expert_tokens)
                
                # Update KV cache
                cache_idx = self.cache_ptr % self.cache_size
                self.kv_cache[i, cache_idx] = expert_output.mean(dim=0)
                self.cache_ptr += 1
                
                # Combine with cached values
                cached_values = self.kv_cache[i].mean(dim=0)  # [C]
                expert_output = expert_output + cached_values
                
                # Place back in output tensor
                output[mask] = expert_output
        
        return output

class StandardMoELLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embedding = nn.Embedding(config['vocab_size'], config['hidden_size'])
        self.position_encoding = PositionalEncoding(config['hidden_size'])
        
        # Transformer layers with Standard MoE
        self.layers = nn.ModuleList([
            StandardMoELayer(
                hidden_size=config['hidden_size'],
                num_heads=config['num_heads'],
                num_experts=config['num_experts'],
                dropout=config['dropout']
            ) for _ in range(config['num_layers'])
        ])
        
        # Output layer
        self.ln_f = nn.LayerNorm(config['hidden_size'])
        self.head = nn.Linear(config['hidden_size'], config['vocab_size'], bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(self, input_ids, attention_mask=None):
        B, T = input_ids.shape
        
        # Get embeddings
        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_encoding(token_embeddings)
        x = position_embeddings
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, attention_mask)
            
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits

class StandardMoELayer(nn.Module):
    def __init__(self, hidden_size, num_heads, num_experts, dropout):
        super().__init__()
        
        # Self-attention
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout)
        self.ln_1 = nn.LayerNorm(hidden_size)
        
        # Standard MoE
        self.moe = StandardMoE(
            hidden_size=hidden_size,
            num_experts=num_experts,
            dropout=dropout
        )
        self.ln_2 = nn.LayerNorm(hidden_size)
        
    def forward(self, x, attention_mask=None):
        # Self-attention
        attn_output, _ = self.attention(x, x, x, attn_mask=attention_mask)
        x = self.ln_1(x + attn_output)
        
        # Standard MoE
        moe_output = self.moe(x)
        x = self.ln_2(x + moe_output)
        
        return x

class StandardMoE(nn.Module):
    def __init__(self, hidden_size, num_experts, dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        
        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size)
            ) for _ in range(num_experts)
        ])
        
        # Router
        self.router = nn.Linear(hidden_size, num_experts)
        
    def forward(self, x):
        B, T, C = x.shape
        
        # Get routing probabilities
        router_logits = self.router(x)  # [B, T, num_experts]
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Get top-k experts
        top_k = 2
        top_k_probs, top_k_indices = torch.topk(router_probs, top_k, dim=-1)
        
        # Initialize output tensor
        output = torch.zeros_like(x)
        
        # Process each expert
        for i in range(self.num_experts):
            # Find tokens routed to this expert
            mask = (top_k_indices == i).any(dim=-1)  # [B, T]
            
            if mask.any():
                # Get tokens for this expert
                expert_tokens = x[mask]  # [num_tokens, C]
                
                # Process through expert
                expert_output = self.experts[i](expert_tokens)
                
                # Place back in output tensor
                output[mask] = expert_output
        
        return output 