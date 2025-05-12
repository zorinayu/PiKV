import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Dict, List, Optional, Union, Any

class PiKVAttention(nn.Module):
    """
    PiKV 注意力机制
    集成了KV缓存并支持流式/高效处理
    """
    def __init__(
        self, 
        hidden_size: int, 
        num_heads: int, 
        head_dim: Optional[int] = None,
        dropout: float = 0.1,
        use_rotary: bool = False,
        max_position_embeddings: int = 2048,
        cache_size: int = 4096
    ):
        super(PiKVAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_heads
        self.dropout = dropout
        self.use_rotary = use_rotary
        self.cache_size = cache_size
        
        # 投影矩阵
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim)
        self.k_proj = nn.Linear(hidden_size, num_heads * self.head_dim)
        self.v_proj = nn.Linear(hidden_size, num_heads * self.head_dim)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        
        # 初始化KV缓存
        # 注意：这些缓存在前向传播中将被流式更新
        self.register_buffer(
            "k_cache", 
            torch.zeros((self.cache_size, self.num_heads, self.head_dim))
        )
        self.register_buffer(
            "v_cache", 
            torch.zeros((self.cache_size, self.num_heads, self.head_dim))
        )
        
        # 缓存指针（当前填充到的位置）
        self.register_buffer("cache_ptr", torch.zeros(1, dtype=torch.long))
        
        # 创建相对位置嵌入（如果使用）
        if use_rotary:
            self.rotary_emb = RotaryEmbedding(self.head_dim, max_position_embeddings)
        
        # 初始化权重
        self._init_weights()
        
        # 记录注意力统计信息
        self.register_buffer("max_attention_score", torch.zeros(1))
        self.register_buffer("min_attention_score", torch.zeros(1))
        self.register_buffer("avg_attention_score", torch.zeros(1))
        self.register_buffer("attention_count", torch.zeros(1))
    
    def _init_weights(self):
        """初始化权重"""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            # 使用截断正态分布进行初始化
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def _update_cache(self, k: torch.Tensor, v: torch.Tensor):
        """更新KV缓存"""
        # k和v的形状: [batch_size, seq_len, num_heads, head_dim]
        
        # 获取缓存更新的数量和偏移量
        batch_size, seq_len = k.size(0), k.size(1)
        update_len = batch_size * seq_len
        ptr = int(self.cache_ptr.item())
        
        # 确保缓存不会溢出
        if ptr + update_len > self.cache_size:
            # 如果缓存空间不足，删除最旧的条目并从头开始
            print(f"Cache overflow: resetting KV cache (ptr={ptr}, update_len={update_len})")
            self.k_cache.zero_()
            self.v_cache.zero_()
            ptr = 0
        
        # 将数据平展为[batch_size*seq_len, num_heads, head_dim]
        k_flat = k.reshape(-1, self.num_heads, self.head_dim)
        v_flat = v.reshape(-1, self.num_heads, self.head_dim)
        
        # 更新缓存
        self.k_cache[ptr:ptr+update_len] = k_flat
        self.v_cache[ptr:ptr+update_len] = v_flat
        
        # 更新指针
        self.cache_ptr[0] = ptr + update_len
        
        return ptr, update_len
    
    def _compute_attention(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        causal_mask: bool = False,
        stream_mode: bool = False,
        stream_position: Optional[int] = None
    ) -> torch.Tensor:
        """计算注意力加权和"""
        # q, k, v形状: [batch_size, seq_len, num_heads, head_dim]
        
        # 转置为注意力形状 [batch_size, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 缩放点积注意力
        attention_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        
        # 应用注意力掩码
        if attention_mask is not None:
            # 注意力掩码: [batch_size, seq_len] 或 [batch_size, 1, seq_len, seq_len]
            if attention_mask.dim() != 4:
                # 将 [batch_size, seq_len] 扩展为 [batch_size, 1, 1, seq_len]
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            
            # 将掩码中的0转换为-infinity，1保持不变
            attention_scores = attention_scores.masked_fill(
                attention_mask == 0, 
                float('-inf')
            )
        
        # 应用因果掩码（仅关注自身和之前的位置）
        if causal_mask:
            seq_len = attention_scores.size(-1)
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=q.device), 
                diagonal=1
            ).bool()
            attention_scores = attention_scores.masked_fill(
                causal_mask.unsqueeze(0).unsqueeze(0), 
                float('-inf')
            )
        
        # 流式处理的特殊处理
        if stream_mode and stream_position is not None:
            # 在流式模式下，只需要计算当前位置的输出
            # 将除了当前位置外的所有查询屏蔽掉
            batch_size, num_heads, seq_len, _ = q.size()
            stream_mask = torch.ones(seq_len, device=q.device).bool()
            stream_mask[stream_position] = False
            attention_scores = attention_scores.masked_fill(
                stream_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1), 
                float('-inf')
            )
        
        # 应用softmax获取注意力权重
        attention_probs = F.softmax(attention_scores, dim=-1)
        
        # 记录注意力统计信息
        with torch.no_grad():
            self.max_attention_score = torch.max(
                self.max_attention_score, 
                attention_probs.max()
            )
            if self.min_attention_score.item() == 0:
                self.min_attention_score = attention_probs.min()
            else:
                self.min_attention_score = torch.min(
                    self.min_attention_score, 
                    attention_probs.min()
                )
            # 更新平均注意力分数
            total = self.avg_attention_score * self.attention_count
            new_count = self.attention_count + 1
            self.avg_attention_score = (total + attention_probs.mean()) / new_count
            self.attention_count = new_count
        
        # 应用dropout
        attention_probs = self.attn_dropout(attention_probs)
        
        # 计算加权和
        context = torch.matmul(attention_probs, v)
        
        # 转置回原始形状 [batch_size, seq_len, num_heads, head_dim]
        context = context.transpose(1, 2)
        
        return context
    
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_ids: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        causal_mask: bool = True,
        stream_mode: bool = False,
        stream_position: Optional[int] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        前向传播
        
        Args:
            hidden_states: 输入张量 [batch_size, seq_len, hidden_size]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            past_key_values: 可选的过去KV值 (k, v)
            position_ids: 可选的位置ID [batch_size, seq_len]
            use_cache: 是否返回KV缓存
            causal_mask: 是否应用因果掩码（生成时为真）
            stream_mode: 是否处于流式处理模式
            stream_position: 在流式模式下的处理位置
            
        Returns:
            output: 注意力输出 [batch_size, seq_len, hidden_size]
            present_key_values: 当use_cache=True时，返回更新后的KV缓存
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # 使用投影矩阵计算Q、K、V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # 重塑为多头形式 [batch_size, seq_len, num_heads, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # 应用旋转位置嵌入（如果使用）
        if self.use_rotary and position_ids is not None:
            q, k = self.rotary_emb(q, k, position_ids)
        
        # 整合过去的KV值（自动递增生成）
        if past_key_values is not None:
            past_k, past_v = past_key_values
            # 拼接过去和当前的KV
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)
        
        # 更新KV缓存
        cache_ptr, cache_len = self._update_cache(k, v)
        
        # 计算注意力
        context = self._compute_attention(
            q, k, v, 
            attention_mask=attention_mask,
            causal_mask=causal_mask,
            stream_mode=stream_mode,
            stream_position=stream_position
        )
        
        # 重塑回原始维度 [batch_size, seq_len, hidden_size]
        context = context.reshape(batch_size, seq_len, self.num_heads * self.head_dim)
        
        # 应用输出投影
        output = self.o_proj(context)
        
        # 如果需要缓存，提取当前批次的KV值
        if use_cache:
            present_k = self.k_cache[cache_ptr:cache_ptr+cache_len].reshape(
                batch_size, seq_len, self.num_heads, self.head_dim
            )
            present_v = self.v_cache[cache_ptr:cache_ptr+cache_len].reshape(
                batch_size, seq_len, self.num_heads, self.head_dim
            )
            present_key_values = (present_k, present_v)
        else:
            present_key_values = None
        
        return output, present_key_values
    
    def get_attention_stats(self) -> Dict[str, float]:
        """获取注意力统计信息"""
        return {
            "max_score": self.max_attention_score.item(),
            "min_score": self.min_attention_score.item(),
            "avg_score": self.avg_attention_score.item(),
            "attention_count": self.attention_count.item()
        }

class RotaryEmbedding(nn.Module):
    """
    旋转位置嵌入 (RoPE)
    提供更高效的相对位置编码
    """
    def __init__(self, dim: int, max_position_embeddings: int = 2048):
        super(RotaryEmbedding, self).__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        
        # 缓存预计算的位置嵌入值
        self.register_buffer(
            "cos_cached", 
            self._compute_cos_sin_cache(max_position_embeddings, dim)[0],
            persistent=False
        )
        self.register_buffer(
            "sin_cached", 
            self._compute_cos_sin_cache(max_position_embeddings, dim)[1],
            persistent=False
        )
    
    def _compute_cos_sin_cache(self, seq_len: int, dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算余弦和正弦缓存"""
        # 位置索引
        position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
        
        # 维度索引
        dim_idx = torch.arange(0, dim, step=2, dtype=torch.float)
        
        # 计算频率
        freq = 1.0 / (10000 ** (dim_idx / dim))
        
        # 计算位置*频率
        pos_freq = position * freq
        
        # 计算余弦和正弦值
        cos = torch.cos(pos_freq)
        sin = torch.sin(pos_freq)
        
        return cos, sin
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        应用旋转位置嵌入
        
        Args:
            q: 查询张量 [batch_size, seq_len, num_heads, head_dim]
            k: 键张量 [batch_size, seq_len, num_heads, head_dim]
            position_ids: 位置ID [batch_size, seq_len]
            
        Returns:
            q_rope: 应用RoPE后的查询
            k_rope: 应用RoPE后的键
        """
        batch_size, seq_len, num_heads, head_dim = q.shape
        
        # 检查维度是偶数
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE需要偶数维度，但得到的是{head_dim}")
        
        # 最大位置ID不应超过缓存长度
        max_pos = position_ids.max().item()
        if max_pos >= self.max_position_embeddings:
            # 如果超出缓存范围，动态扩展缓存
            self.max_position_embeddings = max(self.max_position_embeddings * 2, max_pos + 1)
            cos, sin = self._compute_cos_sin_cache(self.max_position_embeddings, self.dim)
            self.register_buffer("cos_cached", cos, persistent=False)
            self.register_buffer("sin_cached", sin, persistent=False)
        
        # 从缓存中提取当前位置ID对应的余弦和正弦值
        cos = self.cos_cached[position_ids]  # [batch_size, seq_len, dim/2]
        sin = self.sin_cached[position_ids]  # [batch_size, seq_len, dim/2]
        
        # 重塑以便广播
        cos = cos.unsqueeze(2).expand(-1, -1, num_heads, -1)  # [batch_size, seq_len, num_heads, dim/2]
        sin = sin.unsqueeze(2).expand(-1, -1, num_heads, -1)  # [batch_size, seq_len, num_heads, dim/2]
        
        # 拆分顶部一半和底部一半的维度
        q_top, q_bottom = q[..., :head_dim//2], q[..., head_dim//2:]
        k_top, k_bottom = k[..., :head_dim//2], k[..., head_dim//2:]
        
        # 应用旋转
        q_rotate_top = -q_bottom
        q_rotate_bottom = q_top
        k_rotate_top = -k_bottom
        k_rotate_bottom = k_top
        
        # 应用RoPE变换
        q_out_top = q_top * cos + q_rotate_top * sin
        q_out_bottom = q_bottom * cos + q_rotate_bottom * sin
        k_out_top = k_top * cos + k_rotate_top * sin
        k_out_bottom = k_bottom * cos + k_rotate_bottom * sin
        
        # 拼接回完整维度
        q_out = torch.cat([q_out_top, q_out_bottom], dim=-1)
        k_out = torch.cat([k_out_top, k_out_bottom], dim=-1)
        
        return q_out, k_out 