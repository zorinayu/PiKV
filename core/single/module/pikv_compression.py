"""
PiKV Compression Module

实现多种KV缓存压缩策略，包括：
- PyramidCompressor: 层次化金字塔压缩
- SVDCompressor: 基于SVD的低秩压缩
- QuantizedCompressor: 量化压缩
- LoRACompressor: LoRA低秩分解压缩
- LoRaPlusPlusCompressor: LoRA++增强压缩
- PruningCompressor: 剪枝压缩
- DistillationCompressor: 知识蒸馏压缩
- FastVCompressor: FastV高效压缩
- PyramidKVCompressor: PyramidKV层次化压缩
- PiKVCompressor: 自适应综合压缩器
- ChunkKVCompressor: 分块KV缓存压缩
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Dict, List, Optional, Union, Any
from enum import Enum

class CompressionMethod(Enum):
    """Compression method enumeration"""
    PYRAMID = "pyramid"
    SVD = "svd"
    QUANTIZED = "quantized"
    LORA = "lora"
    LORA_PLUS_PLUS = "lora++"
    PRUNING = "pruning"
    DISTILLATION = "distillation"
    FASTV = "fastv"
    PYRAMID_KV = "pyramid_kv"
    CHUNK_KV = "chunk_kv"

class BaseCompressor(nn.Module):
    """
    基础KV缓存压缩器
    提供KV缓存压缩的基本结构和接口
    """
    def __init__(self, hidden_size: int, compression_ratio: float = 0.5):
        super(BaseCompressor, self).__init__()
        self.hidden_size = hidden_size
        self.compression_ratio = compression_ratio
        
        # 压缩统计信息
        self.register_buffer('total_original_size', torch.tensor(0.0))
        self.register_buffer('total_compressed_size', torch.tensor(0.0))
        self.register_buffer('compression_count', torch.tensor(0.0))
        
    def forward(
        self, 
        keys: torch.Tensor, 
        values: torch.Tensor, 
        importance: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        压缩KV缓存
        
        Args:
            keys: 缓存键张量 [batch_size, seq_len, hidden_size]
            values: 缓存值张量 [batch_size, seq_len, hidden_size]
            importance: 可选的重要性分数 [batch_size, seq_len]
            
        Returns:
            compressed_keys: 压缩后的键张量
            compressed_values: 压缩后的值张量
        """
        # 基类不实现具体压缩算法，只返回原始值
        self._update_stats(keys.numel(), keys.numel())
        return keys, values
    
    def get_compression_stats(self) -> Dict[str, float]:
        """
        获取压缩统计信息
        
        Returns:
            stats: 字典包含压缩率、内存减少等信息
        """
        if self.total_original_size.item() > 0:
            actual_ratio = self.total_compressed_size.item() / self.total_original_size.item()
        else:
            actual_ratio = 1.0
            
        return {
            "target_compression_ratio": self.compression_ratio,
            "actual_compression_ratio": actual_ratio,
            "memory_reduction": 1.0 - actual_ratio,
            "compression_count": self.compression_count.item()
        }
    
    def _update_stats(self, original_size: int, compressed_size: int):
        """更新压缩统计信息"""
        self.total_original_size += original_size
        self.total_compressed_size += compressed_size
        self.compression_count += 1
    
    def reset_stats(self):
        """重置压缩统计信息"""
        self.total_original_size.zero_()
        self.total_compressed_size.zero_()
        self.compression_count.zero_()

class PyramidCompressor(BaseCompressor):
    """
    金字塔压缩器
    实现层次化的压缩方案，在不同层级使用不同的压缩比例
    参考PyramidKV论文的分层压缩策略
    """
    def __init__(
        self, 
        hidden_size: int, 
        compression_ratio: float = 0.5,
        num_levels: int = 3,
        decay_factor: float = 0.8,
        use_adaptive_levels: bool = True
    ):
        super(PyramidCompressor, self).__init__(hidden_size, compression_ratio)
        self.num_levels = num_levels
        self.decay_factor = decay_factor
        self.use_adaptive_levels = use_adaptive_levels
        
        # 构建金字塔压缩层
        self.compression_layers = nn.ModuleList()
        
        current_size = hidden_size
        for i in range(num_levels):
            # 计算当前层级的压缩输出大小
            output_size = max(int(current_size * compression_ratio), 1)
            
            # 创建压缩层（编码器-解码器对）
            layer = nn.ModuleDict({
                'encoder': nn.Sequential(
                    nn.Linear(current_size, output_size),
                    nn.LayerNorm(output_size),
                    nn.ReLU()
                ),
                'decoder': nn.Sequential(
                    nn.Linear(output_size, current_size),
                    nn.LayerNorm(current_size)
                )
            })
            self.compression_layers.append(layer)
            
            # 更新当前大小用于下一层
            current_size = output_size
            # 调整下一层的压缩率
            compression_ratio *= decay_factor
        
        # 重要性阈值用于自适应分层
        self.register_buffer('importance_thresholds', 
                           torch.linspace(0.8, 0.2, num_levels))
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化权重"""
        for layer in self.compression_layers:
            for module in layer.values():
                for m in module:
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_normal_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
    
    def _assign_compression_levels(self, importance: torch.Tensor) -> torch.Tensor:
        """根据重要性分配压缩级别"""
        batch_size, seq_len = importance.shape
        levels = torch.zeros_like(importance, dtype=torch.long)
        
        # 根据重要性阈值分配级别
        for level in range(self.num_levels):
            if level == self.num_levels - 1:
                # 最后一级包含所有剩余的token
                mask = levels == 0
            else:
                # 当前级别的阈值
                threshold = self.importance_thresholds[level]
                mask = (importance >= threshold) & (levels == 0)
            
            levels[mask] = level
        
        return levels
    
    def _apply_pyramid_compression(
        self, 
        x: torch.Tensor, 
        levels: torch.Tensor
    ) -> torch.Tensor:
        """应用金字塔压缩"""
        batch_size, seq_len, hidden_size = x.shape
        compressed = torch.zeros_like(x)
        
        for level in range(self.num_levels):
            # 找到属于当前级别的token
            level_mask = (levels == level)
            if not level_mask.any():
                continue
            
            # 提取当前级别的token
            level_tokens = x[level_mask]  # [num_tokens, hidden_size]
            
            if level_tokens.size(0) == 0:
                continue
            
            # 应用压缩（从第0层到第level层）
            compressed_tokens = level_tokens
            for i in range(level + 1):
                if i < len(self.compression_layers):
                    # 编码
                    compressed_tokens = self.compression_layers[i]['encoder'](compressed_tokens)
            
            # 应用解压（从第level层到第0层）
            for i in range(level, -1, -1):
                if i < len(self.compression_layers):
                    # 解码
                    compressed_tokens = self.compression_layers[i]['decoder'](compressed_tokens)
            
            # 添加残差连接
            compressed_tokens = level_tokens + compressed_tokens
            
            # 将压缩后的token放回原位置
            compressed[level_mask] = compressed_tokens
        
        return compressed
    
    def forward(
        self, 
        keys: torch.Tensor, 
        values: torch.Tensor, 
        importance: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_size = keys.shape
        
        # 如果没有提供重要性分数，使用均匀分布
        if importance is None:
            importance = torch.ones(batch_size, seq_len, device=keys.device) * 0.5
        
        # 分配压缩级别
        key_levels = self._assign_compression_levels(importance)
        value_levels = key_levels  # 使用相同的级别分配
        
        # 应用金字塔压缩
        compressed_keys = self._apply_pyramid_compression(keys, key_levels)
        compressed_values = self._apply_pyramid_compression(values, value_levels)
        
        # 更新统计信息
        original_size = keys.numel() + values.numel()
        # 估算压缩大小（基于级别分布）
        avg_compression = sum(
            (self.decay_factor ** level) * (key_levels == level).float().mean().item()
            for level in range(self.num_levels)
        )
        compressed_size = int(original_size * avg_compression)
        self._update_stats(original_size, compressed_size)
        
        return compressed_keys, compressed_values

class LoRACompressor(BaseCompressor):
    """
    LoRA低秩分解压缩器
    使用低秩适应进行KV缓存压缩
    """
    def __init__(
        self, 
        hidden_size: int, 
        rank: int = 64,
        alpha: float = 1.0,
        dropout: float = 0.1,
        init_lora_weights: bool = True
    ):
        super(LoRACompressor, self).__init__(hidden_size, rank/hidden_size)
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA matrices for keys
        self.key_lora_A = nn.Parameter(torch.zeros(hidden_size, rank))
        self.key_lora_B = nn.Parameter(torch.zeros(rank, hidden_size))
        
        # LoRA matrices for values
        self.value_lora_A = nn.Parameter(torch.zeros(hidden_size, rank))
        self.value_lora_B = nn.Parameter(torch.zeros(rank, hidden_size))
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        if init_lora_weights:
            self._init_lora_weights()
    
    def _init_lora_weights(self):
        """Initialize LoRA weights"""
        nn.init.kaiming_uniform_(self.key_lora_A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.value_lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.key_lora_B)
        nn.init.zeros_(self.value_lora_B)
    
    def forward(
        self, 
        keys: torch.Tensor, 
        values: torch.Tensor, 
        importance: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Apply LoRA compression to keys
        key_adaptation = self.dropout(keys @ self.key_lora_A @ self.key_lora_B) * self.scaling
        compressed_keys = keys + key_adaptation
        
        # Apply LoRA compression to values
        value_adaptation = self.dropout(values @ self.value_lora_A @ self.value_lora_B) * self.scaling
        compressed_values = values + value_adaptation
        
        # Update statistics
        original_size = keys.numel() + values.numel()
        compressed_size = int(original_size * self.rank / self.hidden_size)
        self._update_stats(original_size, compressed_size)
        
        return compressed_keys, compressed_values

class LoRaPlusPlusCompressor(LoRACompressor):
    """
    LoRA++ 增强压缩器
    使用更先进的低秩分解技术，包括RSLoRA和DoRA
    """
    def __init__(
        self, 
        hidden_size: int, 
        rank: int = 64,
        alpha: float = 1.0,
        dropout: float = 0.1,
        use_rslora: bool = True,
        use_dora: bool = True,
        magnitude_vector_dim: Optional[int] = None
    ):
        super(LoRaPlusPlusCompressor, self).__init__(hidden_size, rank, alpha, dropout)
        self.use_rslora = use_rslora
        self.use_dora = use_dora
        
        if use_rslora:
            # RSLoRA scaling
            self.scaling = alpha / math.sqrt(rank)
        
        if use_dora:
            # DoRA magnitude vectors
            mag_dim = magnitude_vector_dim or hidden_size
            self.key_magnitude = nn.Parameter(torch.ones(mag_dim))
            self.value_magnitude = nn.Parameter(torch.ones(mag_dim))
    
    def forward(
        self, 
        keys: torch.Tensor, 
        values: torch.Tensor, 
        importance: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Standard LoRA adaptation
        key_lora = self.dropout(keys @ self.key_lora_A @ self.key_lora_B) * self.scaling
        value_lora = self.dropout(values @ self.value_lora_A @ self.value_lora_B) * self.scaling
        
        if self.use_dora:
            # Apply DoRA magnitude scaling
            key_norm = torch.norm(keys + key_lora, dim=-1, keepdim=True)
            value_norm = torch.norm(values + value_lora, dim=-1, keepdim=True)
            
            compressed_keys = (keys + key_lora) / key_norm * self.key_magnitude.unsqueeze(0).unsqueeze(0)
            compressed_values = (values + value_lora) / value_norm * self.value_magnitude.unsqueeze(0).unsqueeze(0)
        else:
            compressed_keys = keys + key_lora
            compressed_values = values + value_lora
        
        # Update statistics
        original_size = keys.numel() + values.numel()
        compressed_size = int(original_size * self.rank / self.hidden_size)
        self._update_stats(original_size, compressed_size)
        
        return compressed_keys, compressed_values

class PruningCompressor(BaseCompressor):
    """
    剪枝压缩器
    基于重要性的结构化剪枝
    """
    def __init__(
        self, 
        hidden_size: int, 
        pruning_ratio: float = 0.5,
        structured: bool = True,
        importance_metric: str = "magnitude"
    ):
        super(PruningCompressor, self).__init__(hidden_size, pruning_ratio)
        self.pruning_ratio = pruning_ratio
        self.structured = structured
        self.importance_metric = importance_metric
        
        # Learnable pruning masks
        self.key_mask = nn.Parameter(torch.ones(hidden_size))
        self.value_mask = nn.Parameter(torch.ones(hidden_size))
    
    def _compute_importance(self, x: torch.Tensor) -> torch.Tensor:
        """Compute importance scores for pruning"""
        if self.importance_metric == "magnitude":
            return torch.abs(x).mean(dim=(0, 1))
        elif self.importance_metric == "variance":
            return torch.var(x, dim=(0, 1))
        elif self.importance_metric == "gradient":
            if x.requires_grad and x.grad is not None:
                return torch.abs(x.grad).mean(dim=(0, 1))
            else:
                return torch.abs(x).mean(dim=(0, 1))
        else:
            return torch.abs(x).mean(dim=(0, 1))
    
    def _generate_pruning_mask(self, importance: torch.Tensor) -> torch.Tensor:
        """Generate binary pruning mask based on importance"""
        sorted_importance, _ = torch.sort(importance, descending=True)
        keep_count = int(len(importance) * (1 - self.pruning_ratio))
        if keep_count > 0:
            threshold = sorted_importance[keep_count - 1]
        else:
            threshold = sorted_importance[-1]
        
        mask = (importance >= threshold).float()
        return mask
    
    def forward(
        self, 
        keys: torch.Tensor, 
        values: torch.Tensor, 
        importance: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Compute importance if not provided
        if importance is None:
            key_importance = self._compute_importance(keys)
            value_importance = self._compute_importance(values)
        else:
            if importance.dim() == 2:
                key_importance = importance.mean(dim=(0, 1)).expand(self.hidden_size)
                value_importance = key_importance
            else:
                key_importance = importance
                value_importance = importance
        
        # Generate pruning masks
        key_pruning_mask = self._generate_pruning_mask(key_importance)
        value_pruning_mask = self._generate_pruning_mask(value_importance)
        
        # Apply masks with learnable parameters
        final_key_mask = torch.sigmoid(self.key_mask) * key_pruning_mask
        final_value_mask = torch.sigmoid(self.value_mask) * value_pruning_mask
        
        # Apply pruning
        compressed_keys = keys * final_key_mask.unsqueeze(0).unsqueeze(0)
        compressed_values = values * final_value_mask.unsqueeze(0).unsqueeze(0)
        
        # Update statistics
        original_size = keys.numel() + values.numel()
        remaining_ratio = (final_key_mask.sum() + final_value_mask.sum()) / (2 * self.hidden_size)
        compressed_size = int(original_size * remaining_ratio.item())
        self._update_stats(original_size, compressed_size)
        
        return compressed_keys, compressed_values

class DistillationCompressor(BaseCompressor):
    """
    知识蒸馏压缩器
    使用教师-学生网络进行压缩
    """
    def __init__(
        self, 
        hidden_size: int, 
        teacher_hidden_size: Optional[int] = None,
        student_hidden_size: Optional[int] = None,
        temperature: float = 3.0,
        alpha: float = 0.7
    ):
        teacher_hidden_size = teacher_hidden_size or hidden_size * 2
        student_hidden_size = student_hidden_size or hidden_size // 2
        
        super(DistillationCompressor, self).__init__(hidden_size, student_hidden_size/hidden_size)
        self.teacher_hidden_size = teacher_hidden_size
        self.student_hidden_size = student_hidden_size
        self.temperature = temperature
        self.alpha = alpha
        
        # Teacher network (larger capacity)
        self.teacher_key_proj = nn.Linear(hidden_size, teacher_hidden_size)
        self.teacher_value_proj = nn.Linear(hidden_size, teacher_hidden_size)
        self.teacher_key_output = nn.Linear(teacher_hidden_size, hidden_size)
        self.teacher_value_output = nn.Linear(teacher_hidden_size, hidden_size)
        
        # Student network (smaller capacity)
        self.student_key_proj = nn.Linear(hidden_size, student_hidden_size)
        self.student_value_proj = nn.Linear(hidden_size, student_hidden_size)
        self.student_key_output = nn.Linear(student_hidden_size, hidden_size)
        self.student_value_output = nn.Linear(student_hidden_size, hidden_size)
        
        # Distillation projectors
        self.key_distill_proj = nn.Linear(teacher_hidden_size, student_hidden_size)
        self.value_distill_proj = nn.Linear(teacher_hidden_size, student_hidden_size)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self, 
        keys: torch.Tensor, 
        values: torch.Tensor, 
        importance: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Teacher forward pass
        teacher_key_hidden = F.relu(self.teacher_key_proj(keys))
        teacher_value_hidden = F.relu(self.teacher_value_proj(values))
        teacher_key_output = self.teacher_key_output(teacher_key_hidden)
        teacher_value_output = self.teacher_value_output(teacher_value_hidden)
        
        # Student forward pass
        student_key_hidden = F.relu(self.student_key_proj(keys))
        student_value_hidden = F.relu(self.student_value_proj(values))
        student_key_output = self.student_key_output(student_key_hidden)
        student_value_output = self.student_value_output(student_value_hidden)
        
        # Knowledge distillation
        if self.training:
            teacher_key_projected = self.key_distill_proj(teacher_key_hidden)
            teacher_value_projected = self.value_distill_proj(teacher_value_hidden)
            
            self.distillation_loss = (
                F.mse_loss(student_key_hidden, teacher_key_projected.detach()) +
                F.mse_loss(student_value_hidden, teacher_value_projected.detach())
            )
        
        # Combine outputs
        if self.training:
            compressed_keys = self.alpha * teacher_key_output + (1 - self.alpha) * student_key_output
            compressed_values = self.alpha * teacher_value_output + (1 - self.alpha) * student_value_output
        else:
            compressed_keys = student_key_output
            compressed_values = student_value_output
        
        # Update statistics
        original_size = keys.numel() + values.numel()
        compressed_size = int(original_size * self.student_hidden_size / self.hidden_size)
        self._update_stats(original_size, compressed_size)
        
        return compressed_keys, compressed_values
    
    def get_distillation_loss(self) -> torch.Tensor:
        """Get the distillation loss for training"""
        return getattr(self, 'distillation_loss', torch.tensor(0.0))

class FastVCompressor(BaseCompressor):
    """
    FastV 压缩器
    基于FastV算法的高效KV缓存压缩
    参考: https://github.com/pkunlp-icler/FastV
    """
    def __init__(
        self, 
        hidden_size: int, 
        compression_ratio: float = 0.5,
        window_size: int = 64,
        sink_tokens: int = 4,
        recent_tokens: Optional[int] = None
    ):
        super(FastVCompressor, self).__init__(hidden_size, compression_ratio)
        self.window_size = window_size
        self.sink_tokens = sink_tokens
        self.recent_tokens = recent_tokens or int(window_size * 0.3)
        
        # Attention score predictor
        self.attention_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # Token importance estimator
        self.importance_estimator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _compute_token_scores(self, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """Compute importance scores for each token"""
        attention_scores = self.attention_predictor(keys).squeeze(-1)
        importance_scores = self.importance_estimator(values).squeeze(-1)
        combined_scores = 0.6 * attention_scores + 0.4 * importance_scores
        return combined_scores
    
    def _select_tokens_fastv(self, scores: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Select tokens using FastV strategy"""
        batch_size = scores.size(0)
        keep_count = max(int(seq_len * (1 - self.compression_ratio)), self.sink_tokens + self.recent_tokens)
        
        selection_mask = torch.zeros_like(scores, dtype=torch.bool)
        
        for b in range(batch_size):
            # Always keep sink tokens
            selection_mask[b, :self.sink_tokens] = True
            
            # Always keep recent tokens
            selection_mask[b, -self.recent_tokens:] = True
            
            # Select important tokens from middle
            middle_start = self.sink_tokens
            middle_end = seq_len - self.recent_tokens
            
            if middle_end > middle_start:
                middle_scores = scores[b, middle_start:middle_end]
                middle_keep = keep_count - self.sink_tokens - self.recent_tokens
                
                if middle_keep > 0:
                    _, middle_indices = torch.topk(middle_scores, min(middle_keep, len(middle_scores)))
                    global_indices = middle_indices + middle_start
                    selection_mask[b, global_indices] = True
        
        return selection_mask
    
    def forward(
        self, 
        keys: torch.Tensor, 
        values: torch.Tensor, 
        importance: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_size = keys.shape
        
        # Compute token importance scores
        if importance is not None:
            token_scores = importance
        else:
            token_scores = self._compute_token_scores(keys, values)
        
        # Select tokens using FastV strategy
        selection_mask = self._select_tokens_fastv(token_scores, seq_len)
        
        # Apply selection mask
        compressed_keys = keys * selection_mask.unsqueeze(-1).float()
        compressed_values = values * selection_mask.unsqueeze(-1).float()
        
        # Update statistics
        original_size = keys.numel() + values.numel()
        kept_ratio = selection_mask.float().mean().item()
        compressed_size = int(original_size * kept_ratio)
        self._update_stats(original_size, compressed_size)
        
        return compressed_keys, compressed_values

class PyramidKVCompressor(BaseCompressor):
    """
    PyramidKV 压缩器
    实现层次化的KV缓存压缩策略
    """
    def __init__(
        self, 
        hidden_size: int, 
        compression_ratio: float = 0.5,
        num_pyramid_levels: int = 4,
        level_ratios: Optional[List[float]] = None
    ):
        super(PyramidKVCompressor, self).__init__(hidden_size, compression_ratio)
        self.num_levels = num_pyramid_levels
        self.level_ratios = level_ratios or [0.5 ** i for i in range(num_pyramid_levels)]
        
        # Pyramid level predictors
        self.level_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 4),
                nn.ReLU(),
                nn.Linear(hidden_size // 4, 1),
                nn.Sigmoid()
            ) for _ in range(num_pyramid_levels)
        ])
        
        # Level-specific compression layers
        self.level_compressors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, int(hidden_size * ratio)),
                nn.ReLU(),
                nn.Linear(int(hidden_size * ratio), hidden_size)
            ) for ratio in self.level_ratios
        ])
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _assign_pyramid_levels(self, x: torch.Tensor) -> torch.Tensor:
        """Assign each token to a pyramid level"""
        batch_size, seq_len, hidden_size = x.shape
        
        # Compute level probabilities for each token
        level_probs = []
        for predictor in self.level_predictors:
            probs = predictor(x)
            level_probs.append(probs)
        
        # Stack and find best level for each token
        level_probs = torch.cat(level_probs, dim=-1)
        level_assignments = torch.argmax(level_probs, dim=-1)
        
        return level_assignments
    
    def forward(
        self, 
        keys: torch.Tensor, 
        values: torch.Tensor, 
        importance: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_size = keys.shape
        
        # Assign tokens to pyramid levels
        key_levels = self._assign_pyramid_levels(keys)
        value_levels = self._assign_pyramid_levels(values)
        
        # Initialize outputs
        compressed_keys = torch.zeros_like(keys)
        compressed_values = torch.zeros_like(values)
        
        # Process each level
        for level in range(self.num_levels):
            # Create masks for current level
            key_mask = (key_levels == level)
            value_mask = (value_levels == level)
            
            if key_mask.any():
                # Extract tokens for this level
                level_keys = keys[key_mask]
                if len(level_keys.shape) == 1:
                    level_keys = level_keys.unsqueeze(0)
                
                # Apply level-specific compression
                compressed_level_keys = self.level_compressors[level](level_keys)
                
                # Put back into output tensor
                compressed_keys[key_mask] = compressed_level_keys.reshape(-1, hidden_size)
            
            if value_mask.any():
                # Extract tokens for this level
                level_values = values[value_mask]
                if len(level_values.shape) == 1:
                    level_values = level_values.unsqueeze(0)
                
                # Apply level-specific compression
                compressed_level_values = self.level_compressors[level](level_values)
                
                # Put back into output tensor
                compressed_values[value_mask] = compressed_level_values.reshape(-1, hidden_size)
        
        # Update statistics
        original_size = keys.numel() + values.numel()
        total_compression = sum(
            self.level_ratios[level] * (key_levels == level).sum().item() 
            for level in range(self.num_levels)
        )
        avg_compression = total_compression / (batch_size * seq_len * 2)
        compressed_size = int(original_size * avg_compression)
        self._update_stats(original_size, compressed_size)
        
        return compressed_keys, compressed_values

class SVDCompressor(BaseCompressor):
    """
    基于SVD的压缩器
    使用奇异值分解进行低秩近似
    """
    def __init__(
        self, 
        hidden_size: int, 
        rank: Optional[int] = None,
        compression_ratio: float = 0.5,
        adaptive_rank: bool = True,
        energy_threshold: float = 0.95
    ):
        super(SVDCompressor, self).__init__(hidden_size, compression_ratio)
        self.rank = rank or int(hidden_size * compression_ratio)
        self.adaptive_rank = adaptive_rank
        self.energy_threshold = energy_threshold
        
        # Initialize projection matrices
        self.key_projector = nn.Linear(hidden_size, self.rank)
        self.key_reconstructor = nn.Linear(self.rank, hidden_size)
        
        self.value_projector = nn.Linear(hidden_size, self.rank)
        self.value_reconstructor = nn.Linear(self.rank, hidden_size)
        
        # Statistics
        self.register_buffer('effective_rank', torch.tensor(float(self.rank)))
        self.register_buffer('energy_preserved', torch.tensor(0.0))
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _compute_adaptive_rank(self, x: torch.Tensor) -> int:
        """Compute adaptive rank based on energy preservation"""
        if not self.adaptive_rank:
            return self.rank
        
        try:
            # Flatten for SVD
            x_flat = x.reshape(-1, x.size(-1))
            
            # Compute covariance
            cov = torch.mm(x_flat.t(), x_flat) / x_flat.size(0)
            
            # Compute eigenvalues
            eigenvalues, _ = torch.linalg.eigh(cov)
            eigenvalues = eigenvalues.flip(0)  # Descending order
            
            # Find rank that preserves energy threshold
            total_energy = torch.sum(eigenvalues)
            energy_ratios = torch.cumsum(eigenvalues, dim=0) / total_energy
            
            # Find effective rank
            preserved_indices = (energy_ratios < self.energy_threshold)
            if preserved_indices.any():
                effective_rank = preserved_indices.sum().item() + 1
            else:
                effective_rank = 1
            
            # Update statistics
            self.effective_rank = torch.tensor(float(effective_rank), device=x.device)
            if effective_rank > 0:
                self.energy_preserved = energy_ratios[min(effective_rank - 1, len(energy_ratios) - 1)]
            
            return max(int(effective_rank), 1)
            
        except Exception:
            return self.rank
    
    def forward(
        self, 
        keys: torch.Tensor, 
        values: torch.Tensor, 
        importance: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Compute adaptive rank
        adaptive_rank = self._compute_adaptive_rank(keys)
        
        # Compress keys
        compressed_keys_hidden = self.key_projector(keys)
        if adaptive_rank < self.rank:
            compressed_keys_hidden = compressed_keys_hidden[..., :adaptive_rank]
            # Pad back to full rank for reconstruction
            padding = torch.zeros(
                compressed_keys_hidden.shape[:-1] + (self.rank - adaptive_rank,),
                device=compressed_keys_hidden.device,
                dtype=compressed_keys_hidden.dtype
            )
            compressed_keys_hidden = torch.cat([compressed_keys_hidden, padding], dim=-1)
        
        reconstructed_keys = self.key_reconstructor(F.relu(compressed_keys_hidden))
        
        # Compress values
        compressed_values_hidden = self.value_projector(values)
        if adaptive_rank < self.rank:
            compressed_values_hidden = compressed_values_hidden[..., :adaptive_rank]
            # Pad back to full rank for reconstruction
            padding = torch.zeros(
                compressed_values_hidden.shape[:-1] + (self.rank - adaptive_rank,),
                device=compressed_values_hidden.device,
                dtype=compressed_values_hidden.dtype
            )
            compressed_values_hidden = torch.cat([compressed_values_hidden, padding], dim=-1)
        
        reconstructed_values = self.value_reconstructor(F.relu(compressed_values_hidden))
        
        # Update statistics
        original_size = keys.numel() + values.numel()
        compressed_size = int(original_size * adaptive_rank / self.hidden_size)
        self._update_stats(original_size, compressed_size)
        
        return reconstructed_keys, reconstructed_values
    
    def get_compression_stats(self) -> Dict[str, float]:
        stats = super().get_compression_stats()
        stats.update({
            "effective_rank": self.effective_rank.item(),
            "energy_preserved": self.energy_preserved.item(),
            "adaptive_rank": self.adaptive_rank,
            "rank": self.rank
        })
        return stats

class QuantizedCompressor(BaseCompressor):
    """
    量化压缩器
    通过量化减少KV缓存的存储需求
    """
    def __init__(
        self, 
        hidden_size: int, 
        bits: int = 8,
        group_size: int = 128,
        symmetric: bool = False,
        dynamic: bool = True
    ):
        super(QuantizedCompressor, self).__init__(hidden_size, bits/32)
        self.bits = bits
        self.group_size = min(group_size, hidden_size)
        self.symmetric = symmetric
        self.dynamic = dynamic
        
        # Quantization parameters
        self.register_buffer('scale', torch.ones(1))
        self.register_buffer('zero_point', torch.zeros(1))
        
        # Statistics
        self.register_buffer('quantization_error', torch.zeros(1))
        self.calibrated = False
    
    def _quantize_tensor(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize tensor"""
        if self.bits == 8:
            if self.symmetric:
                qmin, qmax = -128, 127
            else:
                qmin, qmax = 0, 255
        elif self.bits == 4:
            if self.symmetric:
                qmin, qmax = -8, 7
            else:
                qmin, qmax = 0, 15
        elif self.bits == 16:
            if self.symmetric:
                qmin, qmax = -32768, 32767
            else:
                qmin, qmax = 0, 65535
        else:
            qmin, qmax = 0, 2**self.bits - 1
        
        if self.dynamic:
            # Dynamic quantization
            x_min = x.min()
            x_max = x.max()
            
            if x_max == x_min:
                x_max = x_min + 1e-5
            
            if self.symmetric:
                abs_max = torch.max(torch.abs(x_min), torch.abs(x_max))
                scale = 2 * abs_max / (qmax - qmin)
                zero_point = torch.tensor(0.0, device=x.device)
            else:
                scale = (x_max - x_min) / (qmax - qmin)
                zero_point = qmin - torch.round(x_min / scale)
        else:
            # Static quantization
            if not self.calibrated:
                x_min = x.min()
                x_max = x.max()
                
                if x_max == x_min:
                    x_max = x_min + 1e-5
                
                if self.symmetric:
                    abs_max = torch.max(torch.abs(x_min), torch.abs(x_max))
                    self.scale = 2 * abs_max / (qmax - qmin)
                    self.zero_point = torch.tensor(0.0)
                else:
                    self.scale = (x_max - x_min) / (qmax - qmin)
                    self.zero_point = qmin - torch.round(x_min / self.scale)
                
                self.calibrated = True
            
            scale = self.scale
            zero_point = self.zero_point
        
        # Quantize
        q_x = torch.clamp(torch.round(x / scale + zero_point), qmin, qmax)
        
        # Dequantize
        dq_x = (q_x - zero_point) * scale
        
        # Update error statistics
        with torch.no_grad():
            self.quantization_error = torch.mean((x - dq_x)**2)
        
        return q_x, dq_x, scale
    
    def forward(
        self, 
        keys: torch.Tensor, 
        values: torch.Tensor, 
        importance: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        original_shape = keys.shape
        
        # Reshape for group quantization
        keys_flat = keys.reshape(-1, original_shape[-1])
        values_flat = values.reshape(-1, original_shape[-1])
        
        # Initialize outputs
        dq_keys = torch.zeros_like(keys_flat)
        dq_values = torch.zeros_like(values_flat)
        
        # Group-wise quantization
        def quantize_by_groups(tensor):
            dq_tensor = torch.zeros_like(tensor)
            for i in range(0, tensor.size(-1), self.group_size):
                end = min(i + self.group_size, tensor.size(-1))
                _, dq_group, _ = self._quantize_tensor(tensor[:, i:end])
                dq_tensor[:, i:end] = dq_group
            return dq_tensor
        
        dq_keys = quantize_by_groups(keys_flat)
        dq_values = quantize_by_groups(values_flat)
        
        # Reshape back
        dq_keys = dq_keys.reshape(original_shape)
        dq_values = dq_values.reshape(original_shape)
        
        # Update statistics
        original_size = keys.numel() + values.numel()
        compressed_size = int(original_size * self.bits / 32)
        self._update_stats(original_size, compressed_size)
        
        return dq_keys, dq_values
    
    def get_compression_stats(self) -> Dict[str, float]:
        stats = super().get_compression_stats()
        stats.update({
            "bits": self.bits,
            "quantization_error": self.quantization_error.item(),
            "group_size": self.group_size
        })
        return stats
class ChunkKVCompressor(BaseCompressor):
    """
    ChunkKV 压缩器（完全自定义实现）
    将 KV 序列划分为多个 chunk，对每个 chunk 计算重要性分数，保留重要 chunk。
    参考自 ChunkKV: https://arxiv.org/abs/2502.00299
    """
    def __init__(
        self,
        hidden_size: int,
        compression_ratio: float = 0.5,
        chunk_length: int = 20,
        scoring_mode: str = "mean"  # or "max"
    ):
        super().__init__(hidden_size, compression_ratio)
        self.chunk_length = chunk_length
        self.scoring_mode = scoring_mode

        # 简单的打分器（可替换成更复杂的投影）
        self.scoring_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.scoring_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _chunk_scores(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        hidden: [B, T, D] → scores: [B, num_chunks]
        """
        B, T, D = hidden.shape
        L = self.chunk_length
        num_chunks = (T + L - 1) // L  # ceil division

        pad_len = num_chunks * L - T
        if pad_len > 0:
            pad = torch.zeros(B, pad_len, D, device=hidden.device)
            hidden = torch.cat([hidden, pad], dim=1)

        # reshape to [B, num_chunks, L, D]
        chunks = hidden.view(B, num_chunks, L, D)
        scores = self.scoring_mlp(chunks)  # → [B, num_chunks, L, 1]
        scores = scores.squeeze(-1)        # → [B, num_chunks, L]

        if self.scoring_mode == "max":
            chunk_scores = scores.max(dim=-1).values  # [B, num_chunks]
        else:
            chunk_scores = scores.mean(dim=-1)        # [B, num_chunks]

        return chunk_scores  # [B, num_chunks]

    def forward(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        importance: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        keys, values: [B, T, D]
        """
        B, T, D = keys.shape
        L = self.chunk_length
        num_chunks = (T + L - 1) // L
        keep_chunks = max(1, int(num_chunks * (1 - self.compression_ratio)))

        # 得分基于 keys + values 的平均表示
        combined = (keys + values) / 2
        chunk_scores = self._chunk_scores(combined)  # [B, num_chunks]

        topk = torch.topk(chunk_scores, keep_chunks, dim=-1).indices  # [B, keep_chunks]

        # 构造 token 保留 mask
        keep_mask = torch.zeros(B, num_chunks * L, device=keys.device, dtype=torch.bool)
        for b in range(B):
            for idx in topk[b]:
                start = idx.item() * L
                end = start + L
                keep_mask[b, start:end] = True
        keep_mask = keep_mask[:, :T]  # trim padding

        # 应用 mask
        keep_mask_exp = keep_mask.unsqueeze(-1).expand(-1, -1, D)
        compressed_keys = keys[keep_mask_exp].view(B, -1, D)
        compressed_values = values[keep_mask_exp].view(B, -1, D)

        # 更新统计
        original_size = keys.numel() + values.numel()
        compressed_size = compressed_keys.numel() + compressed_values.numel()
        self._update_stats(original_size, compressed_size)

        return compressed_keys, compressed_values


class PiKVCompressor(nn.Module):
    """
    PiKV综合压缩器
    结合多种压缩策略，自适应选择最佳压缩方法
    """
    def __init__(
        self, 
        hidden_size: int, 
        compression_methods: Optional[List[str]] = None,
        importance_threshold: float = 0.5,
        adaptive_selection: bool = True
    ):
        super(PiKVCompressor, self).__init__()
        self.hidden_size = hidden_size
        self.compression_methods = compression_methods or ["lora", "pyramid_kv", "fastv", "distillation", "chunk_kv"]
        self.importance_threshold = importance_threshold
        self.adaptive_selection = adaptive_selection
        
        # Create compressor instances
        self.compressors = nn.ModuleDict()
        
        for method in self.compression_methods:
            if method == "pyramid":
                self.compressors[method] = PyramidCompressor(hidden_size)
            elif method == "svd":
                self.compressors[method] = SVDCompressor(hidden_size)
            elif method == "quantized":
                self.compressors[method] = QuantizedCompressor(hidden_size)
            elif method == "lora":
                self.compressors[method] = LoRACompressor(hidden_size)
            elif method == "lora++":
                self.compressors[method] = LoRaPlusPlusCompressor(hidden_size)
            elif method == "pruning":
                self.compressors[method] = PruningCompressor(hidden_size)
            elif method == "distillation":
                self.compressors[method] = DistillationCompressor(hidden_size)
            elif method == "fastv":
                self.compressors[method] = FastVCompressor(hidden_size)
            elif method == "pyramid_kv":
                self.compressors[method] = PyramidKVCompressor(hidden_size)
            elif method == "chunk_kv":
                self.compressors[method] = ChunkKVCompressor(hidden_size)
        
        # Method selection network
        if self.adaptive_selection:
            self.method_selector = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, len(self.compression_methods)),
                nn.Softmax(dim=-1)
            )
        
        # Usage statistics
        self.register_buffer('method_usage_count', torch.zeros(len(self.compression_methods)))
        self.register_buffer('total_compressions', torch.tensor(0.0))
        
        # Method index mapping
        self.method_to_idx = {method: i for i, method in enumerate(self.compression_methods)}
    
    def _select_compression_method(
        self, 
        x: torch.Tensor, 
        importance: Optional[torch.Tensor] = None
    ) -> str:
        """Select the best compression method"""
        if not self.adaptive_selection:
            # Default to first method
            return self.compression_methods[0]
        
        # Get feature representation
        x_mean = x.mean(dim=[0, 1])
        
        # Predict method probabilities
        method_probs = self.method_selector(x_mean)
        
        # Select method with highest probability
        method_idx = torch.argmax(method_probs).item()
        selected_method = self.compression_methods[int(method_idx)]
        
        # Adjust based on importance if provided
        if importance is not None:
            mean_importance = importance.mean().item()
            
            # High importance -> prefer quality-preserving methods
            if mean_importance > self.importance_threshold:
                if "distillation" in self.compression_methods:
                    selected_method = "distillation"
                elif "lora++" in self.compression_methods:
                    selected_method = "lora++"
                elif "lora" in self.compression_methods:
                    selected_method = "lora"
            # Low importance -> prefer aggressive compression
            else:
                if "quantized" in self.compression_methods:
                    selected_method = "quantized"
                elif "pruning" in self.compression_methods:
                    selected_method = "pruning"
        
        # Update usage statistics
        method_idx = self.method_to_idx[selected_method]
        self.method_usage_count[method_idx] += 1
        self.total_compressions += 1
        
        return selected_method
    
    def forward(
        self, 
        keys: torch.Tensor, 
        values: torch.Tensor, 
        importance: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Select compression method
        method = self._select_compression_method(keys, importance)
        
        # Apply selected compression
        compressor = self.compressors[method]
        compressed_keys, compressed_values = compressor(keys, values, importance)
        
        return compressed_keys, compressed_values
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Get comprehensive compression statistics"""
        stats = {}
        
        # Collect individual compressor stats
        for method, compressor in self.compressors.items():
            stats[method] = compressor.get_compression_stats()
        
        # Usage statistics
        usage_stats = {}
        if self.total_compressions > 0:
            for i, method in enumerate(self.compression_methods):
                usage_ratio = self.method_usage_count[i].item() / self.total_compressions.item()
                usage_stats[f"{method}_usage"] = usage_ratio
        
        stats["usage"] = usage_stats
        stats["total_compressions"] = self.total_compressions.item()
        
        return stats
    
    def reset_stats(self):
        """Reset all compression statistics"""
        self.method_usage_count.zero_()
        self.total_compressions.zero_()
        
        for compressor in self.compressors.values():
            if hasattr(compressor, 'reset_stats'):
                compressor.reset_stats()
            else:
                # Reset basic stats
                if hasattr(compressor, 'total_original_size'):
                    compressor.total_original_size.zero_()
                if hasattr(compressor, 'total_compressed_size'):
                    compressor.total_compressed_size.zero_()
                if hasattr(compressor, 'compression_count'):
                    compressor.compression_count.zero_()
    
    def print_stats(self):
        """Print compression statistics"""
        stats = self.get_compression_stats()
        
        print("\n===== PiKV Comprehensive Compression Stats =====")
        print(f"Total compressions: {stats['total_compressions']:.0f}")
        
        print("\nMethod usage:")
        for method, ratio in stats["usage"].items():
            print(f"  {method}: {ratio * 100:.2f}%")
        
        print("\nMethod details:")
        for method, method_stats in stats.items():
            if method not in ["usage", "total_compressions"]:
                print(f"\n{method.upper()} Compressor:")
                for stat_name, value in method_stats.items():
                    if isinstance(value, (int, float)):
                        print(f"  {stat_name}: {value:.4f}")
                    else:
                        print(f"  {stat_name}: {value}")
        
        print("\n" + "="*50)

# Convenience function for creating compressors
def create_compressor(
    method: str, 
    hidden_size: int, 
    **kwargs
) -> BaseCompressor:
    """Create a compressor instance"""
    method_map = {
        CompressionMethod.PYRAMID.value: PyramidCompressor,
        CompressionMethod.SVD.value: SVDCompressor,
        CompressionMethod.QUANTIZED.value: QuantizedCompressor,
        CompressionMethod.LORA.value: LoRACompressor,
        CompressionMethod.LORA_PLUS_PLUS.value: LoRaPlusPlusCompressor,
        CompressionMethod.PRUNING.value: PruningCompressor,
        CompressionMethod.DISTILLATION.value: DistillationCompressor,
        CompressionMethod.FASTV.value: FastVCompressor,
        CompressionMethod.PYRAMID_KV.value: PyramidKVCompressor,
    }
    
    if method not in method_map:
        raise ValueError(f"Unknown compression method: {method}")
    
    return method_map[method](hidden_size, **kwargs) 