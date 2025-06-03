"""
PiKV Compression Module

实现多种KV缓存压缩策略，包括：
- PyramidCompressor: 层次化金字塔压缩
- SVDCompressor: 基于SVD的低秩压缩
- QuantizedCompressor: 量化压缩
- PiKVCompressor: 自适应综合压缩器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Dict, List, Optional, Union, Any

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
        self.register_buffer('total_compressed_size', torch.tensor(0.0))
        self.register_buffer('total_original_size', torch.tensor(0.0))
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
        return keys, values
    
    def get_compression_stats(self) -> Dict[str, float]:
        """
        获取压缩统计信息
        
        Returns:
            stats: 字典包含压缩率、内存减少等信息
        """
        if self.compression_count > 0:
            actual_ratio = self.total_compressed_size / self.total_original_size
        else:
            actual_ratio = 1.0
            
        return {
            "target_compression_ratio": self.compression_ratio,
            "actual_compression_ratio": actual_ratio.item(),
            "memory_reduction": 1.0 - actual_ratio.item(),
            "compression_count": self.compression_count.item()
        }
    
    def _update_stats(self, original_size: int, compressed_size: int):
        """更新压缩统计信息"""
        self.total_original_size += original_size
        self.total_compressed_size += compressed_size
        self.compression_count += 1

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
        
        # 如果没有提供重要性，均匀分配到所有层级
        if importance is None:
            importance = torch.ones(batch_size, seq_len, device=keys.device) * 0.5
        
        # 分配压缩级别
        if self.use_adaptive_levels:
            key_levels = self._assign_compression_levels(importance)
            value_levels = key_levels.clone()
        else:
            # 使用简单的轮换分配
            key_levels = torch.arange(seq_len, device=keys.device) % self.num_levels
            key_levels = key_levels.unsqueeze(0).expand(batch_size, -1)
            value_levels = key_levels.clone()
        
        # 应用金字塔压缩
        compressed_keys = self._apply_pyramid_compression(keys, key_levels)
        compressed_values = self._apply_pyramid_compression(values, value_levels)
        
        # 更新统计信息
        original_size = keys.numel() + values.numel()
        # 压缩大小基于平均压缩层级
        avg_level = key_levels.float().mean().item()
        effective_ratio = self.compression_ratio ** (avg_level + 1)
        compressed_size = int(original_size * effective_ratio)
        self._update_stats(original_size, compressed_size)
        
        return compressed_keys, compressed_values

class SVDCompressor(BaseCompressor):
    """
    基于SVD的压缩器
    使用奇异值分解进行低秩近似压缩
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
        self.rank = rank if rank is not None else max(int(hidden_size * compression_ratio), 1)
        self.adaptive_rank = adaptive_rank
        self.energy_threshold = energy_threshold
        
        # 学习的投影矩阵（替代直接SVD分解）
        self.key_projector = nn.Linear(hidden_size, self.rank, bias=False)
        self.key_reconstructor = nn.Linear(self.rank, hidden_size, bias=False)
        
        self.value_projector = nn.Linear(hidden_size, self.rank, bias=False)
        self.value_reconstructor = nn.Linear(self.rank, hidden_size, bias=False)
        
        # 初始化为正交矩阵
        self._init_weights()
        
        # 记录有效秩和能量保留
        self.register_buffer('effective_rank', torch.tensor(self.rank))
        self.register_buffer('energy_preserved', torch.tensor(0.0))
        
    def _init_weights(self):
        """用正交矩阵初始化权重"""
        nn.init.orthogonal_(self.key_projector.weight)
        nn.init.orthogonal_(self.key_reconstructor.weight)
        nn.init.orthogonal_(self.value_projector.weight)
        nn.init.orthogonal_(self.value_reconstructor.weight)
    
    def _compute_adaptive_rank(self, x: torch.Tensor) -> int:
        """计算自适应秩"""
        if not self.adaptive_rank:
            return self.rank
        
        try:
            # 计算协方差矩阵的特征值
            x_centered = x - x.mean(dim=0, keepdim=True)
            cov_matrix = torch.mm(x_centered.t(), x_centered) / x_centered.size(0)
            
            eigenvalues = torch.linalg.eigvals(cov_matrix).real
            eigenvalues = torch.sort(eigenvalues, descending=True)[0]
            
            # 计算累积能量比例
            total_energy = eigenvalues.sum()
            cumsum_energy = torch.cumsum(eigenvalues, dim=0)
            energy_ratios = cumsum_energy / total_energy
            
            # 找到达到阈值的最小秩
            effective_rank = torch.sum(energy_ratios < self.energy_threshold).item() + 1
            effective_rank = min(effective_rank, self.rank)
            
            # 更新统计信息
            self.effective_rank = torch.tensor(effective_rank, device=x.device)
            self.energy_preserved = energy_ratios[effective_rank - 1]
            
            return max(effective_rank, 1)
            
        except Exception:
            return self.rank
    
    def forward(
        self, 
        keys: torch.Tensor, 
        values: torch.Tensor, 
        importance: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_size = keys.shape
        
        # 重塑为2D进行SVD压缩
        keys_2d = keys.view(-1, hidden_size)
        values_2d = values.view(-1, hidden_size)
        
        # 计算自适应秩
        if self.adaptive_rank:
            key_rank = self._compute_adaptive_rank(keys_2d)
            value_rank = self._compute_adaptive_rank(values_2d)
            effective_rank = min(key_rank, value_rank)
        else:
            effective_rank = self.rank
        
        # 压缩键
        compressed_keys = self.key_projector(keys_2d)
        if effective_rank < self.rank:
            compressed_keys = compressed_keys[..., :effective_rank]
            # 使用相应的解码器部分
            reconstructed_keys = F.linear(
                compressed_keys, 
                self.key_reconstructor.weight[:, :effective_rank]
            )
        else:
            reconstructed_keys = self.key_reconstructor(compressed_keys)
        
        # 压缩值
        compressed_values = self.value_projector(values_2d)
        if effective_rank < self.rank:
            compressed_values = compressed_values[..., :effective_rank]
            # 使用相应的解码器部分
            reconstructed_values = F.linear(
                compressed_values, 
                self.value_reconstructor.weight[:, :effective_rank]
            )
        else:
            reconstructed_values = self.value_reconstructor(compressed_values)
        
        # 重塑回原始形状
        reconstructed_keys = reconstructed_keys.view(batch_size, seq_len, hidden_size)
        reconstructed_values = reconstructed_values.view(batch_size, seq_len, hidden_size)
        
        # 添加残差连接
        reconstructed_keys = keys + reconstructed_keys
        reconstructed_values = values + reconstructed_values
        
        # 更新统计信息
        original_size = keys.numel() + values.numel()
        compressed_size = int(original_size * (effective_rank / self.rank) * self.compression_ratio)
        self._update_stats(original_size, compressed_size)
        
        return reconstructed_keys, reconstructed_values

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
        super(QuantizedCompressor, self).__init__(hidden_size, bits / 32.0)
        self.bits = bits
        self.group_size = min(group_size, hidden_size)
        self.symmetric = symmetric
        self.dynamic = dynamic
        
        # 量化参数
        if not dynamic:
            self.register_buffer('scale', torch.ones(1))
            self.register_buffer('zero_point', torch.zeros(1))
            self.calibrated = False
        
        # 统计信息
        self.register_buffer('quantization_error', torch.tensor(0.0))
        
    def _quantize_tensor(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """量化张量"""
        if self.bits == 8:
            qmin, qmax = -128, 127 if self.symmetric else 0, 255
        elif self.bits == 4:
            qmin, qmax = -8, 7 if self.symmetric else 0, 15
        elif self.bits == 16:
            qmin, qmax = -32768, 32767 if self.symmetric else 0, 65535
        else:
            qmin, qmax = 0, 2**self.bits - 1
        
        if self.dynamic:
            # 动态量化：每次计算量化参数
            x_min = x.min()
            x_max = x.max()
            
            if self.symmetric:
                abs_max = torch.max(torch.abs(x_min), torch.abs(x_max))
                scale = abs_max / (qmax - qmin) * 2
                zero_point = torch.tensor(0.0, device=x.device)
            else:
                scale = (x_max - x_min) / (qmax - qmin)
                zero_point = qmin - torch.round(x_min / scale)
            
            # 避免除零
            scale = torch.clamp(scale, min=1e-8)
        else:
            # 静态量化：使用预校准的参数
            if not self.calibrated:
                # 校准量化参数
                x_min = x.min()
                x_max = x.max()
                
                if self.symmetric:
                    abs_max = torch.max(torch.abs(x_min), torch.abs(x_max))
                    self.scale = abs_max / (qmax - qmin) * 2
                    self.zero_point = torch.tensor(0.0)
                else:
                    self.scale = (x_max - x_min) / (qmax - qmin)
                    self.zero_point = qmin - torch.round(x_min / self.scale)
                
                self.scale = torch.clamp(self.scale, min=1e-8)
                self.calibrated = True
            
            scale = self.scale
            zero_point = self.zero_point
        
        # 量化
        q_x = torch.clamp(torch.round(x / scale + zero_point), qmin, qmax)
        
        # 反量化
        dq_x = (q_x - zero_point) * scale
        
        return q_x, dq_x, scale
    
    def forward(
        self, 
        keys: torch.Tensor, 
        values: torch.Tensor, 
        importance: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        original_shape = keys.shape
        
        # 按组量化以提高精度
        def quantize_by_groups(tensor):
            tensor_flat = tensor.view(-1, original_shape[-1])
            dq_tensor = torch.zeros_like(tensor_flat)
            
            for i in range(0, original_shape[-1], self.group_size):
                end = min(i + self.group_size, original_shape[-1])
                group = tensor_flat[:, i:end]
                
                _, dq_group, _ = self._quantize_tensor(group)
                dq_tensor[:, i:end] = dq_group
            
            return dq_tensor.view(original_shape)
        
        # 量化键和值
        dq_keys = quantize_by_groups(keys)
        dq_values = quantize_by_groups(values)
        
        # 计算量化误差
        with torch.no_grad():
            key_error = F.mse_loss(dq_keys, keys)
            value_error = F.mse_loss(dq_values, values)
            self.quantization_error = (key_error + value_error) / 2
        
        # 更新统计信息
        original_size = keys.numel() + values.numel()
        compressed_size = int(original_size * self.compression_ratio)
        self._update_stats(original_size, compressed_size)
        
        return dq_keys, dq_values
    
    def get_compression_stats(self) -> Dict[str, float]:
        stats = super().get_compression_stats()
        stats.update({
            "bits": self.bits,
            "quantization_error": self.quantization_error.item(),
            "group_size": self.group_size,
            "symmetric": self.symmetric,
            "dynamic": self.dynamic
        })
        return stats

class PiKVCompressor(nn.Module):
    """
    PiKV综合压缩器
    结合多种压缩策略，自适应选择最佳压缩方法
    """
    def __init__(
        self, 
        hidden_size: int, 
        compression_methods: List[str] = ["pyramid", "svd", "quantized"],
        importance_threshold: float = 0.5,
        adaptive_selection: bool = True
    ):
        super(PiKVCompressor, self).__init__()
        self.hidden_size = hidden_size
        self.compression_methods = compression_methods
        self.importance_threshold = importance_threshold
        self.adaptive_selection = adaptive_selection
        
        # 创建各种压缩器
        self.compressors = nn.ModuleDict()
        
        if "pyramid" in compression_methods:
            self.compressors["pyramid"] = PyramidCompressor(
                hidden_size=hidden_size,
                compression_ratio=0.5,
                num_levels=3
            )
        
        if "svd" in compression_methods:
            self.compressors["svd"] = SVDCompressor(
                hidden_size=hidden_size,
                compression_ratio=0.3,
                adaptive_rank=True
            )
        
        if "quantized" in compression_methods:
            self.compressors["quantized"] = QuantizedCompressor(
                hidden_size=hidden_size,
                bits=8,
                dynamic=True
            )
        
        # 自适应选择网络
        if adaptive_selection:
            self.selector = nn.Sequential(
                nn.Linear(hidden_size + 1, hidden_size // 2),  # +1 for importance
                nn.ReLU(),
                nn.Linear(hidden_size // 2, len(compression_methods)),
                nn.Softmax(dim=-1)
            )
        
        # 使用统计
        self.register_buffer('method_usage_count', torch.zeros(len(compression_methods)))
        self.register_buffer('total_usage_count', torch.tensor(0.0))
        
        # 方法索引映射
        self.method_to_idx = {method: i for i, method in enumerate(compression_methods)}
    
    def _select_compression_method(
        self, 
        x: torch.Tensor, 
        importance: Optional[torch.Tensor] = None
    ) -> str:
        """选择最佳压缩方法"""
        if not self.adaptive_selection or not hasattr(self, 'selector'):
            # 基于重要性的简单选择
            if importance is not None:
                mean_importance = importance.mean().item()
                if mean_importance > self.importance_threshold:
                    # 高重要性：使用保真度高的压缩
                    return "svd" if "svd" in self.compression_methods else self.compression_methods[0]
                else:
                    # 低重要性：使用更激进的压缩
                    return "quantized" if "quantized" in self.compression_methods else self.compression_methods[-1]
            else:
                return self.compression_methods[0]
        
        # 自适应网络选择
        batch_size, seq_len, hidden_size = x.shape
        
        # 计算输入特征
        x_features = x.mean(dim=[0, 1])  # [hidden_size]
        
        # 添加重要性特征
        if importance is not None:
            importance_feature = importance.mean()
        else:
            importance_feature = torch.tensor(0.5, device=x.device)
        
        # 组合特征
        features = torch.cat([x_features, importance_feature.unsqueeze(0)])
        
        # 预测最佳方法
        method_probs = self.selector(features)
        method_idx = torch.argmax(method_probs).item()
        
        return self.compression_methods[method_idx]
    
    def forward(
        self, 
        keys: torch.Tensor, 
        values: torch.Tensor, 
        importance: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 选择压缩方法
        method = self._select_compression_method(keys, importance)
        
        # 使用选定的压缩器
        compressor = self.compressors[method]
        compressed_keys, compressed_values = compressor(keys, values, importance)
        
        # 更新使用统计
        with torch.no_grad():
            method_idx = self.method_to_idx[method]
            self.method_usage_count[method_idx] += 1
            self.total_usage_count += 1
        
        return compressed_keys, compressed_values
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """获取综合压缩统计信息"""
        stats = {}
        
        # 收集各压缩器的统计信息
        for method, compressor in self.compressors.items():
            stats[f"{method}_stats"] = compressor.get_compression_stats()
        
        # 使用率统计
        if self.total_usage_count > 0:
            usage_stats = {}
            for method, idx in self.method_to_idx.items():
                usage_ratio = self.method_usage_count[idx] / self.total_usage_count
                usage_stats[f"{method}_usage_ratio"] = usage_ratio.item()
            stats["usage_statistics"] = usage_stats
        
        stats["total_compressions"] = self.total_usage_count.item()
        
        return stats
    
    def reset_stats(self):
        """重置所有统计信息"""
        self.method_usage_count.zero_()
        self.total_usage_count.zero_()
        
        for compressor in self.compressors.values():
            if hasattr(compressor, 'total_compressed_size'):
                compressor.total_compressed_size.zero_()
                compressor.total_original_size.zero_()
                compressor.compression_count.zero_() 