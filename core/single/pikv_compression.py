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
        return {
            "compression_ratio": self.compression_ratio,
            "memory_reduction": 1.0 - self.compression_ratio
        }

class PyramidCompressor(BaseCompressor):
    """
    金字塔压缩器
    实现层次化的压缩方案，在不同层级使用不同的压缩比例
    参考PyramidKV
    """
    def __init__(
        self, 
        hidden_size: int, 
        compression_ratio: float = 0.5,
        num_levels: int = 3,
        decay_factor: float = 0.8
    ):
        super(PyramidCompressor, self).__init__(hidden_size, compression_ratio)
        self.num_levels = num_levels
        self.decay_factor = decay_factor
        
        # 构建金字塔压缩层
        self.encoder_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()
        
        current_size = hidden_size
        for i in range(num_levels):
            # 计算当前层级的压缩输出大小
            output_size = max(int(current_size * compression_ratio), 1)
            
            # 创建编码和解码层
            self.encoder_layers.append(nn.Linear(current_size, output_size))
            self.decoder_layers.append(nn.Linear(output_size, current_size))
            
            # 更新当前大小用于下一层
            current_size = output_size
            
            # 调整下一层的压缩率
            compression_ratio *= decay_factor
        
        # 初始化权重
        self._init_weights()
        
        # 缓存统计信息
        self.register_buffer('compression_stats', torch.zeros(4))  # [总压缩比, 平均压缩比, 最大压缩比, 最小压缩比]
        self.register_buffer('sample_count', torch.tensor(0.0))
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _compress_level(self, x: torch.Tensor, level: int, act_fn: Optional[Any] = None) -> torch.Tensor:
        """单层压缩"""
        if level >= len(self.encoder_layers):
            return x
        
        # 编码
        x = self.encoder_layers[level](x)
        
        # 激活函数
        if act_fn is not None:
            x = act_fn(x)
        
        return x
    
    def _decompress_level(self, x: torch.Tensor, level: int, act_fn: Optional[Any] = None) -> torch.Tensor:
        """单层解压"""
        if level >= len(self.decoder_layers):
            return x
        
        # 解码
        x = self.decoder_layers[level](x)
        
        # 激活函数
        if act_fn is not None:
            x = act_fn(x)
        
        return x
    
    def forward(
        self, 
        keys: torch.Tensor, 
        values: torch.Tensor, 
        importance: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_size = keys.shape
        
        # 如果提供了重要性分数，按照重要性排序并分配到不同压缩层级
        if importance is not None:
            # 将重要性扩展到匹配键值形状
            if importance.dim() == 2:  # [batch_size, seq_len]
                importance = importance.unsqueeze(-1)
            
            # 确定每个缓存项的压缩级别
            # 重要性越高，级别越低，压缩越轻
            importance_flattened = importance.view(-1)
            sorted_indices = torch.argsort(importance_flattened, descending=True)
            
            # 分割成不同级别
            level_sizes = []
            remaining = sorted_indices.size(0)
            for i in range(self.num_levels):
                # 最后一层获取所有剩余项
                if i == self.num_levels - 1:
                    level_sizes.append(remaining)
                else:
                    # 根据重要性比例分配
                    # 重要性高的层获得更多的分配
                    level_size = int(remaining * (1 - self.decay_factor) * (self.decay_factor ** i))
                    level_sizes.append(level_size)
                    remaining -= level_size
            
            # 初始化压缩后的键值
            flattened_keys = keys.reshape(-1, hidden_size)
            flattened_values = values.reshape(-1, hidden_size)
            
            compressed_keys = torch.zeros_like(flattened_keys)
            compressed_values = torch.zeros_like(flattened_values)
            
            # 对每个层级分别压缩
            start_idx = 0
            for level, size in enumerate(level_sizes):
                if size <= 0:
                    continue
                    
                end_idx = start_idx + size
                level_indices = sorted_indices[start_idx:end_idx]
                
                # 提取当前层级的键值
                level_keys = flattened_keys[level_indices]
                level_values = flattened_values[level_indices]
                
                # 应用金字塔压缩
                for i in range(level):
                    level_keys = self._compress_level(level_keys, i, F.relu)
                    level_values = self._compress_level(level_values, i, F.relu)
                
                # 应用相同层数的解压
                for i in range(level-1, -1, -1):
                    level_keys = self._decompress_level(level_keys, i, F.relu)
                    level_values = self._decompress_level(level_values, i, F.relu)
                
                # 将压缩后的结果放回原始位置
                compressed_keys[level_indices] = level_keys
                compressed_values[level_indices] = level_values
                
                start_idx = end_idx
            
            # 重塑回原始形状
            compressed_keys = compressed_keys.reshape(batch_size, seq_len, hidden_size)
            compressed_values = compressed_values.reshape(batch_size, seq_len, hidden_size)
        
        else:
            # 如果没有重要性分数，均匀压缩所有键值
            compressed_keys = keys
            compressed_values = values
            
            # 应用金字塔压缩
            for i in range(self.num_levels):
                compressed_keys = self._compress_level(compressed_keys, i, F.relu)
                compressed_values = self._compress_level(compressed_values, i, F.relu)
            
            # 应用解压
            for i in range(self.num_levels-1, -1, -1):
                compressed_keys = self._decompress_level(compressed_keys, i, F.relu)
                compressed_values = self._decompress_level(compressed_values, i, F.relu)
        
        # 更新压缩统计信息
        with torch.no_grad():
            # 计算压缩比 (压缩后大小 / 原始大小)
            if keys.numel() > 0:
                current_ratio = compressed_keys.numel() / keys.numel()
                self.compression_stats[0] += current_ratio  # 总压缩比
                self.compression_stats[1] = self.compression_stats[0] / (self.sample_count + 1)  # 平均压缩比
                self.compression_stats[2] = max(self.compression_stats[2], current_ratio)  # 最大压缩比
                if self.compression_stats[3] == 0:
                    self.compression_stats[3] = current_ratio
                else:
                    self.compression_stats[3] = min(self.compression_stats[3], current_ratio)  # 最小压缩比
                self.sample_count += 1
        
        return compressed_keys, compressed_values
    
    def get_compression_stats(self) -> Dict[str, float]:
        """获取压缩统计信息"""
        stats = {
            "compression_ratio": self.compression_stats[1].item(),  # 平均压缩比
            "max_compression_ratio": self.compression_stats[2].item(),
            "min_compression_ratio": self.compression_stats[3].item(),
            "memory_reduction": 1.0 - self.compression_stats[1].item(),
            "num_samples": self.sample_count.item()
        }
        return stats

class SVDCompressor(BaseCompressor):
    """
    基于SVD的压缩器
    使用奇异值分解进行低秩近似
    """
    def __init__(
        self, 
        hidden_size: int, 
        rank: int = 64,
        adaptive_rank: bool = False
    ):
        super(SVDCompressor, self).__init__(hidden_size, rank/hidden_size)
        self.rank = rank
        self.adaptive_rank = adaptive_rank
        
        # 初始化投影矩阵
        self.key_projector = nn.Linear(hidden_size, rank)
        self.key_reconstructor = nn.Linear(rank, hidden_size)
        
        self.value_projector = nn.Linear(hidden_size, rank)
        self.value_reconstructor = nn.Linear(rank, hidden_size)
        
        # 初始化权重
        self._init_weights()
        
        # 记录压缩器信息
        self.register_buffer('energy_preserved', torch.zeros(1))
        self.register_buffer('effective_rank', torch.zeros(1))
        self.register_buffer('sample_count', torch.tensor(0.0))
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _compute_adaptive_rank(self, x: torch.Tensor, importance: torch.Tensor = None) -> int:
        """计算自适应秩"""
        if not self.adaptive_rank:
            return self.rank
        
        # 计算x的SVD
        try:
            # 将x压平为2D矩阵
            x_flat = x.reshape(-1, x.size(-1))
            
            # 计算协方差矩阵
            cov = torch.mm(x_flat.t(), x_flat) / x_flat.size(0)
            
            # 计算特征值
            eigenvalues, _ = torch.linalg.eigh(cov)
            
            # 对特征值排序（降序）
            eigenvalues = eigenvalues.flip(0)
            
            # 计算能量保留比例
            total_energy = torch.sum(eigenvalues)
            energy_ratio = torch.cumsum(eigenvalues, dim=0) / total_energy
            
            # 找到保留95%能量所需的秩
            target_energy = 0.95
            effective_rank = torch.sum(energy_ratio < target_energy).item() + 1
            
            # 更新统计信息
            self.energy_preserved = target_energy
            self.effective_rank = min(effective_rank, self.rank)
            
            # 重要性调整
            if importance is not None:
                # 重要的token应该使用更高的秩
                importance_factor = importance.mean().item()
                rank_factor = 0.5 + importance_factor * 0.5  # 范围[0.5, 1.0]
                return max(int(effective_rank * rank_factor), 1)
                
            return max(effective_rank, 1)
            
        except Exception:
            # 如果计算失败，返回默认秩
            return self.rank
    
    def forward(
        self, 
        keys: torch.Tensor, 
        values: torch.Tensor, 
        importance: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 计算自适应秩
        adaptive_rank = self._compute_adaptive_rank(keys, importance)
        
        # 压缩键
        compressed_keys = self.key_projector(keys)
        # 如果自适应秩小于配置的秩，对压缩结果进行截断
        if adaptive_rank < self.rank:
            compressed_keys = compressed_keys[..., :adaptive_rank]
        # 解压缩键
        reconstructed_keys = self.key_reconstructor(F.relu(compressed_keys))
        
        # 压缩值
        compressed_values = self.value_projector(values)
        # 如果自适应秩小于配置的秩，对压缩结果进行截断
        if adaptive_rank < self.rank:
            compressed_values = compressed_values[..., :adaptive_rank]
        # 解压缩值
        reconstructed_values = self.value_reconstructor(F.relu(compressed_values))
        
        # 更新样本计数
        with torch.no_grad():
            self.sample_count += 1
        
        return reconstructed_keys, reconstructed_values
    
    def get_compression_stats(self) -> Dict[str, float]:
        stats = super().get_compression_stats()
        stats.update({
            "effective_rank": self.effective_rank.item(),
            "energy_preserved": self.energy_preserved.item(),
            "adaptive_rank": self.adaptive_rank,
            "rank": self.rank,
            "num_samples": self.sample_count.item()
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
        dynamic_quantization: bool = True
    ):
        super(QuantizedCompressor, self).__init__(hidden_size, bits/32)  # 与全精度相比的压缩比
        self.bits = bits
        self.group_size = min(group_size, hidden_size)
        self.dynamic_quantization = dynamic_quantization
        
        # 权重和偏置用于校准量化
        self.register_buffer('scale', torch.ones(1))
        self.register_buffer('zero_point', torch.zeros(1))
        
        # 是否已初始化校准
        self.calibrated = False
        
        # 统计数据
        self.register_buffer('quantization_error', torch.zeros(1))
        self.register_buffer('sample_count', torch.tensor(0.0))
    
    def _quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """量化张量"""
        if self.dynamic_quantization:
            # 动态计算每个批次的缩放因子和零点
            if self.bits == 8:
                qmin, qmax = 0, 255
            elif self.bits == 4:
                qmin, qmax = 0, 15
            else:
                qmin, qmax = 0, 2**self.bits - 1
            
            # 计算量化参数
            x_min = x.min()
            x_max = x.max()
            
            # 避免除零
            if x_max == x_min:
                x_max = x_min + 1e-5
                
            scale = (x_max - x_min) / (qmax - qmin)
            zero_point = qmin - torch.round(x_min / scale)
            
            # 量化
            q_x = torch.clamp(torch.round(x / scale + zero_point), qmin, qmax)
            
            # 反量化
            dq_x = (q_x - zero_point) * scale
            
            # 更新统计信息
            with torch.no_grad():
                self.quantization_error = torch.mean((x - dq_x)**2)
                self.sample_count += 1
            
            return q_x, dq_x, scale
        else:
            # 使用预先校准的量化参数
            if not self.calibrated:
                # 首次计算量化参数
                if self.bits == 8:
                    qmin, qmax = 0, 255
                elif self.bits == 4:
                    qmin, qmax = 0, 15
                else:
                    qmin, qmax = 0, 2**self.bits - 1
                
                # 计算量化参数
                x_min = x.min()
                x_max = x.max()
                
                # 避免除零
                if x_max == x_min:
                    x_max = x_min + 1e-5
                    
                self.scale = (x_max - x_min) / (qmax - qmin)
                self.zero_point = qmin - torch.round(x_min / self.scale)
                
                self.calibrated = True
            
            # 量化
            if self.bits == 8:
                qmin, qmax = 0, 255
            elif self.bits == 4:
                qmin, qmax = 0, 15
            else:
                qmin, qmax = 0, 2**self.bits - 1
                
            q_x = torch.clamp(torch.round(x / self.scale + self.zero_point), qmin, qmax)
            
            # 反量化
            dq_x = (q_x - self.zero_point) * self.scale
            
            # 更新统计信息
            with torch.no_grad():
                self.quantization_error = torch.mean((x - dq_x)**2)
                self.sample_count += 1
            
            return q_x, dq_x, self.scale
    
    def forward(
        self, 
        keys: torch.Tensor, 
        values: torch.Tensor, 
        importance: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        original_shape = keys.shape
        
        # 重塑为2D以便分组量化
        keys_flat = keys.reshape(-1, original_shape[-1])
        values_flat = values.reshape(-1, original_shape[-1])
        
        # 初始化输出
        dq_keys = torch.zeros_like(keys_flat)
        dq_values = torch.zeros_like(values_flat)
        
        # 按组量化
        for i in range(0, original_shape[-1], self.group_size):
            end = min(i + self.group_size, original_shape[-1])
            
            # 量化键
            _, dq_keys_group, _ = self._quantize(keys_flat[:, i:end])
            dq_keys[:, i:end] = dq_keys_group
            
            # 量化值
            _, dq_values_group, _ = self._quantize(values_flat[:, i:end])
            dq_values[:, i:end] = dq_values_group
        
        # 重塑回原始形状
        dq_keys = dq_keys.reshape(original_shape)
        dq_values = dq_values.reshape(original_shape)
        
        return dq_keys, dq_values
    
    def get_compression_stats(self) -> Dict[str, float]:
        stats = super().get_compression_stats()
        stats.update({
            "bits": self.bits,
            "quantization_error": self.quantization_error.item(),
            "group_size": self.group_size,
            "num_samples": self.sample_count.item()
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
        compressor_types: List[str] = ["pyramid", "svd", "quantization"],
        importance_threshold: float = 0.5
    ):
        super(PiKVCompressor, self).__init__()
        self.hidden_size = hidden_size
        self.compressor_types = compressor_types
        self.importance_threshold = importance_threshold
        
        # 创建各种压缩器
        self.compressors = nn.ModuleDict()
        
        if "pyramid" in compressor_types:
            self.compressors["pyramid"] = PyramidCompressor(
                hidden_size=hidden_size,
                compression_ratio=0.5,
                num_levels=3
            )
        
        if "svd" in compressor_types:
            self.compressors["svd"] = SVDCompressor(
                hidden_size=hidden_size,
                rank=int(hidden_size * 0.25),
                adaptive_rank=True
            )
        
        if "quantization" in compressor_types:
            self.compressors["quantization"] = QuantizedCompressor(
                hidden_size=hidden_size,
                bits=8,
                dynamic_quantization=True
            )
        
        # 压缩器选择网络
        # 输入token的特征，预测最合适的压缩器类型
        self.selector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, len(compressor_types)),
            nn.Softmax(dim=-1)
        )
        
        # 记录各压缩器的使用情况
        self.register_buffer('compression_use_count', torch.zeros(len(compressor_types)))
        self.register_buffer('total_tokens', torch.tensor(0.0))
        
        # 为打印信息做准备
        self.compressor_idx_map = {name: i for i, name in enumerate(compressor_types)}
    
    def _select_compressor(
        self, 
        x: torch.Tensor, 
        importance: Optional[torch.Tensor] = None
    ) -> Tuple[str, int]:
        """选择最合适的压缩器"""
        # 获取特征的均值作为选择依据
        x_mean = x.mean(dim=[0, 1])
        
        # 预测最佳压缩器
        scores = self.selector(x_mean)
        
        # 默认选择概率最高的压缩器
        compressor_idx = torch.argmax(scores).item()
        compressor_name = self.compressor_types[compressor_idx]
        
        # 如果提供了重要性分数，基于重要性调整选择
        if importance is not None:
            # 计算重要性的均值
            mean_importance = importance.mean().item()
            
            # 如果重要性高于阈值，选择保留更多信息的压缩器
            if mean_importance > self.importance_threshold:
                # 选择SVD或金字塔压缩器，它们保留更多信息
                if "svd" in self.compressor_types:
                    compressor_name = "svd"
                    compressor_idx = self.compressor_idx_map["svd"]
                elif "pyramid" in self.compressor_types:
                    compressor_name = "pyramid"
                    compressor_idx = self.compressor_idx_map["pyramid"]
            else:
                # 对于不太重要的token，可以使用更激进的压缩
                if "quantization" in self.compressor_types:
                    compressor_name = "quantization"
                    compressor_idx = self.compressor_idx_map["quantization"]
        
        # 更新使用计数
        with torch.no_grad():
            self.compression_use_count[compressor_idx] += 1
            self.total_tokens += 1
        
        return compressor_name, compressor_idx
    
    def forward(
        self, 
        keys: torch.Tensor, 
        values: torch.Tensor, 
        importance: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 选择压缩器
        compressor_name, _ = self._select_compressor(keys, importance)
        
        # 使用选定的压缩器
        compressor = self.compressors[compressor_name]
        compressed_keys, compressed_values = compressor(keys, values, importance)
        
        return compressed_keys, compressed_values
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """获取综合压缩统计信息"""
        stats = {}
        
        # 收集各压缩器的统计信息
        for name, compressor in self.compressors.items():
            stats[name] = compressor.get_compression_stats()
        
        # 计算使用比例
        usage_stats = {}
        if self.total_tokens > 0:
            for i, name in enumerate(self.compressor_types):
                usage_ratio = self.compression_use_count[i].item() / self.total_tokens.item()
                usage_stats[f"{name}_usage"] = usage_ratio
        
        stats["usage"] = usage_stats
        stats["total_tokens"] = self.total_tokens.item()
        
        return stats
        
    def print_stats(self):
        """打印压缩统计信息"""
        stats = self.get_compression_stats()
        
        print("\n===== PiKV Compressor Stats =====")
        print(f"Total tokens processed: {stats['total_tokens']:.0f}")
        
        print("\nCompressor usage:")
        for name, ratio in stats["usage"].items():
            print(f"  {name}: {ratio * 100:.2f}%")
        
        print("\nCompressor details:")
        for name, compressor_stats in stats.items():
            if name not in ["usage", "total_tokens"]:
                print(f"\n{name.upper()} Compressor:")
                for stat_name, value in compressor_stats.items():
                    print(f"  {stat_name}: {value}")
                    
        print("\n=================================") 