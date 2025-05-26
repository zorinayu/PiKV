import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional, Union, Any
from enum import Enum
from .config import config

class SchedulingPolicy(Enum):
    """缓存调度策略枚举"""
    NONE = "none"
    H2O = "h2o"
    STREAMING_LLM = "streaming_llm"
    QUEST = "quest"
    FLEXGEN = "flexgen"
    LRU = "lru"
    LRU_PLUS = "lru_plus"

class BaseScheduler(nn.Module):
    """基础缓存调度器"""
    def __init__(self, cache_size: int, hidden_size: int):
        super(BaseScheduler, self).__init__()
        self.cache_size = cache_size
        self.hidden_size = hidden_size
        
        # 缓存统计信息
        self.register_buffer('hit_count', torch.tensor(0))
        self.register_buffer('miss_count', torch.tensor(0))
        self.register_buffer('eviction_count', torch.tensor(0))
    
    def get_hit_rate(self) -> float:
        """获取缓存命中率"""
        total = self.hit_count + self.miss_count
        return self.hit_count.float() / total.float() if total > 0 else 0.0
    
    def reset_stats(self):
        """重置统计信息"""
        self.hit_count.zero_()
        self.miss_count.zero_()
        self.eviction_count.zero_()
    
    def should_evict(self, cache_usage: torch.Tensor, new_importance: torch.Tensor) -> torch.Tensor:
        """判断是否需要淘汰缓存项"""
        raise NotImplementedError
    
    def select_eviction_candidates(self, 
                                 keys: torch.Tensor, 
                                 values: torch.Tensor, 
                                 metadata: Dict[str, torch.Tensor]) -> torch.Tensor:
        """选择淘汰候选项"""
        raise NotImplementedError

class H2OScheduler(BaseScheduler):
    """
    H2O (Heavy Hitters Oracle) 调度器
    基于注意力权重的重要性进行缓存管理
    """
    def __init__(self, cache_size: int, hidden_size: int, 
                 heavy_ratio: float = 0.1, recent_ratio: float = 0.1):
        super(H2OScheduler, self).__init__(cache_size, hidden_size)
        self.heavy_ratio = heavy_ratio  # 重要token比例
        self.recent_ratio = recent_ratio  # 最近token比例
        
        # 注意力权重累积器
        self.register_buffer('attention_accumulator', torch.zeros(cache_size))
        self.register_buffer('access_timestamps', torch.zeros(cache_size))
        self.register_buffer('current_time', torch.tensor(0))
    
    def update_attention_scores(self, indices: torch.Tensor, attention_weights: torch.Tensor):
        """更新注意力分数"""
        self.attention_accumulator[indices] += attention_weights
        self.access_timestamps[indices] = self.current_time
        self.current_time += 1
    
    def select_eviction_candidates(self, 
                                 keys: torch.Tensor, 
                                 values: torch.Tensor, 
                                 metadata: Dict[str, torch.Tensor]) -> torch.Tensor:
        """H2O淘汰策略：保留重要token和最近token"""
        cache_len = keys.size(0)
        
        # 计算重要token数量
        num_heavy = max(1, int(cache_len * self.heavy_ratio))
        num_recent = max(1, int(cache_len * self.recent_ratio))
        
        # 获取重要token（基于累积注意力权重）
        _, heavy_indices = torch.topk(self.attention_accumulator[:cache_len], num_heavy)
        
        # 获取最近token（基于时间戳）
        _, recent_indices = torch.topk(self.access_timestamps[:cache_len], num_recent)
        
        # 合并保留的索引
        keep_indices = torch.cat([heavy_indices, recent_indices]).unique()
        
        # 创建淘汰掩码
        evict_mask = torch.ones(cache_len, dtype=torch.bool, device=keys.device)
        evict_mask[keep_indices] = False
        
        return evict_mask

class StreamingLLMScheduler(BaseScheduler):
    """
    StreamingLLM 调度器
    保留初始token和最近的滑动窗口
    """
    def __init__(self, cache_size: int, hidden_size: int, 
                 start_size: int = 4, recent_size: Optional[int] = None):
        super(StreamingLLMScheduler, self).__init__(cache_size, hidden_size)
        self.start_size = start_size  # 保留的初始token数量
        self.recent_size = recent_size if recent_size is not None else (cache_size - start_size)  # 滑动窗口大小
        
        self.register_buffer('position_counter', torch.tensor(0))
    
    def select_eviction_candidates(self, 
                                 keys: torch.Tensor, 
                                 values: torch.Tensor, 
                                 metadata: Dict[str, torch.Tensor]) -> torch.Tensor:
        """StreamingLLM淘汰策略：保留开始token和最近窗口"""
        cache_len = keys.size(0)
        
        if cache_len <= self.cache_size:
            return torch.zeros(cache_len, dtype=torch.bool, device=keys.device)
        
        # 创建淘汰掩码
        evict_mask = torch.ones(cache_len, dtype=torch.bool, device=keys.device)
        
        # 保留开始的token
        evict_mask[:self.start_size] = False
        
        # 保留最近的token
        recent_start = max(self.start_size, cache_len - self.recent_size)
        evict_mask[recent_start:] = False
        
        return evict_mask

class QUESTScheduler(BaseScheduler):
    """
    QUEST (Quality-aware Eviction with Streaming) 调度器
    基于质量感知的流式淘汰策略
    """
    def __init__(self, cache_size: int, hidden_size: int, 
                 quality_threshold: float = 0.1, decay_factor: float = 0.95):
        super(QUESTScheduler, self).__init__(cache_size, hidden_size)
        self.quality_threshold = quality_threshold
        self.decay_factor = decay_factor
        
        # 质量分数和使用频率
        self.register_buffer('quality_scores', torch.zeros(cache_size))
        self.register_buffer('usage_frequency', torch.zeros(cache_size))
        self.register_buffer('last_access_time', torch.zeros(cache_size))
        self.register_buffer('global_time', torch.tensor(0))
    
    def update_quality_scores(self, indices: torch.Tensor, 
                            reconstruction_errors: torch.Tensor):
        """更新质量分数（基于重构误差）"""
        # 质量分数 = 1 / (1 + reconstruction_error)
        quality = 1.0 / (1.0 + reconstruction_errors)
        self.quality_scores[indices] = quality
        
        # 更新访问信息
        self.usage_frequency[indices] += 1
        self.last_access_time[indices] = self.global_time
        self.global_time += 1
    
    def compute_utility_scores(self, cache_len: int) -> torch.Tensor:
        """计算效用分数"""
        # 时间衰减因子
        time_decay = torch.pow(self.decay_factor, 
                              self.global_time - self.last_access_time[:cache_len])
        
        # 效用 = 质量 × 使用频率 × 时间衰减
        utility = (self.quality_scores[:cache_len] * 
                  self.usage_frequency[:cache_len] * 
                  time_decay)
        
        return utility
    
    def select_eviction_candidates(self, 
                                 keys: torch.Tensor, 
                                 values: torch.Tensor, 
                                 metadata: Dict[str, torch.Tensor]) -> torch.Tensor:
        """QUEST淘汰策略：基于效用分数"""
        cache_len = keys.size(0)
        
        if cache_len <= self.cache_size:
            return torch.zeros(cache_len, dtype=torch.bool, device=keys.device)
        
        # 计算效用分数
        utility_scores = self.compute_utility_scores(cache_len)
        
        # 淘汰效用分数低于阈值的项
        evict_mask = utility_scores < self.quality_threshold
        
        # 如果需要淘汰的项太少，选择效用最低的项
        num_to_evict = cache_len - self.cache_size
        if evict_mask.sum() < num_to_evict:
            _, lowest_indices = torch.topk(utility_scores, num_to_evict, largest=False)
            evict_mask = torch.zeros(cache_len, dtype=torch.bool, device=keys.device)
            evict_mask[lowest_indices] = True
        
        return evict_mask

class FlexGenScheduler(BaseScheduler):
    """
    FlexGen 调度器
    基于内存层次结构的自适应缓存管理
    """
    def __init__(self, cache_size: int, hidden_size: int,
                 gpu_ratio: float = 0.3, cpu_ratio: float = 0.5):
        super(FlexGenScheduler, self).__init__(cache_size, hidden_size)
        self.gpu_ratio = gpu_ratio  # GPU缓存比例
        self.cpu_ratio = cpu_ratio  # CPU缓存比例
        
        # 分层缓存大小
        self.gpu_cache_size = int(cache_size * gpu_ratio)
        self.cpu_cache_size = int(cache_size * cpu_ratio)
        self.disk_cache_size = cache_size - self.gpu_cache_size - self.cpu_cache_size
        
        # 访问频率和成本
        self.register_buffer('access_frequency', torch.zeros(cache_size))
        self.register_buffer('access_cost', torch.ones(cache_size))  # 访问成本
        
        # 层级标记 (0: GPU, 1: CPU, 2: Disk)
        self.register_buffer('cache_level', torch.zeros(cache_size, dtype=torch.long))
    
    def update_access_patterns(self, indices: torch.Tensor, access_costs: torch.Tensor):
        """更新访问模式"""
        self.access_frequency[indices] += 1
        self.access_cost[indices] = access_costs
    
    def select_eviction_candidates(self, 
                                 keys: torch.Tensor, 
                                 values: torch.Tensor, 
                                 metadata: Dict[str, torch.Tensor]) -> torch.Tensor:
        """FlexGen淘汰策略：基于访问频率和成本"""
        cache_len = keys.size(0)
        
        if cache_len <= self.cache_size:
            return torch.zeros(cache_len, dtype=torch.bool, device=keys.device)
        
        # 计算效益分数 = 访问频率 / 访问成本
        benefit_scores = self.access_frequency[:cache_len] / self.access_cost[:cache_len]
        
        # 根据效益分数分配到不同层级
        num_to_evict = cache_len - self.cache_size
        _, lowest_indices = torch.topk(benefit_scores, num_to_evict, largest=False)
        
        evict_mask = torch.zeros(cache_len, dtype=torch.bool, device=keys.device)
        evict_mask[lowest_indices] = True
        
        return evict_mask

class LRUScheduler(BaseScheduler):
    """
    LRU (Least Recently Used) 调度器
    """
    def __init__(self, cache_size: int, hidden_size: int):
        super(LRUScheduler, self).__init__(cache_size, hidden_size)
        self.register_buffer('access_times', torch.zeros(cache_size))
        self.register_buffer('global_time', torch.tensor(0))
    
    def update_access_time(self, indices: torch.Tensor):
        """更新访问时间"""
        self.access_times[indices] = self.global_time
        self.global_time += 1
    
    def select_eviction_candidates(self, 
                                 keys: torch.Tensor, 
                                 values: torch.Tensor, 
                                 metadata: Dict[str, torch.Tensor]) -> torch.Tensor:
        """LRU淘汰策略：淘汰最久未使用的项"""
        cache_len = keys.size(0)
        
        if cache_len <= self.cache_size:
            return torch.zeros(cache_len, dtype=torch.bool, device=keys.device)
        
        # 找到最久未使用的项
        num_to_evict = cache_len - self.cache_size
        _, oldest_indices = torch.topk(self.access_times[:cache_len], 
                                     num_to_evict, largest=False)
        
        evict_mask = torch.zeros(cache_len, dtype=torch.bool, device=keys.device)
        evict_mask[oldest_indices] = True
        
        return evict_mask

class LRUPlusScheduler(BaseScheduler):
    """
    LRU++ 调度器
    改进的LRU，考虑访问频率和重要性
    """
    def __init__(self, cache_size: int, hidden_size: int, 
                 frequency_weight: float = 0.3, importance_weight: float = 0.4):
        super(LRUPlusScheduler, self).__init__(cache_size, hidden_size)
        self.frequency_weight = frequency_weight
        self.importance_weight = importance_weight
        self.recency_weight = 1.0 - frequency_weight - importance_weight
        
        self.register_buffer('access_times', torch.zeros(cache_size))
        self.register_buffer('access_frequency', torch.zeros(cache_size))
        self.register_buffer('importance_scores', torch.zeros(cache_size))
        self.register_buffer('global_time', torch.tensor(0))
    
    def update_access_info(self, indices: torch.Tensor, importance: torch.Tensor):
        """更新访问信息"""
        self.access_times[indices] = self.global_time
        self.access_frequency[indices] += 1
        self.importance_scores[indices] = importance
        self.global_time += 1
    
    def compute_priority_scores(self, cache_len: int) -> torch.Tensor:
        """计算优先级分数"""
        # 归一化各个因子
        max_time = self.access_times[:cache_len].max()
        recency = self.access_times[:cache_len] / (max_time + 1e-8)
        
        max_freq = self.access_frequency[:cache_len].max()
        frequency = self.access_frequency[:cache_len] / (max_freq + 1e-8)
        
        importance = self.importance_scores[:cache_len]
        
        # 综合分数
        priority = (self.recency_weight * recency + 
                   self.frequency_weight * frequency + 
                   self.importance_weight * importance)
        
        return priority
    
    def select_eviction_candidates(self, 
                                 keys: torch.Tensor, 
                                 values: torch.Tensor, 
                                 metadata: Dict[str, torch.Tensor]) -> torch.Tensor:
        """LRU++淘汰策略：基于综合优先级分数"""
        cache_len = keys.size(0)
        
        if cache_len <= self.cache_size:
            return torch.zeros(cache_len, dtype=torch.bool, device=keys.device)
        
        # 计算优先级分数
        priority_scores = self.compute_priority_scores(cache_len)
        
        # 淘汰优先级最低的项
        num_to_evict = cache_len - self.cache_size
        _, lowest_indices = torch.topk(priority_scores, num_to_evict, largest=False)
        
        evict_mask = torch.zeros(cache_len, dtype=torch.bool, device=keys.device)
        evict_mask[lowest_indices] = True
        
        return evict_mask

class CacheSchedulingManager(nn.Module):
    """
    缓存调度管理器
    统一管理多种调度策略
    """
    def __init__(self, cache_size: int, hidden_size: int, 
                 policy: SchedulingPolicy = SchedulingPolicy.NONE):
        super(CacheSchedulingManager, self).__init__()
        self.cache_size = cache_size
        self.hidden_size = hidden_size
        self.policy = policy
        
        # 创建调度器
        self.scheduler = self._create_scheduler(policy)
        
        # 缓存状态
        self.register_buffer('cache_keys', torch.zeros(cache_size, hidden_size))
        self.register_buffer('cache_values', torch.zeros(cache_size, hidden_size))
        self.register_buffer('cache_valid', torch.zeros(cache_size, dtype=torch.bool))
        self.register_buffer('cache_size_current', torch.tensor(0))
    
    def _create_scheduler(self, policy: SchedulingPolicy) -> Optional[BaseScheduler]:
        """创建指定的调度器"""
        if policy == SchedulingPolicy.NONE:
            return None
        elif policy == SchedulingPolicy.H2O:
            return H2OScheduler(self.cache_size, self.hidden_size)
        elif policy == SchedulingPolicy.STREAMING_LLM:
            return StreamingLLMScheduler(self.cache_size, self.hidden_size)
        elif policy == SchedulingPolicy.QUEST:
            return QUESTScheduler(self.cache_size, self.hidden_size)
        elif policy == SchedulingPolicy.FLEXGEN:
            return FlexGenScheduler(self.cache_size, self.hidden_size)
        elif policy == SchedulingPolicy.LRU:
            return LRUScheduler(self.cache_size, self.hidden_size)
        elif policy == SchedulingPolicy.LRU_PLUS:
            return LRUPlusScheduler(self.cache_size, self.hidden_size)
        else:
            raise ValueError(f"Unknown scheduling policy: {policy}")
    
    def update_cache(self, keys: torch.Tensor, values: torch.Tensor, 
                    metadata: Optional[Dict[str, torch.Tensor]] = None):
        """更新缓存"""
        if self.scheduler is None:
            return  # 无调度策略时直接返回
        
        batch_size, seq_len, hidden_size = keys.shape
        
        # 展平输入
        keys_flat = keys.view(-1, hidden_size)
        values_flat = values.view(-1, hidden_size)
        
        # 检查是否需要淘汰
        new_items = keys_flat.size(0)
        total_size = self.cache_size_current.item() + new_items
        
        if total_size > self.cache_size:
            # 需要淘汰
            current_keys = self.cache_keys[:self.cache_size_current.item()]
            current_values = self.cache_values[:self.cache_size_current.item()]
            
            # 选择淘汰候选项
            evict_mask = self.scheduler.select_eviction_candidates(
                current_keys, current_values, metadata or {}
            )
            
            # 执行淘汰
            keep_mask = ~evict_mask
            keep_indices = torch.where(keep_mask)[0]
            
            # 压缩缓存
            self.cache_keys[:len(keep_indices)] = current_keys[keep_indices]
            self.cache_values[:len(keep_indices)] = current_values[keep_indices]
            self.cache_size_current.fill_(len(keep_indices))
            
            # 更新统计
            self.scheduler.eviction_count += evict_mask.sum()
        
        # 添加新项
        start_idx = self.cache_size_current.item()
        end_idx = min(start_idx + new_items, self.cache_size)
        actual_new_items = end_idx - start_idx
        
        self.cache_keys[start_idx:end_idx] = keys_flat[:actual_new_items]
        self.cache_values[start_idx:end_idx] = values_flat[:actual_new_items]
        self.cache_size_current.fill_(end_idx)
    
    def get_cache_stats(self) -> Dict[str, float]:
        """获取缓存统计信息"""
        stats = {
            'cache_size': self.cache_size_current.item(),
            'cache_utilization': self.cache_size_current.item() / self.cache_size,
            'policy': self.policy.value
        }
        
        if self.scheduler is not None:
            stats.update({
                'hit_rate': self.scheduler.get_hit_rate(),
                'eviction_count': self.scheduler.eviction_count.item()
            })
        
        return stats
    
    def reset_cache(self):
        """重置缓存"""
        self.cache_keys.zero_()
        self.cache_values.zero_()
        self.cache_valid.zero_()
        self.cache_size_current.zero_()
        
        if self.scheduler is not None:
            self.scheduler.reset_stats()
    
    def change_policy(self, new_policy: SchedulingPolicy):
        """更改调度策略"""
        self.policy = new_policy
        self.scheduler = self._create_scheduler(new_policy)
        self.reset_cache() 