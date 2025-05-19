import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
from enum import IntEnum
from typing import Dict, List, Tuple, Optional, Union, Any

# 加载CUDA内核库
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    print("警告: CuPy未安装，将使用PyTorch实现替代CUDA内核")

# 压缩模式枚举
class CompressionMode(IntEnum):
    NONE = 0
    LORA = 1
    QUANT8 = 2
    MASK = 3

# 淘汰策略枚举
class EvictionPolicy(IntEnum):
    SLIDING = 0
    QUEST = 1

class PiKVKernels:
    """
    PiKV CUDA内核的Python接口
    包装三个核心内核：路由(Routing)、压缩(Compression)和淘汰(Eviction)
    """
    def __init__(self, cuda_lib_path=None):
        # CUDA库路径
        if cuda_lib_path is None:
            cuda_lib_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'cuda', 'libpikv_kernels.so'
            )
        
        self.cuda_enabled = CUPY_AVAILABLE and os.path.exists(cuda_lib_path)
        
        if self.cuda_enabled:
            # 加载自定义CUDA库
            self.lib = cp.cuda.runtime.linkModule(cuda_lib_path)
            print(f"成功加载PiKV CUDA内核库: {cuda_lib_path}")
        else:
            print(f"无法加载CUDA内核库，将使用PyTorch实现")
            self.lib = None
    
    def top_k_routing(
        self,
        routing_logits: torch.Tensor,
        k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        执行TopK路由
        
        Args:
            routing_logits: 路由逻辑张量 [batch_size, num_experts]
            k: 最大专家数
        
        Returns:
            top_k_indices: TopK专家索引 [batch_size, k]
            top_k_values: TopK专家分数 [batch_size, k]
        """
        batch_size, num_experts = routing_logits.shape
        
        if self.cuda_enabled:
            # 使用CUDA内核实现
            top_k_indices = torch.zeros((batch_size, k), dtype=torch.int32, device=routing_logits.device)
            top_k_values = torch.zeros((batch_size, k), dtype=torch.float32, device=routing_logits.device)
            
            # 转换为CuPy数组以便调用CUDA内核
            cp_routing_logits = cp.asarray(routing_logits)
            cp_top_k_indices = cp.asarray(top_k_indices)
            cp_top_k_values = cp.asarray(top_k_values)
            
            # 调用CUDA内核
            self.lib.LaunchTopKRoutingKernel(
                cp_routing_logits.data.ptr,
                cp_top_k_indices.data.ptr,
                cp_top_k_values.data.ptr,
                num_experts,
                batch_size,
                k
            )
            
            # 转回PyTorch张量
            top_k_indices = torch.as_tensor(cp.asnumpy(cp_top_k_indices), device=routing_logits.device)
            top_k_values = torch.as_tensor(cp.asnumpy(cp_top_k_values), device=routing_logits.device)
        else:
            # PyTorch实现替代
            top_k_values, top_k_indices = torch.topk(routing_logits, k, dim=1)
        
        return top_k_indices, top_k_values
    
    def compress_kv(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        compression_mode: CompressionMode,
        lora_params: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        压缩KV缓存
        
        Args:
            keys: 键张量 [batch_size, hidden_dim]
            values: 值张量 [batch_size, hidden_dim]
            compression_mode: 压缩模式
            lora_params: LoRA模式下的参数，包括A和B矩阵
        
        Returns:
            compressed_keys: 压缩后的键
            compressed_values: 压缩后的值
            meta_data: 压缩元数据
        """
        batch_size, hidden_dim = keys.shape
        meta_data = {}
        
        # 默认的LoRA参数
        default_rank = max(int(hidden_dim * 0.1), 1)  # 默认秩为隐藏维度的10%
        lora_rank = default_rank
        
        # 准备LoRA参数
        if compression_mode == CompressionMode.LORA:
            if lora_params is None:
                # 创建随机LoRA参数
                lora_a_k = torch.randn((hidden_dim, lora_rank), dtype=torch.float32, device=keys.device) * 0.1
                lora_b_k = torch.zeros((lora_rank, hidden_dim), dtype=torch.float32, device=keys.device)
                lora_a_v = torch.randn((hidden_dim, lora_rank), dtype=torch.float32, device=values.device) * 0.1
                lora_b_v = torch.zeros((lora_rank, hidden_dim), dtype=torch.float32, device=values.device)
            else:
                lora_a_k = lora_params.get('lora_a_k')
                lora_b_k = lora_params.get('lora_b_k')
                lora_a_v = lora_params.get('lora_a_v')
                lora_b_v = lora_params.get('lora_b_v')
                lora_rank = lora_a_k.shape[1]
        else:
            # 占位符
            lora_a_k = torch.zeros((1, 1), dtype=torch.float32, device=keys.device)
            lora_b_k = torch.zeros((1, 1), dtype=torch.float32, device=keys.device)
            lora_a_v = torch.zeros((1, 1), dtype=torch.float32, device=values.device)
            lora_b_v = torch.zeros((1, 1), dtype=torch.float32, device=values.device)
        
        # 元数据空间（用于量化参数等）
        meta_tensor = torch.zeros((batch_size * 4), dtype=torch.float32, device=keys.device)
        
        if self.cuda_enabled:
            # 分配输出空间
            compressed_keys = torch.zeros_like(keys)
            compressed_values = torch.zeros_like(values)
            
            # 转换为CuPy数组
            cp_keys = cp.asarray(keys)
            cp_values = cp.asarray(values)
            cp_compressed_keys = cp.asarray(compressed_keys)
            cp_compressed_values = cp.asarray(compressed_values)
            cp_lora_a_k = cp.asarray(lora_a_k)
            cp_lora_b_k = cp.asarray(lora_b_k)
            cp_lora_a_v = cp.asarray(lora_a_v)
            cp_lora_b_v = cp.asarray(lora_b_v)
            cp_meta_tensor = cp.asarray(meta_tensor)
            
            # 调用CUDA压缩内核
            self.lib.LaunchCompressKVKernel(
                cp_keys.data.ptr,
                cp_values.data.ptr,
                cp_compressed_keys.data.ptr,
                cp_compressed_values.data.ptr,
                cp_lora_a_k.data.ptr,
                cp_lora_b_k.data.ptr,
                cp_lora_a_v.data.ptr,
                cp_lora_b_v.data.ptr,
                cp_meta_tensor.data.ptr,
                hidden_dim,
                batch_size,
                int(compression_mode),
                lora_rank
            )
            
            # 转回PyTorch张量
            compressed_keys = torch.as_tensor(cp.asnumpy(cp_compressed_keys), device=keys.device)
            compressed_values = torch.as_tensor(cp.asnumpy(cp_compressed_values), device=values.device)
            meta_tensor = torch.as_tensor(cp.asnumpy(cp_meta_tensor), device=keys.device)
        else:
            # PyTorch替代实现
            if compression_mode == CompressionMode.LORA:
                # LoRA压缩
                compressed_keys = keys.clone()
                compressed_values = values.clone()
                
                # 执行LoRA变换
                for i in range(batch_size):
                    # K_hat = K + (K * A) * B
                    k_tmp = F.linear(keys[i:i+1], lora_a_k.t())  # [1, rank]
                    k_delta = F.linear(k_tmp, lora_b_k)  # [1, hidden_dim]
                    compressed_keys[i:i+1] = keys[i:i+1] + k_delta * (lora_rank / hidden_dim)
                    
                    # V_hat = V + (V * A) * B
                    v_tmp = F.linear(values[i:i+1], lora_a_v.t())  # [1, rank]
                    v_delta = F.linear(v_tmp, lora_b_v)  # [1, hidden_dim]
                    compressed_values[i:i+1] = values[i:i+1] + v_delta * (lora_rank / hidden_dim)
                
                meta_data = {
                    'compression_type': 'lora',
                    'lora_rank': lora_rank,
                    'lora_ratio': lora_rank / hidden_dim
                }
                
            elif compression_mode == CompressionMode.QUANT8:
                # 8位量化
                k_min = keys.min(dim=1, keepdim=True)[0]
                k_max = keys.max(dim=1, keepdim=True)[0]
                v_min = values.min(dim=1, keepdim=True)[0]
                v_max = values.max(dim=1, keepdim=True)[0]
                
                k_scale = (k_max - k_min) / 255.0
                v_scale = (v_max - v_min) / 255.0
                
                # 量化
                k_quant = torch.round((keys - k_min) / k_scale).clamp(0, 255).to(torch.uint8)
                v_quant = torch.round((values - v_min) / v_scale).clamp(0, 255).to(torch.uint8)
                
                # 反量化（为了演示，实际应用中通常保持量化状态）
                compressed_keys = k_quant.float() * k_scale + k_min
                compressed_values = v_quant.float() * v_scale + v_min
                
                # 保存元数据
                for i in range(batch_size):
                    meta_tensor[i*4] = k_scale[i, 0].item()
                    meta_tensor[i*4+1] = k_min[i, 0].item()
                    meta_tensor[i*4+2] = v_scale[i, 0].item()
                    meta_tensor[i*4+3] = v_min[i, 0].item()
                
                meta_data = {
                    'compression_type': 'quant8',
                    'k_scale': k_scale,
                    'k_min': k_min,
                    'v_scale': v_scale,
                    'v_min': v_min
                }
                
            elif compression_mode == CompressionMode.MASK:
                # 稀疏掩蔽
                threshold = 0.1  # 相对阈值
                
                compressed_keys = keys.clone()
                compressed_values = values.clone()
                
                # 计算平均幅度
                k_mag = keys.abs().mean(dim=1, keepdim=True)
                v_mag = values.abs().mean(dim=1, keepdim=True)
                
                # 创建掩码
                k_mask = keys.abs() >= (threshold * k_mag)
                v_mask = values.abs() >= (threshold * v_mag)
                
                # 应用掩码
                compressed_keys = compressed_keys * k_mask.float()
                compressed_values = compressed_values * v_mask.float()
                
                meta_data = {
                    'compression_type': 'mask',
                    'threshold': threshold,
                    'k_sparsity': 1.0 - k_mask.float().mean().item(),
                    'v_sparsity': 1.0 - v_mask.float().mean().item()
                }
                
            else:
                # 无压缩
                compressed_keys = keys.clone()
                compressed_values = values.clone()
                meta_data = {'compression_type': 'none'}
        
        # 如果使用CUDA实现，解析元数据
        if self.cuda_enabled and compression_mode == CompressionMode.QUANT8:
            meta_data = {
                'compression_type': 'quant8',
                'meta_tensor': meta_tensor
            }
        
        return compressed_keys, compressed_values, meta_data
    
    def evict_cache(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        timestamps: torch.Tensor,
        usage_counts: torch.Tensor,
        current_time: int,
        policy: EvictionPolicy,
        window_threshold: int = 1024,
        quest_threshold: float = 0.1
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        淘汰KV缓存条目
        
        Args:
            keys: 缓存键 [n, hidden_dim]
            values: 缓存值 [n, hidden_dim]
            timestamps: 时间戳 [n]
            usage_counts: 使用计数 [n]
            current_time: 当前时间戳
            policy: 淘汰策略
            window_threshold: 滑动窗口阈值
            quest_threshold: QUEST策略阈值
        
        Returns:
            new_keys: 压缩后的键
            new_values: 压缩后的值
            new_timestamps: 压缩后的时间戳
            new_usage_counts: 压缩后的使用计数
            new_size: 新缓存大小
        """
        n, hidden_dim = keys.shape
        
        if self.cuda_enabled:
            # 分配有效性标志和索引映射空间
            valid_flags = torch.ones(n, dtype=torch.int32, device=keys.device)
            new_indices = torch.zeros(n, dtype=torch.int32, device=keys.device)
            new_size = torch.zeros(1, dtype=torch.int32, device=keys.device)
            
            # 转换为CuPy数组
            cp_keys = cp.asarray(keys)
            cp_values = cp.asarray(values)
            cp_timestamps = cp.asarray(timestamps)
            cp_usage_counts = cp.asarray(usage_counts)
            cp_valid_flags = cp.asarray(valid_flags)
            
            # 第一步：标记无效条目
            self.lib.LaunchEvictKernel(
                cp_keys.data.ptr,
                cp_values.data.ptr,
                cp_timestamps.data.ptr,
                cp_usage_counts.data.ptr,
                cp_valid_flags.data.ptr,
                n,
                hidden_dim,
                current_time,
                window_threshold,
                quest_threshold,
                int(policy)
            )
            
            # 转回PyTorch
            valid_flags = torch.as_tensor(cp.asnumpy(cp_valid_flags), device=keys.device)
            
            # 第二步：压缩缓存
            cp_new_indices = cp.asarray(new_indices)
            cp_new_size = cp.asarray(new_size)
            
            # 创建输出缓冲区的副本
            cp_new_keys = cp.asarray(keys.clone())
            cp_new_values = cp.asarray(values.clone())
            cp_new_timestamps = cp.asarray(timestamps.clone())
            cp_new_usage_counts = cp.asarray(usage_counts.clone())
            
            self.lib.LaunchCompactKVCacheKernel(
                cp_new_keys.data.ptr,
                cp_new_values.data.ptr,
                cp_new_timestamps.data.ptr,
                cp_new_usage_counts.data.ptr,
                cp_valid_flags.data.ptr,
                cp_new_indices.data.ptr,
                cp_new_size.data.ptr,
                n,
                hidden_dim
            )
            
            # 获取新大小
            new_size_scalar = int(cp.asnumpy(cp_new_size)[0])
            
            # 转回PyTorch并裁剪到新大小
            new_keys = torch.as_tensor(cp.asnumpy(cp_new_keys)[:new_size_scalar], device=keys.device)
            new_values = torch.as_tensor(cp.asnumpy(cp_new_values)[:new_size_scalar], device=values.device)
            new_timestamps = torch.as_tensor(cp.asnumpy(cp_new_timestamps)[:new_size_scalar], device=timestamps.device)
            new_usage_counts = torch.as_tensor(cp.asnumpy(cp_new_usage_counts)[:new_size_scalar], device=usage_counts.device)
            
        else:
            # PyTorch替代实现
            valid_flags = torch.ones(n, dtype=torch.bool, device=keys.device)
            
            if policy == EvictionPolicy.SLIDING:
                # 滑动窗口策略
                valid_flags = (current_time - timestamps) <= window_threshold
            else:
                # QUEST策略
                activity_scores = torch.zeros(n, device=keys.device)
                
                # 计算每个条目的活跃度分数
                for i in range(n):
                    # 计算L2范数
                    key_norm = torch.norm(keys[i])
                    # 活跃度 = 范数 * 使用计数
                    activity_scores[i] = key_norm * usage_counts[i]
                
                # 根据阈值标记有效条目
                valid_flags = activity_scores >= quest_threshold
            
            # 压缩缓存
            new_size = torch.sum(valid_flags).item()
            new_keys = torch.zeros((new_size, hidden_dim), dtype=keys.dtype, device=keys.device)
            new_values = torch.zeros((new_size, hidden_dim), dtype=values.dtype, device=values.device)
            new_timestamps = torch.zeros(new_size, dtype=timestamps.dtype, device=timestamps.device)
            new_usage_counts = torch.zeros(new_size, dtype=usage_counts.dtype, device=usage_counts.device)
            
            # 复制有效条目
            idx = 0
            for i in range(n):
                if valid_flags[i]:
                    new_keys[idx] = keys[i]
                    new_values[idx] = values[i]
                    new_timestamps[idx] = timestamps[i]
                    new_usage_counts[idx] = usage_counts[i]
                    idx += 1
        
        return new_keys, new_values, new_timestamps, new_usage_counts, new_size

    def test_top_k_routing(self, batch_size=16, num_experts=32, k=4):
        """测试TopK路由内核"""
        print("\n=============================================")
        print("测试 TopK Routing 内核")
        print("=============================================")
        
        # 创建随机路由逻辑分数
        routing_logits = torch.rand((batch_size, num_experts), device='cuda')
        
        # 预热
        for _ in range(3):
            _ = self.top_k_routing(routing_logits, k)
        
        # 基准测试
        iterations = 100
        start_time = time.time()
        for _ in range(iterations):
            top_k_indices, top_k_values = self.top_k_routing(routing_logits, k)
        elapsed = time.time() - start_time
        
        # 验证结果
        torch_indices, torch_values = torch.topk(routing_logits, k, dim=1)
        
        # 检查结果
        indices_match = torch.allclose(torch_indices.float(), top_k_values)
        print(f"TopK路由内核测试完成:")
        print(f"  批次大小: {batch_size}, 专家数: {num_experts}, Top-K: {k}")
        print(f"  运行时间: {elapsed*1000/iterations:.2f} ms/批次")
        print(f"  结果验证: {'通过' if indices_match else '不匹配'}")
        
        return top_k_indices, top_k_values
    
    def test_compress_kv(self, batch_size=16, hidden_dim=256):
        """测试KV压缩内核"""
        print("\n=============================================")
        print("测试 Compress KV 内核")
        print("=============================================")
        
        # 创建随机KV对
        keys = torch.randn((batch_size, hidden_dim), device='cuda')
        values = torch.randn((batch_size, hidden_dim), device='cuda')
        
        # 测试所有压缩模式
        for mode in [CompressionMode.NONE, CompressionMode.LORA, CompressionMode.QUANT8, CompressionMode.MASK]:
            print(f"\n测试压缩模式: {mode.name}")
            
            # LoRA参数
            if mode == CompressionMode.LORA:
                lora_rank = max(int(hidden_dim * 0.1), 1)
                lora_params = {
                    'lora_a_k': torch.randn((hidden_dim, lora_rank), device='cuda') * 0.1,
                    'lora_b_k': torch.zeros((lora_rank, hidden_dim), device='cuda'),
                    'lora_a_v': torch.randn((hidden_dim, lora_rank), device='cuda') * 0.1,
                    'lora_b_v': torch.zeros((lora_rank, hidden_dim), device='cuda')
                }
            else:
                lora_params = None
            
            # 预热
            for _ in range(3):
                _ = self.compress_kv(keys, values, mode, lora_params)
            
            # 基准测试
            iterations = 100
            start_time = time.time()
            for _ in range(iterations):
                compressed_keys, compressed_values, meta_data = self.compress_kv(
                    keys, values, mode, lora_params
                )
            elapsed = time.time() - start_time
            
            # 计算压缩率
            compression_ratio = 1.0
            if mode != CompressionMode.NONE:
                original_size = keys.nelement() * keys.element_size() + values.nelement() * values.element_size()
                compressed_size = compressed_keys.nelement() * compressed_keys.element_size() + compressed_values.nelement() * compressed_values.element_size()
                compression_ratio = compressed_size / original_size
            
            # 计算误差
            if mode != CompressionMode.NONE:
                k_error = F.mse_loss(compressed_keys, keys).item()
                v_error = F.mse_loss(compressed_values, values).item()
                avg_error = (k_error + v_error) / 2
            else:
                avg_error = 0.0
            
            print(f"  运行时间: {elapsed*1000/iterations:.2f} ms/批次")
            print(f"  压缩率: {compression_ratio:.4f}")
            print(f"  平均误差: {avg_error:.6f}")
            print(f"  元数据: {meta_data}")
        
        return compressed_keys, compressed_values, meta_data
    
    def test_evict_cache(self, n=1024, hidden_dim=256):
        """测试缓存淘汰内核"""
        print("\n=============================================")
        print("测试 Evict Cache 内核")
        print("=============================================")
        
        # 创建随机缓存数据
        keys = torch.randn((n, hidden_dim), device='cuda')
        values = torch.randn((n, hidden_dim), device='cuda')
        timestamps = torch.randint(0, 2000, (n,), device='cuda')
        usage_counts = torch.rand(n, device='cuda')
        
        current_time = 2000  # 当前时间戳
        
        # 测试所有淘汰策略
        for policy in [EvictionPolicy.SLIDING, EvictionPolicy.QUEST]:
            print(f"\n测试淘汰策略: {policy.name}")
            
            # 参数
            window_threshold = 1000  # 仅保留最近1000个时间单位的条目
            quest_threshold = 0.2    # 活跃度阈值
            
            # 预热
            for _ in range(3):
                _ = self.evict_cache(
                    keys, values, timestamps, usage_counts,
                    current_time, policy, window_threshold, quest_threshold
                )
            
            # 基准测试
            iterations = 10
            start_time = time.time()
            for _ in range(iterations):
                new_keys, new_values, new_timestamps, new_usage_counts, new_size = self.evict_cache(
                    keys, values, timestamps, usage_counts,
                    current_time, policy, window_threshold, quest_threshold
                )
            elapsed = time.time() - start_time
            
            # 计算淘汰率
            eviction_ratio = 1.0 - (new_size / n)
            
            print(f"  运行时间: {elapsed*1000/iterations:.2f} ms")
            print(f"  缓存大小: {n} -> {new_size} (淘汰率: {eviction_ratio*100:.1f}%)")
            
            if policy == EvictionPolicy.SLIDING:
                # 验证滑动窗口策略
                expected_valid = (current_time - timestamps) <= window_threshold
                expected_size = torch.sum(expected_valid).item()
                print(f"  验证: 预期大小 {expected_size}, 实际大小 {new_size}")
            
        return new_keys, new_values, new_timestamps, new_usage_counts

def run_benchmarks():
    """运行所有基准测试"""
    # 初始化PiKV内核
    pikv_kernels = PiKVKernels()
    
    # 测试TopK路由
    pikv_kernels.test_top_k_routing(batch_size=32, num_experts=64, k=8)
    
    # 测试KV压缩
    pikv_kernels.test_compress_kv(batch_size=32, hidden_dim=512)
    
    # 测试缓存淘汰
    pikv_kernels.test_evict_cache(n=2048, hidden_dim=512)

if __name__ == "__main__":
    run_benchmarks() 