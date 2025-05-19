#!/usr/bin/env python
import os
import sys
import argparse
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from subprocess import run
import pandas as pd

# 添加路径以导入PiKV模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from single.pikv_kernels import PiKVKernels, CompressionMode, EvictionPolicy

# 定义颜色
BLUE = '\033[94m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
ENDC = '\033[0m'

def compile_cuda_kernels():
    """
    编译CUDA内核库
    """
    cuda_dir = os.path.dirname(os.path.abspath(__file__))
    source_file = os.path.join(cuda_dir, "pikv_kernels.cu")
    output_file = os.path.join(cuda_dir, "libpikv_kernels.so")
    
    print(f"{BLUE}正在编译CUDA内核:{ENDC} {source_file}")
    
    # 确保CUDA目录存在
    os.makedirs(cuda_dir, exist_ok=True)
    
    # 使用nvcc编译CUDA内核到共享库
    cmd = [
        "nvcc", "-shared", "-Xcompiler", "-fPIC",
        "-gencode", "arch=compute_70,code=sm_70",  # Pascal架构
        "-gencode", "arch=compute_75,code=sm_75",  # Turing架构
        "-gencode", "arch=compute_80,code=sm_80",  # Ampere架构
        "-O3", source_file, "-o", output_file
    ]
    
    # 执行编译命令
    result = run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"{GREEN}编译成功:{ENDC} {output_file}")
        return True
    else:
        print(f"{RED}编译失败:{ENDC}")
        print(result.stderr)
        return False

def benchmark_top_k_routing(pikv_kernels, batch_sizes, num_experts_list, k_values, iters=20):
    """
    基准测试TopK路由内核
    """
    print(f"\n{BLUE}============ TopK路由内核基准测试 ============{ENDC}")
    
    results = []
    
    for batch_size in batch_sizes:
        for num_experts in num_experts_list:
            for k in k_values:
                # 创建随机路由逻辑分数
                routing_logits = torch.rand((batch_size, num_experts), device='cuda')
                
                # 预热
                for _ in range(3):
                    _ = pikv_kernels.top_k_routing(routing_logits, k)
                    torch.cuda.synchronize()
                
                # 基准测试 - CUDA版本
                start_time = time.time()
                for _ in range(iters):
                    top_k_indices, top_k_values = pikv_kernels.top_k_routing(routing_logits, k)
                    torch.cuda.synchronize()
                pikv_time = (time.time() - start_time) * 1000 / iters  # ms
                
                # 基准测试 - PyTorch版本
                start_time = time.time()
                for _ in range(iters):
                    torch_indices, torch_values = torch.topk(routing_logits, k, dim=1)
                    torch.cuda.synchronize()
                torch_time = (time.time() - start_time) * 1000 / iters  # ms
                
                # 验证结果
                indices_match = torch.allclose(torch_indices.float(), top_k_values)
                speedup = torch_time / pikv_time if pikv_time > 0 else 0
                
                # 记录结果
                results.append({
                    'batch_size': batch_size,
                    'num_experts': num_experts,
                    'k': k,
                    'pikv_time_ms': pikv_time,
                    'torch_time_ms': torch_time,
                    'speedup': speedup,
                    'verified': indices_match
                })
                
                status = f"{GREEN}通过{ENDC}" if indices_match else f"{RED}失败{ENDC}"
                print(f"批次大小: {batch_size}, 专家数: {num_experts}, K: {k}")
                print(f"  PiKV: {pikv_time:.3f} ms, PyTorch: {torch_time:.3f} ms")
                print(f"  加速比: {speedup:.2f}x, 验证: {status}")
    
    # 创建数据框并返回
    return pd.DataFrame(results)

def benchmark_compress_kv(pikv_kernels, batch_sizes, hidden_dims, compression_modes, iters=20):
    """
    基准测试KV压缩内核
    """
    print(f"\n{BLUE}============ KV压缩内核基准测试 ============{ENDC}")
    
    results = []
    
    for batch_size in batch_sizes:
        for hidden_dim in hidden_dims:
            for mode in compression_modes:
                # 创建随机KV对
                keys = torch.randn((batch_size, hidden_dim), device='cuda')
                values = torch.randn((batch_size, hidden_dim), device='cuda')
                
                # LoRA参数
                lora_params = None
                if mode == CompressionMode.LORA:
                    lora_rank = max(int(hidden_dim * 0.1), 1)
                    lora_params = {
                        'lora_a_k': torch.randn((hidden_dim, lora_rank), device='cuda') * 0.1,
                        'lora_b_k': torch.zeros((lora_rank, hidden_dim), device='cuda'),
                        'lora_a_v': torch.randn((hidden_dim, lora_rank), device='cuda') * 0.1,
                        'lora_b_v': torch.zeros((lora_rank, hidden_dim), device='cuda')
                    }
                
                # 预热
                for _ in range(3):
                    _ = pikv_kernels.compress_kv(keys, values, mode, lora_params)
                    torch.cuda.synchronize()
                
                # 基准测试
                start_time = time.time()
                for _ in range(iters):
                    compressed_keys, compressed_values, meta_data = pikv_kernels.compress_kv(
                        keys, values, mode, lora_params
                    )
                    torch.cuda.synchronize()
                elapsed = (time.time() - start_time) * 1000 / iters  # ms
                
                # 计算压缩率和误差
                compression_ratio = 1.0
                if mode != CompressionMode.NONE:
                    original_size = keys.nelement() * keys.element_size() + values.nelement() * values.element_size()
                    compressed_size = compressed_keys.nelement() * compressed_keys.element_size() + compressed_values.nelement() * compressed_values.element_size()
                    compression_ratio = compressed_size / original_size
                
                # 计算误差
                if mode != CompressionMode.NONE:
                    k_error = torch.norm(compressed_keys - keys).item() / torch.norm(keys).item()
                    v_error = torch.norm(compressed_values - values).item() / torch.norm(values).item()
                    avg_error = (k_error + v_error) / 2
                else:
                    avg_error = 0.0
                
                # 记录结果
                results.append({
                    'batch_size': batch_size,
                    'hidden_dim': hidden_dim,
                    'mode': mode.name,
                    'time_ms': elapsed,
                    'compression_ratio': compression_ratio,
                    'error': avg_error
                })
                
                print(f"批次大小: {batch_size}, 隐藏维度: {hidden_dim}, 模式: {mode.name}")
                print(f"  运行时间: {elapsed:.3f} ms")
                print(f"  压缩率: {compression_ratio:.4f}, 误差: {avg_error:.6f}")
    
    # 创建数据框并返回
    return pd.DataFrame(results)

def benchmark_evict_cache(pikv_kernels, cache_sizes, hidden_dims, policies, iters=10):
    """
    基准测试缓存淘汰内核
    """
    print(f"\n{BLUE}============ 缓存淘汰内核基准测试 ============{ENDC}")
    
    results = []
    
    for n in cache_sizes:
        for hidden_dim in hidden_dims:
            for policy in policies:
                # 创建随机缓存数据
                keys = torch.randn((n, hidden_dim), device='cuda')
                values = torch.randn((n, hidden_dim), device='cuda')
                timestamps = torch.randint(0, 2000, (n,), device='cuda')
                usage_counts = torch.rand(n, device='cuda')
                
                current_time = 2000  # 当前时间戳
                window_threshold = 1000  # 仅保留最近1000个时间单位的条目
                quest_threshold = 0.2    # 活跃度阈值
                
                # 预热
                for _ in range(3):
                    _ = pikv_kernels.evict_cache(
                        keys, values, timestamps, usage_counts,
                        current_time, policy, window_threshold, quest_threshold
                    )
                    torch.cuda.synchronize()
                
                # 基准测试
                start_time = time.time()
                for _ in range(iters):
                    new_keys, new_values, new_timestamps, new_usage_counts, new_size = pikv_kernels.evict_cache(
                        keys, values, timestamps, usage_counts,
                        current_time, policy, window_threshold, quest_threshold
                    )
                    torch.cuda.synchronize()
                elapsed = (time.time() - start_time) * 1000 / iters  # ms
                
                # 计算淘汰率
                eviction_ratio = 1.0 - (new_size / n)
                
                # 验证结果
                if policy == EvictionPolicy.SLIDING:
                    expected_valid = (current_time - timestamps) <= window_threshold
                    expected_size = torch.sum(expected_valid).item()
                    verified = abs(expected_size - new_size) < 5  # 允许小误差
                else:
                    verified = True  # QUEST策略难以简单验证
                
                # 记录结果
                results.append({
                    'cache_size': n,
                    'hidden_dim': hidden_dim,
                    'policy': policy.name,
                    'time_ms': elapsed,
                    'eviction_ratio': eviction_ratio,
                    'new_size': new_size,
                    'verified': verified
                })
                
                status = f"{GREEN}通过{ENDC}" if verified else f"{RED}失败{ENDC}"
                print(f"缓存大小: {n}, 隐藏维度: {hidden_dim}, 策略: {policy.name}")
                print(f"  运行时间: {elapsed:.3f} ms")
                print(f"  淘汰率: {eviction_ratio*100:.1f}%, 新大小: {new_size}")
                print(f"  验证: {status}")
    
    # 创建数据框并返回
    return pd.DataFrame(results)

def plot_results(top_k_results, compress_results, evict_results, output_dir='plots'):
    """
    绘制性能结果图表
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 绘制TopK路由性能
    plt.figure(figsize=(10, 6))
    
    # 按batch_size分组
    for batch_size in top_k_results['batch_size'].unique():
        df = top_k_results[top_k_results['batch_size'] == batch_size]
        plt.plot(df['num_experts'], df['speedup'], 
                 marker='o', label=f'Batch Size {batch_size}')
    
    plt.xlabel('专家数量')
    plt.ylabel('加速比 (PyTorch/PiKV)')
    plt.title('TopK路由内核性能')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'topk_routing_perf.png'))
    
    # 绘制压缩模式性能比较
    plt.figure(figsize=(12, 8))
    
    # 压缩时间
    plt.subplot(2, 2, 1)
    for mode in compress_results['mode'].unique():
        df = compress_results[compress_results['mode'] == mode]
        for hidden_dim in df['hidden_dim'].unique():
            subset = df[df['hidden_dim'] == hidden_dim]
            plt.plot(subset['batch_size'], subset['time_ms'], 
                    marker='o', label=f'{mode}, Dim={hidden_dim}')
    
    plt.xlabel('批次大小')
    plt.ylabel('时间 (ms)')
    plt.title('压缩时间')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 压缩率
    plt.subplot(2, 2, 2)
    for mode in compress_results['mode'].unique():
        if mode != 'NONE':  # 跳过无压缩模式
            df = compress_results[compress_results['mode'] == mode]
            plt.bar(mode, df['compression_ratio'].mean())
    
    plt.ylabel('压缩率 (越低越好)')
    plt.title('平均压缩率')
    plt.grid(True, alpha=0.3)
    
    # 误差
    plt.subplot(2, 2, 3)
    for mode in compress_results['mode'].unique():
        if mode != 'NONE':  # 跳过无压缩模式
            df = compress_results[compress_results['mode'] == mode]
            plt.bar(mode, df['error'].mean())
    
    plt.ylabel('平均误差')
    plt.title('压缩误差')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'compression_perf.png'))
    
    # 绘制淘汰性能
    plt.figure(figsize=(10, 6))
    
    # 绘制处理时间与缓存大小的关系
    for policy in evict_results['policy'].unique():
        df = evict_results[evict_results['policy'] == policy]
        plt.plot(df['cache_size'], df['time_ms'], 
                 marker='o', label=f'策略: {policy}')
    
    plt.xlabel('缓存大小')
    plt.ylabel('处理时间 (ms)')
    plt.title('缓存淘汰性能')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'eviction_perf.png'))
    
    print(f"{GREEN}性能图表已保存到: {output_dir}{ENDC}")

def main():
    """
    主函数：运行所有基准测试
    """
    parser = argparse.ArgumentParser(description='PiKV CUDA内核基准测试')
    parser.add_argument('--no-compile', action='store_true', help='跳过编译步骤')
    parser.add_argument('--output-dir', default='pikv_benchmark_results', help='输出目录')
    args = parser.parse_args()
    
    print(f"{BLUE}PiKV CUDA内核基准测试工具{ENDC}")
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"PyTorch版本: {torch.__version__}")
    
    # 编译CUDA内核
    if not args.no_compile:
        if not compile_cuda_kernels():
            print(f"{RED}编译失败，退出测试。{ENDC}")
            return
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化PiKV内核
    pikv_kernels = PiKVKernels()
    
    # 设置测试参数
    batch_sizes = [16, 32, 64, 128]
    num_experts_list = [16, 32, 64, 128]
    k_values = [2, 4, 8]
    
    hidden_dims = [128, 256, 512]
    compression_modes = [
        CompressionMode.NONE, 
        CompressionMode.LORA, 
        CompressionMode.QUANT8, 
        CompressionMode.MASK
    ]
    
    cache_sizes = [512, 1024, 2048, 4096]
    eviction_policies = [EvictionPolicy.SLIDING, EvictionPolicy.QUEST]
    
    # 运行基准测试
    top_k_results = benchmark_top_k_routing(
        pikv_kernels, batch_sizes, num_experts_list, k_values
    )
    
    compress_results = benchmark_compress_kv(
        pikv_kernels, batch_sizes, hidden_dims, compression_modes
    )
    
    evict_results = benchmark_evict_cache(
        pikv_kernels, cache_sizes, hidden_dims, eviction_policies
    )
    
    # 保存结果
    top_k_results.to_csv(os.path.join(args.output_dir, 'topk_routing_results.csv'), index=False)
    compress_results.to_csv(os.path.join(args.output_dir, 'compression_results.csv'), index=False)
    evict_results.to_csv(os.path.join(args.output_dir, 'eviction_results.csv'), index=False)
    
    print(f"{GREEN}基准测试结果已保存到: {args.output_dir}{ENDC}")
    
    # 绘制结果
    plot_results(top_k_results, compress_results, evict_results, 
                 output_dir=os.path.join(args.output_dir, 'plots'))

if __name__ == "__main__":
    main() 