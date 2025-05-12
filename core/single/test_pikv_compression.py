import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import os
import sys

# 添加路径以导入所需模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from single.pikv_compression import PyramidCompressor, SVDCompressor, QuantizedCompressor, PiKVCompressor
from single.config import config

class CompressionTester:
    """
    测试KV缓存压缩器的性能和质量
    """
    def __init__(self, hidden_size: int = 256):
        self.hidden_size = hidden_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = []
        
        # 创建输出目录
        os.makedirs("results", exist_ok=True)
    
    def generate_test_data(
        self, 
        batch_size: int = 8, 
        seq_len: int = 128,
        importance_level: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """生成测试数据"""
        keys = torch.randn(batch_size, seq_len, self.hidden_size, device=self.device)
        values = torch.randn(batch_size, seq_len, self.hidden_size, device=self.device)
        
        # 生成重要性分数 - 从均匀分布生成，或者根据importance_level设置固定值
        if importance_level < 0:  # 随机重要性
            importance = torch.rand(batch_size, seq_len, device=self.device)
        else:  # 固定重要性
            importance = torch.ones(batch_size, seq_len, device=self.device) * importance_level
            
        return keys, values, importance
    
    def measure_compression_performance(
        self, 
        compressor: nn.Module,
        keys: torch.Tensor,
        values: torch.Tensor,
        importance: Optional[torch.Tensor] = None,
        repeat: int = 10
    ) -> Dict[str, float]:
        """测量压缩性能和质量"""
        # 准备结果字典
        results = {}
        
        # 缓存一份原始数据用于计算精度损失
        keys_cpu = keys.detach().cpu()
        values_cpu = values.detach().cpu()
        
        # 预热
        for _ in range(3):
            with torch.no_grad():
                _ = compressor(keys, values, importance)
        
        # 测量压缩时间
        start_time = time.time()
        for _ in range(repeat):
            with torch.no_grad():
                compressed_keys, compressed_values = compressor(keys, values, importance)
        compression_time = (time.time() - start_time) / repeat
        
        # 计算压缩比和内存占用
        original_size = keys.element_size() * keys.nelement() + values.element_size() * values.nelement()
        compressed_size = compressed_keys.element_size() * compressed_keys.nelement() + compressed_values.element_size() * compressed_values.nelement()
        compression_ratio = compressed_size / original_size
        memory_saved = 1.0 - compression_ratio
        
        # 计算压缩质量 (MSE)
        with torch.no_grad():
            key_mse = F.mse_loss(compressed_keys.cpu(), keys_cpu).item()
            value_mse = F.mse_loss(compressed_values.cpu(), values_cpu).item()
            avg_mse = (key_mse + value_mse) / 2
            
            # 计算相似度 (余弦相似度)
            keys_flat = keys_cpu.reshape(-1, self.hidden_size)
            compressed_keys_flat = compressed_keys.cpu().reshape(-1, self.hidden_size)
            values_flat = values_cpu.reshape(-1, self.hidden_size)
            compressed_values_flat = compressed_values.cpu().reshape(-1, self.hidden_size)
            
            key_sim = F.cosine_similarity(keys_flat, compressed_keys_flat).mean().item()
            value_sim = F.cosine_similarity(values_flat, compressed_values_flat).mean().item()
            avg_sim = (key_sim + value_sim) / 2
        
        # 收集结果
        results["compression_time"] = compression_time
        results["compression_ratio"] = compression_ratio
        results["memory_saved"] = memory_saved
        results["key_mse"] = key_mse
        results["value_mse"] = value_mse
        results["avg_mse"] = avg_mse
        results["key_sim"] = key_sim
        results["value_sim"] = value_sim
        results["avg_sim"] = avg_sim
        
        # 添加来自压缩器的统计信息
        if hasattr(compressor, "get_compression_stats"):
            compressor_stats = compressor.get_compression_stats()
            for key, value in compressor_stats.items():
                if isinstance(value, (int, float)):
                    results[f"compressor_{key}"] = value
        
        return results
    
    def run_compression_test(
        self,
        compressor_name: str,
        compressor_params: Dict,
        batch_sizes: List[int] = [8],
        seq_lens: List[int] = [128],
        importance_levels: List[float] = [0.5],
        repeat: int = 10
    ) -> List[Dict]:
        """运行压缩测试"""
        test_results = []
        
        # 创建压缩器
        if compressor_name == "pyramid":
            compressor = PyramidCompressor(self.hidden_size, **compressor_params).to(self.device)
        elif compressor_name == "svd":
            compressor = SVDCompressor(self.hidden_size, **compressor_params).to(self.device)
        elif compressor_name == "quantized":
            compressor = QuantizedCompressor(self.hidden_size, **compressor_params).to(self.device)
        elif compressor_name == "pikv":
            compressor = PiKVCompressor(self.hidden_size, **compressor_params).to(self.device)
        else:
            raise ValueError(f"未知的压缩器类型: {compressor_name}")
        
        # 运行所有测试组合
        for batch_size in batch_sizes:
            for seq_len in seq_lens:
                for importance_level in importance_levels:
                    # 生成测试数据
                    keys, values, importance = self.generate_test_data(
                        batch_size=batch_size,
                        seq_len=seq_len,
                        importance_level=importance_level
                    )
                    
                    # 测量性能
                    result = self.measure_compression_performance(
                        compressor=compressor,
                        keys=keys,
                        values=values,
                        importance=importance,
                        repeat=repeat
                    )
                    
                    # 添加测试参数
                    result["compressor"] = compressor_name
                    result["batch_size"] = batch_size
                    result["seq_len"] = seq_len
                    result["importance_level"] = importance_level
                    result["hidden_size"] = self.hidden_size
                    
                    # 添加压缩器参数
                    for param_name, param_value in compressor_params.items():
                        result[f"param_{param_name}"] = param_value
                    
                    # 打印单次测试结果
                    print(f"\nTest {len(test_results)+1}:")
                    print(f"Compressor: {compressor_name}, Batch: {batch_size}, Seq: {seq_len}, Imp: {importance_level:.2f}")
                    print(f"Compression Ratio: {result['compression_ratio']:.4f} (Memory Saved: {result['memory_saved']:.2%})")
                    print(f"Avg MSE: {result['avg_mse']:.6f}, Avg Similarity: {result['avg_sim']:.4f}")
                    print(f"Compression Time: {result['compression_time']*1000:.2f} ms")
                    
                    # 保存结果
                    test_results.append(result)
                    self.results.extend(test_results)
        
        return test_results
    
    def run_ablation_study(self):
        """运行消融实验，测试不同压缩参数的效果"""
        print("\n===== 开始PiKV压缩器消融实验 =====")
        
        # 测试PyramidCompressor的不同压缩率
        pyramid_compression_ratios = [0.9, 0.7, 0.5, 0.3, 0.1]
        for ratio in pyramid_compression_ratios:
            print(f"\nTesting PyramidCompressor with compression_ratio={ratio}")
            self.run_compression_test(
                compressor_name="pyramid",
                compressor_params={"compression_ratio": ratio, "num_levels": 3},
                importance_levels=[0.5]
            )
        
        # 测试SVDCompressor的不同秩
        svd_ranks = [int(self.hidden_size * r) for r in [0.8, 0.5, 0.25, 0.1]]
        for rank in svd_ranks:
            print(f"\nTesting SVDCompressor with rank={rank}")
            self.run_compression_test(
                compressor_name="svd",
                compressor_params={"rank": rank, "adaptive_rank": False},
                importance_levels=[0.5]
            )
        
        # 测试QuantizedCompressor的不同位宽
        quant_bits = [16, 8, 4]
        for bits in quant_bits:
            print(f"\nTesting QuantizedCompressor with bits={bits}")
            self.run_compression_test(
                compressor_name="quantized",
                compressor_params={"bits": bits, "dynamic_quantization": True},
                importance_levels=[0.5]
            )
        
        # 测试综合压缩器的不同配置
        compressor_types_options = [
            ["pyramid"],
            ["svd"],
            ["quantization"],
            ["pyramid", "svd"],
            ["pyramid", "quantization"],
            ["svd", "quantization"],
            ["pyramid", "svd", "quantization"]
        ]
        
        for compressor_types in compressor_types_options:
            print(f"\nTesting PiKVCompressor with compressor_types={compressor_types}")
            self.run_compression_test(
                compressor_name="pikv",
                compressor_params={"compressor_types": compressor_types},
                importance_levels=[0.2, 0.5, 0.8]  # 测试不同重要性级别
            )
        
        print("\n===== 完成PiKV压缩器消融实验 =====")
    
    def run_accuracy_vs_compression_test(self):
        """测试压缩率与准确率之间的关系"""
        print("\n===== 测试压缩率与准确率关系 =====")
        
        # 创建一系列压缩率和模拟准确率测试点
        compression_ratios = []
        accuracy_scores = []
        acceleration_rates = []
        
        # 测试PyramidCompressor
        pyramid_compression_ratios = [0.9, 0.7, 0.5, 0.3, 0.1]
        for ratio in pyramid_compression_ratios:
            # 创建压缩器
            compressor = PyramidCompressor(self.hidden_size, compression_ratio=ratio).to(self.device)
            
            # 生成测试数据
            keys, values, importance = self.generate_test_data(
                batch_size=16,
                seq_len=256,
                importance_level=0.5
            )
            
            # 测量性能
            result = self.measure_compression_performance(
                compressor=compressor,
                keys=keys,
                values=values, 
                importance=importance
            )
            
            # 记录压缩率
            compression_ratios.append(result["compression_ratio"])
            
            # 计算模拟的准确率损失
            # 假设准确率与相似度正相关（这里只是模拟）
            base_accuracy = 0.95  # 基准准确率
            accuracy_degradation = (1.0 - result["avg_sim"]) * 0.5  # 准确率下降比例
            accuracy = base_accuracy - accuracy_degradation
            accuracy_scores.append(accuracy)
            
            # 计算加速率（与压缩率关系）
            # 假设加速率与内存节省正相关
            acceleration = 1.0 + result["memory_saved"]  # 加速比例
            acceleration_rates.append(acceleration)
            
            print(f"压缩率: {result['compression_ratio']:.4f}, 模拟准确率: {accuracy:.4f}, 加速率: {acceleration:.2f}")
        
        # 绘制压缩率与准确率/加速率的关系
        self.plot_compression_vs_accuracy(compression_ratios, accuracy_scores, acceleration_rates)
        
        print("\n===== 完成压缩率与准确率测试 =====")
    
    def run_speed_benchmark(self):
        """测试不同压缩方法的速度基准"""
        print("\n===== 速度基准测试 =====")
        
        # 创建各种压缩器
        compressors = {
            "PyramidCompressor (0.5)": PyramidCompressor(self.hidden_size, compression_ratio=0.5),
            "SVDCompressor (r=64)": SVDCompressor(self.hidden_size, rank=64),
            "QuantizedCompressor (8-bit)": QuantizedCompressor(self.hidden_size, bits=8),
            "PiKVCompressor": PiKVCompressor(self.hidden_size)
        }
        
        # 移动到设备
        for name, compressor in compressors.items():
            compressors[name] = compressor.to(self.device)
        
        # 测试不同序列长度
        seq_lengths = [128, 256, 512, 1024, 2048]
        
        # 测试结果
        time_results = {name: [] for name in compressors.keys()}
        memory_results = {name: [] for name in compressors.keys()}
        
        for seq_len in seq_lengths:
            print(f"\n测试序列长度: {seq_len}")
            
            # 生成测试数据
            keys, values, importance = self.generate_test_data(
                batch_size=4,
                seq_len=seq_len,
                importance_level=0.5
            )
            
            # 测试每个压缩器
            for name, compressor in compressors.items():
                # 预热
                for _ in range(3):
                    with torch.no_grad():
                        _ = compressor(keys, values, importance)
                
                # 测量时间
                start_time = time.time()
                for _ in range(10):
                    with torch.no_grad():
                        compressed_keys, compressed_values = compressor(keys, values, importance)
                compression_time = (time.time() - start_time) / 10
                
                # 计算内存节省
                original_size = keys.element_size() * keys.nelement() + values.element_size() * values.nelement()
                compressed_size = compressed_keys.element_size() * compressed_keys.nelement() + compressed_values.element_size() * compressed_values.nelement()
                memory_saved = 1.0 - (compressed_size / original_size)
                
                # 记录结果
                time_results[name].append(compression_time * 1000)  # 毫秒
                memory_results[name].append(memory_saved * 100)  # 百分比
                
                print(f"{name}: {compression_time*1000:.2f} ms, 内存节省: {memory_saved:.2%}")
        
        # 绘制速度基准图
        self.plot_speed_benchmark(seq_lengths, time_results, memory_results)
        
        print("\n===== 完成速度基准测试 =====")
    
    def plot_compression_vs_accuracy(
        self, 
        compression_ratios: List[float], 
        accuracy_scores: List[float],
        acceleration_rates: List[float]
    ):
        """绘制压缩率与准确率/加速率的关系图"""
        plt.figure(figsize=(12, 8))
        
        # 创建两个y轴
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        
        # 绘制准确率曲线
        accuracy_line = ax1.plot(compression_ratios, accuracy_scores, 'b-o', label='准确率')
        ax1.set_xlabel('压缩率 (compressed size / original size)')
        ax1.set_ylabel('模拟准确率', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.set_ylim([0.7, 1.0])  # 调整准确率的y轴范围
        
        # 绘制加速率曲线
        acceleration_line = ax2.plot(compression_ratios, acceleration_rates, 'r-^', label='加速率')
        ax2.set_ylabel('加速率', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        # 添加图例
        lines = accuracy_line + acceleration_line
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc='lower right')
        
        plt.title('PiKV压缩率与准确率/加速率关系')
        plt.grid(True, alpha=0.3)
        
        # 保存图像
        plt.savefig("results/compression_vs_accuracy.png")
        plt.close()
    
    def plot_speed_benchmark(
        self, 
        seq_lengths: List[int], 
        time_results: Dict[str, List[float]],
        memory_results: Dict[str, List[float]]
    ):
        """绘制速度基准图"""
        plt.figure(figsize=(15, 10))
        
        # 创建子图
        plt.subplot(2, 1, 1)
        for name, times in time_results.items():
            plt.plot(seq_lengths, times, '-o', label=name)
        
        plt.xlabel('序列长度')
        plt.ylabel('压缩时间 (ms)')
        plt.title('不同压缩方法的压缩时间')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(2, 1, 2)
        for name, memory in memory_results.items():
            plt.plot(seq_lengths, memory, '-o', label=name)
        
        plt.xlabel('序列长度')
        plt.ylabel('内存节省 (%)')
        plt.title('不同压缩方法的内存节省')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig("results/speed_benchmark.png")
        plt.close()
    
    def save_results(self):
        """保存测试结果为CSV"""
        if not self.results:
            print("没有测试结果可保存")
            return
        
        # 将结果转换为DataFrame
        df = pd.DataFrame(self.results)
        
        # 保存到CSV
        csv_path = "results/compression_test_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"测试结果已保存到: {csv_path}")

def main():
    """主函数"""
    # 创建测试器
    tester = CompressionTester(hidden_size=config["hidden_size"])
    
    # 运行消融实验
    tester.run_ablation_study()
    
    # 测试压缩率与准确率关系
    tester.run_accuracy_vs_compression_test()
    
    # 运行速度基准测试
    tester.run_speed_benchmark()
    
    # 保存结果
    tester.save_results()

if __name__ == "__main__":
    main() 