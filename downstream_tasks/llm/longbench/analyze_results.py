#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# 获取脚本所在目录的绝对路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_results(model_name, test_type="standard"):
    """加载测试结果文件"""
    result_file = os.path.join(SCRIPT_DIR, "results", f"{model_name}_{test_type}_results.json")
    
    if not os.path.exists(result_file):
        raise FileNotFoundError(f"结果文件 {result_file} 不存在，请先运行测试")
    
    with open(result_file, "r") as f:
        results = json.load(f)
    
    return results

def analyze_domain_performance(results):
    """分析各领域表现"""
    domain_accuracy = results["domain_accuracy"]
    
    # 创建DataFrame以便于分析
    df = pd.DataFrame([
        {"domain": domain, "accuracy": accuracy}
        for domain, accuracy in domain_accuracy.items()
    ])
    
    # 按准确率排序
    df_sorted = df.sort_values("accuracy", ascending=False)
    
    return df_sorted

def analyze_by_context_length(results):
    """根据上下文长度分析表现"""
    # 提取测试结果
    data = results["results"]
    
    # 根据上下文长度分组
    length_bins = [(0, 10000), (10000, 20000), (20000, 50000), (50000, 100000), (100000, float('inf'))]
    length_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    
    for item in data:
        context_length = item["context_length"]
        is_correct = item["is_correct"]
        
        for lower, upper in length_bins:
            if lower <= context_length < upper:
                bin_name = f"{lower//1000}k-{upper//1000}k" if upper != float('inf') else f"{lower//1000}k+"
                length_stats[bin_name]["total"] += 1
                if is_correct:
                    length_stats[bin_name]["correct"] += 1
    
    # 计算每个长度区间的准确率
    length_accuracy = {}
    for bin_name, stats in length_stats.items():
        if stats["total"] > 0:
            length_accuracy[bin_name] = stats["correct"] / stats["total"]
    
    # 转换为DataFrame
    df = pd.DataFrame([
        {"length_bin": bin_name, "accuracy": accuracy, "samples": length_stats[bin_name]["total"]}
        for bin_name, accuracy in length_accuracy.items()
    ])
    
    return df

def plot_domain_accuracy(df, model_name, test_type, output_file=None):
    """绘制各领域准确率图表"""
    plt.figure(figsize=(10, 6))
    ax = plt.barh(df["domain"], df["accuracy"], color="skyblue")
    
    # 添加数据标签
    for i, v in enumerate(df["accuracy"]):
        plt.text(v + 0.01, i, f"{v:.4f}", va="center")
    
    plt.xlim(0, 1.0)
    plt.xlabel("准确率")
    plt.ylabel("领域")
    plt.title(f"{model_name} ({test_type}) - 各领域准确率")
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file)
        print(f"领域准确率图表已保存至 {output_file}")
    else:
        plt.show()

def plot_length_accuracy(df, model_name, test_type, output_file=None):
    """绘制不同长度的准确率图表"""
    plt.figure(figsize=(10, 6))
    
    # 确保长度区间按正确顺序排序
    df["sort_key"] = df["length_bin"].apply(lambda x: int(x.split("k")[0]))
    df = df.sort_values("sort_key")
    
    ax = plt.bar(df["length_bin"], df["accuracy"], color="lightgreen")
    
    # 添加数据标签
    for i, v in enumerate(df["accuracy"]):
        plt.text(i, v + 0.01, f"{v:.4f}", ha="center")
    
    # 添加样本数量标签
    for i, (_, row) in enumerate(df.iterrows()):
        plt.text(i, 0.02, f"n={row['samples']}", ha="center", color="darkblue")
    
    plt.ylim(0, 1.0)
    plt.xlabel("上下文长度")
    plt.ylabel("准确率")
    plt.title(f"{model_name} ({test_type}) - 不同长度的准确率")
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file)
        print(f"长度准确率图表已保存至 {output_file}")
    else:
        plt.show()

def compare_models(model_names, test_type="standard"):
    """比较多个模型的表现"""
    if not model_names or len(model_names) < 2:
        print("至少需要两个模型才能进行比较")
        return None, None  # 返回None防止解包错误
    
    all_results = {}
    for model_name in model_names:
        try:
            results = load_results(model_name, test_type)
            all_results[model_name] = results
        except FileNotFoundError:
            print(f"警告: 无法找到模型 {model_name} 的结果")
    
    if len(all_results) < 2:
        print("至少需要两个有效的模型结果才能进行比较")
        return None, None  # 返回None防止解包错误
    
    # 比较总体准确率
    overall_comparison = pd.DataFrame([
        {"model": model_name, "accuracy": results["accuracy"]}
        for model_name, results in all_results.items()
    ])
    
    # 比较各领域准确率
    domains = set()
    for results in all_results.values():
        domains.update(results["domain_accuracy"].keys())
    
    domain_comparison = {domain: [] for domain in domains}
    
    for model_name, results in all_results.items():
        for domain in domains:
            accuracy = results["domain_accuracy"].get(domain, 0)
            domain_comparison[domain].append({"model": model_name, "accuracy": accuracy})
    
    # 转换为DataFrame
    domain_df = {
        domain: pd.DataFrame(data) 
        for domain, data in domain_comparison.items()
    }
    
    return overall_comparison, domain_df

def plot_model_comparison(overall_df, domain_dfs, test_type, output_file=None):
    """绘制模型比较图表"""
    # 确保有有效数据
    if overall_df is None or domain_dfs is None:
        print("没有有效的比较数据可供绘图")
        return
    
    # 绘制总体准确率比较
    plt.figure(figsize=(10, 6))
    ax = plt.bar(overall_df["model"], overall_df["accuracy"], color="skyblue")
    
    # 添加数据标签
    for i, v in enumerate(overall_df["accuracy"]):
        plt.text(i, v + 0.01, f"{v:.4f}", ha="center")
    
    plt.ylim(0, 1.0)
    plt.xlabel("模型")
    plt.ylabel("准确率")
    plt.title(f"模型比较 ({test_type}) - 总体准确率")
    plt.tight_layout()
    
    if output_file:
        plt.savefig(f"{output_file}_overall.png")
        print(f"总体准确率比较图表已保存至 {output_file}_overall.png")
    else:
        plt.show()
    
    # 绘制各领域准确率比较
    if domain_dfs and len(domain_dfs) > 0:
        num_domains = len(domain_dfs)
        fig, axes = plt.subplots(nrows=num_domains, figsize=(12, 4 * num_domains))
        
        if num_domains == 1:
            axes = [axes]
        
        for i, (domain, df) in enumerate(domain_dfs.items()):
            ax = axes[i]
            ax.bar(df["model"], df["accuracy"], color="lightgreen")
            
            # 添加数据标签
            for j, v in enumerate(df["accuracy"]):
                ax.text(j, v + 0.01, f"{v:.4f}", ha="center")
            
            ax.set_ylim(0, 1.0)
            ax.set_xlabel("模型")
            ax.set_ylabel("准确率")
            ax.set_title(f"领域: {domain}")
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(f"{output_file}_domains.png")
            print(f"领域准确率比较图表已保存至 {output_file}_domains.png")
        else:
            plt.show()

def main():
    parser = argparse.ArgumentParser(description="分析LongBench测试结果")
    parser.add_argument("--model", type=str, required=True, help="要分析的模型名称")
    parser.add_argument("--test_type", type=str, default="standard", help="测试类型")
    parser.add_argument("--output_file", type=str, default=None, help="输出JSON文件路径")
    parser.add_argument("--compare", nargs="+", default=[], help="要比较的其他模型")
    parser.add_argument("--plot", action="store_true", help="生成可视化图表")
    
    args = parser.parse_args()
    
    # 确保结果目录存在
    results_dir = os.path.join(SCRIPT_DIR, "results")
    plots_dir = os.path.join(SCRIPT_DIR, "plots")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    try:
        # 加载结果
        results = load_results(args.model, args.test_type)
        
        # 分析各领域表现
        domain_df = analyze_domain_performance(results)
        print("\n各领域准确率:")
        print(domain_df)
        
        # 分析不同长度的表现
        length_df = analyze_by_context_length(results)
        print("\n不同长度的准确率:")
        print(length_df)
        
        # 绘制图表
        if args.plot:
            plot_domain_accuracy(
                domain_df, args.model, args.test_type,
                output_file=os.path.join(plots_dir, f"{args.model}_{args.test_type}_domains.png")
            )
            
            plot_length_accuracy(
                length_df, args.model, args.test_type,
                output_file=os.path.join(plots_dir, f"{args.model}_{args.test_type}_lengths.png")
            )
        
        # 比较模型
        if args.compare:
            compare_models_list = [args.model] + args.compare
            overall_comparison, domain_comparison = compare_models(compare_models_list, args.test_type)
            
            if overall_comparison is not None:
                print("\n模型比较 - 总体准确率:")
                print(overall_comparison)
                
                if args.plot:
                    plot_model_comparison(
                        overall_comparison, domain_comparison, args.test_type,
                        output_file=os.path.join(plots_dir, f"comparison_{args.test_type}")
                    )
        
        # 导出结果
        if args.output_file:
            output_path = args.output_file
            if not os.path.isabs(output_path):
                output_path = os.path.join(results_dir, output_path)
                
            summary = {
                "model": args.model,
                "test_type": args.test_type,
                "accuracy": results["accuracy"],
                "domain_accuracy": results["domain_accuracy"],
                "length_performance": length_df.to_dict(orient="records")
            }
            
            with open(output_path, "w") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            
            print(f"\n分析结果已保存至 {output_path}")
    
    except Exception as e:
        print(f"分析过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 