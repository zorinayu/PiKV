#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LongBench 测试示例脚本

本脚本展示了如何使用LongBench对PiKV模型进行测试的简单示例。
"""

import os
import argparse
import sys

# 获取脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 确保能导入当前目录下的模块
sys.path.append(SCRIPT_DIR)

from run_longbench import evaluate_model
from analyze_results import analyze_domain_performance, analyze_by_context_length, load_results

def example_test(model_name, test_type="standard"):
    """运行测试示例"""
    print(f"=====================================================")
    print(f"开始对模型 {model_name} 进行 {test_type} 测试")
    print(f"=====================================================")
    
    # 检查配置
    config_dir = os.path.join(SCRIPT_DIR, "config")
    config_file = os.path.join(config_dir, "models.json")
    if not os.path.exists(config_file):
        print(f"错误: 配置文件不存在，请确保 {config_file} 文件已创建")
        return
    
    # 创建必要的目录
    results_dir = os.path.join(SCRIPT_DIR, "results")
    plots_dir = os.path.join(SCRIPT_DIR, "plots")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    print(f"配置文件检查通过，开始测试...")
    
    try:
        # 运行评估
        results = evaluate_model(model_name, test_type)
        
        # 显示简要分析
        print("\n结果摘要:")
        print(f"总体准确率: {results['accuracy']:.4f}")
        print("\n前三个领域表现:")
        domain_df = analyze_domain_performance(results)
        print(domain_df.head(3))
        
        print("\n🎉 测试完成! 您可以使用 analyze_results.py 进行更详细的分析。")
        print(f"例如: python {os.path.join(SCRIPT_DIR, 'analyze_results.py')} --model {model_name} --test_type {test_type} --plot")
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="LongBench测试示例")
    parser.add_argument("--model", type=str, default="pikv", help="要测试的模型名称")
    parser.add_argument("--test_type", type=str, default="standard", 
                        choices=["standard", "cot", "no_context", "rag"],
                        help="测试类型")
    parser.add_argument("--api_url", type=str, default=None,
                      help="模型API端点URL，默认为http://localhost:8000/v1/completions")
    parser.add_argument("--api_key", type=str, default=None,
                      help="API密钥，默认为token-abc123")
    
    args = parser.parse_args()
    
    # 运行示例测试
    example_test(args.model, args.test_type)

if __name__ == "__main__":
    main() 