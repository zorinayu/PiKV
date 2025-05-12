#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import time
import argparse
import requests
from tqdm import tqdm
from datasets import load_dataset
import json
import torch
import numpy as np

# 获取脚本所在目录的绝对路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_model_config(model_name):
    """加载模型配置"""
    config_path = os.path.join(SCRIPT_DIR, "config", "models.json")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件 {config_path} 不存在")
    
    with open(config_path, "r") as f:
        models_config = json.load(f)
    
    if model_name not in models_config:
        raise ValueError(f"模型 {model_name} 在配置文件中未找到")
    
    return models_config[model_name]

def load_longbench_data():
    """加载LongBench数据集"""
    try:
        # 尝试从Hugging Face加载
        dataset = load_dataset('THUDM/LongBench-v2', split='train')
        # 尝试获取数据集样本数
        samples_count = None
        try:
            if hasattr(dataset, "__len__"):
                samples_count = len(dataset)
                print(f"成功加载 LongBench-v2 数据集，共 {samples_count} 个样本")
            else:
                # 如果无法直接获取长度，尝试转换为列表并计数
                first_few = list(dataset.take(5))
                print(f"成功加载 LongBench-v2 数据集，已获取前 {len(first_few)} 个样本")
        except Exception as e:
            print(f"成功加载 LongBench-v2 数据集，但无法确定样本数: {e}")
        
        return dataset
    except Exception as e:
        print(f"从Hugging Face加载数据集失败: {e}")
        print("请确保您已安装datasets库并且有互联网连接")
        print("或者您可以从GitHub仓库手动下载数据")
        exit(1)

def format_prompt(example, test_type="standard"):
    """根据测试类型格式化提示"""
    question = example["question"]
    context = example["context"] if test_type != "no_context" else ""
    
    choices = "\n".join([
        f"A. {example['choice_A']}",
        f"B. {example['choice_B']}",
        f"C. {example['choice_C']}",
        f"D. {example['choice_D']}"
    ])
    
    if test_type == "cot":
        # 思维链提示
        prompt = f"""请根据以下上下文回答问题。逐步思考并给出答案。 上下文: {context} 问题: {question} 选项: {choices} 首先，让我逐步思考:"""
    else:
        # 标准提示
        prompt = f"""请根据以下上下文回答问题。从给定的选项中选择一个最合适的答案，只需回答选项字母（A、B、C或D）。 上下文: {context} 问题:
{question} 选项: {choices} 答案:"""
    
    return prompt

def call_model_api(prompt, model_config, api_url="http://localhost:8000/v1/completions", api_key="token-abc123"):
    """调用模型API进行推理"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    data = {
        "model": model_config["path"],
        "prompt": prompt,
        "max_tokens": 1024,
        "temperature": 0.1,
        "stop": ["</s>", "<|im_end|>"]
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["text"].strip()
    except Exception as e:
        print(f"API调用失败: {e}")
        return ""

def parse_answer(response, test_type="standard"):
    """解析模型响应，提取答案"""
    if test_type == "cot":
        # 对于思维链回答，尝试提取最后的答案
        answer_line = ""
        for line in response.split("\n"):
            if "答案" in line or "选择" in line or "我选" in line:
                answer_line = line
        
        # 从答案行中提取A/B/C/D
        options = ["A", "B", "C", "D"]
        for option in options:
            if option in answer_line:
                return option
        
        # 如果未找到明确答案，返回整个响应的最后一行
        last_line = response.strip().split("\n")[-1]
        for option in options:
            if option in last_line:
                return option
    else:
        # 标准回答，直接寻找A/B/C/D
        response = response.strip()
        options = ["A", "B", "C", "D"]
        for option in options:
            if response.startswith(option) or response == option:
                return option
    
    # 如果无法解析答案，返回空字符串
    return ""

def evaluate_model(model_name, test_type="standard", api_url=None, api_key=None):
    """评估模型在LongBench上的表现"""
    model_config = load_model_config(model_name)
    dataset = load_longbench_data()
    
    results = []
    correct = 0
    total = 0
    
    # 按领域分类的统计
    domain_stats = {}
    
    # 创建进度条
    try:
        # 首先尝试创建有总数的进度条
        if hasattr(dataset, "__len__"):
            pbar = tqdm(dataset, desc=f"评估 {model_name}", total=len(dataset))
        else:
            # 如果无法获取总数，则创建无总数的进度条
            pbar = tqdm(dataset, desc=f"评估 {model_name}")
    except:
        # 如果tqdm有问题，使用普通迭代
        print(f"开始评估 {model_name}...")
        pbar = dataset
    
    for idx, example in enumerate(pbar):
        # 准备提示
        prompt = format_prompt(example, test_type)
        
        # 调用模型API
        response = call_model_api(prompt, model_config, 
                                  api_url=api_url if api_url else "http://localhost:8000/v1/completions", 
                                  api_key=api_key if api_key else "token-abc123")
        
        # 解析答案
        predicted = parse_answer(response, test_type)
        correct_answer = example["answer"]
        
        # 记录结果
        is_correct = predicted == correct_answer
        if is_correct:
            correct += 1
        
        total += 1
        
        # 更新领域统计
        domain = example["domain"]
        if domain not in domain_stats:
            domain_stats[domain] = {"correct": 0, "total": 0}
        
        domain_stats[domain]["total"] += 1
        if is_correct:
            domain_stats[domain]["correct"] += 1
        
        # 保存单个示例结果
        results.append({
            "id": example["_id"],
            "domain": domain,
            "sub_domain": example["sub_domain"],
            "question": example["question"],
            "context_length": len(example["context"].split()),
            "correct_answer": correct_answer,
            "predicted": predicted,
            "is_correct": is_correct,
            "response": response
        })
        
        # 每10个样本输出一次中间结果
        if (idx + 1) % 10 == 0:
            current_accuracy = correct / total if total > 0 else 0
            print(f"已处理 {total} 个样本，当前准确率: {current_accuracy:.4f}")
    
    # 计算总体准确率
    accuracy = correct / total if total > 0 else 0
    
    # 计算各领域准确率
    domain_accuracy = {}
    for domain, stats in domain_stats.items():
        domain_accuracy[domain] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
    
    # 汇总结果
    evaluation_result = {
        "model": model_name,
        "test_type": test_type,
        "accuracy": accuracy,
        "domain_accuracy": domain_accuracy,
        "correct": correct,
        "total": total,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": results
    }
    
    # 保存结果
    results_dir = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    output_file = os.path.join(results_dir, f"{model_name}_{test_type}_results.json")
    
    with open(output_file, "w") as f:
        json.dump(evaluation_result, f, ensure_ascii=False, indent=2)
    
    print(f"\n评估完成! 结果已保存至 {output_file}")
    print(f"总体准确率: {accuracy:.4f} ({correct}/{total})")
    print("\n各领域准确率:")
    for domain, acc in domain_accuracy.items():
        print(f"  {domain}: {acc:.4f}")
    
    return evaluation_result

def main():
    parser = argparse.ArgumentParser(description="使用LongBench评估大语言模型的长文本理解能力")
    parser.add_argument("--model", type=str, required=True, help="要评估的模型名称")
    parser.add_argument("--test_type", type=str, default="standard", 
                        choices=["standard", "cot", "no_context", "rag"],
                        help="测试类型: standard(标准), cot(思维链), no_context(无上下文), rag(检索增强)")
    parser.add_argument("--api_url", type=str, default=None,
                        help="模型API端点URL，默认为http://localhost:8000/v1/completions")
    parser.add_argument("--api_key", type=str, default=None,
                        help="API密钥，默认为token-abc123")
    parser.add_argument("--sample_limit", type=int, default=None,
                        help="测试样本数量限制，默认为全部")
    
    args = parser.parse_args()
    
    # 运行评估
    evaluate_model(args.model, args.test_type, args.api_url, args.api_key)

if __name__ == "__main__":
    main() 