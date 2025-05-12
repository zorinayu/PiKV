# LongBench 测试

run
```sh
python downstream_tasks/llm/longbench/example.py --model pikv --test_type standard
```

本目录包含使用 [LongBench](https://github.com/THUDM/LongBench) 对 PiKV 模型进行长文本理解能力评估的代码。

## 简介

LongBench 是一个用于评估大型语言模型长文本理解能力的基准测试框架，由清华大学知识工程实验室开发。它包含多种长文本理解任务，涉及多个领域，包括：

- 单文档QA
- 多文档QA
- 长上下文学习
- 长对话历史理解
- 代码仓库理解
- 长结构化数据理解

该测试可以评估模型在处理长文本时的深度理解和推理能力。

## 安装要求

在使用本测试前，请确保已安装以下依赖：

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 准备模型

首先使用 vLLM 部署要测试的模型：

```bash
vllm serve [MODEL_PATH] --api-key token-abc123 --tensor-parallel-size 4 --gpu-memory-utilization 0.95 --max_model_len 131072 --trust-remote-code
```

根据您的模型和硬件情况调整以下参数：
- `--tensor-parallel-size`：张量并行度，较大模型需要更高的值
- `--gpu-memory-utilization`：GPU内存利用率
- `--max_model_len`：模型的上下文窗口长度

### 2. 运行测试

部署模型后，运行以下命令开始测试：

```bash
python run_longbench.py --model [MODEL_NAME] --test_type [TEST_TYPE]
```

参数说明：
- `--model`：要测试的模型名称，必须与 `config/models.json` 中的配置匹配
- `--test_type`：测试类型，可选值包括：
  - `standard`：标准测试
  - `cot`：思维链测试
  - `no_context`：无上下文测试
  - `rag`：检索增强生成测试

### 3. 查看结果

测试完成后，使用以下命令分析并导出结果：

```bash
python analyze_results.py --model [MODEL_NAME] --output_file results.json
```

## 示例

使用 PiKV 模型进行标准测试：

```bash
# 部署模型
vllm serve models/pikv --api-key token-abc123 --tensor-parallel-size 4 --gpu-memory-utilization 0.95 --max_model_len 131072 --trust-remote-code

# 运行测试
python run_longbench.py --model pikv --test_type standard

# 分析结果
python analyze_results.py --model pikv --output_file pikv_results.json
```

## 结果解读

测试结果会包含以下指标：
- 每个任务类别的准确率
- 综合准确率
- 不同文本长度下的模型表现
- 与基准模型的对比结果

更多详细信息，请参阅 [LongBench 官方文档](https://github.com/THUDM/LongBench)。 