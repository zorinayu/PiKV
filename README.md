# PiKV - Parallel Distributed MoE KV Cache Design

PiKV (Parallel Distributed Key-Value Cache Design with Routing) is a serving framework designed to optimize the management and compression of Key-Value (KV) caches in Large Language Models training and inference. By leveraging parallel, distributed, sparsity and routing strategies, PiKV can improve training and inference efficiency and reduces memory and computation overhead.

## Features

- Routing (PiKVRouting) - Reference to DeepSeek-V2, Sparsely-Gated MoE Layer; Futher research will look into Faster Transformer Decoding (kernel design) and Switch Transformers (Large Scale)
- Compression (PiKVCompression) - Reference to LoRA/LoRA+, PyramidKV/FastV, Distillation
- Streaming/Scheduling (PiKV Attention) - Reference to Quest and StreamingLLM

<!-- - **Parallel Prefetching and Communication Overlap**: PiKV overlaps memory read operations for model weights and KV-cache with collective communication, effectively hiding communication latency and improving throughput. This is inspired by the PRESERVE framework.

- **Dynamic KV Cache Compression**: PiKV uses attention-based pyramidal patterns to allocate more KV memory to lower layers and less to higher layers, following the insights from PyramidKV.

- **Query-Aware KV Cache Selection**: PiKV prioritizes important tokens for caching using query-attention relevance, loading only the most critical KV pairs, as proposed in Quest.

- **KV Cache Streaming**: PiKV supports efficient KV streaming and checkpoint recovery using techniques influenced by MOONCAKE, enabling fast failover and state persistence in distributed inference.

- **Memory Expansion Techniques**: Leveraging CXL-based memory disaggregation, PiKV enables KV cache offloading to host or external memory, reducing GPU memory bottlenecks. -->

## Installation

```bash
# Clone the repository
git clone https://github.com/NoakLiu/PiKV.git
cd PiKV

# Install the package in development mode
pip install -e .

# Install additional dependencies
pip install -r requirements.txt
```

## Usage

#### Must Use Before for Distributed Computing models defined in the files in the Single Machine folder
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

### 1. LLM Next Token Prediction

single GPU
```bash
# Generate training and testing data
python downstream_tasks/llm/next_tok_pred/generate_data.py

# Train single-node model with LoRA
python downstream_tasks/llm/next_tok_pred/train_llm.py --model_type single --use_lora
```


8 GPUs
```bash
# Train distributed model with LoRA (using 8 GPUs)
torchrun --nproc_per_node=4 downstream_tasks/llm/next_tok_pred/train_llm.py --model_type distributed --use_lora
```
or
```bash
torchrun --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=23458 \
    downstream_tasks/llm/next_tok_pred/d_transformers.py
```

Result on 8xA100
```
(mka) jovyan@w-lenge-large-4ceda59b6605447685173387f3a3f682-6d7cd77666-tk4vq:~/workspace/PiKV$ python downstream_tasks/llm/next_tok_pred/s_ntp.py 
Training model...
Epoch 1/5: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00, 10.78it/s, loss=2.4]
Epoch 2/5: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00, 130.57it/s, loss=4.57]
Epoch 3/5: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00, 137.13it/s, loss=4.32]
Epoch 4/5: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00, 135.16it/s, loss=4.07]
Epoch 5/5: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00, 136.52it/s, loss=3.81]

Testing predictions:
Input sequence: [1, 2, 3, 4, 5, 6, 7]
Predicted next token: 9
```

```
(mka) jovyan@w-lenge-large-4ceda59b6605447685173387f3a3f682-6d7cd77666-tk4vq:~/workspace/PiKV$ python downstream_tasks/llm/next_tok_pred/s_transformers.py 
/opt/saturncloud/envs/mka/lib/python3.10/site-packages/huggingface_hub/file_download.py:896: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
  warnings.warn(
/opt/saturncloud/envs/mka/lib/python3.10/site-packages/huggingface_hub/file_download.py:896: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
  warnings.warn(

Prompt: The quick brown fox

Generated: The quick brown foxes that are so popular with young children are in fact very similar to the foxes that are used in the wild. A fox is a fox that has a very long, long tail and is much larger than a human.

The foxes

Prompt: Once upon a time
Generated: Once upon a time, the world was filled with the endless stream of stars, and the universe was filled with stars. The universe was filled with stars. And the universe was filled with stars. And the universe was filled with stars. And the universe was filled with stars

Prompt: In a galaxy far far away
Generated: In a galaxy far far away, the world's first space station is located in the middle of a desert. A pair of huge red eyes meet in the sky.

"You're right. I'm a little bit surprised that you're here, but you're a human
```

Distributed on 8 A100 GPUs results
```
(mka) jovyan@w-lenge-large-4ceda59b6605447685173387f3a3f682-6d7cd77666-tk4vq:~/workspace/PiKV$ torchrun --nproc_per_node=8     --nnodes=1     --node_rank=0     --master_addr=localhost     --master_port=23459     downstream_tasks/llm/next_tok_pred/d_transformers.py
[2025-04-21 03:03:27,367] torch.distributed.run: [WARNING] 
[2025-04-21 03:03:27,367] torch.distributed.run: [WARNING] *****************************************
[2025-04-21 03:03:27,367] torch.distributed.run: [WARNING] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
[2025-04-21 03:03:27,367] torch.distributed.run: [WARNING] *****************************************
Initializing DistributedPiKVCache...
Initializing distributed environment on rank 3, local_rank 3
Successfully initialized distributed environment on rank 3
Process started with rank 3, local_rank 3, world_size 8
Rank 3: Loading model and tokenizer...
/opt/saturncloud/envs/mka/lib/python3.10/site-packages/huggingface_hub/file_download.py:896: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
  warnings.warn(
Initializing DistributedPiKVCache...
Initializing distributed environment on rank 0, local_rank 0
Successfully initialized distributed environment on rank 0
Process started with rank 0, local_rank 0, world_size 8
Rank 0: Loading model and tokenizer...
/opt/saturncloud/envs/mka/lib/python3.10/site-packages/huggingface_hub/file_download.py:896: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
  warnings.warn(
Initializing DistributedPiKVCache...
Initializing distributed environment on rank 2, local_rank 2
Successfully initialized distributed environment on rank 2
Process started with rank 2, local_rank 2, world_size 8
Rank 2: Loading model and tokenizer...
/opt/saturncloud/envs/mka/lib/python3.10/site-packages/huggingface_hub/file_download.py:896: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
  warnings.warn(
Initializing DistributedPiKVCache...
Initializing distributed environment on rank 7, local_rank 7
Successfully initialized distributed environment on rank 7
Process started with rank 7, local_rank 7, world_size 8
Rank 7: Loading model and tokenizer...
/opt/saturncloud/envs/mka/lib/python3.10/site-packages/huggingface_hub/file_download.py:896: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
  warnings.warn(
Initializing DistributedPiKVCache...
Initializing distributed environment on rank 1, local_rank 1
Initializing DistributedPiKVCache...
Initializing distributed environment on rank 4, local_rank 4
Successfully initialized distributed environment on rank 4
Process started with rank 4, local_rank 4, world_size 8
Rank 4: Loading model and tokenizer...
Successfully initialized distributed environment on rank 1
Process started with rank 1, local_rank 1, world_size 8
Rank 1: Loading model and tokenizer...
/opt/saturncloud/envs/mka/lib/python3.10/site-packages/huggingface_hub/file_download.py:896: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
  warnings.warn(
/opt/saturncloud/envs/mka/lib/python3.10/site-packages/huggingface_hub/file_download.py:896: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
  warnings.warn(
Initializing DistributedPiKVCache...
Initializing distributed environment on rank 6, local_rank 6
Successfully initialized distributed environment on rank 6
Process started with rank 6, local_rank 6, world_size 8
Rank 6: Loading model and tokenizer...
/opt/saturncloud/envs/mka/lib/python3.10/site-packages/huggingface_hub/file_download.py:896: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
  warnings.warn(
Initializing DistributedPiKVCache...
Initializing distributed environment on rank 5, local_rank 5
Successfully initialized distributed environment on rank 5
Process started with rank 5, local_rank 5, world_size 8
Rank 5: Loading model and tokenizer...
/opt/saturncloud/envs/mka/lib/python3.10/site-packages/huggingface_hub/file_download.py:896: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
  warnings.warn(
/opt/saturncloud/envs/mka/lib/python3.10/site-packages/huggingface_hub/file_download.py:896: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
  warnings.warn(
/opt/saturncloud/envs/mka/lib/python3.10/site-packages/huggingface_hub/file_download.py:896: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
  warnings.warn(
Rank 3: Initializing DistributedPiKVMoE...
Rank 3: Moving model to device cuda:3
Rank 2: Initializing DistributedPiKVMoE...
Rank 2: Moving model to device cuda:2
/opt/saturncloud/envs/mka/lib/python3.10/site-packages/huggingface_hub/file_download.py:896: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
  warnings.warn(
/opt/saturncloud/envs/mka/lib/python3.10/site-packages/huggingface_hub/file_download.py:896: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
  warnings.warn(
/opt/saturncloud/envs/mka/lib/python3.10/site-packages/huggingface_hub/file_download.py:896: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
  warnings.warn(
/opt/saturncloud/envs/mka/lib/python3.10/site-packages/huggingface_hub/file_download.py:896: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
  warnings.warn(
/opt/saturncloud/envs/mka/lib/python3.10/site-packages/huggingface_hub/file_download.py:896: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
  warnings.warn(
Rank 7: Initializing DistributedPiKVMoE...
/opt/saturncloud/envs/mka/lib/python3.10/site-packages/huggingface_hub/file_download.py:896: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
  warnings.warn(
Rank 7: Moving model to device cuda:7
Rank 5: Initializing DistributedPiKVMoE...
Rank 6: Initializing DistributedPiKVMoE...
Rank 5: Moving model to device cuda:5
Rank 6: Moving model to device cuda:6
Rank 1: Initializing DistributedPiKVMoE...
Rank 1: Moving model to device cuda:1
Rank 4: Initializing DistributedPiKVMoE...
Rank 4: Moving model to device cuda:4
Rank 0: Initializing DistributedPiKVMoE...
Rank 2: Initialization complete
Rank 0: Moving model to device cuda:0
Rank 3: Initialization complete
Rank 7: Initialization complete
Rank 6: Initialization complete
Rank 5: Initialization complete
Rank 4: Initialization complete
Rank 1: Initialization complete
Rank 0: Initialization complete

Prompt: The quick brown fox
Generating token 1/50
Generating token 2/50
Generating token 3/50
Generating token 4/50
Generating token 5/50
Generating token 6/50
Generating token 7/50
Generating token 8/50
Generating token 9/50
Generating token 10/50
Generating token 11/50
Generating token 12/50
Generating token 13/50
Generating token 14/50
Generating token 15/50
Generating token 16/50
Generating token 17/50
Generating token 18/50
Generating token 19/50
Generating token 20/50
Generating token 21/50
Generating token 22/50
Generating token 23/50
Generating token 24/50
Generating token 25/50
Generating token 26/50
Generating token 27/50
Generating token 28/50
Generating token 29/50
Generating token 30/50
Generating token 31/50
Generating token 32/50
Generating token 33/50
Generating token 34/50
Generating token 35/50
Generating token 36/50
Generating token 37/50
Generating token 38/50
Generating token 39/50
Generating token 40/50
Generating token 41/50
Generating token 42/50
Generating token 43/50
Generating token 44/50
Generating token 45/50
Generating token 46/50
Generating token 47/50
Generating token 48/50
Generating token 49/50
Generating token 50/50
Generated: The quick brown foxes can be found on the eastern coast of North America. In the wild they are small and harmless, but they are very intelligent and intelligent in their environment.

They are known to mate with any mammal that they see and even many birds.

Prompt: Once upon a time
Generating token 1/50
Generating token 2/50
Generating token 3/50
Generating token 4/50
Generating token 5/50
Generating token 6/50
Generating token 7/50
Generating token 8/50
Generating token 9/50
Generating token 10/50
Generating token 11/50
Generating token 12/50
Generating token 13/50
Generating token 14/50
Generating token 15/50
Generating token 16/50
Generating token 17/50
Generating token 18/50
Generating token 19/50
Generating token 20/50
Generating token 21/50
Generating token 22/50
Generating token 23/50
Generating token 24/50
Generating token 25/50
Generating token 26/50
Generating token 27/50
Generating token 28/50
Generating token 29/50
Generating token 30/50
Generating token 31/50
Generating token 32/50
Generating token 33/50
Generating token 34/50
Generating token 35/50
Generating token 36/50
Generating token 37/50
Generating token 38/50
Generating token 39/50
Generating token 40/50
Generating token 41/50
Generating token 42/50
Generating token 43/50
Generating token 44/50
Generating token 45/50
Generating token 46/50
Generating token 47/50
Generating token 48/50
Generating token 49/50
Generating token 50/50
Generated: Once upon a time, there were two paths. One path was straight, and the other path was straight.

When I was young, I used to think that if I could do it, I could do it. When I was a child, I thought I

Prompt: In a galaxy far far away
Generating token 1/50
Generating token 2/50
Generating token 3/50
Generating token 4/50
Generating token 5/50
Generating token 6/50
Generating token 7/50
Generating token 8/50
Generating token 9/50
Generating token 10/50
Generating token 11/50
Generating token 12/50
Generating token 13/50
Generating token 14/50
Generating token 15/50
Generating token 16/50
Generating token 17/50
Generating token 18/50
Generating token 19/50
Generating token 20/50
Generating token 21/50
Generating token 22/50
Generating token 23/50
Generating token 24/50
Generating token 25/50
Generating token 26/50
Generating token 27/50
Generating token 28/50
Generating token 29/50
Generating token 30/50
Generating token 31/50
Generating token 32/50
Generating token 33/50
Generating token 34/50
Generating token 35/50
Generating token 36/50
Generating token 37/50
Generating token 38/50
Generating token 39/50
Generating token 40/50
Generating token 41/50
Generating token 42/50
Generating token 43/50
Generating token 44/50
Generating token 45/50
Generating token 46/50
Generating token 47/50
Generating token 48/50
Generating token 49/50
Generating token 50/50
Generated: In a galaxy far far away, the sun is about half as bright as the sun, and the distance between us and the sun is about half that of the galaxy. The distance between us and the sun is about five times that of the galaxy.

This is why we
Distributed environment cleaned up
Distributed environment cleaned up
Distributed environment cleaned up
Distributed environment cleaned up
Distributed environment cleaned up
Distributed environment cleaned up
Distributed environment cleaned up
Distributed environment cleaned up
```

Result on 8xA800

```
(pikv) root@dsw-238653-84cd4c6f57-5cdq7:/mnt/workspace/PiKV# export PYTHONPATH=$PYTHONPATH:$(pwd)
(pikv) root@dsw-238653-84cd4c6f57-5cdq7:/mnt/workspace/PiKV# torchrun --nproc_per_node=8     --nnodes=1     --node_rank=0     --master_addr=localhost     --master_port=23458     downstream_tasks/llm/next_tok_pred/d_transformers.py
W0512 15:37:07.092000 47050 site-packages/torch/distributed/run.py:793] 
W0512 15:37:07.092000 47050 site-packages/torch/distributed/run.py:793] *****************************************
W0512 15:37:07.092000 47050 site-packages/torch/distributed/run.py:793] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
W0512 15:37:07.092000 47050 site-packages/torch/distributed/run.py:793] *****************************************
Initializing DistributedPiKVCache...
Initializing distributed environment on rank 5, local_rank 5
Successfully initialized distributed environment on rank 5
Process started with rank 5, local_rank 5, world_size 8
Rank 5: Loading model and tokenizer...
Initializing DistributedPiKVCache...
Initializing distributed environment on rank 1, local_rank 1
Successfully initialized distributed environment on rank 1
Process started with rank 1, local_rank 1, world_size 8
Rank 1: Loading model and tokenizer...
Initializing DistributedPiKVCache...
Initializing distributed environment on rank 7, local_rank 7
Successfully initialized distributed environment on rank 7
Process started with rank 7, local_rank 7, world_size 8
Rank 7: Loading model and tokenizer...
Initializing DistributedPiKVCache...
Initializing distributed environment on rank 2, local_rank 2
Successfully initialized distributed environment on rank 2
Process started with rank 2, local_rank 2, world_size 8
Rank 2: Loading model and tokenizer...
Initializing DistributedPiKVCache...
Initializing distributed environment on rank 4, local_rank 4
Successfully initialized distributed environment on rank 4
Process started with rank 4, local_rank 4, world_size 8
Rank 4: Loading model and tokenizer...
Initializing DistributedPiKVCache...
Initializing distributed environment on rank 6, local_rank 6
Successfully initialized distributed environment on rank 6
Process started with rank 6, local_rank 6, world_size 8
Rank 6: Loading model and tokenizer...
Initializing DistributedPiKVCache...
Initializing distributed environment on rank 3, local_rank 3
Successfully initialized distributed environment on rank 3
Process started with rank 3, local_rank 3, world_size 8
Rank 3: Loading model and tokenizer...
Initializing DistributedPiKVCache...
Initializing distributed environment on rank 0, local_rank 0
Successfully initialized distributed environment on rank 0
Process started with rank 0, local_rank 0, world_size 8
Rank 0: Loading model and tokenizer...
/root/miniconda3/envs/pikv/lib/python3.11/site-packages/_distutils_hack/__init__.py:53: UserWarning: Reliance on distutils from stdlib is deprecated. Users must rely on setuptools to provide the distutils module. Avoid importing distutils or import setuptools first, and avoid setting SETUPTOOLS_USE_DISTUTILS=stdlib. Register concerns at https://github.com/pypa/setuptools/issues/new?template=distutils-deprecation.yml
  warnings.warn(
/root/miniconda3/envs/pikv/lib/python3.11/site-packages/_distutils_hack/__init__.py:53: UserWarning: Reliance on distutils from stdlib is deprecated. Users must rely on setuptools to provide the distutils module. Avoid importing distutils or import setuptools first, and avoid setting SETUPTOOLS_USE_DISTUTILS=stdlib. Register concerns at https://github.com/pypa/setuptools/issues/new?template=distutils-deprecation.yml
  warnings.warn(
/root/miniconda3/envs/pikv/lib/python3.11/site-packages/_distutils_hack/__init__.py:53: UserWarning: Reliance on distutils from stdlib is deprecated. Users must rely on setuptools to provide the distutils module. Avoid importing distutils or import setuptools first, and avoid setting SETUPTOOLS_USE_DISTUTILS=stdlib. Register concerns at https://github.com/pypa/setuptools/issues/new?template=distutils-deprecation.yml
  warnings.warn(
/root/miniconda3/envs/pikv/lib/python3.11/site-packages/_distutils_hack/__init__.py:53: UserWarning: Reliance on distutils from stdlib is deprecated. Users must rely on setuptools to provide the distutils module. Avoid importing distutils or import setuptools first, and avoid setting SETUPTOOLS_USE_DISTUTILS=stdlib. Register concerns at https://github.com/pypa/setuptools/issues/new?template=distutils-deprecation.yml
  warnings.warn(
/root/miniconda3/envs/pikv/lib/python3.11/site-packages/_distutils_hack/__init__.py:53: UserWarning: Reliance on distutils from stdlib is deprecated. Users must rely on setuptools to provide the distutils module. Avoid importing distutils or import setuptools first, and avoid setting SETUPTOOLS_USE_DISTUTILS=stdlib. Register concerns at https://github.com/pypa/setuptools/issues/new?template=distutils-deprecation.yml
  warnings.warn(
/root/miniconda3/envs/pikv/lib/python3.11/site-packages/_distutils_hack/__init__.py:53: UserWarning: Reliance on distutils from stdlib is deprecated. Users must rely on setuptools to provide the distutils module. Avoid importing distutils or import setuptools first, and avoid setting SETUPTOOLS_USE_DISTUTILS=stdlib. Register concerns at https://github.com/pypa/setuptools/issues/new?template=distutils-deprecation.yml
  warnings.warn(
/root/miniconda3/envs/pikv/lib/python3.11/site-packages/_distutils_hack/__init__.py:53: UserWarning: Reliance on distutils from stdlib is deprecated. Users must rely on setuptools to provide the distutils module. Avoid importing distutils or import setuptools first, and avoid setting SETUPTOOLS_USE_DISTUTILS=stdlib. Register concerns at https://github.com/pypa/setuptools/issues/new?template=distutils-deprecation.yml
  warnings.warn(
/root/miniconda3/envs/pikv/lib/python3.11/site-packages/_distutils_hack/__init__.py:53: UserWarning: Reliance on distutils from stdlib is deprecated. Users must rely on setuptools to provide the distutils module. Avoid importing distutils or import setuptools first, and avoid setting SETUPTOOLS_USE_DISTUTILS=stdlib. Register concerns at https://github.com/pypa/setuptools/issues/new?template=distutils-deprecation.yml
  warnings.warn(
model.safetensors: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 548M/548M [00:49<00:00, 11.2MB/s]
generation_config.json: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 124/124 [00:00<00:00, 1.08MB/s]
Rank 4: Initializing DistributedPiKVMoE...
Rank 5: Initializing DistributedPiKVMoE...
Rank 0: Initializing DistributedPiKVMoE...
Rank 5: Moving model to device cuda:5
Rank 4: Moving model to device cuda:4
Rank 2: Initializing DistributedPiKVMoE...
Rank 6: Initializing DistributedPiKVMoE...
Rank 0: Moving model to device cuda:0
Rank 2: Moving model to device cuda:2
Rank 6: Moving model to device cuda:6
Rank 3: Initializing DistributedPiKVMoE...
Rank 7: Initializing DistributedPiKVMoE...
Rank 1: Initializing DistributedPiKVMoE...
Rank 3: Moving model to device cuda:3
Rank 7: Moving model to device cuda:7
Rank 1: Moving model to device cuda:1
Rank 4: Initialization complete
Rank 0: Initialization complete

Prompt: The quick brown fox
Rank 5: Initialization complete
Generating token 1/50
Rank 2: Initialization complete
Rank 6: Initialization complete
Rank 1: Initialization complete
Rank 3: Initialization complete
Rank 7: Initialization complete
Generating token 2/50
Generating token 3/50
Generating token 4/50
Generating token 5/50
Generating token 6/50
Generating token 7/50
Generating token 8/50
Generating token 9/50
Generating token 10/50
Generating token 11/50
Generating token 12/50
Generating token 13/50
Generating token 14/50
Generating token 15/50
Generating token 16/50
Generating token 17/50
Generating token 18/50
Generating token 19/50
Generating token 20/50
Generating token 21/50
Generating token 22/50
Generating token 23/50
Generating token 24/50
Generating token 25/50
Generating token 26/50
Generating token 27/50
Generating token 28/50
Generating token 29/50
Generating token 30/50
Generating token 31/50
Generating token 32/50
Generating token 33/50
Generating token 34/50
Generating token 35/50
Generating token 36/50
Generating token 37/50
Generating token 38/50
Generating token 39/50
Generating token 40/50
Generating token 41/50
Generating token 42/50
Generating token 43/50
Generating token 44/50
Generating token 45/50
Generating token 46/50
Generating token 47/50
Generating token 48/50
Generating token 49/50
Generating token 50/50
Generated: The quick brown fox will often have a soft spot on the face.

A great way to get the fox is to bring it into your home and put it on the fire.

If you are using a grill and want to get the fox to move away

Prompt: Once upon a time
Generating token 1/50
Generating token 2/50
Generating token 3/50
Generating token 4/50
Generating token 5/50
Generating token 6/50
Generating token 7/50
Generating token 8/50
Generating token 9/50
Generating token 10/50
Generating token 11/50
Generating token 12/50
Generating token 13/50
Generating token 14/50
Generating token 15/50
Generating token 16/50
Generating token 17/50
Generating token 18/50
Generating token 19/50
Generating token 20/50
Generating token 21/50
Generating token 22/50
Generating token 23/50
Generating token 24/50
Generating token 25/50
Generating token 26/50
Generating token 27/50
Generating token 28/50
Generating token 29/50
Generating token 30/50
Generating token 31/50
Generating token 32/50
Generating token 33/50
Generating token 34/50
Generating token 35/50
Generating token 36/50
Generating token 37/50
Generating token 38/50
Generating token 39/50
Generating token 40/50
Generating token 41/50
Generating token 42/50
Generating token 43/50
Generating token 44/50
Generating token 45/50
Generating token 46/50
Generating token 47/50
Generating token 48/50
Generating token 49/50
Generating token 50/50
Generated: Once upon a time, he had never been to a doctor, and had never been to a doctor's office.

But now, he was in the hospital, and he had never been to a doctor's office before.

"I was very angry,"

Prompt: In a galaxy far far away
Generating token 1/50
Generating token 2/50
Generating token 3/50
Generating token 4/50
Generating token 5/50
Generating token 6/50
Generating token 7/50
Generating token 8/50
Generating token 9/50
Generating token 10/50
Generating token 11/50
Generating token 12/50
Generating token 13/50
Generating token 14/50
Generating token 15/50
Generating token 16/50
Generating token 17/50
Generating token 18/50
Generating token 19/50
Generating token 20/50
Generating token 21/50
Generating token 22/50
Generating token 23/50
Generating token 24/50
Generating token 25/50
Generating token 26/50
Generating token 27/50
Generating token 28/50
Generating token 29/50
Generating token 30/50
Generating token 31/50
Generating token 32/50
Generating token 33/50
Generating token 34/50
Generating token 35/50
Generating token 36/50
Generating token 37/50
Generating token 38/50
Generating token 39/50
Generating token 40/50
Generating token 41/50
Generating token 42/50
Generating token 43/50
Generating token 44/50
Generating token 45/50
Generating token 46/50
Generating token 47/50
Generating token 48/50
Generating token 49/50
Generating token 50/50
Generated: In a galaxy far far away, a huge black hole is lurking in the dark, and the only way to find it is to travel there.

The discovery is the latest in a series of high-profile discoveries made by astronomers and scientists at the University of Texas at Austin
Distributed environment cleaned up
Distributed environment cleaned up
Distributed environment cleaned up
Distributed environment cleaned up
Distributed environment cleaned up
Distributed environment cleaned up
Distributed environment cleaned up
Distributed environment cleaned up
```


Ablation Tests
| Model    | Inference Time (s) | Perplexity |
|:---------|-------------------:|-----------:|
| STANDARD | 0.0003             | 4.603      |
| LORA     | 0.0010             | 4.502      |
| ADAPTIVE | 0.0513             | 4.490      |
| PIKV     | 0.0606             | 4.196      |


### 2. RAG (Retrieval-Augmented Generation)

```bash
# Prepare RAG data
python downstream_tasks/rag/prepare_data.py --dataset_name wikipedia --chunk_size 512

# Train single-node RAG model
python downstream_tasks/rag/train.py \
    --model_type single \
    --use_lora \
    --retriever_type dense \
    --index_type faiss \
    --embedding_dim 768

# Train distributed RAG model
torchrun --nproc_per_node=4 downstream_tasks/rag/train.py \
    --model_type distributed \
    --use_lora \
    --retriever_type dense \
    --index_type faiss \
    --embedding_dim 768
```

### 3. Vision Tasks

```bash
# Prepare vision dataset
python downstream_tasks/vision/prepare_data.py --dataset_name imagenet --image_size 224

# Train single-node vision model
python downstream_tasks/vision/train.py \
    --model_type single \
    --use_lora \
    --task classification \
    --backbone vit \
    --image_size 224

# Train distributed vision model
torchrun --nproc_per_node=4 downstream_tasks/vision/train.py \
    --model_type distributed \
    --use_lora \
    --task classification \
    --backbone vit \
    --image_size 224
```

### Command Line Arguments

Common arguments for all tasks:

- `--model_type`: Model type (`single` or `distributed`)
- `--use_lora`: Enable LoRA fine-tuning
- `--rank`: LoRA rank (default: 4)
- `--alpha`: LoRA alpha (default: 1.0)
- `--batch_size`: Batch size (default: 32)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--num_epochs`: Number of training epochs (default: 10)

RAG-specific arguments:
- `--retriever_type`: Type of retriever (`dense` or `sparse`)
- `--index_type`: Type of index (`faiss` or `annoy`)
- `--embedding_dim`: Dimension of embeddings
- `--chunk_size`: Size of text chunks for retrieval

Vision-specific arguments:
- `--task`: Vision task type (`classification`, `detection`, or `segmentation`)
- `--backbone`: Backbone model (`vit`, `resnet`, or `efficientnet`)
- `--image_size`: Input image size

### Distributed Training Configuration

For distributed training, you can set the following environment variables:

```bash
# Set distributed training parameters
export WORLD_SIZE=4
export RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=12345

# Run distributed training
torchrun --nproc_per_node=$WORLD_SIZE \
    --nnodes=1 \
    --node_rank=$RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    downstream_tasks/llm/next_tok_pred/train_llm.py \
    --model_type distributed \
    --use_lora
```

## Mathematical Formulation

### Dynamic KV Allocation

Let the total number of layers be `L`, and the cache size for layer  `i` be `C_i`. A pyramidal allocation policy is defined as:

$$
C_i = C_1 - (i - 1) \cdot d
$$

Where `C_1` is the cache size at the bottom layer and `d` is the step decrement.

### Query-Aware Token Importance

To compute the importance `I_t` of token `t`:

$$
I_t = \sum_{h=1}^{H} \text{softmax}\left( \frac{Q_h K_t^T}{\sqrt{d_k}} \right)
$$

Where `Q_h` is the query vector of attention head `h`, and `K_t` is the key vector of token `t`.

## Usage Example

```python
from pikv import PiKVCache

pikv_cache = PiKVCache(model_cfg, runtime_cfg)

for step in range(seq_len):
    q, k, v = get_current_step_kv()
    pikv_cache.update(layer_id, token_id, k, v)
    context_kv = pikv_cache.retrieve(layer_id)
```

## Architecture

PiKV supports the following core modules:

- `ExpertKVCache`: per-expert low-rank sliding cache
- `Router`: latency-aware top-k expert selection
- `Compressor`: LoRA + INT8/4 quantization
- `Streamer`: stateful KV streaming and checkpointing

## Citation

If you use PiKV in your work, please cite:

```bibtex
@misc{pikv2025,
  title={PiKV: Parallel Distributed MoE KV Cache Design},
  author={Dong Liu},
  year={2025},
  howpublished={\url{https://github.com/NoakLiu/PiKV}}
}
```