# PiKV - Parallel Distributed MoE KV Cache Design

PiKV (Parallel Distributed Mixture of Experts Key-Value Cache Design) is an advanced framework designed to optimize the management and compression of Key-Value (KV) caches in distributed Mixture of Experts (MoE) models. By leveraging parallel and distributed strategies, PiKV enhances inference efficiency and reduces memory and compute overhead in large-scale MoE deployments.

## Features

- **Parallel Prefetching and Communication Overlap**: PiKV overlaps memory read operations for model weights and KV-cache with collective communication, effectively hiding communication latency and improving throughput. This is inspired by the PRESERVE framework.

- **Dynamic KV Cache Compression**: PiKV uses attention-based pyramidal patterns to allocate more KV memory to lower layers and less to higher layers, following the insights from PyramidKV.

- **Query-Aware KV Cache Selection**: PiKV prioritizes important tokens for caching using query-attention relevance, loading only the most critical KV pairs, as proposed in Quest.

- **KV Cache Streaming**: PiKV supports efficient KV streaming and checkpoint recovery using techniques influenced by MOONCAKE, enabling fast failover and state persistence in distributed inference.

- **Memory Expansion Techniques**: Leveraging CXL-based memory disaggregation, PiKV enables KV cache offloading to host or external memory, reducing GPU memory bottlenecks.

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

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

### 1. LLM Next Token Prediction

```bash
# Generate training and testing data
python downstream_tasks/llm/next_tok_pred/generate_data.py

# Train single-node model with LoRA
python downstream_tasks/llm/next_tok_pred/train_llm.py --model_type single --use_lora

# Train distributed model with LoRA (using 4 GPUs)
torchrun --nproc_per_node=4 downstream_tasks/llm/next_tok_pred/train_llm.py --model_type distributed --use_lora
```

Result
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