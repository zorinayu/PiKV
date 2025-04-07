# PiKV - Parallel Distributed MoE KV Cache Design

PiKV (Parallel Distributed Mixture of Experts Key-Value Cache Design) is an advanced framework designed to optimize the management and compression of Key-Value (KV) caches in distributed Mixture of Experts (MoE) models. By leveraging parallel and distributed strategies, PiKV enhances inference efficiency and reduces memory and compute overhead in large-scale MoE deployments.

## Features

- **Parallel Prefetching and Communication Overlap**: PiKV overlaps memory read operations for model weights and KV-cache with collective communication, effectively hiding communication latency and improving throughput. This is inspired by the PRESERVE framework.

- **Dynamic KV Cache Compression**: PiKV uses attention-based pyramidal patterns to allocate more KV memory to lower layers and less to higher layers, following the insights from PyramidKV.

- **Query-Aware KV Cache Selection**: PiKV prioritizes important tokens for caching using query-attention relevance, loading only the most critical KV pairs, as proposed in Quest.

- **KV Cache Streaming**: PiKV supports efficient KV streaming and checkpoint recovery using techniques influenced by MOONCAKE, enabling fast failover and state persistence in distributed inference.

- **Memory Expansion Techniques**: Leveraging CXL-based memory disaggregation, PiKV enables KV cache offloading to host or external memory, reducing GPU memory bottlenecks.

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
@misc{pikv2024,
  title={PiKV: Parallel Distributed MoE KV Cache Design},
  author={Dong Liu},
  year={2024},
  howpublished={\url{https://github.com/your_org/pikv}}
}
```