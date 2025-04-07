# PiKV - Parallel Distributed MoE KV Cache Design

# PiKV: Parallel Distributed MoE KV Cache Design

PiKV (Parallel Distributed Mixture of Experts Key-Value Cache Design) is an advanced framework aimed at optimizing the management and compression of Key-Value (KV) caches in distributed Mixture of Experts (MoE) models. By leveraging parallel and distributed strategies, PiKV enhances inference efficiency and reduces resource consumption in large-scale MoE deployments.

## Features

- **Parallel Prefetching and Communication Overlap**: PiKV employs a strategy that overlaps memory read operations for model weights and KV-cache with collective communication operations, effectively hiding communication latency and improving inference efficiency. This approach is inspired by the PRESERVE framework, which demonstrated up to 1.6× end-to-end speedup on state-of-the-art, open-source LLMs. :contentReference[oaicite:0]{index=0}

- **Dynamic KV Cache Compression**: Utilizing the Pyramidal Information Funneling pattern observed in attention mechanisms, PiKV dynamically adjusts the KV cache size across different layers. More cache is allocated to lower layers where attention is more dispersed, and less to higher layers where attention focuses on critical tokens. This method significantly reduces memory usage while maintaining model performance, as demonstrated by PyramidKV. :contentReference[oaicite:1]{index=1}

- **Query-Aware KV Cache Selection**: By evaluating the criticality of KV cache pages using query vectors, PiKV loads only the most important KV cache pages for attention computation. This query-aware sparsity approach significantly speeds up self-attention without sacrificing accuracy, achieving up to 2.23× speedup. :contentReference[oaicite:2]{index=2}

- **KV Cache Streaming**: PiKV introduces an efficient KV cache streaming mechanism that enables rapid state recovery and fault tolerance, ensuring stable operation in distributed environments. This design is influenced by the MOONCAKE architecture, which features a KVCache-centric disaggregated architecture for serving LLM chatbots. :contentReference[oaicite:3]{index=3}

- **Memory Expansion Techniques**: Leveraging technologies like Compute Express Link (CXL), PiKV stores KV caches in expanded memory, alleviating GPU memory limitations and enhancing service capacity. Studies have shown that CXL-based KV cache storage can achieve competitive time-to-first-token (TTFT) performance, making it a promising solution for large-scale LLM serving. :contentReference[oaicite:4]{index=4}

## Mathematical Formulation

### Dynamic KV Cache Allocation

The allocation of KV cache across different layers is guided by the attention distribution patterns. Let \( L \) denote the total number of layers, and \( C_i \) represent the cache size allocated to the \( i \)-th layer. The allocation follows an arithmetic sequence:

\[
C_i = C_1 - (i - 1) \cdot d
\]

where \( C_1 \) is the cache size for the first layer, and \( d \) is the common difference determined based on the total cache budget and the desired pyramidal shape.

### Query-Aware KV Cache Selection

For each token \( t \) in the sequence, an importance score \( I_t \) is computed to determine its relevance:

\[
I_t = \sum_{h=1}^{H} \text{softmax}\left( \frac{Q_h K_t^\top}{\sqrt{d_k}} \right)
\]

where \( H \) is the number of attention heads, \( Q_h \) is the query vector for head \( h \), \( K_t \) is the key vector for token \( t \), and \( d_k \) is the dimension of the key vectors. Tokens with higher \( I_t \) are prioritized in the KV cache.
