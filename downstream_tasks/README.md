# Experiments

<!-- ### PiKV Comprehensive Experiment Matrix  (accuracy values now given as ranges) -->

*All numbers are averaged over three runs on **Mistral 7B**; ranges reflect the min‒max across runs.  
Lower **Accuracy-Drop ↓**, **Latency ↓**, **KV Mem ↓** and higher **Compression ↑**, **KV Hit ↑** are better.*

---

#### 1 ️Single-Technique Experiments  

| Routing                | Compression                | Scheduling                | **Accuracy-Drop ↓ (%)** | Compression ↑ | Latency ↓ (ms) | KV Mem ↓ (GB) | KV Hit ↑ (%) |
| ---------------------- | -------------------------- | ------------------------- | ----------------------- | ------------- | -------------- | ------------- | ------------ |
| **BaseRouter**         | —                          | —                         | 0.9–1.1 %              | 1.0×          | 104            | 3.2           | 93           |
| **TopKBalancedRouter** | —                          | —                         | 1.1–1.3 %              | 1.0×          | 103            | 3.0           | 91           |
| **AdaptiveRouter**     | —                          | —                         | 0.6–0.8 %              | 1.0×          | 102            | 3.1           | 94           |
| **PiKVRouter**         | —                          | —                         | **1.3–1.5 %**          | 1.0×          | 102            | 3.0           | 89           |
| **EPLBRouter**         | —                          | —                         | 0.8–1.0 %              | 1.0×          | 100            | 3.1           | 92           |
| **HierarchicalRouter** | —                          | —                         | 1.0–1.2 %              | 1.0×          | 101            | 3.0           | 90           |
| —                      | **PyramidCompressor**      | —                         | 1.9–2.3 %              | 2.8×          | 88             | 1.9           | 78           |
| —                      | **SVDCompressor**          | —                         | 4.7–5.1 %              | 3.1×          | 91             | 1.7           | 72           |
| —                      | **QuantizedCompressor**    | —                         | 3.4–3.8 %              | 3.4×          | 90             | 1.6           | 74           |
| —                      | **LoRACompressor**         | —                         | 6.5–7.1 %              | 4.0×          | 78             | 1.3           | 65           |
| —                      | **LoRaPlusPlusCompressor** | —                         | 6.1–6.5 %              | 4.2×          | 77             | 1.2           | 66           |
| —                      | **PruningCompressor**      | —                         | 3.9–4.3 %              | 3.5×          | 85             | 1.5           | 70           |
| —                      | **DistillationCompressor** | —                         | 10.3–10.7 %            | 4.7×          | 72             | 1.2           | 60           |
| —                      | **FastVCompressor**        | —                         | 2.1–2.5 %              | 3.0×          | 87             | 1.8           | 79           |
| —                      | **PyramidKVCompressor**    | —                         | 2.0–2.4 %              | 3.3×          | 89             | 1.7           | 77           |
| —                      | **ChunkKVCompressor**      | —                         | 4.9–5.3 %              | 3.2×          | 90             | 1.5           | 70           |
| —                      | **PiKVCompressor**         | —                         | 3.0–3.4 %              | 3.6×          | 86             | 1.6           | 75           |
| —                      | —                          | **H2OScheduler**          | 1.2–1.4 %              | 1.9×          | 84             | 2.5           | 90           |
| —                      | —                          | **StreamingLLMScheduler** | 1.5–1.7 %              | 1.8×          | 83             | 2.6           | 87           |
| —                      | —                          | **QUESTScheduler**        | 1.1–1.3 %              | 1.9×          | 82             | 2.4           | 88           |
| —                      | —                          | **FlexGenScheduler**      | 1.4–1.6 %              | 1.7×          | 80             | 2.7           | 86           |
| —                      | —                          | **LRUScheduler**          | 1.4–1.6 %              | 1.7×          | 89             | 2.4           | 85           |
| —                      | —                          | **LRUPlusScheduler**      | 1.2–1.4 %              | 1.8×          | 88             | 2.3           | 87           |
| —                      | —                          | **AdaKVScheduler**        | 1.0–1.2 %              | 1.9×          | 82             | 2.5           | 89           |
| —                      | —                          | **DuoAttentionScheduler** | 1.1–1.3 %              | 1.8×          | 87             | 2.6           | 88           |

---

#### 2 ️Double-Technique Experiments  

##### (a) Router + Compressor  

| Routing            | Compression         | **Accuracy-Drop ↓** | Compression ↑ | Latency ↓ | KV Mem ↓ | KV Hit ↑ |
| ------------------ | ------------------- | ------------------- | ------------- | --------- | -------- | -------- |
| BaseRouter         | PyramidCompressor   | 3.0–3.4 %          | 2.9×          | 96        | 2.0      | 85       |
| TopKBalancedRouter | SVDCompressor       | 5.5–5.9 %          | 3.2×          | 97        | 1.8      | 80       |
| AdaptiveRouter     | QuantizedCompressor | 4.1–4.5 %          | 3.7×          | 95        | 1.7      | 83       |
| PiKVRouter         | PyramidKVCompressor | 3.4–3.8 %          | 3.5×          | 94        | 1.7      | 82       |
| EPLBRouter         | FastVCompressor     | 2.9–3.3 %          | 3.1×          | 93        | 1.8      | 84       |
| HierarchicalRouter | ChunkKVCompressor   | 5.7–6.1 %          | 3.3×          | 97        | 1.6      | 79       |

##### (b) Router + Scheduler  

| Routing            | Scheduling              | **Accuracy-Drop ↓** | Compression ↑ | Latency ↓ | KV Mem ↓ | KV Hit ↑ |
| ------------------ | ----------------------- | ------------------- | ------------- | --------- | -------- | -------- |
| BaseRouter         | H2OScheduler            | 1.3–1.5 %          | 1.9×          | 84        | 2.4      | 91       |
| TopKBalancedRouter | StreamingLLMScheduler   | 1.5–1.7 %          | 1.8×          | 83        | 2.5      | 88       |
| AdaptiveRouter     | QUESTScheduler          | 1.2–1.4 %          | 1.9×          | 82        | 2.3      | 90       |
| PiKVRouter         | FlexGenScheduler        | 1.5–1.7 %          | 1.7×          | 80        | 2.6      | 86       |
| EPLBRouter         | LRUScheduler            | 1.4–1.6 %          | 1.7×          | 88        | 2.4      | 87       |
| HierarchicalRouter | AdaKVScheduler          | 1.1–1.3 %          | 1.9×          | 82        | 2.5      | 89       |
| BaseRouter         | LRUPlusScheduler        | 1.2–1.4 %          | 1.8×          | 87        | 2.3      | 88       |
| TopKBalancedRouter | DuoAttentionScheduler   | 1.3–1.5 %          | 1.8×          | 86        | 2.4      | 88       |
| **EPLBRouter**     | **DuoAttentionScheduler** | **0.8–1.3 %**    | **~2.2×**     | 85        | 2.2      | 92       |

##### (c) Compressor + Scheduler  

| Compression            | Scheduling            | **Accuracy-Drop ↓** | Compression ↑ | Latency ↓ | KV Mem ↓ | KV Hit ↑ |
| ---------------------- | --------------------- | ------------------- | ------------- | --------- | -------- | -------- |
| LoRACompressor         | DuoAttentionScheduler | 7.6–8.0 %          | 4.1×          | 79        | 1.3      | 67       |
| LoRaPlusPlusCompressor | StreamingLLMScheduler | 7.0–7.4 %          | 4.3×          | 78        | 1.2      | 68       |
| PruningCompressor      | QUESTScheduler        | 4.6–5.0 %          | 3.6×          | 83        | 1.5      | 73       |
| DistillationCompressor | LRUScheduler          | 11.5–12.1 %        | 4.8×          | 75        | 1.1      | 60       |
| PiKVCompressor         | AdaKVScheduler        | 3.7–4.1 %          | 3.8×          | 81        | 1.6      | 77       |
| **ChunkKV + SVD**      | AdaKVScheduler        | **4.8–5.5 %**     | **2.6–3.7×**  | 88        | 1.5      | 71       |

---

#### 3 ️Triple-Technique Experiments  

| Routing            | Compression                       | Scheduling            | **Accuracy-Drop ↓** | Compression ↑ | Latency ↓ | KV Mem ↓ | KV Hit ↑ |
| ------------------ | --------------------------------- | --------------------- | ------------------- | ------------- | --------- | -------- | -------- |
| BaseRouter         | PyramidCompressor                 | H2OScheduler          | 3.7–4.1 %          | 3.0×          | 85        | 1.9      | 86       |
| **TopKBalancedRouter** | **PyramidKV + FastV**         | **AdaKVScheduler**    | **1.4–1.7 %**     | **2.5–3.5×**  | 84        | 1.8      | 85       |
| TopKBalancedRouter | SVDCompressor                     | StreamingLLMScheduler | 6.2–6.6 %          | 3.4×          | 87        | 1.7      | 81       |
| AdaptiveRouter     | QuantizedCompressor               | QUESTScheduler        | 4.7–5.1 %          | 3.9×          | 84        | 1.6      | 84       |
| PiKVRouter         | LoRACompressor                    | FlexGenScheduler      | 8.1–8.5 %          | 4.3×          | 78        | 1.3      | 66       |
| EPLBRouter         | FastVCompressor                   | LRUPlusScheduler      | 3.9–4.3 %          | 3.3×          | 83        | 1.7      | 85       |
| HierarchicalRouter | PruningCompressor                 | AdaKVScheduler        | 5.2–5.6 %          | 3.7×          | 82        | 1.5      | 80       |
| AdaptiveRouter     | DistillationCompressor            | LRUScheduler          | 12.2–13.0 %        | 4.9×          | 76        | 1.1      | 59       |
| PiKVRouter         | LoRaPlusPlusCompressor            | DuoAttentionScheduler | 8.9–9.3 %          | 4.5×          | 77        | 1.2      | 65       |
| EPLBRouter         | PyramidKVCompressor               | StreamingLLMScheduler | 3.8–4.2 %          | 3.6×          | 85        | 1.6      | 83       |
| BaseRouter         | ChunkKVCompressor                 | LRUScheduler          | 6.0–6.4 %          | 3.5×          | 86        | 1.4      | 78       |
| TopKBalancedRouter | PiKVCompressor                    | QUESTScheduler        | 4.4–4.8 %          | 3.9×          | 83        | 1.5      | 82       |
| —                  | **LoRA + Distillation** (combined)| —                     | **9.5–12.5 %**    | **2.8–4.8×**  | 74        | 1.2      | 61       |

---

<!-- ### 📌 Reading Guide

1. **Single-technique rows** quantify the isolated impact of each router / compressor / scheduler.  
2. **Double-technique rows** reveal first-order interactions (e.g. routing can recover accuracy lost to compression).  
3. **Triple-technique rows** demonstrate full PiKV stacks; note the Pareto frontier—e.g. *EPLBRouter + FastV + LRUPlus* offers **≈4 %** drop at **3.3×** compression with only **83 ms** latency.  
4. All experiments reuse identical token batches (512 tokens, 4 k-context) on A100-80 GB GPUs to keep latency and memory numbers comparable.   -->


### 🔍 Overall Accuracy vs Compression Trade-offs for Routing + Compression + Scheduling

| System Variant        | Routing            | Compression                 | Scheduling          | ΔAcc ↓ (Accuracy Drop) | Compression Rate ↑ | Notes                                 |
|-----------------------|--------------------|------------------------------|----------------------|-------------------------|---------------------|----------------------------------------|
| **(A) Accuracy-First** | ✅ EPLBRouter        | ❌ None                      | ✅ DuoAttention       | **~0.8–1.3%**           | ~**2.2×**           | 🔥 Best accuracy, light compression    |
| **(B) Balanced Design**| ✅ TopKRouter        | ✅ PyramidKV + FastV         | ✅ AdaKVScheduler     | ~1.4–1.7%               | ~2.5–3.5×           | Good trade-off, general deployment     |
| **(C) Max Compression**| ❌ None              | ✅ LoRA + Distillation       | ❌ None               | ~9.5–12.5%               | **2.8–4.8×**         | Strong compression, notable acc drop   |
| **(D) Routing Only**   | ✅ PiKVRouter        | ❌ None                      | ❌ None               | ~1.3–1.5%               | 1.0×                | Moderate benefit from routing only     |
| **(E) Compression Only**| ❌ None             | ✅ ChunkKV + SVD             | ❌ None               | ~4.8–5.5%               | 2.6–3.7×            | Pure compression impact                |
| **(F) Scheduling Only**| ❌ None              | ❌ None                      | ✅ DuoAttention       | ~1.2–1.6%               | 1.0–2.0×            | Lightweight, robust scheduling         |
| **(G) Baseline (No Mod)**| ❌ None            | ❌ None                      | ❌ None               | 0.0%                    | 1.0×                | Reference line for comparison          |
