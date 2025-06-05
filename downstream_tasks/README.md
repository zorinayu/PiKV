# Experiments

### 🔍 Accuracy vs Compression Trade-offs for Routing + Compression + Scheduling

| System Variant        | Routing            | Compression                 | Scheduling          | ΔAcc ↓ (Accuracy Drop) | Compression Rate ↑ | Notes                                 |
|-----------------------|--------------------|------------------------------|----------------------|-------------------------|---------------------|----------------------------------------|
| **(A) Accuracy-First** | ✅ EPLBRouter        | ❌ None                      | ✅ DuoAttention       | **~0.8–1.3%**           | ~**2.2×**           | 🔥 Best accuracy, light compression    |
| **(B) Balanced Design**| ✅ TopKRouter        | ✅ PyramidKV + FastV         | ✅ AdaKVScheduler     | ~1.4–1.7%               | ~3.5–4.5×           | Good trade-off, general deployment     |
| **(C) Max Compression**| ❌ None              | ✅ LoRA + Distillation       | ❌ None               | ~9.5–12.5%               | **2.8–4.8×**         | Strong compression, notable acc drop   |
| **(D) Routing Only**   | ✅ PiKVRouter        | ❌ None                      | ❌ None               | ~1.3–1.5%               | 1.0×                | Moderate benefit from routing only     |
| **(E) Compression Only**| ❌ None             | ✅ ChunkKV + SVD             | ❌ None               | ~4.8–5.5%               | 2.6–4.0×            | Pure compression impact                |
| **(F) Scheduling Only**| ❌ None              | ❌ None                      | ✅ DuoAttention       | ~1.2–1.6%               | 1.0–2.0×            | Lightweight, robust scheduling         |
| **(G) Baseline (No Mod)**| ❌ None            | ❌ None                      | ❌ None               | 0.0%                    | 1.0×                | Reference line for comparison          |
