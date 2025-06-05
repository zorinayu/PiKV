import torch
from core.single.module.pikv_compression import PyramidCompressor, LoRACompressor, SVDCompressor
from core.single.module.pikv_routing import BaseRouter, TopKBalancedRouter, AdaptiveRouter
from core.single.module.pikv_scheduling import H2OScheduler, StreamingLLMScheduler, QUESTScheduler

class ModulePiKV:
    def __init__(self, hidden_size=512, num_experts=4, cache_size=1024):
        # Initialize compressors
        self.pyramid_compressor = PyramidCompressor(hidden_size=hidden_size)
        self.lora_compressor = LoRACompressor(hidden_size=hidden_size)
        self.svd_compressor = SVDCompressor(hidden_size=hidden_size)

        # Initialize routers
        self.base_router = BaseRouter(hidden_size=hidden_size, num_experts=num_experts)
        self.topk_router = TopKBalancedRouter(hidden_size=hidden_size, num_experts=num_experts)
        self.adaptive_router = AdaptiveRouter(hidden_size=hidden_size, num_experts=num_experts)

        # Initialize schedulers
        self.h2o_scheduler = H2OScheduler(cache_size=cache_size, hidden_size=hidden_size)
        self.streaming_scheduler = StreamingLLMScheduler(cache_size=cache_size, hidden_size=hidden_size)
        self.quest_scheduler = QUESTScheduler(cache_size=cache_size, hidden_size=hidden_size)

    def compress(self, keys, values):
        # Example compression using PyramidCompressor
        return self.pyramid_compressor(keys, values)

    def route(self, hidden_states):
        # Example routing using BaseRouter
        return self.base_router(hidden_states)

    def schedule(self, keys, values, metadata):
        # Example scheduling using H2OScheduler
        return self.h2o_scheduler.select_eviction_candidates(keys, values, metadata) 