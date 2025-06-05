import torch
from core.single.module.pikv_scheduling import H2OScheduler, StreamingLLMScheduler, QUESTScheduler

# Initialize schedulers
h2o_scheduler = H2OScheduler(cache_size=1024, hidden_size=512)
streaming_scheduler = StreamingLLMScheduler(cache_size=1024, hidden_size=512)
quest_scheduler = QUESTScheduler(cache_size=1024, hidden_size=512)

# Dummy data
keys = torch.randn(10, 512)
values = torch.randn(10, 512)
metadata = {}

# Evaluate scheduling
candidates = h2o_scheduler.select_eviction_candidates(keys, values, metadata)
print("H2O Scheduler Candidates:", candidates.shape)

candidates = streaming_scheduler.select_eviction_candidates(keys, values, metadata)
print("Streaming Scheduler Candidates:", candidates.shape)

candidates = quest_scheduler.select_eviction_candidates(keys, values, metadata)
print("QUEST Scheduler Candidates:", candidates.shape) 