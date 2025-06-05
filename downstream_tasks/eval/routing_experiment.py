import torch
from core.single.module.pikv_routing import BaseRouter, TopKBalancedRouter, AdaptiveRouter

# Initialize routers
base_router = BaseRouter(hidden_size=512, num_experts=4)
topk_router = TopKBalancedRouter(hidden_size=512, num_experts=4)
adaptive_router = AdaptiveRouter(hidden_size=512, num_experts=4)

# Dummy data
hidden_states = torch.randn(10, 512)

# Evaluate routing
output = base_router(hidden_states)
print("Base Router Output:", output[0].shape)

output = topk_router(hidden_states)
print("TopK Router Output:", output[0].shape)

output = adaptive_router(hidden_states)
print("Adaptive Router Output:", output[0].shape) 