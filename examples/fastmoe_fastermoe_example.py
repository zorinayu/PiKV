#!/usr/bin/env python3
"""
FastMoE and FasterMoE Example
Demonstrates the usage of FastMoE and FasterMoE routers in PiKV
"""

import torch
import torch.nn as nn
import sys
import os

# Add the core directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core', 'single'))

from moe import create_moe

def test_fastmoe():
    """Test FastMoE router with dynamic shadowing and smart scheduling"""
    print("Testing FastMoE Router...")
    
    # Create FastMoE with dynamic shadowing and smart scheduling
    fastmoe = create_moe(
        hidden_size=512,
        num_experts=8,
        router_type="fastmoe",
        top_k=2,
        use_normalization=True,
        use_lora=True,
        enable_dynamic_shadowing=True,
        enable_fuse=True
    )
    
    # Test input
    x = torch.randn(2, 64, 512)
    
    # Forward pass
    output, aux_loss = fastmoe(x)
    
    print(f"FastMoE output shape: {output.shape}")
    print(f"FastMoE auxiliary loss: {aux_loss:.4f}")
    print(f"FastMoE parameters: {sum(p.numel() for p in fastmoe.parameters()):,}")
    
    return fastmoe


def test_fastermoe():
    """Test FasterMoE router with hierarchical intelligent routing"""
    print("\nTesting FasterMoE Router...")
    
    # Create FasterMoE with all optimizations
    fastermoe = create_moe(
        hidden_size=512,
        num_experts=8,
        router_type="fastermoe",
        top_k=2,
        use_normalization=True,
        use_lora=True,
        enable_dynrep=True,
        enable_fuse=True,
        enable_hir_gate=True
    )
    
    # Test input
    x = torch.randn(2, 64, 512)
    
    # Forward pass
    output, aux_loss = fastermoe(x)
    
    print(f"FasterMoE output shape: {output.shape}")
    print(f"FasterMoE auxiliary loss: {aux_loss:.4f}")
    print(f"FasterMoE parameters: {sum(p.numel() for p in fastermoe.parameters()):,}")
    
    # Test performance tracking (FasterMoE specific feature)
    if hasattr(fastermoe.router, 'update_expert_performance'):
        for i in range(8):
            fastermoe.router.update_expert_performance(i, torch.rand(1).item())
        print("FasterMoE expert performance updated")
    
    return fastermoe


def test_comparison():
    """Compare different MoE routers"""
    print("\nComparing MoE Routers...")
    
    routers = {
        "Base": "base",
        "EPLB": "eplb", 
        "Hierarchical": "hierarchical",
        "FastMoE": "fastmoe",
        "FasterMoE": "fastermoe"
    }
    
    results = {}
    x = torch.randn(2, 64, 512)
    
    for name, router_type in routers.items():
        print(f"\nTesting {name} Router...")
        
        # Create MoE with specific router
        if router_type == "fastmoe":
            moe = create_moe(
                hidden_size=512,
                num_experts=8,
                router_type=router_type,
                enable_dynamic_shadowing=True,
                enable_fuse=True
            )
        elif router_type == "fastermoe":
            moe = create_moe(
                hidden_size=512,
                num_experts=8,
                router_type=router_type,
                enable_dynrep=True,
                enable_fuse=True,
                enable_hir_gate=True
            )
        else:
            moe = create_moe(
                hidden_size=512,
                num_experts=8,
                router_type=router_type
            )
        
        # Measure performance
        import time
        start_time = time.time()
        
        output, aux_loss = moe(x)
        
        end_time = time.time()
        inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
        results[name] = {
            "output_shape": output.shape,
            "aux_loss": aux_loss,
            "parameters": sum(p.numel() for p in moe.parameters()),
            "inference_time_ms": inference_time
        }
        
        print(f"  Output shape: {output.shape}")
        print(f"  Auxiliary loss: {aux_loss:.4f}")
        print(f"  Parameters: {results[name]['parameters']:,}")
        print(f"  Inference time: {inference_time:.2f} ms")
    
    # Print comparison summary
    print("\n" + "="*60)
    print("MoE Router Comparison Summary")
    print("="*60)
    print(f"{'Router':<15} {'Parameters':<12} {'Aux Loss':<10} {'Time (ms)':<10}")
    print("-" * 60)
    
    for name, result in results.items():
        print(f"{name:<15} {result['parameters']:<12,} {result['aux_loss']:<10.4f} {result['inference_time_ms']:<10.2f}")
    
    return results


def test_advanced_features():
    """Test advanced features of FastMoE and FasterMoE"""
    print("\nTesting Advanced Features...")
    
    # Test FastMoE with different configurations
    print("\nFastMoE Configurations:")
    
    configs = [
        {"enable_dynamic_shadowing": True, "enable_fuse": False},
        {"enable_dynamic_shadowing": False, "enable_fuse": True},
        {"enable_dynamic_shadowing": True, "enable_fuse": True}
    ]
    
    x = torch.randn(2, 64, 512)
    
    for i, config in enumerate(configs):
        print(f"\nConfig {i+1}: {config}")
        
        fastmoe = create_moe(
            hidden_size=512,
            num_experts=8,
            router_type="fastmoe",
            **config
        )
        
        output, aux_loss = fastmoe(x)
        print(f"  Output shape: {output.shape}, Aux loss: {aux_loss:.4f}")
    
    # Test FasterMoE with different configurations
    print("\nFasterMoE Configurations:")
    
    configs = [
        {"enable_dynrep": True, "enable_fuse": False, "enable_hir_gate": False},
        {"enable_dynrep": False, "enable_fuse": True, "enable_hir_gate": False},
        {"enable_dynrep": False, "enable_fuse": False, "enable_hir_gate": True},
        {"enable_dynrep": True, "enable_fuse": True, "enable_hir_gate": True}
    ]
    
    for i, config in enumerate(configs):
        print(f"\nConfig {i+1}: {config}")
        
        fastermoe = create_moe(
            hidden_size=512,
            num_experts=8,
            router_type="fastermoe",
            **config
        )
        
        output, aux_loss = fastermoe(x)
        print(f"  Output shape: {output.shape}, Aux loss: {aux_loss:.4f}")


def main():
    """Main function to run all tests"""
    print("FastMoE and FasterMoE Example")
    print("=" * 40)
    
    try:
        # Test individual routers
        fastmoe_model = test_fastmoe()
        fastermoe_model = test_fastermoe()
        
        # Compare all routers
        comparison_results = test_comparison()
        
        # Test advanced features
        test_advanced_features()
        
        print("\n" + "="*40)
        print("All tests completed successfully!")
        print("FastMoE and FasterMoE are working correctly.")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
