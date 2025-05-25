#!/usr/bin/env python3
"""
测试PiKV分布式知识蒸馏功能
"""

import torch
import os
import sys
import subprocess
import tempfile
from pathlib import Path

def test_single_gpu_distillation():
    """测试单GPU蒸馏功能"""
    print("=" * 60)
    print("测试单GPU知识蒸馏功能")
    print("=" * 60)
    
    try:
        # 设置环境变量
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['LOCAL_RANK'] = '0'
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12345'
        
        # 导入模块
        from d_transformers_distillation import DistributedPiKVCacheWithDistillation
        
        # 初始化模型
        print("初始化DistributedPiKVCacheWithDistillation...")
        pikv_cache = DistributedPiKVCacheWithDistillation(
            model_name="gpt2",
            max_length=512,
            use_distillation=True,
            teacher_hidden_size=1536,
            distillation_temperature=4.0,
            distillation_alpha=0.7
        )
        
        print("✓ 模型初始化成功")
        
        # 测试生成功能
        print("\n测试文本生成...")
        test_prompt = "The future of artificial intelligence"
        generated_text = pikv_cache.generate_with_distillation(
            test_prompt,
            max_new_tokens=20,
            temperature=0.7,
            use_teacher=True
        )
        
        print(f"输入: {test_prompt}")
        print(f"输出: {generated_text}")
        print("✓ 文本生成成功")
        
        # 测试训练步骤
        print("\n测试蒸馏训练步骤...")
        
        # 创建虚拟训练数据
        batch_size = 2
        seq_len = 10
        vocab_size = pikv_cache.tokenizer.vocab_size
        
        input_data = torch.randint(0, vocab_size, (batch_size, seq_len), device=pikv_cache.device)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=pikv_cache.device)
        
        # 创建优化器
        optimizer = torch.optim.Adam(pikv_cache.model.parameters(), lr=1e-4)
        
        # 执行训练步骤
        loss_info = pikv_cache.distillation_training_step(
            input_data=input_data,
            targets=targets,
            optimizer=optimizer
        )
        
        print("训练损失信息:")
        for loss_name, loss_value in loss_info.items():
            print(f"  {loss_name}: {loss_value:.4f}")
        
        print("✓ 蒸馏训练步骤成功")
        
        # 测试检查点保存和加载
        print("\n测试检查点保存和加载...")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = os.path.join(temp_dir, "test_checkpoint.pth")
            
            # 保存检查点
            pikv_cache.save_checkpoint(checkpoint_path)
            print(f"✓ 检查点保存成功: {checkpoint_path}")
            
            # 加载检查点
            pikv_cache.load_checkpoint(checkpoint_path)
            print("✓ 检查点加载成功")
        
        print("\n" + "=" * 60)
        print("单GPU蒸馏测试全部通过！")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"❌ 单GPU蒸馏测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_torchrun_command():
    """测试torchrun命令是否可以正常执行"""
    print("=" * 60)
    print("测试torchrun命令执行")
    print("=" * 60)
    
    try:
        # 构建torchrun命令
        script_path = Path(__file__).parent / "d_transformers_distillation.py"
        
        cmd = [
            "torchrun",
            "--nproc_per_node=1",
            "--nnodes=1", 
            "--node_rank=0",
            "--master_addr=localhost",
            "--master_port=23456",
            str(script_path),
            "--use_distillation",
            "--model", "gpt2",
            "--max_tokens", "10"
        ]
        
        print(f"执行命令: {' '.join(cmd)}")
        
        # 执行命令
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120  # 2分钟超时
        )
        
        if result.returncode == 0:
            print("✓ torchrun命令执行成功")
            print("标准输出:")
            print(result.stdout[-500:])  # 显示最后500个字符
            return True
        else:
            print(f"❌ torchrun命令执行失败，返回码: {result.returncode}")
            print("标准错误:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ torchrun命令执行超时")
        return False
    except Exception as e:
        print(f"❌ torchrun命令测试失败: {e}")
        return False

def test_multi_gpu_simulation():
    """模拟多GPU环境测试"""
    print("=" * 60)
    print("模拟多GPU环境测试")
    print("=" * 60)
    
    try:
        # 检查CUDA是否可用
        if not torch.cuda.is_available():
            print("⚠️  CUDA不可用，跳过多GPU测试")
            return True
        
        gpu_count = torch.cuda.device_count()
        print(f"检测到 {gpu_count} 个GPU")
        
        if gpu_count < 2:
            print("⚠️  GPU数量不足，跳过多GPU测试")
            return True
        
        # 构建多GPU torchrun命令
        script_path = Path(__file__).parent / "d_transformers_distillation.py"
        
        cmd = [
            "torchrun",
            f"--nproc_per_node={min(2, gpu_count)}",
            "--nnodes=1",
            "--node_rank=0", 
            "--master_addr=localhost",
            "--master_port=23457",
            str(script_path),
            "--use_distillation",
            "--model", "gpt2",
            "--max_tokens", "5"
        ]
        
        print(f"执行多GPU命令: {' '.join(cmd)}")
        
        # 执行命令
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=180  # 3分钟超时
        )
        
        if result.returncode == 0:
            print("✓ 多GPU测试成功")
            return True
        else:
            print(f"❌ 多GPU测试失败，返回码: {result.returncode}")
            print("标准错误:")
            print(result.stderr[-1000:])  # 显示最后1000个字符
            return False
            
    except Exception as e:
        print(f"❌ 多GPU测试失败: {e}")
        return False

def check_dependencies():
    """检查依赖项"""
    print("=" * 60)
    print("检查依赖项")
    print("=" * 60)
    
    dependencies = [
        "torch",
        "transformers", 
        "numpy"
    ]
    
    missing_deps = []
    
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✓ {dep}")
        except ImportError:
            print(f"❌ {dep} (缺失)")
            missing_deps.append(dep)
    
    # 检查torchrun
    try:
        result = subprocess.run(["torchrun", "--help"], capture_output=True)
        if result.returncode == 0:
            print("✓ torchrun")
        else:
            print("❌ torchrun (不可用)")
            missing_deps.append("torchrun")
    except FileNotFoundError:
        print("❌ torchrun (未找到)")
        missing_deps.append("torchrun")
    
    if missing_deps:
        print(f"\n缺失依赖项: {', '.join(missing_deps)}")
        print("请安装缺失的依赖项后重新运行测试")
        return False
    
    print("\n✓ 所有依赖项检查通过")
    return True

def main():
    """主测试函数"""
    print("PiKV 分布式知识蒸馏测试套件")
    print("=" * 60)
    
    # 检查依赖项
    if not check_dependencies():
        sys.exit(1)
    
    test_results = []
    
    # 运行测试
    tests = [
        ("单GPU蒸馏功能", test_single_gpu_distillation),
        ("torchrun命令执行", test_torchrun_command),
        ("多GPU环境模拟", test_multi_gpu_simulation)
    ]
    
    for test_name, test_func in tests:
        print(f"\n开始测试: {test_name}")
        try:
            result = test_func()
            test_results.append((test_name, result))
        except Exception as e:
            print(f"测试 {test_name} 出现异常: {e}")
            test_results.append((test_name, False))
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✓ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！分布式蒸馏功能正常工作。")
        sys.exit(0)
    else:
        print("⚠️  部分测试失败，请检查错误信息。")
        sys.exit(1)

if __name__ == "__main__":
    main() 