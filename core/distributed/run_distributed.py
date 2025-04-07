import os
import torch
import torch.distributed as dist
from distributed_pikv import DistributedPiKVManager
from distributed_config import distributed_config as dconfig
from config import config

def main():
    # 设置环境变量
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '23456'
    
    # 初始化分布式环境
    dist.init_process_group(
        backend=dconfig['dist_backend'],
        init_method='env://',
        world_size=dconfig['world_size'],
        rank=dconfig['rank']
    )
    
    # 创建分布式PiKV管理器
    pikv_manager = DistributedPiKVManager()
    
    # 生成示例数据
    batch_size = config['batch_size']
    hidden_size = config['hidden_size']
    data = torch.randn(batch_size, hidden_size).to(dconfig['device'])
    target = torch.randn(batch_size, hidden_size).to(dconfig['device'])
    
    # 训练循环
    for epoch in range(config['epochs']):
        loss = pikv_manager.train_step(data, target)
        
        # 只在主进程打印信息
        if dconfig['rank'] == 0:
            print(f"Epoch {epoch+1}/{config['epochs']}, Loss: {loss:.4f}")
        
        # 定期保存检查点
        if (epoch + 1) % dconfig['checkpoint_interval'] == 0:
            pikv_manager.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt")
    
    # 清理分布式环境
    dist.destroy_process_group()

if __name__ == "__main__":
    main() 