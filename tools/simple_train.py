"""
Simplified Training Script
"""

from pointcept.engines.simple_train import TRAINERS
from pointcept.utils.config import Config  # 假设 Config 是用来加载配置文件的

def main():
    # 直接加载配置文件
    config_file = "/root/autodl-tmp/Pointcept/configs/s3dis/semseg-pt-v3m1-1-rpe.py"  # 配置文件路径
    cfg = Config.fromfile(config_file)

    # 构建训练器
    trainer = TRAINERS.build(dict(type=cfg.train.type, cfg=cfg))

    # 开始训练
    trainer.train()

if __name__ == "__main__":
    main()