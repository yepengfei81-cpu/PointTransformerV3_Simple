import sys
import os
from pathlib import Path

# Get project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pointcept.engines.simple_train import TRAINERS
from pointcept.utils.config import Config


def main():
    print("=" * 80)
    print("Testing RegressionTrainer")
    print("=" * 80)
    
    # 加载配置文件
    config_file = str(project_root / "configs" / "s3dis" / "semseg-pt-v3m1-gelsight.py")
    cfg = Config.fromfile(config_file)
    
    print(f"\n📋 Original config:")
    print(f"   Save path: {cfg.save_path}")
    print(f"   Eval epochs: {cfg.eval_epoch}")
    
    # 🔥 修改配置
    cfg.eval_epoch = 2
    cfg.epoch = 2
    cfg.enable_wandb = False
    cfg.mix_prob = 0.0
    
    # 🔥 设置绝对路径并创建目录
    test_save_dir = project_root / "exp" / "gelsight_test"
    test_save_dir.mkdir(parents=True, exist_ok=True)
    cfg.save_path = str(test_save_dir)
    
    print(f"\n📋 Modified config:")
    print(f"   Save path: {cfg.save_path}")
    print(f"   Eval epochs: {cfg.eval_epoch}")
    print(f"   Mix prob: {cfg.mix_prob}")
    
    print(f"\n🔧 Building trainer...")
    trainer = TRAINERS.build(dict(type=cfg.train.type, cfg=cfg))
    
    print(f"\n🚀 Starting training...")
    print("=" * 80 + "\n")
    
    trainer.train()
    
    print("\n" + "=" * 80)
    print("✅ Test completed successfully!")
    print(f"📂 Logs saved to: {cfg.save_path}/train.log")
    print("=" * 80)


if __name__ == "__main__":
    main()