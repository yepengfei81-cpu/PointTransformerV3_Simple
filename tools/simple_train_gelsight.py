import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pointcept.engines.simple_train import TRAINERS
from pointcept.utils.config import Config


def main():
    config_file = str(project_root / "configs" / "s3dis" / "semseg-pt-v3m1-gelsight.py")
    cfg = Config.fromfile(config_file)

    test_save_dir = project_root / "exp" / "gelsight_test"
    test_save_dir.mkdir(parents=True, exist_ok=True)
    cfg.save_path = str(test_save_dir)
        
    print("=" * 80)
    print("GelSight Contact Position Regression Training")
    print("=" * 80)
    print(f"Config: {config_file}")
    print(f"Save path: {cfg.save_path}")
    print(f"Epochs: {cfg.epoch}")
    print(f"Eval epochs: {cfg.eval_epoch}")
    print("=" * 80 + "\n")
    
    trainer = TRAINERS.build(dict(type=cfg.train.type, cfg=cfg))
    trainer.train()


if __name__ == "__main__":
    main()