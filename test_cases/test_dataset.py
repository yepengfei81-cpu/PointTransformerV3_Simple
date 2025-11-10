import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pointcept.datasets.builder import build_dataset
from pointcept.utils.config import Config


def test_dataset_loading():
    print("=" * 80)
    print("Testing Dataset Loading...")
    print("=" * 80)
    
    cfg = Config.fromfile("/home/ypf/PointTransformerV3_Simple/configs/s3dis/semseg-pt-v3m1-gelsight.py")
    
    train_dataset = build_dataset(cfg.data.train)
    
    print(f"\n✅ Dataset created successfully!")
    print(f"   - Type: {type(train_dataset).__name__}")
    print(f"   - Total samples: {len(train_dataset)}")
    print(f"   - Data root: {train_dataset.data_root}")
    print(f"   - Split: {train_dataset.split}")
    
    print("\n" + "=" * 80)
    print("Testing Single Sample...")
    print("=" * 80)
    
    sample = train_dataset[0]
    
    print(f"\n✅ Sample loaded successfully!")
    print(f"   Keys: {list(sample.keys())}")
    
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"   - {key}: shape={value.shape}, dtype={value.dtype}")
        elif isinstance(value, np.ndarray):
            print(f"   - {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"   - {key}: type={type(value)}, value={value}")
    
    print("\n" + "=" * 80)
    print("Testing Multiple Samples...")
    print("=" * 80)
    
    for i in range(min(3, len(train_dataset))):
        sample = train_dataset[i]
        name = sample.get("name", "unknown")
        coord_shape = sample["coord"].shape if "coord" in sample else "N/A"
        gt_pos = sample.get("gt_position", None)
        cat_id = sample.get("category_id", None)
        
        print(f"\n   Sample {i}:")
        print(f"   - Name: {name}")
        print(f"   - Coord shape: {coord_shape}")
        print(f"   - GT position: {gt_pos}")
        print(f"   - Category ID: {cat_id}")
    
    print("\n" + "=" * 80)
    print("✅ All dataset tests passed!")
    print("=" * 80)


if __name__ == "__main__":
    test_dataset_loading()