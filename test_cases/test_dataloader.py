import sys
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from functools import partial

sys.path.insert(0, str(Path(__file__).parent.parent))

from pointcept.datasets.builder import build_dataset
from pointcept.datasets.utils import point_collate_fn
from pointcept.utils.config import Config


def test_real_dataloader():
    print("=" * 80)
    print("Testing Real DataLoader (with point_collate_fn)...")
    print("=" * 80)
    
    cfg = Config.fromfile("/home/ypf/PointTransformerV3_Simple/configs/s3dis/semseg-pt-v3m1-gelsight.py")
    # cfg = Config.fromfile("/home/ypf/PointTransformerV3_Simple/configs/s3dis/semseg-pt-v3m1-1-rpe.py")

    train_dataset = build_dataset(cfg.data.train)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        collate_fn=partial(point_collate_fn, mix_prob=0.0),
        pin_memory=False,
    )
    
    print(f"\n✅ DataLoader created successfully!")
    print(f"   - Batch size: 4")
    print(f"   - Total batches: {len(train_loader)}")
    print(f"   - Collate function: point_collate_fn")
    
    print("\n" + "=" * 80)
    print("Testing Batch Loading...")
    print("=" * 80)
    
    for i, batch in enumerate(train_loader):
        if i >= 2:
            break
        
        print(f"\n   Batch {i}:")
        print(f"   - Type: {type(batch)}")
        print(f"   - Keys: {list(batch.keys())}")
        
        print(f"\n   Tensor shapes:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"      - {key}: {value.shape}, dtype={value.dtype}")
            elif isinstance(value, list):
                print(f"      - {key}: list of {len(value)} items")
            else:
                print(f"      - {key}: {type(value)}")
        
        if "offset" in batch:
            offset = batch["offset"]
            print(f"\n   Offset analysis:")
            print(f"      - Offset values: {offset}")
            
            num_samples = len(offset)
            print(f"      - Number of samples: {num_samples}")
            
            print(f"      - Sample 0: {offset[0]} points")
            for i in range(1, len(offset)):
                n_points = offset[i] - offset[i - 1]
                print(f"      - Sample {i}: {n_points} points")
        
        if "gt_position" in batch:
            print(f"\n   Ground truth:")
            for j in range(batch['gt_position'].shape[0]):
                print(f"      - Sample {j}: {batch['gt_position'][j]}")
        
        if "name" in batch:
            print(f"\n   Sample names:")
            for j, name in enumerate(batch['name']):
                print(f"      - Sample {j}: {name}")
    
    print("\n" + "=" * 80)
    print("✅ All real DataLoader tests passed!")
    print("=" * 80)


if __name__ == "__main__":
    test_real_dataloader()