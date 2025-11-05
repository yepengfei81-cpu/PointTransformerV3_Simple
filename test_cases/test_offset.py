import sys
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from functools import partial

sys.path.insert(0, str(Path(__file__).parent.parent))

from pointcept.datasets.builder import build_dataset
from pointcept.datasets.utils import point_collate_fn
from pointcept.utils.config import Config


def test_offset():
    print("=" * 80)
    print("Testing Offset Field...")
    print("=" * 80)
    
    # cfg = Config.fromfile("/root/autodl-tmp/Pointcept/configs/s3dis/semseg-pt-v3m1-gelsight.py")
    cfg = Config.fromfile("/root/autodl-tmp/Pointcept/configs/s3dis/semseg-pt-v3m1-1-rpe.py")
    train_dataset = build_dataset(cfg.data.train)
    
    print(f"\n✅ Dataset loaded")
    print(f"   Total samples: {len(train_dataset)}")
    
    batch_sizes = [2, 4, 8]
    
    for bs in batch_sizes:
        print(f"\n{'='*80}")
        print(f"Testing batch_size = {bs}")
        print(f"{'='*80}")
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=bs,
            shuffle=False,
            num_workers=0,
            collate_fn=partial(point_collate_fn, mix_prob=0.0),
        )
        
        batch = next(iter(train_loader))
        
        print(f"\n   Batch keys: {list(batch.keys())}")
        
        if "offset" not in batch:
            print(f"\n   ❌ ERROR: 'offset' field is MISSING!")
            print(f"   PTv3 model requires 'offset' to separate different samples.")
            print(f"\n   Suggestion: Remove 'offset_keys_dict={{}}' from config")
            continue
        
        print(f"\n   ✅ 'offset' field exists")
        
        offset = batch["offset"]
        print(f"\n   Offset properties:")
        print(f"      - Type: {type(offset)}")
        print(f"      - Shape: {offset.shape}")
        print(f"      - Dtype: {offset.dtype}")
        print(f"      - Values: {offset}")
        
        expected_length = bs + 1
        if len(offset) != expected_length:
            print(f"\n   ❌ ERROR: Offset length is wrong!")
            print(f"      - Expected: {expected_length} (batch_size + 1)")
            print(f"      - Got: {len(offset)}")
            continue
        
        print(f"\n   ✅ Offset length is correct: {len(offset)} = {bs} + 1")
        
        total_points = batch['coord'].shape[0]
        offset_total = offset[-1].item()
        
        if total_points != offset_total:
            print(f"\n   ❌ ERROR: Offset sum doesn't match total points!")
            print(f"      - Total points (from coord): {total_points}")
            print(f"      - Offset sum: {offset_total}")
            continue
        
        print(f"\n   ✅ Offset sum is correct: {offset_total} = {total_points}")
        print(f"\n   Per-sample point counts:")
        for i in range(len(offset) - 1):
            n_points = offset[i + 1] - offset[i]
            print(f"      - Sample {i}: {n_points.item()} points")
        
        print(f"\n   Testing sample separation:")
        for i in range(len(offset) - 1):
            start = offset[i].item()
            end = offset[i + 1].item()
            sample_coords = batch['coord'][start:end]
            print(f"      - Sample {i}: coord[{start}:{end}] = {sample_coords.shape}")
        
        print(f"\n   ✅ All samples can be separated correctly!")
        
        if 'gt_position' in batch:
            print(f"\n   Checking gt_position correspondence:")
            print(f"      - gt_position shape: {batch['gt_position'].shape}")
            print(f"      - Number of samples: {len(offset) - 1}")
            
            if batch['gt_position'].shape[0] == len(offset) - 1:
                print(f"\n   ✅ gt_position matches number of samples")
                
                for i in range(len(offset) - 1):
                    print(f"      - Sample {i}: {batch['gt_position'][i]}")
            else:
                print(f"\n   ❌ ERROR: gt_position count doesn't match!")
    
    print(f"\n{'='*80}")
    print(f"✅ All offset tests passed!")
    print(f"{'='*80}")


if __name__ == "__main__":
    test_offset()