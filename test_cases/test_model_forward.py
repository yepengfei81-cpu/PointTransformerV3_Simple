import sys
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from functools import partial

sys.path.insert(0, str(Path(__file__).parent.parent))

from pointcept.datasets.builder import build_dataset
from pointcept.datasets.utils import point_collate_fn
from pointcept.utils.config import Config
from pointcept.models import build_model


def test_model_forward():
    print("=" * 80)
    print("Testing Model Forward Pass...")
    print("=" * 80)
    
    cfg = Config.fromfile("/root/autodl-tmp/Pointcept/configs/s3dis/semseg-pt-v3m1-gelsight.py")
    # cfg = Config.fromfile("/root/autodl-tmp/Pointcept/configs/s3dis/semseg-pt-v3m1-1-rpe.py")
    # Build dataset
    train_dataset = build_dataset(cfg.data.train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=partial(point_collate_fn, mix_prob=0.0),
    )
    
    # Build model
    print(f"\n🔧 Building model...")
    try:
        model = build_model(cfg.model)
        model.eval()
        print(f"   ✅ Model built successfully")
        print(f"   Type: {type(model).__name__}")
    except Exception as e:
        print(f"   ❌ Model build failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Load batch
    print(f"\n🔧 Loading batch...")
    batch = next(iter(train_loader))
    
    print(f"   ✅ Batch loaded")
    print(f"   Keys: {list(batch.keys())}")
    print(f"   - coord: {batch['coord'].shape}")
    print(f"   - offset: {batch['offset']}")
    print(f"   - gt_position: {batch['gt_position'].shape}")
    
    # Test forward
    print(f"\n🔧 Testing forward pass...")
    try:
        with torch.no_grad():
            output = model(batch)
        
        print(f"\n   ✅ Forward pass successful!")
        print(f"   Output type: {type(output)}")
        
        if isinstance(output, dict):
            print(f"   Output keys: {list(output.keys())}")
            for key, value in output.items():
                if isinstance(value, torch.Tensor):
                    print(f"      - {key}: shape={value.shape}, dtype={value.dtype}")
        elif isinstance(output, torch.Tensor):
            print(f"   Output shape: {output.shape}")
        
    except Exception as e:
        print(f"\n   ❌ Forward pass failed!")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error message: {str(e)}")
        print(f"\n   Full traceback:")
        import traceback
        traceback.print_exc()
        
        print(f"\n   💡 This is expected!")
        print(f"   PTv3's DefaultSegmentor outputs per-point predictions,")
        print(f"   but you need per-sample predictions (3D position).")
        print(f"\n   Next step: Modify the model to output (batch_size, 3)")


if __name__ == "__main__":
    test_model_forward()