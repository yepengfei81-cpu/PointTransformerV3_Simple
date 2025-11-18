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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device: {device}")
    if not torch.cuda.is_available():
        print("   ⚠️  Warning: CUDA not available, but PTv3 requires GPU!")
        print("   ⚠️  This test will likely fail on CPU.")
    
    cfg = Config.fromfile("/root/autodl-tmp/Pointcept/configs/s3dis/semseg-pt-v3m1-gelsight.py")
    
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
        model = model.to(device)
        model.eval()
        print(f"   ✅ Model built successfully")
        print(f"   Type: {type(model).__name__}")
        print(f"   Device: {next(model.parameters()).device}")
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
    
    for key in batch.keys():
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)
    
    print(f"   - coord device: {batch['coord'].device}")
    
    # Test forward
    print(f"\n🔧 Testing forward pass...")
    try:
        with torch.no_grad():
            output = model(batch)
        
        print(f"\n   ✅ Forward pass successful!")
        print(f"   Output type: {type(output)}")
        
        if isinstance(output, dict):
            print(f"   Output keys: {list(output.keys())}")

            if "loss" in output:
                loss = output["loss"]
                print(f"\n   📉 Loss Information:")
                print(f"      - loss value: {loss.item():.6f}")
                
                if torch.isnan(loss):
                    print(f"      ❌ Loss is NaN!")
                elif torch.isinf(loss):
                    print(f"      ❌ Loss is Inf!")
                else:
                    print(f"      ✅ Loss is valid")
            
            if "pred_position" in output and "gt_position" in batch:
                pred_norm = output["pred_position"].cpu()
                gt_norm = batch["gt_position"].cpu()
                
                print(f"\n   📍 Predictions (Normalized Space):")
                for i in range(len(pred_norm)):
                    pred_i = pred_norm[i]
                    gt_i = gt_norm[i]
                    error_norm = torch.norm(pred_i - gt_i).item()
                    print(f"      Sample {i}:")
                    print(f"         Pred: [{pred_i[0]:.6f}, {pred_i[1]:.6f}, {pred_i[2]:.6f}]")
                    print(f"         GT:   [{gt_i[0]:.6f}, {gt_i[1]:.6f}, {gt_i[2]:.6f}]")
                    print(f"         Error: {error_norm:.6f}")
                
                if "pcd_min" in batch and "pcd_size" in batch:
                    pcd_min = batch["pcd_min"].cpu()
                    pcd_size = batch["pcd_size"].cpu()
                    
                    # real = norm * size + min
                    pred = pred_norm * pcd_size + pcd_min
                    gt = gt_norm * pcd_size + pcd_min
                    
                    print(f"\n   📍 Predictions (Real Space):")
                    for i in range(len(pred)):
                        pred_i = pred[i]
                        gt_i = gt[i]
                        error = torch.norm(pred_i - gt_i).item()
                        print(f"      Sample {i}:")
                        print(f"         Pred: [{pred_i[0]:.6f}, {pred_i[1]:.6f}, {pred_i[2]:.6f}] m")
                        print(f"         GT:   [{gt_i[0]:.6f}, {gt_i[1]:.6f}, {gt_i[2]:.6f}] m")
                        print(f"         Error: {error:.6f} m ({error*1000:.2f} mm)")

            for key, value in output.items():
                if isinstance(value, torch.Tensor):
                    print(f"      - {key}: shape={value.shape}, dtype={value.dtype}, device={value.device}")
                    
                    if key == "position_pred" and "gt_position" in batch:
                        print(f"\n   📊 Predictions vs Ground Truth:")
                        pred = value.cpu()
                        gt = batch["gt_position"].cpu()
                        for i in range(len(pred)):
                            print(f"      Sample {i}: pred={pred[i].tolist()}, gt={gt[i].tolist()}")
                            
        elif isinstance(output, torch.Tensor):
            print(f"   Output shape: {output.shape}")
        
    except Exception as e:
        print(f"\n   ❌ Forward pass failed!")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error message: {str(e)}")
        print(f"\n   Full traceback:")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_model_forward()