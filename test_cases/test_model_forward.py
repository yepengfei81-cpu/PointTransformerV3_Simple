import sys
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from functools import partial

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pointcept.datasets.builder import build_dataset
from pointcept.datasets.utils import point_collate_fn
from pointcept.utils.config import Config
from pointcept.models import build_model


def recursive_to_device(data, device):
    if isinstance(data, dict):
        return {key: recursive_to_device(value, device) for key, value in data.items()}
    elif isinstance(data, (list, tuple)):
        result = [recursive_to_device(item, device) for item in data]
        return type(data)(result)
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        return data
    
def test_model_forward():
    print("=" * 80)
    print("Testing Model Forward Pass...")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device: {device}")
    if not torch.cuda.is_available():
        print("   ⚠️  Warning: CUDA not available, but PTv3 requires GPU!")
        print("   ⚠️  This test will likely fail on CPU.")
    
    config_path = project_root / "configs" / "s3dis" / "semseg-pt-v3m1-gelsight.py"
    print(f"\n📄 Config file: {config_path}")
    
    if not config_path.exists():
        print(f"   ❌ Config file not found!")
        return
    
    cfg = Config.fromfile(str(config_path))
    
    # 🔥 新增：打印模型配置
    print(f"\n📋 Model configuration:")
    if hasattr(cfg, 'model'):
        if 'use_parent_cloud' in cfg.model:
            print(f"   use_parent_cloud: {cfg.model['use_parent_cloud']}")
        if 'fusion_type' in cfg.model:
            print(f"   fusion_type: {cfg.model['fusion_type']}")
        if 'parent_backbone' in cfg.model:
            if cfg.model['parent_backbone'] is None:
                print(f"   parent_backbone: None (shared weights)")
            else:
                print(f"   parent_backbone: Independent")
    
    # Build dataset
    print(f"\n🔧 Building dataset...")
    try:
        train_dataset = build_dataset(cfg.data.train)
        train_loader = DataLoader(
            train_dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,
            collate_fn=partial(point_collate_fn, mix_prob=0.0),
        )
        print(f"   ✅ Dataset built successfully")
        print(f"   Total samples: {len(train_dataset)}")
    except Exception as e:
        print(f"   ❌ Dataset build failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Build model
    print(f"\n🔧 Building model...")
    try:
        model = build_model(cfg.model)
        model = model.to(device)
        model.eval()
        print(f"   ✅ Model built successfully")
        print(f"   Type: {type(model).__name__}")
        print(f"   Device: {next(model.parameters()).device}")
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   Total params: {total_params:,}")
        print(f"   Trainable params: {trainable_params:,}")
        
    except Exception as e:
        print(f"   ❌ Model build failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Load batch
    print(f"\n🔧 Loading batch...")
    try:
        batch = next(iter(train_loader))
    except Exception as e:
        print(f"   ❌ Batch load failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\n   📋 Checking nested structure...")
    top_level_required = ['local', 'parent', 'gt_position']
    missing_required = []

    for key in top_level_required:
        if key in batch:
            if isinstance(batch[key], dict):
                print(f"      ✅ {key}: {list(batch[key].keys())}")
            elif isinstance(batch[key], torch.Tensor):
                print(f"      ✅ {key}: {batch[key].shape}")
        else:
            print(f"      ❌ {key}: MISSING!")
            missing_required.append(key)

    if missing_required:
        print(f"\n   ❌ Missing required fields: {missing_required}")
        return

    if "local" in batch:
        print(f"\n   📋 Local cloud:")
        for key in ['coord', 'grid_coord', 'feat', 'offset']:
            if key in batch["local"] and isinstance(batch["local"][key], torch.Tensor):
                print(f"      {key}: {batch['local'][key].shape}")

    if "parent" in batch:
        print(f"\n   📋 Parent cloud:")
        for key in ['coord', 'grid_coord', 'feat', 'offset']:
            if key in batch["parent"] and isinstance(batch["parent"][key], torch.Tensor):
                print(f"      {key}: {batch['parent'][key].shape}")

    print(f"\n   📋 Normalization:")
    for key in ['norm_offset', 'norm_scale', 'category_id']:
        if key in batch and isinstance(batch[key], torch.Tensor):
            print(f"      {key}: {batch[key].shape}")

    print(f"\n   📊 Point cloud stats:")
    if "local" in batch and "offset" in batch["local"]:
        local_offset = batch["local"]["offset"]
        print(f"      Local - Total: {batch['local']['coord'].shape[0]}, Batch: {len(local_offset)}")

    if "parent" in batch and "offset" in batch["parent"]:
        parent_offset = batch["parent"]["offset"]
        print(f"      Parent - Total: {batch['parent']['coord'].shape[0]}, Batch: {len(parent_offset)}")            
    
    # Move to device
    batch = recursive_to_device(batch, device)
    # for key in batch.keys():
    #     if isinstance(batch[key], torch.Tensor):
    #         batch[key] = batch[key].to(device)
    
    print(f"\n   ✅ Batch moved to device: {device}")
    
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
                    print(f"      - {key}: shape={value.shape}, dtype={value.dtype}, device={value.device}")
            
            # 预测结果分析
            if "pred_position" in output and "gt_position" in batch:
                pred = output["pred_position"].cpu()
                gt_norm = batch["gt_position"].cpu()
                
                # 🔥 手动反归一化 GT（用于显示）
                if "norm_offset" in batch and "norm_scale" in batch:
                    norm_offset = batch["norm_offset"].cpu()
                    norm_scale = batch["norm_scale"].cpu()
                    gt = gt_norm * norm_scale + norm_offset  # 反归一化
                else:
                    gt = gt_norm
                
                # 显示
                for i in range(len(pred)):
                    pred_i = pred[i]
                    gt_i = gt[i]
                    error = torch.norm(pred_i - gt_i).item()
                    print(f"   Sample {i}: Pred={pred_i.tolist()}, GT={gt_i.tolist()}, Error={error:.4f}")
            
            # 检查 loss
            if "loss" in output:
                loss = output["loss"]
                print(f"\n      📉 Loss: {loss.item():.6f}")
                
                if torch.isnan(loss):
                    print(f"         ❌ Loss is NaN!")
                elif torch.isinf(loss):
                    print(f"         ❌ Loss is Inf!")
                else:
                    print(f"         ✅ Loss is valid")
            
            # 🔥 新增：检查融合特征（如果 return_point=True）
            if "local_feat" in output:
                print(f"\n      🔍 Feature analysis:")
                print(f"         - local_feat: {output['local_feat'].shape}")
                if "parent_feat" in output and output["parent_feat"] is not None:
                    print(f"         - parent_feat: {output['parent_feat'].shape}")
                    print(f"         ✅ Cross-cloud fusion is working")
                else:
                    print(f"         ⚠️  No parent features (single cloud mode)")
                if "global_feat" in output:
                    print(f"         - global_feat: {output['global_feat'].shape}")
                    
        elif isinstance(output, torch.Tensor):
            print(f"   Output shape: {output.shape}")
        
        print(f"\n" + "=" * 80)
        print(f"✅ Test completed successfully!")
        print(f"=" * 80)
        
    except Exception as e:
        print(f"\n   ❌ Forward pass failed!")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error message: {str(e)}")
        print(f"\n   Full traceback:")
        import traceback
        traceback.print_exc()
        
        print(f"\n" + "=" * 80)
        print(f"❌ Test failed!")
        print(f"=" * 80)


if __name__ == "__main__":
    test_model_forward()