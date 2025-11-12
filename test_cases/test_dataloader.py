import sys
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from functools import partial

sys.path.insert(0, str(Path(__file__).parent.parent))

from pointcept.datasets.builder import build_dataset
from pointcept.datasets.utils import point_collate_fn
from pointcept.utils.config import Config


def print_separator(title="", width=80):
    """打印分隔线"""
    if title:
        print("\n" + "=" * width)
        print(title.center(width))
        print("=" * width)
    else:
        print("=" * width)


def analyze_batch(batch, batch_idx):
    """详细分析一个 batch 的内容"""
    print(f"\n{'─' * 80}")
    print(f"📦 Batch {batch_idx} Analysis")
    print(f"{'─' * 80}")
    
    print(f"\n1️⃣  基本信息:")
    print(f"   - Type: {type(batch)}")
    print(f"   - Keys: {list(batch.keys())}")
    
    # 🔥 分析所有张量
    print(f"\n2️⃣  张量形状:")
    tensor_keys = ['coord', 'grid_coord', 'feat', 'offset', 'gt_position', 
                   'category_id', 'batch', 'parent_coord', 'parent_color',
                   'norm_offset', 'norm_scale']
    
    for key in tensor_keys:
        if key in batch:
            value = batch[key]
            if isinstance(value, torch.Tensor):
                print(f"   ✅ {key:20s}: shape={str(value.shape):20s} dtype={value.dtype}")
            elif isinstance(value, list):
                print(f"   ✅ {key:20s}: list of {len(value)} items")
            else:
                print(f"   ⚠️  {key:20s}: {type(value)}")
    
    # 🔥 分析 offset（局部点云的分割）
    if "offset" in batch:
        offset = batch["offset"]
        print(f"\n3️⃣  Offset 分析 (局部点云分割):")
        print(f"   - Offset tensor: {offset}")
        print(f"   - Batch size: {len(offset)}")
        
        print(f"\n   各样本的局部点云点数:")
        print(f"      Sample 0: {offset[0]:6d} points")
        for i in range(1, len(offset)):
            n_points = offset[i] - offset[i - 1]
            print(f"      Sample {i}: {n_points:6d} points")
        
        total_local_points = offset[-1].item() if len(offset) > 0 else 0
        print(f"\n   总局部点数: {total_local_points}")
    
    # 🔥 分析父点云
    if "parent_coord" in batch:
        parent_coord = batch["parent_coord"]
        print(f"\n4️⃣  父点云分析:")
        print(f"   - parent_coord shape: {parent_coord.shape}")
        print(f"   - parent_color shape: {batch['parent_color'].shape if 'parent_color' in batch else 'N/A'}")
        
        if "batch" in batch:
            batch_idx_tensor = batch["batch"]
            
            # 统计每个样本的父点云点数
            unique_batches = torch.unique(batch_idx_tensor)
            print(f"\n   各样本的父点云点数:")
            for b_idx in unique_batches:
                mask = batch_idx_tensor == b_idx
                n_parent_points = mask.sum().item()
                print(f"      Sample {b_idx}: {n_parent_points:6d} points")
            
            print(f"\n   总父点云点数: {len(batch_idx_tensor)}")
    
    # 🔥 分析归一化参数
    if "norm_offset" in batch or "norm_scale" in batch:
        print(f"\n5️⃣  归一化参数:")
        if "norm_offset" in batch:
            norm_offset = batch["norm_offset"]
            print(f"   - norm_offset shape: {norm_offset.shape}")
            if norm_offset.dim() == 2:
                for i in range(norm_offset.shape[0]):
                    print(f"      Sample {i}: {norm_offset[i].tolist()}")
        
        if "norm_scale" in batch:
            norm_scale = batch["norm_scale"]
            print(f"   - norm_scale shape: {norm_scale.shape}")
            if norm_scale.dim() == 1:
                for i in range(norm_scale.shape[0]):
                    print(f"      Sample {i}: {norm_scale[i].item():.6f}")
    
    # 🔥 分析 GT
    if "gt_position" in batch:
        gt_position = batch["gt_position"]
        print(f"\n6️⃣  Ground Truth:")
        print(f"   - gt_position shape: {gt_position.shape}")
        for j in range(gt_position.shape[0]):
            print(f"      Sample {j}: [{gt_position[j, 0]:.6f}, {gt_position[j, 1]:.6f}, {gt_position[j, 2]:.6f}]")
    
    # 🔥 分析类别
    if "category_id" in batch:
        category_id = batch["category_id"]
        print(f"\n7️⃣  类别 ID:")
        print(f"   - category_id shape: {category_id.shape}")
        
        category_names = {0: "Scissors", 1: "Cup", 2: "Avocado"}
        for j in range(category_id.shape[0]):
            cat_id = category_id[j].item()
            cat_name = category_names.get(cat_id, "Unknown")
            print(f"      Sample {j}: {cat_id} ({cat_name})")
    
    # 🔥 分析样本名称
    if "name" in batch:
        print(f"\n8️⃣  样本名称:")
        for j, name in enumerate(batch['name']):
            print(f"      Sample {j}: {name}")
    
    print(f"\n{'─' * 80}\n")


def test_train_dataloader():
    """测试训练集 DataLoader"""
    print_separator("🚂 测试训练集 DataLoader")
    
    cfg = Config.fromfile("/home/ypf/PointTransformerV3_Simple/configs/s3dis/semseg-pt-v3m1-gelsight.py")
    
    print(f"\n📂 加载训练集...")
    train_dataset = build_dataset(cfg.data.train)
    
    print(f"   ✅ 数据集加载成功!")
    print(f"      - 数据集类型: {type(train_dataset).__name__}")
    print(f"      - 样本数量: {len(train_dataset)}")
    print(f"      - Split: {train_dataset.split}")
    print(f"      - Parent cache size: {train_dataset.max_cache_size}")
    
    batch_size = 4
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=partial(point_collate_fn, mix_prob=0.0),
        pin_memory=False,
    )
    
    print(f"\n📦 创建 DataLoader:")
    print(f"   ✅ DataLoader 创建成功!")
    print(f"      - Batch size: {batch_size}")
    print(f"      - Total batches: {len(train_loader)}")
    print(f"      - Collate function: point_collate_fn (mix_prob=0.0)")
    
    print_separator("🔍 测试前 2 个 Batch")
    
    for i, batch in enumerate(train_loader):
        if i >= 2:
            break
        
        analyze_batch(batch, i)
    
    print_separator("✅ 训练集 DataLoader 测试完成")


def test_val_dataloader():
    """测试验证集 DataLoader"""
    print_separator("🔍 测试验证集 DataLoader")
    
    cfg = Config.fromfile("/home/ypf/PointTransformerV3_Simple/configs/s3dis/semseg-pt-v3m1-gelsight.py")
    
    print(f"\n📂 加载验证集...")
    val_dataset = build_dataset(cfg.data.val)
    
    print(f"   ✅ 数据集加载成功!")
    print(f"      - 数据集类型: {type(val_dataset).__name__}")
    print(f"      - 样本数量: {len(val_dataset)}")
    print(f"      - Split: {val_dataset.split}")
    
    batch_size = 2
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=partial(point_collate_fn, mix_prob=0.0),
        pin_memory=False,
    )
    
    print(f"\n📦 创建 DataLoader:")
    print(f"   ✅ DataLoader 创建成功!")
    print(f"      - Batch size: {batch_size}")
    print(f"      - Total batches: {len(val_loader)}")
    
    print_separator("🔍 测试前 1 个 Batch")
    
    for i, batch in enumerate(val_loader):
        if i >= 1:
            break
        
        analyze_batch(batch, i)
    
    print_separator("✅ 验证集 DataLoader 测试完成")


def test_test_dataloader():
    """测试测试集 DataLoader（注意：测试集返回的数据结构不同）"""
    print_separator("🧪 测试测试集 DataLoader")
    
    cfg = Config.fromfile("/home/ypf/PointTransformerV3_Simple/configs/s3dis/semseg-pt-v3m1-gelsight.py")
    
    print(f"\n📂 加载测试集...")
    test_dataset = build_dataset(cfg.data.test)
    
    print(f"   ✅ 数据集加载成功!")
    print(f"      - 数据集类型: {type(test_dataset).__name__}")
    print(f"      - 样本数量: {len(test_dataset)}")
    print(f"      - Split: {test_dataset.split}")
    print(f"      - Test mode: {test_dataset.test_mode}")
    
    print(f"\n⚠️  注意：测试集返回的数据结构不同（包含 fragment_list）")
    
    # 测试单个样本
    print_separator("🔍 测试单个样本")
    
    sample = test_dataset[0]
    
    print(f"\n📦 Sample 0:")
    print(f"   - Type: {type(sample)}")
    print(f"   - Outer keys: {list(sample.keys())}")
    
    # 🔥 分析外层字段
    print(f"\n1️⃣  外层字段:")
    for key in ['gt_position', 'name', 'category_id', 'parent_coord', 
                'parent_color', 'norm_offset', 'norm_scale']:
        if key in sample:
            value = sample[key]
            if isinstance(value, torch.Tensor):
                print(f"   ✅ {key:20s}: shape={value.shape}, dtype={value.dtype}")
            elif isinstance(value, np.ndarray):
                print(f"   ✅ {key:20s}: shape={value.shape}, dtype={value.dtype}")
            elif isinstance(value, (int, float)):
                print(f"   ✅ {key:20s}: {value}")
            else:
                print(f"   ✅ {key:20s}: {type(value)}")
    
    # 🔥 分析 fragment_list
    if 'fragment_list' in sample:
        print(f"\n2️⃣  Fragment List:")
        print(f"   - Number of fragments: {len(sample['fragment_list'])}")
        
        for i, fragment in enumerate(sample['fragment_list']):
            print(f"\n   Fragment {i}:")
            print(f"      Keys: {list(fragment.keys())}")
            
            for key in ['coord', 'grid_coord', 'feat', 'index', 'name',
                       'parent_coord', 'parent_color', 'norm_offset', 'norm_scale']:
                if key in fragment:
                    value = fragment[key]
                    if isinstance(value, torch.Tensor):
                        print(f"      - {key:20s}: shape={value.shape}")
    
    print_separator("✅ 测试集 DataLoader 测试完成")


def test_collate_fn():
    """测试 collate_fn 是否正确处理父点云"""
    print_separator("🔧 测试 point_collate_fn")
    
    cfg = Config.fromfile("/home/ypf/PointTransformerV3_Simple/configs/s3dis/semseg-pt-v3m1-gelsight.py")
    
    print(f"\n📂 加载训练集...")
    train_dataset = build_dataset(cfg.data.train)
    
    # 手动获取几个样本
    print(f"\n📦 手动获取 2 个样本...")
    samples = [train_dataset[i] for i in range(2)]
    
    print(f"\n   Sample 0 keys: {list(samples[0].keys())}")
    print(f"   Sample 1 keys: {list(samples[1].keys())}")
    
    # 使用 collate_fn
    print(f"\n🔧 调用 point_collate_fn...")
    batch = point_collate_fn(samples, mix_prob=0.0)
    
    print(f"\n   ✅ Collate 成功!")
    print(f"   Batch keys: {list(batch.keys())}")
    
    # 分析结果
    analyze_batch(batch, 0)
    
    print_separator("✅ point_collate_fn 测试完成")


def main():
    """运行所有测试"""
    import numpy as np  # 需要在这里导入（用于测试集分析）
    
    print("\n" + "🚀" * 40)
    print("开始测试带父点云的 DataLoader".center(80))
    print("🚀" * 40)
    
    try:
        # 1. 测试训练集
        test_train_dataloader()
        
        # 2. 测试验证集
        test_val_dataloader()
        
        # 3. 测试 collate_fn
        test_collate_fn()
        
        # 4. 测试测试集（可选）
        # test_test_dataloader()
        
        print("\n" + "🎉" * 40)
        print("所有测试通过！".center(80))
        print("🎉" * 40 + "\n")
        
    except Exception as e:
        print("\n" + "❌" * 40)
        print(f"测试失败: {e}".center(80))
        print("❌" * 40 + "\n")
        raise


if __name__ == "__main__":
    main()