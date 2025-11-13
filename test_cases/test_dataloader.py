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
    tensor_keys = [
        'coord', 'grid_coord', 'feat', 'color', 'offset', 'grid_size',
        'gt_position', 'category_id', 'norm_offset', 'norm_scale',
        'parent_coord', 'parent_color', 'parent_grid_coord', 
        'parent_grid_size', 'parent_batch', 'parent_offset',  # 🔥 新增父点云字段
        'batch', 'name'
    ]
    
    for key in tensor_keys:
        if key in batch:
            value = batch[key]
            if isinstance(value, torch.Tensor):
                print(f"   ✅ {key:20s}: shape={str(value.shape):20s} dtype={value.dtype}")
            elif isinstance(value, list):
                print(f"   ✅ {key:20s}: list of {len(value)} items")
            elif isinstance(value, (int, float)):
                print(f"   ✅ {key:20s}: {type(value).__name__}={value}")
            else:
                print(f"   ⚠️  {key:20s}: {type(value)}")
    
    # 🔥 分析局部点云的 offset
    if "offset" in batch:
        offset = batch["offset"]
        print(f"\n3️⃣  局部点云 Offset 分析:")
        print(f"   - Offset tensor: {offset.tolist()}")
        print(f"   - Batch size: {len(offset) - 1}")
        
        print(f"\n   各样本的局部点云点数:")
        for i in range(len(offset) - 1):
            n_points = offset[i + 1] - offset[i]
            print(f"      Sample {i}: {n_points:6d} points (range: [{offset[i]:6d}, {offset[i+1]:6d}))")
        
        total_local_points = offset[-1].item()
        print(f"\n   ✅ 总局部点数: {total_local_points}")
        
        # 🔥 验证 offset 的正确性
        if "coord" in batch:
            actual_points = batch["coord"].shape[0]
            if actual_points == total_local_points:
                print(f"   ✅ Offset 验证通过: coord.shape[0] ({actual_points}) == offset[-1] ({total_local_points})")
            else:
                print(f"   ❌ Offset 验证失败: coord.shape[0] ({actual_points}) != offset[-1] ({total_local_points})")
    
    # 🔥 分析父点云的 parent_offset
    if "parent_offset" in batch:
        parent_offset = batch["parent_offset"]
        print(f"\n4️⃣  父点云 Parent Offset 分析:")
        print(f"   - Parent Offset tensor: {parent_offset.tolist()}")
        print(f"   - Batch size: {len(parent_offset) - 1}")
        
        print(f"\n   各样本的父点云点数:")
        for i in range(len(parent_offset) - 1):
            n_points = parent_offset[i + 1] - parent_offset[i]
            print(f"      Sample {i}: {n_points:6d} points (range: [{parent_offset[i]:6d}, {parent_offset[i+1]:6d}))")
        
        total_parent_points = parent_offset[-1].item()
        print(f"\n   ✅ 总父点云点数: {total_parent_points}")
        
        # 🔥 验证 parent_offset 的正确性
        if "parent_coord" in batch:
            actual_points = batch["parent_coord"].shape[0]
            if actual_points == total_parent_points:
                print(f"   ✅ Parent Offset 验证通过: parent_coord.shape[0] ({actual_points}) == parent_offset[-1] ({total_parent_points})")
            else:
                print(f"   ❌ Parent Offset 验证失败: parent_coord.shape[0] ({actual_points}) != parent_offset[-1] ({total_parent_points})")
        
        # 🔥 与 parent_batch 对比验证
        if "parent_batch" in batch:
            parent_batch = batch["parent_batch"]
            print(f"\n   🔍 与 parent_batch 对比验证:")
            
            # 从 parent_batch 计算预期的 parent_offset
            counts = torch.bincount(parent_batch.long())
            expected_offset = torch.cat([
                torch.tensor([0]),
                torch.cumsum(counts, dim=0)
            ])
            
            print(f"      - 从 parent_batch 计算的 offset: {expected_offset.tolist()}")
            print(f"      - 实际 parent_offset:           {parent_offset.tolist()}")
            
            if torch.equal(expected_offset, parent_offset):
                print(f"      ✅ Parent Offset 与 parent_batch 一致！")
            else:
                print(f"      ❌ Parent Offset 与 parent_batch 不一致！")
    else:
        print(f"\n4️⃣  父点云 Parent Offset 分析:")
        print(f"   ❌ 缺少 'parent_offset' 字段！")
        print(f"   ⚠️  请检查 point_collate_fn 是否正确生成了 parent_offset")
    
    # 🔥 分析父点云（使用 parent_batch）
    if "parent_coord" in batch:
        parent_coord = batch["parent_coord"]
        print(f"\n5️⃣  父点云详细分析:")
        print(f"   - parent_coord shape: {parent_coord.shape}")
        if "parent_color" in batch:
            print(f"   - parent_color shape: {batch['parent_color'].shape}")
        if "parent_grid_coord" in batch:
            print(f"   - parent_grid_coord shape: {batch['parent_grid_coord'].shape}")
        
        if "parent_batch" in batch:
            parent_batch = batch["parent_batch"]
            print(f"   - parent_batch shape: {parent_batch.shape}")
            
            # 统计每个样本的父点云点数
            unique_batches = torch.unique(parent_batch)
            print(f"\n   各样本的父点云点数（从 parent_batch 统计）:")
            for b_idx in unique_batches:
                mask = parent_batch == b_idx
                n_parent_points = mask.sum().item()
                print(f"      Sample {b_idx}: {n_parent_points:6d} points")
            
            print(f"\n   总父点云点数: {len(parent_batch)}")
    
    # 🔥 对比局部点云和父点云的点数
    if "offset" in batch and "parent_offset" in batch:
        print(f"\n6️⃣  局部点云 vs 父点云 点数对比:")
        offset = batch["offset"]
        parent_offset = batch["parent_offset"]
        
        print(f"   {'Sample':<10} {'Local Points':<15} {'Parent Points':<15} {'Ratio':<10}")
        print(f"   {'-'*10} {'-'*15} {'-'*15} {'-'*10}")
        
        for i in range(len(offset) - 1):
            local_n = (offset[i + 1] - offset[i]).item()
            parent_n = (parent_offset[i + 1] - parent_offset[i]).item()
            ratio = parent_n / local_n if local_n > 0 else 0
            print(f"   {i:<10} {local_n:<15} {parent_n:<15} {ratio:<10.2f}x")
    
    # 🔥 分析归一化参数
    if "norm_offset" in batch or "norm_scale" in batch:
        print(f"\n7️⃣  归一化参数:")
        if "norm_offset" in batch:
            norm_offset = batch["norm_offset"]
            print(f"   - norm_offset shape: {norm_offset.shape}")
            if norm_offset.dim() == 2:
                for i in range(min(norm_offset.shape[0], 5)):  # 最多显示 5 个
                    print(f"      Sample {i}: [{norm_offset[i, 0]:.3f}, {norm_offset[i, 1]:.3f}, {norm_offset[i, 2]:.3f}]")
        
        if "norm_scale" in batch:
            norm_scale = batch["norm_scale"]
            print(f"   - norm_scale shape: {norm_scale.shape}")
            if norm_scale.dim() == 1:
                for i in range(min(norm_scale.shape[0], 5)):  # 最多显示 5 个
                    print(f"      Sample {i}: {norm_scale[i].item():.6f}")
    
    # 🔥 分析 GT
    if "gt_position" in batch:
        gt_position = batch["gt_position"]
        print(f"\n8️⃣  Ground Truth:")
        print(f"   - gt_position shape: {gt_position.shape}")
        for j in range(min(gt_position.shape[0], 5)):  # 最多显示 5 个
            print(f"      Sample {j}: [{gt_position[j, 0]:.6f}, {gt_position[j, 1]:.6f}, {gt_position[j, 2]:.6f}]")
    
    # 🔥 分析类别
    if "category_id" in batch:
        category_id = batch["category_id"]
        print(f"\n9️⃣  类别 ID:")
        print(f"   - category_id shape: {category_id.shape}")
        
        category_names = {0: "Scissors", 1: "Cup", 2: "Avocado"}
        for j in range(min(category_id.shape[0], 5)):  # 最多显示 5 个
            cat_id = category_id[j].item()
            cat_name = category_names.get(cat_id, "Unknown")
            print(f"      Sample {j}: {cat_id} ({cat_name})")
    
    # 🔥 分析样本名称
    if "name" in batch:
        print(f"\n🔟 样本名称:")
        for j, name in enumerate(batch['name'][:5]):  # 最多显示 5 个
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


def test_collate_fn():
    """测试 collate_fn 是否正确处理父点云"""
    print_separator("🔧 测试 point_collate_fn")
    
    cfg = Config.fromfile("/home/ypf/PointTransformerV3_Simple/configs/s3dis/semseg-pt-v3m1-gelsight.py")
    
    print(f"\n📂 加载训练集...")
    train_dataset = build_dataset(cfg.data.train)
    
    # 手动获取几个样本
    print(f"\n📦 手动获取 3 个样本...")
    samples = [train_dataset[i] for i in range(3)]
    
    print(f"\n   各样本的 keys:")
    for i, sample in enumerate(samples):
        print(f"   Sample {i}: {list(sample.keys())}")
        
        # 🔥 显示每个样本的点数
        if "coord" in sample:
            local_n = sample["coord"].shape[0]
            print(f"      - 局部点数: {local_n}")
        
        if "parent_coord" in sample:
            parent_n = sample["parent_coord"].shape[0]
            print(f"      - 父点云点数: {parent_n}")
    
    # 使用 collate_fn
    print(f"\n🔧 调用 point_collate_fn...")
    batch = point_collate_fn(samples, mix_prob=0.0)
    
    print(f"\n   ✅ Collate 成功!")
    print(f"   Batch keys: {list(batch.keys())}")
    
    # 🔥 重点检查 parent_offset
    if "parent_offset" in batch:
        print(f"\n   ✅ 成功生成 parent_offset!")
        print(f"      - parent_offset: {batch['parent_offset'].tolist()}")
    else:
        print(f"\n   ❌ 未生成 parent_offset!")
        print(f"   ⚠️  可能的问题:")
        print(f"      1. point_collate_fn 没有生成 parent_offset")
        print(f"      2. parent_data 中没有 parent_batch")
    
    # 分析结果
    analyze_batch(batch, 0)
    
    print_separator("✅ point_collate_fn 测试完成")


def test_single_sample():
    """测试单个样本的数据结构"""
    print_separator("🔬 测试单个样本")
    
    cfg = Config.fromfile("/home/ypf/PointTransformerV3_Simple/configs/s3dis/semseg-pt-v3m1-gelsight.py")
    
    print(f"\n📂 加载训练集...")
    train_dataset = build_dataset(cfg.data.train)
    
    print(f"\n📦 获取 Sample 0...")
    sample = train_dataset[0]
    
    print(f"\n   ✅ 样本获取成功!")
    print(f"   - Type: {type(sample)}")
    print(f"   - Keys: {list(sample.keys())}")
    
    print(f"\n   详细字段分析:")
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"   ✅ {key:20s}: shape={str(value.shape):20s} dtype={value.dtype}")
        elif isinstance(value, (int, float)):
            print(f"   ✅ {key:20s}: {type(value).__name__}={value}")
        elif isinstance(value, str):
            print(f"   ✅ {key:20s}: '{value}'")
        else:
            print(f"   ⚠️  {key:20s}: {type(value)}")
    
    # 🔥 检查父点云字段
    print(f"\n   父点云字段检查:")
    parent_fields = ["parent_coord", "parent_color", "parent_grid_coord"]
    for field in parent_fields:
        if field in sample:
            value = sample[field]
            if isinstance(value, torch.Tensor):
                print(f"      ✅ {field}: shape={value.shape}")
            else:
                print(f"      ✅ {field}: {type(value).__name__}={value}")
        else:
            print(f"      ❌ {field}: 缺失")
    
    print_separator("✅ 单个样本测试完成")


def main():
    """运行所有测试"""
    print("\n" + "🚀" * 40)
    print("开始测试带父点云的 DataLoader".center(80))
    print("🚀" * 40)
    
    try:
        # 1. 测试单个样本
        test_single_sample()
        
        # 2. 测试 collate_fn
        test_collate_fn()
        
        # 3. 测试训练集
        test_train_dataloader()
        
        # 4. 测试验证集
        test_val_dataloader()
        
        print("\n" + "🎉" * 40)
        print("所有测试通过！".center(80))
        print("🎉" * 40 + "\n")
        
    except Exception as e:
        print("\n" + "❌" * 40)
        print(f"测试失败: {e}".center(80))
        print("❌" * 40 + "\n")
        
        import traceback
        print("\n完整错误信息:")
        traceback.print_exc()
        
        raise


if __name__ == "__main__":
    main()