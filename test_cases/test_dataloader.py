import sys
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from functools import partial
import numpy as np


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


def analyze_batch(batch, batch_idx, is_test=False):
    """详细分析一个 batch 的内容（嵌套结构）"""
    print(f"\n{'─' * 80}")
    print(f"📦 Batch {batch_idx} Analysis {'(Test Mode)' if is_test else '(Train/Val Mode)'}")
    print(f"{'─' * 80}")
    
    print(f"\n1️⃣  基本信息:")
    print(f"   - Type: {type(batch)}")
    print(f"   - Top-level Keys: {list(batch.keys())}")
    
    # 分析局部点云（local）
    if "local" in batch:
        print(f"\n2️⃣  局部点云 (local):")
        local = batch["local"]
        print(f"   - Keys: {list(local.keys())}")
        
        # 🔥 更新：添加新字段
        for key in ["coord", "grid_coord", "feat", "offset", 
                    "gt_position", "coord_centroid",  # 🔥 新增
                    "name", "category_id", "parent_id"]:  # 🔥 新增
            if key in local:
                value = local[key]
                if isinstance(value, torch.Tensor):
                    print(f"   ✅ {key:25s}: shape={str(value.shape):20s} dtype={value.dtype}")
                elif isinstance(value, list):
                    print(f"   ✅ {key:25s}: list of {len(value)} items")
                else:
                    print(f"   ✅ {key:25s}: {type(value).__name__}")
        
        if "offset" in local:
            offset = local["offset"]
            print(f"\n   Offset 分析:")
            print(f"   - Offset: {offset.tolist()}")
            print(f"   - Batch size: {len(offset)}")
            
            print(f"\n   各样本的局部点云点数:")
            start = 0
            for i in range(len(offset)):
                n_points = offset[i] - start
                print(f"      Sample {i}: {n_points:6d} points (range: [{start:6d}, {offset[i]:6d}))")
                start = offset[i]
            
            total_points = offset[-1].item()
            print(f"\n   ✅ 总局部点数: {total_points}")
            
            if "coord" in local:
                actual_points = local["coord"].shape[0]
                if actual_points == total_points:
                    print(f"   ✅ Offset 验证通过: coord.shape[0] == offset[-1]")
                else:
                    print(f"   ❌ Offset 验证失败: {actual_points} != {total_points}")
    
    # 分析父点云（parent）
    if "parent" in batch:
        print(f"\n3️⃣  父点云 (parent):")
        parent = batch["parent"]
        print(f"   - Keys: {list(parent.keys())}")
        
        for key in ["coord", "grid_coord", "feat", "offset", "name"]:
            if key in parent:
                value = parent[key]
                if isinstance(value, torch.Tensor):
                    print(f"   ✅ {key:15s}: shape={str(value.shape):20s} dtype={value.dtype}")
                elif isinstance(value, list):
                    print(f"   ✅ {key:15s}: list of {len(value)} items")
                else:
                    print(f"   ✅ {key:15s}: {type(value).__name__}")
        
        if "offset" in parent:
            offset = parent["offset"]
            print(f"\n   Offset 分析:")
            print(f"   - Offset: {offset.tolist()}")
            print(f"   - Batch size: {len(offset)}")
            
            print(f"\n   各样本的父点云点数:")
            start = 0
            for i in range(len(offset)):
                n_points = offset[i] - start
                print(f"      Sample {i}: {n_points:6d} points (range: [{start:6d}, {offset[i]:6d}))")
                start = offset[i]
            
            total_points = offset[-1].item()
            print(f"\n   ✅ 总父点云点数: {total_points}")
            
            if "coord" in parent:
                actual_points = parent["coord"].shape[0]
                if actual_points == total_points:
                    print(f"   ✅ Offset 验证通过: coord.shape[0] == offset[-1]")
                else:
                    print(f"   ❌ Offset 验证失败: {actual_points} != {total_points}")
    
    # 对比局部点云和父点云
    if "local" in batch and "parent" in batch:
        if "offset" in batch["local"] and "offset" in batch["parent"]:
            print(f"\n4️⃣  局部点云 vs 父点云:")
            local_offset = batch["local"]["offset"]
            parent_offset = batch["parent"]["offset"]
            
            print(f"   {'Sample':<10} {'Local Points':<15} {'Parent Points':<15} {'Ratio':<10}")
            print(f"   {'-'*10} {'-'*15} {'-'*15} {'-'*10}")
            
            local_start = 0
            parent_start = 0
            for i in range(len(local_offset)):
                local_n = local_offset[i] - local_start
                parent_n = parent_offset[i] - parent_start
                ratio = parent_n / local_n if local_n > 0 else 0
                print(f"   {i:<10} {local_n:<15} {parent_n:<15} {ratio:<10.2f}x")
                local_start = local_offset[i]
                parent_start = parent_offset[i]
    
    # 分析归一化参数
    if "norm_offset" in batch or "norm_scale" in batch:
        print(f"\n5️⃣  归一化参数:")
        if "norm_offset" in batch:
            norm_offset = batch["norm_offset"]
            if isinstance(norm_offset, torch.Tensor):
                print(f"   - norm_offset shape: {norm_offset.shape}")
                if norm_offset.dim() == 2:
                    for i in range(min(norm_offset.shape[0], 3)):
                        print(f"      Sample {i}: [{norm_offset[i, 0]:.3f}, {norm_offset[i, 1]:.3f}, {norm_offset[i, 2]:.3f}]")
            else:
                print(f"   - norm_offset: list of {len(norm_offset)} items")
        
        if "norm_scale" in batch:
            norm_scale = batch["norm_scale"]
            if isinstance(norm_scale, torch.Tensor):
                print(f"   - norm_scale shape: {norm_scale.shape}")
                if norm_scale.dim() == 1:
                    for i in range(min(norm_scale.shape[0], 3)):
                        print(f"      Sample {i}: {norm_scale[i].item():.6f}")
                elif norm_scale.dim() == 2:
                    for i in range(min(norm_scale.shape[0], 3)):
                        print(f"      Sample {i}: [{norm_scale[i, 0]:.6f}, {norm_scale[i, 1]:.6f}, {norm_scale[i, 2]:.6f}]")
            else:
                print(f"   - norm_scale: list of {len(norm_scale)} items")
    
    # 🔥 分析 GT（训练/验证集）
    if not is_test and "local" in batch:
        print(f"\n6️⃣  Ground Truth:")
        
        # GT Position（绝对位置）
        if "gt_position" in batch["local"]:
            gt_pos = batch["local"]["gt_position"]
            print(f"   - gt_position shape: {gt_pos.shape}")
            print(f"   - gt_position dtype: {gt_pos.dtype}")
            for j in range(min(gt_pos.shape[0], 3)):
                print(f"      Sample {j}: [{gt_pos[j, 0]:.6f}, {gt_pos[j, 1]:.6f}, {gt_pos[j, 2]:.6f}]")
        
        # Coord Centroid（用于调试）
        if "coord_centroid" in batch["local"]:
            centroid = batch["local"]["coord_centroid"]
            print(f"   - coord_centroid shape: {centroid.shape}")
            for j in range(min(centroid.shape[0], 3)):
                print(f"      Sample {j}: [{centroid[j, 0]:.6f}, {centroid[j, 1]:.6f}, {centroid[j, 2]:.6f}]")
    
    # 🔥 测试集：只有 coord_centroid
    if is_test and "local" in batch:
        print(f"\n6️⃣  推理信息 (无 GT):")
        
        if "coord_centroid" in batch["local"]:
            centroid = batch["local"]["coord_centroid"]
            print(f"   - coord_centroid shape: {centroid.shape}")
            for j in range(min(centroid.shape[0], 3)):
                print(f"      Sample {j}: [{centroid[j, 0]:.6f}, {centroid[j, 1]:.6f}, {centroid[j, 2]:.6f}]")
        else:
            print(f"   ⚠️  测试集缺少 coord_centroid（无法恢复绝对位置）")
    
    # 分析样本名称
    if "local" in batch and "name" in batch["local"]:
        print(f"\n7️⃣  样本名称:")
        print(f"   局部点云:")
        for j, name in enumerate(batch["local"]["name"][:3]):
            print(f"      Sample {j}: {name}")
    
    if "parent" in batch and "name" in batch["parent"]:
        print(f"   父点云:")
        for j, name in enumerate(batch["parent"]["name"][:3]):
            print(f"      Sample {j}: {name}")
    
    print(f"\n{'─' * 80}\n")


def test_single_sample():
    """测试单个样本的数据结构"""
    print_separator("🔬 测试单个样本")
    
    # 🔥 修改配置文件路径
    cfg = Config.fromfile("/home/ypf/PointTransformerV3_Simple/configs/s3dis/semseg-pt-v3m1-gelsight.py")
    
    print(f"\n📂 加载训练集...")
    train_dataset = build_dataset(cfg.data.train)
    
    print(f"\n📦 获取 Sample 0...")
    sample = train_dataset[0]
    
    print(f"\n   ✅ 样本获取成功!")
    print(f"   - Type: {type(sample)}")
    print(f"   - Top-level Keys: {list(sample.keys())}")
    
    # 分析局部点云
    if "local" in sample:
        print(f"\n   局部点云 (local):")
        local = sample["local"]
        print(f"   - Keys: {list(local.keys())}")
        for key, value in local.items():
            if isinstance(value, torch.Tensor):
                print(f"      ✅ {key:25s}: shape={str(value.shape):20s} dtype={value.dtype}")
            else:
                print(f"      ✅ {key:25s}: {type(value).__name__}")
    
    # 分析父点云
    if "parent" in sample:
        print(f"\n   父点云 (parent):")
        parent = sample["parent"]
        print(f"   - Keys: {list(parent.keys())}")
        for key, value in parent.items():
            if isinstance(value, torch.Tensor):
                print(f"      ✅ {key:15s}: shape={str(value.shape):20s} dtype={value.dtype}")
            else:
                print(f"      ✅ {key:15s}: {type(value).__name__}")
    
    # 🔥 验证 GT 字段
    if "gt_position" in sample:
        print(f"\n   验证 GT 字段:")
        gt_pos = sample["gt_position"]
        print(f"   - gt_position: {gt_pos.numpy()}")
        
        if "coord_centroid" in sample:
            centroid = sample["coord_centroid"]
            print(f"   - coord_centroid: {centroid.numpy()}")
            
            # 验证 gt_position 是否远离原点（绝对位置）
            gt_norm = torch.norm(gt_pos).item()
            print(f"   - gt_position norm: {gt_norm:.6f} {'✅ (绝对位置)' if gt_norm > 0.1 else '⚠️ (接近原点)'}")

    if "local" in sample and "coord" in sample["local"]:
        print(f"\n   验证坐标去中心化:")
        coord = sample["local"]["coord"]
        coord_mean = coord.mean(dim=0).numpy()
        coord_norm = np.linalg.norm(coord_mean)
        
        print(f"   - coord.mean(): [{coord_mean[0]:.6f}, {coord_mean[1]:.6f}, {coord_mean[2]:.6f}]")
        print(f"   - coord.mean() norm: {coord_norm:.2e}")
        
        if coord_norm < 1e-4:
            print(f"   ✅ coord 已去中心化（均值接近原点）")
        else:
            print(f"   ❌ coord 未去中心化（均值远离原点）")
        
        # 验证恢复
        if "coord_centroid" in sample:
            centroid = sample["coord_centroid"]
            coord_recovered = coord + centroid
            coord_recovered_mean = coord_recovered.mean(dim=0).numpy()
            
            print(f"\n   恢复原始坐标:")
            print(f"   - (coord + centroid).mean(): [{coord_recovered_mean[0]:.6f}, {coord_recovered_mean[1]:.6f}, {coord_recovered_mean[2]:.6f}]")
            print(f"   - 应该等于 coord_centroid: [{centroid[0]:.6f}, {centroid[1]:.6f}, {centroid[2]:.6f}]")
            
            error = np.linalg.norm(coord_recovered_mean - centroid.numpy())
            if error < 1e-5:
                print(f"   ✅ 恢复验证通过（误差 {error:.2e}）")
            else:
                print(f"   ❌ 恢复验证失败（误差 {error:.2e}）")

    # 分析归一化参数
    if "norm_offset" in sample:
        print(f"\n   归一化参数:")
        print(f"   - norm_offset: {sample['norm_offset']}")
        print(f"   - norm_scale: {sample['norm_scale']}")
    
    print_separator("✅ 单个样本测试完成")


def test_collate_fn():
    """测试 collate_fn"""
    print_separator("🔧 测试 point_collate_fn")
    
    cfg = Config.fromfile("/home/ypf/PointTransformerV3_Simple/configs/s3dis/semseg-pt-v3m1-gelsight.py")
    
    print(f"\n📂 加载训练集...")
    train_dataset = build_dataset(cfg.data.train)
    
    print(f"\n📦 手动获取 3 个样本...")
    samples = [train_dataset[i] for i in range(3)]
    
    print(f"\n   各样本的结构:")
    for i, sample in enumerate(samples):
        print(f"   Sample {i}:")
        print(f"      - Top-level keys: {list(sample.keys())}")
        
        if "local" in sample and "coord" in sample["local"]:
            local_n = sample["local"]["coord"].shape[0]
            print(f"      - 局部点数: {local_n}")
        
        if "parent" in sample and "coord" in sample["parent"]:
            parent_n = sample["parent"]["coord"].shape[0]
            print(f"      - 父点云点数: {parent_n}")
    
    print(f"\n🔧 调用 point_collate_fn...")
    batch = point_collate_fn(samples, mix_prob=0.0)
    
    print(f"\n   ✅ Collate 成功!")
    print(f"   Batch top-level keys: {list(batch.keys())}")
    
    if "local" in batch:
        print(f"   Batch local keys: {list(batch['local'].keys())}")
    if "parent" in batch:
        print(f"   Batch parent keys: {list(batch['parent'].keys())}")
    
    analyze_batch(batch, 0, is_test=False)
    
    print_separator("✅ point_collate_fn 测试完成")


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
    
    print_separator("🔍 测试前 2 个 Batch")
    
    for i, batch in enumerate(train_loader):
        if i >= 2:
            break
        analyze_batch(batch, i, is_test=False)
    
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
        analyze_batch(batch, i, is_test=False)
    
    print_separator("✅ 验证集 DataLoader 测试完成")


def main():
    """运行所有测试"""
    print("\n" + "🚀" * 40)
    print("开始测试带 CentroidShift 的 DataLoader".center(80))
    print("🚀" * 40)
    
    try:
        test_single_sample()
        test_collate_fn()
        test_train_dataloader()
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