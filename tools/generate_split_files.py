# tools/generate_split_files.py

import json
import torch
import numpy as np
from pathlib import Path
import random

# 🔥 定义正确的类别映射
CATEGORY_MAP = {
    "Scissors": 0,
    "Cup": 1,
    "Avocado": 2,
}

def fix_single_file(pth_file, correct_category_id):
    """
    修复单个文件的 category_id
    
    Args:
        pth_file: .pth 文件路径
        correct_category_id: 正确的类别 ID
    
    Returns:
        bool: 是否修复成功
    """
    try:
        # 加载数据
        data = torch.load(pth_file, weights_only=False)
        
        # 检查并修复 category_id
        if "category_id" not in data:
            data["category_id"] = correct_category_id
            torch.save(data, pth_file)
            return True
        
        old_id = data["category_id"]
        
        # 处理不同类型
        if isinstance(old_id, np.ndarray):
            old_id = int(old_id.item())
        elif isinstance(old_id, (list, tuple)):
            old_id = int(old_id[0])
        else:
            old_id = int(old_id)
        
        # 如果不正确，修复
        if old_id != correct_category_id:
            data["category_id"] = correct_category_id
            torch.save(data, pth_file)
            return True
        
        return False
        
    except Exception as e:
        print(f"   ❌ Error fixing {pth_file.name}: {e}")
        return False


def generate_split_files(data_root, train_ratio=0.7, val_ratio=0.2, fix_category_id=True):
    """
    生成 train.txt, val.txt, test.txt，并修复 category_id
    
    Args:
        data_root: 数据根目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        fix_category_id: 是否自动修复 category_id
    
    目录结构：
        data_root/
        ├── Scissors/patches/*.pth
        ├── Cup/patches/*.pth
        └── Avocado/patches/*.pth
    
    生成文件：
        data_root/train.txt  # Scissors/patches/patch_000001.pth
        data_root/val.txt
        data_root/test.txt
    """
    data_root = Path(data_root)
    
    # 🔥 删除旧的划分文件
    print("🗑️  删除旧的划分文件...")
    for split_file in ["train.txt", "val.txt", "test.txt"]:
        split_path = data_root / split_file
        if split_path.exists():
            split_path.unlink()
            print(f"   ✅ 删除: {split_file}")
    
    print("\n" + "=" * 80)
    print("开始生成新的划分文件")
    print("=" * 80)
    
    all_samples = []
    category_stats = {}
    
    # 遍历每个类别
    for category_name, correct_category_id in CATEGORY_MAP.items():
        category_dir = data_root / category_name
        
        if not category_dir.exists():
            print(f"\n⚠️  跳过 {category_name}（目录不存在）")
            continue
        
        patches_dir = category_dir / "patches"
        if not patches_dir.exists():
            print(f"\n⚠️  跳过 {category_name}（没有 patches 目录）")
            continue
        
        print(f"\n📦 处理 {category_name} (ID={correct_category_id})...")
        
        # 收集该类别的所有样本
        category_samples = []
        pth_files = sorted(patches_dir.glob("patch_*.pth"))
        
        if len(pth_files) == 0:
            print(f"   ⚠️  没有找到任何 .pth 文件")
            continue
        
        fixed_count = 0
        
        for pth_file in pth_files:
            # 格式：Scissors/patches/patch_000001.pth
            relative_path = pth_file.relative_to(data_root)
            category_samples.append(str(relative_path))
            
            # 🔥 修复 category_id
            if fix_category_id:
                if fix_single_file(pth_file, correct_category_id):
                    fixed_count += 1
        
        print(f"   📊 样本数: {len(category_samples)}")
        if fix_category_id:
            print(f"   🔧 修复数: {fixed_count}")
        
        all_samples.extend(category_samples)
        category_stats[category_name] = len(category_samples)
    
    if len(all_samples) == 0:
        print("\n❌ 错误：没有找到任何样本！")
        return
    
    # 🔥 打乱（固定随机种子）
    print(f"\n🔀 打乱样本...")
    random.seed(42)
    random.shuffle(all_samples)
    
    # 🔥 划分
    n_total = len(all_samples)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_samples = all_samples[:n_train]
    val_samples = all_samples[n_train:n_train + n_val]
    test_samples = all_samples[n_train + n_val:]
    
    # 🔥 保存
    print(f"\n💾 保存划分文件...")
    
    with open(data_root / "train.txt", 'w') as f:
        f.write('\n'.join(train_samples))
    print(f"   ✅ train.txt ({len(train_samples)} 样本)")
    
    with open(data_root / "val.txt", 'w') as f:
        f.write('\n'.join(val_samples))
    print(f"   ✅ val.txt ({len(val_samples)} 样本)")
    
    with open(data_root / "test.txt", 'w') as f:
        f.write('\n'.join(test_samples))
    print(f"   ✅ test.txt ({len(test_samples)} 样本)")
    
    # 🔥 统计信息
    print("\n" + "=" * 80)
    print("✅ 生成完成")
    print("=" * 80)
    
    print(f"\n📊 总体统计：")
    print(f"   📁 数据根目录: {data_root}")
    print(f"   📊 总样本数: {n_total}")
    print(f"   🔹 Train: {len(train_samples):4d} ({len(train_samples)/n_total*100:5.1f}%)")
    print(f"   🔹 Val:   {len(val_samples):4d} ({len(val_samples)/n_total*100:5.1f}%)")
    print(f"   🔹 Test:  {len(test_samples):4d} ({len(test_samples)/n_total*100:5.1f}%)")
    
    # 🔥 每个类别的分布
    print(f"\n📊 各类别样本数：")
    for category_name, count in category_stats.items():
        print(f"   {category_name:10s}: {count:4d} 样本")
    
    print(f"\n📊 各划分的类别分布：")
    for split_name, split_samples in [
        ("Train", train_samples),
        ("Val", val_samples),
        ("Test", test_samples)
    ]:
        category_counts = {}
        for sample in split_samples:
            category = sample.split('/')[0]
            category_counts[category] = category_counts.get(category, 0) + 1
        
        print(f"\n   {split_name}:")
        for category_name in CATEGORY_MAP.keys():
            count = category_counts.get(category_name, 0)
            if count > 0:
                percentage = count / len(split_samples) * 100
                print(f"      {category_name:10s}: {count:4d} ({percentage:5.1f}%)")


def verify_splits(data_root):
    """
    验证生成的划分文件
    
    检查：
    1. 文件是否存在
    2. 样本数量
    3. category_id 是否正确
    """
    data_root = Path(data_root)
    
    print("\n" + "=" * 80)
    print("验证划分文件")
    print("=" * 80)
    
    for split_name in ["train", "val", "test"]:
        split_file = data_root / f"{split_name}.txt"
        
        if not split_file.exists():
            print(f"\n❌ {split_name}.txt 不存在")
            continue
        
        print(f"\n📋 验证 {split_name}.txt...")
        
        # 读取文件列表
        with open(split_file, 'r') as f:
            file_list = [line.strip() for line in f if line.strip()]
        
        print(f"   样本数: {len(file_list)}")
        
        # 检查前3个样本的 category_id
        print(f"   检查前3个样本:")
        for i, rel_path in enumerate(file_list[:3]):
            full_path = data_root / rel_path
            
            if not full_path.exists():
                print(f"      {i+1}. ❌ 文件不存在: {rel_path}")
                continue
            
            try:
                data = torch.load(full_path, weights_only=False)
                category_id = data.get("category_id", -1)
                
                # 从路径推断期望的类别
                category_name = rel_path.split('/')[0]
                expected_id = CATEGORY_MAP.get(category_name, -1)
                
                status = "✅" if category_id == expected_id else "❌"
                cat_name = ["Scissors", "Cup", "Avocado"][category_id] if 0 <= category_id < 3 else "Unknown"
                
                print(f"      {i+1}. {status} {rel_path}")
                print(f"          category_id={category_id} ({cat_name}), expected={expected_id}")
                
            except Exception as e:
                print(f"      {i+1}. ❌ 加载失败: {rel_path}")
                print(f"          Error: {e}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate train/val/test split files")
    parser.add_argument(
        "--data_root",
        type=str,
        default="/root/autodl-tmp/touch_processed_data/",
        help="Path to dataset root",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="Training set ratio (default: 0.7)",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="Validation set ratio (default: 0.2)",
    )
    parser.add_argument(
        "--fix_category_id",
        action="store_true",
        default=True,
        help="Fix category_id while generating splits (default: True)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify splits after generation",
    )
    
    args = parser.parse_args()
    
    # 生成划分文件
    generate_split_files(
        data_root=args.data_root,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        fix_category_id=args.fix_category_id,
    )
    
    # 验证
    if args.verify:
        verify_splits(args.data_root)