import json
from pathlib import Path
import random

def generate_split_files(data_root, train_ratio=0.7, val_ratio=0.2):
    """
    生成 train.txt, val.txt, test.txt
    
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
    all_samples = []
    
    for category_dir in data_root.iterdir():
        if not category_dir.is_dir():
            continue
        
        category_name = category_dir.name
        if category_name in ['train.txt', 'val.txt', 'test.txt']:
            continue
        
        patches_dir = category_dir / "patches"
        if not patches_dir.exists():
            print(f"⚠️ 跳过 {category_name}（没有 patches 目录）")
            continue
        
        # 收集该类别的所有样本
        category_samples = []
        for pth_file in sorted(patches_dir.glob("patch_*.pth")):
            # 格式：Scissors/patches/patch_000001.pth
            relative_path = pth_file.relative_to(data_root)
            category_samples.append(str(relative_path))
        
        print(f"📦 {category_name}: {len(category_samples)} 样本")
        all_samples.extend(category_samples)
    
    # 打乱
    random.seed(42)  # 🔥 固定随机种子，确保可复现
    random.shuffle(all_samples)
    
    # 划分
    n_total = len(all_samples)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_samples = all_samples[:n_train]
    val_samples = all_samples[n_train:n_train + n_val]
    test_samples = all_samples[n_train + n_val:]
    
    # 保存
    with open(data_root / "train.txt", 'w') as f:
        f.write('\n'.join(train_samples))
    
    with open(data_root / "val.txt", 'w') as f:
        f.write('\n'.join(val_samples))
    
    with open(data_root / "test.txt", 'w') as f:
        f.write('\n'.join(test_samples))
    
    print(f"\n✅ 生成完成：")
    print(f"   📁 数据根目录: {data_root}")
    print(f"   📊 总样本数: {n_total}")
    print(f"   🔹 Train: {len(train_samples)} ({train_ratio*100:.0f}%)")
    print(f"   🔹 Val:   {len(val_samples)} ({val_ratio*100:.0f}%)")
    print(f"   🔹 Test:  {len(test_samples)} ({(1-train_ratio-val_ratio)*100:.0f}%)")
    
    # 显示每个类别的分布
    print(f"\n📊 类别分布：")
    for split_name, split_samples in [("Train", train_samples), ("Val", val_samples), ("Test", test_samples)]:
        category_counts = {}
        for sample in split_samples:
            category = sample.split('/')[0]
            category_counts[category] = category_counts.get(category, 0) + 1
        print(f"   {split_name}: {category_counts}")


if __name__ == "__main__":
    generate_split_files(
        "/root/autodl-tmp/touch_processed_data/",  # 🔥 你的数据根目录
        train_ratio=0.7,
        val_ratio=0.2
    )