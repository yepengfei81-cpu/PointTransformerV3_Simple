"""
🎯 将 GelSight 采集的局部点云 (.ply) 转换为推理用的 .pth 文件
- 使用训练集的全局归一化参数
- 数据结构与训练集完全一致
"""

import numpy as np
import open3d as o3d
import torch
from pathlib import Path
import json
import argparse


class LocalPCDConverter:
    def __init__(self, normalization_source):
        self.normalization_source = Path(normalization_source)
        self.global_min = None
        self.global_max = None
        self.global_range = None
        
        self._load_normalization_params()
        
        print(f"✅ 全局归一化参数加载完成:")
        print(f"   global_min:   {self.global_min}")
        print(f"   global_max:   {self.global_max}")
        print(f"   global_range: {self.global_range}")
    
    def _load_normalization_params(self):
        """从训练集加载全局归一化参数"""
        source = self.normalization_source
        
        if source.suffix == '.pth':
            print(f"📂 从样本文件加载归一化参数: {source}")
            data_dict = torch.load(source, weights_only=False)
            
            if 'norm_offset' not in data_dict or 'norm_scale' not in data_dict:
                raise ValueError(f"❌ {source} 中缺少 norm_offset 或 norm_scale 字段")
            
            self.global_min = data_dict['norm_offset']
            self.global_range = data_dict['norm_scale']
            self.global_max = self.global_min + self.global_range
            
            # 确保是 numpy 数组
            if isinstance(self.global_min, torch.Tensor):
                self.global_min = self.global_min.cpu().numpy()
            if isinstance(self.global_range, torch.Tensor):
                self.global_range = self.global_range.cpu().numpy()
            if isinstance(self.global_max, torch.Tensor):
                self.global_max = self.global_max.cpu().numpy()
        else:
            raise ValueError(
                f"❌ 不支持的归一化参数来源: {source}\n"
                f"   请提供训练集的 .pth 样本文件"
            )
        
        # 类型检查
        self.global_min = self.global_min.astype(np.float32)
        self.global_max = self.global_max.astype(np.float32)
        self.global_range = self.global_range.astype(np.float32)
    
    def load_local_pointcloud(self, ply_path):
        ply_path = Path(ply_path)
        
        if not ply_path.exists():
            raise FileNotFoundError(f"❌ 文件不存在: {ply_path}")
        
        print(f"\n📂 加载局部点云: {ply_path.name}")
        
        pcd = o3d.io.read_point_cloud(str(ply_path))
        
        if len(pcd.points) == 0:
            raise ValueError(f"❌ 点云为空: {ply_path}")
        
        local_coord = np.asarray(pcd.points, dtype=np.float32)
        
        if pcd.has_colors():
            local_color = np.asarray(pcd.colors, dtype=np.float32)
            print(f"   ✅ 加载 {len(local_coord)} 个点（带颜色）")
        else:
            local_color = np.ones((len(local_coord), 3), dtype=np.float32) * 0.5
            print(f"   ⚠️  加载 {len(local_coord)} 个点（无颜色，使用默认灰色）")
        
        # 打印坐标范围
        coord_min = local_coord.min(axis=0)
        coord_max = local_coord.max(axis=0)
        coord_range = coord_max - coord_min
        
        print(f"   坐标范围:")
        print(f"      X: [{coord_min[0]:.6f}, {coord_max[0]:.6f}] (range: {coord_range[0]:.6f})")
        print(f"      Y: [{coord_min[1]:.6f}, {coord_max[1]:.6f}] (range: {coord_range[1]:.6f})")
        print(f"      Z: [{coord_min[2]:.6f}, {coord_max[2]:.6f}] (range: {coord_range[2]:.6f})")
        
        return local_coord, local_color
    
    def convert_to_pth(
        self,
        ply_path,
        output_path=None,
        category="Unknown",
        category_id=0,
        sample_name=None,
        gt_position=None,
        bigpcd_id=None,
        bigpcd_name=None,
    ):
        ply_path = Path(ply_path)
        
        # 🔥 自动生成 sample_name
        if sample_name is None:
            sample_name = ply_path.stem
        
        # 1. 加载点云
        local_coord_original, local_color = self.load_local_pointcloud(ply_path)
        
        # 2. 归一化局部点云
        local_coord_normalized = (local_coord_original - self.global_min) / self.global_range
        local_coord_normalized = local_coord_normalized.astype(np.float32)
        
        print(f"\n🌍 归一化结果:")
        print(f"   归一化坐标范围: [{local_coord_normalized.min():.6f}, {local_coord_normalized.max():.6f}]")
        
        # ⚠️ 检查是否在 [0, 1] 范围内
        if local_coord_normalized.min() < -0.1 or local_coord_normalized.max() > 1.1:
            print(f"   ⚠️  警告：归一化坐标超出 [0, 1] 范围较多！")
            print(f"       这可能意味着当前点云与训练集的空间范围差异较大")
        elif local_coord_normalized.min() < 0 or local_coord_normalized.max() > 1:
            print(f"   ⚠️  注意：归一化坐标略微超出 [0, 1] 范围（可接受）")
        else:
            print(f"   ✅ 归一化坐标在 [0, 1] 范围内")
        
        # 🔥 3. 处理 GT 位置
        if gt_position is not None:
            gt_position_original = np.array(gt_position, dtype=np.float32)
            gt_position_normalized = (gt_position_original - self.global_min) / self.global_range
            gt_position_normalized = gt_position_normalized.astype(np.float32)
            
            gt_available = True
            
            print(f"\n✅ GT 位置:")
            print(f"   原始空间:   [{gt_position_original[0]:.6f}, {gt_position_original[1]:.6f}, {gt_position_original[2]:.6f}] 米")
            print(f"   归一化空间: [{gt_position_normalized[0]:.6f}, {gt_position_normalized[1]:.6f}, {gt_position_normalized[2]:.6f}]")
        else:
            centroid_original = local_coord_original.mean(axis=0).astype(np.float32)
            gt_position_original = centroid_original
            gt_position_normalized = (centroid_original - self.global_min) / self.global_range
            gt_position_normalized = gt_position_normalized.astype(np.float32)
            
            gt_available = False
            
            print(f"\n⚠️  无 GT 位置（真实推理场景）")
            print(f"   使用局部点云质心作为占位符（不参与误差计算）:")
            print(f"      原始空间:   [{gt_position_original[0]:.6f}, {gt_position_original[1]:.6f}, {gt_position_original[2]:.6f}] 米")
            print(f"      归一化空间: [{gt_position_normalized[0]:.6f}, {gt_position_normalized[1]:.6f}, {gt_position_normalized[2]:.6f}]")
        
        # 🔥 4. 处理父点云信息
        if bigpcd_id is not None and bigpcd_id >= 0:
            if bigpcd_name is None:
                if isinstance(bigpcd_id, int):
                    bigpcd_name = f"bigpointcloud_{bigpcd_id:03d}.ply"
                else:
                    bigpcd_name = f"bigpointcloud_{str(bigpcd_id).zfill(3)}.ply"
            
            print(f"\n✅ 父点云信息:")
            print(f"   ID:   {bigpcd_id}")
            print(f"   文件: {bigpcd_name}")
        else:
            bigpcd_id = -1
            bigpcd_name = "unknown"
            
            print(f"\n⚠️  未指定父点云（推理时不会尝试加载父点云）")
        
        # 5. 构造数据字典
        data_dict = {
            "local_coord": local_coord_normalized,
            "local_color": local_color,
            "gt_position": gt_position_normalized,
            "gt_available": gt_available,
            "norm_offset": self.global_min,
            "norm_scale": self.global_range,
            "local_coord_original": local_coord_original,
            "gt_position_original": gt_position_original,
            "extraction_method": "real_gelsight",
            "extraction_radius": 0.0,
            "category": category,
            "category_id": category_id,
            "bigpcd_name": bigpcd_name,
            "bigpcd_id": bigpcd_id,
            "parent_id": bigpcd_id,
            "sample_id": -1,
            "name": sample_name,
        }
        
        # 6. 保存
        if output_path is None:
            output_path = ply_path.with_suffix('.pth')
        else:
            output_path = Path(output_path)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data_dict, output_path)
        
        print(f"\n✅ 转换完成！")
        print(f"   输入: {ply_path}")
        print(f"   输出: {output_path}")
        print(f"   样本名称: {sample_name}")
        print(f"   点数: {len(local_coord_normalized)}")
        print(f"   GT 可用: {'是' if gt_available else '否'}")
        print(f"   父点云 ID: {bigpcd_id}")
        
        return output_path
    
    def verify_pth(self, pth_path):
        """验证生成的 .pth 文件"""
        pth_path = Path(pth_path)
        
        print(f"\n{'='*70}")
        print(f"🔍 验证生成的 .pth 文件")
        print(f"{'='*70}")
        print(f"文件: {pth_path}\n")
        
        data_dict = torch.load(pth_path, weights_only=False)
        
        print("📋 字段检查:")
        required_keys = [
            "local_coord", "local_color", "gt_position", "gt_available",
            "norm_offset", "norm_scale",
            "local_coord_original", "gt_position_original",
            "extraction_method", "extraction_radius",
            "category", "category_id", "bigpcd_name", "bigpcd_id", 
            "parent_id", "sample_id", "name"
        ]
        
        for key in required_keys:
            if key in data_dict:
                value = data_dict[key]
                if isinstance(value, np.ndarray):
                    print(f"   ✅ {key:25s} shape={str(value.shape):15s} dtype={value.dtype}")
                elif value is None:
                    print(f"   ✅ {key:25s} value=None")
                elif isinstance(value, (bool, np.bool_)):
                    print(f"   ✅ {key:25s} value={value}")
                else:
                    print(f"   ✅ {key:25s} value={value}")
            else:
                print(f"   ❌ {key:25s} 缺失")
        
        # 🔥 反归一化验证
        gt_available = data_dict.get('gt_available', False)
        
        print(f"\n🔄 反归一化验证:")
        print(f"   GT 可用: {'是' if gt_available else '否'}")
        
        local_coord_norm = data_dict['local_coord']
        norm_offset = data_dict['norm_offset']
        norm_scale = data_dict['norm_scale']
        local_coord_original = data_dict['local_coord_original']
        
        # 反归一化局部点云
        local_coord_denorm = local_coord_norm * norm_scale + norm_offset
        coord_diff = np.abs(local_coord_denorm - local_coord_original).max()
        
        print(f"\n   局部点云反归一化:")
        print(f"      坐标最大差异: {coord_diff:.6e} 米 = {coord_diff*1000:.6f} 毫米")
        
        if coord_diff < 1e-6:
            print(f"      ✅ 反归一化结果与原始数据完全一致")
        elif coord_diff < 1e-3:
            print(f"      ✅ 反归一化结果与原始数据基本一致（精度损失可接受）")
        else:
            print(f"      ⚠️  反归一化结果与原始数据差异较大")
        
        # 🔥 如果有 GT，验证 GT 位置
        if gt_available:
            gt_position_norm = data_dict['gt_position']
            gt_position_original = data_dict['gt_position_original']
            
            gt_position_denorm = gt_position_norm * norm_scale + norm_offset
            gt_diff = np.linalg.norm(gt_position_denorm - gt_position_original)
            
            print(f"\n   GT 位置反归一化:")
            print(f"      位置差异: {gt_diff:.6e} 米 = {gt_diff*1000:.6f} 毫米")
            
            if gt_diff < 1e-6:
                print(f"      ✅ 反归一化结果与原始数据完全一致")
            elif gt_diff < 1e-3:
                print(f"      ✅ 反归一化结果与原始数据基本一致（精度损失可接受）")
            else:
                print(f"      ⚠️  反归一化结果与原始数据差异较大")
        
        print(f"\n{'='*70}\n")


def batch_convert(
    input_dir,
    output_dir,
    normalization_source,
    category="Unknown",
    category_id=0,
    config_file=None,  # 🔥 改名：更通用的配置文件
):
    """
    批量转换目录下的所有 .ply 文件
    
    Args:
        config_file: (可选) 配置文件（JSON 格式），支持:
                     {
                       "sample1.ply": {
                         "gt_position": [x, y, z],
                         "bigpcd_id": 1,
                         "bigpcd_name": "bigpointcloud_001.ply"  // 可选
                       },
                       "sample2.ply": {
                         "bigpcd_id": 2
                       },
                       ...
                     }
                     
                     或简化格式（仅 GT）:
                     {
                       "sample1.ply": [x, y, z],
                       "sample2.ply": [x, y, z],
                       ...
                     }
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ply_files = sorted(input_dir.glob("*.ply"))
    
    if len(ply_files) == 0:
        print(f"❌ 在 {input_dir} 中未找到 .ply 文件")
        return
    
    # 🔥 加载配置文件
    config_dict = {}
    if config_file is not None:
        config_file = Path(config_file)
        if config_file.exists():
            with open(config_file, 'r') as f:
                config_dict = json.load(f)
            print(f"✅ 加载配置文件: {config_file}")
            print(f"   包含 {len(config_dict)} 个样本的配置")
        else:
            print(f"⚠️  配置文件不存在: {config_file}")
    
    print(f"\n{'='*70}")
    print(f"📦 批量转换模式")
    print(f"{'='*70}")
    print(f"   输入目录: {input_dir}")
    print(f"   输出目录: {output_dir}")
    print(f"   发现 {len(ply_files)} 个 .ply 文件")
    print(f"   配置样本: {len(config_dict)}/{len(ply_files)}")
    print(f"{'='*70}\n")
    
    converter = LocalPCDConverter(normalization_source)
    
    success_count = 0
    failed_count = 0
    
    for i, ply_path in enumerate(ply_files, 1):
        print(f"\n{'='*70}")
        print(f"[{i}/{len(ply_files)}] 处理: {ply_path.name}")
        print(f"{'='*70}")
        
        output_path = output_dir / ply_path.with_suffix('.pth').name
        
        # 🔥 解析配置
        sample_config = config_dict.get(ply_path.name, {})
        
        # 支持两种格式：
        # 1. 完整配置: {"gt_position": [x,y,z], "bigpcd_id": 1, ...}
        # 2. 简化配置: [x, y, z]  (仅 GT 位置)
        if isinstance(sample_config, list):
            # 简化格式：直接是 GT 位置
            gt_position = sample_config
            bigpcd_id = None
            bigpcd_name = None
        elif isinstance(sample_config, dict):
            # 完整格式
            gt_position = sample_config.get('gt_position', None)
            bigpcd_id = sample_config.get('bigpcd_id', None)
            bigpcd_name = sample_config.get('bigpcd_name', None)
        else:
            # 无配置
            gt_position = None
            bigpcd_id = None
            bigpcd_name = None
        
        try:
            converter.convert_to_pth(
                ply_path=ply_path,
                output_path=output_path,
                category=category,
                category_id=category_id,
                sample_name=ply_path.stem,  # 🔥 使用文件名作为样本名
                gt_position=gt_position,
                bigpcd_id=bigpcd_id,
                bigpcd_name=bigpcd_name,
            )
            success_count += 1
        except Exception as e:
            print(f"❌ 转换失败: {e}")
            import traceback
            traceback.print_exc()
            failed_count += 1
            continue
    
    print(f"\n{'='*70}")
    print(f"✅ 批量转换完成！")
    print(f"   成功: {success_count}/{len(ply_files)}")
    print(f"   失败: {failed_count}/{len(ply_files)}")
    print(f"   输出目录: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="🎯 将 GelSight 局部点云 (.ply) 转换为推理用的 .pth 文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:

1. 单文件转换（无 GT，无父点云）:
   python convert_local_pcd_to_pth.py \\
       --input sample.ply \\
       --normalization data_root/Scissors/patches/patch_000001.pth \\
       --category Scissors \\
       --category_id 0

2. 单文件转换（有 GT，有父点云）:
   python convert_local_pcd_to_pth.py \\
       --input sample.ply \\
       --normalization data_root/Scissors/patches/patch_000001.pth \\
       --category Scissors \\
       --category_id 0 \\
       --gt_x 0.012 --gt_y 0.034 --gt_z 0.056 \\
       --bigpcd_id 1

3. 批量转换（无配置文件）:
   python convert_local_pcd_to_pth.py \\
       --input_dir gelsight_samples/ \\
       --output_dir inference_data/ \\
       --normalization data_root/Scissors/patches/patch_000001.pth \\
       --category Scissors \\
       --category_id 0

4. 批量转换（有配置文件）:
   python convert_local_pcd_to_pth.py \\
       --input_dir gelsight_samples/ \\
       --output_dir inference_data/ \\
       --normalization data_root/Scissors/patches/patch_000001.pth \\
       --category Scissors \\
       --category_id 0 \\
       --config samples_config.json

   配置文件格式 (samples_config.json):
   {
     "sample1.ply": {
       "gt_position": [0.012, 0.034, 0.056],
       "bigpcd_id": 1,
       "bigpcd_name": "bigpointcloud_001.ply"
     },
     "sample2.ply": {
       "bigpcd_id": 2
     },
     "sample3.ply": [0.023, 0.045, 0.067]
   }

5. 验证转换结果:
   python convert_local_pcd_to_pth.py \\
       --verify output.pth
        """
    )
    
    # 输入/输出参数
    parser.add_argument(
        "--input", type=str, default=None,
        help="输入的 .ply 文件路径（单文件模式）"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="输出的 .pth 文件路径（默认：与输入同名同目录）"
    )
    parser.add_argument(
        "--input_dir", type=str, default=None,
        help="输入目录（批量模式）"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="输出目录（批量模式）"
    )
    
    # 归一化参数
    parser.add_argument(
        "--normalization", type=str, required=False,
        help="归一化参数来源：训练集的 .pth 样本"
    )
    
    # 类别参数
    parser.add_argument(
        "--category", type=str, default="Unknown",
        help="类别名称（例如: Scissors, Cup, Avocado）"
    )
    parser.add_argument(
        "--category_id", type=int, default=0,
        help="类别 ID（0=Scissors, 1=Cup, 2=Avocado）"
    )
    
    # 🔥 GT 位置参数（单文件模式）
    parser.add_argument(
        "--gt_x", type=float, default=None,
        help="GT 位置的 X 坐标（米，真实空间）"
    )
    parser.add_argument(
        "--gt_y", type=float, default=None,
        help="GT 位置的 Y 坐标（米，真实空间）"
    )
    parser.add_argument(
        "--gt_z", type=float, default=None,
        help="GT 位置的 Z 坐标（米，真实空间）"
    )
    
    # 🔥 父点云参数（单文件模式）
    parser.add_argument(
        "--bigpcd_id", type=int, default=None,
        help="父点云 ID（例如 1, 2, 3）。如果不提供，将不尝试加载父点云"
    )
    parser.add_argument(
        "--bigpcd_name", type=str, default=None,
        help="父点云文件名（例如 bigpointcloud_001.ply）。可选，默认根据 ID 生成"
    )
    
    # 🔥 配置文件（批量模式）
    parser.add_argument(
        "--config", type=str, default=None,
        help="配置文件（JSON 格式，批量模式）。包含 GT 位置和父点云 ID"
    )
    
    # 验证模式
    parser.add_argument(
        "--verify", type=str, default=None,
        help="验证 .pth 文件"
    )
       
    args = parser.parse_args()
    
    if args.verify:
        # 🔥 验证模式（不需要 normalization 参数）
        converter_temp = LocalPCDConverter.__new__(LocalPCDConverter)
        converter_temp.verify_pth(args.verify)
    
    elif args.input_dir and args.output_dir and args.normalization:
        # 批量模式
        batch_convert(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            normalization_source=args.normalization,
            category=args.category,
            category_id=args.category_id,
            config_file=args.config,  # 🔥 改为 config_file
        )
    
    elif args.input and args.normalization:
        # 单文件模式
        converter = LocalPCDConverter(args.normalization)
        
        # 🔥 构造 GT 位置（如果提供）
        gt_position = None
        if args.gt_x is not None and args.gt_y is not None and args.gt_z is not None:
            gt_position = [args.gt_x, args.gt_y, args.gt_z]
        
        converter.convert_to_pth(
            ply_path=args.input,
            output_path=args.output,
            category=args.category,
            category_id=args.category_id,
            sample_name=None,  # 🔥 会自动使用文件名
            gt_position=gt_position,
            bigpcd_id=args.bigpcd_id,
            bigpcd_name=args.bigpcd_name,             
        )
    
    else:
        parser.print_help()