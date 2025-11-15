import numpy as np
import open3d as o3d
import torch
from pathlib import Path
import json
from tqdm import tqdm
import argparse
import shutil

class PTv3DatasetGenerator:
    """
    PTv3 数据集生成器（支持混合提取方法 + 全局归一化）
    """
    
    def __init__(self, input_dir, output_dir, category_name, samples_per_bigpcd=200, 
                 radius=0.01, method='sphere', sphere_samples=None, cube_samples=None):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.category_name = category_name
        self.samples_per_bigpcd = samples_per_bigpcd
        self.radius = radius
        self.method = method
        
        # 🔥 混合提取模式
        self.sphere_samples = sphere_samples
        self.cube_samples = cube_samples
        
        # 🔥 新增：全局归一化参数（步骤1会计算）
        self.global_min = None
        self.global_max = None
        self.global_range = None
        
        # 如果指定了混合模式，检查参数
        if sphere_samples is not None and cube_samples is not None:
            self.mixed_mode = True
            self.samples_per_bigpcd = sphere_samples + cube_samples
            print(f"🔄 混合提取模式:")
            print(f"   球体方法: {sphere_samples} 个样本/大点云")
            print(f"   立方体方法: {cube_samples} 个样本/大点云")
            print(f"   总计: {self.samples_per_bigpcd} 个样本/大点云")
        else:
            self.mixed_mode = False
        
        # 创建输出目录结构
        self.category_dir = self.output_dir / category_name
        self.patches_dir = self.category_dir / "patches"
        self.patches_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"📂 输入目录: {self.input_dir}")
        print(f"📁 输出目录: {self.output_dir}")
        print(f"🏷️  类别名称: {self.category_name}")
    
    def _compute_global_normalization(self, ply_files):
        """
        🌍 扫描所有父点云，计算全局边界
        """
        print("\n" + "="*70)
        print("🌍 步骤1：计算全局归一化参数")
        print("="*70)
        
        global_min = np.array([np.inf, np.inf, np.inf], dtype=np.float32)
        global_max = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float32)
        
        print(f"📂 扫描 {len(ply_files)} 个点云文件...")
        
        for ply_file in tqdm(ply_files, desc="   扫描点云"):
            try:
                pcd = o3d.io.read_point_cloud(str(ply_file))
                coord = np.asarray(pcd.points, dtype=np.float32)
                
                if len(coord) == 0:
                    print(f"   ⚠️  {ply_file.name} 点云为空，跳过")
                    continue
                
                # 更新全局边界
                global_min = np.minimum(global_min, coord.min(axis=0))
                global_max = np.maximum(global_max, coord.max(axis=0))
                
            except Exception as e:
                print(f"   ⚠️  加载 {ply_file.name} 失败: {e}")
                continue
        
        # 🔥 计算全局范围
        global_range = global_max - global_min
        
        # 🔥 保存到实例变量
        self.global_min = global_min
        self.global_max = global_max
        self.global_range = global_range  # (3,)
        
        print("\n✅ 全局归一化参数计算完成:")
        print(f"   global_min:   [{global_min[0]:.6f}, {global_min[1]:.6f}, {global_min[2]:.6f}]")
        print(f"   global_max:   [{global_max[0]:.6f}, {global_max[1]:.6f}, {global_max[2]:.6f}]")
        print(f"   global_range: [{global_range[0]:.6f}, {global_range[1]:.6f}, {global_range[2]:.6f}]")  # 🔥 打印 range
        print("="*70 + "\n")
    
    def _load_big_pointcloud(self, pcd_path):
        """加载大点云并提取特征"""
        pcd = o3d.io.read_point_cloud(str(pcd_path))
        
        if len(pcd.points) == 0:
            raise ValueError(f"点云为空: {pcd_path}")
        
        global_coord = np.asarray(pcd.points).astype(np.float32)
        
        if pcd.has_colors():
            global_color = np.asarray(pcd.colors).astype(np.float32)
        else:
            global_color = np.ones((len(global_coord), 3), dtype=np.float32) * 0.5
            print(f"⚠️ {pcd_path.name} 没有颜色，使用默认灰色")
        
        # 计算单个点云的归一化参数（用于半径调整）
        pcd_min = global_coord.min(axis=0)
        pcd_max = global_coord.max(axis=0)
        pcd_size = pcd_max - pcd_min
        
        pcd_info = {
            'min': pcd_min,
            'max': pcd_max,
            'center': global_coord.mean(axis=0),
            'size': pcd_size,
            'pcd': pcd,
            'points': global_coord,
        }
        
        return global_coord, global_color, pcd_info
    
    def _is_region_valid(self, center, radius, pcd_info, method='sphere', 
                        min_points_ratio=0.5, coverage_threshold=0.5):
        """检查提取区域是否有效"""
        center = np.array(center).reshape(3)
        points = pcd_info['points']
        
        # 边界检查
        margin = radius * 1.1
        if np.any(center - margin < pcd_info['min']) or np.any(center + margin > pcd_info['max']):
            return False, 0
        
        # 点密度检查
        if method == 'sphere':
            distances = np.linalg.norm(points - center, axis=1)
            num_points = np.sum(distances < radius)
        else:  # cube
            diff = np.abs(points - center)
            mask = np.all(diff < radius, axis=1)
            num_points = np.sum(mask)
        
        # 估算理论点数
        total_points = len(points)
        cloud_volume = np.prod(pcd_info['size'])
        point_density = total_points / cloud_volume if cloud_volume > 0 else 0
        
        if method == 'sphere':
            region_volume = (4/3) * np.pi * (radius ** 3)
        else:
            region_volume = (2 * radius) ** 3
        
        expected_points = point_density * region_volume * coverage_threshold
        is_valid = num_points >= max(expected_points * min_points_ratio, 50)
        
        return is_valid, num_points
    
    def _find_valid_radius(self, pcd_info, initial_radius, method='sphere', max_attempts=5):
        """自动寻找合适的提取半径"""
        pcd_size = pcd_info['size']
        min_size = np.min(pcd_size)
        
        # 检查初始半径
        if initial_radius * 2.4 > min_size:
            adjusted_radius = min_size * 0.3
            print(f"⚠️ 初始半径 {initial_radius:.6f} 太大，自动调整为: {adjusted_radius:.6f}")
            return adjusted_radius
        
        radius = initial_radius
        
        for attempt in range(max_attempts):
            safe_margin = radius * 1.2
            safe_min = pcd_info['min'] + safe_margin
            safe_max = pcd_info['max'] - safe_margin
            
            if np.any(safe_min >= safe_max):
                radius = radius * 0.7
                print(f"   尝试 {attempt+1}: 调整半径为 {radius:.6f}")
                continue
            
            test_center = np.random.uniform(safe_min, safe_max)
            is_valid, num_points = self._is_region_valid(test_center, radius, pcd_info, method)
            
            if is_valid:
                if attempt > 0:
                    print(f"   ✅ 找到合适半径: {radius:.6f} (调整了 {attempt} 次)")
                return radius
            else:
                radius = radius * 0.7
                print(f"   尝试 {attempt+1}: 点数不足 ({num_points}), 调整半径为 {radius:.6f}")
        
        print(f"   ❌ 无法找到合适的半径")
        return None
    
    def _extract_random_patch(self, global_coord, global_color, pcd_info, radius, method):
        """
        🔥 步骤2：从大点云中随机提取一个小点云（使用全局归一化）
        """
        max_attempts = 50
        
        safe_margin = radius * 1.2
        safe_min = pcd_info['min'] + safe_margin
        safe_max = pcd_info['max'] - safe_margin
        
        if np.any(safe_min >= safe_max):
            return None, None, None, None, None, None, None, False
        
        for attempt in range(max_attempts):
            center = np.random.uniform(safe_min, safe_max).astype(np.float32)
            
            # 根据 method 参数选择提取方式
            if method == 'sphere':
                distances = np.linalg.norm(global_coord - center, axis=1)
                mask = distances < radius
            else:  # cube
                diff = np.abs(global_coord - center)
                mask = np.all(diff < radius, axis=1)
            
            indices = np.where(mask)[0]
            
            if len(indices) < 50:
                continue
            
            local_points = global_coord[indices]
            local_colors = global_color[indices]
            
            actual_center = local_points.mean(axis=0).astype(np.float32)
            
            gt_position_normalized = (actual_center - self.global_min) / self.global_range
            local_coord_normalized = (local_points - self.global_min) / self.global_range

            # 确保类型正确
            gt_position_normalized = gt_position_normalized.astype(np.float32)
            local_coord_normalized = local_coord_normalized.astype(np.float32)

            local_coord_original = local_points.astype(np.float32)
            gt_position_original = actual_center.astype(np.float32)

            # 返回时增加原始坐标
            return (local_coord_normalized, local_colors.astype(np.float32), gt_position_normalized,
                    local_coord_original, gt_position_original, radius, method, True)
        
        return None, None, None, None, None, None, None, False
    
    def process_single_bigpcd(self, pcd_path, bigpcd_id, global_sample_id_start, category_id):
        """处理单个大点云"""
        print(f"\n📂 处理: {pcd_path.name}")
        global_coord, global_color, pcd_info = self._load_big_pointcloud(pcd_path)
        print(f"   点数: {len(global_coord)}")
        print(f"   范围: X[{pcd_info['size'][0]:.6f}] Y[{pcd_info['size'][1]:.6f}] Z[{pcd_info['size'][2]:.6f}]")
        
        # 🔥 混合模式：为球体和立方体分别找合适的半径
        if self.mixed_mode:
            print(f"\n   🔄 混合提取模式")
            
            # 球体半径
            print(f"   ⚪ 球体方法:")
            sphere_radius = self._find_valid_radius(pcd_info, self.radius, method='sphere')
            if sphere_radius is None:
                print(f"   ❌ 球体方法无法找到合适半径")
                sphere_radius = 0
            
            # 立方体半径
            print(f"   🟦 立方体方法:")
            cube_radius = self._find_valid_radius(pcd_info, self.radius, method='cube')
            if cube_radius is None:
                print(f"   ❌ 立方体方法无法找到合适半径")
                cube_radius = 0
            
            if sphere_radius == 0 and cube_radius == 0:
                print(f"   ❌ 两种方法都无法找到合适半径，跳过")
                return [], global_sample_id_start
            
            # 🔥 生成提取计划
            extraction_plan = []
            
            if sphere_radius > 0:
                extraction_plan.extend([('sphere', sphere_radius)] * self.sphere_samples)
            
            if cube_radius > 0:
                extraction_plan.extend([('cube', cube_radius)] * self.cube_samples)
            
            # 打乱顺序（可选）
            import random
            random.shuffle(extraction_plan)
            
        else:
            # 单一模式
            adjusted_radius = self._find_valid_radius(pcd_info, self.radius, self.method)
            
            if adjusted_radius is None:
                print(f"   ❌ 无法找到合适的提取半径，跳过")
                return [], global_sample_id_start
            
            if abs(adjusted_radius - self.radius) > 1e-6:
                print(f"   📏 使用调整后的半径: {adjusted_radius:.6f}")
            
            extraction_plan = [(self.method, adjusted_radius)] * self.samples_per_bigpcd
        
        # 生成样本
        samples = []
        success_count = 0
        failed_count = 0
        sphere_count = 0
        cube_count = 0
        current_sample_id = global_sample_id_start
        
        pbar = tqdm(extraction_plan, desc=f"   提取样本", leave=False)
        
        parent_id_str = f"{bigpcd_id:03d}"  # "001", "002", ...
        
        for i, (method, radius) in enumerate(pbar):
            # 🔥 使用指定的方法和半径提取（步骤2会用全局参数归一化）
            (local_coord, local_color, gt_position,
            local_coord_original, gt_position_original,
            actual_radius, used_method, success) = self._extract_random_patch(global_coord, global_color, pcd_info, radius, method)
            
            if not success:
                failed_count += 1
                pbar.set_postfix({
                    '成功': success_count, 
                    '失败': failed_count,
                    '⚪球': sphere_count,
                    '🟦方': cube_count
                })
                continue
            
            # 统计
            if used_method == 'sphere':
                sphere_count += 1
            else:
                cube_count += 1
            
            # 🔥 步骤2：保存时使用全局归一化参数
            data_dict = {
                "local_coord": local_coord,  # 已用全局参数归一化
                "local_color": local_color,
                "gt_position": gt_position,  # 已用全局参数归一化
                
                # 🔥 保存全局归一化参数（所有样本相同）
                "norm_offset": self.global_min,   # 全局 min
                "norm_scale": self.global_range,  # 全局 scale

                "local_coord_original": local_coord_original,
                "gt_position_original": gt_position_original,      
                
                # 保留单个点云参数（用于调试和验证）
                "pcd_min": pcd_info['min'],
                "pcd_max": pcd_info['max'],
                "pcd_size": pcd_info['size'],
                
                # 提取方法
                "extraction_method": used_method,
                "extraction_radius": float(actual_radius),
                
                # 元信息
                "category": self.category_name,
                "category_id": category_id,
                "bigpcd_name": pcd_path.name,
                "bigpcd_id": bigpcd_id,
                "parent_id": parent_id_str,
                "sample_id": current_sample_id,
                "name": f"{self.category_name}_{parent_id_str}_{used_method[0]}{i:05d}",
            }

            # 文件名包含 parent_id
            output_filename = f"patch_{current_sample_id:06d}.pth"
            output_path = self.patches_dir / output_filename

            torch.save(data_dict, output_path)
            
            samples.append({
                'sample_id': current_sample_id,
                'name': data_dict['name'],
                'category': self.category_name,
                'method': used_method,
                'num_local_points': len(local_coord),
            })
            
            success_count += 1
            current_sample_id += 1
            
            pbar.set_postfix({
                '成功': success_count, 
                '失败': failed_count,
                '⚪球': sphere_count,
                '🟦方': cube_count
            })
        
        if success_count > 0:
            print(f"   ✅ 生成 {success_count} 个样本 (失败: {failed_count})")
            if self.mixed_mode:
                print(f"      ⚪ 球体: {sphere_count} 个")
                print(f"      🟦 立方体: {cube_count} 个")
        else:
            print(f"   ❌ 该点云无法生成有效样本")
        
        return samples, current_sample_id
    
    def generate_dataset(self, category_id=0):
        """生成完整数据集"""
        print(f"\n{'='*70}")
        print(f"🚀 生成 PTv3 训练数据集（全局归一化版本）")
        print(f"{'='*70}")
        print(f"   输入目录: {self.input_dir}")
        print(f"   输出目录: {self.output_dir}")
        print(f"   类别名称: {self.category_name}")
        
        if self.mixed_mode:
            print(f"   🔄 混合提取模式:")
            print(f"      球体样本: {self.sphere_samples}/大点云")
            print(f"      立方体样本: {self.cube_samples}/大点云")
            print(f"      总计: {self.samples_per_bigpcd}/大点云")
        else:
            print(f"   每个大点云生成: {self.samples_per_bigpcd} 个样本")
            print(f"   提取方法: {self.method}")
        
        print(f"   初始提取半径: {self.radius}")
        print(f"{'='*70}\n")
        
        # 查找所有 .ply 文件
        ply_files = sorted(self.input_dir.glob("*.ply"))
        
        if len(ply_files) == 0:
            print(f"❌ 在 {self.input_dir} 中未找到 .ply 文件")
            return None
        
        print(f"📂 发现 {len(ply_files)} 个点云文件:")
        for f in ply_files:
            print(f"   - {f.name}")
        
        # 🔥 步骤1：计算全局归一化参数
        self._compute_global_normalization(ply_files)
        
        # 处理每个大点云
        all_samples = []
        global_sample_id = 0
        
        for bigpcd_id, pcd_path in enumerate(ply_files, start=1):
            samples, global_sample_id = self.process_single_bigpcd(
                pcd_path, bigpcd_id, global_sample_id, category_id
            )
            all_samples.extend(samples)
        
        # 统计
        sphere_total = sum(1 for s in all_samples if s['method'] == 'sphere')
        cube_total = sum(1 for s in all_samples if s['method'] == 'cube')
        
        # 保存数据集信息
        dataset_info = {
            'total_samples': len(all_samples),
            'category': self.category_name,
            'category_id': category_id,
            'num_bigpcds': len(ply_files),
            'samples_per_bigpcd': self.samples_per_bigpcd,
            'initial_radius': float(self.radius),
            
            # 🔥 保存全局归一化参数
            'global_normalization': {
                'global_min': self.global_min.tolist(),
                'global_max': self.global_max.tolist(),
                'global_range': self.global_range.tolist(),  # 🔥 保存 range 而不是 scale
            }
        }
        
        if self.mixed_mode:
            dataset_info['mixed_mode'] = True
            dataset_info['sphere_samples_per_bigpcd'] = self.sphere_samples
            dataset_info['cube_samples_per_bigpcd'] = self.cube_samples
            dataset_info['sphere_total'] = sphere_total
            dataset_info['cube_total'] = cube_total
        else:
            dataset_info['mixed_mode'] = False
            dataset_info['method'] = self.method
        
        info_path = self.category_dir / "category_info.json"
        with open(info_path, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"🎉 数据集生成完成！")
        print(f"{'='*70}")
        print(f"✅ 总样本数: {len(all_samples)}")
        
        if self.mixed_mode:
            print(f"   ⚪ 球体样本: {sphere_total}")
            print(f"   🟦 立方体样本: {cube_total}")
        
        print(f"\n🌍 全局归一化参数:")
        print(f"   global_min:   {self.global_min}")
        print(f"   global_max:   {self.global_max}")
        print(f"   global_range: {self.global_range}")
        
        print(f"\n📁 输出目录: {self.category_dir}")
        print(f"📁 样本目录: {self.patches_dir}")
        print(f"📄 信息文件: {info_path}")
        print(f"{'='*70}\n")
        
        return dataset_info


def merge_categories(output_dir, categories):
    """合并多个类别的数据集信息"""
    output_dir = Path(output_dir)
    
    print(f"\n{'='*70}")
    print(f"📦 合并数据集信息")
    print(f"{'='*70}\n")
    
    category_to_id = {cat: i for i, cat in enumerate(sorted(categories))}
    
    all_info = {
        'categories': list(category_to_id.keys()),
        'category_to_id': category_to_id,
        'samples_by_category': {},
        'total_samples': 0,
    }
    
    for category in categories:
        category_dir = output_dir / category
        info_path = category_dir / "category_info.json"
        
        if not info_path.exists():
            print(f"⚠️  {category}: 信息文件不存在，跳过")
            continue
        
        with open(info_path) as f:
            cat_info = json.load(f)
        
        all_info['samples_by_category'][category] = cat_info['total_samples']
        all_info['total_samples'] += cat_info['total_samples']
        
        print(f"✅ {category:15s}: {cat_info['total_samples']:5d} 样本")
    
    # 保存全局信息
    global_info_path = output_dir / "dataset_info.json"
    with open(global_info_path, 'w') as f:
        json.dump(all_info, f, indent=2)
    
    print(f"\n📄 全局信息已保存: {global_info_path}")
    print(f"✅ 总样本数: {all_info['total_samples']}")
    print(f"{'='*70}\n")
    
    return all_info


def verify_single_sample(pth_path):
    """验证单个样本"""
    print(f"\n{'='*70}")
    print(f"🔍 验证样本: {pth_path}")
    print(f"{'='*70}\n")
    
    data_dict = torch.load(pth_path, weights_only=False)
    
    required_keys = [
        "local_coord", "local_color", "gt_position",
        "norm_offset", "norm_scale",  # 🔥 新增检查
        "pcd_min", "pcd_max", "pcd_size",
        "extraction_method", "extraction_radius",
        "category", "category_id", "bigpcd_name", "bigpcd_id", "parent_id", "sample_id", "name"
    ]
    
    print("字段检查:")
    for key in required_keys:
        if key in data_dict:
            value = data_dict[key]
            if isinstance(value, np.ndarray):
                print(f"   ✅ {key:20s} shape={str(value.shape):15s} dtype={value.dtype}")
            else:
                print(f"   ✅ {key:20s} value={value}")
        else:
            print(f"   ⚠️  {key:20s} 缺失")
    
    print(f"\n数据范围:")
    print(f"   Local coord:  [{data_dict['local_coord'].min():.4f}, {data_dict['local_coord'].max():.4f}]")
    print(f"   Local color:  [{data_dict['local_color'].min():.4f}, {data_dict['local_color'].max():.4f}]")
    
    gt_pos = data_dict['gt_position']
    print(f"   GT position:  [{gt_pos[0]:.6f}, {gt_pos[1]:.6f}, {gt_pos[2]:.6f}]")
    
    if 'extraction_method' in data_dict:
        print(f"\n提取方法:")
        print(f"   Method: {data_dict['extraction_method']}")
        print(f"   Radius: {data_dict['extraction_radius']:.6f}")
    
    # 🔥 新增：显示全局归一化参数
    if 'norm_offset' in data_dict and 'norm_scale' in data_dict:
        print(f"\n🌍 全局归一化参数:")
        norm_offset = data_dict['norm_offset']
        norm_scale = data_dict['norm_scale']
        
        print(f"   norm_offset (global_min): {norm_offset}")
        
        # 🔥 检查 norm_scale 的形状
        if isinstance(norm_scale, np.ndarray):
            if norm_scale.shape == (3,):
                print(f"   norm_scale (global_range): {norm_scale}")  # ✅ 正确
    
    if 'pcd_min' in data_dict:
        print(f"\n📦 单个点云参数（调试用）:")
        print(f"   pcd_min:  {data_dict['pcd_min']}")
        print(f"   pcd_max:  {data_dict['pcd_max']}")
        print(f"   pcd_size: {data_dict['pcd_size']}")
    
    print(f"\n坐标范围检查:")
    local_min = data_dict['local_coord'].min()
    local_max = data_dict['local_coord'].max()
    gt_min = gt_pos.min()
    gt_max = gt_pos.max()
    
    if 0 <= local_min and local_max <= 1:
        print(f"   ✅ 小点云坐标在 [0, 1] 范围内")
    else:
        print(f"   ⚠️  小点云坐标超出 [0, 1] 范围: [{local_min:.4f}, {local_max:.4f}]")
    
    if 0 <= gt_min and gt_max <= 1:
        print(f"   ✅ GT 位置在 [0, 1] 范围内")
    else:
        print(f"   ⚠️  GT 位置超出 [0, 1] 范围: [{gt_min:.4f}, {gt_max:.4f}]")
    
    local_center = data_dict['local_coord'].mean(axis=0)
    distance_to_gt = np.linalg.norm(local_center - gt_pos)
    print(f"\n质心检查:")
    print(f"   小点云质心: [{local_center[0]:.4f}, {local_center[1]:.4f}, {local_center[2]:.4f}]")
    print(f"   GT 位置:    [{gt_pos[0]:.4f}, {gt_pos[1]:.4f}, {gt_pos[2]:.4f}]")
    print(f"   距离:       {distance_to_gt:.6f}")
    
    if distance_to_gt < 0.01:
        print(f"   ✅ 质心位置合理")
    else:
        print(f"   ⚠️  质心距离 GT 较远")

    # 🔥 新增：反归一化验证
    if 'norm_offset' in data_dict and 'norm_scale' in data_dict:
        print(f"\n{'='*70}")
        print(f"🔄 反归一化验证")
        print(f"{'='*70}")
        
        norm_offset = data_dict['norm_offset']
        norm_scale = data_dict['norm_scale']
        
        # 反归一化局部坐标
        local_coord_normalized = data_dict['local_coord']
        local_coord_denormalized = local_coord_normalized * norm_scale + norm_offset
        
        # 反归一化 GT 位置
        gt_position_normalized = data_dict['gt_position']
        gt_position_denormalized = gt_position_normalized * norm_scale + norm_offset
        
        print(f"\n📊 反归一化结果:")
        print(f"   局部坐标范围:")
        print(f"      归一化:   [{local_coord_normalized.min():.6f}, {local_coord_normalized.max():.6f}]")
        print(f"      反归一化: [{local_coord_denormalized.min():.6f}, {local_coord_denormalized.max():.6f}]")
        
        print(f"\n   GT 位置:")
        print(f"      归一化:   [{gt_position_normalized[0]:.6f}, {gt_position_normalized[1]:.6f}, {gt_position_normalized[2]:.6f}]")
        print(f"      反归一化: [{gt_position_denormalized[0]:.6f}, {gt_position_denormalized[1]:.6f}, {gt_position_denormalized[2]:.6f}]")
        
        # 🔥 检查反归一化后的质心
        local_center_denormalized = local_coord_denormalized.mean(axis=0)
        distance_denormalized = np.linalg.norm(local_center_denormalized - gt_position_denormalized)
        
        print(f"\n   质心检查（反归一化后）:")
        print(f"      局部质心: [{local_center_denormalized[0]:.6f}, {local_center_denormalized[1]:.6f}, {local_center_denormalized[2]:.6f}]")
        print(f"      GT 位置:  [{gt_position_denormalized[0]:.6f}, {gt_position_denormalized[1]:.6f}, {gt_position_denormalized[2]:.6f}]")
        print(f"      距离:     {distance_denormalized:.6f} 米 = {distance_denormalized*1000:.2f} 毫米")
        
        if distance_denormalized < 0.001:  # 1mm 以内
            print(f"      ✅ 质心位置非常精确")
        elif distance_denormalized < 0.005:  # 5mm 以内
            print(f"      ✅ 质心位置合理")
        else:
            print(f"      ⚠️  质心距离 GT 较远")
        
        # 🔥 与单个点云参数对比（可选）
        if 'pcd_min' in data_dict and 'pcd_size' in data_dict:
            print(f"\n   📦 与单个点云参数对比:")
            
            # 使用单个点云参数反归一化
            pcd_min = data_dict['pcd_min']
            pcd_size = data_dict['pcd_size']
            
            local_coord_denorm_old = local_coord_normalized * pcd_size + pcd_min
            gt_position_denorm_old = gt_position_normalized * pcd_size + pcd_min
            
            print(f"      使用单个点云参数反归一化:")
            print(f"         局部坐标范围: [{local_coord_denorm_old.min():.6f}, {local_coord_denorm_old.max():.6f}]")
            print(f"         GT 位置: [{gt_position_denorm_old[0]:.6f}, {gt_position_denorm_old[1]:.6f}, {gt_position_denorm_old[2]:.6f}]")
            
            # 🔥 检查差异
            coord_diff = np.abs(local_coord_denormalized - local_coord_denorm_old).max()
            gt_diff = np.linalg.norm(gt_position_denormalized - gt_position_denorm_old)
            
            print(f"\n      差异分析:")
            print(f"         坐标最大差异: {coord_diff:.6f} 米 = {coord_diff*1000:.2f} 毫米")
            print(f"         GT 位置差异:  {gt_diff:.6f} 米 = {gt_diff*1000:.2f} 毫米")
            
            if coord_diff < 1e-6 and gt_diff < 1e-6:
                print(f"         ✅ 两种参数结果一致（说明是同一个父点云）")
            else:
                print(f"         ⚠️  两种参数结果不同（说明使用了不同的归一化空间）")
        
        print(f"{'='*70}\n")

    print(f"\n{'='*70}")
    print(f"✅ 验证完成")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成 PTv3 训练数据集（支持混合提取 + 全局归一化）")
    
    parser.add_argument("--input_dir", type=str, default=None,
                       help="输入目录（包含原始 .ply 文件）")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="输出根目录")
    parser.add_argument("--category", type=str, default=None,
                       help="类别名称（例如 Scissors, Cup, Avocado）")
    parser.add_argument("--category_id", type=int, default=0,
                       help="类别 ID（默认 0）")
    parser.add_argument("--samples_per_bigpcd", type=int, default=None,
                       help="每个大点云提取的样本数（单一模式）")
    parser.add_argument("--radius", type=float, default=0.01,
                       help="初始提取半径")
    parser.add_argument("--method", type=str, default="sphere",
                       choices=["sphere", "cube"],
                       help="提取方法（单一模式）")
    
    # 🔥 混合模式参数
    parser.add_argument("--sphere_samples", type=int, default=None,
                       help="球体方法提取的样本数/大点云（混合模式）")
    parser.add_argument("--cube_samples", type=int, default=None,
                       help="立方体方法提取的样本数/大点云（混合模式）")
    
    parser.add_argument("--verify", type=str, default=None,
                       help="验证单个样本")
    parser.add_argument("--merge", type=str, nargs='+', default=None,
                       help="合并多个类别的信息，例如: --merge Scissors Cup Avocado")
    parser.add_argument("--merge_dir", type=str, default=None,
                       help="合并时的输出根目录")
    
    args = parser.parse_args()
    
    if args.verify:
        # 验证模式
        verify_single_sample(args.verify)
    
    elif args.merge and args.merge_dir:
        # 合并模式
        merge_categories(args.merge_dir, args.merge)
    
    elif args.input_dir and args.output_dir and args.category:
        # 🔥 检查模式
        if args.sphere_samples is not None and args.cube_samples is not None:
            # 混合模式
            generator = PTv3DatasetGenerator(
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                category_name=args.category,
                radius=args.radius,
                sphere_samples=args.sphere_samples,
                cube_samples=args.cube_samples
            )
        elif args.samples_per_bigpcd is not None:
            # 单一模式
            generator = PTv3DatasetGenerator(
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                category_name=args.category,
                samples_per_bigpcd=args.samples_per_bigpcd,
                radius=args.radius,
                method=args.method
            )
        else:
            parser.error("请指定 --samples_per_bigpcd（单一模式）或 --sphere_samples + --cube_samples（混合模式）")
            exit(1)
        
        generator.generate_dataset(category_id=args.category_id)
    
    else:
        parser.error("请指定正确的参数组合")
        print("\n使用示例:")
        print("\n1. 单一模式（只用球体）:")
        print("   python script.py --input_dir scans --output_dir data_root --category Scissors --samples_per_bigpcd 100 --method sphere")
        print("\n2. 🔥 混合模式（球体50个 + 立方体50个）:")
        print("   python script.py --input_dir scans --output_dir data_root --category Scissors --sphere_samples 50 --cube_samples 50")
        print("\n3. 验证样本:")
        print("   python script.py --verify data_root/Scissors/patches/patch_001_000001.pth")
        print("\n4. 合并类别:")
        print("   python script.py --merge Scissors Cup Avocado --merge_dir data_root")