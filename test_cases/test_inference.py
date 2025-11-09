import sys
from pathlib import Path
import torch
import numpy as np
import open3d as o3d
import argparse
import json
import csv
from typing import Dict, Tuple, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pointcept.models import build_model
from pointcept.utils.config import Config


def extract_sample_id_from_patch_name(patch_name: str) -> Optional[int]:
    """
    从 patch 文件名提取样本 ID
    
    Examples:
        >>> extract_sample_id_from_patch_name("patch_000044.pth")
        44
        >>> extract_sample_id_from_patch_name("patch_000198.pth")
        198
    """
    patch_name = Path(patch_name).stem
    
    if patch_name.startswith('patch_'):
        try:
            return int(patch_name.split('_')[1])
        except:
            pass
    
    try:
        return int(patch_name.split('_')[-1])
    except:
        return None


class PTv3ContactMatcher:
    """PTv3 接触点位置预测器"""
    
    def __init__(self, config_path: str, checkpoint_path: str, device=None):
        self.config_path = Path(config_path)
        self.checkpoint_path = Path(checkpoint_path)
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        print(f"🔧 加载配置: {self.config_path}")
        self.cfg = Config.fromfile(str(self.config_path))
        
        print(f"🔧 构建模型...")
        self.model = build_model(self.cfg.model).to(self.device)
        
        print(f"🔧 加载权重: {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        self.model.load_state_dict(new_state_dict)
        self.model.eval()
        
        print(f"✅ 模型加载完成")
        if 'epoch' in checkpoint:
            print(f"   训练轮数: {checkpoint['epoch']}")
        if 'best_metric_value' in checkpoint:
            print(f"   最佳指标: {checkpoint['best_metric_value']:.6f}")
        print(f"   推理设备: {self.device}")
    
    def predict(self, input_dict: Dict, verbose: bool = False) -> np.ndarray:
        """
        预测接触点位置
        
        Args:
            input_dict: 包含点云数据的字典
            verbose: 是否打印详细信息
        
        Returns:
            pred_position: (3,) 预测的接触点位置（归一化空间）
        """
        model_input = {}
        
        num_points = None
        
        for key in ['coord', 'feat', 'grid_coord']:
            if key in input_dict:
                value = input_dict[key]
                
                if isinstance(value, np.ndarray):
                    value = torch.from_numpy(value).float()
                elif not isinstance(value, torch.Tensor):
                    value = torch.tensor(value).float()
                
                if num_points is None:
                    num_points = value.shape[0]
                
                if key in ['coord', 'feat']:
                    value = value.float()
                elif key == 'grid_coord':
                    value = value.long()
                
                model_input[key] = value.to(self.device)
        
        if 'batch' not in input_dict:
            model_input['batch'] = torch.zeros(num_points, dtype=torch.long).to(self.device)
        else:
            batch = input_dict['batch']
            if isinstance(batch, np.ndarray):
                batch = torch.from_numpy(batch).long()
            elif not isinstance(batch, torch.Tensor):
                batch = torch.tensor(batch).long()
            model_input['batch'] = batch.to(self.device)
        
        if 'offset' not in input_dict:
            model_input['offset'] = torch.tensor([num_points], dtype=torch.long).to(self.device)
        else:
            offset = input_dict['offset']
            if isinstance(offset, np.ndarray):
                offset = torch.from_numpy(offset).long()
            elif not isinstance(offset, torch.Tensor):
                offset = torch.tensor(offset).long()
            model_input['offset'] = offset.to(self.device)
        
        if 'grid_size' not in input_dict:
            grid_size = 0.002
            if hasattr(self.cfg, 'data'):
                data_cfg = self.cfg.data
                if 'train' in data_cfg and 'transform' in data_cfg['train']:
                    for transform in data_cfg['train']['transform']:
                        if isinstance(transform, dict) and transform.get('type') == 'GridSample':
                            grid_size = transform.get('grid_size', 0.002)
                            break
            model_input['grid_size'] = torch.tensor(grid_size, dtype=torch.float32).to(self.device)
        else:
            grid_size = input_dict['grid_size']
            if isinstance(grid_size, (int, float)):
                grid_size = torch.tensor(grid_size, dtype=torch.float32)
            elif isinstance(grid_size, np.ndarray):
                grid_size = torch.from_numpy(grid_size).float()
            elif not isinstance(grid_size, torch.Tensor):
                grid_size = torch.tensor(grid_size).float()
            model_input['grid_size'] = grid_size.to(self.device)
        
        if 'category_id' in input_dict:
            category_id = input_dict['category_id']
            
            if isinstance(category_id, np.ndarray):
                category_id = torch.from_numpy(category_id).long()
            elif isinstance(category_id, (int, np.integer)):
                category_id = torch.tensor(category_id).long()
            elif not isinstance(category_id, torch.Tensor):
                category_id = torch.tensor(category_id).long()
            
            if category_id.dim() == 0:
                category_id = category_id.unsqueeze(0)
            
            model_input['category_id'] = category_id.to(self.device)
            
            if verbose:
                cat_names = ["Scissors", "Cup", "Avocado"]
                cat_id = category_id.item() if category_id.dim() == 1 else category_id[0].item()
                if 0 <= cat_id < len(cat_names):
                    print(f"   🏷️  物体类别: {cat_names[cat_id]} (ID={cat_id})")
        
        if verbose:
            print("\n📊 模型输入:")
            for key, value in model_input.items():
                if isinstance(value, torch.Tensor):
                    if value.dim() == 0:
                        print(f"   {key}: 标量值={value.item()}, dtype={value.dtype}")
                    else:
                        print(f"   {key}: shape={value.shape}, dtype={value.dtype}")
        
        with torch.no_grad():
            output_dict = self.model(model_input)
        
        pred_position = output_dict['pred_position'].cpu().numpy()[0]
        
        return pred_position


def load_patch_data(patch_path: Path, verbose: bool = False) -> Dict[str, torch.Tensor]:
    """
    加载 .pth 格式的小点云数据
    """
    try:
        data = torch.load(patch_path, map_location='cpu', weights_only=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load {patch_path}: {e}")
    
    # 键名映射
    if 'local_coord' in data and 'coord' not in data:
        data['coord'] = data['local_coord']
        if verbose:
            print(f"   ✅ 映射: local_coord → coord")
    
    if 'local_color' in data:
        if 'feat' not in data:
            data['feat'] = data['local_color']
            if verbose:
                print(f"   ✅ 映射: local_color → feat")
        if 'color' not in data:
            data['color'] = data['local_color']
    
    # 检查必要的键
    missing_keys = []
    if 'coord' not in data:
        missing_keys.append('coord')
    if 'gt_position' not in data:
        missing_keys.append('gt_position')
    
    if missing_keys:
        raise KeyError(
            f"Missing required keys: {missing_keys}\n"
            f"Available keys: {list(data.keys())}"
        )
    
    # 处理 feat
    if 'feat' not in data:
        if 'color' in data:
            data['feat'] = data['color']
        elif 'local_color' in data:
            data['feat'] = data['local_color']
        else:
            raise KeyError(f"Missing 'feat' or 'color' in {patch_path}")
    
    # 转换为 torch.Tensor
    for key in ['coord', 'feat', 'grid_coord', 'gt_position']:
        if key in data:
            value = data[key]
            if isinstance(value, np.ndarray):
                data[key] = torch.from_numpy(value)
            elif not isinstance(value, torch.Tensor):
                data[key] = torch.tensor(value)
    
    if 'category_id' in data:
        value = data['category_id']
        if isinstance(value, np.ndarray):
            data['category_id'] = torch.from_numpy(value)
        elif isinstance(value, (int, np.integer)):
            data['category_id'] = torch.tensor(value)
        elif not isinstance(value, torch.Tensor):
            data['category_id'] = torch.tensor(value)
    
    return data


def get_parent_model_from_data(patch_data: Dict) -> Optional[str]:
    """从数据字典中直接读取父点云 ID"""
    if 'bigpcd_id' in patch_data:
        bigpcd_id = patch_data['bigpcd_id']
        if isinstance(bigpcd_id, torch.Tensor):
            bigpcd_id = bigpcd_id.item()
        return f"{int(bigpcd_id):03d}"
    
    if 'bigpcd_name' in patch_data:
        name = patch_data['bigpcd_name']
        if isinstance(name, str):
            try:
                return name.split('_')[-1].split('.')[0].zfill(3)
            except:
                pass
    
    return None


def denormalize_position(position_normalized: np.ndarray, pcd_min, pcd_size) -> np.ndarray:
    """
    反归一化位置坐标
    
    Args:
        position_normalized: 归一化坐标 [0, 1]
        pcd_min: 点云最小值
        pcd_size: 点云尺寸
    
    Returns:
        position_real: 真实坐标（米）
    """
    if isinstance(pcd_min, torch.Tensor):
        pcd_min = pcd_min.cpu().numpy()
    if isinstance(pcd_size, torch.Tensor):
        pcd_size = pcd_size.cpu().numpy()
    
    return position_normalized * pcd_size + pcd_min


def visualize_prediction(
    patch_data: Dict,
    pred_position: np.ndarray,
    gt_position: np.ndarray,
    complete_model_path: Path,
    patch_name: str = "",
    save_dir: Path = None,
    show_window: bool = False
):
    """
    在完整的父点云上可视化预测结果
    
    ⚠️  重要：pred_position 和 gt_position 必须是真实空间的坐标（米）
    """
    geometries = []
    
    if not complete_model_path.exists():
        raise FileNotFoundError(f"完整点云模型不存在: {complete_model_path}")
    
    print(f"📂 加载完整父点云: {complete_model_path.name}")
    complete_pcd = o3d.io.read_point_cloud(str(complete_model_path))
    complete_pcd.paint_uniform_color([0.85, 0.85, 0.85])
    geometries.append(complete_pcd)
    
    complete_points = np.asarray(complete_pcd.points)
    pcd_min_complete = complete_points.min(axis=0)
    pcd_max_complete = complete_points.max(axis=0)
    pcd_size_complete = np.linalg.norm(pcd_max_complete - pcd_min_complete)
    pcd_center = (pcd_min_complete + pcd_max_complete) / 2
    
    # 反归一化局部点云
    coord_normalized = patch_data['coord']
    if isinstance(coord_normalized, torch.Tensor):
        coord_normalized = coord_normalized.cpu().numpy()
    else:
        coord_normalized = np.array(coord_normalized)
    
    data_pcd_min = patch_data.get('pcd_min')
    data_pcd_size = patch_data.get('pcd_size')
    
    if data_pcd_min is not None and data_pcd_size is not None:
        coord_real = denormalize_position(coord_normalized, data_pcd_min, data_pcd_size)
    else:
        coord_real = coord_normalized
    
    patch_pcd = o3d.geometry.PointCloud()
    patch_pcd.points = o3d.utility.Vector3dVector(coord_real)
    patch_pcd.paint_uniform_color([1.0, 0.65, 0.0])
    geometries.append(patch_pcd)
    
    # 创建标记球体
    sphere_radius = pcd_size_complete * 0.01
    
    gt_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
    gt_sphere.translate(gt_position)
    gt_sphere.paint_uniform_color([0, 0, 1])
    gt_sphere.compute_vertex_normals()
    geometries.append(gt_sphere)
    
    pred_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
    pred_sphere.translate(pred_position)
    pred_sphere.paint_uniform_color([1, 0, 0])
    pred_sphere.compute_vertex_normals()
    geometries.append(pred_sphere)
    
    # 连接线
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.array([pred_position, gt_position]))
    line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
    line_set.colors = o3d.utility.Vector3dVector([[1, 1, 0]])
    geometries.append(line_set)
    
    # 坐标系
    coord_size = pcd_size_complete * 0.05
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=coord_size, 
        origin=pcd_center
    )
    geometries.append(coord_frame)
    
    error = np.linalg.norm(pred_position - gt_position)
    
    print(f"\n{'='*70}")
    print(f"🎨 可视化 ({patch_name}):")
    print(f"   误差: {error:.6f} 米 = {error*1000:.2f} 毫米")
    print(f"{'='*70}\n")
    
    # 保存结果
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存 PLY 文件
        combined_pcd = o3d.geometry.PointCloud()
        combined_pcd += complete_pcd
        combined_pcd += patch_pcd
        combined_pcd += gt_sphere.sample_points_uniformly(number_of_points=1000)
        combined_pcd += pred_sphere.sample_points_uniformly(number_of_points=1000)
        
        ply_path = save_dir / f"{patch_name}_visualization.ply"
        o3d.io.write_point_cloud(str(ply_path), combined_pcd)
        print(f"✅ PLY 文件已保存: {ply_path}")
        
        # 保存 JSON
        result = {
            'patch_name': patch_name,
            'pred_position': pred_position.tolist(),
            'gt_position': gt_position.tolist(),
            'error_meters': float(error),
            'error_mm': float(error * 1000),
            'complete_model': str(complete_model_path),
        }
        json_path = save_dir / f"{patch_name}_result.json"
        with open(json_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        # 保存 matplotlib 图
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(15, 5))
            
            views = [
                (0, 90, "Top View (XY)"),
                (0, 0, "Front View (XZ)"),
                (90, 90, "Side View (YZ)")
            ]
            
            for i, (elev, azim, title) in enumerate(views):
                ax = fig.add_subplot(1, 3, i+1, projection='3d')
                
                sample_idx = np.random.choice(len(complete_points), 
                                             min(5000, len(complete_points)), 
                                             replace=False)
                ax.scatter(complete_points[sample_idx, 0], 
                          complete_points[sample_idx, 1], 
                          complete_points[sample_idx, 2],
                          c='gray', s=0.1, alpha=0.3, label='Complete Model')
                
                ax.scatter(coord_real[:, 0], coord_real[:, 1], coord_real[:, 2],
                          c='orange', s=5, alpha=0.8, label='Local Patch')
                
                ax.scatter(gt_position[0], gt_position[1], gt_position[2],
                          c='blue', s=200, marker='o', label='GT Position', 
                          edgecolors='black', linewidths=2)
                
                ax.scatter(pred_position[0], pred_position[1], pred_position[2],
                          c='red', s=200, marker='o', label='Predicted Position',
                          edgecolors='black', linewidths=2)
                
                ax.plot([gt_position[0], pred_position[0]],
                       [gt_position[1], pred_position[1]],
                       [gt_position[2], pred_position[2]],
                       'y-', linewidth=2, label='Error Line')
                
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.set_title(f"{title}\nError: {error*1000:.2f} mm")
                ax.view_init(elev=elev, azim=azim)
                
                if i == 0:
                    ax.legend(fontsize=8, loc='upper right')
            
            plt.suptitle(f"Contact Point Prediction - {patch_name}", fontsize=14)
            plt.tight_layout()
            
            plt_path = save_dir / f"{patch_name}_matplotlib.png"
            plt.savefig(plt_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Matplotlib 图片已保存: {plt_path}")
        except Exception as e:
            print(f"⚠️  Matplotlib 渲染失败: {e}")
    
    if show_window:
        try:
            o3d.visualization.draw_geometries(
                geometries,
                window_name=f"接触点预测 - {patch_name}",
                width=1920,
                height=1080,
                point_show_normal=False
            )
        except Exception as e:
            print(f"⚠️  无法显示窗口: {e}")


def test_all_patches(
    matcher: PTv3ContactMatcher,
    dataset_dir: Path,
    category: str = "Scissors",
    save_dir: Path = Path("inference_results"),
    visualize_best_worst_median: bool = True,
):
    """测试数据集中的所有样本"""
    print(f"\n{'='*80}")
    print(f"🚀 批量测试所有点云")
    print(f"{'='*80}")
    
    category_dir = dataset_dir / category / "patches"
    
    if not category_dir.exists():
        print(f"❌ 目录不存在: {category_dir}")
        return
    
    patch_files = sorted(category_dir.glob("*.pth"))
    
    if not patch_files:
        print(f"❌ 目录中没有 .pth 文件: {category_dir}")
        return
    
    print(f"📂 数据集目录: {category_dir}")
    print(f"📊 找到 {len(patch_files)} 个小点云")
    
    # 查找所有完整点云文件
    complete_model_dir = dataset_dir / category
    complete_models = {}
    
    for ply_file in sorted(complete_model_dir.glob("bigpointcloud_*.ply")):
        model_id = ply_file.stem.split('_')[-1]
        complete_models[model_id] = ply_file
    
    print(f"📂 找到 {len(complete_models)} 个大点云")
    if complete_models:
        print(f"   模型 ID: {list(complete_models.keys())}")
    
    # 构建映射表
    print(f"\n🔍 正在构建 sample_id → bigpcd_id 映射表...")
    
    sample_to_bigpcd = {}
    bigpcd_to_samples = {}
    
    for patch_file in patch_files:
        sample_id = extract_sample_id_from_patch_name(patch_file.name)
        
        if sample_id is None:
            continue
        
        try:
            data = torch.load(patch_file, map_location='cpu', weights_only=False)
            bigpcd_id = get_parent_model_from_data(data)
            
            if bigpcd_id:
                sample_to_bigpcd[sample_id] = bigpcd_id
                
                if bigpcd_id not in bigpcd_to_samples:
                    bigpcd_to_samples[bigpcd_id] = []
                bigpcd_to_samples[bigpcd_id].append(sample_id)
        except Exception as e:
            print(f"   ⚠️  读取失败: {patch_file.name} - {e}")
    
    print(f"\n📋 实际映射关系:")
    for bigpcd_id in sorted(bigpcd_to_samples.keys()):
        sample_ids = sorted(bigpcd_to_samples[bigpcd_id])
        print(f"   bigpointcloud_{bigpcd_id}.ply → {len(sample_ids)} 个小点云")
        print(f"      sample_id 范围: {sample_ids[0]} ~ {sample_ids[-1]}")
    
    # 创建保存目录
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    batch_dir = save_dir / f"batch_{category}_{len(patch_files)}_samples"
    batch_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 结果保存目录: {batch_dir}")
    
    # 逐个测试
    results = []
    failed_samples = []
    
    print(f"\n{'='*80}")
    print(f"🔮 开始批量推理...")
    print(f"{'='*80}\n")
    
    for i, patch_path in enumerate(patch_files):
        print(f"[{i+1}/{len(patch_files)}] 处理: {patch_path.name}")
        
        # 🔥 初始化变量（防止未定义错误）
        complete_model_path = None
        bigpcd_id = None
        patch_name = patch_path.stem
        
        try:
            # 加载数据
            patch_data = load_patch_data(patch_path, verbose=False)
            patch_name = patch_data.get('name', patch_path.stem)
            
            # 🔥 获取 bigpcd_id
            bigpcd_id = get_parent_model_from_data(patch_data)
            
            if bigpcd_id is None:
                sample_id = extract_sample_id_from_patch_name(patch_path.name)
                if sample_id is not None:
                    bigpcd_id = sample_to_bigpcd.get(sample_id)
            
            # 🔥 获取完整点云路径
            if bigpcd_id:
                complete_model_path = complete_models.get(bigpcd_id)
            
            if not complete_model_path:
                if complete_models:
                    complete_model_path = list(complete_models.values())[0]
                    bigpcd_id = complete_model_path.stem.split('_')[-1]
                    print(f"   ⚠️  使用默认: {complete_model_path.name}")
                else:
                    raise FileNotFoundError("找不到完整点云模型")
            
            if not complete_model_path.exists():
                raise FileNotFoundError(f"完整点云文件不存在: {complete_model_path}")
            
            # 预测（得到归一化坐标）
            pred_position_normalized = matcher.predict(patch_data, verbose=False)
            
            # GT 位置（归一化坐标）
            gt_position_normalized = patch_data['gt_position']
            if isinstance(gt_position_normalized, torch.Tensor):
                gt_position_normalized = gt_position_normalized.cpu().numpy()
            else:
                gt_position_normalized = np.array(gt_position_normalized)
            
            # 🔥 反归一化到真实空间
            pred_position = denormalize_position(
                pred_position_normalized, 
                patch_data['pcd_min'], 
                patch_data['pcd_size']
            )
            gt_position = denormalize_position(
                gt_position_normalized,
                patch_data['pcd_min'],
                patch_data['pcd_size']
            )
            
            # 计算误差（真实空间）
            error = np.linalg.norm(pred_position - gt_position)
            
            # 保存结果
            result = {
                'index': i,
                'patch_name': patch_name,
                'patch_path': str(patch_path),
                'complete_model_path': str(complete_model_path),
                'complete_model_name': complete_model_path.name,
                'parent_id': bigpcd_id,
                'pred_position_normalized': pred_position_normalized.tolist(),
                'gt_position_normalized': gt_position_normalized.tolist(),
                'pred_position': pred_position.tolist(),
                'gt_position': gt_position.tolist(),
                'error_meters': float(error),
                'error_mm': float(error * 1000),
                'patch_data': patch_data,
            }
            results.append(result)
            
            print(f"   ✅ 误差: {error*1000:.2f} mm")
            
        except Exception as e:
            print(f"   ❌ 失败: {e}")
            failed_samples.append({
                'patch_name': patch_name,
                'reason': str(e)
            })
            continue  # 🔥 关键：跳过失败的样本
    
    # 统计分析
    print(f"\n{'='*80}")
    print(f"📈 统计分析")
    print(f"{'='*80}")
    
    if not results:
        print("❌ 没有成功的样本！")
        return
    
    errors = np.array([r['error_mm'] for r in results])
    
    print(f"总样本数: {len(patch_files)}")
    print(f"成功样本数: {len(results)}")
    print(f"失败样本数: {len(failed_samples)}")
    print(f"\n误差统计 (毫米):")
    print(f"  平均值:   {errors.mean():.2f} mm")
    print(f"  中位数:   {np.median(errors):.2f} mm")
    print(f"  标准差:   {errors.std():.2f} mm")
    print(f"  最小值:   {errors.min():.2f} mm")
    print(f"  最大值:   {errors.max():.2f} mm")
    
    # 找出最好、最差、中位数样本
    best_idx = np.argmin(errors)
    worst_idx = np.argmax(errors)
    median_idx = np.argmin(np.abs(errors - np.median(errors)))
    
    best_sample = results[best_idx]
    worst_sample = results[worst_idx]
    median_sample = results[median_idx]
    
    print(f"\n🏆 最佳样本: {best_sample['patch_name']} ({best_sample['error_mm']:.2f} mm)")
    print(f"📉 最差样本: {worst_sample['patch_name']} ({worst_sample['error_mm']:.2f} mm)")
    print(f"📊 中位数样本: {median_sample['patch_name']} ({median_sample['error_mm']:.2f} mm)")
    
    # 保存统计结果
    summary = {
        'category': category,
        'total_samples': len(patch_files),
        'successful_samples': len(results),
        'failed_samples': len(failed_samples),
        'statistics': {
            'mean_error_mm': float(errors.mean()),
            'median_error_mm': float(np.median(errors)),
            'std_error_mm': float(errors.std()),
            'min_error_mm': float(errors.min()),
            'max_error_mm': float(errors.max()),
        },
        'best_sample': {
            'name': best_sample['patch_name'],
            'error_mm': best_sample['error_mm'],
        },
        'worst_sample': {
            'name': worst_sample['patch_name'],
            'error_mm': worst_sample['error_mm'],
        },
        'median_sample': {
            'name': median_sample['patch_name'],
            'error_mm': median_sample['error_mm'],
        },
        'failed_samples': failed_samples,
    }
    
    summary_path = batch_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n✅ 统计结果已保存: {summary_path}")
    
    # 保存 CSV
    csv_path = batch_dir / "all_results.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'index', 'patch_name', 'parent_id', 'complete_model_name',
            'pred_x', 'pred_y', 'pred_z',
            'gt_x', 'gt_y', 'gt_z',
            'error_mm'
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({
                'index': r['index'],
                'patch_name': r['patch_name'],
                'parent_id': r['parent_id'],
                'complete_model_name': r['complete_model_name'],
                'pred_x': r['pred_position'][0],
                'pred_y': r['pred_position'][1],
                'pred_z': r['pred_position'][2],
                'gt_x': r['gt_position'][0],
                'gt_y': r['gt_position'][1],
                'gt_z': r['gt_position'][2],
                'error_mm': r['error_mm'],
            })
    print(f"✅ CSV 已保存: {csv_path}")
    
    # 生成误差分布图
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        ax = axes[0, 0]
        ax.hist(errors, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        ax.axvline(errors.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {errors.mean():.2f} mm')
        ax.axvline(np.median(errors), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(errors):.2f} mm')
        ax.set_xlabel('Error (mm)')
        ax.set_ylabel('Frequency')
        ax.set_title('Error Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[0, 1]
        bp = ax.boxplot(errors, vert=True, patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        ax.set_ylabel('Error (mm)')
        ax.set_title('Error Boxplot')
        ax.grid(True, alpha=0.3)
        
        ax = axes[1, 0]
        ax.plot(range(len(errors)), errors, 'o-', markersize=2, linewidth=0.5, alpha=0.6)
        ax.axhline(errors.mean(), color='red', linestyle='--', linewidth=2, label='Mean')
        ax.axhline(np.median(errors), color='green', linestyle='--', linewidth=2, label='Median')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Error (mm)')
        ax.set_title('Error vs Sample Index')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[1, 1]
        sorted_errors = np.sort(errors)
        cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        ax.plot(sorted_errors, cumulative, linewidth=2)
        ax.axvline(errors.mean(), color='red', linestyle='--', linewidth=2, label='Mean')
        ax.axvline(np.median(errors), color='green', linestyle='--', linewidth=2, label='Median')
        ax.set_xlabel('Error (mm)')
        ax.set_ylabel('Cumulative Probability')
        ax.set_title('Cumulative Distribution Function')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Error Analysis - {category} ({len(results)} samples)', fontsize=16)
        plt.tight_layout()
        
        plot_path = batch_dir / "error_analysis.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 误差分析图已保存: {plot_path}")
    except Exception as e:
        print(f"⚠️  生成误差分析图失败: {e}")
    
    # 可视化代表性样本
    if visualize_best_worst_median:
        print(f"\n🎨 生成代表性样本的可视化\n")
        
        samples_to_visualize = [
            (best_sample, "best"),
            (median_sample, "median"),
            (worst_sample, "worst"),
        ]
        
        for sample, label in samples_to_visualize:
            try:
                visualize_prediction(
                    sample['patch_data'],
                    np.array(sample['pred_position']),
                    np.array(sample['gt_position']),
                    Path(sample['complete_model_path']),
                    patch_name=f"{label}_{sample['patch_name']}",
                    save_dir=batch_dir,
                    show_window=False
                )
            except Exception as e:
                print(f"   ⚠️  可视化失败: {e}")
    
    print(f"\n{'='*80}")
    print(f"✅ 批量测试完成！")
    print(f"📂 结果保存在: {batch_dir}")
    print(f"{'='*80}\n")
    
    return results, summary


def main():
    parser = argparse.ArgumentParser(description='PTv3 Contact Position Regression 推理')
    parser.add_argument('--config', type=str, 
                        default='configs/s3dis/semseg-pt-v3m1-gelsight.py')
    parser.add_argument('--checkpoint', type=str,
                        default='exp/gelsight_test/model/model_best.pth')
    parser.add_argument('--dataset_dir', type=str,
                        default='../../touch_processed_data')
    parser.add_argument('--category', type=str,
                        default='Scissors',
                        choices=['Scissors', 'Cup', 'Avocado'])
    parser.add_argument('--save_dir', type=str,
                        default='inference_results')
    parser.add_argument('--no_vis', action='store_true',
                        help='不生成可视化')
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"🚀 初始化 PTv3 Contact Matcher")
    print(f"{'='*80}")
    
    matcher = PTv3ContactMatcher(
        config_path=args.config,
        checkpoint_path=args.checkpoint
    )
    
    test_all_patches(
        matcher,
        dataset_dir=Path(args.dataset_dir),
        category=args.category,
        save_dir=Path(args.save_dir),
        visualize_best_worst_median=not args.no_vis
    )


if __name__ == "__main__":
    main()