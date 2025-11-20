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
        model_input = {
            'local': {},
            'parent': {},
        }
        
        if 'local' not in input_dict:
            raise KeyError("Missing 'local' key in input_dict")
        
        if hasattr(self.model, 'use_parent_cloud') and self.model.use_parent_cloud:
            if 'parent' not in input_dict:
                raise KeyError("Missing 'parent' key in input_dict (required when use_parent_cloud=True)")
        
        local_data = input_dict['local']
        num_local_points = None
        for key in ['coord', 'feat', 'grid_coord']:
            if key not in local_data:
                raise KeyError(f"Missing '{key}' in input_dict['local']")
            
            value = local_data[key]
            
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value)
            elif not isinstance(value, torch.Tensor):
                value = torch.tensor(value)
            
            if num_local_points is None:
                num_local_points = value.shape[0]

            if key in ['coord', 'feat']:
                value = value.float()
            elif key == 'grid_coord':
                value = value.long()
            
            model_input['local'][key] = value.to(self.device) 

        # local point cloud batch and offset
        if 'batch' in local_data:
            batch = local_data['batch']
            if isinstance(batch, np.ndarray):
                batch = torch.from_numpy(batch).long()
            elif not isinstance(batch, torch.Tensor):
                batch = torch.tensor(batch).long()
            model_input['local']['batch'] = batch.to(self.device)
        else:
            model_input['local']['batch'] = torch.zeros(num_local_points, dtype=torch.long, device=self.device)
        
        if 'offset' in local_data:
            offset = local_data['offset']
            if isinstance(offset, np.ndarray):
                offset = torch.from_numpy(offset).long()
            elif not isinstance(offset, torch.Tensor):
                offset = torch.tensor(offset).long()
            model_input['local']['offset'] = offset.to(self.device)
        else:
            model_input['local']['offset'] = torch.tensor([num_local_points], dtype=torch.long, device=self.device)
            
        # deal parent point cloud    
        if hasattr(self.model, 'use_parent_cloud') and self.model.use_parent_cloud:
            parent_data = input_dict['parent']
            num_parent_points = None
            
            for key in ['coord', 'feat', 'grid_coord']:
                if key not in parent_data:
                    raise KeyError(f"Missing '{key}' in input_dict['parent']")
                
                value = parent_data[key]
                if isinstance(value, np.ndarray):
                    value = torch.from_numpy(value)
                elif not isinstance(value, torch.Tensor):
                    value = torch.tensor(value)

                if num_parent_points is None:
                    num_parent_points = value.shape[0]

                if key in ['coord', 'feat']:
                    value = value.float()
                elif key == 'grid_coord':
                    value = value.long()
                
                model_input['parent'][key] = value.to(self.device)
            # parent point cloud batch and offset    
            if 'batch' in parent_data:
                batch = parent_data['batch']
                if isinstance(batch, np.ndarray):
                    batch = torch.from_numpy(batch).long()
                elif not isinstance(batch, torch.Tensor):
                    batch = torch.tensor(batch).long()
                model_input['parent']['batch'] = batch.to(self.device)
            else:
                model_input['parent']['batch'] = torch.zeros(num_parent_points, dtype=torch.long, device=self.device)
            
            if 'offset' in parent_data:
                offset = parent_data['offset']
                if isinstance(offset, np.ndarray):
                    offset = torch.from_numpy(offset).long()
                elif not isinstance(offset, torch.Tensor):
                    offset = torch.tensor(offset).long()
                model_input['parent']['offset'] = offset.to(self.device)
            else:
                model_input['parent']['offset'] = torch.tensor([num_parent_points], dtype=torch.long, device=self.device)     

        # deal whole model input
        if 'grid_size' in input_dict:
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
                category_id = torch.tensor(category_id, dtype=torch.long)
            elif not isinstance(category_id, torch.Tensor):
                category_id = torch.tensor(category_id).long()

            if category_id.dim() == 0:
                category_id = category_id.unsqueeze(0)
            
            model_input['category_id'] = category_id.to(self.device)        

        # print debug information
        print("\n📊 模型输入:")
        print(f"   use_parent_cloud: {getattr(self.model, 'use_parent_cloud', False)}")
        print(f"   fusion_type: {getattr(self.model, 'fusion_type', 'N/A')}")
        
        if 'local' in model_input:
            print(f"\n   局部点云:")
            for key, value in model_input['local'].items():
                if isinstance(value, torch.Tensor):
                    print(f"      {key}: shape={value.shape}, dtype={value.dtype}")
        
        if 'parent' in model_input and len(model_input['parent']) > 0:
            print(f"\n   父点云:")
            for key, value in model_input['parent'].items():
                if isinstance(value, torch.Tensor):
                    print(f"      {key}: shape={value.shape}, dtype={value.dtype}")
        
        print(f"\n   全局参数:")
        if 'grid_size' in model_input:
            print(f"      grid_size: {model_input['grid_size'].item()}")
        if 'category_id' in model_input:
            print(f"      category_id: {model_input['category_id'].item()}")
        
        if 'norm_offset' in input_dict and 'norm_scale' in input_dict:
            print(f"\n📐 归一化参数:")
            print(f"   norm_offset: {input_dict['norm_offset']}")
            print(f"   norm_scale: {input_dict['norm_scale']}")  

        # output final prediction      
        with torch.no_grad():
            output_dict = self.model(model_input)
        
        pred_position = output_dict['pred_position'].cpu().numpy()[0]
        
        return pred_position


def load_patch_data(
    patch_path: Path, 
    parent_pcd_root: Path,
    grid_size: float = 0.002,
    verbose: bool = False
) -> Dict[str, torch.Tensor]:
    """
    加载原始 patch 数据，并转换为模型输入格式
    
    Args:
        patch_path: patch 文件路径 (例如: patch_000120.pth)
        parent_pcd_root: 父点云根目录 (例如: /home/ypf/touch_processed_data)
        grid_size: 体素化网格大小 (默认: 0.002)
        verbose: 是否打印详细信息
    
    Returns:
        处理后的字典（与 DataLoader 输出格式一致）
    """
    try:
        data = torch.load(patch_path, map_location='cpu', weights_only=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load {patch_path}: {e}")
    
    # ✅ 检查原始数据格式（扁平结构）
    required_keys = ['local_coord', 'local_color', 'gt_position', 'norm_offset', 'norm_scale']
    missing_keys = [k for k in required_keys if k not in data]
    if missing_keys:
        raise KeyError(
            f"Missing required keys: {missing_keys}\n"
            f"Available keys: {list(data.keys())}"
        )
    
    # ✅ 1. 提取局部点云数据
    local_coord = data['local_coord']  # (N_local, 3)
    local_color = data['local_color']  # (N_local, 3)
    
    if isinstance(local_coord, np.ndarray):
        local_coord = torch.from_numpy(local_coord).float()
    elif not isinstance(local_coord, torch.Tensor):
        local_coord = torch.tensor(local_coord).float()
    
    if isinstance(local_color, np.ndarray):
        local_color = torch.from_numpy(local_color).float()
    elif not isinstance(local_color, torch.Tensor):
        local_color = torch.tensor(local_color).float()
    
    # ✅ 2. 提取归一化参数
    norm_offset = data['norm_offset']
    norm_scale = data['norm_scale']
    
    if isinstance(norm_offset, np.ndarray):
        norm_offset = torch.from_numpy(norm_offset).float()
    elif not isinstance(norm_offset, torch.Tensor):
        norm_offset = torch.tensor(norm_offset).float()
    
    if isinstance(norm_scale, np.ndarray):
        norm_scale = torch.from_numpy(norm_scale).float()
    elif not isinstance(norm_scale, torch.Tensor):
        norm_scale = torch.tensor(norm_scale).float()
    
    # ✅ 3. 加载父点云
    bigpcd_id = data.get('bigpcd_id', data.get('parent_id'))
    category = data.get('category', 'Unknown')
    
    if bigpcd_id is None:
        raise KeyError("Missing 'bigpcd_id' or 'parent_id' in patch data")
    
    # 格式化 bigpcd_id（确保是 3 位字符串）
    if isinstance(bigpcd_id, (int, np.integer)):
        bigpcd_id_str = f"{bigpcd_id:03d}"
    else:
        bigpcd_id_str = str(bigpcd_id).zfill(3)
    
    # 构造父点云文件路径
    bigpcd_name = data.get('bigpcd_name', f'bigpointcloud_{bigpcd_id_str}.ply')
    
    # 尝试不同的路径格式
    possible_paths = [
        parent_pcd_root / category / bigpcd_name,
        parent_pcd_root / category / f'bigpointcloud_{bigpcd_id_str}.ply',
        parent_pcd_root / category / f'data{bigpcd_id_str}.ply',
        parent_pcd_root / bigpcd_name,
    ]
    
    parent_pcd_path = None
    for path in possible_paths:
        if path.exists():
            parent_pcd_path = path
            break
    
    if parent_pcd_path is None:
        raise FileNotFoundError(
            f"Cannot find parent point cloud for bigpcd_id={bigpcd_id}\n"
            f"Category: {category}\n"
            f"Tried paths:\n" + "\n".join([f"  - {p}" for p in possible_paths])
        )
    
    if verbose:
        print(f"   📂 加载父点云: {parent_pcd_path}")
    
    # 读取父点云
    import open3d as o3d
    parent_pcd = o3d.io.read_point_cloud(str(parent_pcd_path))
    parent_coord = np.asarray(parent_pcd.points).astype(np.float32)
    parent_color = np.asarray(parent_pcd.colors).astype(np.float32)
    
    parent_coord = torch.from_numpy(parent_coord).float()
    parent_color = torch.from_numpy(parent_color).float()
    
    # ✅ 4. 归一化坐标（使用数据集中保存的归一化参数）
    # 注意：局部点云已经是归一化后的，所以不需要再次归一化
    # 但父点云需要归一化
    local_coord_normalized = local_coord  # 已经归一化
    parent_coord_normalized = (parent_coord - norm_offset) / norm_scale
    
    # ✅ 5. 体素化（GridSample）
    local_grid_coord = torch.floor(local_coord_normalized / grid_size).long()
    parent_grid_coord = torch.floor(parent_coord_normalized / grid_size).long()
    
    # ✅ 6. 提取 GT 位置（已经是归一化的）
    gt_position = data['gt_position']
    if isinstance(gt_position, np.ndarray):
        gt_position = torch.from_numpy(gt_position).float()
    elif not isinstance(gt_position, torch.Tensor):
        gt_position = torch.tensor(gt_position).float()
    
    # ✅ 7. 提取 category_id
    category_id = data.get('category_id')
    if category_id is not None:
        if isinstance(category_id, np.ndarray):
            category_id = torch.from_numpy(category_id).long()
        elif isinstance(category_id, (int, np.integer)):
            category_id = torch.tensor(category_id, dtype=torch.long)
        elif not isinstance(category_id, torch.Tensor):
            category_id = torch.tensor(category_id).long()
    
    # ✅ 8. 构造嵌套结构（与 DataLoader 一致）
    result = {
        'local': {
            'coord': local_coord_normalized,  # (N_local, 3) - 已归一化
            'feat': local_color,               # (N_local, 3)
            'grid_coord': local_grid_coord,    # (N_local, 3)
            'offset': torch.tensor([local_coord_normalized.shape[0]], dtype=torch.long),
        },
        'parent': {
            'coord': parent_coord_normalized,  # (N_parent, 3) - 归一化
            'feat': parent_color,              # (N_parent, 3)
            'grid_coord': parent_grid_coord,   # (N_parent, 3)
            'offset': torch.tensor([parent_coord_normalized.shape[0]], dtype=torch.long),
            'name': str(parent_pcd_path.name),
        },
        'gt_position': gt_position,            # (3,) - 已归一化
        'norm_offset': norm_offset,            # (3,)
        'norm_scale': norm_scale,              # (3,) or scalar
        'category_id': category_id,            # scalar
        'name': data.get('name', patch_path.stem),
        'grid_size': grid_size,
        
        # 🔥 额外保存原始数据（用于可视化）
        '_raw_data': data,
        '_parent_pcd_path': parent_pcd_path,
    }
    
    if verbose:
        print(f"\n✅ 数据加载成功:")
        print(f"   局部点云: {result['local']['coord'].shape[0]} 点")
        print(f"   父点云: {result['parent']['coord'].shape[0]} 点")
        print(f"   GT 位置（归一化）: {gt_position.tolist()}")
        print(f"   归一化参数:")
        print(f"      offset: {norm_offset.tolist()}")
        print(f"      scale: {norm_scale.tolist()}")
        if category_id is not None:
            print(f"   类别 ID: {category_id.item()}")
        print(f"   父点云文件: {parent_pcd_path}")
    
    return result


def get_parent_model_from_data(patch_data: Dict) -> Optional[str]:
    if 'parent' in patch_data and 'name' in patch_data['parent']:
        parent_name = patch_data['parent']['name']
        if isinstance(parent_name, str):
            try:
                # 例如：bigpointcloud_001.ply → 001
                return parent_name.split('_')[-1].split('.')[0].zfill(3)
            except:
                pass
    
    return None


def denormalize_position(position_normalized: np.ndarray, norm_offset, norm_scale) -> np.ndarray:
    if isinstance(norm_offset, torch.Tensor):
        norm_offset = norm_offset.cpu().numpy()
    if isinstance(norm_scale, torch.Tensor):
        norm_scale = norm_scale.cpu().numpy()
    
    if isinstance(position_normalized, torch.Tensor):
        position_normalized = position_normalized.cpu().numpy()
    if not isinstance(position_normalized, np.ndarray):
        position_normalized = np.array(position_normalized)
    
    return position_normalized * norm_scale + norm_offset


def visualize_prediction(
    patch_data: Dict,
    pred_position: np.ndarray,
    gt_position: np.ndarray,
    complete_model_path: Path = None,  # 🔥 改为可选
    patch_name: str = "",
    save_dir: Path = None,
    show_window: bool = False
):
    """可视化预测结果"""
    geometries = []
    
    # 🔥 如果没有传入完整点云路径，从 patch_data 中提取
    if complete_model_path is None:
        if '_parent_pcd_path' in patch_data:
            complete_model_path = patch_data['_parent_pcd_path']
        else:
            raise ValueError("需要提供 complete_model_path 或在 patch_data 中包含 '_parent_pcd_path'")
    
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
    
    # 🔥 反归一化局部点云（用于可视化）
    if '_raw_data' in patch_data:
        # 使用原始数据（未归一化）
        coord_real = patch_data['_raw_data']['local_coord']
        if isinstance(coord_real, torch.Tensor):
            coord_real = coord_real.cpu().numpy()
    else:
        # 从归一化坐标反推
        coord_normalized = patch_data['local']['coord']
        if isinstance(coord_normalized, torch.Tensor):
            coord_normalized = coord_normalized.cpu().numpy()
        
        norm_offset = patch_data['norm_offset']
        norm_scale = patch_data['norm_scale']
        coord_real = denormalize_position(coord_normalized, norm_offset, norm_scale)
    
    patch_pcd = o3d.geometry.PointCloud()
    patch_pcd.points = o3d.utility.Vector3dVector(coord_real)
    patch_pcd.paint_uniform_color([1.0, 0.65, 0.0])
    geometries.append(patch_pcd)
    
    # 创建标记球体（保持不变）
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
    grid_size: float = 0.002,  # 🔥 新增参数
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
        
        patch_name = patch_path.stem
        complete_model_path = None
        
        try:
            # 🔥 加载数据（使用新的函数）
            patch_data = load_patch_data(
                patch_path, 
                parent_pcd_root=dataset_dir,
                grid_size=grid_size,
                verbose=False
            )
            
            patch_name = patch_data.get('name', patch_path.stem)
            complete_model_path = patch_data['_parent_pcd_path']
            
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
                patch_data['norm_offset'],
                patch_data['norm_scale']
            )
            gt_position = denormalize_position(
                gt_position_normalized,
                patch_data['norm_offset'],
                patch_data['norm_scale']
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
                'parent_id': patch_data['_raw_data'].get('parent_id', 'unknown'),
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
                'patch_path': str(patch_path),
                'reason': str(e)
            })
            continue
    
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
    parser.add_argument('--grid_size', type=float, default=0.002,
                        help='体素化网格大小（默认: 0.002）')
    
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
        visualize_best_worst_median=not args.no_vis,
        grid_size=args.grid_size,  # 🔥 传入 grid_size
    )


if __name__ == "__main__":
    main()