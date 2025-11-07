import sys
from pathlib import Path
import torch
import numpy as np
import open3d as o3d
import argparse
from typing import Dict, Tuple, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pointcept.models import build_model
from pointcept.utils.config import Config


class PTv3ContactMatcher:
    """PTv3 接触点位置预测器"""
    
    def __init__(self, config_path: str, checkpoint_path: str, device=None):
        """
        Args:
            config_path: 配置文件路径
            checkpoint_path: 模型权重路径
            device: 推理设备
        """
        self.config_path = Path(config_path)
        self.checkpoint_path = Path(checkpoint_path)
        
        # 设备选择
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # 加载配置
        print(f"🔧 加载配置: {self.config_path}")
        self.cfg = Config.fromfile(str(self.config_path))
        
        # 构建模型
        print(f"🔧 构建模型...")
        self.model = build_model(self.cfg.model).to(self.device)
        
        # 加载权重
        print(f"🔧 加载权重: {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # 移除 'module.' 前缀（如果有）
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
    
    def predict(self, input_dict: Dict) -> np.ndarray:
        """
        预测接触点位置
        
        Args:
            input_dict: 包含点云数据的字典（可能是 torch.Tensor 或 numpy.ndarray）
                - 'coord': (N, 3) 点坐标
                - 'feat': (N, 3) 点特征
                - 'category_id': (1,) 类别 ID（可选）
        
        Returns:
            pred_position: (3,) 预测的接触点位置
        """
        # 准备输入
        model_input = {}
        
        # 🔥 首先获取点云数量
        num_points = None
        
        for key in ['coord', 'feat', 'grid_coord']:
            if key in input_dict:
                value = input_dict[key]
                
                # 统一转换为 torch.Tensor
                if isinstance(value, np.ndarray):
                    value = torch.from_numpy(value).float()
                elif not isinstance(value, torch.Tensor):
                    value = torch.tensor(value).float()
                
                # 记录点数
                if num_points is None:
                    num_points = value.shape[0]
                
                # 确保正确的数据类型
                if key in ['coord', 'feat']:
                    value = value.float()
                elif key == 'grid_coord':
                    value = value.long()
                
                model_input[key] = value.to(self.device)
        
        # 🔥 添加 batch 键
        if 'batch' not in input_dict:
            model_input['batch'] = torch.zeros(num_points, dtype=torch.long).to(self.device)
        else:
            batch = input_dict['batch']
            if isinstance(batch, np.ndarray):
                batch = torch.from_numpy(batch).long()
            elif not isinstance(batch, torch.Tensor):
                batch = torch.tensor(batch).long()
            model_input['batch'] = batch.to(self.device)
        
        # 🔥 添加 offset 键
        if 'offset' not in input_dict:
            model_input['offset'] = torch.tensor([num_points], dtype=torch.long).to(self.device)
        else:
            offset = input_dict['offset']
            if isinstance(offset, np.ndarray):
                offset = torch.from_numpy(offset).long()
            elif not isinstance(offset, torch.Tensor):
                offset = torch.tensor(offset).long()
            model_input['offset'] = offset.to(self.device)
        
        # 🔥 添加 grid_size（从配置文件中获取，或使用默认值）
        if 'grid_size' not in input_dict:
            # 从配置文件中获取 grid_size
            grid_size = None
            
            # 查找配置中的 grid_size
            if hasattr(self.cfg, 'data'):
                data_cfg = self.cfg.data
                if 'train' in data_cfg and 'transform' in data_cfg['train']:
                    for transform in data_cfg['train']['transform']:
                        if isinstance(transform, dict) and transform.get('type') == 'GridSample':
                            grid_size = transform.get('grid_size', 0.002)
                            break
            
            # 如果还是没找到，使用默认值
            if grid_size is None:
                grid_size = 0.002  # 默认 2mm
            
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
        
        # 🔥 添加类别信息（你的模型需要这个！）
        if 'category_id' in input_dict:
            category_id = input_dict['category_id']
            
            # 统一转换为 torch.Tensor
            if isinstance(category_id, np.ndarray):
                category_id = torch.from_numpy(category_id).long()
            elif isinstance(category_id, (int, np.integer)):
                category_id = torch.tensor(category_id).long()
            elif not isinstance(category_id, torch.Tensor):
                category_id = torch.tensor(category_id).long()
            
            # 处理标量
            if category_id.dim() == 0:
                category_id = category_id.unsqueeze(0)
            
            model_input['category_id'] = category_id.to(self.device)
            
            # 打印类别信息
            cat_names = ["Scissors", "Cup", "Avocado"]
            cat_id = category_id.item() if category_id.dim() == 1 else category_id[0].item()
            if 0 <= cat_id < len(cat_names):
                print(f"   🏷️  物体类别: {cat_names[cat_id]} (ID={cat_id})")
        else:
            print("   ⚠️  没有提供 category_id，模型可能需要这个信息！")
        
        # 🔥 打印输入信息（调试用）
        print("\n📊 模型输入:")
        for key, value in model_input.items():
            if isinstance(value, torch.Tensor):
                if value.dim() == 0:  # 标量
                    print(f"   {key}: 标量值={value.item()}, dtype={value.dtype}")
                else:
                    print(f"   {key}: shape={value.shape}, dtype={value.dtype}")
                    if key == 'batch' and value.numel() > 0:
                        unique_batches = torch.unique(value)
                        print(f"      唯一批次: {unique_batches.tolist()}")
                    elif key == 'offset':
                        print(f"      值: {value.tolist()}")
                    elif key == 'category_id':
                        print(f"      值: {value.tolist()}")
            else:
                print(f"   {key}: {value}")
        
        # 推理
        with torch.no_grad():
            output_dict = self.model(model_input)
        
        # 提取预测位置
        pred_position = output_dict['pred_position'].cpu().numpy()[0]  # (3,)
        
        return pred_position


def load_patch_data(patch_path: Path, verbose: bool = False) -> Dict[str, torch.Tensor]:
    """
    加载 .pth 格式的小点云数据
    
    要求：
        - 必须有 'coord'（点坐标）
        - 必须有 'gt_position'（真实位置）
        - 必须有 'feat' 或 'color'（颜色特征）
    
    Args:
        patch_path: .pth 文件路径
        verbose: 是否打印详细信息
    
    Returns:
        data: 包含点云数据的字典（统一转为 torch.Tensor）
    """
    # 加载数据（PyTorch 2.6+ 需要 weights_only=False）
    try:
        data = torch.load(patch_path, map_location='cpu', weights_only=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load {patch_path}: {e}")
    
    if verbose:
        print(f"\n   原始数据键: {list(data.keys())}")
    
    # 1. 检查必要的键
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
    
    # 2. 🔥 处理 feat（必须是 color）
    if 'feat' not in data:
        if 'color' in data:
            if verbose:
                print(f"   ✅ 'feat' 不存在，使用 'color' 作为 'feat'")
            data['feat'] = data['color']
        else:
            raise KeyError(
                f"Missing 'feat' or 'color' in {patch_path}\n"
                f"feat 必须来自颜色信息，不能用坐标代替！\n"
                f"Available keys: {list(data.keys())}"
            )
    
    # 🔥 3. 统一转换为 torch.Tensor（如果是 numpy.ndarray）
    for key in ['coord', 'feat', 'grid_coord', 'gt_position']:
        if key in data:
            value = data[key]
            if isinstance(value, np.ndarray):
                data[key] = torch.from_numpy(value)
            elif not isinstance(value, torch.Tensor):
                data[key] = torch.tensor(value)
    
    # 处理 category_id
    if 'category_id' in data:
        value = data['category_id']
        if isinstance(value, np.ndarray):
            data['category_id'] = torch.from_numpy(value)
        elif isinstance(value, (int, np.integer)):
            data['category_id'] = torch.tensor(value)
        elif not isinstance(value, torch.Tensor):
            data['category_id'] = torch.tensor(value)
    
    if verbose:
        print(f"   最终数据键: {list(data.keys())}")
        print(f"   数据类型:")
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                print(f"      {key}: shape={value.shape}, dtype={value.dtype}, type=Tensor")
            elif isinstance(value, np.ndarray):
                print(f"      {key}: shape={value.shape}, dtype={value.dtype}, type=ndarray")
            else:
                print(f"      {key}: type={type(value)}")
    
    return data


def visualize_prediction(
    patch_data: Dict[str, torch.Tensor],
    pred_position: np.ndarray,
    gt_position: np.ndarray,
    patch_name: str = "",
    window_title: str = "接触点预测结果"
):
    """
    在 Open3D 中可视化预测结果（只显示局部点云）
    
    Args:
        patch_data: 小点云数据（可能是 torch.Tensor 或 numpy.ndarray）
        pred_position: 预测的接触点位置 (3,) numpy
        gt_position: 真实的接触点位置 (3,) numpy
        patch_name: 样本名称
        window_title: 窗口标题
    """
    # 🔥 修复：统一转换为 numpy
    coord = patch_data['coord']
    if isinstance(coord, torch.Tensor):
        coord = coord.cpu().numpy()
    else:
        coord = np.array(coord)
    
    # 创建小点云
    patch_pcd = o3d.geometry.PointCloud()
    patch_pcd.points = o3d.utility.Vector3dVector(coord)
    patch_pcd.paint_uniform_color([0.7, 0.7, 0.7])  # 灰色
    
    # 计算点云范围
    pcd_min = coord.min(axis=0)
    pcd_max = coord.max(axis=0)
    pcd_center = (pcd_min + pcd_max) / 2
    pcd_size = np.linalg.norm(pcd_max - pcd_min)
    
    # 创建预测位置标记（红色球体）
    sphere_radius = pcd_size * 0.02
    pred_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
    pred_sphere.translate(pred_position)
    pred_sphere.paint_uniform_color([1, 0, 0])  # 红色
    
    # 创建 GT 位置标记（蓝色球体）
    gt_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
    gt_sphere.translate(gt_position)
    gt_sphere.paint_uniform_color([0, 0, 1])  # 蓝色
    
    # 创建连接线
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.array([pred_position, gt_position]))
    line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
    line_set.colors = o3d.utility.Vector3dVector([[1, 0.5, 0]])  # 橙色
    
    # 创建坐标系
    coord_size = pcd_size * 0.15
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=coord_size, 
        origin=pcd_center
    )
    
    # 计算误差
    error = np.linalg.norm(pred_position - gt_position)
    
    # 打印信息
    print(f"\n🎨 可视化说明:")
    print(f"   🟦 蓝色球体 = GT 接触点位置")
    print(f"   🟥 红色球体 = 预测接触点位置")
    print(f"   ⬜ 灰色点云 = 输入的局部点云")
    print(f"   🟧 橙色线段 = 预测误差连线")
    print(f"\n📊 预测结果 ({patch_name}):")
    print(f"   预测位置: [{pred_position[0]:.6f}, {pred_position[1]:.6f}, {pred_position[2]:.6f}]")
    print(f"   GT 位置:   [{gt_position[0]:.6f}, {gt_position[1]:.6f}, {gt_position[2]:.6f}]")
    print(f"   误差: {error:.6f} 米 = {error*1000:.2f} 毫米")
    
    # 可视化
    o3d.visualization.draw_geometries(
        [patch_pcd, pred_sphere, gt_sphere, line_set, coord_frame],
        window_name=window_title,
        width=1280,
        height=720,
        point_show_normal=False
    )

def visualize_with_complete_model(
    patch_data: Dict[str, torch.Tensor],
    pred_position: np.ndarray,
    gt_position: np.ndarray,
    complete_model_path: Path,
    patch_name: str = "",
    window_title: str = "接触点在完整模型上的位置"
):
    """
    在完整点云模型上可视化预测结果
    
    Args:
        patch_data: 小点云数据
        pred_position: 预测的接触点位置 (3,) numpy
        gt_position: 真实的接触点位置 (3,) numpy
        complete_model_path: 完整点云模型路径（.ply/.pcd）
        patch_name: 样本名称
        window_title: 窗口标题
    """
    geometries = []
    
    # 加载完整点云模型
    if complete_model_path.exists():
        print(f"📂 加载完整模型: {complete_model_path}")
        complete_pcd = o3d.io.read_point_cloud(str(complete_model_path))
        complete_pcd.paint_uniform_color([0.8, 0.8, 0.8])  # 浅灰色
        geometries.append(complete_pcd)
        
        # 计算点云尺寸
        points = np.asarray(complete_pcd.points)
        pcd_min = points.min(axis=0)
        pcd_max = points.max(axis=0)
        pcd_size = np.linalg.norm(pcd_max - pcd_min)
    else:
        print(f"⚠️  完整模型不存在: {complete_model_path}")
        # 🔥 修复：统一转换为 numpy
        coord = patch_data['coord']
        if isinstance(coord, torch.Tensor):
            coord = coord.cpu().numpy()
        else:
            coord = np.array(coord)
        
        patch_pcd = o3d.geometry.PointCloud()
        patch_pcd.points = o3d.utility.Vector3dVector(coord)
        patch_pcd.paint_uniform_color([0.7, 0.7, 0.7])
        geometries.append(patch_pcd)
        
        pcd_min = coord.min(axis=0)
        pcd_max = coord.max(axis=0)
        pcd_size = np.linalg.norm(pcd_max - pcd_min)
    
    # 创建预测位置标记（红色球体）
    sphere_radius = pcd_size * 0.015
    pred_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
    pred_sphere.translate(pred_position)
    pred_sphere.paint_uniform_color([1, 0, 0])  # 红色
    geometries.append(pred_sphere)
    
    # 创建 GT 位置标记（蓝色球体）
    gt_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
    gt_sphere.translate(gt_position)
    gt_sphere.paint_uniform_color([0, 0, 1])  # 蓝色
    geometries.append(gt_sphere)
    
    # 创建连接线
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.array([pred_position, gt_position]))
    line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
    line_set.colors = o3d.utility.Vector3dVector([[1, 0.5, 0]])  # 橙色
    geometries.append(line_set)
    
    # 创建坐标系
    coord_size = pcd_size * 0.08
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=coord_size, 
        origin=[0, 0, 0]
    )
    geometries.append(coord_frame)
    
    # 计算误差
    error = np.linalg.norm(pred_position - gt_position)
    
    # 打印信息
    print(f"\n🎨 完整模型可视化 ({patch_name}):")
    print(f"   🟦 蓝色球体 = GT 接触点")
    print(f"   🟥 红色球体 = 预测接触点")
    print(f"   ⬜ 灰色点云 = 完整模型")
    print(f"   误差: {error:.6f} 米 = {error*1000:.2f} 毫米")
    
    # 可视化
    o3d.visualization.draw_geometries(
        geometries,
        window_name=window_title,
        width=1280,
        height=720,
        point_show_normal=False
    )


def test_single_sample(
    matcher: PTv3ContactMatcher,
    patch_path: Path,
    complete_model_path: Optional[Path] = None,
    visualize: bool = True,
    save_result: bool = False
):
    """
    测试单个样本
    
    Args:
        matcher: PTv3ContactMatcher 实例
        patch_path: 局部点云文件路径 (.pth)
        complete_model_path: 完整点云模型路径（可选）
        visualize: 是否可视化
        save_result: 是否保存结果
    """
    print(f"\n{'='*70}")
    print(f"🧪 测试样本: {patch_path}")
    print(f"{'='*70}")
    
    # 加载数据（开启详细输出）
    patch_data = load_patch_data(patch_path, verbose=True)
    
    # 🔥 修复：统一转换为 numpy
    coord = patch_data['coord']
    if isinstance(coord, torch.Tensor):
        coord_np = coord.cpu().numpy()
    else:
        coord_np = np.array(coord)
    
    # 🔥 修复：gt_position 可能是 Tensor 或 ndarray
    gt_position = patch_data['gt_position']
    if isinstance(gt_position, torch.Tensor):
        gt_position = gt_position.cpu().numpy()
    else:
        gt_position = np.array(gt_position)
    
    patch_name = patch_data.get('name', patch_path.stem)
    category_id = patch_data.get('category_id', None)
    
    print(f"\n📂 局部点云:")
    print(f"   名称: {patch_name}")
    print(f"   点数: {coord.shape[0] if isinstance(coord, torch.Tensor) else len(coord)}")
    print(f"   GT 位置: {gt_position}")
    if category_id is not None:
        if isinstance(category_id, torch.Tensor):
            cat_id_value = category_id.item() if category_id.dim() == 0 else category_id[0].item()
        else:
            cat_id_value = int(category_id)
        print(f"   类别 ID: {cat_id_value}")
    
    # 预测
    print(f"\n🔮 正在预测...")
    try:
        pred_position = matcher.predict(patch_data)
    except Exception as e:
        print(f"❌ 预测失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # 计算误差
    error = np.linalg.norm(pred_position - gt_position)
    
    print(f"\n📊 预测结果:")
    print(f"   预测位置: [{pred_position[0]:.6f}, {pred_position[1]:.6f}, {pred_position[2]:.6f}]")
    print(f"   GT 位置:   [{gt_position[0]:.6f}, {gt_position[1]:.6f}, {gt_position[2]:.6f}]")
    print(f"   误差: {error:.6f} 米 = {error*1000:.2f} 毫米")
    
    # 可视化
    if visualize:
        # 可视化1: 只显示局部点云
        print(f"\n📺 窗口1: 局部点云 + 预测结果")
        visualize_prediction(
            patch_data,
            pred_position,
            gt_position,
            patch_name=patch_name,
            window_title=f"局部点云 - {patch_name}"
        )
        
        # 可视化2: 在完整模型上显示（如果提供）
        if complete_model_path:
            print(f"\n📺 窗口2: 完整模型 + 预测结果")
            visualize_with_complete_model(
                patch_data,
                pred_position,
                gt_position,
                complete_model_path,
                patch_name=patch_name,
                window_title=f"完整模型 - {patch_name}"
            )
    
    # 保存结果
    if save_result:
        result = {
            'patch_file': str(patch_path),
            'patch_name': patch_name,
            'pred_position': pred_position.tolist(),
            'gt_position': gt_position.tolist(),
            'error_meters': float(error),
            'error_mm': float(error * 1000)
        }
        
        result_path = Path("inference_results") / f"{patch_path.stem}_result.json"
        result_path.parent.mkdir(exist_ok=True)
        
        import json
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\n💾 结果已保存: {result_path}")
    
    print(f"{'='*70}\n")
    
    return error


def test_dataset(
    matcher: PTv3ContactMatcher,
    dataset_dir: Path,
    category: str = "Scissors",
    num_samples: int = 5,
    complete_model_path: Optional[Path] = None,
    visualize_samples: bool = False
):
    """
    测试数据集中的多个样本
    
    Args:
        matcher: PTv3ContactMatcher 实例
        dataset_dir: 数据集目录
        category: 类别名称
        num_samples: 测试样本数量
        complete_model_path: 完整点云模型路径
        visualize_samples: 是否可视化每个样本
    """
    category_dir = dataset_dir / category / "patches"
    
    if not category_dir.exists():
        print(f"❌ 目录不存在: {category_dir}")
        return
    
    # 获取所有 patch 文件
    patch_files = sorted(category_dir.glob("*.pth"))
    
    if not patch_files:
        print(f"❌ 目录中没有 .pth 文件: {category_dir}")
        return
    
    print(f"\n{'='*70}")
    print(f"📊 数据集批量测试")
    print(f"{'='*70}")
    print(f"   目录: {category_dir}")
    print(f"   总样本数: {len(patch_files)}")
    print(f"   测试样本数: {min(num_samples, len(patch_files))}")
    
    # 随机选择样本
    import random
    test_files = random.sample(patch_files, min(num_samples, len(patch_files)))
    
    errors = []
    
    for i, patch_path in enumerate(test_files):
        print(f"\n{'-'*70}")
        print(f"样本 {i+1}/{len(test_files)}: {patch_path.name}")
        print(f"{'-'*70}")
        
        # 加载数据
        patch_data = load_patch_data(patch_path)
        patch_name = patch_data.get('name', patch_path.stem)
        
        # 预测
        pred_position = matcher.predict(patch_data)
        
        # 🔥 修复：gt_position 可能是 Tensor 或 ndarray
        gt_position = patch_data['gt_position']
        if isinstance(gt_position, torch.Tensor):
            gt_position = gt_position.cpu().numpy()
        else:
            gt_position = np.array(gt_position)
        
        # 计算误差
        error = np.linalg.norm(pred_position - gt_position)
        errors.append(error)
        
        print(f"   名称: {patch_name}")
        print(f"   预测: [{pred_position[0]:.6f}, {pred_position[1]:.6f}, {pred_position[2]:.6f}]")
        print(f"   GT:   [{gt_position[0]:.6f}, {gt_position[1]:.6f}, {gt_position[2]:.6f}]")
        print(f"   误差: {error*1000:.2f} mm")
        
        # 可视化（可选）
        if visualize_samples:
            visualize_prediction(
                patch_data,
                pred_position,
                gt_position,
                patch_name=patch_name,
                window_title=f"样本 {i+1}/{len(test_files)}: {patch_name}"
            )
    
    # 统计
    errors = np.array(errors)
    print(f"\n{'='*70}")
    print(f"📈 统计结果")
    print(f"{'='*70}")
    print(f"   平均误差: {errors.mean()*1000:.2f} mm")
    print(f"   中位数误差: {np.median(errors)*1000:.2f} mm")
    print(f"   标准差: {errors.std()*1000:.2f} mm")
    print(f"   最大误差: {errors.max()*1000:.2f} mm")
    print(f"   最小误差: {errors.min()*1000:.2f} mm")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description='PTv3 Contact Position Regression 推理与可视化')
    parser.add_argument('--config', type=str, 
                        default='configs/s3dis/semseg-pt-v3m1-gelsight.py',
                        help='配置文件路径')
    parser.add_argument('--checkpoint', type=str,
                        default='exp/gelsight_test/model/model_best.pth',
                        help='模型权重路径')
    parser.add_argument('--patch', type=str,
                        help='单个局部点云文件路径 (.pth)')
    parser.add_argument('--dataset', type=str,
                        default='data/gelsight_dataset',
                        help='数据集根目录')
    parser.add_argument('--category', type=str, default='Scissors',
                        help='类别名称（Scissors/Cup/Avocado）')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='批量测试时的样本数量')
    parser.add_argument('--complete_model', type=str,
                        help='完整点云模型路径 (.ply/.pcd)')
    parser.add_argument('--no_vis', action='store_true',
                        help='不显示可视化')
    parser.add_argument('--vis_all', action='store_true',
                        help='批量测试时可视化每个样本（会很慢）')
    parser.add_argument('--save', action='store_true',
                        help='保存结果到 JSON')
    
    args = parser.parse_args()
    
    # 初始化推理器
    print(f"\n{'='*70}")
    print(f"🚀 初始化 PTv3 Contact Matcher")
    print(f"{'='*70}")
    
    matcher = PTv3ContactMatcher(
        config_path=args.config,
        checkpoint_path=args.checkpoint
    )
    
    # 完整模型路径
    complete_model_path = Path(args.complete_model) if args.complete_model else None
    
    if args.patch:
        # 测试单个样本
        test_single_sample(
            matcher,
            Path(args.patch),
            complete_model_path=complete_model_path,
            visualize=not args.no_vis,
            save_result=args.save
        )
    else:
        # 批量测试数据集
        test_dataset(
            matcher,
            Path(args.dataset),
            category=args.category,
            num_samples=args.num_samples,
            complete_model_path=complete_model_path,
            visualize_samples=args.vis_all
        )


if __name__ == "__main__":
    main()