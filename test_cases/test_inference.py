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

def get_parent_model_from_sample_id(sample_id: int, samples_per_bigpcd: int = 50):
    """
    根据样本 ID 计算对应的父点云 ID
    
    Args:
        sample_id: 样本编号（从 patch 文件名提取）
        samples_per_bigpcd: 每个大点云生成的样本数
    
    Returns:
        parent_id: 父点云 ID（例如 "001", "002"）
    
    Examples:
        >>> get_parent_model_from_sample_id(44, 50)
        '001'  # 0-49 → bigpointcloud_001
        >>> get_parent_model_from_sample_id(99, 50)
        '002'  # 50-99 → bigpointcloud_002
        >>> get_parent_model_from_sample_id(150, 50)
        '004'  # 150-199 → bigpointcloud_004
    """
    bigpcd_id = (sample_id // samples_per_bigpcd) + 1
    return f"{bigpcd_id:03d}"


def extract_sample_id_from_patch_name(patch_name: str):
    """
    从 patch 文件名提取样本 ID
    
    Examples:
        >>> extract_sample_id_from_patch_name("patch_000044.pth")
        44
        >>> extract_sample_id_from_patch_name("patch_000198.pth")
        198
    """
    # 移除路径和扩展名
    patch_name = Path(patch_name).stem
    
    # 提取数字部分
    if patch_name.startswith('patch_'):
        try:
            return int(patch_name.split('_')[1])
        except:
            pass
    
    # 尝试从末尾提取数字
    try:
        return int(patch_name.split('_')[-1])
    except:
        return None
        
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
    patch_data: Dict,
    pred_position: np.ndarray,
    gt_position: np.ndarray,
    complete_model_path: Path,
    patch_name: str = "",
    window_title: str = "接触点预测结果",
    save_dir: Path = None,
    show_window: bool = False
):
    """
    在完整的父点云上可视化预测结果
    
    Args:
        patch_data: 小点云数据
        pred_position: 预测的接触点位置 (3,) numpy
        gt_position: 真实的接触点位置 (3,) numpy
        complete_model_path: 完整点云模型路径（.ply/.pcd）
        patch_name: 样本名称
        window_title: 窗口标题
        save_dir: 保存目录
        show_window: 是否显示窗口（远程服务器设为 False）
    """
    geometries = []
    
    # 1. 加载完整的父点云
    if not complete_model_path.exists():
        raise FileNotFoundError(f"完整点云模型不存在: {complete_model_path}")
    
    print(f"📂 加载完整父点云: {complete_model_path}")
    complete_pcd = o3d.io.read_point_cloud(str(complete_model_path))
    complete_pcd.paint_uniform_color([0.85, 0.85, 0.85])  # 浅灰色
    geometries.append(complete_pcd)
    
    # 计算完整点云尺寸
    complete_points = np.asarray(complete_pcd.points)
    pcd_min = complete_points.min(axis=0)
    pcd_max = complete_points.max(axis=0)
    pcd_size = np.linalg.norm(pcd_max - pcd_min)
    pcd_center = (pcd_min + pcd_max) / 2
    
    # 2. 高亮显示局部点云区域
    coord = patch_data['coord']
    if isinstance(coord, torch.Tensor):
        coord = coord.cpu().numpy()
    else:
        coord = np.array(coord)
    
    patch_pcd = o3d.geometry.PointCloud()
    patch_pcd.points = o3d.utility.Vector3dVector(coord)
    patch_pcd.paint_uniform_color([1.0, 0.65, 0.0])  # 橙色高亮
    geometries.append(patch_pcd)
    
    # 3. 创建 GT 位置标记（蓝色球体）
    sphere_radius = pcd_size * 0.01
    gt_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
    gt_sphere.translate(gt_position)
    gt_sphere.paint_uniform_color([0, 0, 1])  # 蓝色
    gt_sphere.compute_vertex_normals()
    geometries.append(gt_sphere)
    
    # 4. 创建预测位置标记（红色球体）
    pred_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
    pred_sphere.translate(pred_position)
    pred_sphere.paint_uniform_color([1, 0, 0])  # 红色
    pred_sphere.compute_vertex_normals()
    geometries.append(pred_sphere)
    
    # 5. 创建连接线
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.array([pred_position, gt_position]))
    line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
    line_set.colors = o3d.utility.Vector3dVector([[1, 1, 0]])  # 黄色
    geometries.append(line_set)
    
    # 6. 创建坐标系
    coord_size = pcd_size * 0.05
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=coord_size, 
        origin=pcd_center
    )
    geometries.append(coord_frame)
    
    # 计算误差
    error = np.linalg.norm(pred_position - gt_position)
    
    # 打印信息
    print(f"\n{'='*70}")
    print(f"🎨 可视化说明 ({patch_name}):")
    print(f"{'='*70}")
    print(f"   🟦 蓝色球体     = GT 接触点位置（真实位置）")
    print(f"   🟥 红色球体     = 预测接触点位置（模型输出）")
    print(f"   ⬜ 灰色点云     = 完整的父点云模型")
    print(f"   🟠 橙色点云     = 输入的局部点云（高亮显示）")
    print(f"   🟨 黄色线段     = 预测误差连线")
    print(f"\n📊 预测结果:")
    print(f"   预测位置: [{pred_position[0]:.6f}, {pred_position[1]:.6f}, {pred_position[2]:.6f}]")
    print(f"   GT 位置:   [{gt_position[0]:.6f}, {gt_position[1]:.6f}, {gt_position[2]:.6f}]")
    print(f"   误差: {error:.6f} 米 = {error*1000:.2f} 毫米")
    print(f"{'='*70}\n")
    
    # 🔥 保存可视化结果
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 🔥 方法 1: 使用自定义回调保存图片（适合无头服务器）
        print("📸 正在生成可视化图片...")
        
        img_path = save_dir / f"{patch_name}_visualization.png"
        
        def capture_image(vis):
            """回调函数：渲染后保存图片"""
            # 设置相机参数
            ctr = vis.get_view_control()
            if ctr is not None:
                # 设置视角
                ctr.set_front([0.5, 0.5, 0.5])
                ctr.set_lookat(pcd_center.tolist())
                ctr.set_up([0, 0, 1])
                ctr.set_zoom(0.8)
            
            # 保存图片
            vis.capture_screen_image(str(img_path), do_render=True)
            return False  # 返回 False 关闭窗口
        
        try:
            # 尝试使用可视化窗口保存
            o3d.visualization.draw_geometries_with_animation_callback(
                geometries,
                capture_image,
                window_name=window_title,
                width=1920,
                height=1080
            )
            print(f"✅ 图片已保存: {img_path}")
        except Exception as e:
            print(f"⚠️  离屏渲染失败: {e}")
            print("   使用备用方案：直接保存点云...")
        
        # 🔥 方法 2: 保存为 PLY 文件（备用方案，总是可用）
        print("💾 保存为 PLY 文件...")
        
        # 合并所有几何体到一个点云
        combined_pcd = o3d.geometry.PointCloud()
        
        # 完整点云（灰色）
        combined_pcd += complete_pcd
        
        # 局部点云（橙色）
        combined_pcd += patch_pcd
        
        # GT 球体（蓝色）- 采样为点云
        gt_sphere_pcd = gt_sphere.sample_points_uniformly(number_of_points=1000)
        combined_pcd += gt_sphere_pcd
        
        # 预测球体（红色）- 采样为点云
        pred_sphere_pcd = pred_sphere.sample_points_uniformly(number_of_points=1000)
        combined_pcd += pred_sphere_pcd
        
        ply_path = save_dir / f"{patch_name}_visualization.ply"
        o3d.io.write_point_cloud(str(ply_path), combined_pcd)
        print(f"✅ PLY 文件已保存: {ply_path}")
        print(f"   可以用 MeshLab/CloudCompare 打开查看")
        
        # 保存预测结果到 JSON
        result = {
            'patch_name': patch_name,
            'pred_position': pred_position.tolist(),
            'gt_position': gt_position.tolist(),
            'error_meters': float(error),
            'error_mm': float(error * 1000),
            'complete_model': str(complete_model_path),
        }
        json_path = save_dir / f"{patch_name}_result.json"
        import json
        with open(json_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"✅ 结果已保存: {json_path}")
        
        # 🔥 方法 3: 使用 matplotlib 生成 2D 投影图（备用方案）
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(15, 5))
            
            # 三个视角
            views = [
                (0, 90, "Top View (XY)"),      # 俯视图
                (0, 0, "Front View (XZ)"),     # 正视图
                (90, 90, "Side View (YZ)")     # 侧视图
            ]
            
            for i, (elev, azim, title) in enumerate(views):
                ax = fig.add_subplot(1, 3, i+1, projection='3d')
                
                # 绘制完整点云（灰色，采样以减少点数）
                sample_idx = np.random.choice(len(complete_points), 
                                             min(5000, len(complete_points)), 
                                             replace=False)
                ax.scatter(complete_points[sample_idx, 0], 
                          complete_points[sample_idx, 1], 
                          complete_points[sample_idx, 2],
                          c='gray', s=0.1, alpha=0.3, label='Complete Model')
                
                # 绘制局部点云（橙色）
                ax.scatter(coord[:, 0], coord[:, 1], coord[:, 2],
                          c='orange', s=5, alpha=0.8, label='Local Patch')
                
                # 绘制 GT 位置（蓝色大点）
                ax.scatter(gt_position[0], gt_position[1], gt_position[2],
                          c='blue', s=200, marker='o', label='GT Position', 
                          edgecolors='black', linewidths=2)
                
                # 绘制预测位置（红色大点）
                ax.scatter(pred_position[0], pred_position[1], pred_position[2],
                          c='red', s=200, marker='o', label='Predicted Position',
                          edgecolors='black', linewidths=2)
                
                # 绘制连接线
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
            
            # 保存 matplotlib 图片
            plt_path = save_dir / f"{patch_name}_matplotlib.png"
            plt.savefig(plt_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Matplotlib 图片已保存: {plt_path}")
        except Exception as e:
            print(f"⚠️  Matplotlib 渲染失败: {e}")
    
    # 显示窗口（可选）
    if show_window:
        try:
            o3d.visualization.draw_geometries(
                geometries,
                window_name=window_title,
                width=1920,
                height=1080,
                point_show_normal=False
            )
        except Exception as e:
            print(f"⚠️  无法显示窗口: {e}")

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
    complete_model_path: Path,
    visualize: bool = True,
    save_result: bool = True,  # 🔥 默认保存
    save_dir: Path = Path("inference_results"),  # 🔥 保存目录
    show_window: bool = False  # 🔥 远程服务器不显示窗口
):
    """
    测试单个样本
    
    Args:
        matcher: PTv3ContactMatcher 实例
        patch_path: 局部点云文件路径 (.pth)
        complete_model_path: 完整点云模型路径（必需）
        visualize: 是否生成可视化
        save_result: 是否保存结果
        save_dir: 保存目录
        show_window: 是否显示窗口（远程服务器设为 False）
    """
    print(f"\n{'='*70}")
    print(f"🧪 测试样本: {patch_path}")
    print(f"{'='*70}")
    
    # 检查完整模型是否存在
    if not complete_model_path.exists():
        raise FileNotFoundError(f"完整点云模型不存在: {complete_model_path}")
    
    # 加载数据
    patch_data = load_patch_data(patch_path, verbose=True)
    
    # 统一转换为 numpy
    coord = patch_data['coord']
    if isinstance(coord, torch.Tensor):
        coord_np = coord.cpu().numpy()
    else:
        coord_np = np.array(coord)
    
    gt_position = patch_data['gt_position']
    if isinstance(gt_position, torch.Tensor):
        gt_position = gt_position.cpu().numpy()
    else:
        gt_position = np.array(gt_position)
    
    patch_name = patch_data.get('name', patch_path.stem)
    category_id = patch_data.get('category_id', None)
    
    print(f"\n📂 局部点云:")
    print(f"   名称: {patch_name}")
    print(f"   点数: {len(coord_np)}")
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
    
    # 可视化并保存
    if visualize:
        print(f"\n📺 生成可视化结果...")
        visualize_prediction(
            patch_data,
            pred_position,
            gt_position,
            complete_model_path,
            patch_name=patch_name,
            window_title=f"接触点预测 - {patch_name}",
            save_dir=save_dir if save_result else None,
            show_window=show_window
        )
    
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


def test_all_patches(
    matcher: PTv3ContactMatcher,
    dataset_dir: Path,
    category: str = "Scissors",
    save_dir: Path = Path("inference_results"),
    visualize_best_worst_median: bool = True,
):
    """
    测试数据集中的所有样本，并生成统计报告
    """
    print(f"\n{'='*80}")
    print(f"🚀 批量测试所有点云")
    print(f"{'='*80}")
    
    # 1. 查找所有 patch 文件
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
    
    # 2. 查找所有完整点云文件
    complete_model_dir = dataset_dir / category
    complete_models = {}
    
    for ply_file in sorted(complete_model_dir.glob("bigpointcloud_*.ply")):
        model_id = ply_file.stem.split('_')[-1]
        complete_models[model_id] = ply_file
    
    num_bigpcds = len(complete_models)
    print(f"📂 找到 {num_bigpcds} 个大点云")
    if complete_models:
        print(f"   模型 ID: {list(complete_models.keys())}")
    
    # 🔥 3. 自动计算每个大点云对应的小点云数量
    if num_bigpcds == 0:
        print(f"❌ 没有找到大点云文件")
        return
    
    samples_per_bigpcd = len(patch_files) // num_bigpcds
    
    print(f"\n💡 自动推断:")
    print(f"   总小点云数: {len(patch_files)}")
    print(f"   总大点云数: {num_bigpcds}")
    print(f"   每个大点云对应: {samples_per_bigpcd} 个小点云")
    
    # 验证是否整除
    if len(patch_files) % num_bigpcds != 0:
        print(f"   ⚠️  警告：{len(patch_files)} 不能被 {num_bigpcds} 整除")
        print(f"   可能有部分大点云的小点云数量不同")
    
    # 4. 打印映射关系
    print(f"\n📋 推断的映射关系:")
    for i, (model_id, model_path) in enumerate(sorted(complete_models.items())):
        start_idx = i * samples_per_bigpcd
        end_idx = start_idx + samples_per_bigpcd - 1
        print(f"   bigpointcloud_{model_id}.ply → patch_{start_idx:06d} ~ patch_{end_idx:06d}")
    
    # 5. 创建保存目录
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    batch_dir = save_dir / f"batch_{category}_{len(patch_files)}_samples"
    batch_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 结果保存目录: {batch_dir}")
    
    # 6. 逐个测试
    results = []
    failed_samples = []
    
    print(f"\n{'='*80}")
    print(f"🔮 开始批量推理...")
    print(f"{'='*80}\n")
    
    for i, patch_path in enumerate(patch_files):
        print(f"[{i+1}/{len(patch_files)}] 处理: {patch_path.name}")
        
        try:
            # 加载数据
            patch_data = load_patch_data(patch_path, verbose=False)
            patch_name = patch_data.get('name', patch_path.stem)
            
            # 🔥 方法 1: 从数据中读取（最准确）
            complete_model_path = None
            parent_id = None
            
            if 'bigpcd_id' in patch_data:
                bigpcd_id = patch_data['bigpcd_id']
                if isinstance(bigpcd_id, torch.Tensor):
                    bigpcd_id = bigpcd_id.item()
                parent_id = f"{int(bigpcd_id):03d}"
                complete_model_path = complete_models.get(parent_id)
                print(f"   📂 从数据读取: bigpcd_id={bigpcd_id} → bigpointcloud_{parent_id}.ply")
            
            # 🔥 方法 2: 从文件名推断（使用样本 ID）
            else:
                sample_id = extract_sample_id_from_patch_name(patch_path.name)
                if sample_id is not None:
                    # 🔥 修复：使用实际计算的 samples_per_bigpcd
                    parent_id = get_parent_model_from_sample_id(sample_id, samples_per_bigpcd)
                    complete_model_path = complete_models.get(parent_id)
                    print(f"   📂 从样本 ID 推断: sample_id={sample_id} → parent_id={parent_id}")
            
            # 🔥 方法 3: 使用第一个（备用）
            if complete_model_path is None or not complete_model_path.exists():
                if complete_models:
                    complete_model_path = list(complete_models.values())[0]
                    parent_id = complete_model_path.stem.split('_')[-1]
                    print(f"   ⚠️  使用默认: {complete_model_path.name}")
                else:
                    print(f"   ❌ 找不到完整点云模型，跳过")
                    failed_samples.append({
                        'patch_name': patch_name,
                        'reason': 'No complete model found'
                    })
                    continue
            
            # 预测
            pred_position = matcher.predict(patch_data)
            
            # GT 位置
            gt_position = patch_data['gt_position']
            if isinstance(gt_position, torch.Tensor):
                gt_position = gt_position.cpu().numpy()
            else:
                gt_position = np.array(gt_position)
            
            # 计算误差
            error = np.linalg.norm(pred_position - gt_position)
            
            # 保存结果
            result = {
                'index': i,
                'patch_name': patch_name,
                'patch_path': str(patch_path),
                'complete_model_path': str(complete_model_path),
                'complete_model_name': complete_model_path.name,
                'parent_id': parent_id,
                'pred_position': pred_position.tolist(),
                'gt_position': gt_position.tolist(),
                'error_meters': float(error),
                'error_mm': float(error * 1000),
                'patch_data': patch_data,
            }
            results.append(result)
            
            print(f"   ✅ 误差: {error*1000:.2f} mm | 父点云: bigpointcloud_{parent_id}.ply")
            
        except Exception as e:
            print(f"   ❌ 失败: {e}")
            import traceback
            traceback.print_exc()
            failed_samples.append({
                'patch_name': patch_path.name,
                'reason': str(e)
            })
    
    # 5. 统计分析
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
    print(f"  25% 分位: {np.percentile(errors, 25):.2f} mm")
    print(f"  75% 分位: {np.percentile(errors, 75):.2f} mm")
    
    # 6. 找出最好、最差、中位数样本
    best_idx = np.argmin(errors)
    worst_idx = np.argmax(errors)
    median_idx = np.argmin(np.abs(errors - np.median(errors)))
    
    best_sample = results[best_idx]
    worst_sample = results[worst_idx]
    median_sample = results[median_idx]
    
    print(f"\n🏆 最佳样本 (最小误差):")
    print(f"  名称: {best_sample['patch_name']}")
    print(f"  误差: {best_sample['error_mm']:.2f} mm")
    
    print(f"\n📉 最差样本 (最大误差):")
    print(f"  名称: {worst_sample['patch_name']}")
    print(f"  误差: {worst_sample['error_mm']:.2f} mm")
    
    print(f"\n📊 中位数样本:")
    print(f"  名称: {median_sample['patch_name']}")
    print(f"  误差: {median_sample['error_mm']:.2f} mm")
    
    # 7. 保存统计结果到 JSON
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
            'percentile_25_mm': float(np.percentile(errors, 25)),
            'percentile_75_mm': float(np.percentile(errors, 75)),
        },
        'best_sample': {
            'name': best_sample['patch_name'],
            'error_mm': best_sample['error_mm'],
            'pred_position': best_sample['pred_position'],
            'gt_position': best_sample['gt_position'],
        },
        'worst_sample': {
            'name': worst_sample['patch_name'],
            'error_mm': worst_sample['error_mm'],
            'pred_position': worst_sample['pred_position'],
            'gt_position': worst_sample['gt_position'],
        },
        'median_sample': {
            'name': median_sample['patch_name'],
            'error_mm': median_sample['error_mm'],
            'pred_position': median_sample['pred_position'],
            'gt_position': median_sample['gt_position'],
        },
        'failed_samples': failed_samples,
    }
    
    summary_path = batch_dir / "summary.json"
    import json
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n✅ 统计结果已保存: {summary_path}")
    
    # 8. 保存所有结果到 CSV
    import csv
    csv_path = batch_dir / "all_results.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'index', 'patch_name', 'parent_id', 'complete_model_name',  # 🔥 添加 parent_id
            'pred_x', 'pred_y', 'pred_z',
            'gt_x', 'gt_y', 'gt_z',
            'error_mm'
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({
                'index': r['index'],
                'patch_name': r['patch_name'],
                'parent_id': r['parent_id'],  # 🔥 添加这一列
                'complete_model_name': r['complete_model_name'],
                'pred_x': r['pred_position'][0],
                'pred_y': r['pred_position'][1],
                'pred_z': r['pred_position'][2],
                'gt_x': r['gt_position'][0],
                'gt_y': r['gt_position'][1],
                'gt_z': r['gt_position'][2],
                'error_mm': r['error_mm'],
            })
    print(f"✅ 详细结果已保存: {csv_path}")
    
    # 9. 生成误差分布图
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 9.1 误差直方图
        ax = axes[0, 0]
        ax.hist(errors, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        ax.axvline(errors.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {errors.mean():.2f} mm')
        ax.axvline(np.median(errors), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(errors):.2f} mm')
        ax.set_xlabel('Error (mm)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Error Distribution', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 9.2 误差箱线图
        ax = axes[0, 1]
        bp = ax.boxplot(errors, vert=True, patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        ax.set_ylabel('Error (mm)', fontsize=12)
        ax.set_title('Error Boxplot', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # 9.3 误差随样本变化
        ax = axes[1, 0]
        ax.plot(range(len(errors)), errors, 'o-', markersize=2, linewidth=0.5, alpha=0.6)
        ax.axhline(errors.mean(), color='red', linestyle='--', linewidth=2, label='Mean')
        ax.axhline(np.median(errors), color='green', linestyle='--', linewidth=2, label='Median')
        ax.set_xlabel('Sample Index', fontsize=12)
        ax.set_ylabel('Error (mm)', fontsize=12)
        ax.set_title('Error vs Sample Index', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 9.4 累积分布函数
        ax = axes[1, 1]
        sorted_errors = np.sort(errors)
        cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        ax.plot(sorted_errors, cumulative, linewidth=2)
        ax.axvline(errors.mean(), color='red', linestyle='--', linewidth=2, label='Mean')
        ax.axvline(np.median(errors), color='green', linestyle='--', linewidth=2, label='Median')
        ax.set_xlabel('Error (mm)', fontsize=12)
        ax.set_ylabel('Cumulative Probability', fontsize=12)
        ax.set_title('Cumulative Distribution Function', fontsize=14)
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
    
    # 10. 可视化最好、最差、中位数样本
    if visualize_best_worst_median:
        print(f"\n{'='*80}")
        print(f"🎨 生成代表性样本的可视化")
        print(f"{'='*80}\n")
        
        samples_to_visualize = [
            (best_sample, "best", "🏆 最佳样本"),
            (median_sample, "median", "📊 中位数样本"),
            (worst_sample, "worst", "📉 最差样本"),
        ]
        
        for sample, label, title in samples_to_visualize:
            print(f"\n{title}: {sample['patch_name']} (误差: {sample['error_mm']:.2f} mm)")
            
            try:
                visualize_prediction(
                    sample['patch_data'],
                    np.array(sample['pred_position']),
                    np.array(sample['gt_position']),
                    Path(sample['complete_model_path']),
                    patch_name=f"{label}_{sample['patch_name']}",
                    window_title=f"{title} - {sample['patch_name']}",
                    save_dir=batch_dir,
                    show_window=False
                )
            except Exception as e:
                print(f"   ⚠️  可视化失败: {e}")
    
    print(f"\n{'='*80}")
    print(f"✅ 批量测试完成！")
    print(f"{'='*80}")
    print(f"📂 所有结果保存在: {batch_dir}")
    print(f"\n生成的文件:")
    print(f"  - summary.json           : 统计摘要")
    print(f"  - all_results.csv        : 所有样本的详细结果")
    print(f"  - error_analysis.png     : 误差分析图")
    print(f"  - best_*                 : 最佳样本的可视化")
    print(f"  - median_*               : 中位数样本的可视化")
    print(f"  - worst_*                : 最差样本的可视化")
    print(f"{'='*80}\n")
    
    return results, summary

def main():
    parser = argparse.ArgumentParser(description='PTv3 Contact Position Regression 推理与可视化')
    parser.add_argument('--config', type=str, 
                        default='configs/s3dis/semseg-pt-v3m1-gelsight.py',
                        help='配置文件路径')
    parser.add_argument('--checkpoint', type=str,
                        default='exp/gelsight_test/model/model_best.pth',
                        help='模型权重路径')
    
    # 🔥 两种模式：单个样本 或 批量测试
    parser.add_argument('--mode', type=str, 
                        choices=['single', 'batch'], 
                        default='single',
                        help='运行模式：single（单个样本）或 batch（批量测试）')
    
    # 单个样本模式参数
    parser.add_argument('--patch', type=str,
                        help='单个局部点云文件路径 (.pth)')
    parser.add_argument('--complete_model', type=str,
                        help='完整点云模型路径 (.ply/.pcd)')
    
    # 批量测试模式参数
    parser.add_argument('--dataset_dir', type=str,
                        default='../../touch_processed_data',
                        help='数据集根目录')
    parser.add_argument('--category', type=str,
                        default='Scissors',
                        choices=['Scissors', 'Cup', 'Avocado'],
                        help='物体类别')
    
    # 通用参数
    parser.add_argument('--save_dir', type=str,
                        default='inference_results',
                        help='保存结果的目录')
    parser.add_argument('--no_vis', action='store_true',
                        help='不生成可视化')
    parser.add_argument('--show_window', action='store_true',
                        help='显示窗口（本地有显示器时使用）')
    
    args = parser.parse_args()
    
    # 初始化推理器
    print(f"\n{'='*80}")
    print(f"🚀 初始化 PTv3 Contact Matcher")
    print(f"{'='*80}")
    
    matcher = PTv3ContactMatcher(
        config_path=args.config,
        checkpoint_path=args.checkpoint
    )
    
    if args.mode == 'single':
        # 单个样本测试
        if not args.patch or not args.complete_model:
            parser.error("单个样本模式需要 --patch 和 --complete_model 参数")
        
        test_single_sample(
            matcher,
            Path(args.patch),
            complete_model_path=Path(args.complete_model),
            visualize=not args.no_vis,
            save_result=True,
            save_dir=Path(args.save_dir),
            show_window=args.show_window
        )
    
    elif args.mode == 'batch':
        # 批量测试
        test_all_patches(
            matcher,
            dataset_dir=Path(args.dataset_dir),
            category=args.category,
            save_dir=Path(args.save_dir),
            visualize_best_worst_median=not args.no_vis
        )


if __name__ == "__main__":
    main()