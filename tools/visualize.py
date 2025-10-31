import numpy as np

# # 文件路径
coord_file = "/root/autodl-tmp/processed_data/Area_5/WC_1/coord.npy"
pred_file = "/root/autodl-tmp/Pointcept/exp/default/result/Area_5-WC_1_pred.npy"

# 加载数据
coords = np.load(coord_file)
pred_labels = np.load(pred_file)

# 检查点数是否一致
print(f"原始点云点数: {coords.shape[0]}")
print(f"预测结果点数: {pred_labels.shape[0]}")

import plotly.express as px
import plotly.io as pio

def visualize(coords, labels, save_path=None):
    fig = px.scatter_3d(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        color=labels,
        color_continuous_scale="Viridis",
        opacity=0.8
    )
    if save_path:
        pio.write_html(fig, save_path)
        print(f"Interactive visualization saved to {save_path}")
    else:
        fig.show()

# 可视化并保存
save_path = "/root/autodl-tmp/Pointcept/exp/default/visualizations/Area_5-WC_1_visualization.html"
visualize(coords, pred_labels, save_path=save_path)