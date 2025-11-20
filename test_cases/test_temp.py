import torch
import torch.nn as nn
from pointcept.models.builder import MODELS, build_model
from pointcept.models.utils import offset2batch
from pointcept.models.point import Point


@MODELS.register_module()
class ContactPositionRegressor(nn.Module):
    """
    Contact Position Regressor with Point-level Cross-Attention
    
    使用逐点 Cross-Attention 融合局部点云和父点云特征
    """
    def __init__(
        self,
        num_outputs=3,
        num_classes=3,
        use_category_condition=True,
        category_emb_dim=32,
        backbone_out_channels=512,
        backbone=None,
        freeze_backbone=False,
        pooling_type="attention",  # "max" | "avg" | "attention"
        parent_backbone=None,  # None = shared backbone
        cross_attn_num_heads=8,
        cross_attn_dropout=0.1,
        use_position_encoding=True,  # 是否使用位置编码
    ):
        super().__init__()
        
        # Backbone
        self.backbone = build_model(backbone)
        self.freeze_backbone = freeze_backbone
        
        if parent_backbone is not None:
            self.parent_backbone = build_model(parent_backbone)
        else:
            self.parent_backbone = self.backbone  # 共享权重
        
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            for p in self.parent_backbone.parameters():
                p.requires_grad = False
        
        # Loss functions
        self.criterion_smooth_l1 = nn.SmoothL1Loss()
        self.criterion_mse = nn.MSELoss()
        
        # Config
        self.use_category_condition = use_category_condition
        self.pooling_type = pooling_type
        self.use_position_encoding = use_position_encoding
        self.backbone_out_channels = backbone_out_channels
        
        # Pooling layers
        if pooling_type == "attention":
            self.attention_pool = nn.Sequential(
                nn.Linear(backbone_out_channels, backbone_out_channels // 4),
                nn.ReLU(inplace=True),
                nn.Linear(backbone_out_channels // 4, 1),
            )
        
        # Position encoding (optional)
        if use_position_encoding:
            self.local_pos_encoder = nn.Sequential(
                nn.Linear(3, backbone_out_channels // 4),
                nn.ReLU(inplace=True),
                nn.Linear(backbone_out_channels // 4, backbone_out_channels),
            )
            self.parent_pos_encoder = nn.Sequential(
                nn.Linear(3, backbone_out_channels // 4),
                nn.ReLU(inplace=True),
                nn.Linear(backbone_out_channels // 4, backbone_out_channels),
            )
        
        # Point-level Cross-Attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=backbone_out_channels,
            num_heads=cross_attn_num_heads,
            dropout=cross_attn_dropout,
            batch_first=True,
        )
        
        # Fusion projection with residual
        self.fusion_proj = nn.Sequential(
            nn.Linear(backbone_out_channels, backbone_out_channels),
            nn.LayerNorm(backbone_out_channels),
            nn.ReLU(inplace=True),
        )
        
        # Category embedding
        if use_category_condition:
            self.category_embedding = nn.Embedding(num_classes, category_emb_dim)
            regression_input_dim = backbone_out_channels + category_emb_dim
        else:
            regression_input_dim = backbone_out_channels
        
        # Regression head
        self.regression_head = nn.Sequential(
            nn.Linear(regression_input_dim, backbone_out_channels // 2),
            nn.LayerNorm(backbone_out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(backbone_out_channels // 2, backbone_out_channels // 4),
            nn.LayerNorm(backbone_out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(backbone_out_channels // 4, num_outputs),
        )
    
    @staticmethod
    def _to_device(obj, device):
        """递归地将数据转移到指定设备"""
        if isinstance(obj, torch.Tensor):
            return obj.to(device, non_blocking=True)
        elif isinstance(obj, dict):
            return {k: ContactPositionRegressor._to_device(v, device) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return type(obj)(ContactPositionRegressor._to_device(v, device) for v in obj)
        else:
            return obj
    
    def extract_point_features(self, point_dict, is_parent=False):
        """
        提取逐点特征（不池化）
        
        Args:
            point_dict: 点云字典
            is_parent: 是否是父点云
        
        Returns:
            point_feat: (N, C) - 逐点特征
        """
        backbone = self.parent_backbone if is_parent else self.backbone
        
        point = Point(point_dict)
        point = backbone(point)
        
        # 提取逐点特征
        if isinstance(point, Point):
            # 处理 pooling 逆映射
            while "pooling_parent" in point.keys():
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = point.feat[inverse]
                point = parent
            feat = point.feat  # (N, C)
        else:
            feat = point
        
        return feat  # (N, C)
    
    def pool_features(self, point_feat, offset):
        """
        将逐点特征池化为全局特征
        
        Args:
            point_feat: (N, C) - 逐点特征
            offset: (batch_size,) - batch offset
        
        Returns:
            global_feat: (batch_size, C) - 全局特征
        """
        batch = offset2batch(offset)
        batch_size = batch.max().item() + 1 if batch.numel() > 0 else 0
        global_feat = []
        
        for i in range(batch_size):
            mask = (batch == i)
            sample_feat = point_feat[mask]  # (N_i, C)
            
            if sample_feat.shape[0] == 0:
                # 空 batch，使用零特征
                sample_global = torch.zeros(point_feat.shape[1], device=point_feat.device)
            else:
                if self.pooling_type == "max":
                    sample_global = sample_feat.max(dim=0)[0]
                elif self.pooling_type == "avg":
                    sample_global = sample_feat.mean(dim=0)
                elif self.pooling_type == "attention":
                    attn_weights = self.attention_pool(sample_feat).softmax(dim=0)
                    sample_global = (sample_feat * attn_weights).sum(dim=0)
                else:
                    raise ValueError(f"Unknown pooling_type: {self.pooling_type}")
            
            global_feat.append(sample_global)
        
        return torch.stack(global_feat)  # (batch_size, C)
    
    def point_level_cross_attention(
        self,
        local_point_feat,
        parent_point_feat,
        local_dict,
        parent_dict,
    ):
        """
        逐点 Cross-Attention 融合
        
        局部点云的每个点都和父点云的所有点做 attention
        
        Args:
            local_point_feat: (N_local, C)
            parent_point_feat: (N_parent, C)
            local_dict: 局部点云字典（包含 offset, coord）
            parent_dict: 父点云字典
        
        Returns:
            global_feat: (batch_size, C) - 融合后的全局特征
        """
        local_offset = local_dict["offset"]
        parent_offset = parent_dict["offset"]
        batch_size = local_offset.shape[0]
        
        local_batch = offset2batch(local_offset)
        parent_batch = offset2batch(parent_offset)
        
        # 位置编码（可选）
        if self.use_position_encoding:
            local_coord = local_dict["coord"]  # (N_local, 3)
            parent_coord = parent_dict["coord"]  # (N_parent, 3)
            
            local_pos_emb = self.local_pos_encoder(local_coord)  # (N_local, C)
            parent_pos_emb = self.parent_pos_encoder(parent_coord)  # (N_parent, C)
            
            local_point_feat = local_point_feat + local_pos_emb
            parent_point_feat = parent_point_feat + parent_pos_emb
        
        # 逐 batch 处理
        fused_feats = []
        for i in range(batch_size):
            local_mask = (local_batch == i)
            parent_mask = (parent_batch == i)
            
            local_feat_i = local_point_feat[local_mask]  # (N_local_i, C)
            parent_feat_i = parent_point_feat[parent_mask]  # (N_parent_i, C)
            
            if local_feat_i.shape[0] == 0 or parent_feat_i.shape[0] == 0:
                # 空 batch，使用零特征
                fused_feats.append(torch.zeros(self.backbone_out_channels, device=local_feat_i.device))
                continue
            
            # Cross-Attention
            # Query: 局部点云的每个点
            # Key/Value: 父点云的每个点
            local_feat_i = local_feat_i.unsqueeze(0)  # (1, N_local_i, C)
            parent_feat_i = parent_feat_i.unsqueeze(0)  # (1, N_parent_i, C)
            
            attended_feat_i, _ = self.cross_attention(
                query=local_feat_i,
                key=parent_feat_i,
                value=parent_feat_i,
            )  # (1, N_local_i, C)
            
            attended_feat_i = attended_feat_i.squeeze(0)  # (N_local_i, C)
            
            # 投影 + 残差连接
            attended_feat_i = self.fusion_proj(attended_feat_i)
            fused_feat_i = local_feat_i.squeeze(0) + attended_feat_i  # (N_local_i, C)
            
            # 池化为全局特征
            if self.pooling_type == "max":
                global_feat_i = fused_feat_i.max(dim=0)[0]
            elif self.pooling_type == "avg":
                global_feat_i = fused_feat_i.mean(dim=0)
            elif self.pooling_type == "attention":
                attn_weights = self.attention_pool(fused_feat_i).softmax(dim=0)
                global_feat_i = (fused_feat_i * attn_weights).sum(dim=0)
            else:
                raise ValueError(f"Unknown pooling_type: {self.pooling_type}")
            
            fused_feats.append(global_feat_i)
        
        return torch.stack(fused_feats)  # (batch_size, C)
    
    def forward(self, input_dict, return_point=False):
        """
        Forward pass
        
        Args:
            input_dict: 包含 'local' 和 'parent' 的字典
            return_point: 是否返回中间特征（用于调试）
        
        Returns:
            return_dict: 包含 'loss' 或 'pred_position'
        """
        device = next(self.parameters()).device
        input_dict = self._to_device(input_dict, device)
        
        local_dict = input_dict["local"]
        parent_dict = input_dict["parent"]
        
        # 1. 提取逐点特征
        local_point_feat = self.extract_point_features(local_dict, is_parent=False)  # (N_local, C)
        parent_point_feat = self.extract_point_features(parent_dict, is_parent=True)  # (N_parent, C)
        
        # 2. 逐点 Cross-Attention 融合
        global_feat = self.point_level_cross_attention(
            local_point_feat=local_point_feat,
            parent_point_feat=parent_point_feat,
            local_dict=local_dict,
            parent_dict=parent_dict,
        )  # (batch_size, C)
        
        # 3. Category Embedding（可选）
        if self.use_category_condition and "category_id" in input_dict:
            category_id = input_dict["category_id"].long()
            category_emb = self.category_embedding(category_id)  # (batch_size, emb_dim)
            global_feat = torch.cat([global_feat, category_emb], dim=-1)
        
        # 4. Regression Head
        position_pred_norm = self.regression_head(global_feat)  # (batch_size, 3)
        
        # 5. 构造返回字典
        return_dict = {}
        
        if return_point:
            return_dict["local_point_feat"] = local_point_feat
            return_dict["parent_point_feat"] = parent_point_feat
            return_dict["global_feat"] = global_feat
        
        if self.training or "gt_position" in input_dict:
            gt_position_norm = input_dict["gt_position"]
            loss_smooth_l1 = self.criterion_smooth_l1(position_pred_norm, gt_position_norm)
            loss_mse = self.criterion_mse(position_pred_norm, gt_position_norm)
            total_loss = loss_smooth_l1 * 1.0 + loss_mse * 0.5
            
            return_dict["loss"] = total_loss
            if not self.training:
                return_dict["pred_position"] = position_pred_norm
        else:
            return_dict["pred_position"] = position_pred_norm
        
        return return_dict