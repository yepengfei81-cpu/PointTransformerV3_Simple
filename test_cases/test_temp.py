import torch
import torch.nn as nn
from pointcept.models.builder import MODELS, build_model
from pointcept.models.utils import offset2batch
from pointcept.models.point import Point


@MODELS.register_module()
class ContactPositionRegressor(nn.Module):
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
            self.parent_backbone = self.backbone  # share weights
        
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
        if isinstance(obj, torch.Tensor):
            return obj.to(device, non_blocking=True)
        elif isinstance(obj, dict):
            return {k: ContactPositionRegressor._to_device(v, device) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return type(obj)(ContactPositionRegressor._to_device(v, device) for v in obj)
        else:
            return obj
    
    def extract_point_features(self, point_dict, is_parent=False):
        backbone = self.parent_backbone if is_parent else self.backbone
        
        point = Point(point_dict)
        point = backbone(point)
        
        if isinstance(point, Point):
            while "pooling_parent" in point.keys():
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = point.feat[inverse]
                point = parent
            feat = point.feat  # (N, C)
        else:
            feat = point
        
        return feat  # (N, C)
    
    def point_level_cross_attention(
        self,
        local_point_feat,
        parent_point_feat,
        local_dict,
        parent_dict,
    ):
        local_offset = local_dict["offset"]
        parent_offset = parent_dict["offset"]
        batch_size = local_offset.shape[0]
        
        local_batch = offset2batch(local_offset)
        parent_batch = offset2batch(parent_offset)
        
        if self.use_position_encoding:
            local_coord = local_dict["coord"]  # (N_local, 3)
            parent_coord = parent_dict["coord"]  # (N_parent, 3)
            
            local_pos_emb = self.local_pos_encoder(local_coord)  # (N_local, C)
            parent_pos_emb = self.parent_pos_encoder(parent_coord)  # (N_parent, C)
            
            local_point_feat = local_point_feat + local_pos_emb
            parent_point_feat = parent_point_feat + parent_pos_emb
        
        fused_feats = []
        for i in range(batch_size):
            local_mask = (local_batch == i)
            parent_mask = (parent_batch == i)
            
            local_feat_i = local_point_feat[local_mask]  # (N_local_i, C)
            parent_feat_i = parent_point_feat[parent_mask]  # (N_parent_i, C)
            
            if local_feat_i.shape[0] == 0 or parent_feat_i.shape[0] == 0:
                fused_feats.append(torch.zeros(self.backbone_out_channels, device=local_feat_i.device))
                continue
            
            # Cross-Attention
            local_feat_i = local_feat_i.unsqueeze(0)  # (1, N_local_i, C)
            parent_feat_i = parent_feat_i.unsqueeze(0)  # (1, N_parent_i, C)
            
            attended_feat_i, _ = self.cross_attention(
                query=local_feat_i,
                key=parent_feat_i,
                value=parent_feat_i,
            )  # (1, N_local_i, C)
            
            attended_feat_i = attended_feat_i.squeeze(0)  # (N_local_i, C)
            attended_feat_i = self.fusion_proj(attended_feat_i)
            fused_feat_i = local_feat_i.squeeze(0) + attended_feat_i  # (N_local_i, C)
            
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
        device = next(self.parameters()).device
        input_dict = self._to_device(input_dict, device)
        
        local_dict = input_dict["local"]
        parent_dict = input_dict["parent"]
        
        local_point_feat = self.extract_point_features(local_dict, is_parent=False)  # (N_local, C)
        parent_point_feat = self.extract_point_features(parent_dict, is_parent=True)  # (N_parent, C)
        
        global_feat = self.point_level_cross_attention(
            local_point_feat=local_point_feat,
            parent_point_feat=parent_point_feat,
            local_dict=local_dict,
            parent_dict=parent_dict,
        )  # (batch_size, C)
        
        if self.use_category_condition and "category_id" in input_dict:
            category_id = input_dict["category_id"].long()
            category_emb = self.category_embedding(category_id)  # (batch_size, emb_dim)
            global_feat = torch.cat([global_feat, category_emb], dim=-1)
        
        # Regression Head
        position_pred_norm = self.regression_head(global_feat)  # (batch_size, 3)
        
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