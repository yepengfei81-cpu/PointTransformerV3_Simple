import torch
import torch.nn as nn
import torch_scatter
import torch_cluster
from peft import LoraConfig, get_peft_model
from collections import OrderedDict

from pointcept.models.losses import build_criteria
from pointcept.models.utils.structure import Point
from pointcept.models.utils import offset2batch, batch2offset
from .builder import MODELS, build_model


@MODELS.register_module()
class DefaultSegmentor(nn.Module):
    def __init__(self, backbone=None, criteria=None):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        if "condition" in input_dict.keys():
            # PPT (https://arxiv.org/abs/2308.09718)
            # currently, only support one batch one condition
            input_dict["condition"] = input_dict["condition"][0]
        seg_logits = self.backbone(input_dict)
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss)
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)


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
        criteria=None,
        freeze_backbone=False,
        pooling_type="attention",  # "max" | "avg" | "attention"
        parent_backbone=None,
        fusion_type="concat",  # "concat" | "cross_attention"
        use_parent_cloud=True,
    ):
        super().__init__()
        
        self.backbone = build_model(backbone)
        self.freeze_backbone = freeze_backbone
        # self.criteria = build_criteria(criteria)
        self.criterion_smooth_l1 = nn.SmoothL1Loss()
        self.criterion_mse = nn.MSELoss()
        self.use_category_condition = use_category_condition
        self.use_parent_cloud = use_parent_cloud
        self.fusion_type = fusion_type           
        # construct parent point cloud backbone 
        if self.use_parent_cloud:
            if parent_backbone is not None:
                self.parent_backbone = build_model(parent_backbone)
                if freeze_backbone:
                    for p in self.parent_backbone.parameters():
                        p.requires_grad = False
            else:
                self.parent_backbone = self.backbone  # share weights
    
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        if use_parent_cloud and fusion_type == "concat":
            fusion_dim = backbone_out_channels * 2
        elif use_parent_cloud and fusion_type == "cross_attention":
            fusion_dim = backbone_out_channels
        else:
            fusion_dim = backbone_out_channels

        if use_category_condition:
            self.category_embedding = nn.Embedding(num_classes, category_emb_dim)
            regression_input_dim = fusion_dim + category_emb_dim
        else:
            regression_input_dim = fusion_dim
        
        # Pooling
        self.pooling_type = pooling_type
        if pooling_type == "attention":
            self.attention_pool = nn.Sequential(
                nn.Linear(backbone_out_channels, backbone_out_channels // 4),
                nn.ReLU(inplace=True),
                nn.Linear(backbone_out_channels // 4, 1),
            )
            if self.use_parent_cloud:
                self.parent_attention_pool = nn.Sequential(
                    nn.Linear(backbone_out_channels, backbone_out_channels // 4),
                    nn.ReLU(inplace=True),
                    nn.Linear(backbone_out_channels // 4, 1),
                )            

        # Cross-attention fusion
        if self.use_parent_cloud and self.fusion_type == "cross_attention":
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=backbone_out_channels,
                num_heads=8,
                batch_first=True,
            )        

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
            # nn.Sigmoid()
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
        
    # Modular feature extraction
    def extract_features(self, point_dict, is_parent=False):
        if is_parent and self.use_parent_cloud:
            backbone = self.parent_backbone
            attention_pool = self.parent_attention_pool if self.pooling_type == "attention" else None
        else:
            backbone = self.backbone
            attention_pool = self.attention_pool if self.pooling_type == "attention" else None
        
        point = Point(point_dict)
        
        point = backbone(point)
        
        if isinstance(point, Point):
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = point.feat[inverse]
                point = parent
            feat = point.feat  # (N, C)
        else:
            feat = point
        
        if "offset" not in point_dict:
            if self.pooling_type == "max":
                global_feat = feat.max(dim=0, keepdim=True)[0]
            elif self.pooling_type == "avg":
                global_feat = feat.mean(dim=0, keepdim=True)
            elif self.pooling_type == "attention":
                attn_weights = attention_pool(feat).softmax(dim=0)
                global_feat = (feat * attn_weights).sum(dim=0, keepdim=True)
            else:
                raise ValueError(f"Unknown pooling_type: {self.pooling_type}")
        else:
            offset = point_dict["offset"]
            batch = offset2batch(offset)
            batch_size = batch.max().item() + 1 if batch.numel() > 0 else 0 
            global_feat = []
            
            for i in range(batch_size):
                mask = (batch == i)
                sample_feat = feat[mask]
                
                if sample_feat.shape[0] == 0:
                    import warnings
                    warnings.warn(f"Empty batch at index {i}")
                    sample_global = torch.zeros(feat.shape[1], device=feat.device)
                else:
                    if self.pooling_type == "max":
                        sample_global = sample_feat.max(dim=0)[0]
                    elif self.pooling_type == "avg":
                        sample_global = sample_feat.mean(dim=0)
                    elif self.pooling_type == "attention":
                        attn_weights = attention_pool(sample_feat).softmax(dim=0)
                        sample_global = (sample_feat * attn_weights).sum(dim=0)
                    else:
                        raise ValueError(f"Unknown pooling_type: {self.pooling_type}")
                
                global_feat.append(sample_global)
            global_feat = torch.stack(global_feat)  # (batch_size, C)
        
        return global_feat
            
    def forward(self, input_dict, return_point=False):      
        device = next(self.parameters()).device
        input_dict = self._to_device(input_dict, device)         
        local_dict = input_dict["local"]
        parent_dict = input_dict["parent"] if self.use_parent_cloud else None
        
        local_feat = self.extract_features(local_dict, is_parent=False)
        parent_feat = None
        if self.use_parent_cloud and parent_dict is not None:
            parent_feat = self.extract_features(parent_dict, is_parent=True)

        if self.use_parent_cloud and parent_feat is not None:
            if self.fusion_type == "concat":
                global_feat = torch.cat([local_feat, parent_feat], dim=-1)  # (batch_size, 2C)
            elif self.fusion_type == "cross_attention":
                local_feat_unsqueeze = local_feat.unsqueeze(1)
                parent_feat_unsqueeze = parent_feat.unsqueeze(1)
                # print(f"\nBefore cross attention:")
                # print(f"   local_feat_unsqueeze: {local_feat_unsqueeze.shape}")
                # print(f"   parent_feat_unsqueeze: {parent_feat_unsqueeze.shape}")
                fused_feat, attn_weights = self.cross_attention(
                    query=local_feat_unsqueeze,
                    key=parent_feat_unsqueeze,
                    value=parent_feat_unsqueeze,
                )
                # print(f"\nAfter cross attention:")
                # print(f"   fused_feat: shape={fused_feat.shape}, mean={fused_feat.mean().item():.6f}, std={fused_feat.std().item():.6f}")
                # print(f"   attn_weights: shape={attn_weights.shape}")
                # print(f"   attn_weights values: {attn_weights.squeeze().detach().cpu().tolist()}")                
                global_feat = fused_feat.squeeze(1)  # (batch_size, C)
            else:
                raise ValueError(f"Unknown fusion_type: {self.fusion_type}")
        else:
            global_feat = local_feat
        
        if self.use_category_condition and "category_id" in input_dict:
            category_id = input_dict["category_id"]  # (batch_size,)
            if category_id.dtype != torch.long:
                category_id = category_id.long()
            
            category_emb = self.category_embedding(category_id)  # (batch_size, category_emb_dim)
            global_feat = torch.cat([global_feat, category_emb], dim=-1)  # (batch_size, C + emb_dim)
        
        position_pred_norm = self.regression_head(global_feat)  # (batch_size, 3)
        return_dict = {}
        
        if return_point:
            return_dict["local_feat"] = local_feat
            return_dict["parent_feat"] = parent_feat
            return_dict["global_feat"] = global_feat

        if self.training or "gt_position" in input_dict:
            gt_position_norm = input_dict["gt_position"]  # (batch_size, 3)
            loss_smooth_l1 = self.criterion_smooth_l1(position_pred_norm, gt_position_norm)
            loss_mse = self.criterion_mse(position_pred_norm, gt_position_norm)
            total_loss = loss_smooth_l1 * 1.0 + loss_mse * 0.5
            
            return_dict["loss"] = total_loss
            if not self.training:
                return_dict["pred_position"] = position_pred_norm
        else:
            return_dict["pred_position"] = position_pred_norm
        
        return return_dict

@MODELS.register_module()
class DefaultSegmentorV2(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        freeze_backbone=False,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, input_dict, return_point=False):
        point = Point(input_dict)
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat and use DefaultSegmentorV2
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            feat = point.feat
        else:
            feat = point
        seg_logits = self.seg_head(feat)
        return_dict = dict()
        if return_point:
            # PCA evaluator parse feat and coord in point
            return_dict["point"] = point
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
            return_dict["seg_logits"] = seg_logits
        # test
        else:
            return_dict["seg_logits"] = seg_logits
        return return_dict


@MODELS.register_module()
class DefaultLORASegmentorV2(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        freeze_backbone=False,
        use_lora=False,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        backbone_path=None,
        keywords=None,
        replacements=None,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.keywords = keywords
        self.replacements = replacements
        self.backbone = build_model(backbone)
        backbone_weight = torch.load(
            backbone_path,
            map_location=lambda storage, loc: storage.cuda(),
        )
        self.backbone_load(backbone_weight)

        self.criteria = build_criteria(criteria)
        self.freeze_backbone = freeze_backbone
        self.use_lora = use_lora

        if self.use_lora:
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["qkv"],
                # target_modules=["query", "value"],
                lora_dropout=lora_dropout,
                bias="none",
            )
            self.backbone.enc = get_peft_model(self.backbone.enc, lora_config)

        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        if self.use_lora:
            for name, param in self.backbone.named_parameters():
                if "lora_" in name:
                    param.requires_grad = True
        self.backbone.enc.print_trainable_parameters()

    def backbone_load(self, checkpoint):
        weight = OrderedDict()
        for key, value in checkpoint["state_dict"].items():
            if not key.startswith("module."):
                key = "module." + key  # xxx.xxx -> module.xxx.xxx
            # Now all keys contain "module." no matter DDP or not.
            if self.keywords in key:
                key = key.replace(self.keywords, self.replacements)
            key = key[7:]  # module.xxx.xxx -> xxx.xxx
            if key.startswith("backbone."):
                key = key[9:]
            weight[key] = value
        load_state_info = self.backbone.load_state_dict(weight, strict=False)
        print(f"Missing keys: {load_state_info[0]}")
        print(f"Unexpected keys: {load_state_info[1]}")

    def forward(self, input_dict, return_point=False):
        point = Point(input_dict)
        if self.freeze_backbone and not self.use_lora:
            with torch.no_grad():
                point = self.backbone(point)
        else:
            point = self.backbone(point)

        if isinstance(point, Point):
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            feat = point.feat
        else:
            feat = point

        seg_logits = self.seg_head(feat)
        return_dict = dict()
        if return_point:
            return_dict["point"] = point

        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
            return_dict["seg_logits"] = seg_logits
        else:
            return_dict["seg_logits"] = seg_logits
        return return_dict


@MODELS.register_module()
class DINOEnhancedSegmentor(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        freeze_backbone=False,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone = build_model(backbone) if backbone is not None else None
        self.criteria = build_criteria(criteria)
        self.freeze_backbone = freeze_backbone
        if self.backbone is not None and self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, input_dict, return_point=False):
        point = Point(input_dict)
        if self.backbone is not None:
            if self.freeze_backbone:
                with torch.no_grad():
                    point = self.backbone(point)
            else:
                point = self.backbone(point)
            point_list = [point]
            while "unpooling_parent" in point_list[-1].keys():
                point_list.append(point_list[-1].pop("unpooling_parent"))
            for i in reversed(range(1, len(point_list))):
                point = point_list[i]
                parent = point_list[i - 1]
                assert "pooling_inverse" in point.keys()
                inverse = point.pooling_inverse
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = point_list[0]
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pooling_inverse
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            feat = [point.feat]
        else:
            feat = []
        dino_coord = input_dict["dino_coord"]
        dino_feat = input_dict["dino_feat"]
        dino_offset = input_dict["dino_offset"]
        idx = torch_cluster.knn(
            x=dino_coord,
            y=point.origin_coord,
            batch_x=offset2batch(dino_offset),
            batch_y=offset2batch(point.origin_offset),
            k=1,
        )[1]

        feat.append(dino_feat[idx])
        feat = torch.concatenate(feat, dim=-1)
        seg_logits = self.seg_head(feat)
        return_dict = dict()
        if return_point:
            # PCA evaluator parse feat and coord in point
            return_dict["point"] = point
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
            return_dict["seg_logits"] = seg_logits
        # test
        else:
            return_dict["seg_logits"] = seg_logits
        return return_dict


@MODELS.register_module()
class DefaultClassifier(nn.Module):
    def __init__(
        self,
        backbone=None,
        criteria=None,
        num_classes=40,
        backbone_embed_dim=256,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.num_classes = num_classes
        self.backbone_embed_dim = backbone_embed_dim
        self.cls_head = nn.Sequential(
            nn.Linear(backbone_embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat
        # And after v1.5.0 feature aggregation for classification operated in classifier
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            point.feat = torch_scatter.segment_csr(
                src=point.feat,
                indptr=nn.functional.pad(point.offset, (1, 0)),
                reduce="mean",
            )
            feat = point.feat
        else:
            feat = point
        cls_logits = self.cls_head(feat)
        if self.training:
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss)
        elif "category" in input_dict.keys():
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss, cls_logits=cls_logits)
        else:
            return dict(cls_logits=cls_logits)
