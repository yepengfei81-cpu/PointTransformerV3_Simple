_base_ = ["../_base_/default_runtime.py"]

batch_size = 8
batch_size_val = 2
num_worker = 8
world_size = 1
batch_size_per_gpu = 4
batch_size_val_per_gpu = 4
num_worker_per_gpu = 24
mix_prob = 0.0
empty_cache = False
enable_amp = False
enable_wandb = True

model = dict(
    type="ContactPositionRegressor",
    num_outputs=3,
    num_classes=3,
    use_category_condition=True,
    category_emb_dim=32,
    backbone_out_channels=512,
    backbone=dict(
        type="PT-v3m1",
        in_channels=3,
        order=["z", "z-trans", "hilbert", "hilbert-trans"],
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(128, 128, 128, 128, 128),
        # dec_depths=(2, 2, 2, 2),
        # dec_channels=(64, 64, 128, 256),
        # dec_num_head=(4, 4, 8, 16),
        # dec_patch_size=(128, 128, 128, 128),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=True,
        enable_flash=False,
        upcast_attention=True,
        upcast_softmax=True,
        enc_mode=True, #False
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=False,
        pdnorm_adaptive=False,
        pdnorm_affine=False,
        pdnorm_conditions=(),
    ),
    # criteria=[
    #     dict(type="SmoothL1Loss", loss_weight=1.0),
    #     dict(type="MSELoss", loss_weight=0.5),
    # ],
    freeze_backbone=False,
    pooling_type="attention",  
    use_parent_cloud=False,
    parent_backbone=None, # shared backbone
    fusion_type="cross_attention",  # "concat" | "cross_attention"  
)

epoch = 500
eval_epoch = 250
optimizer = dict(type="AdamW", lr=0.001, weight_decay=0.01)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.001, 0.001],
    pct_start=0.1,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=0.0001)]

dataset_type = "ContactPositionDataset"
data_root = "/home/ypf/touch_processed_data/"

data = dict(
    ignore_index=-1,
    names=["Scissors", "Cup", "Avocado"],
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        parent_pcd_root=data_root,
        max_cache_size=10,
        loop=epoch // eval_epoch,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            dict(type="ChromaticJitter", p=0.95, std=0.05),
            dict(
                type="GridSample",
                grid_size=0.002,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "gt_position", "category_id", "name"),
                feat_keys=("color",),
                # offset_keys_dict={},
            ),
        ],
        parent_transform=[
            dict(
                type="GridSample",
                grid_size=0.002,  # 与局部点云一致
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
        ],        
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        parent_pcd_root=data_root,
        max_cache_size=10,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="GridSample",
                grid_size=0.002,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "gt_position", "category_id", "name"),
                feat_keys=("color",),
                # offset_keys_dict={},
            ),
        ],
        parent_transform=[
            dict(
                type="GridSample",
                grid_size=0.002,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
        ],        
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        parent_pcd_root=data_root,
        max_cache_size=10,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="NormalizeColor"),
        ],
        parent_transform=[
            dict(
                type="GridSample",
                grid_size=0.002,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
        ],           
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.002,
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
            ),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "name"),
                    feat_keys=("color",),
                    # offset_keys_dict={},
                ),
            ],
            aug_transform=[],
        ),
    ),
)

# adapt new datasets trainer
train = dict(
    type="RegressionTrainer",
)

# TODO：CHECK HOOK LIST
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="RegressionEvaluator", write_per_class=False),
    dict(type="CheckpointSaver", save_freq=None),
]