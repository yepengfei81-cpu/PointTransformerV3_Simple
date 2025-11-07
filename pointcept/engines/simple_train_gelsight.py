import os
import torch
from functools import partial
from pathlib import Path
from tensorboardX import SummaryWriter
import wandb
from .simple_train import TrainerBase, TRAINERS
from .defaults import worker_init_fn
import pointcept.utils.comm as comm
from pointcept.datasets import build_dataset, point_collate_fn
from pointcept.models import build_model
from pointcept.utils.logger import get_root_logger
from pointcept.utils.optimizer import build_optimizer
from pointcept.utils.scheduler import build_scheduler
from pointcept.utils.events import EventStorage, ExceptionWriter


@TRAINERS.register_module("RegressionTrainer")
class RegressionTrainer(TrainerBase):
    def __init__(self, cfg):
        super(RegressionTrainer, self).__init__()
        # Basic attributes
        self.epoch = 0
        self.start_epoch = 0
        self.max_epoch = cfg.eval_epoch
        self.best_metric_value = -torch.inf
        self.cfg = cfg
        
        # Setup logger
        self.logger = get_root_logger(
            log_file=os.path.join(cfg.save_path, "train.log"),
            file_mode="a" if cfg.get("resume", False) else "w",
        )
        
        self.logger.info("=" * 80)
        self.logger.info("Initializing RegressionTrainer")
        self.logger.info("=" * 80)
        self.logger.info(f"Save path: {cfg.save_path}")
        
        # Build components
        self.logger.info("=> Building model ...")
        self.model = self.build_model()
        
        self.logger.info("=> Building writer ...")
        self.writer = self.build_writer()
        
        self.logger.info("=> Building train dataloader ...")
        self.train_loader = self.build_train_loader()
        
        self.logger.info("=> Building val dataloader ...")
        self.val_loader = self.build_val_loader()
        
        self.logger.info("=> Building optimizer ...")
        self.optimizer = self.build_optimizer()
        
        self.logger.info("=> Building scheduler ...")
        self.scheduler = self.build_scheduler()
        
        # Register hooks
        self.logger.info("=> Registering hooks ...")
        self.register_hooks(cfg.hooks)
        
        self.logger.info("✅ RegressionTrainer initialized successfully")
        self.logger.info("=" * 80)
    
    def train(self):
        """Main training loop"""
        with EventStorage() as self.storage, ExceptionWriter():
            self.before_train()
            self.logger.info("\n" + "=" * 80)
            self.logger.info("Start Training")
            self.logger.info("=" * 80)
            
            for self.epoch in range(self.start_epoch, self.max_epoch):
                self.model.train()
                self.data_iterator = enumerate(self.train_loader)
                self.before_epoch()
                
                # Training loop
                for (
                    self.comm_info["iter"],
                    self.comm_info["input_dict"],
                ) in self.data_iterator:
                    self.before_step()
                    self.run_step()
                    self.after_step()
                
                self.after_epoch()
            
            self.after_train()
            self.logger.info("=" * 80)
            self.logger.info("Training Completed")
            self.logger.info("=" * 80)

    def after_epoch(self):
        for h in self.hooks:
            h.after_epoch()
        # Reset storage for next epoch
        self.storage.reset_histories()
        # Clear cache if configured
        if self.cfg.get("empty_cache_per_epoch", False):
            torch.cuda.empty_cache()

    def run_step(self):
        """Execute one training step"""
        input_dict = self.comm_info["input_dict"]
        
        # Move data to GPU
        for key in input_dict.keys():
            if isinstance(input_dict[key], torch.Tensor):
                input_dict[key] = input_dict[key].cuda(non_blocking=True)
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass
        output_dict = self.model(input_dict)
        loss = output_dict["loss"]
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping (if configured)
        if self.cfg.get("clip_grad", None) is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.cfg.clip_grad
            )
        
        # Update parameters
        self.optimizer.step()
        self.scheduler.step()
        
        # Empty cache if needed
        if self.cfg.get("empty_cache", False):
            torch.cuda.empty_cache()
        
        # Store output for hooks
        self.comm_info["model_output_dict"] = output_dict
        
        # Log to storage (for hooks to use)
        self.storage.put_scalar("loss", loss.item())
        self.storage.put_scalar("lr", self.optimizer.param_groups[0]["lr"])
    
    def build_model(self):
        """Build and initialize model"""
        model = build_model(self.cfg.model)
        
        # Count parameters
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"   Model: {type(model).__name__}")
        self.logger.info(f"   Trainable parameters: {n_parameters:,}")
        
        # Move to GPU
        model = model.cuda()
        
        return model

    def build_writer(self):
        writer = SummaryWriter(self.cfg.save_path) if comm.is_main_process() else None
        self.logger.info(f"Tensorboard writer logging dir: {self.cfg.save_path}")
        if self.cfg.enable_wandb and comm.is_main_process():
            tag, name = Path(self.cfg.save_path).parts[-2:]
            wandb.init(
                project=self.cfg.wandb_project,
                name=f"{tag}/{name}",
                tags=[tag],
                dir=self.cfg.save_path,
                settings=wandb.Settings(api_key=self.cfg.wandb_key),
                config=self.cfg,
            )
        return writer

    def build_train_loader(self):
        """Build training data loader"""
        train_data = build_dataset(self.cfg.data.train)
        
        self.logger.info(f"   Train samples: {len(train_data)}")
        
        init_fn = (
            partial(
                worker_init_fn,
                num_workers=self.cfg.num_worker_per_gpu,
                rank=comm.get_rank(),
                seed=self.cfg.get("seed", None),
            )
            if self.cfg.get("seed", None) is not None
            else None
        )
        
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.cfg.batch_size_per_gpu,
            shuffle=True,
            num_workers=self.cfg.num_worker_per_gpu,
            sampler=None,
            collate_fn=partial(point_collate_fn, mix_prob=self.cfg.mix_prob),
            pin_memory=True,
            worker_init_fn=init_fn,
            drop_last=len(train_data) > self.cfg.batch_size_per_gpu,
            persistent_workers=True,
        )
        
        self.logger.info(f"   Batches per epoch: {len(train_loader)}")
        
        return train_loader

    def build_val_loader(self):
        val_loader = None
        
        if self.cfg.get("evaluate", False):
            val_data = build_dataset(self.cfg.data.val)
            self.logger.info(f"   Val samples: {len(val_data)}")
            
            if comm.get_world_size() > 1:
                val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
            else:
                val_sampler = None
            
            val_loader = torch.utils.data.DataLoader(
                val_data,
                batch_size=self.cfg.batch_size_val_per_gpu,
                shuffle=False,
                num_workers=self.cfg.num_worker_per_gpu,
                pin_memory=True,
                sampler=val_sampler,
                collate_fn=partial(point_collate_fn, mix_prob=0.0),  # 🔥 验证时不 mix
            )
            
            self.logger.info(f"   Val batches: {len(val_loader)}")
        else:
            self.logger.info("   Validation disabled")
        
        return val_loader

    def build_optimizer(self):
        """Build optimizer"""
        param_dicts = self.cfg.get("param_dicts", [])
        optimizer = build_optimizer(self.cfg.optimizer, self.model, param_dicts)
        
        self.logger.info(f"   Optimizer: {type(optimizer).__name__}")
        self.logger.info(f"   Base learning rate: {self.cfg.optimizer.lr}")
        
        return optimizer
    
    def build_scheduler(self):
        """Build learning rate scheduler"""
        # Calculate total steps
        total_steps = len(self.train_loader) * self.max_epoch
        self.cfg.scheduler.total_steps = total_steps
        
        scheduler = build_scheduler(self.cfg.scheduler, self.optimizer)
        
        self.logger.info(f"   Scheduler: {type(scheduler).__name__}")
        self.logger.info(f"   Total steps: {total_steps}")
        
        return scheduler