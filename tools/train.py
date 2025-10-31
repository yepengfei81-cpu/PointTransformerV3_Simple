"""
Main Training Script

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from pointcept.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from pointcept.engines.train import TRAINERS
from pointcept.engines.launch import launch
import pointcept.utils.comm as comm


def main_worker(cfg):
    cfg = default_setup(cfg)
    print("Debug: Configuration Parameters")
    print(f"world_size: {comm.get_world_size()}")  # 分布式训练的总进程数
    print(f"batch_size_per_gpu: {cfg.batch_size_per_gpu}")
    print(f"batch_size_val_per_gpu: {cfg.batch_size_val_per_gpu}")
    print(f"num_worker_per_gpu: {cfg.num_worker_per_gpu}")
    print(f"batch_size: {cfg.batch_size}")
    print(f"batch_size_val: {cfg.batch_size_val}")
    print(f"num_worker: {cfg.num_worker}")    
    trainer = TRAINERS.build(dict(type=cfg.train.type, cfg=cfg))
    trainer.train()


def main():
    args = default_argument_parser().parse_args()
    cfg = default_config_parser(args.config_file, args.options)

    launch(
        main_worker,
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        cfg=(cfg,),
    )


if __name__ == "__main__":
    main()
