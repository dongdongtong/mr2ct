import os
import sys
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(project_dir)
import yaml
import importlib

import torch
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group

# from trainers.sup_trainer import SupervisedTrainer
from datasets.create_dataloaders import create_loader, create_patch_loader
from utils.utils import set_random_seed


def ddp_setup(rank, world_size, port):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = port
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def trainer_setup(trainer_name):
    trainer_module = importlib.import_module(f"trainers.{trainer_name}")
    trainer = getattr(trainer_module, "TranslationTrainer")

    return trainer
    

def main(rank: int, world_size: int, args):
    if args.ddp:
        ddp_setup(rank, world_size, args.port)
    
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        random_seed = config['random_seed']

    set_random_seed(random_seed)

    loaders = create_patch_loader(args, rank)
    trainer = trainer_setup(args.trainer)(args, rank)
    
    trainer.train(loaders)
    
    if args.ddp:
        destroy_process_group()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='hematoma distributed training work')
    parser.add_argument('--config', type=str, help='Configuration to run the exp')
    parser.add_argument('--trainer', type=str, help='select one trainer to train')
    parser.add_argument('--fold', type=int, default=0, help='cross validation fold')
    parser.add_argument('--port', default="12345", type=str, help='Distributed Data Parallel starting port')
    parser.add_argument('--ddp', action="store_true", help='Distributed Data Parallel mode or not')
    parser.add_argument('--amp', action="store_true", help='Auto AMP mode or not')
    parser.add_argument('--gradient_accumulation_step', type=int, default=1, help='steps of gradient accumulation, default is 1 for no accumulation')
    parser.add_argument('--resume', action="store_true", help='Resume training or train from scratch')
    parser.add_argument('--resume_path', type=str, default="", help='Resume training path')
    parser.add_argument('--load_weights', action="store_true", help='resume training but only load weights, no epoch, optimizer, scheduler')


    args = parser.parse_args()
    
    # with open(args.config) as f:
    #     config = yaml.load(f, Loader=yaml.FullLoader)
    
    world_size = torch.cuda.device_count()
    if args.ddp:
        mp.spawn(main, args=(world_size, args), nprocs=world_size)
    else:
        main(0, 1, args)