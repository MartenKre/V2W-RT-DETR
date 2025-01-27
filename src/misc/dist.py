"""
reference
- https://github.com/pytorch/vision/blob/main/references/detection/utils.py
- https://github.com/facebookresearch/detr/blob/master/util/misc.py#L406

by lyuwenyu
"""

import random
import numpy as np 

import torch
import torch.nn as nn 
import torch.distributed
import torch.distributed as tdist

from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.data import DistributedSampler
from torch.utils.data.dataloader import DataLoader
from src.data.buoy_dataset import BuoyDataset, collate_fn_adapt


def init_distributed():
    '''
    distributed setup
    args:
        backend (str), ('nccl', 'gloo')
    '''
    try:
        # # https://pytorch.org/docs/stable/elastic/run.html
        # LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  
        # RANK = int(os.getenv('RANK', -1))
        # WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
        
        tdist.init_process_group(init_method='env://', )
        torch.distributed.barrier()

        rank = get_rank()
        device = torch.device(f'cuda:{rank}')
        torch.cuda.set_device(device)

        setup_print(rank == 0)
        print('Initialized distributed mode...')

        return True 

    except:
        print('Not init distributed mode.')
        return False 


def setup_print(is_main):
    '''This function disables printing when not in master process
    '''
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_main or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_available_and_initialized():
    if not tdist.is_available():
        return False
    if not tdist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_available_and_initialized():
        return 0
    return tdist.get_rank()


def get_world_size():
    if not is_dist_available_and_initialized():
        return 1
    return tdist.get_world_size()

    
def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)



def warp_model(model, find_unused_parameters=False, sync_bn=False,):
    if is_dist_available_and_initialized():
        rank = get_rank()
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model) if sync_bn else model 
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=find_unused_parameters)
    return model

# def warp_loader(loader, shuffle=False):        
def warp_loader(path_to_yaml = "",mode="train", batch_size=4, num_workers=4, augment=True, shuffle=True):
    drop_last = True if mode=="train" else False
    dataset=BuoyDataset(yaml_file=path_to_yaml ,mode=mode, transform=False, augment=augment)
    if is_dist_available_and_initialized():
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        loader = DataLoader(dataset, 
                            batch_size, 
                            sampler=sampler, 
                            drop_last=drop_last, 
                            collate_fn=collate_fn_adapt, 
                            num_workers=num_workers, )
    else:
        sampler = torch.utils.data.RandomSampler(dataset) if shuffle==True else torch.utils.data.SequentialSampler(dataset)
        loader = DataLoader(dataset, 
                            batch_size, 
                            sampler=sampler,
                            drop_last=drop_last, 
                            collate_fn=collate_fn_adapt, 
                            num_workers=num_workers)
    return loader




def is_parallel(model) -> bool:
    # Returns True if model is of type DP or DDP
    return type(model) in (torch.nn.parallel.DataParallel, torch.nn.parallel.DistributedDataParallel)


def de_parallel(model) -> nn.Module:
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model


def reduce_dict(data, avg=True):
    '''
    Args 
        data dict: input, {k: v, ...}
        avg bool: true
    '''
    world_size = get_world_size()
    if world_size < 2:
        return data
    
    with torch.no_grad():
        keys, values = [], []
        for k in sorted(data.keys()):
            keys.append(k)
            values.append(data[k])

        values = torch.stack(values, dim=0)
        tdist.all_reduce(values)

        if avg is True:
            values /= world_size
        
        _data = {k: v for k, v in zip(keys, values)}
    
    return _data



def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    data_list = [None] * world_size
    tdist.all_gather_object(data_list, data)
    return data_list

    
import time 
def sync_time():
    '''sync_time
    '''
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return time.time()



def set_seed(seed):
    # fix the seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


