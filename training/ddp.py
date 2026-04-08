import os
import torch
import torch.distributed as dist


def setup_ddp():
    dist.init_process_group("nccl")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()