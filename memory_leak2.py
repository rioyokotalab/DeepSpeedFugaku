import torch
import torch.nn as nn
import os

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.weight = torch.nn.Parameter(torch.randn(1000, 1000))

    def forward(self, x):
        return self.model(x)

def is_last_rank():
    return torch.distributed.get_rank() == (
        torch.distributed.get_world_size() - 1)

def print_rank_last(message):
    """If distributed is initialized, print only on last rank."""
    if torch.distributed.is_initialized():
        if is_last_rank():
            print(message, flush=True)
    else:
        print(message, flush=True)

# distibuted init
init_method = 'tcp://'
master_ip = os.getenv('MASTER_ADDR', 'localhost')
master_port = os.getenv('MASTER_PORT', '6000')
init_method += master_ip + ':' + master_port
rank = int(os.getenv('PMIX_RANK'))
world_size = int(os.getenv("WORLD_SIZE", '1'))

torch.distributed.init_process_group(
    backend='mpi',
    world_size=world_size,
    rank = rank,
    init_method=init_method)

# Create the model
model = MyModel()

# Count the actual number of parameters
actual_params = sum(p.nelement() for p in model.parameters())
print(actual_params)
print(actual_params*4/1024/1024/1024)

# allreduce and memory leak
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    for i in range(300):
        buckets = {}
        for param in model.parameters():
            # if param.grad is None:
            param.grad = torch.zeros_like(param)
            dt = param.data.type()
            if buckets.get(dt) is None:
                buckets[dt] = []
            buckets[dt].append(param)
            param.main_grad = param.grad
        for tp in buckets:
            grads = [param.grad.data for param in buckets[tp]]
            coalesced = torch._utils._flatten_dense_tensors(grads)
            # coalesced /= mpu.get_data_parallel_world_size()
            torch.distributed.all_reduce(coalesced)
            for buf, synced in zip(grads, torch._utils._unflatten_dense_tensors(
                    coalesced, grads)):
                buf.copy_(synced)
        print_rank_last(f'{i=}, {coalesced.shape=}')