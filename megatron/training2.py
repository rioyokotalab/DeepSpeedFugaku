# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pretrain utilities."""

from datetime import datetime
import time
# The earliest we can measure the start time.
_TRAIN_START_TIME = time.time()

import torch
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

from megatron import get_args
from megatron import mpu
from megatron import print_rank_0
from megatron import print_rank_last
from megatron.model import Float16Module
from megatron.initialize import initialize_megatron

def print_datetime(string):
    """Note that this call will sync across all ranks."""
    torch.distributed.barrier()
    time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print_rank_0('[' + string + '] datetime: {} '.format(time_str))

class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.weight = torch.nn.Parameter(torch.randn(3000, 3000))

    def forward(self, x):
        return self.model(x)

def pretrain(train_valid_test_dataset_provider,
             model_provider,
             forward_step_func,
             extra_args_provider=None,
             args_defaults={},
             data_post_process=None):

    # Initalize and get arguments, timers, and Tensorboard writer.
    initialize_megatron(extra_args_provider=extra_args_provider,
                        args_defaults=args_defaults)

    # Adjust the startup time so it reflects the largest value.
    # This will be closer to what scheduler will see (outside of
    # image ... launches.
    global _TRAIN_START_TIME
    start_time_tensor = torch.FloatTensor([_TRAIN_START_TIME])
    torch.distributed.all_reduce(start_time_tensor,
                                 op=torch.distributed.ReduceOp.MIN)
    _TRAIN_START_TIME = start_time_tensor.item()
    print_rank_0('time to initialize megatron (seconds): {:.3f}'.format(
        time.time() - _TRAIN_START_TIME))
    print_datetime('after megatron is initialized')

    # args = get_args()
    model = [MyModel()]
    print_rank_last(model)
    print_datetime('after model, optimizer, and learning rate '
                   'scheduler are built')
    
    def get_model_memory(model: torch.nn.Module) -> int:
        total_memory = 0
        for param in model.parameters():
            total_memory += param.element_size() * param.nelement()
        return total_memory
    
    def allreduce_gradients(module):
        print_rank_last('call allreduce_gradients: _grad_buffers is None')
        buckets = {}
        for param in module.parameters():
            param.grad = torch.zeros_like(param)
            if param.requires_grad and param.grad is not None:
                tp = param.data.type()
                if tp not in buckets:
                    buckets[tp] = []
                buckets[tp].append(param)
                param.main_grad = param.grad

        # For each bucket, all-reduce and copy all-reduced grads.
        print_rank_last(f'call allreduce_gradients: {len(buckets)=}')
        for tp in buckets:
            bucket = buckets[tp]
            grads = [param.grad.data for param in bucket]
            coalesced = torch._utils._flatten_dense_tensors(grads)
            coalesced /= mpu.get_data_parallel_world_size()
            print_rank_last(f'call {coalesced.shape=}')
            torch.distributed.all_reduce(
                coalesced, group=mpu.get_data_parallel_group())
            for buf, synced in zip(grads,torch._utils._unflatten_dense_tensors(
                    coalesced, grads)):
                buf.copy_(synced)

    memory_usage = get_model_memory(model[0])
    print_rank_last(f'{memory_usage=}')
    # total_memory_usage = 0
    print_rank_last('---global vars start---')
    for key, value in globals().items():
        if not key.startswith('__') and not key.endswith('__'):
            print_rank_last(f"{key}: {value}")
        if key == '__file__':
            print_rank_last(f"{key}: {value}")
    print_rank_last('---global vars end---')

    for i in range(100):
        allreduce_gradients(model[0])
