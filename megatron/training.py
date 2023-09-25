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
import math
import sys
import time
import json
import subprocess
# The earliest we can measure the start time.
_TRAIN_START_TIME = time.time()

import torch
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

from megatron import get_args
from megatron import get_timers
from megatron import get_tensorboard_writer
from megatron import get_current_global_batch_size
from megatron import get_num_microbatches
from megatron import is_last_rank
from megatron import update_num_microbatches
from megatron import mpu
from megatron import print_rank_0
from megatron import print_rank_last
from megatron.checkpointing import load_checkpoint
from megatron.checkpointing import save_checkpoint
from megatron.model import Float16Module
from megatron.optimizer import get_megatron_optimizer
from megatron.initialize import initialize_megatron
from megatron.initialize import write_args_to_tensorboard
from megatron.learning_rates import AnnealingLR
from megatron.model import DistributedDataParallel as LocalDDP
from megatron.utils import check_adlr_autoresume_termination
from megatron.utils import unwrap_model
from megatron.data.data_samplers import build_pretraining_data_loader
from megatron.utils import calc_params_l2_norm
from megatron.schedules import forward_backward_no_pipelining
from megatron.schedules import forward_backward_pipelining_without_interleaving
from megatron.schedules import forward_backward_pipelining_with_interleaving
from megatron.utils import report_memory, throughput_calculator, checkpoint_throughput_calculator

import deepspeed
from deepspeed.compression.compress import init_compression, redundancy_clean


from megatron.model.transformer import  ParallelTransformerLayer
from deepspeed.runtime.data_pipeline.data_routing.helper import convert_to_random_ltd

def print_datetime(string):
    """Note that this call will sync across all ranks."""
    torch.distributed.barrier()
    time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print_rank_0('[' + string + '] datetime: {} '.format(time_str))

class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # layers = []
        # # Input layer to first hidden layer
        # layers.append(nn.Linear(1000, 5000))
        # # Hidden layers
        # for _ in range(10):
        #     layers.append(nn.Linear(5000, 5000))
        # # Output layer
        # layers.append(nn.Linear(5000, 1))
        # self.model = nn.Sequential(*layers)
        self.weight = torch.nn.Parameter(torch.randn(1000, 1000))

    def forward(self, x):
        return self.model(x)

def pretrain(train_valid_test_dataset_provider,
             model_provider,
             forward_step_func,
             extra_args_provider=None,
             args_defaults={},
             data_post_process=None):
    """Main training program.

    This function will run the followings in the order provided:
        1) initialize Megatron.
        2) setup model, optimizer and lr schedule using the model_provider.
        3) call train_val_test_data_provider to get train/val/test datasets.
        4) train the modle using the forward_step_func.

    Arguments:
        train_valid_test_dataset_provider: a function that takes the size of
            train/valid/test dataset and returns `train, valid, test` datasets.
        model_provider: a function that returns a vanilla version of the
            model. By vanilla we mean a simple model on cpu with no fp16 or ddp.
        forward_step_func: a function that takes a `data iterator` and `model`,
            and returns a `loss` scalar with a dictionary with key:values being
            the info we would like to monitor during training, for example
            `lm-loss: value`. We also require that this function add
            `batch generator` to the timers class.
        extra_args_provider: a function that takes a parser and adds arguments
            to it. It is used for programs to add their own arguments.
        args_defaults: a dictionary from argument-name to argument-value. It
            to set already parse arguments.
    """

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

    args = get_args()
    # timers = get_timers()

    # if args.deepspeed:
    #     args.deepspeed_configuration = json.load(
    #         open(args.deepspeed_config, 'r', encoding='utf-8'))
    #     if "curriculum_learning" in args.deepspeed_configuration and \
    #         "enabled" in args.deepspeed_configuration["curriculum_learning"]:
    #         args.curriculum_learning_legacy = args.deepspeed_configuration[ \
    #             "curriculum_learning"]["enabled"]
    #     if args.curriculum_learning_legacy and not args.no_pipeline_parallel:
    #         from deepspeed.runtime.data_pipeline.curriculum_scheduler \
    #             import CurriculumScheduler
    #         args.curriculum_scheduler = CurriculumScheduler( \
    #             args.deepspeed_configuration["curriculum_learning"])
    #     if "compression_training" in args.deepspeed_configuration:
    #         args.compression_training = True

    # Model, optimizer, and learning rate.
    # timers('model-and-optimizer-setup').start()
    # model, optimizer, lr_scheduler = setup_model_and_optimizer(
    #     model_provider, teacher=False, data_post_process=data_post_process,
    #     build_train_valid_test_datasets_provider=train_valid_test_dataset_provider)
    model = [MyModel()]
    print_rank_last(model)
    # timers('model-and-optimizer-setup').stop()
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
            # print_rank_last(f'call allreduce_gradients: {param.shape=}, {param.requires_grad=}, {param.grad=}')
            # if param.grad is None:
            param.grad = torch.zeros_like(param)
            if param.requires_grad and param.grad is not None:
                # print_rank_last('call allreduce_gradients: param.requires_grad and param.grad is not None')
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
    for i in range(args.train_iters):
        allreduce_gradients(model[0])

