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

"""Pretrain GPT"""

import torch
from megatron.training import pretrain

import os


if __name__ == "__main__":
    torch.set_flush_denormal(True)
    os.makedirs(os.environ.get('TIMER', 'timer'), exist_ok=True)
    # git_ds_info()
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        pretrain(None, None, None,
                 args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
                 data_post_process=None)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
