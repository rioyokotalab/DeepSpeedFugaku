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

from commons import print_separator
from commons import initialize_distributed
from deepspeed.accelerator import get_accelerator
from megatron import mpu
import torch
import sys
sys.path.append("../..")

num_devices = get_accelerator().device_count()
dev_name = 'cuda' if num_devices > 0 else 'cpus'
# Same as megatron/mpu/random.py
seed_offset = 2718

def test_set_rng_state(tensor_model_parallel_size):

    if torch.distributed.get_rank() == 0:
        print('> testing set_rng_state with size {} ...'.
              format(tensor_model_parallel_size))

    mpu.initialize_model_parallel(tensor_model_parallel_size)
    tensor_model_parallel_size = mpu.get_tensor_model_parallel_world_size()

    size = 123
    seed = 1234
    global num_devices

    if num_devices > 0:
        get_accelerator().manual_seed(seed)
        tensor = get_accelerator().FloatTensor(size)
    else: # CPUs
        torch.manual_seed(seed)
        tensor = torch.FloatTensor(size)

    # Get the state
    if num_devices > 0:
        rng_state = get_accelerator().get_rng_state()
    else:
        rng_state = torch.get_rng_state()
    rng_state_copy = rng_state.clone()

    # Do some stuff.
    for _ in range(5):
        torch.randn(size, out=tensor)
    result_1 = tensor.clone()

    assert rng_state.sub(rng_state_copy).max() == 0
    if num_devices > 0:
        assert get_accelerator().get_rng_state().sub(rng_state_copy).max() > 0
    else:
        assert torch.get_rng_state().sub(rng_state_copy).max() > 0

    # State should be different.
    if num_devices > 0:
        new_rng_state = get_accelerator().get_rng_state()
    else:
        new_rng_state = torch.get_rng_state()
    max_diff = new_rng_state.sub(rng_state).max()
    print('   max diff in rng state (should be non-zero) on global rank {}: {}'.
          format(torch.distributed.get_rank(), max_diff))
    assert max_diff > 0

    # Reset the rng state and do the same stuff.
    if num_devices > 0:
        mpu.random._set_cuda_rng_state(rng_state)
    else:
        torch.set_rng_state(rng_state)
    for _ in range(5):
        torch.randn(size, out=tensor)
    if num_devices > 0:
        mpu.random._set_cuda_rng_state(rng_state)
    else:
        torch.set_rng_state(rng_state)
    for _ in range(5):
        torch.randn(size, out=tensor)
    result_2 = tensor.clone()

    # Results should be the same
    error = result_2.sub(result_1).abs().max()
    print('   max error in generated tensors (should be zero) on '
          'global rank {}: {}'.format(torch.distributed.get_rank(), error))
    assert error < 1.0e-6

    # Input state should have remained intact.
    error = rng_state.sub(rng_state_copy).max()
    print('   max error in rng state (should be zero) on global rank {}: {}'.
          format(torch.distributed.get_rank(), error))
    assert error == 0

    # Reset groups
    mpu.destroy_model_parallel()

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print('>> passed the test :-)')


def test_rng_tracker(tensor_model_parallel_size):
    global num_devices
    global dev_name

    if torch.distributed.get_rank() == 0:
        print('> testing {} rng tracker with size {} ...'.
                  format(dev_name, tensor_model_parallel_size))

    mpu.initialize_model_parallel(tensor_model_parallel_size)
    tensor_model_parallel_size = mpu.get_tensor_model_parallel_world_size()

    seed_1 = 1234
    seed_2 = 4321
    size = [12, 21]

    if num_devices > 0:
        tensor = get_accelerator().FloatTensor(size)
    else:
        tensor = torch.FloatTensor(size)

    # Set to seed_1 and generate two tensors.
    if num_devices > 0:
        get_accelerator().manual_seed(seed_1)
    else:
        torch.manual_seed(seed_1)
    torch.randn(size, out=tensor)
    target_11 = tensor.clone()
    torch.randn(size, out=tensor)
    target_12 = tensor.clone()

    # Set to seed_2 and generate two tensors.
    if num_devices > 0:
        get_accelerator().manual_seed(seed_2)
    else:
        torch.manual_seed(seed_2)

    torch.randn(size, out=tensor)
    target_21 = tensor.clone()
    torch.randn(size, out=tensor)
    target_22 = tensor.clone()

    # Now if we interleave seed_1 and seed_2,
    # we should still get the same tensors
    if num_devices > 0:
        get_accelerator().manual_seed(seed_1)
        mpu.get_cuda_rng_tracker().add('test', seed_2)
    else:
        torch.manual_seed(seed_1)
        mpu.get_cpus_rng_tracker().add('test', seed_2)

    torch.randn(size, out=tensor)
    result_11 = tensor.clone()

    if num_devices > 0:
        with mpu.get_cuda_rng_tracker().fork('test'):
            torch.randn(size, out=tensor)
    else:
        with mpu.get_cpus_rng_tracker().fork('test'):
            torch.randn(size, out=tensor)
    result_21 = tensor.clone()

    torch.randn(size, out=tensor)
    result_12 = tensor.clone()

    if num_devices > 0:
        with mpu.get_cuda_rng_tracker().fork('test'):
            torch.randn(size, out=tensor)
    else:
        with mpu.get_cpus_rng_tracker().fork('test'):
            torch.randn(size, out=tensor)
    result_22 = tensor.clone()

    diff = result_11.sub(result_21).abs().max()
    diff = min(diff, result_12.sub(result_22).abs().max())
    print('   max diff in generated tensors (should be non-zero) on '
          'global rank {}: {}'.format(torch.distributed.get_rank(), diff))
    assert diff > 1.0e-6
    error = max(result_11.sub(target_11).abs().max(),
                result_12.sub(target_12).abs().max())
    error = max(error, result_21.sub(target_21).abs().max())
    error = max(error, result_22.sub(target_22).abs().max())
    print('   max error in generated tensors (should be zero) on '
          'global rank {}: {}'.format(torch.distributed.get_rank(), error))
    assert error < 1.0e-6

    # Reset the tracker
    if num_devices > 0:
        mpu.get_cuda_rng_tracker().reset()
    else:
        mpu.get_cpus_rng_tracker().reset()

    # Reset groups
    mpu.destroy_model_parallel()

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print('>> passed the test :-)')


def test_model_parallel_manual_seed(tensor_model_parallel_size):
    global num_devices
    global dev_name
    global seed_offset

    if torch.distributed.get_rank() == 0:
        print('> testing model parallel {} manual seed with size {} ...'.
                  format(dev_name, tensor_model_parallel_size))

    mpu.initialize_model_parallel(tensor_model_parallel_size)
    tensor_model_parallel_size = mpu.get_tensor_model_parallel_world_size()

    if num_devices > 0:
        mpu.model_parallel_cuda_manual_seed(12345)
        assert get_accelerator().initial_seed() == 12345
    else:
        mpu.model_parallel_cpus_manual_seed(12345)
        assert torch.initial_seed() == 12345

    if num_devices > 0:
        with mpu.get_cuda_rng_tracker().fork():
            assert get_accelerator().initial_seed() == (12345 + seed_offset +
                                                        mpu.get_tensor_model_parallel_rank())
    else:
        with mpu.get_cpus_rng_tracker().fork():
            assert torch.initial_seed() == (12345 +seed_offset +
                                          mpu.get_tensor_model_parallel_rank())

    # Reset the tracker
    if num_devices > 0:
        mpu.get_cuda_rng_tracker().reset()
    else:
        mpu.get_cpus_rng_tracker().reset()

    # Reset groups
    mpu.destroy_model_parallel()

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print('>> passed the test :-)')


if __name__ == '__main__':
    num_devices = get_accelerator().device_count()

    if num_devices > 0:
        dev_name = 'cuda'
    else:
        dev_name = 'cpus'

    initialize_distributed()
    world_size = torch.distributed.get_world_size()

    tensor_model_parallel_size = 1
    while tensor_model_parallel_size <= world_size:
        print_separator('test set rng state')
        test_set_rng_state(tensor_model_parallel_size)
        tensor_model_parallel_size *= 2

    tensor_model_parallel_size = 1
    while tensor_model_parallel_size <= world_size:
        print_separator('test {} rng tracker'.format(dev_name))
        test_rng_tracker(tensor_model_parallel_size)
        tensor_model_parallel_size *= 2

    tensor_model_parallel_size = 1
    while tensor_model_parallel_size <= world_size:
        print_separator('test model parallel {} manual seed'.format(dev_name))
        test_model_parallel_manual_seed(tensor_model_parallel_size)
        tensor_model_parallel_size *= 2
