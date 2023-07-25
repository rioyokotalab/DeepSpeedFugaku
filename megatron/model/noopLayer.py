from megatron import get_timers
import torch

class NoopLayer(torch.nn.Module):
    def __init__(self, name, start):
        super(NoopLayer, self).__init__()
        timers = get_timers()
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                if start:
                    self.register_backward_hook(lambda module, grad_input, grad_output: timers(name).start())
                else:
                    self.register_backward_hook(lambda module, grad_input, grad_output: timers(name).stop())
        else:
            if start:
                self.register_module_full_backward_hook(lambda module, grad_input, grad_output: timers(name).start())
            else:
                self.register_module_full_backward_hook(lambda module, grad_input, grad_output: timers(name).stop())
    def forward(self, x):
        return x