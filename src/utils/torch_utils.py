import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.optim.optimizer import Optimizer

import numpy as np

def keras_decay(step, decay=0.01):
    """Learning rate decay in Keras-style"""
    return 1.0 / (1.0 + decay * step)


class MipLRDecay(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, lr_init, lr_final, max_steps, lr_delay_steps=0, lr_delay_mult=1):
        self.lr_init = lr_init
        self.lr_final = lr_final
        self.max_steps = max_steps
        self.lr_delay_steps = lr_delay_steps
        self.lr_delay_mult = lr_delay_mult
        super(MipLRDecay, self).__init__(optimizer)

    def get_lr(self):
        step = self.last_epoch
        if self.lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = self.lr_delay_mult + (1 - self.lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(self.last_epoch / self.lr_delay_steps, 0, 1))
        else:
            delay_rate = 1.
        t = np.clip(step / self.max_steps, 0, 1)
        log_lerp = np.exp(np.log(self.lr_init) * (1 - t) + np.log(self.lr_final) * t)
        return [delay_rate * log_lerp]
    

class LARC(object):
    """
    :class:`LARC` is a pytorch implementation of both the scaling and clipping variants of LARC,
    in which the ratio between gradient and parameter magnitudes is used to calculate an adaptive 
    local learning rate for each individual parameter. The algorithm is designed to improve
    convergence of large batch training.
     
    See https://arxiv.org/abs/1708.03888 for calculation of the local learning rate.

    In practice it modifies the gradients of parameters as a proxy for modifying the learning rate
    of the parameters. This design allows it to be used as a wrapper around any torch.optim Optimizer.

    ```
    model = ...
    optim = torch.optim.Adam(model.parameters(), lr=...)
    optim = LARC(optim)
    ```

    It can even be used in conjunction with apex.fp16_utils.FP16_optimizer.

    ```
    model = ...
    optim = torch.optim.Adam(model.parameters(), lr=...)
    optim = LARC(optim)
    optim = apex.fp16_utils.FP16_Optimizer(optim)
    ```

    Args:
        optimizer: Pytorch optimizer to wrap and modify learning rate for.
        trust_coefficient: Trust coefficient for calculating the lr. See https://arxiv.org/abs/1708.03888
        clip: Decides between clipping or scaling mode of LARC. If `clip=True` the learning rate is set to `min(optimizer_lr, local_lr)` for each parameter. If `clip=False` the learning rate is set to `local_lr*optimizer_lr`.
        eps: epsilon kludge to help with numerical stability while calculating adaptive_lr
    """

    def __init__(self, optimizer, trust_coefficient=0.02, clip=True, eps=1e-8):
        self.optim = optimizer
        self.trust_coefficient = trust_coefficient
        self.eps = eps
        self.clip = clip

    def __getstate__(self):
        return self.optim.__getstate__()

    def __setstate__(self, state):
        self.optim.__setstate__(state)

    @property
    def state(self):
        return self.optim.state

    def __repr__(self):
        return self.optim.__repr__()

    @property
    def param_groups(self):
        return self.optim.param_groups

    @param_groups.setter
    def param_groups(self, value):
        self.optim.param_groups = value
    
    def state_dict(self):
        return self.optim.state_dict()

    def load_state_dict(self, state_dict):
        self.optim.load_state_dict(state_dict)

    def zero_grad(self):
        self.optim.zero_grad()

    def add_param_group(self, param_group):
        self.optim.add_param_group( param_group)

    def step(self):
        with torch.no_grad():
            weight_decays = []
            for group in self.optim.param_groups:
                # absorb weight decay control from optimizer
                weight_decay = group['weight_decay'] if 'weight_decay' in group else 0
                weight_decays.append(weight_decay)
                group['weight_decay'] = 0
                for p in group['params']:
                    if p.grad is None:
                        continue
                    param_norm = torch.norm(p.data)
                    grad_norm = torch.norm(p.grad.data)

                    if param_norm != 0 and grad_norm != 0:
                        # calculate adaptive lr + weight decay
                        adaptive_lr = self.trust_coefficient * (param_norm) / (grad_norm + param_norm * weight_decay + self.eps)

                        # clip learning rate for LARC
                        if self.clip:
                            # calculation of adaptive_lr so that when multiplied by lr it equals `min(adaptive_lr, lr)`
                            adaptive_lr = min(adaptive_lr/group['lr'], 1)

                        p.grad.data += weight_decay * p.data
                        p.grad.data *= adaptive_lr

        self.optim.step()
        # return weight decay control to optimizer
        for i, group in enumerate(self.optim.param_groups):
            group['weight_decay'] = weight_decays[i]




class LARS(Optimizer):
    """ LARS for PyTorch
    
    Paper: `Large batch training of Convolutional Networks` - https://arxiv.org/pdf/1708.03888.pdf
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional): learning rate (default: 1.0).
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
        trust_coeff (float): trust coefficient for computing adaptive lr / trust_ratio (default: 0.001)
        eps (float): eps for division denominator (default: 1e-8)
        trust_clip (bool): enable LARC trust ratio clipping (default: False)
        always_adapt (bool): always apply LARS LR adapt, otherwise only when group weight_decay != 0 (default: False)
    """

    def __init__(
        self,
        params,
        lr=1.0,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        trust_coeff=0.001,
        eps=1e-8,
        trust_clip=False,
        always_adapt=False,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            trust_coeff=trust_coeff,
            eps=eps,
            trust_clip=trust_clip,
            always_adapt=always_adapt,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        device = self.param_groups[0]['params'][0].device
        one_tensor = torch.tensor(1.0, device=device)  # because torch.where doesn't handle scalars correctly

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            trust_coeff = group['trust_coeff']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad

                # apply LARS LR adaptation, LARC clipping, weight decay
                # ref: https://github.com/NVIDIA/apex/blob/master/apex/parallel/LARC.py
                if weight_decay != 0 or group['always_adapt']:
                    w_norm = p.norm(2.0)
                    g_norm = grad.norm(2.0)
                    trust_ratio = trust_coeff * w_norm / (g_norm + w_norm * weight_decay + eps)
                    # FIXME nested where required since logical and/or not working in PT XLA
                    trust_ratio = torch.where(
                        w_norm > 0,
                        torch.where(g_norm > 0, trust_ratio, one_tensor),
                        one_tensor,
                    )
                    if group['trust_clip']:
                        trust_ratio = torch.minimum(trust_ratio / group['lr'], one_tensor)
                    grad.add(p, alpha=weight_decay)
                    grad.mul_(trust_ratio)

                # apply SGD update https://github.com/pytorch/pytorch/blob/1.7/torch/optim/sgd.py#L100
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(grad).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(grad, alpha=1. - dampening)
                    if nesterov:
                        grad = grad.add(buf, alpha=momentum)
                    else:
                        grad = buf

                p.add_(grad, alpha=-group['lr'])

        return loss