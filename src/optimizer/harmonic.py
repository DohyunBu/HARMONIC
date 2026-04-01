from .harmonized_cone.grad_operator import HARMONIC_update
from .harmonized_cone.utils import get_gradient_vector, apply_gradient_vector

import torch
from torch.optim import Optimizer
class HARMONIC(Optimizer):
    
    def __init__(self, optimizer, model):
        defaults = dict()
        super().__init__(optimizer.param_groups, defaults)
        self.optimizer = optimizer
        self.model = model
        self.device = "cuda:0"
    
    def step(self, closure):
        with torch.enable_grad():
            _ = closure(skip_backward=True)
            losses = self.losses
        grads = []
        for loss in losses:
            self.optimizer.zero_grad(set_to_none=False)
            loss.backward(retain_graph=True)
            grads.append(get_gradient_vector(self.model))
        
        g_InDO = HARMONIC_update(grads)
       
        self.zero_grad(set_to_none=False)
        apply_gradient_vector(self.model, g_InDO)
        self.optimizer.step()