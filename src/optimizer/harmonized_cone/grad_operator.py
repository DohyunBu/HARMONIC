# usr/bin/python3
# -*- coding: UTF-8 -*-
import torch
from typing import Optional, Sequence, Union
from .utils import *
from .length_model import *
import cdd

def extreme_rays_from_tensor(A: torch.tensor):
    A = A.detach().cpu().numpy()
    A = np.concatenate([np.zeros(A.shape[0]).reshape(-1,1), A], axis=1)
    mat = cdd.matrix_from_array(A, rep_type=cdd.RepType.INEQUALITY)
    P = cdd.polyhedron_from_matrix(mat)
    gens = cdd.copy_generators(P).array
    return torch.tensor(gens)[:,1:]

def HARMONIC_update(
    grads: Union[torch.Tensor, Sequence[torch.Tensor]],
    length_model: LengthModel = ProjectionLength(),
    losses: Optional[Sequence] = None
) -> torch.Tensor:
    if not isinstance(grads, torch.Tensor):
        grads = torch.stack(grads)
    
    with torch.no_grad():
        device = grads.device
        dtype = grads.dtype
        YT = grads /grads.norm().item()
        AT_upper = torch.eye(YT.shape[0], dtype=dtype, device=device)     
        AT_lower = YT @ YT.T     
        AT = torch.cat([AT_upper, AT_lower], dim=0)               
        
        try:
            A_rT_inv_tmp = extreme_rays_from_tensor(AT).T
            A_rT_inv_tmp = A_rT_inv_tmp.type(dtype).to(device)
        except:
            A_rT_inv_tmp = torch.linalg.pinv(AT)
            A_rT_inv_tmp = A_rT_inv_tmp.type(dtype).to(device)
        
        E = YT.T @ A_rT_inv_tmp
        E = E / (E.norm(dim=0, keepdim=True) + 1e-8)
        center = E.mean(dim=1).reshape(-1,)
        best_direction = center.reshape(-1,) 
        
        return length_model.rescale_length(target_vector=best_direction,gradients=grads,losses=losses)