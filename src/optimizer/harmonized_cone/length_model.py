from torch import Tensor
import torch
from typing import Optional, Sequence, Union
from .utils import *


class LengthModel:
    def __init__(self):
        pass

    def get_length(
        self,
        target_vector: Optional[torch.Tensor] = None,
        unit_target_vector: Optional[torch.Tensor] = None,
        gradients: Optional[torch.Tensor] = None,
        losses: Optional[Sequence] = None,
    ) -> Union[torch.Tensor, float]:
        
        raise NotImplementedError(
            "This method must be implemented by the subclass.")

    def rescale_length(
        self,
        target_vector: torch.Tensor,
        gradients: Optional[torch.Tensor] = None,
        losses: Optional[Sequence] = None,
    ) -> torch.Tensor:

        unit_target_vector = unit_vector(target_vector)
        return (
            self.get_length(
                target_vector=target_vector,
                unit_target_vector=unit_target_vector,
                gradients=gradients,
                losses=losses,
            )
            * unit_target_vector
        )


class ProjectionLength(LengthModel):
    def __init__(self):
        super().__init__()

    def get_length(
        self,
        target_vector: Optional[torch.Tensor] = None,
        unit_target_vector: Optional[torch.Tensor] = None,
        gradients: Optional[torch.Tensor] = None,
        losses: Optional[Sequence] = None,
    ) -> torch.Tensor:
        
        assert gradients is not None, "The ProjectionLength model requires gradients information."
        if unit_target_vector is None:
            unit_target_vector = unit_vector(target_vector)
        return torch.sum(
            torch.stack([torch.dot(grad_i, unit_target_vector)
                        for grad_i in gradients])
        )