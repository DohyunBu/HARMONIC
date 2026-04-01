# usr/bin/python3
# -*- coding: UTF-8 -*-
import torch
from typing import Sequence, Union, Tuple, Literal
import numpy as np


def get_para_vector(network: torch.nn.Module) -> torch.Tensor:
    with torch.no_grad():
        para_vec = None
        for par in network.parameters():
            viewed = par.data.view(-1)
            if para_vec is None:
                para_vec = viewed
            else:
                para_vec = torch.cat((para_vec, viewed))
        return para_vec


def get_gradient_vector(
    network: torch.nn.Module, none_grad_mode: Literal["raise", "zero", "skip"] = "skip"
) -> torch.Tensor:
    with torch.no_grad():
        grad_vec = None
        for par in network.parameters():
            if par.grad is None:
                if none_grad_mode == "raise":
                    raise RuntimeError("None gradient detected.")
                elif none_grad_mode == "zero":
                    viewed = torch.zeros_like(par.data.view(-1))
                elif none_grad_mode == "skip":
                    continue
                else:
                    raise ValueError(f"Invalid none_grad_mode '{none_grad_mode}'.")
            else:
                viewed = par.grad.data.view(-1)
            if grad_vec is None:
                grad_vec = viewed
            else:
                grad_vec = torch.cat((grad_vec, viewed))
        return grad_vec


def apply_gradient_vector(
    network: torch.nn.Module,
    grad_vec: torch.Tensor,
    none_grad_mode: Literal["zero", "skip"] = "zero",
    zero_grad_mode: Literal["skip", "pad_zero", "pad_value"] = "pad_zero",
) -> None:
    if none_grad_mode == "zero" and zero_grad_mode == "pad_value":
        apply_gradient_vector_para_based(network, grad_vec)
    with torch.no_grad():
        start = 0
        for par in network.parameters():
            if par.grad is None:
                if none_grad_mode == "skip":
                    continue
                elif none_grad_mode == "zero":
                    start = start + par.data.view(-1).shape[0]
                    if zero_grad_mode == "pad_zero":
                        par.grad = torch.zeros_like(par.data)
                    elif zero_grad_mode == "skip":
                        continue
                    else:
                        raise ValueError(f"Invalid zero_grad_mode '{zero_grad_mode}'.")
                else:
                    raise ValueError(f"Invalid none_grad_mode '{none_grad_mode}'.")
            else:
                end = start + par.data.view(-1).shape[0]
                par.grad.data = grad_vec[start:end].view(par.data.shape)
                start = end


def apply_gradient_vector_para_based(
    network: torch.nn.Module,
    grad_vec: torch.Tensor,
) -> None:
    with torch.no_grad():
        start = 0
        for par in network.parameters():
            end = start + par.data.view(-1).shape[0]
            par.grad = grad_vec[start:end].view(par.data.shape)
            start = end


def apply_para_vector(network: torch.nn.Module, para_vec: torch.Tensor) -> None:
    with torch.no_grad():
        start = 0
        for par in network.parameters():
            end = start + par.data.view(-1).shape[0]
            par.data = para_vec[start:end].view(par.data.shape)
            start = end


def get_cos_similarity(vector1: torch.Tensor, vector2: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return torch.dot(vector1, vector2) / vector1.norm() / vector2.norm()


def unit_vector(vector: torch.Tensor, warn_zero: bool = False) -> torch.Tensor:
    with torch.no_grad():
        if vector.norm() == 0:
            if warn_zero:
                print("Detected zero vector when doing normalization.")
            return torch.zeros_like(vector)
        else:
            return vector / vector.norm()


def transfer_coef_double(
    weights: torch.tensor,
    unit_vec_1: torch.tensor,
    unit_vec_2: torch.tensor,
    or_unit_vec_1: torch.tensor,
    or_unit_vec_2: torch.tensor,
) -> tuple:

    return (
        torch.dot(or_unit_vec_2, unit_vec_1)
        / (weights[0] / weights[1] * torch.dot(or_unit_vec_1, unit_vec_2)),
        1,
    )


def estimate_conflict(gradients: torch.Tensor) -> torch.Tensor:
    direct_sum = unit_vector(gradients.sum(dim=0))
    unit_grads = gradients / torch.norm(gradients, dim=1).view(-1, 1)
    return unit_grads @ direct_sum


def has_zero(lists: Sequence) -> bool:
    for i in lists:
        if i == 0:
            return True
    return False


class OrderedSliceSelector:
    def __init__(self):
        self.start_index = 0

    def select(
        self, n: int, source_sequence: Sequence
    ) -> Tuple[Sequence, Union[float, Sequence]]:
        if n > len(source_sequence):
            raise ValueError(
                "n must be less than or equal to the length of the source sequence"
            )
        end_index = self.start_index + n
        if end_index > len(source_sequence) - 1:
            new_start = end_index - len(source_sequence)
            indexes = list(range(self.start_index, len(source_sequence))) + list(
                range(0, new_start)
            )
            self.start_index = new_start
        else:
            indexes = list(range(self.start_index, end_index))
            self.start_index = end_index
        if len(indexes) == 1:
            return indexes, source_sequence[indexes[0]]
        else:
            return indexes, [source_sequence[i] for i in indexes]


class RandomSliceSelector:
    def select(
        self, n: int, source_sequence: Sequence
    ) -> Tuple[Sequence, Union[float, Sequence]]:
        assert n <= len(
            source_sequence
        ), "n can not be larger than or equal to the length of the source sequence"
        indexes = np.random.choice(len(source_sequence), n, replace=False)
        if len(indexes) == 1:
            return indexes, source_sequence[indexes[0]]
        else:
            return indexes, [source_sequence[i] for i in indexes]
