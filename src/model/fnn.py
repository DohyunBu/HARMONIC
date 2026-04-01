import torch
import torch.nn.functional as F

from deepxde import config
from deepxde.nn import NN

initializer_dict = {
    'Glorot normal': torch.nn.init.xavier_normal_,
    'Glorot uniform': torch.nn.init.xavier_uniform_,
    'He normal': torch.nn.init.kaiming_normal_,
    'He uniform': torch.nn.init.kaiming_uniform_,
    'zeros': torch.nn.init.zeros_,
}

activation_dict = {
    "elu": F.elu,
    "relu": F.relu,
    "selu": F.selu,
    "sigmoid": F.sigmoid,
    "silu": F.silu,
    "sin": torch.sin,
    "tanh": torch.tanh,
}


class FNN(NN):
    """Fully-connected neural network."""

    def __init__(self, layer_sizes, activation, kernel_initializer):
        super().__init__()
        self.activation = activation_dict[activation]
        initializer = initializer_dict[kernel_initializer]
        initializer_zero = initializer_dict['zeros']

        self.linears = torch.nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            self.linears.append(torch.nn.Linear(layer_sizes[i - 1], layer_sizes[i], dtype=config.real(torch)))
            initializer(self.linears[-1].weight)
            initializer_zero(self.linears[-1].bias)

    def forward(self, inputs):
        x = inputs
        if self._input_transform is not None:
            x = self._input_transform(x)
        for linear in self.linears[:-1]:
            x = self.activation(linear(x))
        x = self.linears[-1](x)
        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x
