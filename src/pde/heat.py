import numpy as np
import torch
import deepxde as dde
from . import baseclass

class HeatND(baseclass.BaseTimePDE):
    def __init__(self, dim=5, T=1):
        super().__init__()
        self.output_dim = 1
        self.bbox = [-1, 1] * dim + [0, T]
        self.geom = dde.geometry.Hypersphere([0] * dim, 1)
        timedomain = dde.geometry.TimeDomain(0, T)
        self.geomtime = dde.geometry.GeometryXTime(self.geom, timedomain)

        def xnorm(xt):
            return (xt[:, :-1]**2).sum(axis=1).reshape(-1, 1)

        def pde(x, u):
            u_xx = dde.grad.hessian(u, x, i=0, j=0)
            for i in range(1, dim):
                u_xx = u_xx + dde.grad.hessian(u, x, i=i, j=i)
            u_t = dde.grad.jacobian(u, x, i=0, j=dim)

            def f(xt):
                x2, t = xnorm(xt), xt[:, -1:]
                return -1 / dim * x2 * torch.exp(x2 / 2 + t)

            return 1 / dim * u_xx + f(x) - u_t

        self.pde = pde
        self.set_pdeloss(num=1)

        def ref_sol(xt):
            x2, t = xnorm(xt), xt[:, -1:]
            return np.exp(x2 / 2 + t)

        self.ref_sol = ref_sol

        self.add_bcs([{
            'component': 0,
            'function': ref_sol,
            'bc': (lambda _, on_boundary: on_boundary),
            'type': 'neumann',
        }, {
            'component': 0,
            'function': ref_sol,
            'bc': (lambda _, on_initial: on_initial),
            'type': 'ic',
        }])

        self.training_points(mul=4)
