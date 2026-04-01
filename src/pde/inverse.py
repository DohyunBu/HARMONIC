import numpy as np
import torch
import deepxde as dde
from . import baseclass


class HeatInv(baseclass.BaseTimePDE):
    def __init__(self):
        super().__init__()
        self.output_config = [{"name": s} for s in ["u", "a"]]
        self.bbox = [-1, 1] * 2 + [0, 1]
        self.geom = dde.geometry.Rectangle(xmin=[-1, -1], xmax=[1, 1])
        timedomain = dde.geometry.TimeDomain(0, 1)
        self.geomtime = dde.geometry.GeometryXTime(self.geom, timedomain)

        def f(xyt):
            x, y, t = xyt[:, 0:1], xyt[:, 1:2], xyt[:, 2:3]
            s, c, p = torch.sin, torch.cos, np.pi
            return torch.exp(-t) * (
                (4 * p**2 - 1) * s(p * x) * s(p * y)
                + p**2 * (
                    2 * s(p * x) ** 2 * s(p * y) ** 2
                    - c(p * x) ** 2 * s(p * y) ** 2
                    - s(p * x) ** 2 * c(p * y) ** 2
                )
            )

        def u_ref(xyt):
            x, y, t = xyt[:, 0:1], xyt[:, 1:2], xyt[:, 2:3]
            return np.exp(-t) * np.sin(np.pi * x) * np.sin(np.pi * y)

        def a_ref(xyt): 
            x, y, t = xyt[:, 0:1], xyt[:, 1:2], xyt[:, 2:3]
            return 2 + np.sin(np.pi * x) * np.sin(np.pi * y)

        self.f_src = f
        self.u_ref = u_ref
        self.a_ref = a_ref

        def pde(x, ua):
            u, a = ua[:, 0:1], ua[:, 1:2]
            u_x = dde.grad.jacobian(u, x, i=0, j=0)
            u_y = dde.grad.jacobian(u, x, i=0, j=1)
            u_t = dde.grad.jacobian(u, x, i=0, j=2)
            d_au = dde.grad.jacobian(a * u_x, x, i=0, j=0) + dde.grad.jacobian(a * u_y, x, i=0, j=1)

            return u_t - d_au - f(x)

        self.pde = pde
        self.set_pdeloss(num=1)

        self.ref_sol = lambda xyt: np.concatenate((u_ref(xyt), a_ref(xyt)), axis=1)

        data_pts = np.loadtxt("ref/heatinv_points.dat")
        # fmt: off
        self.add_bcs([{
            'component': 0,
            'points': data_pts,
            'values': u_ref(data_pts) + np.random.normal(loc=0, scale=0.1, size=(2500, 1)),
            'type': 'pointset',
            'name': 'data_loss',
        }, {
            'component': 1,
            'function': a_ref,
            'bc': (lambda _, on_boundary: on_boundary),
            'type': 'dirichlet',
            'name': 'bc_a',
        }])

        self.training_points(mul=4)
