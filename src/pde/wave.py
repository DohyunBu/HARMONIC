import numpy as np
import deepxde as dde
from . import baseclass


class Wave1D(baseclass.BasePDE):

    def __init__(self, C=2, bbox=[0, 1, 0, 1], scale=1, a=4):
        super().__init__()
        self.output_dim = 1
        self.bbox = [0, scale, 0, scale]
        self.geom = dde.geometry.Rectangle(xmin=[self.bbox[0], self.bbox[2]], xmax=[self.bbox[1], self.bbox[3]])

        # define PDE
        def wave_pde(x, u):
            u_xx = dde.grad.hessian(u, x, i=0, j=0)
            u_tt = dde.grad.hessian(u, x, i=1, j=1)

            return u_tt - C**2 * u_xx

        self.pde = wave_pde
        self.set_pdeloss(num=1)

        def ref_sol(x):
            x = x / scale
            return (np.sin(np.pi * x[:, 0:1]) * np.cos(2 * np.pi * x[:, 1:2]) + 0.5 * np.sin(a * np.pi * x[:, 0:1]) * np.cos(2 * a * np.pi * x[:, 1:2]))

        self.ref_sol = ref_sol

        def boundary_x0(x, on_boundary):
            return on_boundary and (np.isclose(x[0], self.bbox[0]) or np.isclose(x[0], self.bbox[1]))

        def boundary_t0(x, on_boundary):
            return on_boundary and np.isclose(x[1], self.bbox[2])

        self.add_bcs([{
            'component': 0,
            'function': (lambda _: 0),
            'bc': boundary_t0,
            'type': 'neumann'
        }, {
            'component': 0,
            'function': ref_sol,
            'bc': boundary_t0,
            'type': 'dirichlet'
        }, {
            'component': 0,
            'function': ref_sol,
            'bc': boundary_x0,
            'type': 'dirichlet'
        }])

        self.training_points()
