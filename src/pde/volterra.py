import numpy as np
import torch
import deepxde as dde
from . import baseclass

class Volterra1D(baseclass.BasePDE):
    def __init__(self):
        super().__init__()
        self.output_dim = 2
        self.bbox = [0, 5]
        self.geom = dde.geometry.Interval(*self.bbox)

        def pde(x, y):
            u = y[:, 0:1]
            v = y[:, 1:2]
            u_x = dde.grad.jacobian(u, x, i=0, j=0)
            v_x = dde.grad.jacobian(v, x, i=0, j=0)
            return [u_x + u - v, v_x - u + v]
        
        self.pde = pde
        self.set_pdeloss(num=2)
        
        def ref_sol(x):
            x = np.asarray(x).reshape(-1, 1)
            u = np.exp(-x) * np.cosh(x)
            return u
        
        self.ref_sol = ref_sol
        self.eval_components = [0]

        def boundary_ic(x, on_initial):
            return on_initial and np.isclose(x[0], 0)
        
        self.add_bcs([{
            'component': 0,
            'function': (lambda x: 1.0),
            'bc': boundary_ic,
            'type': 'dirichlet'
        },{
            'component': 1,
            'function': (lambda x: 0.0),
            'bc': boundary_ic,
            'type': 'dirichlet'
        }])
        self.training_points()