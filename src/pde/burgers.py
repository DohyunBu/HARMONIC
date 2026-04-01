import numpy as np
import deepxde as dde
from . import baseclass


class Burgers1D(baseclass.BaseTimePDE):

    def __init__(self, datapath="ref/burgers1d.dat", geom=[-1, 1], time=[0, 1], nu=0.01 / np.pi):
        super().__init__()
        self.output_dim = 1
        self.geom = dde.geometry.Interval(*geom)
        timedomain = dde.geometry.TimeDomain(*time)
        self.geomtime = dde.geometry.GeometryXTime(self.geom, timedomain)
        self.bbox = geom + time

        def burger_pde(x, u):
            u_x = dde.grad.jacobian(u, x, i=0, j=0)
            u_t = dde.grad.jacobian(u, x, i=0, j=1)
            u_xx = dde.grad.hessian(u, x, i=0, j=0)
            return u_t + u * u_x - nu * u_xx

        self.pde = burger_pde
        self.set_pdeloss()

        self.load_ref_data(datapath)
        def ic_func(x):
            return np.sin(-np.pi * x[:, 0:1])

        self.add_bcs([{
            'component': 0,
            'function': ic_func,
            'bc': (lambda _, on_initial: on_initial),
            'type': 'ic'
        }, {
            'component': 0,
            'function': (lambda _: 0),
            'bc': (lambda _, on_boundary: on_boundary),
            'type': 'dirichlet'
        }])

        self.training_points()  
