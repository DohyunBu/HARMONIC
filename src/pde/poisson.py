import numpy as np
import deepxde as dde
from . import baseclass


class Poisson2D_Classic(baseclass.BasePDE):

    def __init__(self, datapath="ref/poisson1_cg_data.dat", scale=1):
        super().__init__()
        self.output_dim = 1
        self.bbox = [-scale / 2, scale / 2, -scale / 2, scale / 2]
        self.geom = dde.geometry.Rectangle(xmin=[-scale / 2, -scale / 2], xmax=[scale / 2, scale / 2])
        circ = np.array([[0.3, 0.3, 0.1], [-0.3, 0.3, 0.1], [0.3, -0.3, 0.1], [-0.3, -0.3, 0.1]]) * scale
        for c in circ:
            disk = dde.geometry.Disk(c[0:2], c[2])
            self.geom = dde.geometry.CSGDifference(self.geom, disk)

        def pde(x, u):
            u_xx = dde.grad.hessian(u, x, i=0, j=0)
            u_yy = dde.grad.hessian(u, x, i=1, j=1)

            return [u_xx + u_yy]

        self.pde = pde
        self.set_pdeloss(num=1)

        def transform_fn(data):
            data[:, :self.input_dim] *= scale
            return data

        self.load_ref_data(datapath, transform_fn=transform_fn)

        def rec_boundary(x, on_boundary):
            return on_boundary and (
                np.isclose(x[0], self.bbox[0]) or np.isclose(x[0], self.bbox[1]) or np.isclose(x[1], self.bbox[2]) or np.isclose(x[1], self.bbox[3])
            )

        def circ_boundary(x, on_boundary):
            return on_boundary and not rec_boundary(x, on_boundary)

        self.add_bcs([{
            'component': 0,
            'function': (lambda _: 1),
            'bc': rec_boundary,
            'type': 'dirichlet'
        }, {
            'component': 0,
            'function': (lambda _: 0),
            'bc': circ_boundary,
            'type': 'dirichlet'
        }])

        self.training_points() 