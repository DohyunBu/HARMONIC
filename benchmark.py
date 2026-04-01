import argparse
import time
import os
from trainer import Trainer

os.environ["DDEBACKEND"] = "pytorch"

import numpy as np
import torch
import deepxde as dde
from src.optimizer import HARMONIC
from src.pde.burgers import Burgers1D
from src.pde.heat import HeatND
from src.pde.poisson import Poisson2D_Classic
from src.pde.wave import Wave1D
from src.pde.volterra import Volterra1D
from src.pde.inverse import HeatInv
from src.utils.args import parse_hidden_layers, parse_loss_weight
from src.utils.callbacks import TesterCallback, PlotCallback, LossCallback
from src.utils.rar import rar_wrapper

pde_list = [(Wave1D, None), (Poisson2D_Classic, None), (HeatND, None), (Volterra1D, [0]), (HeatInv, None)]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PINNBench trainer')
    parser.add_argument('--name', type=str, default="benchmark")
    parser.add_argument('--device', type=str, default="0")
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--hidden-layers', type=str, default="50*3")
    parser.add_argument('--loss-weight', type=str, default="")
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--iter', type=int, default=50000)
    parser.add_argument('--log-every', type=int, default=100)
    parser.add_argument('--plot-every', type=int, default=2000)
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--trainseed', type=str, default="[2025,9,19,98,220621]")
    parser.add_argument('--method', type=str, default="harmonic")
    
    command_args = parser.parse_args()
    seed = command_args.seed
    if seed is not None:
        dde.config.set_random_seed(seed)
    date_str = time.strftime('%m.%d-%H.%M.%S', time.localtime())
    trainer = Trainer(f"{date_str}-{command_args.name}", command_args.device)

    for pde_config in pde_list:

        def get_model_dde():
            pde = pde_config[0]()
            net = dde.nn.FNN([pde.input_dim] + parse_hidden_layers(command_args) + [pde.output_dim], "tanh", "Glorot normal")
            net = net.float()
            loss_weights = np.ones(pde.num_loss)

            opt = torch.optim.Adam(net.parameters(), command_args.lr)
            if command_args.method == "harmonic":
                opt = HARMONIC(opt, net)
            
            model = pde.create_model(net)
            model.compile(opt, loss_weights=loss_weights)

            return model


        trainer.add_task(
            get_model_dde, {
                "iterations": command_args.iter,
                "display_every": command_args.log_every,
                "callbacks": [
                    TesterCallback(log_every=command_args.log_every, components=pde_config[1]),
                    LossCallback(verbose=True),
                ]
            }
        )

    trainer.setup(__file__, seed)
    trainer.set_repeat(command_args.repeat, eval(command_args.trainseed))
    trainer.train_all()
    trainer.summary()
