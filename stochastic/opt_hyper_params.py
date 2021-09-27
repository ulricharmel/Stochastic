"""
Hyper parameter search with optuna
"""
import sys
import numpy as np
import time

from contextlib import ExitStack
from loguru import logger
from stochastic import configure_loguru

from stochastic.utils.utils import create_output_dirs
from stochastic.utils.parser import create_parser, init_learning_rates
from stochastic.data_handling.read_data import set_xds, load_model

import stochastic.opt.train as train
import stochastic.opt.jax_grads as jaxGrads
# from stochastic.opt.custom_grads import update as custom_grads_update 
import stochastic.essays.rime.tools as RT
# again, this only works on startup!
from jax.config import config
config.update("jax_enable_x64", True)

train.NO_OPTUNA = False

import optuna
import logging

def main():
    logger.info("Running optuna for stochastic")
    parser = create_parser()
    args = parser.parse_args()
    create_output_dirs(args.outdir)
    configure_loguru(args.outdir, args.name)

    xds, data_chan_freq, phasedir = set_xds(args.msname, args.datacol, args.weightcol, -1, args.one_corr)
    RT.ra0, RT.dec0 = phasedir
    RT.freq0 = args.freq0 if args.freq0 else data_chan_freq[0] 

    params = load_model(args.init_model)
    error_fn = jaxGrads.get_hessian if args.error_func == "hessian" else jaxGrads.get_fisher

    kwags = [args.epochs, args.delta_loss, args.delta_epoch, args.optimizer, args.name]
    
    def objective(trial):

        lrs = trial.suggest_float("lrs", 1e-3, 1e-1, log=True)
        lrr = trial.suggest_float("lrr", 1e-6, 1e-4, log=True)
        lrsp = trial.suggest_float("lrsp", 1e-3, 10, log=True)
        lrspi = trial.suggest_float("lrspi", 1e-3, 1e-1, log=True)

        batch_size = 1440 #trial.suggest_int("batch", 720, 2016, log=True)

        LR = dict(stokes=lrs, radec=lrr, shape_params=lrsp, alpha=lrspi)

        best_loss = train.train(params, xds, data_chan_freq, batch_size, args.outdir, error_fn, LR, *kwags)

        return best_loss

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    optuna.logging.set_verbosity(optuna.logging.DEBUG)

    study = optuna.create_study() #direction="maximize",
        # pruner=optuna.pruners.MedianPruner(n_startup_trials=2, interval_steps=1000),
    study.optimize(objective, n_trials=20)  #, timeout=600

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    

