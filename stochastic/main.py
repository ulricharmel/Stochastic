import sys
import numpy as np
import time

from contextlib import ExitStack
from loguru import logger
from stochastic import configure_loguru

from stochastic.utils.utils import create_output_dirs
from stochastic.utils.parser import create_parser, init_learning_rates
from stochastic.data_handling.read_data import set_xds, load_model, MSobject
from stochastic.preprocess.skymodel_utils import best_json_to_tigger

import stochastic.opt.train as train
import stochastic.opt.optax_grads as optaxGrads
import stochastic.opt.jax_grads as jaxGrads
# from stochastic.opt.custom_grads import update as custom_grads_update 
import stochastic.rime.tools as RT

import stochastic.opt as opt

# again, this only works on startup!
from jax.config import config
config.update("jax_enable_x64", True)

def _main(exitstack):
    logger.info("Running: stochastic " + " ".join(sys.argv[1:]))
    
    parser = create_parser()
    args = parser.parse_args()
    create_output_dirs(args.outdir)
    configure_loguru(args.outdir, args.name)

    # import pdb; pdb.set_trace()
    kw = vars(args)
    logger.info('Input Options:')
    for key in kw.keys():
        logger.info('     %25s = %s' % (key, kw[key]))

    if args.efrac > 0.1:
        logger.warning("Fraction of data set to use of hessian computation too large. This may throw a segmentation fault")
    
    assert len(args.fr) == 2
    
    if args.wsclean:
        if args.log_spectra:
            jaxGrads.forward_model = opt.foward_pnts_lm_wsclean_log
            optaxGrads.forward_model = opt.foward_pnts_lm_wsclean_log
        else:
            jaxGrads.forward_model = opt.foward_pnts_lm_wsclean
            optaxGrads.forward_model = opt.foward_pnts_lm_wsclean
    else:
        jaxGrads.forward_model = opt.foward_pnts_lm
        optaxGrads.forward_model = opt.foward_pnts_lm

    # xds, data_chan_freq, phasedir = set_xds(args.msname, args.datacol, args.weightcol, 10*args.batch_size, args.one_corr, args.dummy_column, args.log_spectra, args.fr)
    
    xds = MSobject(args.msname, args.datacol, args.weightcol, args.one_corr, args.dummy_column, args.log_spectra, args.fr)
    phasedir = xds.phasedir

    RT.ra0, RT.dec0 = phasedir
    RT.freq0 = args.freq0 if args.freq0 else np.mean(xds.data_chan_freq) # data_chan_freq[0]
    RT.cellsize = args.cellsize
    RT.cx = RT.cy = args.npix/2

    LR = init_learning_rates(args.lr)

    params, d_params, nparams = load_model(args.init_model, args.dummy_model)
    
    opt_args = [args.epochs, args.delta_loss, args.delta_epoch, args.optimizer, args.name, args.report_freq, args.niter]
    # extra_args = dict(d_params=d_params, dummy_column=args.dummy_column, forwardm=forwardm)
    
    # print(minibatch)

    t0 = time.time()
    if args.svrg:
        error_fn = optaxGrads.get_hessian if args.error_func == "hessian" else optaxGrads.get_fisher
        train.train_svrg(params, xds, xds.data_chan_freq, args.batch_size, args.outdir, error_fn, LR, *opt_args, 
                                                           d_params=d_params, dummy_column=args.dummy_column, l1r=args.l1r, l2r=args.l2r, noneg=args.noneg)
    else:
        error_fn = jaxGrads.get_hessian if args.error_func == "hessian" else jaxGrads.get_fisher
        train.train(params, xds, xds.data_chan_freq, args.batch_size, args.outdir, error_fn, LR, *opt_args, 
                                                        d_params=d_params, dummy_column=args.dummy_column, l1r=args.l1r, l2r=args.l2r, noneg=args.noneg)
    
    del xds
    
    logger.info("Saving model to tigger lsm format")
    paramsfile = f"{args.outdir}/{args.name}-params_best.json"
    best_json_to_tigger(args.msname, paramsfile, nparams, args.freq0)
    
    ep_min, ep_hr = np.modf((time.time() - t0)/3600.)
    logger.success("{}hr{:0.2f}mins taken for training.".format(int(ep_hr), ep_min*60))
    
@logger.catch
def main():
    with ExitStack() as stack:
        _main(stack)
