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

def _main(exitstack):
    logger.info("Running: stochastic " + " ".join(sys.argv[1:]))
    parser = create_parser()
    args = parser.parse_args()
    create_output_dirs(args.outdir)
    configure_loguru(args.outdir, args.name)
    if args.efrac > 0.1:
        logger.warning("Fraction of data set to use of hessian computation too large. This may throw a segmentation fault")
    
    kwags = [args.epochs, args.delta_loss, args.delta_epoch, args.optimizer, args.name]
    
    LR = init_learning_rates(args.lr)
    
    xds, data_chan_freq, phasedir = set_xds(args.msname, args.datacol, args.weightcol, args.rowchunks, args.one_corr)
    RT.ra0, RT.dec0 = phasedir
    RT.freq0 = args.freq0 if args.freq0 else data_chan_freq[0] 

    params = load_model(args.init_model)
    error_fn = jaxGrads.get_hessian if args.error_func == "hessian" else jaxGrads.get_fisher
    
    t0 = time.time()
    train.train(params, xds, data_chan_freq, args.batch_size, args.outdir, error_fn, LR, *kwags)
    ep_min, ep_hr = np.modf((time.time() - t0)/3600.)
    logger.success("{}hr{:0.2f}mins taken for training.".format(int(ep_hr), ep_min*60))
    
@logger.catch
def main():
    with ExitStack() as stack:
        _main(stack)
