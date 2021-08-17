import sys
import numpy as np

import time

from contextlib import ExitStack
from loguru import logger
from stochastic import configure_loguru

from stochastic.utils.utils import create_output_dirs, save_output
from stochastic.utils.parser import create_parser
from stochastic.data_handling.read_data import load_data, load_model
from stochastic.opt.jax_grads import update as jax_grads_update
from stochastic.opt.custom_grads import update as custom_grads_update 

import stochastic.essays.rime.tools as RT

# again, this only works on startup!
from jax.config import config
config.update("jax_enable_x64", True)

LEARNING_RATE = dict(stokes=1e-1, radec=0, shape_params=1e-1)
EPOCHS = 400
DELTA_LOSS = 1e-6


def train(params, data_uvw, data_chan_freq, data, batch_size, outdir, update):
    """
    Use Stochastic gradient decent and try to fit for the parameters
    Compute the loss function
    Args:
        Params list with a dictionary)
            flux, radec, shape parameters (ex, ey, pa)
        data_uvw (array)
            uvw coordinates from the measurement set
        data_chan_freq (array)
            frequencies from the measurement set
        data (array)
            data visibilities
        batch_size (int)
            number of visibilities to train in one go
        outdir (str)
            save the fitted parameters and loss function here
    Returns:  
        fitted parameters
    """

    # For now we will aussume a perfect measuremnt set

    nsamples = data.shape[0]
    assert nsamples%batch_size == 0, "Please choose a batch size that equaly divides the number of rows"
    
    inds = np.array([(i,i+batch_size) for i in range(0, nsamples, batch_size)])
    num_batches = len(inds)
    best_loss = 10000.0
    loss_previous = 0
    best_model = params.copy()
    DELTA_EPOCH = 5  

    loss_avg = {}
    for epoch in range(EPOCHS):
        start_time = time.time()
        loss_avg["epoch-%d"%epoch] = []
        arr = np.random.permutation(num_batches)
        d_inds = inds[arr]

        for batch in range(num_batches):
            ts, te = d_inds[batch]
            d_uvw = data_uvw[ts:te]
            d_freq = data_chan_freq.copy()
            d_vis = data[ts:te]
            # import pdb; pdb.set_trace()
            params, loss_i =  update(params, d_uvw, d_freq, d_vis, LEARNING_RATE)
            loss_avg["epoch-%d"%epoch].append(np.asarray(loss_i))
            if batch==0 and epoch==0:
                logger.info("Starting loss is {}", loss_i)

            if np.asarray(loss_i) < best_loss:
                best_loss = loss_i 
                best_model = params.copy()
        
        mean_loss = sum(loss_avg["epoch-%d"%epoch])/len(loss_avg["epoch-%d"%epoch])
        if epoch==0:
            loss_previous = mean_loss
        else:
            if np.abs(mean_loss - loss_previous) < DELTA_LOSS:
                DELTA_EPOCH -= 1

        epoch_time = time.time() - start_time
        logger.info("Epoch {} in {:0.3f} sec and loss is {:0.5f}".format(epoch, epoch_time, mean_loss))
    
        if DELTA_EPOCH == 0:
            logger.info("Early stoppage loss function not changing")
            break
    
    save_output(outdir+"/params.json", params, convert=True)
    save_output(outdir+"/params_best.json", best_model, convert=True)
    save_output(outdir+"/loss.json", loss_avg, convert=True)

    return

def _main(exitstack):
    
    configure_loguru()
    parser = create_parser()
    args = parser.parse_args()
    create_output_dirs(args.outdir)
    update = custom_grads_update if args.custom_grads else jax_grads_update
    data_vis, data_uvw, data_chan_freq, phasedir = load_data(args.msname, args.datacol, args.one_corr)
    RT.ra0, RT.dec0 = phasedir
    params = load_model(args.init_model)
    t0 = time.time()
    train(params, data_uvw, data_chan_freq, data_vis, args.batch_size, args.outdir, update)
    logger.success("{:.2f} seconds taken for training.", time.time() - t0)

@logger.catch
def main():
    with ExitStack() as stack:
        _main(stack)
