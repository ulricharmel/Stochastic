# Module that does all the dirty work.
# The trainging goes here.
from stochastic.opt import forward
import numpy as np
import time
import stochastic.opt.jax_grads as jaxGrads
from stochastic.utils.utils import save_output
from stochastic.data_handling.read_data import getbatch

from loguru import logger

get_iter = lambda epoch, num_batchs, batch : epoch*num_batchs + batch 

NO_OPTUNA = True

def train(params, xds, data_chan_freq, batch_size, outdir, error_fn, LR, *opt_args, **extra_args):
    """
    Use Stochastic gradient decent and try to fit for the parameters
    Compute the loss function
    Args:
        Params list with a dictionary)
            flux, radec, shape parameters (ex, ey, pa)
        xds (Xarray dataset)
            xarray dataset to use 
        data_chan_freq (array)
            frequencies from the measurement set
        batch_size (int)
            number of visibilities to train in one go
        outdir (str)
            save the fitted parameters and loss function here
        error_fn (function)
            function to use to compute the hessian
        LR (dict)
            learning rates for each parameters
    Returns:  
        fitted parameters
    """

    EPOCHS, DELTA_LOSS, DELTA_EPOCH, OPTIMIZER, prefix, REPORT_FREQ = opt_args
    d_params, dummy_column = extra_args["d_params"], extra_args["dummy_column"]

    # For now we will aussume a perfect measuremnt set
    
    nsamples = xds.dims['row']
    # assert nsamples%batch_size == 0, "Please choose a batch size that equaly divides the number of rows"
    allindices = np.random.permutation(np.array(range(nsamples)))
    
    inds = np.array([(i,i+batch_size) for i in range(0, nsamples, batch_size)])
    num_batches = min(len(inds), 500)
    logger.info(f"Number of batches in one epoch is {num_batches}")
    report_batches = list(range(num_batches//REPORT_FREQ, num_batches, num_batches//REPORT_FREQ))
    best_loss, best_iter = 10000.0, 0
    loss_previous = 0
    best_model = params.copy()
    loss_avg = {}
    delta_ratio = 1.2
    STOP_INCREASING_LOSS = False
    
    jaxGrads.LR = LR
    jaxGrads.init_optimizer(OPTIMIZER)
    opt_state = jaxGrads.opt_init(params)
    
    for epoch in range(EPOCHS):
        start_time = time.time()
        loss_avg["epoch-%d"%epoch] = []
        arr = np.random.permutation(num_batches)
        d_inds = inds[arr]

        for batch in range(num_batches):
            ts, te = d_inds[batch]
            indices = allindices[ts:te]
            d_vis, d_weights, d_uvw, d_kwargs = getbatch(indices, xds, d_params, dummy_column)
            d_freq = data_chan_freq.copy()

            iter = get_iter(epoch, num_batches, batch)
            opt_state, loss_i =  jaxGrads.update(iter, opt_state, d_uvw, d_freq, d_vis, d_weights, d_kwargs)
            loss_avg["epoch-%d"%epoch].append(np.asarray(loss_i))
            
            if batch==0 and epoch==0:
                logger.info("Starting loss is {}", loss_i)
                loss_previous = loss_i 
            elif np.abs(loss_i - loss_previous) < DELTA_LOSS:
                DELTA_EPOCH -=1
                if DELTA_EPOCH==0:
                    break
            elif (loss_i>loss_previous and loss_previous < DELTA_LOSS):
                STOP_INCREASING_LOSS = True
                break
            else:
                pass

            if np.asarray(loss_i) < best_loss:
                best_loss = loss_i 
                best_model = jaxGrads.constraint_upd(opt_state)
                best_iter = iter
            
            if batch in report_batches:
                logger.info(f"Epoch {epoch}: after passing through {batch*100./num_batches:.2f}% of the data loss is {loss_i}")
            
            loss_previous = loss_i
        
        mean_loss = sum(loss_avg["epoch-%d"%epoch])/len(loss_avg["epoch-%d"%epoch])

        epoch_t = time.time() - start_time
        logger.info("Epoch {} in {} secs, mean and final loss are {:.2e} and {:.2e}".format(epoch, epoch_t, mean_loss, loss_i))
    
        if DELTA_EPOCH == 0:
            logger.info("Early stoppage loss function not changing")
            break
        elif STOP_INCREASING_LOSS:
            logger.info("Loss value starts increasing, early stoppage to avoid over fitting!")
            break
        else:
            pass
    
    if NO_OPTUNA:
        params = jaxGrads.constraint_upd(opt_state)
        errors = error_fn(best_model, d_uvw, d_freq, d_vis, d_weights, d_kwargs)
        logger.debug(f"Best parameters obtained after {best_iter} iterations!")

        save_output(f"{outdir}/{prefix}-params.json", params, convert=True)
        save_output(f"{outdir}/{prefix}-loss.json", loss_avg, convert=True)
        save_output(f"{outdir}/{prefix}-params_best.json", best_model, convert=True)
        save_output(f"{outdir}/{prefix}-params_best_errors.json", errors, convert=True)
        
    return best_loss