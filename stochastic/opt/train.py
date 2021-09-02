# Module that does all the dirty work.
# The trainging goes here.
import numpy as np
import time
import stochastic.opt.jax_grads as jaxGrads
from stochastic.utils.utils import save_output
from loguru import logger

get_iter = lambda epoch, num_batchs, batch : epoch*num_batchs + batch 

def train(params, data_uvw, data_chan_freq, data, weights, batch_size, outdir, error_fn, efrac, LR, *kwags):
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
        weights (array)
            data weights
        batch_size (int)
            number of visibilities to train in one go
        outdir (str)
            save the fitted parameters and loss function here
        error_fn (function)
            function to use to compute the hessian
        efrac (float)
            fraction of the data to use to compute the hessian
        LR (dict)
            learning rates for each parameters
    Returns:  
        fitted parameters
    """

    EPOCHS, DELTA_LOSS, DELTA_EPOCH, OPTIMIZER, prefix = kwags

    # For now we will aussume a perfect measuremnt set
    
    nsamples = data.shape[0]
    assert nsamples%batch_size == 0, "Please choose a batch size that equaly divides the number of rows"
    
    inds = np.array([(i,i+batch_size) for i in range(0, nsamples, batch_size)])
    num_batches = len(inds)
    report_batches = list(range(num_batches//10, num_batches, num_batches//10))
    best_loss, best_iter = 10000.0, 0
    loss_previous = 0
    best_model = params.copy()
    loss_avg = {}
    delta_ratio = 1.0
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
            d_uvw = data_uvw[ts:te]
            d_freq = data_chan_freq.copy()
            d_vis = data[ts:te]
            d_weights = weights[ts:te]
            iter = get_iter(epoch, num_batches, batch)
            opt_state, loss_i =  jaxGrads.update(iter, opt_state, d_uvw, d_freq, d_vis, d_weights)
            loss_avg["epoch-%d"%epoch].append(np.asarray(loss_i))
            
            if batch==0 and epoch==0:
                logger.info("Starting loss is {}", loss_i)
                loss_previous = loss_i 
            elif np.abs(loss_i - loss_previous) < DELTA_LOSS:
                DELTA_EPOCH -=1
                if DELTA_EPOCH==0:
                    break
            elif loss_i/loss_previous > delta_ratio and loss_previous < DELTA_LOSS:
                STOP_INCREASING_LOSS = True
                break
            else:
                pass

            if np.asarray(loss_i) < best_loss:
                best_loss = loss_i 
                best_model = jaxGrads.constraint_upd(opt_state)
                best_iter = iter
            
            if batch in report_batches:
                logger.info(f"Epoch {epoch}: after passing through {int(batch*100./num_batches)}% of the data loss is {loss_i}")
            
            loss_previous = loss_i
        
        mean_loss = sum(loss_avg["epoch-%d"%epoch])/len(loss_avg["epoch-%d"%epoch])

        epoch_t = time.time() - start_time
        logger.info("Epoch {} in {} secs, mean and final loss are {:0.5f} and {:0.5f}".format(epoch, epoch_t, mean_loss, loss_i))
    
        if DELTA_EPOCH == 0:
            logger.info("Early stoppage loss function not changing")
            break
        elif STOP_INCREASING_LOSS:
            logger.info("Loss value starts increasing, early stoppage to avoid over fitting!")
            break
        else:
            pass
    
    params = jaxGrads.constraint_upd(opt_state)
    # esitmate the hessian matrix with a large fraction of the data
    # samples = np.random.choice(nsamples, int(efrac*nsamples))
    # d_uvw = data_uvw[samples]
    # d_freq = data_chan_freq[samples]
    # d_vis = data[samples]
    errors = error_fn(best_model, d_uvw, d_freq, d_vis, d_weights)
    logger.debug(f"Best parameters obtained after {best_iter} iterations!")

    save_output(f"{outdir}/{prefix}-params.json", params, convert=True)
    save_output(f"{outdir}/{prefix}-loss.json", loss_avg, convert=True)
    save_output(f"{outdir}/{prefix}-params_best.json", best_model, convert=True)
    save_output(f"{outdir}/{prefix}-params_best_errors.json", errors, convert=True)
    
    return