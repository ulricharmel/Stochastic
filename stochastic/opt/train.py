# Module that does all the dirty work.
# The trainging goes here.
import numpy as np
import time
import stochastic.opt.jax_grads as jaxGrads
from stochastic.utils.utils import save_output
from loguru import logger

_get_iter = lambda epoch, num_batchs, batch : epoch*num_batchs + batch 

def train(params, data_uvw, data_chan_freq, data, batch_size, outdir, error_fn, efrac, LR, *kwags):
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
        error_fn (function)
            function to use to compute the hessian
        efrac (float)
            fraction of the data to use to compute the hessian
        LR (dict)
            learning rates for each parameters
    Returns:  
        fitted parameters
    """

    EPOCHS, DELTA_LOSS, DELTA_EPOCH, OPTIMIZER = kwags

    # For now we will aussume a perfect measuremnt set
    
    nsamples = data.shape[0]
    assert nsamples%batch_size == 0, "Please choose a batch size that equaly divides the number of rows"
    
    inds = np.array([(i,i+batch_size) for i in range(0, nsamples, batch_size)])
    num_batches = len(inds)
    best_loss = 10000.0
    loss_previous = 0
    best_model = params.copy()
    loss_avg = {}
    delta_ratio = 100.0
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
            opt_state, loss_i =  jaxGrads.update(_get_iter(epoch, num_batches, batch), opt_state, d_uvw, d_freq, d_vis)
            loss_avg["epoch-%d"%epoch].append(np.asarray(loss_i))
            
            if batch==0 and epoch==0:
                logger.info("Starting loss is {}", loss_i)

            if np.asarray(loss_i) < best_loss:
                best_loss = loss_i 
                best_model = jaxGrads.constraint_upd(opt_state)
        
        mean_loss = sum(loss_avg["epoch-%d"%epoch])/len(loss_avg["epoch-%d"%epoch])
        if epoch==0:
            loss_previous = mean_loss
        elif np.abs(mean_loss - loss_previous) < DELTA_LOSS:
            DELTA_EPOCH -=1
        elif np.abs(mean_loss - loss_previous)/loss_previous > delta_ratio:
            STOP_INCREASING_LOSS = True
        else:
            pass

        loss_previous = mean_loss
        epoch_time = time.time() - start_time
        logger.info("Epoch {} in {:0.3f} sec and loss is {:0.5f}".format(epoch, epoch_time, mean_loss))
    
        if DELTA_EPOCH == 0:
            logger.info("Early stoppage loss function not changing")
            break
        elif STOP_INCREASING_LOSS:
            logger.info("Loss value increased significantly, early stoppage to avoid over fitting!")
            break
        else:
            pass
    
    # params = jaxGrads.constraint_upd(opt_state)
    # esitmate the hessian matrix with a large fraction of the data
    # samples = np.random.choice(nsamples, int(efrac*nsamples))
    # d_uvw = data_uvw[samples]
    # d_freq = data_chan_freq[samples]
    # d_vis = data[samples]
    errors = error_fn(best_model, d_uvw, d_freq, d_vis)

    save_output(outdir+"/params.json", best_model, convert=True)
    save_output(outdir+"/loss.json", loss_avg, convert=True)
    save_output(outdir+"/params_errors.json", errors, convert=True)
    
    return