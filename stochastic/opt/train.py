# Module that does all the dirty work.
# The trainging goes here.
from stochastic.rime.tools import radec2lm, lm2radec, pixel2radec, fraclm2radec
from stochastic.opt import forward
import numpy as np
import time
import stochastic.opt.optax_grads as optaxGrads
import stochastic.opt.jax_grads as jaxGrads
from stochastic.opt.optimizers import optimizer
from stochastic.utils.utils import save_output
from stochastic.data_handling.read_data import getbatch
from collections import OrderedDict
from jax.flatten_util import ravel_pytree

from loguru import logger

get_iter = lambda epoch, num_batchs, batch : epoch*num_batchs + batch 


def train_optax(params_radec, xds, data_chan_freq, batch_size, outdir, error_fn, LR, *opt_args, **extra_args):
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

    EPOCHS, DELTA_LOSS, DELTA_EPOCH, OPTIMIZER, prefix, REPORT_FREQ, NITER = opt_args
    dummy_params, dummy_column = extra_args["d_params"], extra_args["dummy_column"]

    params = {}
    params["stokes"] = params_radec["stokes"]
    params["lm"]  = radec2lm(params_radec["radec"])
    params["alpha"] = params_radec["alpha"]

    # For now we will aussume a perfect measuremnt set
    
    nsamples = xds.dims['row']
    # assert nsamples%batch_size == 0, "Please choose a batch size that equaly divides the number of rows"
    allindices = np.random.permutation(np.array(range(nsamples)))
    
    inds = np.array([(i,i+batch_size) for i in range(0, nsamples, batch_size)])
    num_batches = min(len(inds), NITER)
    logger.info(f"Number of batches in one epoch is {num_batches}")
    report_batches = list(range(num_batches//REPORT_FREQ, num_batches, num_batches//REPORT_FREQ))
    
    CONV = False
    STALL = False

    best_loss, best_iter = 10000.0, 0
    best_model = params.copy()

    loss_p = 0
    loss_avg = {}

    opt_state = optaxGrads.init_optimizer(params, OPTIMIZER, LR)

    for epoch in range(EPOCHS):
        start_time = time.time()
        loss_avg["epoch-%d"%epoch] = []
        arr = np.random.permutation(num_batches)
        d_inds = inds[arr]

        for batch in range(num_batches):
            ts, te = d_inds[batch]
            indices = allindices[ts:te]
            d_vis, d_weights, d_uvw, d_kwargs = getbatch(indices, xds, dummy_params, dummy_column, data_chan_freq)
            d_freq = data_chan_freq.copy()

            iter = get_iter(epoch, num_batches, batch)
            
            x0, _ = ravel_pytree(params)
            params, opt_state, loss_i = optaxGrads.optax_step(opt_state, params, d_uvw, d_freq, d_vis, d_weights, d_kwargs)
            
            loss_avg["epoch-%d"%epoch].append(np.asarray(loss_i))
            
            if batch==0 and epoch==0:
                logger.info("Starting loss is {}", loss_i)

            # check convergence
            xk, _ = ravel_pytree(params)
            eps = np.linalg.norm(xk - x0) / np.linalg.norm(xk)
            if eps < DELTA_LOSS:
                CONV = True 
                break
        
            eps = np.linalg.norm(loss_i-loss_p)/ np.linalg.norm(loss_i)
            if eps < DELTA_LOSS:
                STALL = True
                break

            if np.asarray(loss_i) < best_loss:
                best_loss = loss_i 
                best_model = params
                best_iter = iter
            
            if batch in report_batches:
                logger.info(f"Epoch {epoch}: after passing through {batch*100./num_batches:.2f}% of the data loss is {loss_i}")
            
            loss_p = loss_i
            
        mean_loss = sum(loss_avg["epoch-%d"%epoch])/len(loss_avg["epoch-%d"%epoch])

        epoch_t = time.time() - start_time
        logger.info("Epoch {} in {} secs, mean and final loss are {:.2e} and {:.2e}".format(epoch, epoch_t, mean_loss, loss_i))
    
        if CONV:
            logger.info("Parameters converge")
            break
        
        if STALL:
            logger.info("Loss stall")
            break
    

    errors = error_fn(best_model, d_uvw, d_freq, d_vis, d_weights, d_kwargs)
    logger.debug(f"Best parameters obtained after {best_iter} iterations!")

    params_radec = {}
    params_radec["stokes"] = params["stokes"]
    params_radec["radec"]  = lm2radec(params["lm"])
    params_radec["alpha"]  = params["alpha"]

    best_model_radec = {}
    best_model_radec["stokes"] = best_model["stokes"]
    best_model_radec["radec"]  = lm2radec(best_model["lm"])
    best_model_radec["alpha"]  = best_model["alpha"]

    save_output(f"{outdir}/{prefix}-params.json", params_radec, convert=True)
    save_output(f"{outdir}/{prefix}-loss.json", loss_avg, convert=True)
    save_output(f"{outdir}/{prefix}-params_best.json", best_model_radec, convert=True)
    save_output(f"{outdir}/{prefix}-params_best_errors.json", errors, convert=True)
    
    return best_loss

def train_svrg(params, xds, data_chan_freq, batch_size, outdir, error_fn, LR, *opt_args, **extra_args):
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

    EPOCHS, DELTA_LOSS, DELTA_EPOCH, OPTIMIZER, prefix, REPORT_FREQ, NITER = opt_args
    dummy_params, dummy_column = extra_args["d_params"], extra_args["dummy_column"]

    # radec = params["radec"]
    # logger.info(f"Input pos {radec}")
    params0 = params.copy()

    # params = OrderedDict()
    # params["alpha"] = params_radec["alpha"]
    # params["lm"]  = radec2lm(params_radec["radec"])
    # params["stokes"] = params_radec["stokes"]

    # For now we will aussume a perfect measuremnt set
    
    nsamples = xds.nrows # xds.dims['row']
    # assert nsamples%batch_size == 0, "Please choose a batch size that equaly divides the number of rows"
    # allindices = np.random.permutation(np.array(range(nsamples)))
    
    inds = np.array([(i,i+batch_size) for i in range(0, nsamples, batch_size)])
    num_batches = min(len(inds), NITER)
    logger.info(f"Number of batches in one epoch is {num_batches} out of {len(inds)}")
    report_batches = list(range(num_batches//REPORT_FREQ, num_batches, num_batches//REPORT_FREQ))
    
    CONV = False
    STALL = False

    best_loss, best_iter = 10000.0, 0
    best_model = params.copy()

    loss_p = 0
    loss_avg = {}

    grad_avg = {}

    # optaxGrads.LR = LR
    optaxGrads.init_optimizer(OPTIMIZER)
    opt_state = optaxGrads.opt_init(params)
    iter = 0
    opt_info = (iter, opt_state)

    minibatch = batch_size // 10
    logger.info(f"Minibatch size is {minibatch}!")

    # import pdb; pdb.set_trace()

    for epoch in range(EPOCHS):
        start_time = time.time()
        loss_avg["epoch-%d"%epoch] = []
        arr = np.random.permutation(num_batches)
        d_inds = inds[arr]

        grad_avg["epoch-%d"%epoch] = []


        for batch in range(num_batches):
            # ts, te = d_inds[batch]
            # indices = allindices[ts:te]
            # d_vis, d_weights, d_uvw, d_kwargs = getbatch(d_inds[batch], xds, dummy_params, dummy_column, data_chan_freq)
            
            d_vis, d_weights, d_uvw, d_kwargs =xds.getbatch(d_inds[batch][0], batch_size, dummy_params)
            d_kwargs["alpha_l1"] = extra_args["l1r"]
            d_kwargs["alpha_l2"] = extra_args["l2r"]
            d_kwargs["params0"] = params0
            d_kwargs["noneg"] = extra_args["noneg"]
            
            d_freq = data_chan_freq.copy()
            
            # import pdb; pdb.set_trace()
            if batch==0 and epoch==0:
                optaxGrads.LR = optaxGrads.run_power_method(params, d_uvw, d_freq, d_vis, d_weights, LR, d_kwargs)

            # iter = get_iter(epoch, num_batches, batch)
            
            x0, _ = ravel_pytree(params)
            opt_info, params, loss_values, grad_values = optaxGrads.svrg_step(opt_info, minibatch, LR, params, d_uvw, d_freq, d_vis, d_weights, DELTA_LOSS, d_kwargs)

            loss_avg["epoch-%d"%epoch].extend(np.asarray(loss_values))
            loss_i = loss_values[-1]

            grad_avg["epoch-%d"%epoch].extend(np.asarray(grad_values))
            
            # import pdb; pdb.set_trace()

            if batch==0 and epoch==0:
                logger.info(f"Starting loss {loss_values[0]} current loss {loss_i}")

            # check convergence
            xk, _ = ravel_pytree(params)
            eps = np.linalg.norm(xk - x0) / np.linalg.norm(xk)
            if eps < DELTA_LOSS:
                CONV = True 
                break
        
            eps = np.linalg.norm(loss_i-loss_p) #/ np.linalg.norm(loss_i)
            if loss_i!=0:
                if eps < DELTA_LOSS or loss_i<DELTA_LOSS:
                    STALL = True
                    break

                if np.asarray(loss_i) < best_loss:
                    best_loss = loss_i 
                    best_model = params
                    best_iter = iter
            
            if batch in report_batches:
                logger.info(f"Epoch {epoch}: after passing through {batch*100./num_batches:.2f}% of the data loss is {loss_i}")
            
            loss_p = loss_i
            
        mean_loss = sum(loss_avg["epoch-%d"%epoch])/len(loss_avg["epoch-%d"%epoch])

        epoch_t = time.time() - start_time
        logger.info("Epoch {} in {} secs, mean and final loss are {:.2e} and {:.2e}, iter {}".format(epoch, epoch_t, mean_loss, loss_i, opt_info[0]))
    
        if CONV:
            logger.info("Parameters converge")
            break
        
        if STALL:
            logger.info("Loss stall")
            break
    

    errors = error_fn(best_model, d_uvw, d_freq, d_vis, d_weights, d_kwargs)
    logger.debug(f"Best parameters obtained after {best_iter} iterations!")

    # radec = best_model["radec"]
    # stokes = best_model["stokes"]
    # spi = best_model["alpha"]
    # logger.info(f"Output pos {radec}, flux {stokes} and spi {spi}")

    params_radec = {}
    params_radec["stokes"] = params["stokes"]
    params_radec["radec"]  = pixel2radec(params["radec"])
    params_radec["alpha"]  = params["alpha"]

    best_model_radec = {}
    best_model_radec["stokes"] = best_model["stokes"]
    best_model_radec["radec"]  = pixel2radec(best_model["radec"])
    best_model_radec["alpha"]  = best_model["alpha"]

    save_output(f"{outdir}/{prefix}-params.json", params_radec, convert=True)
    save_output(f"{outdir}/{prefix}-loss.json", loss_avg, convert=True)
    save_output(f"{outdir}/{prefix}-params_best.json", best_model_radec, convert=True)
    save_output(f"{outdir}/{prefix}-params_best_errors.json", errors, convert=True)
    save_output(f"{outdir}/{prefix}-grads.json", grad_avg, grad=True)
    
    return best_loss


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

    # radec = params["radec"]
    # logger.info(f"Input pos {radec}")

    EPOCHS, DELTA_LOSS, DELTA_EPOCH, OPTIMIZER, prefix, REPORT_FREQ, NITER = opt_args
    dummy_params, dummy_column = extra_args["d_params"], extra_args["dummy_column"]

    # params = OrderedDict()
    # params["alpha"] = params_radec["alpha"]
    # params["lm"]  = radec2lm(params_radec["radec"])
    # params["stokes"] = params_radec["stokes"]
    # For now we will aussume a perfect measuremnt set
    
    nsamples = xds.nrows # xds.dims['row']
    # assert nsamples%batch_size == 0, "Please choose a batch size that equaly divides the number of rows"
    # allindices = np.random.permutation(np.array(range(nsamples)))
    
    inds = np.array([(i,i+batch_size) for i in range(0, nsamples, batch_size)])
    num_batches = min(len(inds), NITER)
    logger.info(f"Number of batches in one epoch is {num_batches} out of {len(inds)}")
    report_batches = list(range(num_batches//REPORT_FREQ, num_batches, num_batches//REPORT_FREQ))
    
    best_loss, best_iter = 10000.0, 0
    loss_p = 0
    best_model = params.copy()
    loss_avg = {}
    grad_avg= {}
    
    delta_ratio = 1.2

    CONV = False
    STALL = False
    
    jaxGrads.LR = LR
    jaxGrads.init_optimizer(OPTIMIZER)
    opt_state = jaxGrads.opt_init(params)
    
    for epoch in range(EPOCHS):
        start_time = time.time()
        loss_avg["epoch-%d"%epoch] = []
        arr = np.random.permutation(num_batches)
        d_inds = inds[arr]
        grad_avg["epoch-%d"%epoch] = []

        for batch in range(num_batches):
            # ts, te = d_inds[batch]
            # indices = allindices[ts:te]
            # d_vis, d_weights, d_uvw, d_kwargs = getbatch(d_inds[batch], xds, dummy_params, dummy_column, data_chan_freq)
            
            d_vis, d_weights, d_uvw, d_kwargs = xds.getbatch(d_inds[batch][0], batch_size, dummy_params)
            d_kwargs["alpha_l1"] = extra_args["l1r"]
            d_kwargs["alpha_l2"] = extra_args["l2r"]

            d_freq = data_chan_freq.copy()

            if batch==0 and epoch==0:
                optaxGrads.LR = optaxGrads.run_power_method(params, d_uvw, d_freq, d_vis, d_weights, LR, d_kwargs)

            x0, _ = ravel_pytree(params)
            iter = get_iter(epoch, num_batches, batch)
            opt_state, loss_i, grad_i =  jaxGrads.update(iter, opt_state, d_uvw, d_freq, d_vis, d_weights, d_kwargs)
            loss_avg["epoch-%d"%epoch].append(np.asarray(loss_i))
            grad_avg["epoch-%d"%epoch].append(grad_i)
            
            # import pdb; pdb.set_trace()
            if batch==0 and epoch==0:
                logger.info("Starting loss is {}", loss_i)

            # check convergence
            params = jaxGrads.constraint_upd(opt_state)
            xk, _ = ravel_pytree(params)
            eps = np.linalg.norm(xk - x0) / np.linalg.norm(xk)
            if eps < DELTA_LOSS:
                CONV = True 
                break
        
            eps = np.linalg.norm(loss_i-loss_p) #/ np.linalg.norm(loss_i)
            if loss_i!=0:
                if eps < DELTA_LOSS or loss_i<DELTA_LOSS:
                    STALL = True
                    break

                if np.asarray(loss_i) < best_loss:
                    best_loss = loss_i 
                    best_model = params
                    best_iter = iter
            
            if batch in report_batches:
                logger.info(f"Epoch {epoch}: after passing through {batch*100./num_batches:.2f}% of the data loss is {loss_i}")
            
            loss_p = loss_i
            
        mean_loss = sum(loss_avg["epoch-%d"%epoch])/len(loss_avg["epoch-%d"%epoch])

        epoch_t = time.time() - start_time
        logger.info("Epoch {} in {} secs, mean and final loss are {:.2e} and {:.2e}".format(epoch, epoch_t, mean_loss, loss_i))
    
        if CONV:
            logger.info("Parameters converge iterations")
            break
        
        if STALL:
            logger.info("Loss stall")
            break
    
    errors = error_fn(best_model, d_uvw, d_freq, d_vis, d_weights, d_kwargs)
    logger.debug(f"Best parameters obtained after {best_iter} iterations!")

    # radec = best_model["radec"]
    # logger.info(f"Output pos {radec}")

    params_radec = {}
    params_radec["stokes"] = params["stokes"]
    params_radec["radec"]  = pixel2radec(params["radec"])
    params_radec["alpha"]  = params["alpha"]

    best_model_radec = {}
    best_model_radec["stokes"] = best_model["stokes"]
    best_model_radec["radec"]  = pixel2radec(best_model["radec"])
    best_model_radec["alpha"]  = best_model["alpha"]

    save_output(f"{outdir}/{prefix}-params.json", params_radec, convert=True)
    save_output(f"{outdir}/{prefix}-loss.json", loss_avg, convert=True)
    save_output(f"{outdir}/{prefix}-params_best.json", best_model_radec, convert=True)
    save_output(f"{outdir}/{prefix}-params_best_errors.json", errors, convert=True)
    save_output(f"{outdir}/{prefix}-grads.json", grad_avg, grad=True)
        
    return best_loss
