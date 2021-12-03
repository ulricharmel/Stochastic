import jax
import jax.numpy as jnp
from jax import lax, jit, random, ops
from jax.test_util import check_grads
from stochastic.opt import forward
from stochastic.opt.second_order import hessian_diag, fisher_diag
from stochastic.rime.tools import *
from loguru import logger

import optax

# from jax.experimental import optimizers
import stochastic.opt.optimizers as optimizers

LR = None
forward_model = forward

@jit
def loss_fn(params, data_uvw, data_chan_freq, data, weights, kwargs):
    """
    Compute the loss function
    Args:
        Params list with a dictionary)
            flux, lm, shape parameters (ex, ey, pa)
        data_uvw (array)
            uvw coordinates from the measurement set
        data_chan_freq (array)
            frequencies from the measurement set
        data (array)
            data visibilities
        weights(array)
            data weights
        kwargs (dictionary)
            dummy sources and column if any
        forward_model (function)
            forward model to use for the visibilities
    Returns:  
        loss function
    """
    model_vis = forward_model(params, data_uvw, data_chan_freq, kwargs)

    # import pdb; pdb.set_trace()
    diff = data - model_vis

    l1 = jnp.vdot(diff*weights, diff).real/(2*weights.sum())

    # targets = jnp.vstack((model_vis.real, model_vis.imag))
    # preds  = jnp.vstack((data.real, data.imag))
    # wei =  jnp.vstack((weights, weights))

    # l2 = jnp.sum(wei*(preds - targets)**2)/(wei.sum())

    # l3 = jnp.mean((preds-targets)**2)

    return l1

    

@jit
def log_likelihood(params, data_uvw, data_chan_freq, data, weights, kwargs):
    """
    Compute the negative log_likelihood 
    Args:
        Params list with a dictionary)
            flux, lm, shape parameters (ex, ey, pa)
        data_uvw (array)
            uvw coordinates from the measurement set
        data_chan_freq (array)
            frequencies from the measurement set
        data (array)
            data visibilities
        weights(array)
            data weights
        kwargs (dictionary)
            dummy sources and column if any
        forward_model (function)
            forward model to use for the visibilities
    Returns:  
        negative log likelihood
    """

    model_vis = forward_model(params, data_uvw, data_chan_freq, kwargs)
    diff = data - model_vis
    chi2 = jnp.vdot(diff*weights, diff).real
    loglike = chi2/2.   # + other parts omitted for now. Especially the weights not included negative change to plus

    return loglike


opt_init = opt_update = get_params = None 

# TODO test different optimisers and learning rate scheduling
# Use optimizers to set optimizer initialization and update functions

def init_optimizer(opt="adam"):
    global opt_init, opt_update, get_params
    if opt == "adam":
        opt_init, opt_update, get_params = optimizers.adam(b1=0.9, b2=0.999, eps=1e-8)
        logger.info("ADAM optimizer initialised!")
    elif opt == "sgd":
        opt_init, opt_update, get_params = optimizers.sgd()
        logger.info("SGD optimizer initialised!")
    elif opt == "momentum":
        opt_init, opt_update, get_params = optimizers.momentum(mass=0.8)
        logger.info("Momentum optimizer initialised!")
    else:
        raise NotImplementedError("Choose between adam, momentum and sgd")
    
    return opt_init, opt_update, get_params

@jit
def nonnegative_projector(x):
  return jnp.maximum(x, 0)

@jit
def constraint_upd(opt_state):
    params = get_params(opt_state)
    # params["stokes"] = ops.index_update(params["stokes"], ops.index[:,0], nonnegative_projector(params["stokes"][:,0]))
    # params["shape_params"] = ops.index_update(params["shape_params"], ops.index[:,0:2], jnp.abs(params["shape_params"][:,0:2]))

    return params

@jit
def update(i, opt_state, data_uvw, data_chan_freq, data, weights, kwargs):
    params = constraint_upd(opt_state)
    loss, grads = jax.value_and_grad(loss_fn)(params, data_uvw, data_chan_freq, data, weights, kwargs)

    # import pdb; pdb.set_trace()
    
    # logger.debug("Loss {},  grads, {}", grads, loss)

    return opt_update(i, LR, grads, opt_state), loss
@jit
def get_hessian(params, data_uvw, data_chan_freq, data, weights, kwargs):
    """returns the standard error based on hessian of log_like hood"""
    return hessian_diag(log_likelihood, params, data_uvw, data_chan_freq, data, weights, kwargs)

@jit
def get_fisher(params, data_uvw, data_chan_freq, data, weights, kwargs):
    """returns the error using an approximation of the fisher diag of the log_like hood"""
    return fisher_diag(log_likelihood, params, data_uvw, data_chan_freq, data, weights, kwargs)

