import jax
import jax.numpy as jnp
from jax import lax, jit, random, ops
from jax.test_util import check_grads
import optax
from optax._src import base
from jax.flatten_util import ravel_pytree

from stochastic.opt import forward
from stochastic.opt.second_order import hessian_diag, fisher_diag
from stochastic.essays.rime.tools import *
from loguru import logger

# from jax.experimental import optimizers
import stochastic.opt.optimizers as optimizers

forward_model = forward
optimizer = None

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
    diff = data - model_vis

    return jnp.vdot(diff*weights, diff).real/(2*weights.sum()) # return jnp.sum(diff.real*diff.real*weights + diff.imag*diff.imag*weights)/(2*weights.sum())

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


# TODO test different optimisers and learning rate scheduling
# Use optimizers to set optimizer initialization and update functions

def init_optimizer(params, opt="adam", learning_rate=1e-2):
    global optimizer
    if opt == "adam":
        optimizer = optax.adam(learning_rate, b1=0.9, b2=0.999, eps=1e-8)
        logger.info("ADAM optimizer initialised!")
    elif opt == "sgd":
        optimizer = optax.sgd(learning_rate)
        logger.info("SGD optimizer initialised!")
    elif opt == "momentum":
        optimizer = optax.momentum(learning_rate, mass=0.8)
        logger.info("Momentum optimizer initialised!")
    else:
        raise NotImplementedError("Choose between adam, momentum and sgd")
    
    opt_state = optimizer.init(params)
    
    return opt_state

@jax.jit
def optax_step(opt_state, params, data_uvw, data_chan_freq, data, weights, kwargs):
    loss_value, grads = jax.value_and_grad(loss_fn)(params, data_uvw, data_chan_freq, data, weights, kwargs)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss_value

@jax.jit
def grads_updates(ups, vps, mps, lr):
    """
    compute the corrected gradient for srvg
    Returns:
        Updated parameters, with same structure, shape and type as `params`.
    """
    return jax.tree_multimap(
      lambda u, v, m : -lr*(u - v + m), ups, vps, mps
    )

@jax.jit
def mean_updates(params, N):

    return jax.tree_multimap(
        lambda p: p/N, params
    )


@jax.jit
def svrg_step(minibatch, lr, params, data_uvw, data_chan_freq, data, weights, kwargs):
    
    mean_grad = jax.grad(loss_fn)(params, data_uvw, data_chan_freq, data, weights, kwargs)

    batchsize = data.shape[0]
    n_steps = batchsize//minibatch
    params_tt = params
    
    flatten, func =  ravel_pytree(params)
    params_k = func(flatten*0)

    loss = []
    
    for tt in range(0, batchsize, minibatch):
        data_uvw_tt = data_uvw[tt*minibatch:(tt+1)*minibatch]
        data_tt = data[tt*minibatch:(tt+1)*minibatch]
        weights_tt = weights[tt*minibatch:(tt+1)*minibatch]
        kwargs_tt = {}
        kwargs_tt["dummy_col_vis"] = kwargs["dummy_col_vis"][tt*minibatch:(tt+1)*minibatch]
        
        grad_tt = grads_updates(jax.grad(loss_fn)(params_tt, data_uvw_tt, data_chan_freq, data_tt, weights_tt, kwargs_tt) , 
                                jax.grad(loss_fn)(params, data_uvw_tt, data_chan_freq, data_tt, weights_tt, kwargs_tt) , mean_grad, lr)
        
        params_tt = optax.apply_updates(params_tt, grad_tt)

        params_k = optax.apply_updates(params_k, params_tt)
    
        loss.append(loss_fn(params_tt, data_uvw_tt, data_chan_freq, data_tt, weights_tt, kwargs_tt))
    
    mean_params = mean_updates(params_k, n_steps)

    return mean_params, loss

@jit
def get_hessian(params, data_uvw, data_chan_freq, data, weights, kwargs):
    """returns the standard error based on hessian of log_like hood"""
    return hessian_diag(log_likelihood, params, data_uvw, data_chan_freq, data, weights, kwargs)

@jit
def get_fisher(params, data_uvw, data_chan_freq, data, weights, kwargs):
    """returns the error using an approximation of the fisher diag of the log_like hood"""
    return fisher_diag(log_likelihood, params, data_uvw, data_chan_freq, data, weights, kwargs)
