import jax
import jax.numpy as jnp
from jax import lax, jit, random, ops
from jax.test_util import check_grads
import optax
from optax._src import base
from jax.flatten_util import ravel_pytree

from stochastic.opt import forward
from stochastic.opt.second_order import hessian_diag, fisher_diag
from stochastic.rime.tools import *
from loguru import logger
import numpy as np

# from jax.experimental import optimizers
import stochastic.opt.optimizers as optimizers

import line_profiler
profile = line_profiler.LineProfiler()

forward_model = forward
optimizer = None
LR = None

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
    num = diff.size*2.

    l1 = jnp.vdot(diff*weights, diff).real/num  #/(2*weights.sum()

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


# TODO test different optimisers and learning rate scheduling
# Use optimizers to set optimizer initialization and update functions

def map_nested_fn(fn):
  '''Recursively apply `fn` to the key-value pairs of a nested dict'''
  def map_fn(nested_dict):
    return {k: (map_fn(v) if isinstance(v, dict) else fn(k, v))
            for k, v in nested_dict.items()}
  return map_fn

def init_optimizer(params, opt="adam", LR=dict(stokes=1e-2, lm=1e-5, alpha=1e-2)):
    global optimizer
    label_fn = map_nested_fn(lambda k, _: k)

    if opt == "adam":
        optimizer = optax.multi_transform({"stokes": optax.adam(LR["stokes"]), "lm": optax.adam(LR["lm"]), "alpha": optax.adam(LR["alpha"])}, label_fn)
        logger.info("ADAM optimizer initialised!")
    elif opt == "sgd":
        optimizer = optax.multi_transform({"stokes": optax.sgd(LR["stokes"]), "lm": optax.sgd(LR["lm"]), "alpha": optax.sgd(LR["alpha"])}, label_fn)
        logger.info("SGD optimizer initialised!")
    elif opt == "momentum":
        optimizer = optax.multi_transform({"stokes": optax.momentum(LR["stokes"]), "lm": optax.momentum(LR["lm"]), "alpha": optax.momentum(LR["alpha"])}, label_fn)
        logger.info("Momentum optimizer initialised!")
    else:
        raise NotImplementedError("Choose between adam, momentum and sgd")
    
    opt_state = optimizer.init(params)
    
    return opt_state

@jit
def optax_step(opt_state, params, data_uvw, data_chan_freq, data, weights, kwargs):
    loss_value, grads = jax.value_and_grad(loss_fn)(params, data_uvw, data_chan_freq, data, weights, kwargs)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss_value

#------------------------Add adam ontop of svrg, let see--------------------------------------#
opt_init = opt_update = get_params = None 

# TODO test different optimisers and learning rate scheduling
# Use optimizers to set optimizer initialization and update functions

def init_optimizer(opt="adam"):
    global opt_init, opt_update, get_params
    if opt == "adam":
        opt_init, opt_update, get_params = optimizers.adam(b1=0.9, b2=0.999, eps=1e-8)
        logger.info("SVRG + ADAM optimizer initialised!")
    elif opt == "sgd":
        opt_init, opt_update, get_params = optimizers.sgd()
        logger.info("SVRG + SGD optimizer initialised!")
    elif opt == "momentum":
        opt_init, opt_update, get_params = optimizers.momentum(mass=0.8)
        logger.info("SVRG + Momentum optimizer initialised!")
    else:
        raise NotImplementedError("Choose between adam, momentum and sgd")
    
    return opt_init, opt_update, get_params

@jit
def grads_updates(ups, vps, mps):
    """
    compute the corrected gradient for srvg
    Returns:
        Updated parameters, with same structure, shape and type as `params`.
    """
    return jax.tree_multimap(
      lambda u, v, m: (u - v + m), ups, vps, mps
    )

@jit
def mean_updates(params, N):

    return jax.tree_multimap(
        lambda p: p/N, params
    )


# @jit
@profile
def svrg_step(opt_info, minibatch, lr, params, data_uvw, data_chan_freq, data, weights, eps, kwargs):
    
    mgrad = jax.grad(loss_fn)(params, data_uvw, data_chan_freq, data, weights, kwargs)

    batchsize = data.shape[0]
    n_steps = batchsize//minibatch
    params_tt = params
    steps = np.random.permutation(np.array(list(range(n_steps))))
    
    flatten, func =  ravel_pytree(params)
    params_k = func(flatten*0)

    loss = []
    iter, opt_state = opt_info

    grads_= []

    # import pdb; pdb.set_trace()
    
    for ind, tt in enumerate(steps):
        data_uvw_tt = data_uvw[tt*minibatch:(tt+1)*minibatch]
        data_tt = data[tt*minibatch:(tt+1)*minibatch]
        weights_tt = weights[tt*minibatch:(tt+1)*minibatch]
        kwargs_tt = {}
        kwargs_tt["dummy_col_vis"] = None #kwargs["dummy_col_vis"][tt*minibatch:(tt+1)*minibatch]
        # kwargs_tt["dummy_params"] = kwargs["dummy_params"]
        
        loss_tt, ugrad = jax.value_and_grad(loss_fn)(params_tt, data_uvw_tt, data_chan_freq, data_tt, weights_tt, kwargs_tt)
        vgrad = jax.grad(loss_fn)(params, data_uvw_tt, data_chan_freq, data_tt, weights_tt, kwargs_tt)

        grad_tt = grads_updates(ugrad, vgrad, mgrad)

        iter = iter+1
        opt_state = opt_update(iter, LR, grad_tt, opt_state)
        params_tt = get_params(opt_state)
        
        # params_tmp = optax.apply_updates(params_tt, grad_tt)
        # loss_tt = loss_fn(params_tt, data_uvw_tt, data_chan_freq, data_tt, weights_tt, kwargs_tt)
        # if loss_tmp < loss_tt:
        #     params_tt = params_tmp
        #     # params_k = optax.apply_updates(params_k, params_tt)
    
        #     loss.append(loss_tmp)
        
        # if ind == 0:
        loss.append(loss_tt)
        grads_.append(grad_tt)

        if loss_tt < eps:
            break
    
    opt_info = (iter, opt_state)
    # mean_params = mean_updates(params_k, n_steps)

    # loss_mf = loss_fn(mean_params, data_uvw, data_chan_freq, data, weights, kwargs)

    # import pdb; pdb.set_trace()

    return opt_info, params_tt, loss, grads_

@jit
def get_hessian(params, data_uvw, data_chan_freq, data, weights, kwargs):
    """returns the standard error based on hessian of log_like hood"""
    return hessian_diag(log_likelihood, params, data_uvw, data_chan_freq, data, weights, kwargs)

@jit
def get_fisher(params, data_uvw, data_chan_freq, data, weights, kwargs):
    """returns the error using an approximation of the fisher diag of the log_like hood"""
    return fisher_diag(log_likelihood, params, data_uvw, data_chan_freq, data, weights, kwargs)
