import jax
import jax.numpy as jnp
from jax import lax, jit, random, ops
from jax.test_util import check_grads
from stochastic.opt import forward
from stochastic.essays.rime.tools import *
from loguru import logger


@jit
def loss_fn(params, data_uvw, data_chan_freq, data):
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
    Returns:  
        loss function
    """
    # import pdb; pdb.set_trace()
    model_vis = forward(params, data_uvw, data_chan_freq)
    diff = data - model_vis

    return jnp.mean(diff.real*diff.real+diff.imag*diff.imag)

#@jit
def update(params, data_uvw, data_chan_freq, data, LR):
    
    # print(make_jaxpr(jax.value_and_grad(loss_fn))(params, lm, data_uvw, data_chan_freq, data))
    
    loss, grads = jax.value_and_grad(loss_fn)(params, data_uvw, data_chan_freq, data) 
    # grads["shape_params"] = ops.index_update(grads["shape_params"], ops.index[:, 0:2], grads["shape_params"][:,0:2]*rad2arcsec)
    # grads["shape_params"] = ops.index_update(grads["shape_params"], ops.index[:, 3], grads["shape_params"][:, 3]*rad2deg)
    
    # check_grads(loss_fn, (params, lm, data_uvw, data_chan_freq, data), order=2)
    # Note that `grads` is a pytree with the same structure as `params`.
    # `jax.grad` is one of the many JAX functions that has
    # built-in support for pytrees.
    # import pdb; pdb.set_trace()
    logger.debug("init loss {},  grads, {}", grads, loss)

    # This is handy, because we can apply the SGD update using tree utils:
    return jax.tree_multimap(lambda p, g, r: p - r* g, params, grads, LR), loss