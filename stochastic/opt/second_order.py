# Adapted from the link below.
# ==============================================================================
# Copied from https://github.com/deepmind/optax/blob/master/optax/_src/second_order.py
#===============================================================================
"""Functions for computing diagonals of Hessians & Fisher info of parameters.
Computing the Hessian or Fisher information matrices for neural networks is
typically intractible due to the quadratic memory requirements. Solving for the
diagonals of these matrices is often a better solution.
This module provides two functions for computing these diagonals, `hessian_diag`
and `fisher_diag`., each with sub-quadratic memory requirements.
"""

from typing import Any, Callable

import jax
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp

import numpy as np
from jax.numpy.linalg import norm
from functools import partial

from loguru import logger


# This covers both Jax and Numpy arrays.
# TODO(b/160876114): use the pytypes defined in Chex.
Array = jnp.ndarray
# LossFun of type f(params, inputs, targets).
LossFun = Callable[[Any, Array, Array, Array, Any, Any], Array]
ForwardFun = Callable[[Any, Array, Array, Array, Any], Array]


def ravel(p: Any) -> Array:
    return ravel_pytree(p)[0]


def hvp(
    loss: LossFun,
    v: jnp.DeviceArray,
    params: Any,
    uvw: jnp.DeviceArray,
    freq: jnp.DeviceArray,
    data:jnp.DeviceArray,
    weights: jnp.DeviceArray,
    kwargs: Any,
    # forwardm: Any
            ) -> jnp.DeviceArray:
    """Performs an efficient vector-Hessian (of `loss`) product.
    Args:
    loss: the loss function.
    v: a vector of size `ravel(params)`.
    params: model parameters.
    inputs: inputs at which `loss` is evaluated.
    targets: targets at which `loss` is evaluated.
    Returns:
    An Array corresponding to the product of `v` and the Hessian of `loss`
    evaluated with the current parameters.
    """
    _, unravel_fn = ravel_pytree(params)
    loss_fn = lambda p: loss(p, uvw, freq, data, weights, kwargs)
    return jax.jvp(jax.grad(loss_fn), [params], [unravel_fn(v)])[1]

def hvp2(
    params: Any,
    loss: LossFun,
    v: jnp.DeviceArray,
    uvw: jnp.DeviceArray,
    freq: jnp.DeviceArray,
    data:jnp.DeviceArray,
    weights: jnp.DeviceArray,
    kwargs: Any,
    # forwardm: Any
            ) -> jnp.DeviceArray:
    """Performs an efficient vector-Hessian (of `loss`) product.
    Args:
    loss: the loss function.
    v: a vector of size `ravel(params)`.
    params: model parameters.
    inputs: inputs at which `loss` is evaluated.
    targets: targets at which `loss` is evaluated.
    Returns:
    An Array corresponding to the product of `v` and the Hessian of `loss`
    evaluated with the current parameters.
    """
    _, unravel_fn = ravel_pytree(params)
    loss_fn = lambda p: loss(p, uvw, freq, data, weights, kwargs)
    return ravel(jax.jvp(jax.grad(loss_fn), [params], [unravel_fn(v)])[1])


def hessian_diag(
    loss: LossFun,
    params: Any,
    uvw: jnp.DeviceArray,
    freq: jnp.DeviceArray,
    data: jnp.DeviceArray,
    weights: jnp.DeviceArray,
    kwargs: Any,
    # forwardm: Any
            ) -> jnp.DeviceArray:
    """Computes the diagonal hessian of `loss` at (`inputs`, `targets`).
    Args:
    loss: the loss function.
    params: model parameters.
    uvw: uvw coordinates
    freq: frequency values to use
    data: visibilities
    Returns:
    A DeviceArray corresponding to the product to the Hessian of `loss`
    evaluated with the current parameters`.
    """
    pp, _= ravel_pytree(params)
    
    hess, unravel_fn  = ravel_pytree(hvp(loss, jnp.ones_like(pp), params, uvw, freq, data, weights, kwargs))
    return unravel_fn(1./jnp.sqrt(jnp.abs(hess)))



def fisher_diag(
    negative_log_likelihood: LossFun,
    params: Any,
    uvw: jnp.ndarray,
    freq: jnp.ndarray,
    data: jnp.ndarray,
    weights: jnp.ndarray,
    kwargs: Any,
    # forwardm: Any
            ) -> jnp.DeviceArray:
    """Computes the diagonal of the (observed) Fisher information matrix.
    Args:
    negative_log_likelihood: the negative log likelihood function.
    params: model parameters.
    uvw: uvw coordinates
    freq: frequencies values to use
    data: visibilities
    Returns:
    An Array corresponding to the product to the Hessian of
    `negative_log_likelihood` evaluated at `(params, inputs, targets)`.
    """
    _, unravel_fn = ravel_pytree(params)
    raveled  = jnp.square(
                    ravel(jax.grad(negative_log_likelihood)(params, uvw, freq, data, weights, kwargs)))
    
    return unravel_fn(1./jnp.sqrt(raveled))

def power_method(
        A,
        unravel_fn,
        b0=None,
        tol=1e-6,
        maxit=250,
        verbosity=1,
        report_freq=25):

    b = b0 / norm(b0)
    beta = 1.0
    eps = 1.0
    k = 0
    while eps > tol and k < maxit:
        bp = b
        b = A(unravel_fn(bp))
        bnorm = jnp.linalg.norm(b)
        betap = beta
        beta = jnp.vdot(bp, b) / jnp.vdot(bp, bp)
        b /= bnorm
        eps = np.asarray(jnp.linalg.norm(beta - betap) / betap)
        k += 1

        if not k % report_freq and verbosity > 1:
            logger.info("At iteration %i eps = %f" % (k, eps))

    if k == maxit:
        if verbosity:
            logger.info(
                "Maximum iterations reached. eps = %f, current beta = %f" %
                (eps, beta))
    else:
        if verbosity:
            logger.info(
                "Success, converged after %i iterations. beta = %f" %
                (k, beta))
    return beta, bp

def power_wrapper(
    loss: LossFun,
    params: Any,
    uvw: jnp.DeviceArray,
    freq: jnp.DeviceArray,
    data: jnp.DeviceArray,
    weights: jnp.DeviceArray,
    kwargs: Any,
    # forwardm: Any
            ) -> jnp.DeviceArray:
    """wrapper to run the power method).
    Args:
    loss: the loss function.
    params: model parameters.
    uvw: uvw coordinates
    freq: frequency values to use
    data: visibilities
    Returns:
    A DeviceArray corresponding to the product to the Hessian of `loss`
    evaluated with the current parameters`.
    """
    pp, unravel_fn = ravel_pytree(params)

    hess = partial(hvp2, loss=loss, v=jnp.ones_like(pp), uvw=uvw, freq=freq, data=data, weights=weights, kwargs=kwargs)
    
    beta, bp = power_method(hess, unravel_fn, pp)
    
    return beta, bp
