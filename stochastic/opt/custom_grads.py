import jax
import jax.numpy as jnp
from jax import lax, jit, random, custom_jvp, ops
from jax.test_util import check_grads
from stochastic.opt import forward
from stochastic.rime.tools import *
from stochastic.rime.jax_rime import phase_delay, gaussian
from loguru import logger

@custom_jvp
def loss_g(params, data_uvw, data_chan_freq, data):
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

@jit
def loss_g_div(params, params_dot, uvw, freq, data):
    source = params["stokes"].shape[0]
    shape_params = params["shape_params"]
    lm = radec2lm(params["radec"]) 
    radec_d = radec2lm_deriv(params["radec"])

    model = forward(params, uvw, freq)
    diff = data - model
    vis_out = 0.

    shape_arcsec = jnp.empty_like(shape_params)
    shape_arcsec = ops.index_update(shape_arcsec, ops.index[:, 0:2], shape_params[:,0:2]*arcsec2rad)
    shape_arcsec = ops.index_update(shape_arcsec, ops.index[:, 2], shape_params[:,2]*deg2rad)

    for s in range(source):
        source_delay = phase_delay(lm[s:s+1], uvw, freq)
        gauss_shape = gaussian(uvw, freq, shape_arcsec[s:s+1])
        delay_diff = jnp.einsum('rfc, crf, crf->rfc', diff, gauss_shape, source_delay)

        # stokes derivative 
        vis_out += -2*jnp.mean(delay_diff.real+delay_diff.imag)*params_dot["stokes"][s,0] # remove factor of 2

        # lm derivative - > radec derivatives 
        dphase_l, dphase_m = phase_delay_deriv_lm(lm[s:s+1], uvw, freq)

        dl_diff = jnp.einsum('rfc, crf->rfc', delay_diff, dphase_l)
        vis_out += -2*params["stokes"][s,0]*jnp.mean(dl_diff.real+dl_diff.imag)*params_dot["radec"][s,0]*radec_d[s,0]
        vis_out += -2*params["stokes"][s,0]*jnp.mean(dl_diff.real+dl_diff.imag)*params_dot["radec"][s,1]*radec_d[s,1]

        dm_diff = jnp.einsum('rfc, crf->rfc', delay_diff, dphase_m)
        vis_out += -2*params["stokes"][s,0]*jnp.mean(dm_diff.real+dm_diff.imag)*params_dot["radec"][s,0]*radec_d[s,2]
        vis_out += -2*params["stokes"][s,0]*jnp.mean(dm_diff.real+dm_diff.imag)*params_dot["radec"][s,1]*radec_d[s,3]

        #shape derivative
        d_emin, d_emax, d_pa = gaussian_shape_deriv(uvw, freq, shape_arcsec[s:s+1])

        demin_diff = jnp.einsum('rfc, crf->rfc', delay_diff, d_emin)
        vis_out += 2*params["stokes"][s,0]*jnp.mean(demin_diff.real+demin_diff.imag)*params_dot["shape_params"][s,0]*arcsec2rad

        demax_diff = jnp.einsum('rfc, crf->rfc', delay_diff, d_emax)
        vis_out += 2*params["stokes"][s,0]*jnp.mean(demax_diff.real+demax_diff.imag)*params_dot["shape_params"][s,1]*arcsec2rad

        dpa_diff = jnp.einsum('rfc, crf->rfc', delay_diff, d_pa)
        vis_out += 2*params["stokes"][s,0]*jnp.mean(dpa_diff.real+dpa_diff.imag)*params_dot["shape_params"][s,2]*deg2rad

    return vis_out

@loss_g.defjvp
def loss_g_jvp(primals, tangents):
    params, uvw, freq, data = primals
    params_dot  = tangents[0]
    primal_out = loss_g(params, uvw, freq, data)
    tangent_out = loss_g_div(params, params_dot, uvw, freq, data)

    return primal_out, tangent_out

#@jit
def update(params, data_uvw, data_chan_freq, data, LR):
    
    loss, grads = jax.value_and_grad(loss_g)(params, data_uvw, data_chan_freq, data)
    
    # check_grads(loss_fn, (params, lm, data_uvw, data_chan_freq, data), order=2)
    # Note that `grads` is a pytree with the same structure as `params`.
    # `jax.grad` is one of the many JAX functions that has
    # built-in support for pytrees.
    # import pdb; pdb.set_trace()
    logger.debug("init loss {},  grads, {}", grads, loss)

    # This is handy, because we can apply the SGD update using tree utils:
    return jax.tree_multimap(lambda p, g, r: p - r* g, params, grads, LR), loss

