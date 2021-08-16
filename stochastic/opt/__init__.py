import jax
import jax.numpy as jnp
from jax import lax, jit, random
from jax.test_util import check_grads
from jax import make_jaxpr
from stochastic.essays.rime.jax_rime import fused_rime, fused_rime_sinlge_corr

@jit
def forward(params, data_uvw, data_chan_freq):
    """
    Compute the model visibilities using jax rime
    Args:
        Params (dictionary)
            flux, radec, shape parameters (ex, ey, pa)
        data_uvw (array)
            uvw coordinates from the measurement set
        data_chan_freq (array)
            frequencies from the measurement set
    Returns:  
        Model visibilities (array)
    """

    # lm = params['lm']
    shape_params = params["shape_params"]
    stokes = params["stokes"]
    radec = params["radec"]

    model_vis = fused_rime_sinlge_corr(radec, data_uvw, data_chan_freq, shape_params, stokes)

    return model_vis

