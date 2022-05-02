import jax
import jax.numpy as jnp
from jax import lax, jit, random
from jax.test_util import check_grads
from jax import make_jaxpr
from stochastic.rime.jax_rime import ( 
            fused_rime_sinlge_corr, 
            rime_pnts_lm_single_corr, 
            rime_pnts_wsclean_sc, 
            rime_pnts_wsclean_sc_log,
            rime_gauss_lm_single_corr,
            rime_gauss_wsclean_sc,
            rime_gauss_wsclean_sc_log
            )

from stochastic.rime.jax_1d_rime import (
    fused_rime_sinlge_corr_1d,
    rime_pnts_lm_single_corr_1d,
    rime_pnts_wsclean_sc_1d,
    rime_pnts_wsclean_sc_log_1d,
    rime_gauss_lm_single_corr_1d,
    rime_gauss_wsclean_sc_1d,
    rime_gauss_wsclean_sc_log_1d
)

@jit
def forward(params, data_uvw, data_chan_freq, kwargs):
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
    alpha = params["alpha"]

    model_vis = fused_rime_sinlge_corr(radec, data_uvw, data_chan_freq, shape_params, stokes, alpha)

    return model_vis


@jit
def forward_d_model(params, data_uvw, data_chan_freq, kwargs):
    """
    Compute the model visibilities using jax rime
    Args:
        Params (dictionary)
            flux, radec, shape parameters (ex, ey, pa), spi
        data_uvw (array)
            uvw coordinates from the measurement set
        data_chan_freq (array)
            frequencies from the measurement set
        kwag (dictionary)
            extra args
    Returns:  
        Model visibilities (array)
    """

    shape_params = params["shape_params"]
    stokes = params["stokes"]
    radec = params["radec"]
    alpha = params["alpha"]

    model_vis = fused_rime_sinlge_corr(radec, data_uvw, data_chan_freq, shape_params, stokes, alpha)

    dummy_params = kwargs["dummy_params"]
    d_shape_params = dummy_params["shape_params"]
    d_stokes = dummy_params["stokes"]
    d_radec = dummy_params["radec"]
    d_alpha = dummy_params["alpha"]

    dummy_model_vis = fused_rime_sinlge_corr(d_radec, data_uvw, data_chan_freq, d_shape_params, d_stokes, d_alpha)

    return model_vis + dummy_model_vis

@jit
def forward_d_model_col(params, data_uvw, data_chan_freq, kwargs):
    """
    Compute the model visibilities using jax rime
    Args:
        Params (dictionary)
            flux, radec, shape parameters (ex, ey, pa), spi
        data_uvw (array)
            uvw coordinates from the measurement set
        data_chan_freq (array)
            frequencies from the measurement set
        **kwargs (dictionary)
            extra args
    Returns:  
        Model visibilities (array)
    """

    shape_params = params["shape_params"]
    stokes = params["stokes"]
    radec = params["radec"]
    alpha = params["alpha"]

    model_vis = fused_rime_sinlge_corr(radec, data_uvw, data_chan_freq, shape_params, stokes, alpha)

    dummy_params = kwargs["dummy_params"]
    d_shape_params = dummy_params["shape_params"]
    d_stokes = dummy_params["stokes"]
    d_radec = dummy_params["radec"]
    d_alpha = dummy_params["alpha"]

    dummy_model_vis = fused_rime_sinlge_corr(d_radec, data_uvw, data_chan_freq, d_shape_params, d_stokes, d_alpha)

    dummy_col_vis = kwargs["dummy_col_vis"]

    return model_vis + dummy_model_vis + dummy_col_vis


@jit
def forward_d_col(params, data_uvw, data_chan_freq, kwargs):
    """
    Compute the model visibilities using jax rime
    Args:
        Params (dictionary)
            flux, radec, shape parameters (ex, ey, pa), spi
        data_uvw (array)
            uvw coordinates from the measurement set
        data_chan_freq (array)
            frequencies from the measurement set
        **kwargs (dictionary)
            extra args
    Returns:  
        Model visibilities (array)
    """

    shape_params = params["shape_params"]
    stokes = params["stokes"]
    radec = params["radec"]
    alpha = params["alpha"]

    model_vis = fused_rime_sinlge_corr(radec, data_uvw, data_chan_freq, shape_params, stokes, alpha)

    dummy_col_vis = kwargs["dummy_col_vis"]

    return model_vis + dummy_col_vis



@jit
def foward_pnts_lm_d_col(params, data_uvw, data_chan_freq, kwargs):
    """
    Compute the model visibilities using jax rime
    No beam for now. Assume we have, lm instead of radec
    Args:
        Params (dictionary)
            flux, lm, spi
        data_uvw (array)
            uvw coordinates from the measurement set
        data_chan_freq (array)
            frequencies from the measurement set
        **kwargs (dictionary)
            extra args
    Returns:  
        Model visibilities (array)
    """

    stokes = params["stokes"]
    radec = params["radec"]
    alpha = params["alpha"]

    model_vis = rime_pnts_lm_single_corr(radec, data_uvw, data_chan_freq, stokes, alpha)

    dummy_col_vis = kwargs["dummy_col_vis"]

    return model_vis + dummy_col_vis


# below are the main functions that are currently be used and set in main.py
# each for function for now will have duplicated 1d implementation where the frequncy axis is flatten

@jit
def foward_pnts_lm(params, data_uvw, data_chan_freq, kwargs):
    """
    Compute the model visibilities using jax rime
    No beam for now. Assume we have, lm instead of radec
    Args:
        Params (dictionary)
            flux, lm, spi
        data_uvw (array)
            uvw coordinates from the measurement set
        data_chan_freq (array)
            frequencies from the measurement set
        **kwargs (dictionary)
            extra args
    Returns:  
        Model visibilities (array)
    """

    stokes = params["stokes"]
    radec = params["radec"]
    alpha = params["alpha"]

    model_vis = rime_pnts_lm_single_corr(radec, data_uvw, data_chan_freq, stokes, alpha)

    return model_vis

@jit
def foward_pnts_lm_1d(params, data_uvw, data_chan_freq, kwargs):
    """
    Compute the model visibilities using jax rime
    No beam for now. Assume we have, lm instead of radec
    Args:
        Params (dictionary)
            flux, lm, spi
        data_uvw (array)
            uvw coordinates from the measurement set
        data_chan_freq (array)
            frequencies from the measurement set
        **kwargs (dictionary)
            extra args
    Returns:  
        Model visibilities (array)
    """

    stokes = params["stokes"]
    radec = params["radec"]
    alpha = params["alpha"]

    model_vis = rime_pnts_lm_single_corr_1d(radec, data_uvw, data_chan_freq, stokes, alpha)

    return model_vis

@jit
def foward_gauss_lm(params, data_uvw, data_chan_freq, kwargs):
    """
    Compute the model visibilities using jax rime
    No beam for now. Assume we have, lm instead of radec
    Args:
        Params (dictionary)
            flux, lm, spi
        data_uvw (array)
            uvw coordinates from the measurement set
        data_chan_freq (array)
            frequencies from the measurement set
        **kwargs (dictionary)
            extra args
    Returns:  
        Model visibilities (array)
    """

    stokes = params["stokes"]
    radec = params["radec"]
    alpha = params["alpha"]
    shape_params = params["shape_params"] 

    model_vis = rime_gauss_lm_single_corr(radec, data_uvw, data_chan_freq, shape_params, stokes, alpha)

    return model_vis

@jit
def foward_gauss_lm_1d(params, data_uvw, data_chan_freq, kwargs):
    """
    Compute the model visibilities using jax rime
    No beam for now. Assume we have, lm instead of radec
    Args:
        Params (dictionary)
            flux, lm, spi
        data_uvw (array)
            uvw coordinates from the measurement set
        data_chan_freq (array)
            frequencies from the measurement set
        **kwargs (dictionary)
            extra args
    Returns:  
        Model visibilities (array)
    """

    stokes = params["stokes"]
    radec = params["radec"]
    alpha = params["alpha"]
    shape_params = params["shape_params"] 

    model_vis = rime_gauss_lm_single_corr_1d(radec, data_uvw, data_chan_freq, shape_params, stokes, alpha)

    return model_vis

@jit
def foward_pnts_lm_wsclean(params, data_uvw, data_chan_freq, kwargs):
    """
    Compute the model visibilities using jax rime
    No beam for now. Assume we have, lm instead of radec
    Args:
        Params (dictionary)
            flux, lm, spi
        data_uvw (array)
            uvw coordinates from the measurement set
        data_chan_freq (array)
            frequencies from the measurement set
        **kwargs (dictionary)
            extra args
    Returns:  
        Model visibilities (array)
    """

    stokes = params["stokes"]
    radec = params["radec"]
    alpha = params["alpha"]

    model_vis = rime_pnts_wsclean_sc(radec, data_uvw, data_chan_freq, stokes, alpha)

    return model_vis

@jit
def foward_pnts_lm_wsclean_1d(params, data_uvw, data_chan_freq, kwargs):
    """
    Compute the model visibilities using jax rime
    No beam for now. Assume we have, lm instead of radec
    Args:
        Params (dictionary)
            flux, lm, spi
        data_uvw (array)
            uvw coordinates from the measurement set
        data_chan_freq (array)
            frequencies from the measurement set
        **kwargs (dictionary)
            extra args
    Returns:  
        Model visibilities (array)
    """

    stokes = params["stokes"]
    radec = params["radec"]
    alpha = params["alpha"]

    model_vis = rime_pnts_wsclean_sc_1d(radec, data_uvw, data_chan_freq, stokes, alpha)

    return model_vis


@jit
def foward_pnts_lm_wsclean_log(params, data_uvw, data_chan_freq, kwargs):
    """
    Compute the model visibilities using jax rime
    No beam for now. Assume we have, lm instead of radec
    Args:
        Params (dictionary)
            flux, lm, spi
        data_uvw (array)
            uvw coordinates from the measurement set
        data_chan_freq (array)
            frequencies from the measurement set
        **kwargs (dictionary)
            extra args
    Returns:  
        Model visibilities (array)
    """

    stokes = params["stokes"]
    radec = params["radec"]
    alpha = params["alpha"]

    model_vis = rime_pnts_wsclean_sc_log(radec, data_uvw, data_chan_freq, stokes, alpha)

    return model_vis 

@jit
def foward_pnts_lm_wsclean_log_1d(params, data_uvw, data_chan_freq, kwargs):
    """
    Compute the model visibilities using jax rime
    No beam for now. Assume we have, lm instead of radec
    Args:
        Params (dictionary)
            flux, lm, spi
        data_uvw (array)
            uvw coordinates from the measurement set
        data_chan_freq (array)
            frequencies from the measurement set
        **kwargs (dictionary)
            extra args
    Returns:  
        Model visibilities (array)
    """

    stokes = params["stokes"]
    radec = params["radec"]
    alpha = params["alpha"]

    model_vis = rime_pnts_wsclean_sc_log_1d(radec, data_uvw, data_chan_freq, stokes, alpha)

    return model_vis

@jit
def foward_gauss_lm_wsclean(params, data_uvw, data_chan_freq, kwargs):
    """
    Compute the model visibilities using jax rime
    No beam for now. Assume we have, lm instead of radec
    Args:
        Params (dictionary)
            flux, lm, spi
        data_uvw (array)
            uvw coordinates from the measurement set
        data_chan_freq (array)
            frequencies from the measurement set
        **kwargs (dictionary)
            extra args
    Returns:  
        Model visibilities (array)
    """

    stokes = params["stokes"]
    radec = params["radec"]
    alpha = params["alpha"]
    shape_params = params["shape_params"]

    model_vis = rime_gauss_wsclean_sc(radec, data_uvw, data_chan_freq, shape_params, stokes, alpha)

    return model_vis

@jit
def foward_gauss_lm_wsclean_1d(params, data_uvw, data_chan_freq, kwargs):
    """
    Compute the model visibilities using jax rime
    No beam for now. Assume we have, lm instead of radec
    Args:
        Params (dictionary)
            flux, lm, spi
        data_uvw (array)
            uvw coordinates from the measurement set
        data_chan_freq (array)
            frequencies from the measurement set
        **kwargs (dictionary)
            extra args
    Returns:  
        Model visibilities (array)
    """

    stokes = params["stokes"]
    radec = params["radec"]
    alpha = params["alpha"]
    shape_params = params["shape_params"]

    model_vis = rime_gauss_wsclean_sc_1d(radec, data_uvw, data_chan_freq, shape_params, stokes, alpha)

    return model_vis

@jit
def foward_gauss_lm_wsclean_log(params, data_uvw, data_chan_freq, kwargs):
    """
    Compute the model visibilities using jax rime
    No beam for now. Assume we have, lm instead of radec
    Args:
        Params (dictionary)
            flux, lm, spi
        data_uvw (array)
            uvw coordinates from the measurement set
        data_chan_freq (array)
            frequencies from the measurement set
        **kwargs (dictionary)
            extra args
    Returns:  
        Model visibilities (array)
    """

    stokes = params["stokes"]
    radec = params["radec"]
    alpha = params["alpha"]
    shape_params = params["shape_params"]

    model_vis = rime_gauss_wsclean_sc_log(radec, data_uvw, data_chan_freq, shape_params, stokes, alpha)

    return model_vis 

@jit
def foward_gauss_lm_wsclean_log_1d(params, data_uvw, data_chan_freq, kwargs):
    """
    Compute the model visibilities using jax rime
    No beam for now. Assume we have, lm instead of radec
    Args:
        Params (dictionary)
            flux, lm, spi
        data_uvw (array)
            uvw coordinates from the measurement set
        data_chan_freq (array)
            frequencies from the measurement set
        **kwargs (dictionary)
            extra args
    Returns:  
        Model visibilities (array)
    """

    stokes = params["stokes"]
    radec = params["radec"]
    alpha = params["alpha"]
    shape_params = params["shape_params"]

    model_vis = rime_gauss_wsclean_sc_log_1d(radec, data_uvw, data_chan_freq, shape_params, stokes, alpha)

    return model_vis 




