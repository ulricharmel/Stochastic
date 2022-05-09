from functools import partial
import jax.numpy as jnp
from jax import jit, vmap
from jax import lax, ops
from jax.experimental import loops

from stochastic.rime.tools import *

from jax.config import config
config.update("jax_enable_x64", True)


@jit
def gaussian_1d(uvw, frequency, shape_params):
    # https://en.wikipedia.org/wiki/Full_width_at_half_maximum

    fwhm = 2. * jnp.sqrt(2. * jnp.log(2.))
    fwhminv = 1. / fwhm
    gauss_scale = fwhminv * jnp.sqrt(2.) * jnp.pi / lightspeed

    dtype = jnp.result_type(*(jnp.dtype(a.dtype.name) for
                             a in (uvw, frequency, shape_params)))

    nsrc = shape_params.shape[0]
    nrow = uvw.shape[0]
    # nchan = frequency.shape[0]

    scaled_freq = frequency*gauss_scale


    with loops.Scope() as l2:
        l2.shape = jnp.empty((nsrc, nrow), dtype=dtype)

        l2.emaj, l2.emin, l2.angle = jnp.array([0, 0, 0], dtype=shape_params.dtype)
        l2.el, l2.em, l2.er = jnp.array([0, 0, 0], dtype=shape_params.dtype)

        l2.u, l2.v, l2.w = jnp.array([0, 0, 0], dtype=uvw.dtype)

        l2.u1, l2.v1, l2.fu1, l2.fv1 = jnp.array([0, 0, 0, 0], dtype=dtype)

        for s in l2.range(shape_params.shape[0]):
            l2.emaj, l2.emin, l2.angle = shape_params[s]

            # Convert to l-projection, m-projection, ratio
            l2.el = l2.emaj * jnp.sin(l2.angle)
            l2.em = l2.emaj * jnp.cos(l2.angle)
            l2.er = lax.cond(l2.emaj == 0., lambda _: l2.emin/1., lambda _: l2.emin/l2.emaj, operand=None)

            for r in l2.range(uvw.shape[0]):
                l2.u, l2.v, l2.w = uvw[r]

                l2.u1 = (l2.u*l2.em - l2.v*l2.el)*l2.er
                l2.v1 = l2.u*l2.el + l2.v*l2.em

                l2.fu1 = l2.u1*scaled_freq[r]
                l2.fv1 = l2.v1*scaled_freq[r]
                l2.shape = ops.index_update(l2.shape, (s, r), jnp.exp(-(l2.fu1*l2.fu1 + l2.fv1*l2.fv1)))

    return l2.shape


@jit
def phase_delay_1d(lm, uvw, frequency):
    out_dtype = jnp.result_type(lm, uvw, frequency, jnp.complex64)

    one = lm.dtype.type(1.0)
    neg_two_pi_over_c = lm.dtype.type(minus_two_pi_over_c)
    complex_one = out_dtype.type(1j)

    l = lm[:, 0, None]  # noqa
    m = lm[:, 1, None]

    u = uvw[None, :, 0]
    v = uvw[None, :, 1]
    w = uvw[None, :, 2]

    n = jnp.sqrt(one - l**2 - m**2) - one

    real_phase = (neg_two_pi_over_c *
                  (l * u + m * v + n * w) *
                  frequency[None, :])

    return jnp.exp(complex_one*real_phase)

@jit
def brightness(stokes):
    return jnp.stack([
        stokes[:, 0] + stokes[:, 3],
        stokes[:, 1] + stokes[:, 2]*1j,
        stokes[:, 1] - stokes[:, 2]*1j,
        stokes[:, 0] - stokes[:, 3]],
        axis=1)

@jit
def dummy_brightness():
    return jnp.stack([jnp.array([1.]), jnp.array([0.]), jnp.array([0.]), jnp.array([1.])], axis=1)

@jit
def dummy_brightness_sc():
    return jnp.array([[1.]])

@jit 
def fused_wsclean_rime_1d(radec, uvw, frequency, shape_params, stokes, alpha):

    lm = radec2lm(radec)
    source = lm.shape[0]
    row = uvw.shape[0]
    # chan = frequency.shape[0]
    corr = stokes.shape[1]
    
    shape_arcsec = jnp.empty_like(shape_params)
    shape_arcsec = ops.index_update(shape_arcsec, ops.index[:, 0:2], shape_params[:,0:2]*arcsec2rad)
    shape_arcsec = ops.index_update(shape_arcsec, ops.index[:, 2], shape_params[:,2]*deg2rad)

    dtype = jnp.result_type(lm.dtype, uvw.dtype,
                            frequency.dtype, shape_params.dtype, stokes.dtype,
                            jnp.complex64)
    vis = jnp.empty((row, corr), dtype)

    def body(s, vis):
        phdelay = phase_delay_1d(lm[None, s], uvw, frequency)
        brness = dummy_brightness()
        spectrum = wsclean_spectra(stokes[s, 0], alpha[s], frequency)
        gauss_shape = gaussian_1d(uvw, frequency, shape_arcsec[None, s])
        coh = jnp.einsum("sr,sr,si,sr->ri",
                         phdelay,
                         gauss_shape,
                         brness,
                         spectrum)

        return vis + coh.astype(dtype)

    return lax.fori_loop(0, source, body, vis)

@jit 
def fused_wsclean_log_rime_1d(radec, uvw, frequency, shape_params, stokes, alpha):

    lm = radec2lm(radec)
    source = lm.shape[0]
    row = uvw.shape[0]
    # chan = frequency.shape[0]
    corr = stokes.shape[1]
    
    shape_arcsec = jnp.empty_like(shape_params)
    shape_arcsec = ops.index_update(shape_arcsec, ops.index[:, 0:2], shape_params[:,0:2]*arcsec2rad)
    shape_arcsec = ops.index_update(shape_arcsec, ops.index[:, 2], shape_params[:,2]*deg2rad)

    dtype = jnp.result_type(lm.dtype, uvw.dtype,
                            frequency.dtype, shape_params.dtype, stokes.dtype,
                            jnp.complex64)
    vis = jnp.empty((row, corr), dtype)

    def body(s, vis):
        phdelay = phase_delay_1d(lm[None, s], uvw, frequency)
        brness = dummy_brightness()
        spectrum = wsclean_log_spectra(stokes[s, 0], alpha[s], frequency)
        gauss_shape = gaussian_1d(uvw, frequency, shape_arcsec[None, s])
        coh = jnp.einsum("sr,sr,si,sr->ri",
                         phdelay,
                         gauss_shape,
                         brness,
                         spectrum)

        return vis + coh.astype(dtype)

    return lax.fori_loop(0, source, body, vis)

@jit 
def fused_wsclean_rime_sc_1d(radec, uvw, frequency, shape_params, stokes, alpha):

    lm = radec2lm(radec)
    source = lm.shape[0]
    row = uvw.shape[0]
    # chan = frequency.shape[0]
    corr = stokes.shape[1]
    
    shape_arcsec = jnp.empty_like(shape_params)
    shape_arcsec = ops.index_update(shape_arcsec, ops.index[:, 0:2], shape_params[:,0:2]*arcsec2rad)
    shape_arcsec = ops.index_update(shape_arcsec, ops.index[:, 2], shape_params[:,2]*deg2rad)

    dtype = jnp.result_type(lm.dtype, uvw.dtype,
                            frequency.dtype, shape_params.dtype, stokes.dtype,
                            jnp.complex64)
    vis = jnp.empty((row, corr), dtype)

    def body(s, vis):
        phdelay = phase_delay_1d(lm[None, s], uvw, frequency)
        brness = dummy_brightness_sc()
        spectrum = wsclean_spectra(stokes[s, 0], alpha[s], frequency)
        gauss_shape = gaussian_1d(uvw, frequency, shape_arcsec[None, s])
        coh = jnp.einsum("sr,sr,si,sr->ri",
                         phdelay,
                         gauss_shape,
                         brness,
                         spectrum)

        return vis + coh.astype(dtype)

    return lax.fori_loop(0, source, body, vis)

@jit 
def fused_wsclean_log_rime_sc_1d(radec, uvw, frequency, shape_params, stokes, alpha):

    lm = radec2lm(radec)
    source = lm.shape[0]
    row = uvw.shape[0]
    # chan = frequency.shape[0]
    corr = stokes.shape[1]
    
    shape_arcsec = jnp.empty_like(shape_params)
    shape_arcsec = ops.index_update(shape_arcsec, ops.index[:, 0:2], shape_params[:,0:2]*arcsec2rad)
    shape_arcsec = ops.index_update(shape_arcsec, ops.index[:, 2], shape_params[:,2]*deg2rad)

    dtype = jnp.result_type(lm.dtype, uvw.dtype,
                            frequency.dtype, shape_params.dtype, stokes.dtype,
                            jnp.complex64)
    vis = jnp.empty((row, corr), dtype)

    def body(s, vis):
        phdelay = phase_delay_1d(lm[None, s], uvw, frequency)
        brness = dummy_brightness_sc()
        spectrum = wsclean_log_spectra(stokes[s, 0], alpha[s], frequency)
        gauss_shape = gaussian_1d(uvw, frequency, shape_arcsec[None, s])
        coh = jnp.einsum("sr,sr,si,sr->ri",
                         phdelay,
                         gauss_shape,
                         brness,
                         spectrum)

        return vis + coh.astype(dtype)

    return lax.fori_loop(0, source, body, vis)


@jit 
def rime_gauss_wsclean_sc_1d(radec, uvw, frequency, shape_params, stokes, alpha):

    lm = pixel2lm(radec)
    source = lm.shape[0]
    row = uvw.shape[0]
    # chan = frequency.shape[0]
    corr = stokes.shape[1]
    
    shape_arcsec = jnp.empty_like(shape_params)
    shape_arcsec = ops.index_update(shape_arcsec, ops.index[:, 0:2], shape_params[:,0:2]*arcsec2rad)
    shape_arcsec = ops.index_update(shape_arcsec, ops.index[:, 2], shape_params[:,2]*deg2rad)

    dtype = jnp.result_type(lm.dtype, uvw.dtype,
                            frequency.dtype, shape_params.dtype, stokes.dtype,
                            jnp.complex64)
    vis = jnp.empty((row, corr), dtype)

    def body(s, vis):
        phdelay = phase_delay_1d(lm[None, s], uvw, frequency)
        brness = dummy_brightness_sc()
        spectrum = wsclean_spectra(stokes[s, 0], alpha[s], frequency)
        gauss_shape = gaussian_1d(uvw, frequency, shape_arcsec[None, s])
        coh = jnp.einsum("sr,sr,si,sr->ri",
                         phdelay,
                         gauss_shape,
                         brness,
                         spectrum)

        return vis + coh.astype(dtype)

    return lax.fori_loop(0, source, body, vis)

@jit 
def rime_gauss_wsclean_sc_log_1d(radec, uvw, frequency, shape_params, stokes, alpha):

    lm = pixel2lm(radec)
    source = lm.shape[0]
    row = uvw.shape[0]
    # chan = frequency.shape[0]
    corr = stokes.shape[1]
    
    shape_arcsec = jnp.empty_like(shape_params)
    shape_arcsec = ops.index_update(shape_arcsec, ops.index[:, 0:2], shape_params[:,0:2]*arcsec2rad)
    shape_arcsec = ops.index_update(shape_arcsec, ops.index[:, 2], shape_params[:,2]*deg2rad)

    dtype = jnp.result_type(lm.dtype, uvw.dtype,
                            frequency.dtype, shape_params.dtype, stokes.dtype,
                            jnp.complex64)
    vis = jnp.empty((row, corr), dtype)

    def body(s, vis):
        phdelay = phase_delay_1d(lm[None, s], uvw, frequency)
        brness = dummy_brightness_sc()
        spectrum = wsclean_log_spectra(stokes[s, 0], alpha[s], frequency)
        gauss_shape = gaussian_1d(uvw, frequency, shape_arcsec[None, s])
        coh = jnp.einsum("sr,sr,si,sr->ri",
                         phdelay,
                         gauss_shape,
                         brness,
                         spectrum)

        return vis + coh.astype(dtype)

    return lax.fori_loop(0, source, body, vis)

@jit 
def rime_pnts_wsclean_sc_1d(radec, uvw, frequency, stokes, alpha):

    lm = pixel2lm(radec) #radec2lm(radec) frac2lm
    source = lm.shape[0]
    row = uvw.shape[0]
    chan = frequency.shape[0]
    corr = stokes.shape[1]
    
    dtype = jnp.result_type(lm.dtype, uvw.dtype,
                            frequency.dtype, stokes.dtype,
                            jnp.complex64)
    vis = jnp.empty((row, corr), dtype)

    def body(s, vis):
        phdelay = phase_delay_1d(lm[None, s], uvw, frequency)
        brness = dummy_brightness_sc()
        spectrum = wsclean_spectra(stokes[s, 0], alpha[s], frequency)
        
        coh = jnp.einsum("sr,si,sr->ri",
                         phdelay,
                         brness,
                         spectrum)

        return vis + coh.astype(dtype)

    return lax.fori_loop(0, source, body, vis)

@jit
def rime_pnts_lm_single_corr_1d(radec, uvw, frequency, stokes, alpha):

    lm = pixel2lm(radec) # radec2lm(radec) frac2lm
    source = lm.shape[0]
    row = uvw.shape[0]
    # chan = frequency.shape[0]
    corr = stokes.shape[1]

    dtype = jnp.result_type(lm.dtype, uvw.dtype,
                            frequency.dtype, stokes.dtype,
                            jnp.complex64)
    vis = jnp.empty((row, corr), dtype)

    def body(s, vis):
        phdelay = phase_delay_1d(lm[None, s], uvw, frequency)
        brness = stokes[None, s]
        spectrum = source_spectrum(alpha[s], frequency)

        coh = jnp.einsum("sr,si,sr->ri",
                         phdelay,
                         brness,
                         spectrum)

        return vis + coh.astype(dtype)
    
    return lax.fori_loop(0, source, body, vis)

@jit 
def rime_pnts_wsclean_sc_log_1d(radec, uvw, frequency, stokes, alpha):

    lm = pixel2lm(radec) # radec2lm(radec) pixel2lm frac2lm
    source = lm.shape[0]
    row = uvw.shape[0]
    # chan = frequency.shape[0]
    corr = stokes.shape[1]
    
    dtype = jnp.result_type(lm.dtype, uvw.dtype,
                            frequency.dtype, stokes.dtype,
                            jnp.complex64)
    vis = jnp.empty((row, corr), dtype)

    def body(s, vis):
        phdelay = phase_delay_1d(lm[None, s], uvw, frequency)
        brness = dummy_brightness_sc()
        spectrum = wsclean_log_spectra(stokes[s, 0], alpha[s], frequency)
        
        coh = jnp.einsum("sr,si,sr->ri",
                         phdelay,
                         brness,
                         spectrum)

        return vis + coh.astype(dtype)

    return lax.fori_loop(0, source, body, vis)

@jit
def fused_rime_1d(radec, uvw, frequency, shape_params, stokes, alpha):

    # Full expansion over source axis -- very memory hungry

    # return jnp.einsum("srf,si->rfi",
    #                   phase_delay(lm, uvw, frequency),
    #                   brightness(stokes))

    lm = radec2lm(radec)
    source = lm.shape[0]
    row = uvw.shape[0]
    # chan = frequency.shape[0]
    corr = stokes.shape[1]
    
    shape_arcsec = jnp.empty_like(shape_params)
    shape_arcsec = ops.index_update(shape_arcsec, ops.index[:, 0:2], shape_params[:,0:2]*arcsec2rad)
    shape_arcsec = ops.index_update(shape_arcsec, ops.index[:, 2], shape_params[:,2]*deg2rad)

    dtype = jnp.result_type(lm.dtype, uvw.dtype,
                            frequency.dtype, shape_params.dtype, stokes.dtype,
                            jnp.complex64)
    vis = jnp.empty((row, corr), dtype)

    def body(s, vis):
        phdelay = phase_delay_1d(lm[None, s], uvw, frequency)
        brness = brightness(stokes[None, s])
        spectrum = source_spectrum(alpha[s], frequency)
    
        gauss_shape = gaussian_1d(uvw, frequency, shape_arcsec[None, s])
        coh = jnp.einsum("sr,sr,si,sr->ri",
                         phdelay,
                         gauss_shape,
                         brness,
                         spectrum)

        return vis + coh.astype(dtype)

    return lax.fori_loop(0, source, body, vis)

@jit
def fused_rime_sinlge_corr_1d(radec, uvw, frequency, shape_params, stokes, alpha):

    # Full expansion over source axis -- very memory hungry

    # return jnp.einsum("srf,si->rfi",
    #                   phase_delay(lm, uvw, frequency),
    #                   brightness(stokes))

    lm = radec2lm(radec)
    source = lm.shape[0]
    row = uvw.shape[0]
    # chan = frequency.shape[0]
    corr = stokes.shape[1]
    
    shape_arcsec = jnp.empty_like(shape_params)
    shape_arcsec = ops.index_update(shape_arcsec, ops.index[:, 0:2], shape_params[:,0:2]*arcsec2rad)
    shape_arcsec = ops.index_update(shape_arcsec, ops.index[:, 2], shape_params[:,2]*deg2rad)

    dtype = jnp.result_type(lm.dtype, uvw.dtype,
                            frequency.dtype, shape_params.dtype, stokes.dtype,
                            jnp.complex64)
    vis = jnp.empty((row, corr), dtype)

    def body(s, vis):
        phdelay = phase_delay_1d(lm[None, s], uvw, frequency)
        brness = stokes[None, s]
        spectrum = source_spectrum(alpha[s], frequency)
        gauss_shape = gaussian_1d(uvw, frequency, shape_arcsec[None, s])
        coh = jnp.einsum("sr,sr,si,sr->ri",
                         phdelay,
                         gauss_shape,
                         brness,
                         spectrum)

        return vis + coh.astype(dtype)

    return lax.fori_loop(0, source, body, vis)

@jit
def rime_gauss_lm_single_corr_1d(radec, uvw, frequency, shape_params, stokes, alpha):

    lm = pixel2lm(radec)
    source = lm.shape[0]
    row = uvw.shape[0]
    # chan = frequency.shape[0]
    corr = stokes.shape[1]
    
    shape_arcsec = jnp.empty_like(shape_params)
    shape_arcsec = ops.index_update(shape_arcsec, ops.index[:, 0:2], shape_params[:,0:2]*arcsec2rad)
    shape_arcsec = ops.index_update(shape_arcsec, ops.index[:, 2], shape_params[:,2]*deg2rad)

    dtype = jnp.result_type(lm.dtype, uvw.dtype,
                            frequency.dtype, shape_params.dtype, stokes.dtype,
                            jnp.complex64)
    vis = jnp.empty((row, corr), dtype)

    def body(s, vis):
        phdelay = phase_delay_1d(lm[None, s], uvw, frequency)
        brness = stokes[None, s]
        spectrum = source_spectrum(alpha[s], frequency)
        gauss_shape = gaussian_1d(uvw, frequency, shape_arcsec[None, s])
        coh = jnp.einsum("sr,sr,si,sr->ri",
                         phdelay,
                         gauss_shape,
                         brness,
                         spectrum)

        return vis + coh.astype(dtype)

    return lax.fori_loop(0, source, body, vis)

@jit
def rime_pnts_lm_single_corr_1d(radec, uvw, frequency, stokes, alpha):

    lm = pixel2lm(radec) # radec2lm(radec) frac2lm
    source = lm.shape[0]
    row = uvw.shape[0]
    # chan = frequency.shape[0]
    corr = stokes.shape[1]

    dtype = jnp.result_type(lm.dtype, uvw.dtype,
                            frequency.dtype, stokes.dtype,
                            jnp.complex64)
    vis = jnp.empty((row, corr), dtype)

    def body(s, vis):
        phdelay = phase_delay_1d(lm[None, s], uvw, frequency)
        brness = stokes[None, s]
        spectrum = source_spectrum(alpha[s], frequency)

        coh = jnp.einsum("sr,si,sr->ri",
                         phdelay,
                         brness,
                         spectrum)

        return vis + coh.astype(dtype)
    
    return lax.fori_loop(0, source, body, vis)

@jit
def rime_pnts_fuse_1d(radec, uvw, frequency, stokes, alpha):

    lm = radec2lm(radec)
    source = lm.shape[0]
    row = uvw.shape[0]
    # chan = frequency.shape[0]
    corr = stokes.shape[1]

    dtype = jnp.result_type(lm.dtype, uvw.dtype,
                            frequency.dtype, stokes.dtype,
                            jnp.complex64)
    vis = jnp.empty((row, corr), dtype)

    def body(s, vis):
        phdelay = phase_delay_1d(lm[None, s], uvw, frequency)
        brness = brightness(stokes[None, s])
        spectrum = source_spectrum(alpha[s], frequency)

        coh = jnp.einsum("sr,si,sr->ri",
                         phdelay,
                         brness,
                         spectrum)

        return vis + coh.astype(dtype)
    
    return lax.fori_loop(0, source, body, vis)