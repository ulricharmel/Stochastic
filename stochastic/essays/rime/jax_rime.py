from functools import partial

import numpy as np
from scipy.constants import c as lightspeed

import jax.numpy as jnp
from jax import jit, vmap
from jax import lax, ops
from jax.experimental import loops


minus_two_pi_over_c = -2*jnp.pi/lightspeed

@jit
def gaussian(uvw, frequency, shape_params):
    # https://en.wikipedia.org/wiki/Full_width_at_half_maximum

    fwhm = 2. * jnp.sqrt(2. * jnp.log(2.))
    fwhminv = 1. / fwhm
    gauss_scale = fwhminv * jnp.sqrt(2.) * jnp.pi / lightspeed

    dtype = jnp.result_type(*(jnp.dtype(a.dtype.name) for
                             a in (uvw, frequency, shape_params)))

    nsrc = shape_params.shape[0]
    nrow = uvw.shape[0]
    nchan = frequency.shape[0]
 
    with loops.Scope() as l1:
        l1.scaled_freq = jnp.empty_like(frequency)

        for f in l1.range(frequency.shape[0]):
            l1.scaled_freq = ops.index_update(l1.scaled_freq, f, frequency[f] * gauss_scale)
    
    with loops.Scope() as l2:
        l2.shape = jnp.empty((nsrc, nrow, nchan), dtype=dtype)

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

                for f in l2.range(l1.scaled_freq.shape[0]):
                    l2.fu1 = l2.u1*l1.scaled_freq[f]
                    l2.fv1 = l2.v1*l1.scaled_freq[f]

                    l2.shape = ops.index_update(l2.shape, (s, r, f), jnp.exp(-(l2.fu1*l2.fu1 + l2.fv1*l2.fv1)))

    return l2.shape

@jit
def phase_delay(lm, uvw, frequency):
    out_dtype = jnp.result_type(lm, uvw, frequency, jnp.complex64)

    one = lm.dtype.type(1.0)
    neg_two_pi_over_c = lm.dtype.type(minus_two_pi_over_c)
    complex_one = out_dtype.type(1j)

    l = lm[:, 0, None, None]  # noqa
    m = lm[:, 1, None, None]

    u = uvw[None, :, 0, None]
    v = uvw[None, :, 1, None]
    w = uvw[None, :, 2, None]

    n = jnp.sqrt(one - l**2 - m**2) - one

    real_phase = (neg_two_pi_over_c *
                  (l * u + m * v + n * w) *
                  frequency[None, None, :])

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
def coherency(nsrc, lm, uvw, frequency, stokes):
    return jnp.einsum("srf,si->srfi",
                      phase_delay(lm, uvw, frequency),
                      brightness(stokes))


@jit
def fused_rime(lm, uvw, frequency, shape_params, stokes):

    # Full expansion over source axis -- very memory hungry

    # return jnp.einsum("srf,si->rfi",
    #                   phase_delay(lm, uvw, frequency),
    #                   brightness(stokes))

    source = lm.shape[0]
    row = uvw.shape[0]
    chan = frequency.shape[0]
    corr = stokes.shape[1]

    dtype = jnp.result_type(lm.dtype, uvw.dtype,
                            frequency.dtype, shape_params.dtype, stokes.dtype,
                            jnp.complex64)
    vis = jnp.empty((row, chan, corr), dtype)

    def body(s, vis):
        phdelay = phase_delay(lm[None, s], uvw, frequency)
        brness = brightness(stokes[None, s])
        coh = jnp.einsum("srf,si->rfi",
                         phdelay,
                         brness)
        '''gauss_shape = gaussian(uvw, frequency, shape_params[None, s])
        coh = jnp.einsum("srf,srf,si->rfi",
                         phdelay,
                         gauss_shape,
                         brness)'''

        return vis + coh.astype(dtype)

    return lax.fori_loop(0, source, body, vis)
