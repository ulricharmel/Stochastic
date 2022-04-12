import dask.array as da
import numpy as np
import pytest
import pyrap.tables as pt
import stochastic.rime.tools as RT
import os

from stochastic.rime.jax_rime import fused_rime, fused_rime_sinlge_corr, fused_wsclean_rime
from testing.conftest import freq0

import stochastic.rime.tools as RT

import stochastic.opt.optax_grads as optaxGrads
import stochastic.opt.jax_grads as jaxGrads
# from stochastic.opt.custom_grads import update as custom_grads_update 
import stochastic.rime.tools as RT

import stochastic.opt as opt
from stochastic.data_handling.read_data import set_xds, load_model, MSobject

from functools import partial
from jax.flatten_util import ravel_pytree

import jax.config
import jax.numpy as jnp
from jax.test_util import check_grads

# again, this only works on startup!
from jax.config import config
config.update("jax_enable_x64", True)

wsclean = False
log_spectra = False

if wsclean:
    if log_spectra:
        jaxGrads.forward_model = opt.foward_pnts_lm_wsclean_log
        optaxGrads.forward_model = opt.foward_pnts_lm_wsclean_log
    else:
        jaxGrads.forward_model = opt.foward_pnts_lm_wsclean
        optaxGrads.forward_model = opt.foward_pnts_lm_wsclean
else:
    jaxGrads.forward_model = opt.foward_pnts_lm
    optaxGrads.forward_model = opt.foward_pnts_lm


# define some constants
deg2rad = jnp.pi / 180.0;
arcsec2rad = deg2rad / 3600.0;
uas2rad = 1e-6 * deg2rad / 3600.0;

rad2deg = 180./jnp.pi

arcsec2deg = 1./3600.0
rad2arsec = 1./arcsec2rad

def lm_to_pixel(lm, cellsize, cx, cy):
    
    x = np.zeros_like(lm)
    x[:,1] = lm[:,0]/(cellsize*arcsec2rad) + cx
    x[:,0] = lm[:,1]/(cellsize*arcsec2rad) + cy

    return x 

def pixel_to_lm(x, cellsize, cx, cy):

    lm = np.zeros_like(x)
    lm[:,1] = (x[:,0]-cx)*cellsize*arcsec2rad
    lm[:,0] = (x[:,1]-cy)*cellsize*arcsec2rad

    return lm


@pytest.mark.optimisation
def test_wsclean_grads(msname, initmodel):
    
    xds = MSobject(msname, "DATA", "WEIGHT", False, None, False, [0,-1])
    phasedir = xds.phasedir

    RT.ra0, RT.dec0 = phasedir
    RT.freq0 = np.mean(xds.data_chan_freq)

    LR = dict(alpha=1e-2, radec=1e-4, stokes=1e-2)

    offsets = [0, 1/10., 1/8., 1/4., 1/2.]

    for offset in offsets:
        offby = np.random.uniform(low=-offset, high=offset, size=(1,2))/3600.
        lm_model = np.load(initmodel)

        radec = np.zeros((1,2))

        radec[0, 0] = lm_model[0,1] + offby[0,0]
        radec[0, 1] = lm_model[0,2] + offby[0,1]

        # lm_model[0,0] += 0.4    
        lm_model[:,1:3] =  radec

        init_model = "./tmp-point-init-offset-radec.npy"
        np.save(init_model, lm_model)

        params, d_params, nparams = load_model(init_model, None)
        
        d_vis, d_weights, d_uvw, d_kwargs = xds.getbatch(0, 2016, d_params)
        d_freq = xds.data_chan_freq.copy()

        d_kwargs["alpha_l1"] = 0
        d_kwargs["alpha_l2"] = 0
        d_kwargs["params0"] = params.copy()
    
        loss_fn = jaxGrads.loss_fn

        loss, grads = jax.value_and_grad(loss_fn)(params, d_uvw, d_freq, d_vis, d_weights, d_kwargs)

        print(f"-------ofset fraction in arcseconds {offby*3600}--------------")
        print(f"loss - {loss}")
        print(f"grads - {grads}")

        print("---------Check grads----------------")
        
        loss_g = partial(loss_fn, data_uvw=d_uvw, data_chan_freq=d_freq, data=d_vis, weights=d_weights, kwargs=d_kwargs)

        def loss_gg(pp, unravel_fn):
            params = unravel_fn(pp)
            return loss_g(params)
        
        pp, unravel_fn = ravel_pytree(params)
        my_loss = partial(loss_gg, unravel_fn=unravel_fn)

        check_grads(my_loss, (pp,), order=2)  # check up to 2nd order derivatives
    
    os.system(f"rm {init_model}")

