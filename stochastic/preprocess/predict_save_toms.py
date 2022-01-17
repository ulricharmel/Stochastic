import dask.array as da
import numpy as np
import pytest
import pyrap.tables as pt
from MSUtils.msutils import addcol
import stochastic.rime.tools as RT

from stochastic.rime.jax_rime import fused_rime, fused_wsclean_rime, fused_wsclean_log_rime
from stochastic.preprocess.skymodel_utils import get_field_center

import jax.numpy as jnp

# again, this only works on startup!
from jax.config import config
config.update("jax_enable_x64", True)

def wsclean_rime_to_MS(msname, dummymodel, freq0, datacol, logspi=False):
    # for now simple predict with pyrap
    # hopefull this is enoug for now, else we have to switch to dask-ms

    # get some observation settings from the MS
    tab = pt.table(msname)
    uvw = tab.getcol('UVW')
    tab.close()

    # get frequency info from SPECTRAL_WINDOW subtable
    freqtab = pt.table(msname+'::SPECTRAL_WINDOW')
    freq = freqtab.getcol('CHAN_FREQ')[0]
    freqtab.close()

    phase_centre = get_field_center(msname)
    RT.ra0, RT.dec0 = phase_centre
    RT.freq0 = freq0

    model = np.load(dummymodel)

    nsources = model.shape[0]
    spi_c = model.shape[1] - 6

    stokes = np.zeros((nsources, 4))
    stokes[:,0] = model[:,0]

    alpha = np.zeros((nsources, spi_c))
    alpha[:] = model[:,6:]

    radec = model[:,1:3]

    shape_params = model[:,3:6]
    stokes = jnp.asarray(stokes)
    radec = jnp.asarray(radec)
    shape_params = jnp.asarray(shape_params)
    uvw = jnp.asarray(uvw)
    freq = jnp.asarray(freq)
    alpha = jnp.asarray(alpha)

    if logspi:
        vis = fused_wsclean_log_rime(radec, uvw, freq, shape_params, stokes, alpha)
    else:
        vis = fused_wsclean_rime(radec, uvw, freq, shape_params, stokes, alpha)

    # add datacol if it is not present
    addcol(msname, datacol)	 

    tab = pt.table(msname, readonly=False)
    tab.putcol(datacol, np.array(vis)) # convert JAX array to numpy array
    tab.close()
