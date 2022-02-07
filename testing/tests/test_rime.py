import dask.array as da
import numpy as np
import pytest
import pyrap.tables as pt
import stochastic.rime.tools as RT

from stochastic.rime.jax_rime import fused_rime, fused_rime_sinlge_corr, fused_wsclean_rime
from testing.conftest import freq0

import jax.numpy as jnp

# again, this only works on startup!
from jax.config import config
config.update("jax_enable_x64", True)

def get_field_center(MS):
	t = pt.table(MS+"/FIELD", ack=False)
	phase_centre = (t.getcol("PHASE_DIR"))[0,0,:]
	t.close()
	return phase_centre[0], phase_centre[1]

# @pytest.mark.rime
# def test_wsclean_rime(msname, dummymodel, freq0):
#     # get some observation settings from the MS
#     tab = pt.table(msname)
#     uvw = tab.getcol('UVW')
#     model_vis = tab.getcol("MODEL_DATA")
#     tab.close()

#     # get frequency info from SPECTRAL_WINDOW subtable
#     freqtab = pt.table(msname+'::SPECTRAL_WINDOW')
#     freq = freqtab.getcol('CHAN_FREQ')[0]
#     freqtab.close()

#     phase_centre = get_field_center(msname)
#     RT.ra0, RT.dec0 = phase_centre
#     RT.freq0 = freq0

#     model = np.load(dummymodel)
    
#     nsources = model.shape[0]
#     spi_c = model.shape[1] - 6

#     stokes = np.zeros((nsources, 4))
#     stokes[:,0] = model[:,0]

#     alpha = np.zeros((nsources, spi_c))
#     alpha[:] = model[:,6:]

#     radec = model[:,1:3]

#     shape_params = model[:,3:6]
#     stokes = jnp.asarray(stokes)
#     radec = jnp.asarray(radec)
#     shape_params = jnp.asarray(shape_params)
#     uvw = jnp.asarray(uvw)
#     freq = jnp.asarray(freq)
#     alpha = jnp.asarray(alpha)
    
#     vis = fused_wsclean_rime(radec, uvw, freq, shape_params, stokes, alpha)

#     vis = np.array(vis)

#     assert np.allclose(vis, model_vis)


@pytest.fixture(scope="function", params=[
    {
        "source": (1000,),
        "row": (1000,),
        "chan": (512,),
    }
])
def chunks(request):
    return request.param

@pytest.mark.rime("chunks", [{
        "source": (1000,),
        "row": (1000,)*10,
        "chan": (512,),
}], indirect=True)
def test_dask_rime(chunks):
    src = sum(chunks["source"])
    row = sum(chunks["row"])
    chan = sum(chunks["chan"])

    radec = da.random.random((src, 2), chunks=(chunks["source"], 2))*.0001
    uvw = da.random.random((row, 3), chunks=(chunks["row"], 3))
    freq = da.linspace(.856e9, 2*.856e9, chan, chunks=chunks["chan"])

    stokes = da.random.random((src, 4), chunks=(chunks["source"], 4))
    alpha = da.random.random((src, 3), chunks=(chunks["source"], 3))
    shape_params = da.zeros((src, 3), chunks=(chunks["source"], 3))

    from stochastic.rime.dask_wrappers import rime
    vis = rime(radec, uvw, freq, shape_params, stokes, alpha) 
    data = vis.compute()

    import pdb; pdb.set_trace()







