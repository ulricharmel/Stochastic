import dask.array as da
import numpy as np
import pytest


from ..jax_rime import fused_rime, fused_rime_sinlge_corr

# define some constants
deg2rad = np.pi / 180.0;
arcsec2rad = deg2rad / 3600.0;
uas2rad = 1e-6 * deg2rad / 3600.0;

@pytest.fixture(scope="function", params=[
    {
        "source": (1000,),
        "row": (1000,),
        "chan": (512,),
    }
])
def chunks(request):
    return request.param

def test_rime(chunks):
    src = sum(chunks["source"])
    row = sum(chunks["row"])
    chan = sum(chunks["chan"])

    lm = np.random.random((src, 2))*.00001
    uvw = (np.random.random((row, 3)) - 0.5)*10000
    freq = np.linspace(.856e9, 2*.856e9, chan)

    shape_params = np.random.random((src, 3))*uas2rad
    shape_params[:,2] = 0

    stokes = np.random.random((src, 4))
    alpha = np.random.random((src, 1))

    vis = fused_rime(lm, uvw, freq, shape_params, stokes, alpha)
    print(vis)

def test_rime_single_corr(chunks):
    src = sum(chunks["source"])
    row = sum(chunks["row"])
    chan = sum(chunks["chan"])

    lm = np.random.random((src, 2))*.00001
    uvw = (np.random.random((row, 3)) - 0.5)*10000
    freq = np.linspace(.856e9, 2*.856e9, chan)

    shape_params = np.random.random((src, 3))*uas2rad
    shape_params[:,2] = 0

    stokes = np.random.random((src, 1))
    alpha = np.random.random((src, 1))

    vis = fused_rime_sinlge_corr(lm, uvw, freq, shape_params, stokes, alpha)
    print(vis)


@pytest.mark.parametrize("chunks", [{
        "source": (1000,),
        "row": (1000,)*10,
        "chan": (512,),
}], indirect=True)
def test_dask_rime(chunks):
    src = sum(chunks["source"])
    row = sum(chunks["row"])
    chan = sum(chunks["chan"])

    lm = da.random.random((src, 2), chunks=(chunks["source"], 2))*.0001
    uvw = da.random.random((row, 3), chunks=(chunks["row"], 3))
    freq = da.linspace(.856e9, 2*.856e9, chan, chunks=chunks["chan"])
    
    shape_params = da.random.random((src, 3), chunks=(chunks["source"], 3))*uas2rad
    shape_params[:,2] = 0

    stokes = da.random.random((src, 4), chunks=(chunks["source"], 4))

    from ..dask_wrappers import rime
    vis = rime(lm, uvw, freq, shape_params, stokes)
    vis.compute()