import dask.array as da
import numpy as np
import pytest


from ..jax_rime import fused_rime


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

    stokes = np.random.random((src, 4))

    vis = fused_rime(lm, uvw, freq, stokes)
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

    stokes = da.random.random((src, 4), chunks=(chunks["source"], 4))

    from ..dask_wrappers import rime
    vis = rime(lm, uvw, freq, stokes)
    vis.compute()