import dask
import dask.array as da
import numpy as np

from stochastic.rime.jax_rime import fused_rime


def _wrapper(radec, uvw, frequency, shape_params, stokes, alpha):
    return fused_rime(radec, uvw, frequency, shape_params, stokes, alpha).to_py() # using to_py because you want to fill the back the output to ms


def rime(radec, uvw, frequency, shape_params, stokes, alpha):
    assert radec.ndim == 2
    assert radec.chunks[0] == stokes.chunks[0]
    assert radec.chunks[1] == (2,)
    
    assert shape_params.ndim == 2
    assert shape_params.chunks[0] == stokes.chunks[0]
    assert shape_params.chunks[1] == (3,)

    assert uvw.ndim == 2
    assert uvw.chunks[1] == (3,)

    assert frequency.ndim == 1

    assert stokes.ndim == 2

    dtype = np.result_type(radec, alpha, uvw, frequency, stokes, shape_params, np.complex64)

    return da.blockwise(_wrapper, ("row", "chan", "corr"),
                        radec, ("source", "radec"),
                        uvw, ("row", "uvw"),
                        frequency, ("chan",),
                        shape_params, ("source", "shape_params"),
                        stokes, ("source", "corr"),
                        alpha, ("source", "alpha"),
                        meta=np.empty((0,0,0), dtype),
                        concatenate=True,
                        align_arrays=False)

