import dask
import dask.array as da
import numpy as np

from .jax_rime import fused_rime


def _wrapper(lm, uvw, frequency, shape_params, stokes):
    return fused_rime(lm, uvw, frequency, shape_params, stokes)

def rime(lm, uvw, frequency, shape_params, stokes):
    assert lm.ndim == 2
    assert lm.chunks[0] == stokes.chunks[0]
    assert lm.chunks[1] == (2,)
    
    assert shape_params.ndim == 2
    assert shape_params.chunks[0] == stokes.chunks[0]
    assert shape_params.chunks[1] == (3,)

    assert uvw.ndim == 2
    assert uvw.chunks[1] == (3,)

    assert frequency.ndim == 1

    assert stokes.ndim == 2

    dtype = np.result_type(lm, uvw, frequency, stokes, shape_params, np.complex64)

    return da.blockwise(_wrapper, ("row", "chan", "corr"),
                        lm, ("source", "lm"),
                        uvw, ("row", "uvw"),
                        frequency, ("chan",),
                        shape_params, ("source", "shape_params"),
                        stokes, ("source", "corr"),
                        meta=np.empty((0,0,0), dtype))

