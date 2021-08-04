import dask
import dask.array as da
import numpy as np

from .jax_rime import fused_rime


def _wrapper(lm, uvw, frequency, stokes):
    return fused_rime(lm[0][0], uvw[0], frequency, stokes[0])

def rime(lm, uvw, frequency, stokes):
    assert lm.ndim == 2
    assert lm.chunks[0] == stokes.chunks[0]
    assert lm.chunks[1] == (2,)

    assert uvw.ndim == 2
    assert uvw.chunks[1] == (3,)

    assert frequency.ndim == 1

    assert stokes.ndim == 2

    dtype = np.result_type(lm, uvw, frequency, stokes, np.complex64)

    return da.blockwise(_wrapper, ("row", "chan", "corr"),
                        lm, ("source", "lm"),
                        uvw, ("row", "uvw"),
                        frequency, ("chan",),
                        stokes, ("source", "corr"),
                        meta=np.empty((0,0,0), dtype))

