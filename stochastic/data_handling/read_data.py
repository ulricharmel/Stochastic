import pyrap.tables as pt
import jax.numpy as jnp
import numpy as np
from loguru import logger

def load_data(msname, datacol, single_corr):
    """Open the measurement set return the data
    Args:
        msname (str)
            measurement set name
        datacol (str)
            datacol with visibilities to fit
        single_corr (bool)
            use a single correlation
    return:
        tuple of data visibilites, uvw, frequencies
    """
    
    tab = pt.table(msname, ack=False)
    if single_corr:
        data_vis = tab.getcol(datacol)[:,:,0:1]
    else:
        logger.warning("Only one correlation implemented for now. Will default to the first correlation")
        data_vis = tab.getcol(datacol)[:,:,0:1]
    data_uvw = tab.getcol("UVW")
    tab.close()

    freqtab = pt.table(msname+'::SPECTRAL_WINDOW', ack=False)
    data_chan_freq = freqtab.getcol('CHAN_FREQ')[0]
    freqtab.close()

    fieldtab = pt.table(msname+"::FIELD", ack=False)
    phasedir = (fieldtab.getcol("PHASE_DIR"))[0,0,:]
    fieldtab.close()

    data_vis = jnp.asarray(data_vis)
    data_uvw = jnp.asarray(data_uvw)
    data_chan_freq = jnp.asarray(data_chan_freq)

    return data_vis, data_uvw, data_chan_freq, phasedir

def load_model(modelfile):
    """load model save a npy file.
        Array with shape (nsources x flux x ra x dec x emaj, emin x pa)
    Args:
        numpy file
    Returns:
        dictionary with the intial parameters    
    """

    model = np.load(modelfile)
    stokes = model[:,0:1]
    radec = model[:,1:3]
    shape_params = model[:,3:]

    params = {}
    params["stokes"] = jnp.asarray(stokes)
    params["radec"]  = jnp.asarray(radec)
    params["shape_params"] = jnp.asarray(shape_params)

    return params







