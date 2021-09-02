import pyrap.tables as pt
import jax.numpy as jnp
import numpy as np
from loguru import logger

def load_data(msname, datacol, weightcol, single_corr):
    """Open the measurement set return the data
    Args:
        msname (str)
            measurement set name
        datacol (str)
            datacol with visibilities to fit
        weightcol (str)
            column with weights
        single_corr (bool)
            use a single correlation
    return:
        tuple of data visibilites, uvw, frequencies
    """
    
    tab = pt.table(msname, ack=False)
    if single_corr:
        data_vis = tab.getcol(datacol)[:,:,0:1]
        # Get flag info from MS
        data_flag = tab.getcol('FLAG')[:,:,0:1]
        data_flag_row = tab.getcol('FLAG_ROW')
        data_flag = np.logical_or(data_flag, data_flag_row[:,np.newaxis,np.newaxis])

        data_weights = tab.getcol(weightcol)
        if data_weights.ndim == 3:
            data_weights = data_weights[:,:,0:1]
        else:
            data_weights = np.broadcast_to(data_weights[:,None,0:1], data_vis.shape)
    else:
        logger.warning("Only one correlation implemented for now. Will default to the first correlation")
        data_vis = tab.getcol(datacol)[:,:,0:1]
        
        data_flag = tab.getcol('FLAG')[:,:,0:1]
        data_flag_row = tab.getcol('FLAG_ROW')
        data_flag = np.logical_or(data_flag, data_flag_row[:,np.newaxis,np.newaxis])

        data_weights = tab.getcol(weightcol)
        if data_weights.ndim == 3:
            data_weights = data_weights[:,:,0:1]
        else:
            data_weights = np.broadcast_to(data_weights[:,None,0:1], data_vis.shape)
    
    data_weights.setflags(write=1)
    data_weights *= np.logical_not(data_flag)

    data_uvw = tab.getcol("UVW")
    tab.close()

    freqtab = pt.table(msname+'::SPECTRAL_WINDOW', ack=False)
    data_chan_freq = freqtab.getcol('CHAN_FREQ')[0]
    freqtab.close()

    fieldtab = pt.table(msname+"::FIELD", ack=False)
    phasedir = (fieldtab.getcol("PHASE_DIR"))[0,0,:]
    fieldtab.close()

    data_vis = jnp.asarray(data_vis)
    data_weights = jnp.asarray(data_weights)
    data_uvw = jnp.asarray(data_uvw)
    data_chan_freq = jnp.asarray(data_chan_freq)


    return data_vis, data_weights, data_uvw, data_chan_freq, phasedir

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
    shape_params = model[:,3:6]
    alpha = model[:,6:]

    params = {}
    params["stokes"] = jnp.asarray(stokes)
    params["radec"]  = jnp.asarray(radec)
    params["shape_params"] = jnp.asarray(shape_params)
    params["alpha"] = jnp.asarray(alpha)

    return params







