import pyrap.tables as pt
import jax.numpy as jnp
import numpy as np

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
        data_vis = tab.getcol(datacol)
    data_uvw = tab.getcol("UVW")
    tab.close()

    freqtab = pt.table(msname+'::SPECTRAL_WINDOW', ack=False)
    data_chan_freq = freqtab.getcol('CHAN_FREQ')[0]
    freqtab.close()

    data_vis = jnp.asarray(data_vis)
    data_uvw = jnp.asarray(data_uvw)
    data_chan_freq = jnp.asarray(data_chan_freq)

    return data_vis, data_uvw, data_chan_freq

def load_model(modelfile):
    """load model save a npy file.
        Array with shape (nsources x l x m x emaj, emin x pa)
    Args:
        numpy file
    Returns:
        dictionary with the intial parameters    
    """

    model = np.load(modelfile)
    stokes = model[:,0:1]
    lm = model[:,1:3]
    shape_params = model[:,3:]

    params = {}
    params["lm"] = jnp.asarray(lm)
    params["stokes"] = jnp.asarray(stokes)
    params["shape_params"] = jnp.asarray(shape_params)

    return params







