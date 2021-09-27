import pyrap.tables as pt
import jax.numpy as jnp
import numpy as np
from loguru import logger

from daskms import xds_from_ms, xds_from_table

datacol = "DATA"
weightcol = "WEIGHT"
single_corr = True

def set_xds(msname, data_col, weight_col, rowchunks, singlecorr):
    """
    Read the measuremenet in an xarray dataset
    Args:
        msname (str)
            measurement set name
        datacol (str)
            datacol with visibilities to fit
        weightcol(str)
            column with weights to used
        rowchunks (int)
            rows chunk size for xarray
        singlecorr (bool)
            set the single_corrr argument
    Return:
        Xarray dataset, data_chan_freq, phasedir, rmap
    """

    global datacol, weightcol, single_corr
    datacol, weightcol, single_corr = data_col, weight_col, singlecorr

    freqtab = pt.table(msname+'::SPECTRAL_WINDOW', ack=False)
    data_chan_freq = jnp.asarray(freqtab.getcol('CHAN_FREQ')[0])
    freqtab.close()

    fieldtab = pt.table(msname+"::FIELD", ack=False)
    phasedir = (fieldtab.getcol("PHASE_DIR"))[0,0,:]
    fieldtab.close()

    columns = ["FLAG", "FLAG_ROW", "UVW", "TIME", "ANTENNA1", "ANTENNA2", datacol, weightcol]
    xds = xds_from_ms(msname, columns=columns, chunks={"row":-1})[0]
    
    timecol = xds.TIME.compute().data
    unique  = np.unique(timecol)
    rmap = {x: i for i, x in enumerate(unique)}
    rowmap = jnp.asarray(np.fromiter(list(map(rmap.__getitem__, timecol)), int))
    antenna1 = jnp.asarray(xds.ANTENNA1.compute().data)
    antenna2 = jnp.asarray(xds.ANTENNA2.compute().data)

    return xds, data_chan_freq, phasedir

def getbatch(inds, xds):
    """
    Return the data for the given batch
    Args:
        inds (random indices for batch)
        xds (dataset)
            Xarray dataset
    Return:
        data_vis, data_weights, data_uvw
    """

    if single_corr:
        data_vis = xds[datacol][inds][:,:,0:1].compute().data
        data_flag = xds.FLAG[inds][:,:,0:1].compute().data
        data_flag_row = xds.FLAG_ROW[inds].compute().data
        data_flag = np.logical_or(data_flag, data_flag_row[:,np.newaxis,np.newaxis])

        data_weights = xds[weightcol][inds].compute().data
        if data_weights.ndim == 3:
            data_weights = data_weights[:,:,0:1]
        else:
            data_weights = np.broadcast_to(data_weights[:,None,0:1], data_vis.shape)
    else:
        data_vis = xds[datacol][inds][:,:,0:1].compute().data
        data_flag = xds.FLAG[inds][:,:,0:1].compute().data
        data_flag_row = xds.FLAG_ROW[inds].compute().data
        data_flag = np.logical_or(data_flag, data_flag_row[:,np.newaxis,np.newaxis])

        data_weights = xds[weightcol][inds].compute().data
        if data_weights.ndim == 3:
            data_weights = data_weights[:,:,0:1]
        else:
            data_weights = np.broadcast_to(data_weights[:,None,0:1], data_vis.shape)
    
    data_weights.setflags(write=1)
    data_weights *= np.logical_not(data_flag)

    data_uvw = xds.UVW[inds].compute().data

    data_vis = jnp.asarray(data_vis)
    data_weights = jnp.asarray(data_weights)
    data_uvw = jnp.asarray(data_uvw)

    return data_vis, data_weights, data_uvw

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







