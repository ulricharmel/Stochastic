import pyrap.tables as pt
import jax.numpy as jnp
import numpy as np
from loguru import logger

from daskms import xds_from_ms, xds_from_table
from stochastic.rime.jax_rime import fused_wsclean_log_rime_sc, fused_wsclean_rime_sc

datacol = "DATA"
weightcol = "WEIGHT"
single_corr = True
log_spectra = False
dummyparams = False
freq_range = [0,-1] 

def set_xds(msname, data_col, weight_col, rowchunks, singlecorr, dummy_column, logspectra, freqrange):
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
        dummy_column (str)
            column for dummy visibilities
    Return:
        Xarray dataset, data_chan_freq, phasedir, rmap
    """

    global datacol, weightcol, single_corr, log_spectra, freq_range
    datacol, weightcol, single_corr, log_spectra, freq_range = data_col, weight_col, singlecorr, logspectra, freqrange

    fs = freq_range[0]
    fe = freq_range[1]

    freqtab = pt.table(msname+'::SPECTRAL_WINDOW', ack=False)
    data_chan_freq = jnp.asarray(freqtab.getcol('CHAN_FREQ')[0][fs:fe])
    freqtab.close()

    fieldtab = pt.table(msname+"::FIELD", ack=False)
    phasedir = (fieldtab.getcol("PHASE_DIR"))[0,0,:]
    fieldtab.close()
    columns = ["FLAG", "FLAG_ROW", "UVW", "TIME", "ANTENNA1", "ANTENNA2", datacol, weightcol]
    if dummy_column:
        columns.append(dummy_column)
    
    xds = xds_from_ms(msname, columns=columns, chunks={"row":-1})[0]
    
    timecol = xds.TIME.compute().data
    unique  = np.unique(timecol)
    rmap = {x: i for i, x in enumerate(unique)}
    rowmap = jnp.asarray(np.fromiter(list(map(rmap.__getitem__, timecol)), int))
    antenna1 = jnp.asarray(xds.ANTENNA1.compute().data)
    antenna2 = jnp.asarray(xds.ANTENNA2.compute().data)

    return xds, data_chan_freq, phasedir

def getbatch(inds, xds, d_params, dummy_column, data_chan_freq):
    """
    Return the data for the given batch
    Args:
        inds (random indices for batch)
        xds (dataset)
            Xarray dataset
        d_params (array or None)
            dummy parameters
        dummy_column (str or None)
            dummy visibilities column
        data_chan_freq (array)
            use to commpute dummy visibilities from model
    Return:
        data_vis, data_weights, data_uvw, d_kargs
    """

    fs = freq_range[0]
    fe = freq_range[1]

    if single_corr:
        data_vis = xds[datacol][inds][:,fs:fe,0:1].compute().data
        data_flag = xds.FLAG[inds][:,fs:fe,0:1].compute().data
        data_flag_row = xds.FLAG_ROW[inds].compute().data
        data_flag = np.logical_or(data_flag, data_flag_row[:,np.newaxis,np.newaxis])

        if dummy_column:
            dummy_vis = xds[dummy_column][inds][:,fs:fe,0:1].compute().data
            data_vis -= dummy_vis
        else:
            dummy_vis = None

        data_weights = xds[weightcol][inds].compute().data.real
        if data_weights.ndim == 3:
            data_weights = data_weights[:,fs:fe,0:1]
        else:
            data_weights = np.broadcast_to(data_weights[:,None,0:1], data_vis.shape)
    else:
        data_vis = xds[datacol][inds][:,fs:fe,0:1].compute().data
        data_flag = xds.FLAG[inds][:,fs:fe,0:1].compute().data
        data_flag_row = xds.FLAG_ROW[inds].compute().data
        data_flag = np.logical_or(data_flag, data_flag_row[:,np.newaxis,np.newaxis])

        if dummy_column:
            dummy_vis = xds[dummy_column][inds][:,fs:fe,0:1].compute().data
            data_vis -= dummy_vis
        else:
            dummy_vis = None

        data_weights = xds[weightcol][inds].compute().data.real
        if data_weights.ndim == 3:
            data_weights = data_weights[:,fe:fe,0:1]
        else:
            data_weights = np.broadcast_to(data_weights[:,None,0:1], data_vis.shape)
    
    data_weights.setflags(write=1)
    data_weights *= np.logical_not(data_flag)

    data_uvw = xds.UVW[inds].compute().data

    data_vis = jnp.asarray(data_vis)
    data_weights = jnp.asarray(data_weights)
    data_uvw = jnp.asarray(data_uvw)

    # NOQA: no checks added to ensure single correlation, by default just assume everything is single correlation 
    if dummyparams:
        d_shape_params = d_params["shape_params"]
        d_stokes = d_params["stokes"]
        d_radec = d_params["radec"]
        d_alpha = d_params["alpha"]

        if log_spectra:
            dummy_model_vis = fused_wsclean_log_rime_sc(d_radec, data_uvw, data_chan_freq, d_shape_params, d_stokes, d_alpha)
        else:
            dummy_model_vis = fused_wsclean_rime_sc(d_radec, data_uvw, data_chan_freq, d_shape_params, d_stokes, d_alpha)

        data_vis -= dummy_model_vis

    d_kwargs = {}
    d_kwargs["dummy_params"] = d_params
    d_kwargs["dummy_col_vis"] = dummy_vis

    return data_vis, data_weights, data_uvw, d_kwargs

def load_model(modelfile, dummy_model):
    """load model save a npy file.
        Array with shape (nsources x flux x ra x dec x emaj, emin x pa)
    Args:
        modelfile
            numpy array file
        dummy_model 
            numpy array file
    Returns:
        dictionary with the intial parameters    
    """

    model = np.load(modelfile)
    stokes = model[:,0:1]
    radec = model[:,1:3]
    # shape_params = model[:,3:6]
    alpha = model[:,3:]

    nparams = model.shape[1]+3

    params = {}
    params["stokes"] = jnp.asarray(stokes)
    params["radec"]  = jnp.asarray(radec)
    # params["shape_params"] = jnp.asarray(shape_params)
    params["alpha"] = jnp.asarray(alpha)

    if dummy_model:
        global dummyparams
        dummyparams = True
        d_model = np.load(dummy_model)
        d_stokes = d_model[:,0:1]
        d_radec = d_model[:,1:3]
        d_shape_params = d_model[:,3:6]
        d_alpha = d_model[:,6:]

        d_params = {}
        d_params["stokes"] = jnp.asarray(d_stokes)
        d_params["radec"]  = jnp.asarray(d_radec)
        d_params["shape_params"] = jnp.asarray(d_shape_params)
        d_params["alpha"] = jnp.asarray(d_alpha)
    else:
        d_params = None

    return params, d_params, nparams







