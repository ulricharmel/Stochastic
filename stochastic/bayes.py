#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, time
import argparse
import pyrap.tables as pt
import numpy as np
import jax.numpy as jnp
from jax import lax, jit, random

from jaxns.nested_sampling import NestedSampler, save_results, load_results
from jaxns.prior_transforms import PriorChain, UniformPrior
from jaxns.plotting import plot_cornerplot, plot_diagnostics
from jaxns.utils import summary

from zagros.vardefs import *
from zagros.essays.rime.jax_rime import fused_rime

# set global variables 
data_vis = None # variable to hold input data matrix
data_ant1 = None
data_ant2 = None
data_uvw = None
data_nant = None
data_nbl = None

data_uniqtimes = None
data_uniqtime_indices = None
data_ntime = None
data_inttime = None

data_flag = None
data_flag_row = None

data_chan_freq=None # NB: Can handle only one SPW.
data_nchan = None
data_chanwidth = None

baseline_dict = None # Constructed in main()
weight_vector = None
weight_vector_flattened = None
per_bl_sig = None

# define some constants
uas2rad = 1e-6*jnp.pi/180.0/3600.0


def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument("ms", help="Input MS name")
    p.add_argument("col", help="Name of the data column from MS")
    return p


def make_baseline_dictionary(ant_unique):
    return dict([((x, y), np.where((data_ant1 == x) & (data_ant2 == y))[0]) for x in ant_unique for y in ant_unique if y > x])


def log_likelihood(fd, l, m, **kwargs):
    """
    Compute the loglikelihood function.
    NOTE: Not called directly by user code; the function signature must
          correspond to the requirements of the numerical sampler used.
    Parameters
    ----------
    theta : Input parameter vector

    Returns
    -------
    loglike : float
    """

    # set up the arrays necessary for forward modelling
    lm = jnp.array([[l, m]])
    gauss_shape = jnp.array([[l, m]]) # just to match fused_rime signature; not used
    #gauss_shape = jnp.array([[emaj, emin, pa]])
    stokes = jnp.array([[fd, 0, 0, 0]])

    # Use jax to predict vis 
    model_vis = fused_rime(lm, data_uvw, data_chan_freq, gauss_shape, stokes)

    # Compute chi-squared and loglikelihood
    diff = model_vis - data_vis
    chi2 = jnp.sum((diff.real*diff.real+diff.imag*diff.imag) * weight_vector)
    loglike = -chi2/2. - jnp.log(2.*jnp.pi/weight_vector_flattened).sum()

    return loglike


def main(args):

    global data_vis, data_ant1, data_ant2, basline_dict, data_uvw, data_nant, data_nbl, data_uniqtimes, data_uniqtime_indices, \
           data_ntime, data_inttime, data_flag, data_flag_row, data_chan_freq, data_nchan, data_chanwidth, weight_vector, \
           weight_vector_flattened, per_bl_sig

    ####### Read data from MS
    tab = pt.table(args.ms).query("ANTENNA1 != ANTENNA2"); # INI: always exclude autocorrs for our purposes
    data_vis = tab.getcol(args.col)
    data_ant1 = tab.getcol('ANTENNA1')
    data_ant2 = tab.getcol('ANTENNA2')
    ant_unique = np.unique(np.hstack((data_ant1, data_ant2)))
    baseline_dict = make_baseline_dictionary(ant_unique)

    # Read uvw coordinates; necessary for computing the source coherency matrix
    data_uvw = tab.getcol('UVW')

    # get data from ANTENNA subtable
    anttab = pt.table(args.ms+'::ANTENNA')
    stations = anttab.getcol('STATION')
    data_nant = len(stations)
    data_nbl = int((data_nant*(data_nant-1))/2)
    anttab.close()

    # Obtain indices of unique times in 'TIME' column
    data_uniqtimes, data_uniqtime_indices = np.unique(tab.getcol('TIME'), return_inverse=True)
    data_ntime = data_uniqtimes.shape[0]
    data_inttime = tab.getcol('EXPOSURE', 0, data_nbl)

    # Get flag info from MS
    data_flag = tab.getcol('FLAG')
    data_flag_row = tab.getcol('FLAG_ROW')
    data_flag = np.logical_or(data_flag, data_flag_row[:,np.newaxis,np.newaxis])

    tab.close()

    # get frequency info from SPECTRAL_WINDOW subtable
    freqtab = pt.table(args.ms+'::SPECTRAL_WINDOW')
    data_chan_freq = freqtab.getcol('CHAN_FREQ')[0]
    data_nchan = freqtab.getcol('NUM_CHAN')[0]
    data_chanwidth = freqtab.getcol('CHAN_WIDTH')[0,0]
    freqtab.close()

    # compute weight vector
    weight_vector = np.zeros(data_vis.shape) # weight_vector same for both real and imag parts of the vis.
    if noise_per_vis == None:
        per_bl_sig = np.zeros((data_nbl))
        bl_incr = 0;
        for a1 in np.arange(data_nant):
          for a2 in np.arange(a1+1,data_nant):cost function with weights
            per_bl_sig[bl_incr] = (1./corr_eff) * np.sqrt((sefds[a1]*sefds[a2])/(2*data_chanwidth*data_inttime[bl_incr])) 
            weight_vector[baseline_dict[(a1,a2)]] = 1./np.power(per_bl_sig[bl_incr], 2)
            bl_incr += 1;
    else:
        weight_vector[:] = 1./np.power(noise_per_vis, 2)

    weight_vector *= np.logical_not(data_flag)

    # convert ndarrays to jax arrays
    data_vis = jnp.asarray(data_vis)
    data_uvw = jnp.asarray(data_uvw)
    data_chan_freq = jnp.asarray(data_chan_freq)
    weight_vector = jnp.asarray(weight_vector)
    weight_vector_flattened = weight_vector.flatten()#[jnp.nonzero(weight_vector.flatten())]

    # set up priors
    prior_chain = PriorChain() \
            .push(UniformPrior('fd', Smin, Smax)) \
            .push(UniformPrior('l', dxmin*uas2rad, dxmax*uas2rad)) \
            .push(UniformPrior('m', dymin*uas2rad, dymax*uas2rad))

    # Run jaxns
    print("Starting nested sampling...")
    ns = NestedSampler(log_likelihood, prior_chain, num_live_points=nlive_factor*prior_chain.U_ndims)
    results = jit(ns)(key=random.PRNGKey(seed), termination_frac=termination_frac)

    print("Saving results to npz file...")
    save_results(results, 'output.npz')

    summary(results)
    plot_diagnostics(results, save_name='diagnostics.png')
    plot_cornerplot(results, save_name='cornerplot.png')

    return 0

if __name__ == '__main__':
    args = create_parser().parse_args()
    ret = main(args)
    sys.exit(ret)
