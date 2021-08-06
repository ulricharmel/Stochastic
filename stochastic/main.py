#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, time
import argparse
import pyrap.tables as pt
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax, jit, random
from jax.test_util import check_grads
import time

import traceback

try:
	import ipdb as pdb
except:
	import pdb

from stochastic.utils.utils import create_output_dirs, save_output
from stochastic.utils.parser import create_parser
from stochastic.data_handling.read_data import load_data, load_model 

from stochastic.essays.rime.jax_rime import fused_rime, fused_rime_sinlge_corr

LEARNING_RATE = 1e-1
EPOCHS = 2

def forward(params, data_uvw, data_chan_freq):
    """
    Compute the model visibilities using jax rime
    Args:
        Params (dictionary)
            flux, lm, shape parameters (ex, ey, pa)
        data_uvw (array)
            uvw coordinates from the measurement set
        data_chan_freq (array)
            frequencies from the measurement set
    Returns:  
        Model visibilities (array)
    """

    lm = params['lm']
    shape_params = params["shape_params"]
    stokes = params["stokes"]

    model_vis = fused_rime_sinlge_corr(lm, data_uvw, data_chan_freq, shape_params, stokes)

    return model_vis

def loss_fn(params, data_uvw, data_chan_freq, data):
    """
    Compute the loss function
    Args:
        Params list with a dictionary)
            flux, lm, shape parameters (ex, ey, pa)
        data_uvw (array)
            uvw coordinates from the measurement set
        data_chan_freq (array)
            frequencies from the measurement set
        data (array)
            data visibilities
    Returns:  
        loss function
    """
    import pdb; pdb.set_trace()
    model_vis = forward(params, data_uvw, data_chan_freq)
    diff = data - model_vis

    return jnp.mean(diff.real*diff.real+diff.imag*diff.imag)

# @jax.jit
def update(params, data_uvw, data_chan_freq, data):
    loss, grads = jax.value_and_grad(loss_fn)(params, data_uvw, data_chan_freq, data)
    # check_grads(loss_fn, (params, data_uvw, data_chan_freq, data), order=2)
    # Note that `grads` is a pytree with the same structure as `params`.
    # `jax.grad` is one of the many JAX functions that has
    # built-in support for pytrees.
    
    import pdb; pdb.set_trace()

    # This is handy, because we can apply the SGD update using tree utils:
    return jax.tree_multimap(lambda p, g: p - LEARNING_RATE * g, params, grads), loss


def train(params, data_uvw, data_chan_freq, data, batch_size, outdir):
    """
    Use Stochastic gradient decent and try to fit for the parameters
    Compute the loss function
    Args:
        Params list with a dictionary)
            flux, lm, shape parameters (ex, ey, pa)
        data_uvw (array)
            uvw coordinates from the measurement set
        data_chan_freq (array)
            frequencies from the measurement set
        data (array)
            data visibilities
        batch_size (int)
            number of visibilities to train in one go
        outdir (str)
            save the fitted parameters and loss function here
    Returns:  
        fitted parameters
    """

    # For now we will aussume a perfect measuremnt set

    nsamples = data.shape[0]
    assert nsamples%batch_size == 0, "Please choose a batch size that equaly divides the number of rows"
    
    inds = np.array([(i,i+batch_size) for i in range(0, nsamples, batch_size)])
    num_batches = len(inds)

    loss_avg = {}
    for epoch in range(EPOCHS):
        start_time = time.time()
        loss_avg["epoch-%d"%epoch] = []
        arr = np.random.permutation(num_batches)
        d_inds = inds[arr]

        for batch in range(num_batches):
            ts, te = d_inds[batch]
            d_uvw = data_uvw[ts:te]
            d_freq = data_chan_freq.copy()
            d_vis = data[ts:te]
            params, loss_i =  update(params, d_uvw, d_freq, d_vis)
            loss_i = np.asarray(loss_i)
            print(loss_i)
            loss_avg["epoch-%d"%epoch].append(loss_i)

        mean_loss = sum(loss_avg["epoch-%d"%epoch])/len(loss_avg["epoch-%d"%epoch])

        epoch_time = time.time() - start_time
        print("Epoch {} in {:0.2f} sec and loss is {:0.2f}".format(epoch, epoch_time, mean_loss))
    
    save_output(outdir+"/params.json", params, convert=True)
    save_output(outdir+"/loss.json", loss_avg, convert=True)

    return

def main():
    
    parser = create_parser()
    args = parser.parse_args()
    
    try:
        create_output_dirs(args.outdir)
        data_vis, data_uvw, data_chan_freq = load_data(args.msname, args.datacol, args.one_corr)
        params = load_model(args.init_model)
        train(params, data_uvw, data_chan_freq, data_vis, args.batch_size, args.outdir)

    except:
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)

