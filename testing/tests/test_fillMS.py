import sys
import argparse
import numpy as np
import pyrap.tables as pt

import jax.config
import jax.numpy as jnp

from stochastic.essays.rime.jax_rime import fused_rime, fused_rime_sinlge_corr

# define some constants
deg2rad = jnp.pi / 180.0;
arcsec2rad = deg2rad / 3600.0;
uas2rad = 1e-6 * deg2rad / 3600.0;

# define global variables
lm = jnp.array([[0*uas2rad, 0*uas2rad]]) # dims: nsrc x 2
shape_params = jnp.array([[40*arcsec2rad, 60*arcsec2rad, jnp.deg2rad(0)]])
stokes = jnp.array([[10,0,0,0]]) # dims: nsrc x 4
stokes2 = jnp.array([[10]])

noise_per_vis = 0.1 # error on each visibility in Jy. None -> fit it
sefds = np.array([6000,1300,560,220,2000,1600,5000,1600,4500]) # station SEFDs in Jy - from EHT2017_station_info
corr_eff = 0.88

deg2rad = lm.dtype.type(deg2rad)
arcsec2rad = lm.dtype.type(arcsec2rad)
uas2rad = lm.dtype.type(uas2rad)

def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument("ms", help="Input MS name")
    #p.add_argument('--hypo', type=int, choices=[0,1], required=True)
    return p

def main(args):

    if args.ms[-1] == '/':
        args.ms = args.ms[:-1]

    # get some observation settings from the MS
    tab = pt.table(args.ms)
    uvw = tab.getcol('UVW')
    tab.close()

    # get frequency info from SPECTRAL_WINDOW subtable
    freqtab = pt.table(args.ms+'::SPECTRAL_WINDOW')
    freq = freqtab.getcol('CHAN_FREQ')[0]
    freqtab.close()

    # Use jax to predict vis 
    vis = fused_rime(lm, uvw, freq, shape_params, stokes)
    
    # vis2 = fused_rime_sinlge_corr(lm, uvw, freq, shape_params, stokes2)

    # print(vis[:,:,0]-vis2)

    # add noise
    noise = np.random.normal(0, noise_per_vis, size=vis.shape) + 1j*np.random.normal(0, noise_per_vis, size=vis.shape)
    noise = jnp.asarray(noise)
    # vis = vis + noise

    tab = pt.table(args.ms, readonly=False)
    tab.putcol('DATA', np.array(vis)) # convert JAX array to numpy array
    tab.close()

    return 0

if __name__ == '__main__':
    jax.config.update("jax_enable_x64", True)
    args = create_parser().parse_args()
    ret = main(args)
    sys.exit(ret)
