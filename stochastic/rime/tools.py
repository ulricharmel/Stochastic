import jax.numpy as jnp
from jax import jit, vmap
from jax import lax, ops
from jax.experimental import loops
from scipy.constants import c as lightspeed

# define some constants
deg2rad = jnp.pi / 180.0;
arcsec2rad = deg2rad / 3600.0;
uas2rad = 1e-6 * deg2rad / 3600.0;
rad2deg = 1./deg2rad
rad2arcsec = 1./arcsec2rad
deg2arcsec = deg2rad*rad2arcsec

minus_two_pi_over_c = 2*jnp.pi/lightspeed # remove -2
ra0 = 0 # overwite this in main
dec0 = 0

freq0 = 1e9 # reference frequency for source spectrum

# only cellsize should be necessary
cellsize = 1.
cx, cy = 1024, 1024
fov = 1.


@jit
def lm2radec(lm):
    #let say the ra and dec in radians
    source = lm.shape[0]
    radec = jnp.empty((source, 2), lm.dtype)
    
    sin_pc_dec = jnp.sin(dec0)
    cos_pc_dec = jnp.cos(dec0) 

    def body(s, radec):
        l,m = lm[s]
        n = jnp.sqrt(1.0 -l**2 -m**2)

        ra = ra0 + jnp.arctan2(l, (n*cos_pc_dec - m*sin_pc_dec))
        dec = jnp.arcsin(m*cos_pc_dec + n*sin_pc_dec)

        radec = ops.index_update(radec, (s, 0), ra)
        radec = ops.index_update(radec, (s, 1), dec)

        return radec
     
    return lax.fori_loop(0, source, body, radec)
        

@jit
def radec2lm(radec):
    source = radec.shape[0]
    lm = jnp.empty((source, 2), radec.dtype)

    def body(s, lm):
        ra, dec = radec[s]*deg2rad
        delta_ra = ra - ra0
        l = jnp.cos(dec)*jnp.sin(delta_ra)
        m = jnp.sin(dec)*jnp.cos(dec0) - jnp.cos(dec)*jnp.sin(dec0)*jnp.cos(delta_ra)
   
        lm = ops.index_update(lm, (s, 0), l)
        lm = ops.index_update(lm, (s, 1), m)

        return lm
  
    return lax.fori_loop(0, source, body, lm)

@jit
def source_spectrum(alpha, freqs):
    # for now we assume the refrencece frequency is the first frequency
    # freq0 is imported from tools.py and it value is updated from main
    frf = freqs/freq0
    logfr = jnp.log10(frf)
    
    spectrum = frf ** sum([a * jnp.power(logfr, n) for n, a in enumerate(alpha)])
    return spectrum[None, :]

@jit 
def wsclean_spectra(flux, alpha, freqs):
    one = flux.dtype.type(1.0)
    frf = freqs/freq0 - one
    spectrum = flux + sum([a * jnp.power(frf, n+1) for n, a in enumerate(alpha)])

    return spectrum[None, :]

@jit 
def wsclean_log_spectra(flux, alpha, freqs):

    logfr = jnp.log(freqs/freq0)
    log_spectrum = jnp.log(flux) + sum([a * jnp.power(logfr, n+1) for n, a in enumerate(alpha)])

    spectrum = jnp.exp(log_spectrum)

    return spectrum[None, :]

@jit
def frac2lm(xy):
    source = xy.shape[0]
    lm = jnp.empty((source, 2), xy.dtype)

    def body(s, lm):
        l = xy[s,0]*fov*arcsec2rad*cellsize
        m = xy[s,1]*fov*arcsec2rad*cellsize

        lm = ops.index_update(lm, (s, 0), l)
        lm = ops.index_update(lm, (s, 1), m)

        return lm
  
    return lax.fori_loop(0, source, body, lm)

@jit
def lm2frac(lm):
    source = lm.shape[0]
    xy = jnp.empty((source, 2), lm.dtype)

    def body(s, xy):
        x = lm[s,0]/(fov*cellsize*arcsec2rad)
        y = lm[s,1]/(fov*cellsize*arcsec2rad)

        xy = ops.index_update(xy, (s, 0), x)
        xy = ops.index_update(xy, (s, 1), y)

        return xy
  
    return lax.fori_loop(0, source, body, xy)


@jit
def pixel2lm(xy):
    source = xy.shape[0]
    lm = jnp.empty((source, 2), xy.dtype)

    def body(s, lm):
        l = -(xy[s,0]-cy)*cellsize*arcsec2rad
        m = (xy[s,1]-cx)*cellsize*arcsec2rad
   
        lm = ops.index_update(lm, (s, 0), l)
        lm = ops.index_update(lm, (s, 1), m)

        return lm
  
    return lax.fori_loop(0, source, body, lm)

@jit
def lm2pixel(lm):
    source = lm.shape[0]
    xy = jnp.empty((source, 2), lm.dtype)

    def body(s, xy):
        x = -lm[s,0]/(cellsize*arcsec2rad) + cx
        y = lm[s,1]/(cellsize*arcsec2rad) + cy
   
        xy = ops.index_update(xy, (s, 0), x)
        xy = ops.index_update(xy, (s, 1), y)

        return xy
  
    return lax.fori_loop(0, source, body, xy)


def radec2pixel(radec):
    return lm2pixel(radec2lm(radec))

def pixel2radec(xy):
    return lm2radec(pixel2lm(xy))*rad2deg

def radec2fraclm(radec):
    return lm2frac(radec2lm(radec))

def fraclm2radec(xy):
    return lm2radec(frac2lm(xy))*rad2deg


@jit
def radec2lm_deriv(radec):
    source = radec.shape[0]
    radec_d = jnp.empty((source, 4), radec.dtype)

    def body(s, radec_d):
        ra, dec = radec[s]*deg2rad
        delta_ra = ra - ra0
        d_ra_abs = jnp.abs(ra -ra0)
        d_dec_abs = jnp.abs(dec - dec0)

        lr = deg2rad*jnp.cos(dec)*jnp.cos(delta_ra)*d_ra_abs  
        ldec = -deg2rad*jnp.sin(delta_ra)*jnp.sin(dec)*d_dec_abs

        mr = deg2rad*jnp.sin(dec0)*jnp.cos(dec)*jnp.sin(delta_ra)*d_ra_abs
        mdec = deg2rad*(jnp.sin(dec0)*jnp.cos(delta_ra)*jnp.sin(dec) 
                    + jnp.cos(dec)*jnp.cos(dec))*d_dec_abs
    
        radec_d = ops.index_update(radec_d, (s,0), lr)
        radec_d = ops.index_update(radec_d, (s,1), ldec)
        radec_d = ops.index_update(radec_d, (s,2), mr)
        radec_d = ops.index_update(radec_d, (s,3), mdec)

        return radec_d 
  
    return lax.fori_loop(0, source, body, radec_d)

@jit
def gaussian_shape_deriv(uvw, frequency, shape_params):
    
    fwhm = 2. * jnp.sqrt(2. * jnp.log(2.))
    fwhminv = 1. / fwhm
    gauss_scale = fwhminv * jnp.sqrt(2.) * jnp.pi / lightspeed
    dtype = jnp.result_type(*(jnp.dtype(a.dtype.name) for
                                a in (uvw, frequency, shape_params)))

    u = uvw[None, :, 0, None]
    v = uvw[None, :, 1, None]

    scaled_freq_2 = 2.*(frequency*gauss_scale)**2

    emaj, emin, pa = shape_params[0]

    emin_ = u*jnp.cos(pa) - v*jnp.sin(pa)
    emaj_ = u*jnp.sin(pa) + v*jnp.cos(pa)

    d_emin = (emin*emin_**2)*scaled_freq_2[None, None, :]

    d_emax = (emaj*emaj_**2)*scaled_freq_2[None, None, :]

    d_pa = ((emaj**2 - emin**2)*emin_*emaj_
                        *scaled_freq_2[None, None,:])

    return d_emax, d_emin, d_pa

@jit
def phase_delay_deriv_lm(lm, uvw, frequency):
    out_dtype = jnp.result_type(lm, uvw, frequency, jnp.complex64)

    one = lm.dtype.type(1.0)
    neg_two_pi_over_c = lm.dtype.type(minus_two_pi_over_c)
    complex_one = out_dtype.type(1j)

    l = lm[:, 0, None, None]  # noqa
    m = lm[:, 1, None, None]

    u = uvw[None, :, 0, None]
    v = uvw[None, :, 1, None]
    w = uvw[None, :, 2, None]

    dn = jnp.sqrt(one - l**2 - m**2)

    dphase_l = (neg_two_pi_over_c *
                    complex_one*(u - w*l/dn) *
                    frequency[None, None, :])

    dphase_m = (neg_two_pi_over_c *
                    complex_one*(v - w*m/dn) *
                    frequency[None, None, :])

    return dphase_l, dphase_m