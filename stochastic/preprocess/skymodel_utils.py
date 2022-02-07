import sys
import os
import pyrap.tables as pt
import numpy as np
import json

def lsmconvert(skymodel):
	"""
	Converting textfile to lsm.html format
	"""
	os.system("tigger-convert %s -f"%(skymodel))
	return "%s.lsm.html"%(skymodel[:-4])

def get_ms_freq0(MS):
	freqtab = pt.table(MS+"/SPECTRAL_WINDOW", ack=False)
	chan_freq = freqtab.getcol("CHAN_FREQ").squeeze()
	freqtab.close()
	if chan_freq.size == 1:
		return chan_freq
	else:
		return np.mean(chan_freq)

def get_field_center(MS):
	t = pt.table(MS+"/FIELD", ack=False)
	phase_centre = (t.getcol("PHASE_DIR"))[0,0,:]
	t.close()
	return phase_centre[0], phase_centre[1]

def save_text_model(msname, sources, outfile, prefix="src", freq0=None):
    """Two source model to test the stochastic code"""

    str_out = "#format: name ra_d dec_d  emaj_s emin_s pa_d i q u v spi freq0 \n"
    filemode = "w"

    if freq0==None:
        freq0 = get_ms_freq0(msname)

    for i in range(len(sources)):
        amp, ra_d, dec_d, emaj_s, emin_s, pa_d, spi = sources[i, 0:7]
        name = prefix+str(i)
        # if amp>0:
        str_out += "%s %.12g %.12g %.12g %.12g %.5f %.5g 0 0 0 %.5f %f\n"%(name, ra_d, dec_d, emaj_s, emin_s, pa_d, amp, spi, freq0)

    skymodel = open(outfile,filemode)
    skymodel.write(str_out)
    skymodel.close()
    lsm = lsmconvert(outfile)

    return lsm

def save_text_model_polyspi(msname, sources, outfile, prefix="src", freq0=None):
    """Two source model to test the stochastic code"""

    nparams = sources.shape[1]
    spi_c = nparams - 6
    spi = f"spi"  #+curvature

    for i in range(spi_c-1):
        spi = spi + f" spi{i+2}"

    str_out = f"#format: name ra_d dec_d emaj_s emin_s pa_d i q u v {spi} freq0 \n"
    filemode = "w"

    if freq0==None:
        freq0 = get_ms_freq0(msname)

    for i in range(len(sources)):
        amp, ra_d, dec_d, emaj_s, emin_s, pa_d = sources[i, 0:6]
        spi_v = str(sources[i,6:]).strip("[]")
        name = prefix+str(i)
        # if amp>0:
        str_out += f"{name} {ra_d} {dec_d} {emaj_s} {emin_s} {pa_d} {amp} 0 0 0 {spi_v} {freq0}\n"
        # "%s %.12g %.12g %.12g %.12g %.5f %.5g 0 0 0 %.5f %f\n"%(name, ra_d, dec_d, emaj_s, emin_s, pa_d, amp, spi, freq0)

    skymodel = open(outfile,filemode)
    skymodel.write(str_out)
    skymodel.close()
    lsm = lsmconvert(outfile)

    return lsm

def best_json_to_tigger(msname, paramsfile, nparams, freq0):
    """load the output paramsfile and save it as a tigger skymodel"""
    tf = open(paramsfile)
    params = json.load(tf)
    nsources = len(params["radec"])
    model = np.zeros((nsources, nparams))
    model[:,0] = np.asarray(params["stokes"])[:,0]
    model[:,1:3] = np.asarray(params["radec"])
    # model[:,3:6] = np.asarray(params["shape_params"])
    model[:,6:] = np.asarray(params["alpha"])

    outfile = paramsfile[:-5]+".txt" 
    save_text_model_polyspi(msname, model, outfile, prefix="src", freq0=freq0)


