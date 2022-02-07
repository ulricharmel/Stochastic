# -*- coding: utf-8 -*-
# copy from codex-africanus wsclean

import math
import re
import warnings
import os

import numpy as np
import Tigger

from stochastic.preprocess.skymodel_utils import *

# define some constants
deg2rad = np.pi / 180.0;
arcsec2rad = deg2rad / 3600.0;
uas2rad = 1e-6 * deg2rad / 3600.0;

rad2deg = 180./np.pi

arcsec2deg = 1./3600.0
rad2arsec = 1./arcsec2rad


def deg2HMS(ra='', dec='', round=False):
  RA, DEC, rs, ds = '', '', '', ''
  if dec:
    if str(dec)[0] == '-':
      ds, dec = '-', abs(dec)
    deg = int(dec)
    decM = abs(int((dec-deg)*60))
    if round:
      decS = int((abs((dec-deg)*60)-decM)*60)
    else:
      decS = (abs((dec-deg)*60)-decM)*60
    DEC = '{0}{1}.{2}.{3}'.format(ds, deg, decM, decS)
  
  if ra:
    if str(ra)[0] == '-':
      rs, ra = '-', abs(ra)
    raH = int(ra/15)
    raM = int(((ra/15)-raH)*60)
    if round:
      raS = int(((((ra/15)-raH)*60)-raM)*60)
    else:
      raS = ((((ra/15)-raH)*60)-raM)*60
    RA = '{0}{1}:{2}:{3}'.format(rs, raH, raM, raS)
  
  if ra and dec:
    return (RA, DEC)
  else:
    return RA or DEC

hour_re = re.compile(r"(?P<sign>[+-]*)"
                     r"(?P<hours>\d+):"
                     r"(?P<mins>\d+):"
                     r"(?P<secs>\d+\.?\d*)")

deg_re = re.compile(r"(?P<sign>[+-])*"
                    r"(?P<degs>\d+)\."
                    r"(?P<mins>\d+)\."
                    r"(?P<secs>\d+\.?\d*)")




def _hour_converter(hour_str):
    m = hour_re.match(hour_str)

    if not m:
        raise ValueError("Error parsing '%s'" % hour_str)

    value = float(m.group("hours")) / 24.0
    value += float(m.group("mins")) / (24.0*60.0)
    value += float(m.group("secs")) / (24.0*60.0*60.0)

    if m.group("sign") == '-':
        value = -value

    return np.rad2deg(2.0 * math.pi * value)


def _deg_converter(deg_str):
    m = deg_re.match(deg_str)

    if not m:
        raise ValueError(f"Error parsing '{deg_str}'")

    value = float(m.group("degs")) / 360.0
    value += float(m.group("mins")) / (360.0*60.0)
    value += float(m.group("secs")) / (360.0*60.0*60.0)

    if m.group("sign") == '-':
        value = -value

    return np.rad2deg(2.0 * math.pi * value)


def arcsec2rad(arcseconds=0.0):
    return float(arcseconds) # np.rad2deg(float(arcseconds) / 3600.)

def orientation(pa_d=0):
    return float(pa_d)

def spi_converter(spi):
    if spi == '[]':
        return [0]
    else:
        base = [] # 0
        base.extend([float(c) for c in spi.strip("[]").split(",")])
        return base


_COLUMN_CONVERTERS = {
    'Name': str,
    'Type': str,
    'Ra': _hour_converter,
    'Dec': _deg_converter,
    'I': float,
    'SpectralIndex': spi_converter,
    'LogarithmicSI': lambda x: bool(x == "true" or x=="True"),
    'ReferenceFrequency': float,
    'MajorAxis': arcsec2rad,
    'MinorAxis': arcsec2rad,
    'Orientation': orientation,
} 
# lambda x=0.0: np.deg2rad(float(x)),


# Split on commas, ignoring within [] brackets
_COMMA_SPLIT_RE = re.compile(r',\s*(?=[^\]]*(?:\[|$))')

# Parse columm headers, handling possible defaults
_COL_HEADER_RE = re.compile(r"^\s*?(?P<name>.*?)"
                            r"(\s*?=\s*?'(?P<default>.*?)'\s*?){0,1}$")


def _parse_col_descriptor(column_descriptor):
    components = [c.strip() for c in column_descriptor.split(",")]

    columns = []
    defaults = []

    for column in components:
        m = _COL_HEADER_RE.search(column)

        if m is None:
            raise ValueError(f"'{column}' is not a valid column header")

        name, default = m.group('name', 'default')

        columns.append(name)
        defaults.append(default)

    return columns, defaults


def _parse_header(header):
    format_str, col_desc = (c.strip() for c in header.split("=", 1))

    if format_str != "Format":
        raise ValueError(f"'{format_str}' does not "
                         f"appear to be a wsclean header")

    return _parse_col_descriptor(col_desc)


def _parse_lines(fh, line_nr, column_names, defaults, converters):
    source_data = [[] for _ in range(len(column_names))]

    for line_nr, line in enumerate(fh, line_nr):
        components = [c.strip() for c in re.split(_COMMA_SPLIT_RE, line)]

        if len(components) != len(column_names):
            raise ValueError(f"line {line_nr} '{line}' should "
                             f"have {len(column_names)} components")

        # Iterate through each column's data
        it = zip(column_names, components, converters, source_data, defaults)

        for name, comp, conv, data_list, default in it:
            if not comp:
                if default is None:
                    try:
                        default = conv()
                    except Exception as e:
                        raise ValueError(
                            f"No value supplied for column '{name}' "
                            f"on line {line_nr} and no default was "
                            f"supplied either. Attempting to "
                            f"generate a default produced the "
                            f"following exception {e}")

                value = default
            else:
                value = comp

            data_list.append(conv(value))

    columns = dict(zip(*(column_names, source_data)))

    # Zero any spectral model's with nan/inf values
    try:
        name_column = columns["Name"]
        flux_column = columns["I"]
        spi_column = columns["SpectralIndex"]
        log_spi_column = columns["LogarithmicSI"]
    except KeyError as e:
        raise ValueError(f"WSClean Model File missing "
                         f"required column {str(e)}")

    it = zip(name_column, flux_column, spi_column, log_spi_column)

    # Zero flux and spi's in-place
    for i, (name, flux, spi, log_spi) in enumerate(it):
        good = True

        if not math.isfinite(flux):
            warnings.warn(f"Non-finite I {flux} encountered "
                          f"for source {name}. This source model will "
                          f"be zeroed.")
            good = False

        if not all(map(math.isfinite, spi)):
            warnings.warn(f"Non-finite SpectralIndex {spi} encountered "
                          f"for source {name}. This source model will "
                          f"be zeroed.")
            good = False

        if good:
            continue

        # np.log(1.0) = 0.0
        flux_column[i] = 1.0 if log_spi else 0.0

        for j in range(len(spi)):
            spi[j] = 0.0

    return list(columns.items())


def load(filename):
    """
    Loads wsclean component model.

    .. code-block:: python

        sources = load("components.txt")
        sources = dict(sources)  # Convert to dictionary

        I = sources["I"]
        ref_freq = sources["ReferenceFrequency"]

    See the `WSClean Component List
    <https://sourceforge.net/p/wsclean/wiki/ComponentList/>`_
    for further details.

    Parameters
    ----------
    filename : str or iterable
        Filename of wsclean model file or iterable
        producing the lines of the file.

    See Also
    --------
    africanus.model.wsclean.spectra

    Returns
    -------
    list of (name, list of values) tuples
        list of column (name, value) tuples
    """

    if isinstance(filename, str):
        fh = open(filename, "r")
        fh = iter(fh)
        close_filename = True
    else:
        fh = iter(filename)
        close_filename = False

    try:
        # Search for a header until we find a non-empty string
        header = ''
        line_nr = 1

        for headers in fh:
            header = headers.split("#", 1)[0].strip()

            if header:
                break

            line_nr += 1

        if not header:
            raise ValueError(f"'{filename}' does not contain "
                             f"a valid wsclean header")

        column_names, defaults = _parse_header(header)

        try:
            converters = [_COLUMN_CONVERTERS[n] for n in column_names]
        except KeyError as e:
            raise ValueError(f"No converter registered for column {str(e)}")

        return _parse_lines(fh, line_nr, column_names, defaults, converters)

    finally:
        if close_filename:
            fh.close()


def convert_tigger_cc(ccfile, label, n_components=None):
    """convert wsclean components to tigger model"""

    # import pdb; pdb.set_trace()
    listmodel = load(ccfile)
    spi_c = len(str(listmodel[5][1][0]).strip("[]").split(','))

    spi = f"spi"  #+curvature
    for i in range(spi_c-1):
        spi = spi + f" spi{i+2}"

    if n_components == None:
        n_components = len(listmodel[0][1])
    str_out = f"#format: name ra_d dec_d emaj_s emin_s pa_d i q u v {spi} freq0 \n"
    filemode = "w"  

    # import pdb; pdb.set_trace()

    for s in range(n_components):
        amp = listmodel[4][1][s]
        ra_d = listmodel[2][1][s]
        dec_d = listmodel[3][1][s]
        spi_v = str(listmodel[5][1][s]).strip("[]").replace(",", " ")
        freq0 = listmodel[7][1][s]
        logspi = listmodel[6][1][s]
        name = listmodel[0][1][s]
        emaj_s = listmodel[8][1][s]
        emin_s = listmodel[9][1][s]
        pa_d = listmodel[10][1][s]
        str_out += f"{name} {ra_d} {dec_d} {emaj_s} {emin_s} {pa_d} {amp} 0 0 0 {spi_v} {freq0}\n"


    outdir = os.path.dirname(ccfile)
    outdir = "." if outdir == '' else outdir

    outfile = outdir+"/%s-init-cc-model.txt"%label

    skymodel = open(outfile,filemode)
    skymodel.write(str_out)
    skymodel.close()

    os.system("tigger-convert %s -f --rename"%(outfile))

    return outfile[:-4]+".lsm.html", freq0, logspi, spi_c

def tigger_to_wsclean(lsmfile, spi_c, freq0, label, log_spi="false"):
    """Convert tigger catalog to wsclean format for crystalball"""

    str_out= f"Format = Name, Type, Ra, Dec, I, SpectralIndex, LogarithmicSI, ReferenceFrequency='1300000000', MajorAxis, MinorAxis, Orientation\n"
    filemode = "w"

    if spi_c==1:
        spi_string = str([])
    else:
        spi_string = str([" "]*(spi_c-1)).strip("[]")
    shape_string = f"{0},{0},{0}"

    model = Tigger.load(lsmfile)
    for src in model.sources:
        name =  src.name
        s_type = "GAUSSIAN" if src.shape else "POINT"
        ra, dec =  deg2HMS(ra=np.rad2deg(src.pos.ra), dec=np.rad2deg(src.pos.dec), round=False)
        if src.spectrum:
            spi = str(src.spectrum.spi[:]) # because of that silly, you can't use the first one #1:
        else:
            spi = spi_string
        if src.shape:
            ex,ey,pa = src.shape.ex*rad2arsec, src.shape.ey*rad2arsec, src.shape.pa*rad2deg
            shape = f"{ex},{ey},{pa}"
        else:
            shape = shape_string

        str_out += f"{name},{s_type},{ra},{dec},{src.flux.I},{spi},{log_spi},{freq0},{shape}\n"

    outdir = os.path.dirname(lsmfile)
    outdir = "." if outdir == '' else outdir

    outfile = outdir+"/%s-cc-model.txt"%label

    skymodel = open(outfile,filemode)
    skymodel.write(str_out)
    skymodel.close()

    return outfile


def split_model(ccmodel, label, threshold=1e-2):
    """split the wsclean component model into two skymodel
    -one containing points sources to fit and the other
    containing gaussian sources
    Args:
        ccmodel (str)
            wclean component model in Tigger lsm format
        label (str)
            predix label for the naming of the files
        threshold (flux)
            only fit point sources with flux > than this threshold
            ignore negative sources aswell
    returns:
        pointsource model and gaussian source model
    """

    model = Tigger.load(ccmodel)

    pointsources = []
    dummysources = []

    for src in model.sources:
        if src.shape == None:
            if src.get_attr("cluster_lead"):
                if src.cluster_flux > threshold:
                    src.flux.I = src.cluster_flux # set the init flux to the cluster flux then
                    pointsources.append(src)
                elif src.cluster_flux > 0:
                    dummysources.append(src)
                else:
                    dummysources.append(src) #pass
        else:
            dummysources.append(src)

    outdir = os.path.dirname(ccmodel)
    outdir = "." if outdir == '' else outdir
    
    model.setSources(pointsources)
    pmodel = outdir+"/%s-point-sources.lsm.html"%label
    model.save(pmodel)

    model.setSources(dummysources)
    gmodel = outdir+"/%s-dummy-sources.lsm.html"%label
    model.save(gmodel)

    return pmodel, gmodel

def lsm_cc_init(tiggermodel, label, dummy=False, spi_c=1):
    """
    Initiliaise model from tigger file of clean component
    """

    # import pdb; pdb.set_trace()

    outdir = os.path.dirname(tiggermodel)
    outdir = "." if outdir == '' else outdir

    model  = Tigger.load(tiggermodel)
    sources = []
    freq0 = None

    # import pdb; pdb.set_trace()

    for src in model.sources:
        if src.get_attr("spectrum"):
            spi = src.spectrum.spi
            freq0 = src.spectrum.freq0
            if type(spi) != list:
                spi = [spi]
            if len(spi) != spi_c:
                n_ext = spi_c - len(spi)
                spi_ext = [0]*n_ext
                spi.extend(spi_ext)
        else:
            spi = [0]*spi_c

        ra, dec = rad2deg*src.pos.ra, rad2deg*src.pos.dec

        if dummy:
            flux = src.flux.I 
            if src.get_attr("shape"):
                ex, ey, pa = src.shape.ex*rad2arsec, src.shape.ey*rad2arsec, src.shape.pa*rad2deg
            else:
                ex=ey=pa = 0
            source = [flux, ra, dec, ex, ey, pa]
            source.extend(spi)
            sources.append(source)
        else:
            flux = src.cluster_flux
            spi = [0]*spi_c
            source = [flux, ra, dec]
            source.extend(spi)
            sources.append(source)

    model = np.array(sources)
    if dummy:
        initmodel = outdir+"/%s-dummy-cc-model.npy"%label
    else:
        initmodel = outdir+"/%s-init-point-cc-model.npy"%label


    np.save(initmodel, model)

    return initmodel, freq0


def convert_numpy(listmodel, label):
    """convert list model to numpy"""

    n_components = len(listmodel[0,1])
    sources = np.zeros((n_components, 6))

    for s in range(n_components):
        sources[s,0] = listmodel[4,s]
        sources[s,1] = listmodel[2,s]
        sources[s,2] = listmodel[3,s]
        sources[s,3] = listmodel[8,s]
        sources[s,4] = listmodel[9,s]
        sources[s,5] = listmodel[10,s]

    outdir = os.path.dirname(listmodel)
    outdir = "." if outdir == '' else outdir

    initmodel = outdir+"/%s-init-cc-model.npy"%label
    np.save(initmodel, sources)

