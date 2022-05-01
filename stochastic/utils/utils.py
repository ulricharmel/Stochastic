#!/usr/bin/env python

# INI: Adapted from casacore.tables.msutil.msregularize:
import os
import sys
import argparse
import numpy as np
import pyrap.tables as pt
import json
from loguru import logger

def save_output(filname, mydict, convert=False, grad=False, gauss=False):
    if convert:
        mydict_convert = {}
        for key in mydict:
            mydict_convert[key] = np.asarray(mydict[key]).tolist()
    elif grad:
        mydict_convert = {}
        allkeys = ["alpha", "shape_params", "radec", "stokes"] if gauss else ["alpha", "radec", "stokes"]
        for key in mydict:
               mydict_convert[key] = {}
               for key2 in allkeys:
                   values = [x[key2] for x in mydict[key]]
                   mydict_convert[key][key2] = np.asarray(values).tolist()
    else:
        mydict_convert = mydict

    tf = open(filname, "w")
    json.dump(mydict_convert,tf)
    tf.close()

def create_output_dirs(outdir):
    """create ouput directory for log files and fitted information"""
    if os.path.isdir(outdir):
        logger.info("Output directory already exit from previous run!")
    else:
        os.mkdir(outdir)

def regularize_ms(msname):
    """ Regularize an MS

    The output MS will have the same number of baselines for each time stamp.
    All new rows are fully flagged. 

    First, missing rows are written into a separate MS <msname>_missing.MS,
    which is concatenated with the original MS and sorted in order of TIME,
    DATADESC_ID, ANTENNA1, ANTENNA2 to form a new regular MS. This MS is
    a 'deep' copy copy of the original MS.

    If no rows were missing, no new MS is created.
    """

    msprefix = msname.rsplit('.', maxsplit=1)[0]

    # Get all baselines
    tab = pt.table(msname)

    # Make list of missing antennas
    ant1=tab.getcol('ANTENNA1')
    ant2=tab.getcol('ANTENNA2')

    ants_present = np.unique(np.hstack((ant1,ant2)))

    anttab=pt.table(msname+'::ANTENNA')
    ants_anttab = np.arange(anttab.nrows())
    anttab.close()

    ants_missing = [x for x in ants_anttab if x not in ants_present]
    ants_missing = np.array(ants_missing)

    combos = []
    for ii in ants_missing:
        for jj in ants_present:
            if ii > jj:
                combos.append((jj,ii))
            else:
                combos.append((ii,jj))

    for ii in np.arange(0, ants_missing.shape[0]):
        for jj in np.arange(ii+1, ants_missing.shape[0]):
            combos.append((ants_missing[ii], ants_missing[jj]))

    combos.sort()
    print(f'Missing baselines: {combos}')

    # Ensure the unique antenna permutations in $t1 also contain antennas in ANTENNA that are missing from MAIN
    t1=tab.sort('unique ANTENNA1,ANTENNA2')
    t1.copy('uniqants.ms',deep=True)
    t1.close()

    # Assign values to be used in TaQL
    data = tab.getcol('DATA')
    nrows = data.shape[0]
    nchan = data.shape[1]
    ncorr = data.shape[2]

    t1=pt.table('uniqants.ms')
    for bl in combos:
        pt.taql('insert into $t1 (ANTENNA1,ANTENNA2,DATA,FLAG) VALUES ($bl[0], $bl[1], array([0+0i], [$nchan, $ncorr]), array([True], [$nchan, $ncorr]))')

    if 'MODEL_DATA' in tab.colnames():
        pt.taql('update $t1 set MODEL_DATA=array([0+0i], [$nchan, $ncorr])')
    if 'CORRECTED_DATA' in tab.colnames():
        pt.taql('update $t1 set CORRECTED_DATA=array([0+0i], [$nchan, $ncorr])')
    if 'WEIGHT' in tab.colnames():
        pt.taql('update $t1 set WEIGHT=array([1], [$ncorr])')
    if 'SIGMA' in tab.colnames():
        pt.taql('update $t1 set SIGMA=array([1], [$ncorr])')
    if 'WEIGHT_SPECTRUM' in tab.colnames():
        pt.taql('update $t1 set WEIGHT_SPECTRUM=array([1], [$nchan, $ncorr])')
    if 'SIGMA_SPECTRUM' in tab.colnames():
        pt.taql('update $t1 set SIGMA_SPECTRUM=array([1], [$nchan, $ncorr])')

    t1.sort('unique ANTENNA1,ANTENNA2').rename('uniqants_sorted.ms')
    t1.close()

    t1 = pt.table('uniqants_sorted.ms')
    nadded = 0

    # Iterate in time and band over the MS
    for tsub in tab.iter(['TIME','DATA_DESC_ID']):
        nmissing = t1.nrows() - tsub.nrows()

        if nmissing < 0:
            raise ValueError("A time/band chunk has too many rows")
        elif nmissing > 0:
            ant1 = list(tsub.getcol('ANTENNA1'))
            ant2 = list(tsub.getcol('ANTENNA2'))

            # select baseline permutations that are missing from the current 'tsub'
            t2 = pt.taql('select from $t1 where !any(ANTENNA1 == $ant1 && ANTENNA2 == $ant2)')
            if t2.nrows() != nmissing:
                raise ValueError("A time/band chunk behaves strangely")

            # for the first iteration, create a new table and open for writing
            if nadded == 0:
                tnew = t2.copy(msprefix+"_missing.MS", deep=True)
                tnew = pt.table(msprefix+"_missing.MS", readonly=False)
            else:
                t2.copyrows(tnew)

            # set the correct time and band in the new rows.
            tnew.putcell('TIME', np.arange(nadded, nadded+nmissing), tsub.getcell('TIME',0))
            tnew.putcell('DATA_DESC_ID', np.arange(nadded, nadded+nmissing), tsub.getcell('DATA_DESC_ID',0))
            nadded += nmissing # update nadded

    # close tables
    t1.close()

    # combine the new table with the existing one
    if nadded > 0:
        # initialize DATA with zeros and flag the added rows
        pt.taql('update $tnew set DATA=0+0i')
        pt.taql('update $tnew set FLAG=True')
        pt.taql('update $tnew set FLAG_ROW=True')

        tcombs = pt.table([tab, tnew]).sort('TIME,DATA_DESC_ID,ANTENNA1,ANTENNA2')
        tcombs.copy(msprefix+"_regularized.MS", deep=True)

        # close and/or delete temporary tables
        tnew.close()
        pt.tabledelete(msprefix+"_missing.MS") # tnew
        t2.close()
        tcombs.close()

    # close and delete tables before exiting
    t1.close()
    pt.tabledelete('uniqants_sorted.ms')
    pt.tabledelete('uniqants.ms')
    tab.close()

    if nadded > 0:
        print(msprefix+"_regularized.MS", 'contains the regularized MS.')
    else:
        print(f"{msname} is already regularized. No changes made.")

def main():
    # Parse the input arguments
    parser = argparse.ArgumentParser(description="Regularize an MS so that all timestamps have entries for all baselines")
    parser.add_argument('msname', type=str, help="Name of the input MS to be regularized")
    args = parser.parse_args()

    regularize_ms(args.msname)

if __name__ == '__main__':

    ret=main()
    sys.exit(ret)

