import argparse
import os
from collections import OrderedDict

def create_parser():
    p = argparse.ArgumentParser()
    
    p.add_argument("--msname", "-ms", type=str, dest="msname", required=True, help="measurement set")
    
    p.add_argument("--datacol", "-dc", dest="datacol",type=str, help="datacol to fit", required=True)

    p.add_argument("--weightcol", "-wc", dest="weightcol", type=str, help="datacol to fit", default="WEIGHT")
    
    p.add_argument("--init-model", "-im", type=str, help="initial model file", required=True)

    p.add_argument("--dummy-model", "-dm", type=str, help="dummy model file")

    p.add_argument("--dummy-column", "-dmc", type=str, help="dummy model column")
    
    p.add_argument("--batch-size", "-bs", default=2016, type=int, help="Batch size")

    p.add_argument("--cellsize", "-cz", default=1, type=float, help="Image cellsize in arcseconds, please set this to the cellsize of your image")
    
    p.add_argument("--npix", "-npix", default=2048, type=float, help="Number of pixel to define image center")

    p.add_argument("--report-freq", "-rf", default=10, type=int, help="Reporting frequency")
    
    p.add_argument('--outdir', "-od", type=str, default="stochastic",  help="output directory, default is created in current working directory")

    p.add_argument('--name', "-name", type=str, default="out",  help="prefix to use for output files")

    p.add_argument("--one-corr", "-oc", help="use a single correlation",  action="store_true")

    p.add_argument("--feed-type", "-feed", help="feed type when handle multicorrelations", choices=["linear", "circular"], default="linear")

    p.add_argument("--wsclean", "-wsclean", help="fit wsclean spectra model",  action="store_true")

    p.add_argument("--gauss", "-gauss", help="fit a gaussian model",  action="store_true")
    
    p.add_argument("--log-spectra", "-logsp", help="use log spectra for wsclean components",  action="store_true")

    p.add_argument("--svrg", "-sv", help="use svrg",  action="store_true")

    p.add_argument("--noneg", "-noneg", help="non negative components, set components flux of negative components to zero",  action="store_true")

    p.add_argument("--drop-flags", "-dfgs", help="Flatten the data and drop flagged rows instead of just setting their weights to 0",  action="store_true")

    p.add_argument("--learning-rate", "-lr", dest="lr", type=float, nargs="+",
                        help="leaarning rates to. Either use a single value or list for each parameter (stokes, radec, shape_params)",  
                                default=[1e-2, 1e-5, 1e-2])

    p.add_argument("--frequency-range", "-fr", dest="fr", type=int, nargs="+",
                        help="start and end frequency channel to use. Most often frequency edges are completely flagged",  
                                default=[0, -1])

    p.add_argument("--error-functon", "-ef", dest="error_func", help="which function to use for error estimation, diagonals of Hessian or Fisher matrix", 
                      default="hessian", choices=["hessian", "fisher"])

    p.add_argument("--optimizer", "-op", dest="optimizer", help="which optimisation to use ADAM, SGD, Momentum", 
                      default="sgd", choices=["adam", "sgd", "momentum"])

    p.add_argument("--error-fraction", "-efrac", dest="efrac", 
                help="fraction of the data to use for hessian estimation, note a large value will cause an error",  default=0.02)

    p.add_argument("--freq0", "-freq0", type=float, 
                        help="Refrence frequency for spi fitting. We assume all the sources have the same reference, default is channel 0")

    p.add_argument("--epochs", "-eps", default=20, type=int, help="Number of epochs")

    p.add_argument("--niter", "-niter", default=2000, type=int, help="Max number of iteration per epochs")

    p.add_argument("--delta-loss", "-dl", default=1e-6, type=float, help="Minimum change in loss function to actiavte early stoppage")

    p.add_argument("--delta-epoch", "-de", default=5, type=int, help="Number of epochs after whcih early stoppage is activated")

    p.add_argument("--l2r", "-l2r", default=0, type=float, help="L2 regularisation parameter")

    p.add_argument("--l1r", "-l1r", default=0, type=float, help="L1 regularisation parameter")

    return p

def init_learning_rates(lr, gauss=False):
    """
    initialise a dictionary of learning rates
    args:
        lr (list)
    returns:
        dictionary (stokes, radec, shape_params, spi)
    """

    if gauss:
        assert len(lr)==1 or len(lr) == 4, "Either set a constant learning rate or set a different learning rate for each parameter"
        if len(lr) == 1:
            return dict(alpha=float(lr[0]),  shape_params=float(lr[0]), radec=float(lr[0]), stokes=float(lr[0]))
        else:
            return dict(alpha=float(lr[3]),  shape_params=float(lr[2]), radec=float(lr[1]), stokes=float(lr[0]))

    else: 
        assert len(lr)==1 or len(lr) == 3, "Either set a constant learning rate or set a different learning rate for each parameter"

        if len(lr) == 1:
            return dict(alpha=float(lr[0]), radec=float(lr[0]), stokes=float(lr[0]))
        else:
            return dict(alpha=float(lr[2]), radec=float(lr[1]), stokes=float(lr[0]))



