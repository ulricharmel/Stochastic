import argparse
import os

def create_parser():
    p = argparse.ArgumentParser()
    
    p.add_argument("--msname", "-ms", type=str, dest="msname", required=True, help="measurement set")
    
    p.add_argument("--datacol", "-dc", dest="datacol",type=str, help="datacol to fit")
    
    p.add_argument("--init-model", "-im", type=str, help="initial model file", required=True)
    
    p.add_argument("--batch_size", "-bs", default=2016, type=int, help="Batch size")
    
    p.add_argument('--outdir', "-od", type=str, default="stochastic",  help="output directory, default is created in current working directory")

    p.add_argument("--one-corr", "-oc", help="use a single correlation",  action="store_true")

    return p

