import argparse
import os
import sys

from contextlib import ExitStack
from loguru import logger
from stochastic import configure_loguru
from stochastic.preprocess.wsclean_utils import convert_tigger_cc, split_model, tigger_to_wsclean, lsm_cc_init
from stochastic.preprocess.predict_save_toms import wsclean_rime_to_MS

def create_parser():
    p = argparse.ArgumentParser()
    
    p.add_argument("--msname", "-ms", type=str, dest="msname", required=True, help="measurement set")
    
    p.add_argument("--sourcelist", "-sl", dest="sourcelist",type=str, help="wsclean source list", required=True)

    p.add_argument("--prefix", dest="prefix", type=str, help="prefix for output files", default="cc")

    p.add_argument("--threshold", "-th", dest="threshold", type=float, help="fit point sources brigter than this flux value, default=1e-2", default=1e-2)

    p.add_argument("--dummycol", "-dmc", dest="dummycol", type=str, help="If given the dummy model will be predicted here directly")

    return p

def _main(exitstack):
    logger.info("Running: the preprocess step with options " + " ".join(sys.argv[1:]))
    parser = create_parser()
    args = parser.parse_args()

    cclsm, freq0, logspi, spi_c = convert_tigger_cc(args.sourcelist, args.prefix)
    pmodel, gmodel = split_model(cclsm, args.prefix, threshold=args.threshold)

    initmodel ,_ = lsm_cc_init(pmodel, args.prefix, dummy=False, spi_c=spi_c)
    dummymodel, _ = lsm_cc_init(gmodel, args.prefix, dummy=True, spi_c=spi_c)
    skymodel = tigger_to_wsclean(gmodel, spi_c, freq0, args.prefix+"-dummy-", log_spi=str(logspi))

    logger.info(f"Freq0, logspi and spi_c are {freq0}, {logspi} and {spi_c}.")
    logger.info(f"Initial model and dummy model are {initmodel} and {dummymodel}")

    if args.dummycol:
        logger.info("Dummy col set, will predict the dummy visibilities. This may take a while depending the size of the measurement set!")
        wsclean_rime_to_MS(args.msname, dummymodel, freq0, args.dummycol, logspi=logspi)

    logger.success("Everything ran, will exit now!")


@logger.catch
def main():
    with ExitStack() as stack:
        _main(stack)
