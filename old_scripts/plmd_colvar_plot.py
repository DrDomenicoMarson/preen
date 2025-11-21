#!/usr/bin/env python3

from argparse import ArgumentParser
from MyPlumed.plmd_functions import plot_colvar

parser = ArgumentParser()
parser.add_argument("--basedir", dest="basedir",
                    type=str, metavar="./", default="./",
                    help="base directory where to find the walkers directories")
parser.add_argument("--colvarbasen", dest="colvarbasen",
                    type=str, metavar="COLVAR", default="COLVAR",
                    help="base name for COLVAR files")
parser.add_argument("--marker", dest="marker",
                    type=str, metavar=",", default=",",
                    help="symbol for the plots, def ',', but you can use '.' that's bigger")
parser.add_argument("--ploteach", dest="ploteach", action="store_true", default=False,
                    help="plot in every directory, not only the merged file (in multidir runs)")
parser.add_argument("--ext", dest="ext",
                    type=str, metavar="png", default="png",
                    help="extension (format) of the output file")
opt = parser.parse_args()


plot_colvar(
    basedir=opt.basedir,
    colvar_base = opt.colvarbasen,
    marker=opt.marker,
    ext=opt.ext,
    plot_each=opt.ploteach,
    )
