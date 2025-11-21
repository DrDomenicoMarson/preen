#!/usr/bin/env python3

from argparse import ArgumentParser
from MyPlumed.plmd_functions import merge_colvar_files

parser = ArgumentParser()
parser.add_argument("--basedir", dest="basedir",
                    type=str, metavar="./", default="./",
                    help="base directory where to find the walkers directories")
parser.add_argument("--colvarbasen", dest="colvarbasen",
                    type=str, metavar="COLVAR", default="COLVAR",
                    help="base name for COLVAR files")
parser.add_argument("--perc2discard", dest="perc2discard",
                    type=float, metavar="10.0", default=10.0,
                    help="discard that amount %% from the beginning of each file")
parser.add_argument("--keeporder", dest="keeporder", action="store_true", default=False,
                    help="preserve the order of the frames (first line from all the files, than the next and so on...)")
opt = parser.parse_args()


merge_colvar_files(basedir = opt.basedir,
                   colvar_base = opt.colvarbasen,
                   perc_colvar_to_discard = opt.perc2discard,
                   keep_order=opt.keeporder)
