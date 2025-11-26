#! /usr/bin/env python3

"""
Example script demonstrating how to use the FESutils library to calculate FES from COLVAR data.
"""

import os
from FESutils.fes_config import FESConfig
from FESutils.colvar_api import calculate_fes
from functools import partial

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    example_data_dir = os.path.join(base_dir, 'data')
    output_dir = os.path.join(base_dir, 'output')

    PartConfig = partial(FESConfig,
                        input_file=os.path.join(example_data_dir, 'COLVAR_merged.tgz'),
                         temp=300.0, plot=True, backup=False, bias_spec='opes.bias',
                         sigma=(0.05, 5.0),
                         cv_spec=('dT.z', 'tiltAvg'),
                         grid_bin=(50, 50),
                         )

    calculate_fes(PartConfig(outfile=os.path.join(output_dir, 'fes2d-rew.dat'),))

    calculate_fes(PartConfig(outfile=os.path.join(output_dir, 'fes2d-rew.dat'),
                             stride=500000))

    calculate_fes(PartConfig(outfile=os.path.join(output_dir, 'fes2d-rew.dat'),
                             blocks_num=8))

if __name__ == "__main__":
    main()
