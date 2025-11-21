#! /usr/bin/env python3

"""
Example script demonstrating how to use the FESutils library to calculate FES from COLVAR data.
"""

import os
from FESutils.fes_config import FESConfig
from FESutils.api import calculate_fes
from functools import partial

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    example_data_dir = os.path.join(base_dir, 'data')
    output_dir = os.path.join(base_dir, 'output')

    PartConfig = partial(FESConfig,
                         filename=os.path.join(example_data_dir, 'COLVAR_merged.tgz'),
                         temp=300.0, plot=True, backup=True, bias_spec='opes.bias',
                         )

    calculate_fes(PartConfig(
        outfile=os.path.join(output_dir, 'fes2d-rew.dat'),
        sigma=(0.05, 5.0),
        cv_spec=('dT.z', 'tiltAvg'),
        grid_bin=(50, 50),
    ))

    print("Running FES calculation (Reweighting)...")
    calculate_fes(PartConfig(
        outfile=os.path.join(output_dir, 'fes2d-rew-strided.dat'),
        sigma=(0.05, 5.0),
        cv_spec=('dT.z', 'tiltAvg'),
        grid_bin=(50, 50),
        stride=100000,
    ))

    print("Running FES calculation (Reweighting)...")
    calculate_fes(PartConfig(
        outfile=os.path.join(output_dir, 'fes2d-rew-blocks.dat'),
        sigma=(0.05, 5.0),
        cv_spec=('dT.z', 'tiltAvg'),
        grid_bin=(50, 50),
        blocks_num=8,
    ))

if __name__ == "__main__":
    main()
