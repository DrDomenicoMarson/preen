#! /usr/bin/env python3

"""
Example script demonstrating how to use the FESutils library to calculate FES from COLVAR data.
"""

import os
from FESutils.fes_config import FESConfig
from FESutils.api import calculate_fes

def main():
    # Define configuration
    # In a real scenario, these might come from arguments or a config file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    example_data_dir = os.path.join(base_dir, 'data')
    
    config = FESConfig(
        filename=os.path.join(example_data_dir, 'COLVAR_merged.tgz'),
        outfile='fes-rew.dat',
        temp=300.0,
        sigma=(0.05, 5.0),
        cv_spec=('dT.z', 'tiltAvg'),
        bias_spec='opes.bias',
        grid_bin=(50, 50),
        plot=True,
        backup=True
    )
    
    print("Running FES calculation (Reweighting)...")
    try:
        calculate_fes(config)
        print("Success!")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
