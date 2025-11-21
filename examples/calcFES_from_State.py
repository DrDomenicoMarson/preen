#! /usr/bin/env python3

"""
Example script demonstrating how to use the FESutils library to calculate FES from STATE file.
"""

import os
from FESutils.fes_config import FESConfig
from FESutils.api import calculate_fes_from_state

def main():
    # Define configuration
    base_dir = os.path.dirname(os.path.abspath(__file__))
    example_data_dir = os.path.join(base_dir, 'data')
    output_dir = os.path.join(base_dir, 'output')

    state_file = os.path.join(example_data_dir, 'KERNELSforRST')

    config = FESConfig(
        filename=state_file,
        outfile=os.path.join(output_dir, 'fes-state.dat'),
        temp=300.0,
        grid_bin=(100, 100),
        plot=True,
        backup=True
    )

    print("Running FES calculation (From State)...")
    calculate_fes_from_state(config)
    print("Success!")

if __name__ == "__main__":
    main()
