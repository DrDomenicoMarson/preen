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
    example_data_dir = os.path.join(base_dir, 'example_data')
    
    # Note: KERNELSforRST might not be compressed in example_data by default, 
    # but our API supports both.
    state_file = os.path.join(example_data_dir, 'KERNELSforRST')
    
    config = FESConfig(
        filename=state_file,
        outfile='fes-state.dat',
        sigma=(0.0,), # Not used for state file (read from file), but required by config init
        kbt=2.49433863, # 300 K
        cv_spec=(), # Not used for state file (read from file)
        bias_spec='', # Not used
        grid_bin=(100, 100),
        plot=True,
        backup=True
    )
    
    print("Running FES calculation (From State)...")
    try:
        calculate_fes_from_state(config)
        print("Success!")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
