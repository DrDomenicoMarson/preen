import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from FESutils.colvar_api import calculate_fes
from FESutils.fes_config import FESConfig

def create_symmetric_data(filename):
    """
    Create a dummy COLVAR file with data symmetric around 0.
    Simulates a particle crossing a membrane (z coordinate).
    """
    n_samples = 10000
    # Two wells at -2 and +2
    z = np.concatenate([
        np.random.normal(-2.0, 0.5, n_samples // 2),
        np.random.normal(2.0, 0.5, n_samples // 2)
    ])
    # Another CV, e.g., distance from pore axis (r), always positive
    r = np.random.normal(1.0, 0.2, n_samples)
    
    # Bias (dummy)
    bias = np.zeros(n_samples)
    
    df = pd.DataFrame({'time': np.arange(n_samples), 'z': z, 'r': r, 'bias': bias})
    
    with open(filename, 'w') as f:
        f.write("#! FIELDS time z r bias\n")
        df.to_csv(f, sep=' ', header=False, index=False)
    print(f"Created dummy data: {filename}")

def run_1d_symmetrization():
    print("\n--- Running 1D Symmetrization Example ---")
    input_file = "example_sym_data.dat"
    create_symmetric_data(input_file)
    
    # 1. Standard FES (Full range -4 to 4)
    print("Calculating standard FES...")
    config_std = FESConfig(
        input_file=input_file,
        outfile="fes_std.dat",
        temp=300.0,
        sigma=(0.2,),
        cv_spec=('z',),
        bias_spec='bias',
        grid_min=(-4.0,),
        grid_max=(4.0,),
        grid_bin=(100,),
        plot=True
    )
    calculate_fes(config_std)
    
    # 2. Symmetrized FES (Folded range 0 to 4)
    print("Calculating symmetrized FES...")
    config_sym = FESConfig(
        input_file=input_file,
        outfile="fes_sym.dat",
        temp=300.0,
        sigma=(0.2,),
        cv_spec=('z',),
        bias_spec='bias',
        symmetrize_cvs=['z'], # <--- New feature
        grid_min=(0.0,),      # Grid starts at 0
        grid_max=(4.0,),
        grid_bin=(50,),
        plot=True
    )
    calculate_fes(config_sym)
    
    # Plot comparison
    data_std = np.loadtxt("fes_std.dat")
    data_sym = np.loadtxt("fes_sym.dat")
    
    plt.figure(figsize=(10, 5))
    plt.plot(data_std[:, 0], data_std[:, 1], label="Standard FES (z)")
    plt.plot(data_sym[:, 0], data_sym[:, 1], '--', label="Symmetrized FES (|z|)")
    plt.xlabel("z coordinate")
    plt.ylabel("Free Energy (kJ/mol)")
    plt.legend()
    plt.title("Comparison of Standard vs Symmetrized FES")
    plt.grid(True)
    plt.savefig("comparison_1d.png")
    print("Generated comparison_1d.png")

def run_2d_symmetrization():
    print("\n--- Running 2D Symmetrization Example ---")
    input_file = "example_sym_data.dat"
    # We use the same data file
    
    # Symmetrize only 'z', keep 'r' as is
    print("Calculating 2D FES (z symmetrized, r normal)...")
    config_2d = FESConfig(
        input_file=input_file,
        outfile="fes_2d_sym.dat",
        temp=300.0,
        sigma=(0.2, 0.1),
        cv_spec=('z', 'r'),
        bias_spec='bias',
        symmetrize_cvs=['z'], # <--- Only z is symmetrized
        grid_min=(0.0, 0.0),
        grid_max=(4.0, 2.0),
        grid_bin=(50, 50),
        plot=True
    )
    calculate_fes(config_2d)
    print("Generated fes_2d_sym.dat and fes_2d_sym.png")

if __name__ == "__main__":
    run_1d_symmetrization()
    run_2d_symmetrization()
    
    # # Cleanup
    # if os.path.exists("example_sym_data.dat"):
    #     os.remove("example_sym_data.dat")
