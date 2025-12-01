import pytest
import os
import shutil
import numpy as np
import pandas as pd
from FESutils.state_api import calculate_fes_from_state
from FESutils.colvar_api import calculate_fes
from FESutils.fes_config import FESStateConfig, FESConfig

@pytest.fixture(scope="module")
def sym_env():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    test_dir = os.path.join(base_dir, 'test_symmetrization')
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)
    yield test_dir
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

def create_dummy_state_file(filename, dim2=False):
    with open(filename, 'w') as f:
        # Header block (Fixed 10 lines expected by state_api.py)
        # Line 0: FIELDS
        if not dim2:
            f.write("#! FIELDS time cv1 sigma_cv1 height\n")
        else:
            f.write("#! FIELDS time cv1 cv2 sigma_cv1 sigma_cv2 height\n")
            
        # Line 1: action
        f.write("#! SET action OPES_METAD_state\n")
        # Line 2: biasfactor
        f.write("#! SET biasfactor 10.0\n")
        # Line 3: epsilon
        f.write("#! SET epsilon 1.0\n")
        # Line 4: kernel_cutoff
        f.write("#! SET kernel_cutoff 6.25\n")
        # Line 5: ignored (usually compression_threshold)
        f.write("#! SET compression_threshold 1\n")
        # Line 6: zed
        f.write("#! SET zed 1.0\n")
        # Line 7: sum_weights (since action is OPES_METAD_state)
        f.write("#! SET sum_weights 1.0\n")
        # Line 8: ignored
        f.write("#! SET ignored 0\n")
        # Line 9: ignored
        f.write("#! SET ignored 0\n")
        
        # Periodicity info comes after header block
        f.write("#! SET min_cv1 -4.0\n")
        f.write("#! SET max_cv1 4.0\n")
        if dim2:
            f.write("#! SET min_cv2 -4.0\n")
            f.write("#! SET max_cv2 4.0\n")
        
        # Data: Symmetric kernels at -2 and +2
        # Height 1.0
        # Format must match FIELDS
        if not dim2:
            # time cv1 sigma height
            f.write("1.0 -2.0 0.1 1.0\n")
            f.write("2.0  2.0 0.1 1.0\n")
        else:
            # time cv1 cv2 sigma1 sigma2 height
            f.write("1.0 -2.0 0.0 0.1 0.1 1.0\n")
            f.write("2.0  2.0 0.0 0.1 0.1 1.0\n")

def create_dummy_colvar_file(filename, dim2=False):
    # Create samples at -2 and +2
    n_samples = 1000
    cv1 = np.concatenate([np.random.normal(-2.0, 0.1, n_samples), np.random.normal(2.0, 0.1, n_samples)])
    if dim2:
        cv2 = np.random.normal(0.0, 0.1, 2 * n_samples)
    else:
        cv2 = np.zeros(2 * n_samples)
        
    bias = np.zeros(2 * n_samples) # No bias for simplicity
    
    df = pd.DataFrame({'time': np.arange(2 * n_samples), 'cv1': cv1, 'cv2': cv2, 'bias': bias})
    
    with open(filename, 'w') as f:
        f.write("#! FIELDS time cv1 cv2 bias\n")
        df.to_csv(f, sep=' ', header=False, index=False)

def test_state_symmetrization_1d(sym_env):
    state_file = os.path.join(sym_env, 'state_1d.dat')
    create_dummy_state_file(state_file, dim2=False)
    outfile = os.path.join(sym_env, 'fes_state_1d.dat')
    
    config = FESStateConfig(
        input_file=state_file,
        outfile=outfile,
        temp=300.0,
        grid_bin=(100,),
        symmetrize_cvs=['cv1'],
        grid_min=(0.0,),
        grid_max=(4.0,)
    )
    
    calculate_fes_from_state(config)
    
    data = np.loadtxt(outfile)
    cv = data[:, 0]
    fes = data[:, 1]
    
    # Check that grid starts at 0
    assert cv[0] >= 0.0
    
    # Check for minimum around 2.0 (since -2 is folded to 2)
    min_idx = np.argmin(fes)
    assert cv[min_idx] == pytest.approx(2.0, abs=0.2)

def test_colvar_symmetrization_1d(sym_env):
    colvar_file = os.path.join(sym_env, 'colvar_1d.dat')
    create_dummy_colvar_file(colvar_file, dim2=False)
    outfile = os.path.join(sym_env, 'fes_colvar_1d.dat')
    
    config = FESConfig(
        input_file=colvar_file,
        outfile=outfile,
        temp=300.0,
        sigma=(0.1,),
        cv_spec=('cv1',),
        bias_spec='bias',
        symmetrize_cvs=['cv1'],
        grid_min=(0.0,),
        grid_max=(4.0,)
    )
    
    calculate_fes(config)
    
    data = np.loadtxt(outfile)
    cv = data[:, 0]
    fes = data[:, 1]
    
    # Check that grid starts at 0
    assert cv[0] >= 0.0
    
    # Check for minimum around 2.0
    min_idx = np.argmin(fes)
    assert cv[min_idx] == pytest.approx(2.0, abs=0.2)

def test_state_symmetrization_2d(sym_env):
    state_file = os.path.join(sym_env, 'state_2d.dat')
    create_dummy_state_file(state_file, dim2=True)
    outfile = os.path.join(sym_env, 'fes_state_2d.dat')
    
    config = FESStateConfig(
        input_file=state_file,
        outfile=outfile,
        temp=300.0,
        grid_bin=(50, 50),
        symmetrize_cvs=['cv1'], # Only symmetrize cv1
        grid_min=(0.0, -1.0),
        grid_max=(4.0, 1.0)
    )
    
    calculate_fes_from_state(config)
    
    # Check output file exists
    assert os.path.exists(outfile)
    
    # Load data (format is x y z)
    data = np.loadtxt(outfile)
    cv1 = data[:, 0]
    
    # Check cv1 is positive
    assert np.all(cv1 >= 0.0)

def test_colvar_symmetrization_2d(sym_env):
    colvar_file = os.path.join(sym_env, 'colvar_2d.dat')
    create_dummy_colvar_file(colvar_file, dim2=True)
    outfile = os.path.join(sym_env, 'fes_colvar_2d.dat')
    
    config = FESConfig(
        input_file=colvar_file,
        outfile=outfile,
        temp=300.0,
        sigma=(0.1, 0.1),
        cv_spec=('cv1', 'cv2'),
        bias_spec='bias',
        symmetrize_cvs=['cv1'],
        grid_min=(0.0, -1.0),
        grid_max=(4.0, 1.0)
    )
    
    calculate_fes(config)
    
    assert os.path.exists(outfile)
    data = np.loadtxt(outfile)
    cv1 = data[:, 0]
    assert np.all(cv1 >= 0.0)
