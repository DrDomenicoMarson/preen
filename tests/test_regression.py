import pytest
import os
import shutil
import numpy as np
import glob
import FESutils.api
from FESutils.fes_config import FESConfig
from FESutils.constants import KB_KJ_MOL

# Use Agg backend for headless testing
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

@pytest.fixture(scope="module")
def regression_env():
    """Set up and tear down the regression test environment."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # example_data is now in ../examples/data relative to tests/
    example_data_dir = os.path.abspath(os.path.join(base_dir, '..', 'examples', 'data'))
    reference_dir = os.path.join(example_data_dir, 'res_fromKERNELS_reference')
    colvar_file = os.path.join(example_data_dir, 'COLVAR_merged.tgz')
    test_output_dir = os.path.join(base_dir, 'test_regression_output')
    
    if not os.path.exists(test_output_dir):
        os.makedirs(test_output_dir)
        
    env = {
        'base_dir': base_dir,
        'example_data_dir': example_data_dir,
        'reference_dir': reference_dir,
        'colvar_file': colvar_file,
        'test_output_dir': test_output_dir
    }
    
    yield env
    
    if os.path.exists(test_output_dir):
        shutil.rmtree(test_output_dir)

def run_fes(config):
    """Helper to run FES calculation with given config."""
    FESutils.api.calculate_fes(config)

def compare_files(test_file, ref_file):
    """Compare two FES output files numerically."""
    assert os.path.exists(test_file), f"Test file {test_file} not found"
    assert os.path.exists(ref_file), f"Reference file {ref_file} not found"

    # Read data
    try:
        test_data = np.loadtxt(test_file)
        ref_data = np.loadtxt(ref_file)
    except ValueError as e:
        pytest.fail(f"Could not load data from files: {e}")

    # Compare data with tolerance
    np.testing.assert_allclose(
        test_data, 
        ref_data, 
        rtol=1e-5, 
        atol=1e-5, 
        err_msg=f"Data mismatch in {os.path.basename(test_file)}"
    )

def test_1_dZ(regression_env):
    """Test 1: 1D FES for dT.z"""
    outfile = os.path.join(regression_env['test_output_dir'], 'dZ.dat')
    config = FESConfig(
        filename=regression_env['colvar_file'],
        outfile=outfile,
        kbt=300.0 * KB_KJ_MOL,
        grid_bin=(100,),
        sigma=(0.05,),
        cv_spec=('dT.z',),
        bias_spec='opes.bias',
        plot=True
    )
    run_fes(config)
    compare_files(outfile, os.path.join(regression_env['reference_dir'], 'dZ.dat'))

def test_2_dZ_16block(regression_env):
    """Test 2: 1D FES with blocks"""
    outfile = os.path.join(regression_env['test_output_dir'], 'dZ_16block.dat')
    config = FESConfig(
        filename=regression_env['colvar_file'],
        outfile=outfile,
        kbt=300.0 * KB_KJ_MOL,
        grid_bin=(100,),
        sigma=(0.05,),
        cv_spec=('dT.z',),
        bias_spec='opes.bias',
        blocks_num=16,
        plot=True
    )
    run_fes(config)
    # When blocks are used, the final output is in a 'block' subdirectory
    block_outfile = os.path.join(os.path.dirname(outfile), 'block', os.path.basename(outfile))
    # Reference file is also in a block subdirectory
    ref_outfile = os.path.join(regression_env['reference_dir'], 'block', 'dZ_16block.dat')
    compare_files(block_outfile, ref_outfile)

def test_3_dZ_stride(regression_env):
    """Test 3: 1D FES with stride"""
    outfile = os.path.join(regression_env['test_output_dir'], 'dZ_500000stride.dat')
    config = FESConfig(
        filename=regression_env['colvar_file'],
        outfile=outfile,
        kbt=300.0 * KB_KJ_MOL,
        grid_bin=(100,),
        sigma=(0.05,),
        cv_spec=('dT.z',),
        bias_spec='opes.bias',
        stride=500000,
        plot=True
    )
    run_fes(config)
    
    stride_dir = os.path.join(os.path.dirname(outfile), 'stride')
    assert os.path.exists(stride_dir)
    # Check if files are there
    files = glob.glob(os.path.join(stride_dir, '*.dat'))
    assert len(files) > 0

def test_4_2D_simple(regression_env):
    """Test 6: Simple 2D FES execution (no blocks/stride)"""
    outfile = os.path.join(regression_env['test_output_dir'], '2D_simple.dat')
    config = FESConfig(
        filename=regression_env['colvar_file'],
        outfile=outfile,
        kbt=300.0 * KB_KJ_MOL,
        grid_bin=(50, 50),
        sigma=(0.05, 5.0),
        cv_spec=('dT.z', 'tiltAvg'),
        bias_spec='opes.bias',
        plot=True
    )
    print("\nRunning simple 2D test...")
    run_fes(config)
    assert os.path.exists(outfile)
