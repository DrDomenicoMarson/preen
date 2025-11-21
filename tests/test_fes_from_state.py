import pytest
import os
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import FESutils.api
from FESutils.fes_config import FESConfig
from FESutils.constants import KB_KJ_MOL

@pytest.fixture(scope="module")
def state_env():
    """Set up and tear down the state test environment."""
    base_dir = os.getcwd()
    test_dir = os.path.join(base_dir, 'test_state_output')
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)
    
    # example_data is in ../examples/data
    kernels_file = os.path.abspath(os.path.join(base_dir, '..', 'examples', 'data', 'KERNELSforRST'))
    outfile = os.path.join(test_dir, 'fes_from_state.dat')
    
    env = {
        'base_dir': base_dir,
        'test_dir': test_dir,
        'kernels_file': kernels_file,
        'outfile': outfile
    }
    
    yield env
    
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

def test_run_kernels_for_rst(state_env):
    """Test running calcFES_from_State.py on KERNELSforRST."""
    if not os.path.exists(state_env['kernels_file']):
        pytest.skip(f"KERNELSforRST not found at {state_env['kernels_file']}")

    config = FESConfig(
        filename=state_env['kernels_file'],
        outfile=state_env['outfile'],
        kbt=300.0 * KB_KJ_MOL,
        grid_bin=(100, 100),
        sigma=(0.0,), # Dummy
        cv_spec=(), # Dummy
        bias_spec='', # Dummy
        backup=False,
        plot=True
    )
    
    # Run main
    FESutils.api.calculate_fes_from_state(config)
    
    # Check output exists
    assert os.path.exists(state_env['outfile'])
    
    # Check PNG exists (default behavior)
    png_file = state_env['outfile'].replace('.dat', '.png')
    assert os.path.exists(png_file), f"PNG file {png_file} was not created"
    
    # Check content
    with open(state_env['outfile'], 'r') as f:
        lines = f.readlines()
    
    # Check header
    assert any('FIELDS' in l for l in lines)
    assert any('SET min_' in l for l in lines)
    
    # Check data
    data = np.loadtxt(state_env['outfile'])
    assert data.shape[0] > 0
    assert data.shape[1] >= 2 # CV, FES
    
    # Basic sanity check on FES values
    # FES should be dimensionless (energy/kbt) or energy?
    # Script says: fes = -kbt * sf * log(prob/max_prob)
    # So it is energy units.
    # Min value should be 0 (mintozero default)
    fes_col = 1
    min_fes = np.min(data[:, fes_col])
    assert min_fes == pytest.approx(0.0, abs=1e-4)
