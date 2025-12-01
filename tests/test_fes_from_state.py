import pytest
import os
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
from FESutils.state_api import calculate_fes_from_state
from FESutils.fes_config import FESStateConfig
from FESutils.constants import KB_KJ_MOL

@pytest.fixture(scope="module")
def state_env():
    """Set up and tear down the state test environment."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
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

    config = FESStateConfig(
        input_file=state_env['kernels_file'],
        outfile=state_env['outfile'],
        temp=300.0,
        grid_bin=(100, 100),
        backup=False,
        plot=True
    )
    
    # Run main
    calculate_fes_from_state(config)
    
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


def _write_simple_state_2d(path):
    """Write a minimal 2D STATE file that supports derivative evaluation."""
    lines = [
        "#! FIELDS time cv1 cv2 sigma_cv1 sigma_cv2 height",
        "#! SET action OPES_METAD_state",
        "#! SET biasfactor 10.0",
        "#! SET epsilon 1.0",
        "#! SET kernel_cutoff 6.25",
        "#! SET compression_threshold 1",
        "#! SET zed 1.0",
        "#! SET sum_weights 1.0",
        "#! SET ignored 0",
        "#! SET ignored 0",
        "#! SET min_cv1 -4.0",
        "#! SET max_cv1 4.0",
        "#! SET min_cv2 -1.0",
        "#! SET max_cv2 1.0",
        # time  cv1  cv2  sigma1 sigma2 height
        "1.0 -1.0 0.0 0.2 0.3 1.0",
        "2.0  1.0 0.0 0.2 0.3 1.0",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def test_state_derivatives_2d(tmp_path):
    """Ensure 2D STATE derivative computation runs and writes derivative columns."""
    state_file = tmp_path / "state_2d.dat"
    _write_simple_state_2d(state_file)
    outfile = tmp_path / "fes_state_der.dat"
    config = FESStateConfig(
        input_file=str(state_file),
        outfile=str(outfile),
        temp=300.0,
        grid_bin=(10, 10),
        calc_der=True,
        backup=False,
        plot=False,
    )

    calculate_fes_from_state(config)

    assert outfile.exists()
    data = np.loadtxt(outfile)
    # Columns: cv1, cv2, fes, der_cv1, der_cv2
    assert data.shape[1] == 5
