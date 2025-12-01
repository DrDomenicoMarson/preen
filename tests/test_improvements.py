import pytest
import os
import shutil
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
from FESutils.colvar_api import calculate_fes
from FESutils.fes_config import FESConfig
from FESutils.fes_output import backup_file
from FESutils.constants import KB_KJ_MOL

@pytest.fixture
def backup_env(tmp_path):
    """Set up a temporary environment for backup tests."""
    test_dir = tmp_path / "backup_test"
    test_dir.mkdir()
    colvar_file = test_dir / "COLVAR"
    
    # Create dummy COLVAR
    with open(colvar_file, 'w') as f:
        f.write('#! FIELDS time cv1 cv2 cv3 .bias\n')
        f.write('#! SET min_cv1 -3.14\n')
        f.write('#! SET max_cv1 3.14\n')
        for i in range(100):
            f.write(f'{i} {np.sin(i/10)} {np.cos(i/10)} {np.sin(i/5)} {np.random.rand()}\n')
            
    return {
        'test_dir': test_dir,
        'colvar_file': str(colvar_file)
    }

def test_backup_functionality(backup_env):
    """Test that backup_file creates backups correctly."""
    test_file = backup_env['test_dir'] / 'test.dat'
    
    # Create first file
    with open(test_file, 'w') as f:
        f.write('content 1')
    
    # Backup
    backup_file(str(test_file))
    
    # Check if backup exists (test.dat -> test.dat.1)
    assert not test_file.exists()
    assert (backup_env['test_dir'] / 'test.dat.1').exists()
    
    # Create file again
    with open(test_file, 'w') as f:
        f.write('content 2')
        
    # Backup again
    backup_file(str(test_file))
    
    assert (backup_env['test_dir'] / 'test.dat.1').exists() # Old backup
    assert (backup_env['test_dir'] / 'test.dat.2').exists() # New backup

def test_config_validation(backup_env):
    """Test that invalid configuration raises ValueError."""
    # Test invalid dimension mismatch (sigma has 3 elements, cv_spec has 3 elements, but code expects 1 or 2)
    # Wait, FESConfig doesn't validate on init, but calculate_fes does.
    
    # Actually, calculate_fes checks len(sigma) vs dim.
    # Let's try to trigger a validation error.
    
    # If I pass 3 CVs, load_colvar_data might fail or calculate_fes might fail.
    # The original test was checking parse_cli_args.
    
    outfile = str(backup_env['test_dir'] / 'out.dat')
    config = FESConfig(
        input_file=backup_env['colvar_file'],
        outfile=outfile,
        kbt=300.0 * KB_KJ_MOL,
        grid_bin=(10,),
        sigma=(0.1, 0.1, 0.1), # 3 sigmas
        cv_spec=('cv1', 'cv2', 'cv3'), # 3 CVs
        bias_spec='.bias'
    )
    
    # calculate_fes should raise ValueError because only 1D or 2D supported
    with pytest.raises(ValueError, match="only 1 or 2 dimensional bias are supported"):
        calculate_fes(config)

def test_no_backup_option(backup_env):
    """Test that backup=False disables backup creation."""
    outfile = backup_env['test_dir'] / 'no_backup.dat'
    outfile_str = str(outfile)
    
    config = FESConfig(
        input_file=backup_env['colvar_file'],
        outfile=outfile_str,
        kbt=300.0 * KB_KJ_MOL,
        grid_bin=(10,),
        sigma=(0.1,),
        cv_spec=('cv1',),
        bias_spec='.bias',
        backup=False
    )
    
    # First run
    calculate_fes(config)
    assert outfile.exists()
    
    mtime1 = outfile.stat().st_mtime
    
    # Second run (should overwrite without backup)
    time.sleep(0.1)
    calculate_fes(config)
    
    assert outfile.exists()
    assert not (backup_env['test_dir'] / 'no_backup.dat.1').exists()
    
    mtime2 = outfile.stat().st_mtime
    assert mtime1 != mtime2
