import os
import shutil
import sys
import unittest
from unittest.mock import MagicMock
import numpy as np

# Add current directory to path
sys.path.append(os.getcwd())

# Mock matplotlib before importing modules that use it
sys.modules['matplotlib'] = MagicMock()
sys.modules['matplotlib.pyplot'] = MagicMock()

import calcFES
from FESutils.fes_output import backup_file

class TestFESImprovements(unittest.TestCase):
    def setUp(self):
        self.test_dir = 'test_output'
        os.makedirs(self.test_dir, exist_ok=True)
        self.colvar_file = str(os.path.join(self.test_dir, 'COLVAR'))
        self.create_dummy_colvar()

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def create_dummy_colvar(self):
        with open(self.colvar_file, 'w') as f:
            f.write('#! FIELDS time cv1 cv2 .bias\n')
            f.write('#! SET min_cv1 -3.14\n')
            f.write('#! SET max_cv1 3.14\n')
            for i in range(100):
                f.write(f'{i} {np.sin(i/10)} {np.cos(i/10)} {np.random.rand()}\n')

    def test_backup_functionality(self):
        """Test that backup_file creates backups correctly."""
        test_file = os.path.join(self.test_dir, 'test.dat')
        
        # Create first file
        with open(test_file, 'w') as f:
            f.write('content 1')
        
        # Backup
        backup_file(test_file)
        
        # Check if backup exists (should not exist yet as file didn't exist before first write? 
        # Wait, backup_file is called BEFORE writing.
        # So if I call it now, it should move test.dat to test.dat.1
        
        self.assertFalse(os.path.exists(test_file))
        self.assertTrue(os.path.exists(f'{test_file}.1'))
        
        # Create file again
        with open(test_file, 'w') as f:
            f.write('content 2')
            
        # Backup again
        backup_file(test_file)
        
        self.assertTrue(os.path.exists(f'{test_file}.1')) # Old backup
        self.assertTrue(os.path.exists(f'{test_file}.2')) # New backup
        
    def test_exception_handling(self):
        """Test that invalid arguments raise SystemExit (caught in main) or ValueError."""
        # Test invalid dimension
        with self.assertRaises(ValueError):
            calcFES.parse_cli_args(['--colvar', self.colvar_file, '--sigma', '0.1,0.1,0.1', '--temp', '300', '--cv', '2,3,4'])


    def test_no_backup_option(self):
        """Test that --no-backup disables backup creation."""
        outfile = os.path.join(self.test_dir, 'no_backup.dat')
        
        # First run
        args1 = [
            '--colvar', self.colvar_file,
            '--outfile', outfile,
            '--temp', '300',
            '--bin', '10',
            '--sigma', '0.1',
            '--cv', 'cv1',
            '--no-backup'
        ]
        calcFES.main(args1)
        self.assertTrue(os.path.exists(outfile))
        
        # Get modification time
        mtime1 = os.path.getmtime(outfile)
        
        # Second run (should overwrite without backup)
        # Sleep briefly to ensure mtime difference if FS resolution is low
        import time
        time.sleep(0.1)
        
        calcFES.main(args1)
        self.assertTrue(os.path.exists(outfile))
        self.assertFalse(os.path.exists(f'{outfile}.1'))
        
        mtime2 = os.path.getmtime(outfile)
        self.assertNotEqual(mtime1, mtime2)

if __name__ == '__main__':
    unittest.main()
