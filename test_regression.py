import unittest
import os
import shutil
import sys
import numpy as np
import glob
import calcFES

# Mock matplotlib before importing modules that use it
from unittest.mock import MagicMock
mock_plt = MagicMock()

def mock_subplots(*args, **kwargs):
    fig = MagicMock()
    ax = MagicMock()
    return fig, ax

mock_plt.subplots.side_effect = mock_subplots
mock_plt.get_cmap.return_value = MagicMock()
mock_plt.Normalize.return_value = MagicMock()
mock_plt.cm.ScalarMappable.return_value = MagicMock()

sys.modules['matplotlib'] = MagicMock()
sys.modules['matplotlib.pyplot'] = mock_plt


class TestFESRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_dir = os.path.dirname(os.path.abspath(__file__))
        cls.example_data_dir = os.path.join(cls.base_dir, 'example_data')
        cls.reference_dir = os.path.join(cls.example_data_dir, 'res_fromKERNELS_reference')
        cls.colvar_file = os.path.join(cls.example_data_dir, 'COLVAR_merged.tgz')
        cls.test_output_dir = os.path.join(cls.base_dir, 'test_regression_output')
        
        if not os.path.exists(cls.test_output_dir):
            os.makedirs(cls.test_output_dir)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_output_dir):
            shutil.rmtree(cls.test_output_dir)

    def run_fes(self, args):
        """Helper to run calcFESs with given arguments."""
        calcFES.main(args)

    def compare_files(self, test_file, ref_file):
        """Compare two FES output files numerically."""
        self.assertTrue(os.path.exists(test_file), f"Test file {test_file} not found")
        self.assertTrue(os.path.exists(ref_file), f"Reference file {ref_file} not found")

        # Compare headers (ignoring file paths if any)
        # We might need to be selective here if headers contain timestamps or paths
        # For now, let's assume headers should be identical except maybe for some specific fields
        # with open(test_file, 'r') as f:
        #     test_lines = [l for l in f.readlines() if l.startswith('#!')]
        # with open(ref_file, 'r') as f:
        #     ref_lines = [l for l in f.readlines() if l.startswith('#!')]
        # self.assertEqual(test_lines, ref_lines, "Headers do not match")

        # Read data
        try:
            test_data = np.loadtxt(test_file)
            ref_data = np.loadtxt(ref_file)
        except ValueError as e:
            self.fail(f"Could not load data from files: {e}")

        # Compare data with tolerance
        np.testing.assert_allclose(test_data, ref_data, rtol=1e-5, atol=1e-5, err_msg=f"Data mismatch in {os.path.basename(test_file)}")

    def test_1_dZ(self):
        """Test 1: 1D FES for dT.z"""
        outfile = os.path.join(self.test_output_dir, 'dZ.dat')
        args = [
            '--colvar', self.colvar_file,
            '--outfile', outfile,
            '--temp', '300',
            '--bin', '100',
            '--sigma', '0.05',
            '--cv', 'dT.z',
            '--bias', 'opes.bias',
        ]
        self.run_fes(args)
        self.compare_files(outfile, os.path.join(self.reference_dir, 'dZ.dat'))

    def test_2_dZ_16block(self):
        """Test 2: 1D FES with blocks"""
        outfile = os.path.join(self.test_output_dir, 'dZ_16block.dat')
        args = [
            '--colvar', self.colvar_file,
            '--outfile', outfile,
            '--temp', '300',
            '--bin', '100',
            '--sigma', '0.05',
            '--cv', 'dT.z',
            '--bias', 'opes.bias',
            '--blocks', '16',
        ]
        self.run_fes(args)
        # When blocks are used, the final output is in a 'block' subdirectory
        block_outfile = os.path.join(os.path.dirname(outfile), 'block', os.path.basename(outfile))
        # Reference file is also in a block subdirectory
        ref_outfile = os.path.join(self.reference_dir, 'block', 'dZ_16block.dat')
        self.compare_files(block_outfile, ref_outfile)

    def test_3_dZ_stride(self):
        """Test 3: 1D FES with stride"""
        outfile = os.path.join(self.test_output_dir, 'dZ_500000stride.dat')
        args = [
            '--colvar', self.colvar_file,
            '--outfile', outfile,
            '--temp', '300',
            '--bin', '100',
            '--sigma', '0.05',
            '--cv', 'dT.z',
            '--bias', 'opes.bias',
            '--stride', '500000',
        ]
        self.run_fes(args)
        
        stride_dir = os.path.join(os.path.dirname(outfile), 'stride')
        self.assertTrue(os.path.exists(stride_dir))
        # Check if files are there
        files = glob.glob(os.path.join(stride_dir, '*.dat'))
        self.assertTrue(len(files) > 0)


    def test_4_2D_simple(self):
        """Test 6: Simple 2D FES execution (no blocks/stride)"""
        outfile = os.path.join(self.test_output_dir, '2D_simple.dat')
        args = [
            '--colvar', self.colvar_file,
            '--outfile', outfile,
            '--temp', '300',
            '--bin', '50,50',
            '--sigma', '0.05,5',
            '--cv', 'dT.z,tiltAvg',
            '--bias', 'opes.bias',
        ]
        print("\nRunning simple 2D test...")
        self.run_fes(args)
        self.assertTrue(os.path.exists(outfile))

if __name__ == '__main__':
    unittest.main()
