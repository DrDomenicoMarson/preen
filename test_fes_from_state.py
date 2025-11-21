import unittest
import os
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import FESutils.api
from FESutils.fes_config import FESConfig
from FESutils.constants import KB_KJ_MOL

class TestFESFromState(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_dir = os.getcwd()
        cls.test_dir = os.path.join(cls.base_dir, 'test_state_output')
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)
        os.makedirs(cls.test_dir)
        
        cls.kernels_file = os.path.join(cls.base_dir, 'example_data', 'KERNELSforRST')
        cls.outfile = os.path.join(cls.test_dir, 'fes_from_state.dat')

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)

    def test_run_kernels_for_rst(self):
        """Test running calcFES_from_State.py on KERNELSforRST."""
        if not os.path.exists(self.kernels_file):
            self.skipTest(f"KERNELSforRST not found at {self.kernels_file}")

        config = FESConfig(
            filename=self.kernels_file,
            outfile=self.outfile,
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
        self.assertTrue(os.path.exists(self.outfile))
        
        # Check PNG exists (default behavior)
        png_file = self.outfile.replace('.dat', '.png')
        self.assertTrue(os.path.exists(png_file), f"PNG file {png_file} was not created")
        
        # Check content
        with open(self.outfile, 'r') as f:
            lines = f.readlines()
        
        # Check header
        self.assertTrue(any('FIELDS' in l for l in lines))
        self.assertTrue(any('SET min_' in l for l in lines))
        
        # Check data
        data = np.loadtxt(self.outfile)
        self.assertTrue(data.shape[0] > 0)
        self.assertTrue(data.shape[1] >= 2) # CV, FES
        
        # Basic sanity check on FES values
        # FES should be dimensionless (energy/kbt) or energy?
        # Script says: fes = -kbt * sf * log(prob/max_prob)
        # So it is energy units.
        # Min value should be 0 (mintozero default)
        fes_col = 1
        min_fes = np.min(data[:, fes_col])
        self.assertAlmostEqual(min_fes, 0.0, places=4)

if __name__ == '__main__':
    unittest.main()
