"""
Example: merge multiple basenames per run and compute a FES in memory.

Layout expected under `base_dir` (one file per basename in each run directory):
    base_dir/run0/COLVAR
    base_dir/run0/CV_DIHEDRALS
    base_dir/run1/COLVAR
    base_dir/run1/CV_DIHEDRALS

Adjust `basenames`, `cv_columns`, and `bias_spec` to match your data.
"""

from pathlib import Path

from FESutils import FESConfig, calculate_fes, merge_multiple_colvar_files
from FESutils.constants import KB_KJ_MOL


def main():
    base_dir = Path("examples/data")  # change to your dataset root

    # Basenames to merge per run (must exist side-by-side in each run directory)
    basenames = ("COLVAR", "CV_DIHEDRALS")

    cv_columns = ("dT.z", "phi1")  # example CV names spanning both files
    bias_spec = "opes.bias"  # can also be explicit names or column numbers

    # Only build the needed columns to keep memory use small
    requested_columns = cv_columns + (bias_spec,) if bias_spec != "no" else cv_columns

    merged = merge_multiple_colvar_files(
        base_dir=base_dir,
        basenames=basenames,
        discard_fraction=0.50,
        time_ordered=True,
        requested_columns=requested_columns,
        verbose=True,
    )

    # Configure reweighting; supply sigma/grid settings for your system
    config = FESConfig(
        filename=None,
        outfile="fes_multi.dat",
        kbt=300.0 * KB_KJ_MOL,
        grid_bin=(100,),
        sigma=(0.05,),
        cv_spec=cv_columns,
        bias_spec=bias_spec,
        backup=False,
        plot=True,
    )

    calculate_fes(config, merge_result=merged)
    print("Done. Outputs: fes_multi.dat (+ fes_multi.png if plot=True)")


if __name__ == "__main__":
    main()
