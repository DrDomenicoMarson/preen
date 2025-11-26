from pathlib import Path

from FESutils import FESConfig, calculate_fes, merge_runs_multiple_colvar_files
from FESutils.constants import KB_KJ_MOL, DEG2RAD


def main():

    cv_columns = ("dT.z", "phi1")
    bias_spec = "opes.bias"
    requested_columns = cv_columns + (bias_spec,) if bias_spec != "no" else cv_columns

    merged = merge_runs_multiple_colvar_files(
        run_dirs=[Path("big_data/run_1"), Path("big_data/run_2")],
        basenames=["COLVAR", "CV_DIHEDRALS"],
        discard_fractions=[0.4, 0.02],
        time_ordered=True,
        requested_columns=requested_columns,
        verbose=True,
    )

    config = FESConfig(
        input_file=None,
        outfile="fes_multi_run.dat",
        kbt=300.0 * KB_KJ_MOL,
        grid_bin=(100, 90),
        sigma=(0.05, 5*DEG2RAD),
        cv_spec=cv_columns,
        bias_spec=bias_spec,
        backup=False,
        plot=True,
    )

    calculate_fes(config, merge_result=merged)


if __name__ == "__main__":
    main()
