from FESutils import calculate_fes, merge_runs_multiple_colvar_files, FESConfig
from pathlib import Path

def run_1d_symmetrization():
    cv_columns = ("dT.z", )
    bias_spec = "opes.bias"
    requested_columns = cv_columns + (bias_spec,) if bias_spec != "no" else cv_columns

    merged = merge_runs_multiple_colvar_files(
        run_dirs=[Path("big_data/run_1"), Path("big_data/run_2")],
        basenames=["COLVAR"], # "CV_DIHEDRALS"
        discard_fractions=[0.7, 0.1],
        time_ordered=True,
        requested_columns=requested_columns,
        verbose=True,
    )

    config = FESConfig(
        outfile="fes_multi_run_nosym.dat",
        temp=300.0,
        grid_bin=(100,),
        sigma=(0.05,),
        cv_spec=cv_columns,
        bias_spec=bias_spec,
        backup=False,
        plot=True,
    )
    calculate_fes(config, merge_result=merged)

    config = FESConfig(
        outfile="fes_multi_run_sym.dat",
        temp=300.0,
        grid_bin=(100,),
        sigma=(0.05,),
        cv_spec=cv_columns,
        symmetrize_cvs=('dT.z',), # <--- New feature
        bias_spec=bias_spec,
        backup=False,
        plot=True,
    )
    calculate_fes(config, merge_result=merged)


def run_2d_symmetrization():
    cv_columns = ("dT.z", "tiltAvg")
    bias_spec = "opes.bias"
    requested_columns = cv_columns + (bias_spec,) if bias_spec != "no" else cv_columns

    merged = merge_runs_multiple_colvar_files(
        run_dirs=[Path("big_data/run_1"), Path("big_data/run_2")],
        basenames=["COLVAR"], # "CV_DIHEDRALS"
        discard_fractions=[0.7, 0.1],
        time_ordered=True,
        requested_columns=requested_columns,
        verbose=True,
    )

    config = FESConfig(
        outfile="fes_multi_run_nosym2D.dat",
        temp=300.0,
        grid_bin=(100, 90),
        sigma=(0.05, 5),
        cv_spec=cv_columns,
        bias_spec=bias_spec,
        backup=False,
        plot=True,
    )
    calculate_fes(config, merge_result=merged)

    config = FESConfig(
        outfile="fes_multi_run_sym2D.dat",
        temp=300.0,
        grid_bin=(100, 90),
        sigma=(0.05, 5),
        cv_spec=cv_columns,
        symmetrize_cvs=('dT.z',), # <--- New feature
        bias_spec=bias_spec,
        backup=False,
        plot=True,
    )
    calculate_fes(config, merge_result=merged)

if __name__ == "__main__":
    #run_1d_symmetrization()
    run_2d_symmetrization()
