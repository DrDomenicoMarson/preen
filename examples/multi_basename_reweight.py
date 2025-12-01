from pathlib import Path
from FESutils.constants import DEG2RAD
from FESutils import FESConfig, calculate_fes, merge_multiple_colvar_files

def main():
    base_dir = Path("big_data/run_2")  # change to your dataset root

    basenames = ("COLVAR", "CV_DIHEDRALS")

    cv_columns = ("dT.z", "phi1")  # example CV names spanning both files
    bias_spec = "opes.bias"  # can also be explicit names or column numbers

    requested_columns = cv_columns + (bias_spec,) if bias_spec != "no" else cv_columns

    merged = merge_multiple_colvar_files(
        base_dir=base_dir,
        basenames=basenames,
        discard_fraction=0.0,
        time_ordered=True,
        requested_columns=requested_columns,
        verbose=True,
    )

    #print(merged.dataframe[["dT.z", "phi1", "opes.bias"]])

    config = FESConfig(
        outfile="fes_multi.dat",
        temp=300.0,
        grid_bin=(100, 90),
        sigma=(0.05,5*DEG2RAD),
        cv_spec=cv_columns,
        bias_spec=bias_spec,
        plot=True,
    )

    calculate_fes(config, merge_result=merged)


if __name__ == "__main__":
    main()
