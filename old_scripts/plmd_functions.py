#!/usr/bin/env python3

from pathlib import Path
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm
from MyPlumed.plmd_classes import PlumedFesFile, PlumedOut

sns.set_context("paper", font_scale=0.5)

def get_file(stem_str, dir_to_check: Path) -> None | Path :
    for f in dir_to_check.iterdir():
        if f.is_file() and f.stem == stem_str and f.suffix not in [".png", ".pdf"]:
            return f
    return None


def get_files_from_multirun(basedir: Path, basefilen: str) -> list[Path]:
    found_files = []
    for subdir in sorted(basedir.iterdir()):
        if subdir.is_dir() and not "unbiased_run" in subdir.name:
            found_file = get_file(stem_str=basefilen, dir_to_check=subdir)
            if found_file is not None:
                found_files.append(found_file)
    return found_files

def get_headers(headers_line):
    return headers_line.lstrip("#! FIELDS").rstrip("\n").split()

def merge_colvar_files(
        basedir: str|Path = "./",
        colvar_base: str = "COLVAR",
        perc_colvar_to_discard: float = 10.0,
        keep_order: bool = False,
        onlyfirst: bool = False):
    basedir = Path(basedir)
    colvar_files = sorted(get_files_from_multirun(basedir=basedir, basefilen=colvar_base))
    print(f"> will work on the {colvar_base} files found in {basedir} subdirs:")
    colvar_files_str = '\n\t'.join(map(str, colvar_files))
    print(f"\t{colvar_files_str}")
    outf = f"{colvar_base}_merged.dat"
    print(f"> will keep all but the first {perc_colvar_to_discard}% of each file")
    print(f"> will write the merged data to the {outf} file")

    with open(colvar_files[0]) as f:
        header_line = f.readline()
        concat_lines: dict[int, list[str]] = {0: [header_line]}
        num_fields = len(get_headers(headers_line=header_line))
        tot_lines = sum(1 for _ in f)

    read_from = round(perc_colvar_to_discard*tot_lines/100)

    if onlyfirst:
        colvar_files = colvar_files[:1]

    for colvar in tqdm(colvar_files, desc="Loading the data..."):
        with open(colvar, "r") as f:
            for _ in range(read_from):
                next(f)
            for idx, l in enumerate(f, start=1):
                if l.startswith("#") or len(l.split()) != num_fields:
                    continue
                if not l.endswith('\n'):
                    l += '\n'
                if keep_order:
                    try:
                        concat_lines[idx].append(l)
                    except KeyError:
                        concat_lines[idx] = [l]
                else:
                    concat_lines[0].append(l)


    with open(outf, 'w') as f:
        for frame_list in tqdm(concat_lines.values(), desc="Writing..."):
            for l in frame_list:
                f.write(l)

def plot_colvar(
        basedir: str|Path = "./",
        colvar_base: str = "COLVAR",
        hills_base: str = "HILLS",
        colvar_file: None | Path = None,
        hills_file: None | Path = None,
        ext: str = "png",
        marker: str = ',',
        plot_each: bool = True):
    """
    plots the colvar files in a semi-automatic way;
    - if colvar_file is provided, it just plots the specific file
    - otherwise:
        - checks if it is a multidir run (it checks every subdir of 'basedir')
            - plots every 'colvar_base' in its subdir
            - plots a 'merged' file in working_dir with all the 'colvar_base' files found
        - if it doesn't detect a multidir run
            - it search for a 'colvar_base' file in the 'basedir' and plots it
    """
    basedir = Path(basedir)
    if colvar_file is not None:
        print(f"> plotting the provided file: {colvar_file.absolute()} ")
        plmd_file = PlumedOut(
            colvar_file=colvar_file, hills_file=hills_file, ext=ext, marker=marker)
    else:
        colvar_files = get_files_from_multirun(basedir=basedir, basefilen=colvar_base)
        hills_files = get_files_from_multirun(basedir=basedir, basefilen=hills_base)
        if len(colvar_files) != len(hills_files) and hills_files:
            msg = f"each directory should have both a {colvar_base} and a {hills_base} files"
            msg += f", or just a {colvar_base} file"
            raise FileNotFoundError(msg)
        if colvar_files:
            print(f"> working on the {colvar_base} files found in {basedir} subdirs:")
            colvar_files_str = '\n\t'.join(map(str, sorted(colvar_files)))
            print(f">\t{colvar_files_str}")
            fig = plt.figure()
            axes = None
            for dir_idx, colvar in enumerate(tqdm(colvar_files)):
                hills = hills_files[dir_idx] if hills_files else None
                plmd_file = PlumedOut(
                        colvar_file=colvar, hills_file=hills, ext=ext, marker=marker,
                        plot=plot_each)
                fig, axes = plmd_file.plot(append_to=(fig, axes))
            print(f"> plotting the {colvar_base} files found in the subdirs, merged together")
            suptitle = str(basedir.absolute())
            fig.suptitle(suptitle)
            fig.tight_layout()
            fig.savefig(str(f"{colvar_base}_merged.{ext}"), dpi=600)
            plt.close(fig)
        else:
            colvar = get_file(stem_str=colvar_base, dir_to_check=basedir)
            hills = get_file(stem_str=hills_base, dir_to_check=basedir)
            _plmd_file = PlumedOut(colvar_file=colvar, hills_file=hills, ext=ext, marker=marker)

# def run_sum_hills():
#     out_dir = basedir/"results_fes"
#     out_dir.mkdir(parents=True, exist_ok=True)
#     sum_hills_base = "export PLUMED_MAXBACKUP=0 ; plumed sum_hills"
#     sum_hills_base += f" --hills {basedir/opt.hills} --mintozero"
#     sum_hills_base += f" --min {opt.min} --max {opt.max} --bin {opt.bin}"
#     sum_hills_all = f"{sum_hills_base} --outfile {out_dir/'fes_all.dat'}"
#     run(sum_hills_all, shell=True, check=True)
#     if opt.stride:
#         stride = int(line_count(basedir/opt.hills)/abs(opt.stride)) if opt.stride < 0 else opt.stride
#         strided_dir = out_dir/f"strided_{abs(opt.stride)}.chunks_{stride}" if opt.stride < 0 else out_dir/f"strided_{abs(opt.stride)}.abs"
#         strided_dir.mkdir(parents=True, exist_ok=True)
#         sum_hills_stride = f"{sum_hills_base} --stride {stride} --outfile {strided_dir/'fes_strided.'}"
#         run(sum_hills_stride, shell=True, check=True)

def plot_fes(basedir, input_fes="fes_all.dat", gradient_levels=50, max_fes=None, min_fes=None, stride=0):
    out_dir = Path(basedir)/"results_fes"
    _ = PlumedFesFile(
        out_dir/input_fes, plot=True,
        gradient_levels=gradient_levels,
        max_fes_to_plot=max_fes, min_fes_to_plot=min_fes)
    try:
        strided_dir = list(out_dir.glob(f"strided_{abs(stride)}.*"))[0]
    except IndexError:
        strided_dir = None

    if strided_dir:
        plot_type = "2D"
        with open(strided_dir/"fes_strided.0.dat") as f:
            for l in f:
                if l.split()[4] == "file.free":
                    plot_type = "3D"
                else:
                    raise ValueError("Cannot identify if the FES is 2D or 3D")
                break

        fes_files = list(strided_dir.glob("fes*"))
        n_fes = len(fes_files)
        cols = int(n_fes**0.5)
        rows = n_fes // cols
        rows += n_fes % cols
        if n_fes+1 > cols*rows and plot_type == "2D":  # account for the possible cumulative plot
            n_fes += 1
            cols = int(n_fes**0.5)
            rows = n_fes // cols
            rows += n_fes % cols

        fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True)
        for idx, ax in enumerate(axes.flat):
            curr_file = strided_dir/f"fes_strided.{idx}.dat"
            if curr_file.is_file():
                plmd_fes = PlumedFesFile(
                    curr_file, ax=ax, fig=fig, label=idx, plot=True,
                    gradient_levels=gradient_levels,
                    max_fes_to_plot=max_fes)
                if plot_type == "2D":
                    plmd_fes.plot(ax=axes.flat[-1], fig=fig, label=idx)

        if plot_type == "2D":
            axes.flat[-1].set_title("all in")
            axes.flat[-1].legend()

        out_file = out_dir/f"fes_strided_{abs(stride)}.pdf"
        fig.suptitle(str(out_dir))
        fig.tight_layout()
        fig.savefig(out_file)
        plt.close(fig)
