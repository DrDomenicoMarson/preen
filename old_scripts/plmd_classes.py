#!/usr/bin/env python3

from pathlib import Path
from dataclasses import dataclass
from warnings import catch_warnings, simplefilter
from numpy import loadtxt, asarray, linspace, isinf, isneginf, isnan, ndarray
import seaborn as sns
from matplotlib import pyplot as plt, cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pandas.errors import ParserError
sns.set_context("paper", font_scale=0.5)
# this is needed to address a DeprecationWarning,
# caused by SeaBorn-Matplotlib interaction,
# may be removed in the future
# plt.rcParams['axes.grid'] = False


@dataclass
class CV:
    name: str
    min: float|None = None
    max: float|None = None
    nbins_original: int|None = None
    periodic: bool|None = None
    values: 'Data3D | ndarray | None' = None


class Data3D:
    def __init__(self, nbins1, nbins2, vals_plumed=None, vals_reshaped=None):
        self.nbins1 = nbins1
        self.nbins2 = nbins2
        if vals_plumed is not None:
            self.plumed_style = vals_plumed
        elif vals_reshaped is not None:
            self.reshaped = vals_reshaped
        else:
            raise ValueError("vals_plumed XOR vals_reshaped need to be provided!")

    def to_plumed_style(self, vals_array):
        return vals_array.reshape(1, self.nbins2 * self.nbins1)[0]

    def to_reshaped(self, vals_array):
        return vals_array.reshape(self.nbins2, self.nbins1)

    @property
    def max(self):
        return next(x for x in sorted(self._plumed_style, reverse=True) if not isinf(x) and not isnan(x))

    @property
    def min(self):
        return next(x for x in sorted(self._plumed_style) if not isneginf(x) and not isnan(x))

    @property
    def plumed_style(self):
        return self._plumed_style

    @property
    def reshaped(self):
        return self._reshaped

    @plumed_style.setter
    def plumed_style(self, vals):
        vals = asarray(vals)
        self._plumed_style = vals
        self._reshaped = self.to_reshaped(self._plumed_style)

    @reshaped.setter
    def reshaped(self, vals):
        vals = asarray(vals)
        self._reshaped = vals
        self._plumed_style = self.to_plumed_style(self._reshaped)

    def update_reshaped(self, idx_to_update, new_values):
        self._reshaped[idx_to_update] = new_values
        self._plumed_style = self.to_plumed_style(self._reshaped)

    def from_reshaped_to_plumed_style(self):
        self._plumed_style = self.to_plumed_style(self._reshaped)

    def from_plumed_style_to_reshaped(self):
        self._reshaped = self.to_reshaped(self._plumed_style)


class PlumedBase:
    read_as_pandas = None
    def set_read_as_pandas(self, use_plumed=False):
        if PlumedBase.read_as_pandas is None:
            from MyPlumed.plmd_fake import read_as_pandas as my_read_as_pandas
            if not use_plumed:
                PlumedBase.read_as_pandas = my_read_as_pandas
            else:
                try:
                    import plumed # type: ignore
                    from functools import partial
                    PLUMED_KERNEL = plumed.Plumed()
                    PlumedBase.read_as_pandas = partial(plumed.read_as_pandas, kernel=PLUMED_KERNEL)
                except ImportError:
                    print("Cannot import plumed, module not found, using internal fake_read_as_pandas")
                    PlumedBase.read_as_pandas = my_read_as_pandas
                except RuntimeError as err:
                    print("Cannot import plumed, probably a library mismatch? Here's the error:")
                    print("\n", err, "\n")
                    print("...using internal fake_read_as_pandas")
                    PlumedBase.read_as_pandas = my_read_as_pandas


class PlumedFesFile(PlumedBase):
    def __init__(self, in_file, plot=False, load_with_plumed=False, **kwargs):
        self.set_read_as_pandas(use_plumed=load_with_plumed)
        self.in_file = in_file
        self.load_CV_infos()
        self.load_fes_data(load_with_plumed, **kwargs)
        if plot:
            self.plot(**kwargs)

    def load_CV_infos(self):
        cvs: list[CV] = []
        add_new_cv, cv_idx = True, 0
        with open(self.in_file, 'r') as f:
            fields = []
            for idx, line in enumerate(f):
                if not line.startswith("#!"):
                    break
                items = line.split()
                if idx == 0:
                    fields = items[2:]
                    continue
                if add_new_cv:
                    cvs.append(CV(name=fields[cv_idx]))
                    add_new_cv = False
                if items[2].startswith("normalisation"):
                    self.normalisation = float(items[3])
                if items[2].startswith("min"):
                    cvs[-1].min=float(items[3])
                if items[2].startswith("max"):
                    cvs[-1].max=float(items[3])
                if items[2].startswith("nbins"):
                    cvs[-1].nbins_original=int(items[3])
                if items[2].startswith("periodic"):
                    cvs[-1].periodic=bool(items[3])
                    cv_idx += 1
                    add_new_cv = True
        self.dimensions = len(cvs)
        if self.dimensions == 1:
            self.cv1: CV = cvs[0]
            self.fes_name: str = fields[1]
        elif self.dimensions == 2:
            self.cv1: CV = cvs[0]
            self.cv2: CV = cvs[1]
            self.fes_name: str = fields[2]
            #self.dfes_name1 = fields[3]
            #self.dfes_name2 = fields[4]
            if self.fes_name != "file.free":
                assert self.cv1.nbins_original is not None
                assert self.cv2.nbins_original is not None
                self.cv1.nbins_original += 1
                self.cv2.nbins_original += 1
        for idx, cv in enumerate(cvs):
            print(f"> found cv{idx} with name {cv.name}, min {cv.min}, max {cv.max}, nbins {cv.nbins_original}")

    def load_fes_data(self, load_with_plumed=True, set_min_to_0=False, **kwargs):
        if self.dimensions == 1:
            _data = loadtxt(self.in_file, unpack=True)
            self.cv1.values = _data[0]
            self.free = _data[1]            
        elif self.dimensions == 2:
            assert PlumedBase.read_as_pandas is not None
            data = PlumedBase.read_as_pandas(str(self.in_file))
            self.cv1.values = Data3D(vals_plumed=data[self.cv1.name], nbins2=self.cv2.nbins_original, nbins1=self.cv1.nbins_original)
            self.cv2.values = Data3D(vals_plumed=data[self.cv2.name], nbins2=self.cv2.nbins_original, nbins1=self.cv1.nbins_original)
            self.free = Data3D(vals_plumed=data[self.fes_name], nbins2=self.cv2.nbins_original, nbins1=self.cv1.nbins_original)
            #self.dfree1 = Data3D(vals_plumed=data[self.dfes_name1], nbins2=self.cv2.nbins_original, nbins1=self.cv1.nbins_original)
            #self.dfree2 = Data3D(vals_plumed=data[self.dfes_name2], nbins2=self.cv2.nbins_original, nbins1=self.cv1.nbins_original)

            if set_min_to_0:
                self.free.plumed_style -= self.free.min

            print()
            #self.max_free = next(x for x in sorted(self.free.plumed_style, reverse=True) if not isinf(x))
            #self.min_free = next(x for x in sorted(self.free.plumed_style) if not isneginf(x))
            # print(self.min_free, self.max_free)
            #self.free[self.free == -inf] = nan
            #self.free[self.free == inf] = nan
            # NOTE maybe: must implement this in the new style
            # self.free[self.free == self.max_free] = nan

    def plot(self, ax=None, fig=None, label=None, gradient_levels=11, max_fes_to_plot=None, min_fes_to_plot=None):
        
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            save_the_figure = True
        else:
            save_the_figure = False

        if self.dimensions == 1:
            assert isinstance(self.cv1.values, ndarray)
            assert isinstance(self.free, ndarray)
            _ = ax.plot(self.cv1.values, self.free, label=label, linewidth=0.2)
            ax.set_xlabel(self.cv1.name)
            ax.set_ylabel("FES [kcal/mol]")
        elif self.dimensions == 2:
            assert isinstance(self.cv1.values, Data3D)
            assert isinstance(self.cv2.values, Data3D)
            assert fig is not None
            max_to_plot = max_fes_to_plot if max_fes_to_plot is not None else self.free.max
            min_to_plot = min_fes_to_plot if min_fes_to_plot is not None else self.free.min
            levels_grad = linspace(min_to_plot, max_to_plot, num=gradient_levels, endpoint=True)
            cntr = ax.contourf(self.cv1.values.reshaped, self.cv2.values.reshaped, self.free.reshaped, levels=levels_grad, cmap="viridis")
            cbar = fig.colorbar(cntr, ax=ax, orientation="vertical", fraction=0.1, pad=0.04, shrink=0.85)
            cbar.ax.tick_params(labelsize=12)
            cbar.set_label(label="Free Energy", size=16)
            ax.set_xlabel(self.cv1.name)
            ax.set_ylabel(self.cv2.name)

        if label is None:
            ax.set_title(self.in_file)
        else:
            ax.set_title(label)

        if save_the_figure:
            assert fig is not None
            in_f_str = str(self.in_file.absolute())
            out_name = f"{in_f_str[:-4]}.pdf" if in_f_str[-4]=="." else f"{in_f_str}.pdf"
            fig.suptitle("/".join(in_f_str.split("/")[-3:]))
            fig.tight_layout()
            fig.savefig(out_name)
            plt.close(fig)


class PlumedOut(PlumedBase):
    def __init__(
            self,
            colvar_file: str|Path,
            colvar_dir: str|Path = "./",
            hills_file=None, hills_dir="./", 
            ext="pdf", plot=True, marker=",",
            load_with_plumed=False):
        self.set_read_as_pandas(use_plumed=load_with_plumed)
        self.rootdir = Path(".")
        assert isinstance(colvar_file, (str, Path)), "colvar_file must be str or Path"
        assert isinstance(colvar_dir, (str, Path)), "colvar_dir must be str or Path"
        self.colvar_file = colvar_file
        self.colvar_full_path = self.rootdir/colvar_dir/colvar_file
        self.df = self.read_data(self.colvar_full_path)
        self.df.replace('-', "NaN", inplace=True) # type: ignore
        #print(self.df.tail)
        self.hills = self.read_data(self.rootdir/hills_dir/hills_file) if hills_file else None
        self.marker = marker
        if plot: self.plot(ext=ext)

    def read_data(self, input_plumed_file):
        if not input_plumed_file.is_file():
            print(f"WARNING: the file {input_plumed_file} was not found...")
            return None
        input_plumed_file = str(input_plumed_file)
        with catch_warnings():
            simplefilter("ignore")
            assert PlumedBase.read_as_pandas is not None
            try:
                return PlumedBase.read_as_pandas(str(input_plumed_file))
            except ParserError as e:
                len_original_columns = (int(str(e).split()[6]))
                return PlumedBase.read_as_pandas(str(input_plumed_file), usecols=range(len_original_columns))


    def plot(self, linewidth=0.2, ext="pdf", append_to=None):
        tot_colvar_plots = self.df.shape[1]-1 # type: ignore
        tot_plots = tot_colvar_plots if self.hills is None else tot_colvar_plots + 1

        cols = int(tot_plots**0.5)
        rows = tot_plots // cols
        rows += tot_plots % cols

        if append_to is None:
            fig, axes = plt.subplots(rows, cols)
        else:
            fig, axes = append_to
            if axes is None:
                axes = fig.subplots(rows, cols)

        for k, ax in enumerate(axes.flat):
            try:
                ax.set_axis_on()
                title = self.df.columns[k+1] # type: ignore
                if "wall" in title or ".rct" in title or ".nker" in title or ".neff" in title or ".zed" in title:
                    ax.plot(self.df.iloc[:, 0], self.df.iloc[:, k+1], linewidth=linewidth) # type: ignore
                else:
                    ax.plot(self.df.iloc[:, 0], self.df.iloc[:, k+1], marker=self.marker, markersize=0.4, linewidth=0, markeredgewidth=0.2) # type: ignore
                if ".nker" in title:
                    title += " , # of compressed kernels"
                elif ".neff" in title:
                    title += " , effective sample size"
                ax.set_title(title)
            except IndexError:
                ax.set_axis_off()
                # pass

        if self.hills is not None:
            ax = axes.flat[-1]
            ax.set_axis_on()
            ax.plot(self.hills["time"], self.hills["height"], marker=self.marker, markersize=0.4, linewidth=0, markeredgewidth=0.2)
            ax.set_title("height")

        if append_to is None:
            suptitle = f"{self.rootdir}"
            fig.suptitle(suptitle)
            fig.tight_layout()
            plot_base = str(self.colvar_full_path.absolute())
            plot_file = plot_base[:-4] if plot_base[-4]=="." else plot_base
            fig.savefig(f"{plot_file}.{ext}", dpi=1200)
            plt.close(fig)
        else:
            return fig, axes


class PlumedDataFile:
    def __init__(self, in_file, print_info=False, is_path_data=False, select_path_kind="", file_idx=None):

        self.file_idx = file_idx
        self.data = {}
        self.file_path = in_file
        self.read_plumed_data_file(self.file_path, print_info)
        self.path = {}
        if is_path_data:
            self.find_path_data("s", select_path_kind=select_path_kind)
            self.find_path_data("z", select_path_kind=select_path_kind)

    def read_plumed_data_file(self, in_file, print_info=False):
        with open(in_file, 'r') as f:
            index_to_key_correspndance = {}
            for line in f:
                if line.startswith("#! FIELDS"):
                    items = line.split()
                    if not self.data:
                        index_to_key_correspndance = {}
                        for idx, item in enumerate(items[2:]):
                            self.data[item] = []
                            index_to_key_correspndance[idx] = item
                    else:
                        index_to_key_correspndance = {}
                        for idx, item in enumerate(items[2:]):
                            if item not in self.data:
                                tmp_data = []
                                for val in self.data.values():
                                    for _ in val:
                                        tmp_data.append(None)
                                    break
                                self.data[item] = tmp_data
                                index_to_key_correspndance[idx] = item
                            else:
                                index_to_key_correspndance[idx] = item
                elif line.startswith("\n") or line.startswith("#"):
                    continue
                else:
                    for idx, item in enumerate(line.split()):
                        self.data[index_to_key_correspndance[idx]].append(float(item))

        for self.k in self.data:
            if self.k == "time":
                self.tmp_time_list = []
                self.add_time, self.prev_time = 0, 0
                for self.time_idx, self.time in enumerate(self.data["time"]):
                    if self.time < self.prev_time:
                        self.delta_t = self.data["time"][self.time_idx - 1] - \
                                       self.data["time"][self.time_idx - 2]
                        self.add_time += self.prev_time + self.delta_t
                    self.prev_time = self.time
                    self.time += self.add_time
                    self.tmp_time_list.append(self.time)
                self.data["time"] = self.tmp_time_list
            self.data[self.k] = asarray(self.data[self.k])

        if print_info: print("> PlumedFile data keys:\n", "; ".join(self.data.keys()))

    def find_path_data(self, path_letter, select_path_kind):
        for self.k in self.data:
            if not self.k.startswith("der") or self.k.startswith("sigma"):
                if select_path_kind == 3:
                    if self.k.endswith(".%s%s%s" % (path_letter, path_letter, path_letter)):
                        self.path[path_letter] = self.data[self.k]
                elif select_path_kind == "gpath":
                    if self.k.endswith(".g%spath" % path_letter):
                        self.path[path_letter] = self.data[self.k]
                elif select_path_kind == "path":
                    if self.k.endswith(".%spath" % path_letter):
                        self.path[path_letter] = self.data[self.k]
                else:
                    if self.k.endswith(".%s%s%s" % (path_letter, path_letter, path_letter)) or \
                            self.k.endswith(".g%spath" % path_letter) or self.k.endswith(".%spath" % path_letter):
                        self.path[path_letter] = self.data[self.k]
