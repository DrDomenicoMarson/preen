import math
from dataclasses import dataclass
from collections.abc import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


@dataclass
class StrideEntry:
    label: str
    surface: NDArray


class PlotManager:
    def __init__(
        self,
        enabled: bool,
        dim2: bool,
        axis_x,
        axis_y,
        mintozero: bool,
        mesh: tuple[NDArray, NDArray] | None,
    ):
        self.enabled = enabled
        self.dim2 = dim2
        self.axis_x = axis_x
        self.axis_y = axis_y
        self.mintozero = mintozero
        self.mesh = mesh
        self.standard_surface: NDArray | None = None
        self.stride_entries: list[StrideEntry] = []

    def record_standard_surface(self, surface: NDArray) -> None:
        if not self.enabled:
            return
        self.standard_surface = np.array(surface, copy=True)

    def add_stride_surface(self, surface: NDArray, label: str) -> None:
        if not self.enabled:
            return
        self.stride_entries.append(
            StrideEntry(label=label, surface=np.array(surface, copy=True))
        )

    def save_standard_plot(self, path: str, names: tuple[str, str | None]) -> None:
        if not self.enabled or self.standard_surface is None:
            return
        _plot_single_surface(
            path,
            self.standard_surface,
            self.axis_x,
            self.axis_y,
            self.mintozero,
            self.mesh,
            self.dim2,
            names,
            title="Free energy surface",
        )

    def save_stride_plots(self, path: str, names: tuple[str, str | None]) -> None:
        if not self.enabled or not self.stride_entries:
            return
        if not self.dim2:
            _plot_stride_1d(
                path, self.stride_entries, self.axis_x, self.mintozero, names[0]
            )
        else:
            _plot_stride_2d(
                path,
                self.stride_entries,
                self.axis_x,
                self.axis_y,
                self.mintozero,
                self.mesh,
                names,
            )

    def save_block_plots(
        self,
        path: str,
        fes_surface: NDArray,
        error_surface: NDArray,
        names: tuple[str, str | None],
    ) -> None:
        if not self.enabled:
            return
        if not self.dim2:
            _plot_block_error_1d(
                path, self.axis_x, fes_surface, error_surface, self.mintozero, names[0]
            )
        else:
            _plot_block_error_2d(
                path,
                fes_surface,
                error_surface,
                self.axis_x,
                self.axis_y,
                self.mesh,
                names,
                self.mintozero,
            )


def _plot_single_surface(
    path, surface, axis_x, axis_y, mintozero, mesh, dim2, names, title
):
    data = _apply_shift(surface, mintozero)
    if not dim2:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(axis_x.values, data, label="FES")
        ax.set_xlabel(names[0])
        ax.set_ylabel("Free energy")
        ax.set_title(title)
        ax.legend(fontsize=7)
    else:
        fig, ax = plt.subplots(figsize=(6, 5))
        x_vals = axis_x.values
        y_vals = axis_y.values
        extent = [axis_x.minimum, axis_x.maximum, axis_y.minimum, axis_y.maximum]
        im = ax.imshow(data.T, origin="lower", extent=extent, aspect="auto")
        ax.set_xlabel(names[0])
        ax.set_ylabel(names[1] if names[1] is not None else "CV2")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label="Free energy")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _plot_stride_1d(path, entries: Sequence[StrideEntry], axis_x, mintozero, name_x):
    fig, ax = plt.subplots(figsize=(7, 4))
    cmap = plt.get_cmap("viridis")
    n = len(entries)
    use_colorbar = n > 6
    for idx, entry in enumerate(entries):
        data = _apply_shift(entry.surface, mintozero)
        color = cmap(idx / (n - 1)) if n > 1 else cmap(0.0)
        label = entry.label if not use_colorbar else None
        ax.plot(axis_x.values, data, label=label, alpha=0.9, color=color)
    ax.set_xlabel(name_x)
    ax.set_ylabel("Free energy")
    ax.set_title("Stride evolution")
    if not use_colorbar:
        ax.legend(fontsize=7)
    else:
        norm = plt.Normalize(1, n)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label="Stride index")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _plot_stride_2d(
    path, entries: Sequence[StrideEntry], axis_x, axis_y, mintozero, mesh, names
):
    n = len(entries)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    axes = np.atleast_2d(axes)
    data_list = [_apply_shift(entry.surface, mintozero) for entry in entries]
    vmin = min(item.min() for item in data_list)
    vmax = max(item.max() for item in data_list)
    extent = [axis_x.minimum, axis_x.maximum, axis_y.minimum, axis_y.maximum]
    for idx, entry in enumerate(entries):
        r = idx // cols
        c = idx % cols
        ax = axes[r, c]
        data = _apply_shift(entry.surface, mintozero)
        im = ax.imshow(
            data.T, origin="lower", extent=extent, aspect="auto", vmin=vmin, vmax=vmax
        )
        ax.set_title(entry.label)
    for idx in range(n, rows * cols):
        r = idx // cols
        c = idx % cols
        axes[r, c].axis("off")
    # add labels to bottom-left subplot
    axes[-1, 0].set_xlabel(names[0])
    axes[-1, 0].set_ylabel(names[1] if names[1] is not None else "CV2")
    fig.tight_layout()
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.65, label="Free energy")
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _plot_block_error_1d(path, axis_x, fes_surface, error_surface, mintozero, name_x):
    fes = _apply_shift(fes_surface, mintozero)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(axis_x.values, fes, label="Average FES")
    ax.fill_between(
        axis_x.values,
        fes - error_surface,
        fes + error_surface,
        alpha=0.3,
        label="Uncertainty",
    )
    ax.set_xlabel(name_x)
    ax.set_ylabel("Free energy")
    ax.set_title("Block-averaged FES with uncertainty")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _plot_block_error_2d(
    path, fes_surface, error_surface, axis_x, axis_y, mesh, names, mintozero
):
    extent = [axis_x.minimum, axis_x.maximum, axis_y.minimum, axis_y.maximum]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    data = _apply_shift(fes_surface, mintozero)
    im0 = axes[0].imshow(data.T, origin="lower", extent=extent, aspect="auto")
    axes[0].set_title("Block-averaged FES")
    axes[0].set_xlabel(names[0])
    axes[0].set_ylabel(names[1] if names[1] is not None else "CV2")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, label="Free energy")
    im1 = axes[1].imshow(error_surface.T, origin="lower", extent=extent, aspect="auto")
    axes[1].set_title("Uncertainty")
    axes[1].set_xlabel(names[0])
    axes[1].set_ylabel(names[1] if names[1] is not None else "CV2")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, label="Free energy uncertainty")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _apply_shift(surface: NDArray, mintozero: bool) -> NDArray:
    if not mintozero:
        return surface
    return surface - np.min(surface)
