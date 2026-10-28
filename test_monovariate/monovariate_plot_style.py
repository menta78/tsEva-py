"""Consistent presentation-only figure formatting for monovariate tsEva demos.

Only changes Matplotlib artists (sizes, tick density, legends, and saved PNGs).
Does not edit input data, fitted EVA parameters, or return levels.
"""
from pathlib import Path
import math
import numpy as np
from datetime import datetime
from matplotlib.ticker import FuncFormatter
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator


def configure_plotting():
    mpl.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 17,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 18,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'axes.titlepad': 14,
    })


def _limit_fixed_xticks(ax, max_ticks=9):
    """Prevent crowded years while preserving original date representation."""
    if ax.get_xscale() != 'linear' or getattr(ax, 'name', '') == '3d':
        return
    vals = ax.get_xticks()
    if len(vals) <= max_ticks or len(vals) > 100:
        return
    loc = ax.xaxis.get_major_locator()
    # For fixed labels, preserve the matching tick-label text.
    if isinstance(loc, FixedLocator):
        labels = [t.get_text() for t in ax.get_xticklabels()]
        stride = math.ceil(len(vals)/max_ticks)
        idx = list(range(0, len(vals), stride))
        if labels and len(labels) == len(vals) and all(labels[i] for i in idx):
            ax.set_xticks(vals[idx], [labels[i] for i in idx])
        else:
            ax.set_xticks(vals[idx])


def _format_matlab_years(ax):
    """tsEva axes store MATLAB datenums, which Matplotlib dates mislabels by ~1970 years."""
    if ax.name == '3d' or ax.get_xscale() != 'linear':
        return False
    xmin, xmax = ax.get_xlim()
    if not (500000 < xmin < xmax < 1000000):
        return False
    try:
        start, stop = datetime.fromordinal(max(1, int(xmin - 366))), datetime.fromordinal(int(xmax - 366))
    except (ValueError, OverflowError):
        return False
    years = max(1, stop.year - start.year)
    step = 1 if years <= 8 else (2 if years <= 18 else (5 if years <= 45 else 10))
    ystart = ((start.year + step - 1) // step) * step
    positions = [datetime(y, 1, 1).toordinal() + 366 for y in range(ystart, stop.year + 1, step)]
    if positions:
        if not ax.get_xlabel():
            ax.set_xlabel('Year')
        ax.set_xticks(positions)
        ax.xaxis.set_major_formatter(FuncFormatter(
            lambda x, pos: f'{datetime.fromordinal(int(x - 366)).year}'
            if 366 < x < 1000000 else ''))
    return True


def polish_figure(fig=None):
    """Fit ticks, titles, and legends to a readable 16:9-ish export canvas."""
    fig = fig if fig is not None else plt.gcf()
    if not fig.axes:
        return fig
    is_3d = any(ax.name == '3d' for ax in fig.axes)
    fig.set_size_inches(13.5, 7.8 if is_3d else 7.25, forward=True)
    legend_rows = 0
    main_title = ''
    collected = []
    for ax in fig.axes:
        is_colorbar = ax.get_label() == '<colorbar>' or hasattr(ax, '_colorbar')
        ax.tick_params(axis='both', which='major', labelsize=10, pad=6)
        if ax.name == '3d':
            ax.tick_params(axis='z', which='major', labelsize=9, pad=3)
            ax.zaxis.label.set_fontsize(11)
        for label in [ax.xaxis.label, ax.yaxis.label]:
            label.set_fontsize(12 if not is_colorbar else 11)
            label.set_weight('normal')
        if ax.title.get_text():
            if not main_title and not is_colorbar:
                main_title = ax.title.get_text()
                ax.set_title('')
            else:
                ax.title.set_fontsize(15)
        for text in ax.texts:
            if text.get_fontsize() > 12:
                text.set_fontsize(11)
        if is_colorbar:
            continue
        # A hard-coded y-limit must not cut off return-level confidence bands.
        if 'return period' in ax.get_xlabel().lower() and ax.lines:
            values = []
            for line in ax.lines:
                yy = np.asarray(line.get_ydata(), dtype=float).reshape(-1)
                yy = yy[np.isfinite(yy)]
                if yy.size:
                    values.append(yy)
            if values:
                low = min(float(a.min()) for a in values)
                high = max(float(a.max()) for a in values)
                a, b = ax.get_ylim()
                if low < a or high > b:
                    span = max(1e-6, high - low)
                    ax.set_ylim(min(a, low - .05 * span), max(b, high + .05 * span))
        if not _format_matlab_years(ax):
            _limit_fixed_xticks(ax)
        for tick in ax.get_xticklabels():
            tick.set_rotation(0)
            tick.set_ha('center')
        if ax.name != '3d':
            ax.grid(True, which='major', alpha=0.19, linewidth=0.65)
            ax.grid(False, which='minor', axis='x')
            ax.set_axisbelow(True)
        legend = ax.get_legend()
        if legend:
            # Matplotlib 3.6/3.7: legendHandles; 3.8+: legend_handles.
            # Keep both APIs supported without changing the underlying plot.
            handles = getattr(legend, 'legend_handles', None)
            if handles is None:
                handles = getattr(legend, 'legendHandles', None)
            if handles is None:
                handles, _ = ax.get_legend_handles_labels()
            handles = list(handles)
            labels = [t.get_text() for t in legend.get_texts()]
            if handles and labels:
                ncol = min(4, len(labels))
                legend_rows = max(legend_rows, math.ceil(len(labels)/ncol))
                collected.append((handles, labels, ncol))
            legend.remove()
    if main_title:
        fig.suptitle(main_title, fontsize=17, fontweight='semibold', y=0.985)
    if collected:
        # One figure-level legend row (or two for six curves), independent of data axes.
        handles, labels, ncol = collected[0]
        fig.legend(handles, labels, loc='upper center',
                   bbox_to_anchor=(0.5, 0.925 if main_title else 0.975),
                   ncol=ncol, frameon=False, fontsize=10,
                   columnspacing=1.6, handlelength=2.6)
    if is_3d:
        try:
            fig.subplots_adjust(left=0.09, right=0.91, bottom=0.11, top=0.87)
        except Exception:
            pass
    else:
        if collected:
            top = 0.74 if legend_rows >= 2 and main_title else (0.80 if main_title else (0.82 if legend_rows >= 2 else 0.88))
        else:
            top = 0.89 if main_title else 0.94
        try:
            fig.tight_layout(pad=1.65)
            if len(fig.axes) == 1:
                fig.subplots_adjust(left=0.085, right=0.97, bottom=0.12, top=top)
            else:
                fig.subplots_adjust(top=top)
        except Exception:
            fig.subplots_adjust(left=0.10, right=0.94, bottom=0.13, top=top)
    return fig


def save_figure(filename, fig=None, **kwargs):
    fig = polish_figure(fig)
    save_options = {'dpi': 160, 'bbox_inches': 'tight', 'pad_inches': 0.22}
    save_options.update(kwargs)
    fig.savefig(filename, **save_options)
    return Path(filename)
