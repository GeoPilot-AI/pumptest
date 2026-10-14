"""Draw the documentation figures from the validation tests.

    python scripts/figures.py     # writes docs/assets/*.png

Every curve is a fresh fit, so a figure cannot show a result the code does
not produce.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogFormatterSciNotation, NullFormatter

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from validation_report import FITTED, WELLTESTPY, model_for  # noqa: E402

from pumptest import datasets, fit  # noqa: E402

OUT = REPO / "docs" / "assets"
# Categorical slots in fixed order (one per observation well), then ink.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "DejaVu Sans",
    }
)


def _grid(ax: plt.Axes) -> None:
    ax.grid(True, which="major", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    # Log axes label their minor ticks by default, which collides on short spans.
    for axis in (ax.xaxis, ax.yaxis):
        if axis.get_scale() == "log":
            axis.set_minor_formatter(NullFormatter())
            axis.set_major_formatter(LogFormatterSciNotation())


def _shown(time: np.ndarray, count: int = 60) -> np.ndarray:
    """Indices of about ``count`` log-spaced points, so dense logger records stay legible."""

    marks = np.geomspace(time.min(), time.max(), count)
    return np.unique(np.searchsorted(time, marks).clip(0, time.size - 1))


def fits_figure() -> Path:
    names = (*FITTED, *WELLTESTPY)
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.6), constrained_layout=True)
    for ax, name in zip(axes.flat, names, strict=False):
        test, model = datasets.load(name), model_for(name)
        result = fit(test, model)
        floor, top = np.inf, 0.0
        for colour, obs in zip([*SERIES, "#8a5cd1"], test.observations, strict=False):
            keep = (obs.drawdown > 0) & (obs.time >= 10.0 / 86400.0)
            t, s = obs.time[keep], obs.drawdown[keep]
            idx = _shown(t)
            floor, top = min(floor, float(s[idx].min())), max(top, float(s[idx].max()))
            t_line = np.logspace(np.log10(t.min()), np.log10(t.max()), 200)
            ax.loglog(t[idx], s[idx], "o", ms=3.5, mfc="none", mec=colour, mew=1.0, label=obs.name)
            ax.loglog(t_line, model.drawdown(result.values, test, obs, t_line), color=colour)
        # The model falls towards zero at early time far from the well; show
        # only the range the instruments measured.
        ax.set_ylim(0.5 * floor, 2.0 * top)
        ax.set_title(
            f"{test.name}: {model.name}\nT = {result.values['T']:.4g} m²/d, "
            f"RMSE {result.rmse * 1000:.1f} mm",
            color=INK,
        )
        ax.set_xlabel("time since pumping started (d)")
        ax.set_ylabel("drawdown (m)")
        ax.legend(loc="lower right")
        _grid(ax)
    for ax in axes.flat[len(names) :]:
        ax.axis("off")
    fig.suptitle(
        "Measured drawdown (circles) and the fitted solution (lines)", color=INK, fontsize=12
    )
    path = OUT / "fits.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def parity_figure() -> Path:
    fig, (ax_t, ax_s) = plt.subplots(1, 2, figsize=(9, 4.2), constrained_layout=True)
    for name in (*FITTED, *WELLTESTPY):
        test, pub = datasets.load(name), datasets.published(name)
        ours = fit(test, model_for(name)).values
        colour = SERIES[0] if name in FITTED else SERIES[1]
        for ax, key in ((ax_t, "T"), (ax_s, "S")):
            ax.loglog(pub.values[key], ours[key], "o", ms=7, color=colour)
            ax.annotate(
                test.name,
                (pub.values[key], ours[key]),
                textcoords="offset points",
                xytext=(6, -10),
                fontsize=7.5,
                color=MUTED,
            )
    for ax, label, unit in ((ax_t, "Transmissivity", " (m²/d)"), (ax_s, "Storativity", "")):
        lo, hi = ax.get_xlim()
        lo, hi = min(lo, ax.get_ylim()[0]), max(hi, ax.get_ylim()[1])
        ax.loglog([lo, hi], [lo, hi], color=MUTED, lw=1, ls="--", label="1:1")
        ax.plot([], [], "o", color=SERIES[0], label="AQTESOLV")
        ax.plot([], [], "o", color=SERIES[1], label="welltestpy (raw records here)")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(f"published{unit}")
        ax.set_ylabel(f"pumptest{unit}")
        ax.set_title(label, color=INK)
        ax.legend(loc="upper left")
        _grid(ax)
    fig.suptitle("pumptest against published estimates", color=INK, fontsize=12)
    path = OUT / "parity.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for path in (fits_figure(), parity_figure()):
        print(f"wrote {path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
