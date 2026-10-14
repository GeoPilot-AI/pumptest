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

from validation_report import FITTED, model_for  # noqa: E402

from pumptest import (  # noqa: E402
    Observation,
    PapadopulosCooper,
    PumpingTest,
    datasets,
    fit,
    log_derivative,
    step_drawdown,
)

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


def fits_figure() -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.6), constrained_layout=True)
    for ax, name in zip(axes.flat, (*FITTED, "vennebulten"), strict=True):
        test, model = datasets.load(name), model_for(name)
        result = fit(test, model)
        for colour, obs in zip(SERIES, test.observations, strict=False):
            t_line = np.logspace(np.log10(obs.time.min()), np.log10(obs.time.max()), 200)
            ax.loglog(
                obs.time,
                obs.drawdown,
                "o",
                ms=4,
                mfc="none",
                mec=colour,
                mew=1.1,
                label=obs.name,
            )
            ax.loglog(t_line, model.drawdown(result.values, test, obs, t_line), color=colour)
        ax.set_title(
            f"{test.name}: {model.name}\nT = {result.values['T']:.4g} m²/d, "
            f"RMSE {result.rmse * 1000:.1f} mm",
            color=INK,
        )
        ax.set_xlabel("time since pumping started (d)")
        ax.set_ylabel("drawdown (m)")
        ax.legend(loc="lower right")
        _grid(ax)
    fig.suptitle(
        "Measured drawdown (circles) and the fitted solution (lines)", color=INK, fontsize=12
    )
    path = OUT / "fits.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def parity_figure() -> Path:
    fig, (ax_t, ax_s) = plt.subplots(1, 2, figsize=(9, 4.2), constrained_layout=True)
    for name in FITTED:
        test, pub = datasets.load(name), datasets.published(name).values
        ours = fit(test, model_for(name)).values
        for ax, key in ((ax_t, "T"), (ax_s, "S")):
            ax.loglog(pub[key], ours[key], "o", ms=7, color=SERIES[0])
            ax.annotate(
                test.name,
                (pub[key], ours[key]),
                textcoords="offset points",
                xytext=(6, -10),
                fontsize=7.5,
                color=MUTED,
            )
    for ax, label, unit in ((ax_t, "Transmissivity", " (m²/d)"), (ax_s, "Storativity", "")):
        lo, hi = ax.get_xlim()
        lo, hi = min(lo, ax.get_ylim()[0]), max(hi, ax.get_ylim()[1])
        ax.loglog([lo, hi], [lo, hi], color=MUTED, lw=1, ls="--", label="1:1")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(f"published{unit}")
        ax.set_ylabel(f"pumptest{unit}")
        ax.set_title(label, color=INK)
        ax.legend(loc="upper left")
        _grid(ax)
    fig.suptitle("pumptest against AQTESOLV and Kruseman and de Ridder", color=INK, fontsize=12)
    path = OUT / "parity.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def single_well_figure() -> Path:
    hz = datasets.load("hoogezand")
    t, s = hz.observations[0].time, hz.observations[0].drawdown
    # From 5 s, as on the validation page; half a second early so that the 5 s
    # point is kept whichever way its conversion to days rounded.
    keep = t >= 4.5 / 86400.0
    test = PumpingTest(hz.well, hz.schedule, [Observation("pumped well", 0.0, t[keep], s[keep])])
    model = PapadopulosCooper(skin=True, casing_radius=True)
    result = fit(test, model, fixed={"S": 4e-4})
    q, sw = datasets.step_test()
    st = step_drawdown(q, sw)

    fig, (ax_h, ax_w) = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    seconds = t[keep] * 86400.0
    line_t = np.logspace(np.log10(seconds.min()), np.log10(seconds.max()), 200)
    ax_h.semilogx(seconds, s[keep], "o", ms=4, mfc="none", mec=SERIES[0], mew=1.1, label="measured")
    ax_h.semilogx(
        line_t,
        model.drawdown(result.values, test, test.observations[0], line_t / 86400.0),
        color=SERIES[0],
        label="fitted",
    )
    dt, ds = log_derivative(t[keep], s[keep], window=0.3)
    ax_h.semilogx(dt * 86400.0, ds, "s", ms=3.5, color=SERIES[1], label="ds/d ln t")
    ax_h.set_title(
        f"Hoogezand single-well test\nT = {result.values['T']:.0f} m²/d (book 1800), "
        f"skin {result.values['Sk']:.1f}, rc {result.values['rc']:.3f} m",
        color=INK,
    )
    ax_h.set_xlabel("time (s)")
    ax_h.set_ylabel("drawdown (m)")
    ax_h.legend(loc="center right")
    _grid(ax_h)

    qq = np.linspace(0, q.max() * 1.1, 100)
    ax_w.plot(q, sw, "o", ms=6, mfc="none", mec=SERIES[0], mew=1.2, label="measured")
    ax_w.plot(
        qq,
        st.aquifer_loss_coefficient * qq + st.well_loss_coefficient * qq**2,
        color=SERIES[0],
        label="s = BQ + CQ², pumptest",
    )
    ax_w.plot(qq, 3.26e-3 * qq + 1.45e-7 * qq**2, color=MUTED, ls="--", lw=1, label="book line")
    ax_w.plot(qq, st.aquifer_loss_coefficient * qq, color=SERIES[2], lw=1, label="aquifer loss BQ")
    ax_w.set_title(
        f"Step-drawdown test 'Well 1'\nB = {st.aquifer_loss_coefficient:.3g}, "
        f"C = {st.well_loss_coefficient:.3g}",
        color=INK,
    )
    ax_w.set_xlabel("pumping rate (m³/d)")
    ax_w.set_ylabel("drawdown in the well (m)")
    ax_w.legend(loc="upper left")
    _grid(ax_w)
    path = OUT / "single_well.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for path in (fits_figure(), parity_figure(), single_well_figure()):
        print(f"wrote {path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
