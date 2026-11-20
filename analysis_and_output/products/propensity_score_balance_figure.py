"""
propensity_score_balance_figure.py
==================================
Draws the propensity-score (PS) balance figure (.png, .pdf) from its
source-data workbook (figure_propensity_score_balance.xlsx, written by
propensity_score.py). Computes nothing: every plotted value (SMDs, densities,
the common-support bounds, the group sizes) is read from the workbook; the
counts printed on the figure are formatted by render_utils.

Layout (3 rows x 2 columns):
  a, c, e  love plots, all patients / women / men: SMD of each covariate,
           AID+ minus AID-, unweighted (open circle) and weighted (filled
           circle); rows shaded by role (balanced by design, balance check);
           dashed lines at |SMD| = 0.10 (no line at 0). One symmetric
           x-range for the three panels, at least -0.80 to 0.80.
  b, d, f  PS density per arm on the shared grid, with the common-support
           band; n per arm in the legend. One y-range and one set of y ticks
           for the three panels, so the rows can be compared.

The workbook is checked before drawing (it stops on a missing sheet or column,
an unknown role, a missing density panel, density grids that differ between
b, d and f, a grid whose length is not the stated number of points, a balance
threshold other than the drawn 0.10, or a love plot whose n differs from its
density panel's), so the caption can give each stratum's n once.

Captions: when a caption file is given, the title and caption wording come
from it; this script fills in the numbers and writes them to a .docx beside
the figure.

Run from the data folder, after propensity_score.py.
"""

import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, MaxNLocator  # noqa: E402

import render_utils as ru  # noqa: E402

# =============================================================================
# PATHS (relative to the data folder the script is run from)
# =============================================================================
SOURCE_DATA_DIR = ""  # folder with figure_propensity_score_balance.xlsx written by propensity_score.py (sheets a-f, common_support, parameters)
FIGURES_DIR = ""  # folder for the figure files (.png, .pdf) written here
CAPTIONS_PATH = ""  # optional JSON with the title and caption; blank = no caption file

WORKBOOK = "figure_propensity_score_balance"
STEM = "figure_propensity_score_balance"

LOVE_COLUMNS = [
    "row_label", "variable", "role", "n_aid_pos", "n_aid_neg",
    "mean_aid_pos_unweighted", "sd_aid_pos_unweighted",
    "mean_aid_neg_unweighted", "sd_aid_neg_unweighted", "smd_unweighted",
    "mean_aid_pos_weighted", "sd_aid_pos_weighted",
    "mean_aid_neg_weighted", "sd_aid_neg_weighted", "smd_weighted",
]
DENSITY_COLUMNS = ["ps_score", "density_aid_pos", "density_aid_neg"]
SUPPORT_COLUMNS = [
    "panel", "stratum", "n_aid_pos", "n_aid_neg",
    "ps_min_aid_pos", "ps_max_aid_pos", "ps_min_aid_neg", "ps_max_aid_neg",
    "common_support_lower", "common_support_upper",
]
PARAMETERS = ["kde_bandwidth_factor", "kde_grid_points", "kde_grid_padding",
              "trim_percentile_per_tail", "balance_threshold"]
ROLES = ("Balanced by design", "Balance check")
DRAWN_THRESHOLD = 0.10   # the dashed lines of the love plots

# =============================================================================
# STYLE (Nature Aging: Arial/Helvetica; panel letters 8 pt, axis labels 7 pt,
# ticks and in-panel text 6-6.5 pt, threshold labels 5.5 pt; no stroke below
# 1.0 pt; 180 mm wide; open vs filled markers and shading, so color is never
# the only carrier of information; vector .pdf and 300 dpi .png)
# =============================================================================
plt.rcParams.update({
    "font.family":           "sans-serif",
    "font.sans-serif":       ["Arial", "Helvetica Neue", "Helvetica", "DejaVu Sans"],
    "font.size":             7,
    "axes.titlesize":        7,
    "axes.labelsize":        7,
    "xtick.labelsize":       6.5,
    "ytick.labelsize":       6.5,
    "legend.fontsize":       6,
    "legend.title_fontsize": 6,
    "axes.linewidth":        1.0,
    "xtick.major.width":     1.0,
    "ytick.major.width":     1.0,
    "xtick.major.size":      2.0,
    "ytick.major.size":      2.0,
    "axes.spines.top":       False,
    "axes.spines.right":     False,
    "lines.linewidth":       1.0,
    "patch.linewidth":       1.0,
    "pdf.fonttype":          42,
    "ps.fonttype":           42,
    "figure.dpi":            300,
    "savefig.dpi":           300,
})

MM    = 1 / 25.4
MAX_W = 180 * MM
FIG_H = 8.2
DPI   = 300

C_DESIGN = "#C0392B"
C_CHECK  = "#666666"
C_AID_P  = "#C0392B"
C_AID_N  = "#5B9EC9"

ROLE_SHADES = {
    "Balanced by design": "#EEF4FB",
    "Balance check":      "#F5F5F5",
}
ROLE_LABELS = {
    "Balanced by design": "Balanced\nby design",
    "Balance check":      "Balance\ncheck",
}
ROLE_COLORS = {
    "Balanced by design": C_DESIGN,
    "Balance check":      C_CHECK,
}

SMD_XLABEL = ("Standardized mean difference  (AID+ − AID−)\n"
              "← AID− higher                        AID+ higher →")

# (love panel, density panel, stratum heading), top to bottom
STRATA = (("a", "b", "All patients"), ("c", "d", "Women"), ("e", "f", "Men"))


# =============================================================================
# LOAD AND CHECK THE SOURCE DATA
# =============================================================================
def read_sheet(path, name, columns):
    df = pd.read_excel(path, sheet_name=name)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise RuntimeError(f"Sheet '{name}' is missing columns: {missing}")
    return df


def love_rows(df, name):
    bad_roles = sorted(set(df["role"]) - set(ROLES))
    if bad_roles:
        raise RuntimeError(f"Sheet '{name}' has unrecognized roles: {bad_roles}")
    # The sheet lists rows top to bottom; the axes are drawn bottom to top.
    df = df.iloc[::-1].reset_index(drop=True)
    return [(r.row_label, r.smd_unweighted, r.smd_weighted, r.role)
            for r in df.itertuples(index=False)]


def density_curves(df):
    return (df["ps_score"].to_numpy(float),
            df["density_aid_pos"].to_numpy(float),
            df["density_aid_neg"].to_numpy(float))


def load_source(path):
    """Every value the figure and caption use, checked."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path}: run propensity_score.py first")
    love_df = {p: read_sheet(path, p, LOVE_COLUMNS) for p in ("a", "c", "e")}
    love = {p: love_rows(df, p) for p, df in love_df.items()}
    density = {p: density_curves(read_sheet(path, p, DENSITY_COLUMNS)) for p in ("b", "d", "f")}
    support = read_sheet(path, "common_support", SUPPORT_COLUMNS).set_index("panel")
    for p in ("b", "d", "f"):
        if p not in support.index:
            raise RuntimeError(f"Sheet 'common_support' has no row for panel {p}")
    par = read_sheet(path, "parameters", ["parameter", "value"]).set_index("parameter")["value"]
    missing = [k for k in PARAMETERS if k not in par.index]
    if missing:
        raise RuntimeError(f"Sheet 'parameters' is missing: {missing}")

    # The three densities share one grid (the pooled PS range widened on both
    # sides), so its ends are the shared x-limits of b, d and f.
    grid = density["b"][0]
    for p in ("d", "f"):
        if not np.array_equal(density[p][0], grid):
            raise RuntimeError(f"Sheet '{p}' does not use the same ps_score grid as sheet 'b'")
    if len(grid) != int(par["kde_grid_points"]):
        raise RuntimeError(f"the density grid has {len(grid)} points, the parameters sheet "
                           f"states {int(par['kde_grid_points'])}")
    if float(par["balance_threshold"]) != DRAWN_THRESHOLD:
        raise RuntimeError(f"balance_threshold is {par['balance_threshold']}; the love plots "
                           f"draw |SMD| = {DRAWN_THRESHOLD}")

    # Each love plot and its density panel show the same patients.
    for lp, dp, _ in STRATA:
        s = support.loc[dp]
        for arm in ("n_aid_pos", "n_aid_neg"):
            ns = set(love_df[lp][arm].astype(int))
            if ns != {int(s[arm])}:
                raise RuntimeError(f"Sheet '{lp}' {arm} {sorted(ns)} differs from the density "
                                   f"panel {dp} ({int(s[arm])})")

    # One x-range for the three love plots: symmetric about zero, wide enough
    # for the largest plotted |SMD|, and never narrower than +/-0.80.
    smd_vals = [v for rows in love.values() for (_, uw, wt, _) in rows
                for v in (uw, wt) if not pd.isna(v)]
    reach = max([abs(v) for v in smd_vals] + [0.80]) + 0.06

    # One y-range and one set of ticks for the three density panels: the
    # highest curve of b, d and f plus headroom for the legend, ticks chosen
    # once on that range.
    ps_ytop = max(max(y_p.max(), y_n.max()) for (_, y_p, y_n) in density.values()) * 1.25
    ps_yticks = [t for t in MaxNLocator(nbins=10, steps=[1, 2, 2.5, 5, 10]).tick_values(0, ps_ytop)
                 if 0 <= t <= ps_ytop]
    return {"love": love, "density": density, "support": support, "parameters": par,
            "xlim_smd": (-reach, reach), "ps_xlim": (float(grid[0]), float(grid[-1])),
            "ps_ylim": (0.0, ps_ytop), "ps_yticks": ps_yticks}


# =============================================================================
# DRAWING
# =============================================================================
def strip_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def panel_tag(ax, letter, dx=-0.30, dy=1.06):
    ax.text(dx, dy, letter, transform=ax.transAxes,
            fontsize=8, fontweight="bold", va="top", ha="left", clip_on=False)


def stratum_tag(ax, text):
    ax.text(0.0, 1.045, text, transform=ax.transAxes,
            fontsize=7, fontweight="bold", color="#333333",
            va="bottom", ha="left", clip_on=False)


def draw_love(ax, smd_rows, xlim_smd, show_legend=False):
    """Open marker = unweighted SMD, filled marker = weighted SMD."""
    n_vars = len(smd_rows)
    y_pos  = np.arange(n_vars) * 1.0
    ylim_lo = -0.52
    ylim_hi = y_pos.max() + 0.52

    # Contiguous runs of the same role, in plotted order.
    blocks, prev, start = [], None, 0
    for i, (_, _, _, role) in enumerate(smd_rows):
        if role != prev:
            if prev is not None:
                blocks.append((prev, start, i - 1))
            prev, start = role, i
    blocks.append((prev, start, n_vars - 1))

    for role, i0, i1 in blocks:
        ax.axhspan(y_pos[i0] - 0.48, y_pos[i1] + 0.48,
                   color=ROLE_SHADES[role], zorder=0, lw=0)
        frac = ((y_pos[i0] + y_pos[i1]) / 2.0 - ylim_lo) / (ylim_hi - ylim_lo)
        ax.text(1.02, frac, ROLE_LABELS[role],
                ha="left", va="center", fontsize=6, color=ROLE_COLORS[role],
                style="italic", transform=ax.transAxes, clip_on=False)

    ax.axvline( 0.10, color="#AAAAAA", lw=1.0, ls="--", zorder=1)
    ax.axvline(-0.10, color="#AAAAAA", lw=1.0, ls="--", zorder=1)
    for _x, _txt in ((-0.10, "−0.10"), (0.10, "0.10")):
        ax.text(_x, ylim_hi + 0.10, _txt, fontsize=5.5, color="#8C8C8C",
                ha="center", va="bottom", clip_on=False)

    for i, (_, s_uw, s_wt, role) in enumerate(smd_rows):
        if pd.isna(s_uw) or pd.isna(s_wt):
            continue
        col = ROLE_COLORS[role]
        yi  = y_pos[i]
        ax.plot([s_uw, s_wt], [yi, yi], color=col, lw=1.0, alpha=0.55, zorder=3)
        ax.scatter(s_uw, yi, marker="o", s=18, facecolors="white",
                   edgecolors=col, linewidths=1.0, zorder=4, clip_on=False)
        ax.scatter(s_wt, yi, marker="o", s=18, facecolors=col,
                   edgecolors=col, linewidths=1.0, zorder=5, clip_on=False)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([r[0] for r in smd_rows], fontsize=6.5)
    _t = np.arange(-2.0, 2.0001, 0.2)
    ax.set_xticks([t for t in _t if xlim_smd[0] <= t <= xlim_smd[1]])
    ax.set_xlim(*xlim_smd)
    ax.set_ylim(ylim_lo, ylim_hi)
    ax.xaxis.grid(True, color="#F0F0F0", lw=1.0, zorder=0)
    ax.yaxis.grid(False)
    strip_axes(ax)

    if show_legend:
        handles = []
        for role, tag in (("Balanced by design", "balanced by design"),
                          ("Balance check", "balance check")):
            col = ROLE_COLORS[role]
            for face, kind in (("white", "Unweighted"), (col, "Weighted")):
                handles.append(Line2D(
                    [0], [0], marker="o", color="w", markerfacecolor=face,
                    markeredgecolor=col, markeredgewidth=1.0, markersize=4.5,
                    label=f"{kind}  |  {tag}"))
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.09),
                  ncol=2, frameon=False, fontsize=6, borderaxespad=0.0,
                  borderpad=0.3, handletextpad=0.4,
                  labelspacing=0.35, columnspacing=1.4)


def draw_support(ax, curves, s, ps_xlim, ps_ylim, ps_yticks, show_xlabel=True):
    """Propensity-score density in each arm, with the common-support band.
    ps_ylim and ps_yticks are shared by b, d and f."""
    x, y_p, y_n = curves
    cs_lo, cs_hi = float(s["common_support_lower"]), float(s["common_support_upper"])
    y_top = ps_ylim[1]

    ax.fill_between(x, y_p, alpha=0.32, color=C_AID_P)
    ax.fill_between(x, y_n, alpha=0.28, color=C_AID_N)
    ax.plot(x, y_p, color=C_AID_P, lw=1.0)
    ax.plot(x, y_n, color=C_AID_N, lw=1.0)

    ax.axvspan(cs_lo, cs_hi, alpha=0.10, color="#AAAAAA", zorder=0)
    ax.axvline(cs_lo, color="#888888", lw=1.0, ls="--", zorder=1)
    ax.axvline(cs_hi, color="#888888", lw=1.0, ls="--", zorder=1)
    ax.text(cs_hi - 0.045 * (ps_xlim[1] - ps_xlim[0]), y_top * 0.46,
            "Common\nsupport",
            ha="right", va="center", fontsize=6, color="#888888")

    if show_xlabel:
        ax.set_xlabel("Propensity score", fontsize=7, labelpad=3)
    ax.set_ylabel("Density", fontsize=7)
    ax.set_xlim(*ps_xlim)
    ax.set_ylim(*ps_ylim)
    ax.yaxis.set_major_locator(FixedLocator(ps_yticks))
    strip_axes(ax)

    handles = [
        mpatches.Patch(facecolor=C_AID_P, alpha=0.55,
                       label=f"AID+  (n={ru.fmt_count(s['n_aid_pos'])})"),
        mpatches.Patch(facecolor=C_AID_N, alpha=0.55,
                       label=f"AID−  (n={ru.fmt_count(s['n_aid_neg'])})"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.16, 1.0),
              frameon=False, fontsize=6, borderpad=0.4, labelspacing=0.35)


def build_figure(src):
    fig = plt.figure(figsize=(MAX_W, FIG_H), dpi=DPI)
    fig.patch.set_facecolor("white")

    gs = gridspec.GridSpec(
        3, 2, figure=fig,
        width_ratios=[1.35, 1.0],
        height_ratios=[1.0, 0.94, 0.94],
        wspace=0.55, hspace=0.45,
        left=0.13, right=0.95, top=0.955, bottom=0.065,
    )

    for row, (love_panel, support_panel, stratum) in enumerate(STRATA):
        last_row = row == len(STRATA) - 1

        ax_love = fig.add_subplot(gs[row, 0])
        draw_love(ax_love, src["love"][love_panel], src["xlim_smd"], show_legend=(row == 0))
        stratum_tag(ax_love, stratum)
        panel_tag(ax_love, love_panel)
        if last_row:
            ax_love.set_xlabel(SMD_XLABEL, fontsize=7, labelpad=3)

        ax_support = fig.add_subplot(gs[row, 1])
        draw_support(ax_support, src["density"][support_panel],
                     src["support"].loc[support_panel], src["ps_xlim"],
                     src["ps_ylim"], src["ps_yticks"], show_xlabel=last_row)
        stratum_tag(ax_support, stratum)
        panel_tag(ax_support, support_panel, dx=-0.26)
    return fig


# =============================================================================
# CAPTION
# =============================================================================
def ordinal(k):
    k = int(k)
    suffix = "th" if 10 <= k % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(k % 10, "th")
    return f"{k}{suffix}"


def caption_parts(cap, src):
    c, s, p = cap["figure"], src["support"], src["parameters"]
    trim = float(p["trim_percentile_per_tail"])
    if trim != int(trim):
        raise RuntimeError(f"trim_percentile_per_tail {trim} is not a whole percentile")
    values = {"threshold": ru.fmt_num(p["balance_threshold"], 2),
              "bandwidth": ru.fmt_num(p["kde_bandwidth_factor"], 2),
              "grid_points": ru.fmt_count(p["kde_grid_points"]),
              "grid_padding": ru.fmt_num(p["kde_grid_padding"], 2),
              "trim_low": ordinal(trim), "trim_high": ordinal(100 - trim)}
    for panel, tag in (("b", "all"), ("d", "female"), ("f", "male")):
        values[f"n_aid_pos_{tag}"] = ru.fmt_count(s.loc[panel, "n_aid_pos"])
        values[f"n_aid_neg_{tag}"] = ru.fmt_count(s.loc[panel, "n_aid_neg"])
    return c["title"], [ru.fill(c[k], values) for k in
                        ("intro", "love", "rows", "density", "model", "tables",
                         "abbreviations")]


def main():
    cap = ru.load_captions(CAPTIONS_PATH, __file__)
    src = load_source(os.path.join(SOURCE_DATA_DIR, WORKBOOK + ".xlsx"))
    fig = build_figure(src)
    for ext in ("pdf", "png"):
        ru.save_figure(fig, os.path.join(FIGURES_DIR, f"{STEM}.{ext}"), dpi=DPI,
                       facecolor="white", bbox_inches="tight")
    plt.close(fig)
    if cap:
        title, parts = caption_parts(cap, src)
        ru.write_caption(os.path.join(FIGURES_DIR, STEM + "_caption.docx"), title, parts)


if __name__ == "__main__":
    main()