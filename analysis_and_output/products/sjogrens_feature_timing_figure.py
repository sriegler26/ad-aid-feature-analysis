"""
sjogrens_feature_timing_figure.py
=================================
Draws the Sjogren's timing figure from its source-data workbook
(figure_sjogrens_feature_timing.xlsx, written by aid_condition_analysis.py);
computes nothing.
Panels:
  a  gap between the Sjogren's diagnosis and the first counted core feature,
     AID+ women with Sjogren's: the gaps in 0.5-year bins (amber, within 1 year of
     the diagnosis; blue, outside), with the median tick, the Q1-Q3 bar, the
     bootstrap CI band of the median and the 1-year band (sheets a_bins,
     a_summary); the source data hold no patient rows, so the gaps are
     drawn as binned counts;
  b  share of women concurrent (Wilson CI) vs their share of the core-feature
     burden (cluster bootstrap CI), women with a dated diagnosis (sheet b_burden).
When a caption file is given, the title and caption go to a .docx beside the
figure.

Run from the data folder, after aid_condition_analysis.py.
"""

import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

import render_utils as ru  # noqa: E402

# =============================================================================
# PATHS (relative to the data folder the script is run from)
# =============================================================================
SOURCE_DATA_DIR = ""  # folder of the source-data workbooks (.xlsx) written by aid_condition_analysis.py
FIGURES_DIR = ""  # folder for the figure files (.png, .pdf) written here
CAPTIONS_PATH = ""  # optional JSON with the title and caption; blank = no caption file

WORKBOOK = "figure_sjogrens_feature_timing"
STEM = "figure_sjogrens_feature_timing"
MM = 1 / 25.4
MAX_W = 180 * MM
FIG_H = 62 * MM
DPI = 300
C_SIG, C_TEXT, C_BLUE, C_AMBER, C_LGREY = "#1A1A1A", "#333333", "#2166AC", "#C77B30", "#D8D8D8"
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica Neue", "Helvetica", "DejaVu Sans"],
    "font.size": 6, "axes.labelsize": 6.5, "xtick.labelsize": 6, "ytick.labelsize": 6,
    "legend.fontsize": 5.0, "axes.linewidth": 1.0, "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})


def stars(p):
    if p is None or not np.isfinite(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def main():
    cap = ru.load_captions(CAPTIONS_PATH, __file__)
    path = os.path.join(SOURCE_DATA_DIR, WORKBOOK + ".xlsx")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path}: run aid_condition_analysis.py first")
    wb = pd.read_excel(path, sheet_name=None)
    bins, a, b = wb["a_bins"], wb["a_summary"].iloc[0], wb["b_burden"].iloc[0]
    p = wb["parameters"].set_index("parameter")["value"]
    cy = float(p["concurrent_yrs"])
    fig = plt.figure(figsize=(MAX_W, FIG_H))
    ax1 = fig.add_axes([0.070, 0.220, 0.400, 0.620])
    ax2 = fig.add_axes([0.610, 0.220, 0.360, 0.620])
    fig.text(0.012, 0.965, "a", fontsize=8, fontweight="bold", va="top", ha="left")
    fig.text(0.552, 0.965, "b", fontsize=8, fontweight="bold", va="top", ha="left")

    # ---- panel a: binned gaps --------------------------------------------------
    ax1.axvspan(-cy, cy, color="#F2F2F2", zorder=0, lw=0)
    ax1.axvline(0, color=C_SIG, lw=0.8, zorder=1)
    top = 1
    if len(bins):
        w = bins["bin_end_yrs"] - bins["bin_start_yrs"]
        ax1.bar(bins["bin_start_yrs"], bins["n_concurrent"], width=w, align="edge",
                color=C_AMBER, edgecolor="white", linewidth=0.25, zorder=3)
        ax1.bar(bins["bin_start_yrs"], bins["n_not_concurrent"], width=w, align="edge",
                bottom=bins["n_concurrent"], color=C_BLUE, edgecolor="white", linewidth=0.25,
                zorder=3)
        top = max(1, int((bins["n_concurrent"] + bins["n_not_concurrent"]).max()))
    yb = -0.18 * top
    if a["n"] > 0 and np.isfinite(a.get("median_yrs", np.nan)):
        ax1.plot([a["q1_yrs"], a["q3_yrs"]], [yb, yb], color=C_SIG, lw=1.5,
                 solid_capstyle="butt", zorder=4)
        if np.isfinite(a["median_ci_low"]):
            ax1.plot([a["median_ci_low"], a["median_ci_high"]], [yb, yb], color=C_SIG, lw=3.2,
                     solid_capstyle="butt", alpha=0.32, zorder=4)
        ax1.plot([a["median_yrs"]], [yb], marker="|", color=C_SIG, markersize=6.5, mew=1.5,
                 zorder=5)
        ax1.annotate(f"Median {stars(a['p_wilcoxon'])}".rstrip(), (a["median_yrs"], yb),
                     textcoords="offset points", xytext=(6, -8), ha="left", va="top",
                     fontsize=4.8, color=C_TEXT)
    ax1.set_ylim(-0.36 * top, top * 1.08)
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))  # counts: whole-number ticks only
    ax1.set_yticks([t for t in ax1.get_yticks() if 0 <= t <= top])
    ax1.set_ylabel("Women, n", fontsize=5.4, labelpad=2)
    ax1.tick_params(axis="both", labelsize=5.0, length=2, width=0.6, pad=1.5)
    ax1.set_xlabel("Time from Sjögren's diagnosis to first documented core feature (years)",
                   fontsize=5.4, labelpad=2)
    for s in ["top", "right"]:
        ax1.spines[s].set_visible(False)
    ax1.spines["bottom"].set_linewidth(0.6)
    ax1.spines["left"].set_linewidth(0.6)

    # ---- panel b: share of women vs burden share -----------------------------
    if b["n_carriers"] > 0:
        for (val, lo, hi), y in (((b["share_carriers_pct"], b["share_carriers_ci_low"],
                                   b["share_carriers_ci_high"]), 1),
                                 ((b["share_burden_pct"], b["share_burden_ci_low"],
                                   b["share_burden_ci_high"]), 0)):
            ax2.barh(y, val, height=0.5, color=C_AMBER, edgecolor=C_SIG, linewidth=0.6, zorder=2)
            ax2.barh(y, 100 - val, height=0.5, left=val, color=C_LGREY, edgecolor=C_SIG,
                     linewidth=0.6, zorder=2)
            if np.isfinite(lo):
                ax2.plot([lo, hi], [y, y], color=C_SIG, lw=1.0, zorder=4)
                for e in (lo, hi):
                    ax2.plot([e, e], [y - 0.10, y + 0.10], color=C_SIG, lw=1.0, zorder=4)
            ax2.text(val, y + 0.34, ru.fmt_num(val, 0) + "%", ha="center", va="bottom",
                     fontsize=5.0, color=C_TEXT, fontweight="bold")
    ax2.set_yticks([1, 0])
    ax2.set_yticklabels(["Share of\nwomen", "Share of\ncore-feature burden"], fontsize=5.2,
                        linespacing=1.2)
    ax2.set_xlim(0, 100)
    ax2.set_ylim(-0.55, 1.72)
    ax2.set_xticks([0, 25, 50, 75, 100])
    ax2.tick_params(axis="x", labelsize=5.0, length=2, width=0.6, pad=1.5)
    ax2.tick_params(axis="y", length=0, pad=2)
    ax2.set_xlabel("Share (%), women with Sjögren's syndrome and a dated diagnosis", fontsize=5.4,
                   labelpad=2)
    for s in ["top", "right", "left"]:
        ax2.spines[s].set_visible(False)
    ax2.spines["bottom"].set_linewidth(0.6)
    ax2.legend(handles=[
        mpatches.Patch(facecolor=C_AMBER, edgecolor=C_SIG, linewidth=0.6,
                       label=f"First feature within ±{cy:g} year of diagnosis"),
        mpatches.Patch(facecolor=C_LGREY, edgecolor=C_SIG, linewidth=0.6,
                       label=f"First feature more than {cy:g} year from diagnosis")],
        loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=1, frameon=False, fontsize=4.6,
        handlelength=1.3, handletextpad=0.45)
    for ext in ("png", "pdf"):
        ru.save_figure(fig, os.path.join(FIGURES_DIR, f"{STEM}.{ext}"), dpi=DPI,
                       facecolor="white")
    plt.close(fig)
    if cap:
        c = cap["figure"]
        n = {"n": ru.fmt_count(a["n"]), "n_boot": ru.fmt_count(p["n_boot"]),
             "n_perm": ru.fmt_count(p["n_permutations"]), "cy": f"{cy:g}",
             "bin": ru.fmt_num(p["gap_bin_yrs"], 1), "min_group": int(p["min_group_n"])}
        parts = [ru.fill(c["body"], n), c["features"], ru.fill(c["a"], n), c["tests"],
                 ru.fill(c["b"], n)]
        if a["status"] != "tested":
            parts.append(ru.fill(c["not_tested"], n))
        parts.append(c["stars"])
        ru.write_caption(os.path.join(FIGURES_DIR, STEM + "_caption.docx"), c["title"], parts)


if __name__ == "__main__":
    main()