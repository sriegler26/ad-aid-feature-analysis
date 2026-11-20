"""
dlb_core_feature_cooccurrence_figure.py
=======================================
Draws the DLB core feature co-occurrence figure (.png, .pdf) from the
source-data workbook written by dlb_core_analysis.py. Fits no model and
computes no statistic: every plotted value (percentages, Wilson CIs, Q values,
the window counts, the RBD baseline) is read from the workbook.

Layout:
- a: percentage of patients with each pair and the triad of the core features
  (parkinsonism, fluctuating cognition, visual hallucinations), AID+ and AID-,
  for women, all patients; women, coded DLB excluded; men, all patients; Wilson
  CIs; significance symbols from the Q of the AID+ versus AID- odds ratio.
- b: among AID+ women positive for each combination, the share with all, some
  or none of its features first documented by the window after the AD
  diagnosis date; coded DLB excluded beside coded DLB (disjoint groups).
- c: REM sleep behavior disorder (RBD) among AID+ women by core-feature burden (0-1 versus 2-3), coded DLB
  excluded beside all patients; dashed line, all AID+ women.

Captions: when a caption file is given, the title and caption wording come
from it; this script fills in the numbers and writes them to a .docx beside
the figure.

Run from the data folder, after dlb_core_analysis.py.
"""

import os
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import render_utils as ru

warnings.filterwarnings("ignore")

# =============================================================================
# PATHS (relative to the data folder the script is run from)
# =============================================================================
SOURCE_DATA_DIR = ""  # folder of the source-data workbooks (.xlsx) written by dlb_core_analysis.py
FIGURES_DIR = ""  # folder for the figure files written here
CAPTIONS_PATH = ""  # optional JSON with the title and caption; blank = no caption file

FIGURE_FILE = "figure_dlb_core_feature_cooccurrence"

# =============================================================================
# DRAWING SETTINGS
# =============================================================================
C_F_POS = "#C0392B"
C_M_POS = "#1A5276"
M = ru.MINUS

COMBOS = ["park_fluct", "park_vis_hall", "fluct_vis_hall", "triad"]
COMBO_SHORT = {"park_fluct": "Park. +\nFluct. cog.",
               "park_vis_hall": "Park. +\nVis. hall.",
               "fluct_vis_hall": "Fluct. cog. +\nVis. hall.",
               "triad": "All three"}
BURDEN = ["0_1", "2_3"]

# =============================================================================
# STYLE
# =============================================================================
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica Neue", "Helvetica", "DejaVu Sans"],
    "font.size": 6,
    "axes.titlesize": 6,
    "axes.labelsize": 6.5,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 5.5,
    "axes.linewidth": 1.0,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.major.size": 2.0,
    "ytick.major.size": 2.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.0,
    "patch.linewidth": 1.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# =============================================================================
# WORKBOOK
# =============================================================================
def workbook():
    path = os.path.join(SOURCE_DATA_DIR, FIGURE_FILE + ".xlsx")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path}: run dlb_core_analysis.py first")
    return pd.read_excel(path, sheet_name=None)


def parameter(wb, key):
    p = wb["parameters"].set_index("parameter")["value"]
    if key not in p.index:
        raise KeyError(f"parameters: {key} missing")
    return p[key]


# =============================================================================
# HELPERS
# =============================================================================
def save_fig(fig, name):
    for ext in ("pdf", "png"):
        ru.save_figure(fig, os.path.join(FIGURES_DIR, f"{name}.{ext}"), dpi=300,
                       facecolor="white", bbox_inches="tight")
    plt.close(fig)


def strip_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def panel_tag(ax, letter, dx=-0.18, dy=1.05):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=8,
            fontweight="bold", va="top", ha="left", clip_on=False)


def q_to_stars(q):
    if q is None or np.isnan(q):
        return ""
    if q < 0.001:
        return "***"
    if q < 0.01:
        return "**"
    if q < 0.05:
        return "*"
    return ""


# =============================================================================
# DATA
# =============================================================================
def panel_a_data(wb):
    a = wb["a"]
    a = a[a["set"] == "primary"]
    out = {}
    for _, r in a.iterrows():
        n = int(r["n"])
        out[(r["combination"], r["sex"], r["arm"], r["aid"])] = (
            (float(r["pct"]), float(r["ci_low"]), float(r["ci_high"]), n) if n > 0
            else (np.nan, np.nan, np.nan, 0))
    q = {}
    for _, r in wb["a_odds_ratios"].iterrows():
        q[(r["sex"], r["arm"], r["combination"])] = (
            float(r["q"]) if r["status"] == "fitted" else np.nan)
    return out, q


def panel_b_data(wb):
    b = wb["b"]
    out = {}
    for grp in ("dlb_excluded", "coded_dlb"):
        out[grp] = {}
        for c in COMBOS:
            m = b[(b["group"] == grp) & (b["combination"] == c)]
            n = int(m["n"].iloc[0]) if len(m) else 0
            if n == 0:
                out[grp][c] = dict(n=n, all=np.nan, some=np.nan, none=np.nan)
                continue
            r = m.iloc[0]
            out[grp][c] = dict(n=n, all=100 * r["all_in_window"] / n,
                               some=100 * r["some_in_window"] / n,
                               none=100 * r["none_in_window"] / n)
    return out


def panel_c_data(wb):
    c = wb["c"]
    out = {}
    for arm in ("dlb_excluded", "all"):
        out[arm] = {}
        for b in BURDEN:
            m = c[(c["arm"] == arm) & (c["core_burden"] == b)]
            n = int(m["n"].iloc[0]) if len(m) else 0
            out[arm][b] = ((float(m["pct"].iloc[0]), float(m["ci_low"].iloc[0]),
                            float(m["ci_high"].iloc[0]), n, int(m["events"].iloc[0]))
                           if n > 0 else (np.nan, np.nan, np.nan, 0, 0))
    base = wb["c_baseline"].iloc[0]
    return out, base


# =============================================================================
# FIGURE
# =============================================================================
def draw(wb):
    panelA, panelA_q = panel_a_data(wb)
    panelB = panel_b_data(wb)
    panelC, base = panel_c_data(wb)
    yrs = parameter(wb, "pre_peri_window_yrs_figure")
    window = f"by {yrs:g} yr after Dx"

    fig = plt.figure(figsize=(180 / 25.4, 5.4), dpi=300)
    gs = gridspec.GridSpec(2, 2, figure=fig, left=0.08, right=0.98, top=0.94, bottom=0.10,
                           hspace=0.34, wspace=0.42, height_ratios=[1.15, 1.0])
    ax_A = fig.add_subplot(gs[0, :])
    ax_B = fig.add_subplot(gs[1, 0])
    ax_C = fig.add_subplot(gs[1, 1])

    # ---- Panel a ----
    # color: sex; marker: AID status (diamond AID+, circle AID-); fill: arm
    # (filled all patients, open coded DLB excluded)
    X_COMBOS = np.arange(len(COMBOS), dtype=float)
    C_FEMALE = C_F_POS
    C_MALE = C_M_POS
    GROUPS_A = [
        # (sex, arm, aid, color, marker, size, filled)
        ("Female", "all", "AID+", C_FEMALE, "D", 16, True),
        ("Female", "all", "AID-", C_FEMALE, "o", 13, True),
        ("Female", "dlb_excluded", "AID+", C_FEMALE, "D", 16, False),
        ("Female", "dlb_excluded", "AID-", C_FEMALE, "o", 13, False),
        ("Male", "all", "AID+", C_MALE, "D", 16, True),
        ("Male", "all", "AID-", C_MALE, "o", 13, True),
    ]
    PAIR_CENTERS = [-0.22, -0.22, 0.0, 0.0, 0.22, 0.22]
    INTRA_SIGN = [-1, +1, -1, +1, -1, +1]
    INTRA_GAP = 0.045
    OFF = [c + s * INTRA_GAP for c, s in zip(PAIR_CENTERS, INTRA_SIGN)]
    SIG_PAIRS = [
        ((0, 1), "Female", "all", C_FEMALE),
        ((2, 3), "Female", "dlb_excluded", C_FEMALE),
        ((4, 5), "Male", "all", C_MALE),
    ]
    empty = (np.nan,) * 3 + (0,)

    for xi, combo in enumerate(COMBOS):
        for gi, (sex, arm, aid, color, marker, size, filled) in enumerate(GROUPS_A):
            pv, lo, hi, n = panelA.get((combo, sex, arm, aid), empty)
            if np.isnan(pv):
                continue
            x = xi + OFF[gi]
            ax_A.plot([x, x], [lo, hi], color=color,
                      lw=0.9 if filled else 0.7,
                      ls="-" if filled else (0, (1.4, 0.9)),
                      solid_capstyle="round", zorder=3)
            ax_A.scatter(x, pv, marker=marker, s=size,
                         facecolor=color if filled else "white",
                         edgecolor=color, linewidths=0 if filled else 0.8, zorder=5)
    ax_A.set_xticks(X_COMBOS)
    ax_A.set_xticklabels([COMBO_SHORT[c] for c in COMBOS], fontsize=5.5)
    ax_A.set_xlabel("Core DLB feature combination", fontsize=6.5)
    ax_A.set_ylabel("Patients with the\ncombination (%)", fontsize=6.5)
    ax_A.set_xlim(-0.5, len(COMBOS) - 0.5)
    _his = [panelA.get((c, g[0], g[1], g[2]), empty)[2] for c in COMBOS for g in GROUPS_A]
    _his = [h for h in _his if not np.isnan(h)]
    _ymax_a = (max(_his) if _his else 1.0) * 1.28
    ax_A.set_ylim(0, _ymax_a)
    ax_A.yaxis.grid(True, color="#F0F0F0", lw=0.5, zorder=0)
    ax_A.set_axisbelow(True)
    strip_axes(ax_A)

    _star_pad = _ymax_a * 0.025
    for xi, combo in enumerate(COMBOS):
        for (i1, i2), sex, arm, color in SIG_PAIRS:
            his = []
            for i in (i1, i2):
                hi = panelA.get((combo, GROUPS_A[i][0], GROUPS_A[i][1], GROUPS_A[i][2]), empty)[2]
                if not np.isnan(hi):
                    his.append(hi)
            if not his:
                continue
            stars = q_to_stars(panelA_q.get((sex, arm, combo), np.nan))
            if not stars:
                continue
            ax_A.text(xi + PAIR_CENTERS[i1], max(his) + _star_pad, stars, ha="center",
                      va="bottom", fontsize=6.0, color=color, fontweight="bold", zorder=6)
    leg_A = [
        Line2D([0], [0], marker="D", color="w", markerfacecolor=C_FEMALE,
               markeredgecolor=C_FEMALE, markersize=4.5, label="AID+, women, all"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_FEMALE,
               markeredgecolor=C_FEMALE, markersize=4.0, label=f"AID{M}, women, all"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="white",
               markeredgecolor=C_FEMALE, markersize=4.5, label="AID+, women, coded DLB excl."),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
               markeredgecolor=C_FEMALE, markersize=4.0, label=f"AID{M}, women, coded DLB excl."),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=C_MALE,
               markeredgecolor=C_MALE, markersize=4.5, label="AID+, men, all"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_MALE,
               markeredgecolor=C_MALE, markersize=4.0, label=f"AID{M}, men, all"),
    ]
    ax_A.legend(handles=leg_A, loc="upper right", frameon=True, fontsize=5.0,
                framealpha=0.95, edgecolor="#CCCCCC", borderpad=0.4,
                handletextpad=0.3, labelspacing=0.3, ncol=3)
    panel_tag(ax_A, "a", dx=-0.06)

    # ---- Panel b: paired stacked bars per combination ----
    ARMS_B = ["dlb_excluded", "coded_dlb"]
    X_B = np.arange(len(COMBOS), dtype=float)
    BAR_W = 0.30
    BAR_OFF = {"dlb_excluded": -0.17, "coded_dlb": +0.17}
    SEG_COLORS = {
        "dlb_excluded": ["#7A1E1E", "#C0392B", "#F0C7C2"],
        "coded_dlb": ["#123A54", "#2E7BA6", "#C3DCEA"],
    }
    seg_labels = [f"Both/all {window}", f"Some {window}", f"None {window}"]
    seg_keys = ["all", "some", "none"]

    for arm in ARMS_B:
        bottoms = np.zeros(len(COMBOS))
        for seg_key, color in zip(seg_keys, SEG_COLORS[arm]):
            vals = np.array([panelB[arm][c][seg_key] if not np.isnan(panelB[arm][c][seg_key]) else 0
                             for c in COMBOS], dtype=float)
            ax_B.bar(X_B + BAR_OFF[arm], vals, bottom=bottoms, width=BAR_W,
                     color=color, zorder=3)
            bottoms += vals
        for xi, combo in enumerate(COMBOS):
            ax_B.text(xi + BAR_OFF[arm], 101.5, ru.fmt_count(panelB[arm][combo]["n"]),
                      ha="center", va="bottom", fontsize=4.4, color="#444444")
    ax_B.text(-0.48, 101.5, "n=", ha="left", va="bottom", fontsize=4.4, color="#444444")
    ax_B.set_xticks(X_B)
    ax_B.set_xticklabels([COMBO_SHORT[c] for c in COMBOS], fontsize=5.2)
    ax_B.set_xlabel("Core DLB feature combination", fontsize=6.5)
    ax_B.set_ylabel("Patients (%)", fontsize=6.5)
    ax_B.set_xlim(-0.55, len(COMBOS) - 0.45)
    ax_B.set_ylim(0, 108)
    ax_B.set_yticks([0, 25, 50, 75, 100])
    strip_axes(ax_B)
    leg_B = ([Patch(facecolor=c, edgecolor="none", label=f"{l} (coded DLB excl.)")
              for c, l in zip(SEG_COLORS["dlb_excluded"], seg_labels)] +
             [Patch(facecolor=c, edgecolor="none", label=f"{l} (coded DLB)")
              for c, l in zip(SEG_COLORS["coded_dlb"], seg_labels)])
    ax_B.legend(handles=leg_B, loc="lower center", frameon=True, fontsize=4.2,
                framealpha=0.95, edgecolor="#CCCCCC", borderpad=0.3,
                handletextpad=0.3, labelspacing=0.25, columnspacing=0.8, ncol=2)
    ax_B.set_title("AID+ women", fontsize=5.2, style="italic", color="#555555", pad=6)
    panel_tag(ax_B, "b")

    # ---- Panel c: RBD by core-feature burden ----
    ARMS_C = ["dlb_excluded", "all"]
    X_C = np.arange(2, dtype=float)
    BAR_W_C = 0.32
    BAR_OFF_C = {"dlb_excluded": -0.18, "all": +0.18}
    COLOR_C = {"dlb_excluded": C_F_POS, "all": C_M_POS}
    for arm in ARMS_C:
        xs = [xi + BAR_OFF_C[arm] for xi in range(2)]
        ys = [panelC[arm][b][0] for b in BURDEN]
        ax_C.bar(xs, ys, width=BAR_W_C, color=COLOR_C[arm], zorder=3)
        for xi, b in enumerate(BURDEN):
            pv, lo, hi, n, k = panelC[arm][b]
            if np.isnan(pv):
                continue
            x = xi + BAR_OFF_C[arm]
            ax_C.plot([x, x], [lo, hi], color="black", lw=0.8,
                      solid_capstyle="round", zorder=4)
    base_pct = float(base["pct"]) if int(base["n"]) > 0 else np.nan
    if not np.isnan(base_pct):
        ax_C.axhline(base_pct, color="#888888", lw=0.7, ls=(0, (2, 1.5)), zorder=1)
    ax_C.set_xticks(X_C)
    ax_C.set_xticklabels(["0–1 core\nfeatures", "2–3 core\nfeatures"], fontsize=5.5)
    ax_C.set_xlabel("Core McKeith feature burden", fontsize=6.5)
    ax_C.set_ylabel("REM sleep behavior disorder (RBD)\namong AID+ women (%)",
                    fontsize=6.5)
    ax_C.set_xlim(-0.55, 1.55)
    _hi_vals = [panelC[a][b][2] for a in ARMS_C for b in BURDEN if not np.isnan(panelC[a][b][2])]
    _ymax_c = max(max(_hi_vals) if _hi_vals else 1.0,
                  base_pct if not np.isnan(base_pct) else 0.0) * 1.15
    ax_C.set_ylim(0, _ymax_c)
    ax_C.yaxis.grid(True, color="#F0F0F0", lw=0.5, zorder=0)
    ax_C.set_axisbelow(True)
    strip_axes(ax_C)
    leg_C = [Patch(facecolor=C_F_POS, edgecolor="none", label="Coded DLB excluded"),
             Patch(facecolor=C_M_POS, edgecolor="none", label="All patients")]
    ax_C.legend(handles=leg_C, loc="upper left", frameon=True, fontsize=4.8, framealpha=0.95,
                edgecolor="#CCCCCC", borderpad=0.3, handletextpad=0.3, labelspacing=0.3)
    ax_C.set_title("AID+ women", fontsize=5.2, style="italic", color="#555555", pad=6)
    panel_tag(ax_C, "c")

    save_fig(fig, FIGURE_FILE)
    return dict(panelB=panelB, panelC=panelC, base=base, yrs=yrs)


# =============================================================================
# CAPTION
# =============================================================================
def captions(cap, wb, info):
    c = cap["caption"]
    n = lambda key: ru.fmt_count(parameter(wb, key))
    a = wb["a"]

    def arm_n(sex, arm, aid):
        m = a[(a["sex"] == sex) & (a["arm"] == arm) & (a["aid"] == aid)]
        return ru.fmt_count(m["n"].iloc[0]) if len(m) else ru.DASH
    out = [c["body"],
           ru.fill(c["a"], {"aid_pos_female": arm_n("Female", "all", "AID+"),
                            "aid_neg_female": arm_n("Female", "all", "AID-"),
                            "aid_pos_female_x": arm_n("Female", "dlb_excluded", "AID+"),
                            "aid_neg_female_x": arm_n("Female", "dlb_excluded", "AID-"),
                            "aid_pos_male": arm_n("Male", "all", "AID+"),
                            "aid_neg_male": arm_n("Male", "all", "AID-")}),
           c["a_symbols"],
           ru.fill(c["b"], {"yrs": f"{info['yrs']:g}"}),
           c["c"]]
    base = info["base"]
    out.append(ru.fill(c["c_baseline"], {"events": ru.fmt_count(base["events"]),
                                         "n": ru.fmt_count(base["n"]),
                                         "pct": ru.fmt_pct(base["pct"])}))
    out.append(c["ci"])
    out.append(c["ceiling"])
    return out


def main():
    cap = ru.load_captions(CAPTIONS_PATH, __file__)
    print(f"  Title and caption: {'written' if cap else 'none (no caption file set)'}")
    wb = workbook()
    info = draw(wb)
    if cap:
        ru.write_caption(os.path.join(FIGURES_DIR, FIGURE_FILE + "_caption.docx"),
                         cap["title"], captions(cap, wb, info))


if __name__ == "__main__":
    main()