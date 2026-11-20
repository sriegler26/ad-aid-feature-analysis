"""
aid_condition_figure.py
=======================
Draws the AID condition figure and the extended data figure of
core-feature combinations by condition from the source-data workbooks written
by aid_condition_analysis.py; computes nothing.

AID condition figure (180 x 210 mm; panels in narrative order; workbook
figure_aid_condition_spread.xlsx):
  a  conditions ranked by the mutually adjusted rate ratio of core features in
     women (core features relative to expected, 95% CI; 1, as in AID- women),
     marker area by women with the condition (quartile tiers),
     pooled value (line) and 95% prediction interval (band), x for a condition
     not estimated; beside it, membership in five published groupings
     (filled, the class named in the header; half-filled, a middle class where the
     grouping has one: limited or conflicting PD-risk evidence, weak PD
     pleiotropy; open, the
     contrasting class; blank, not classified), with a key below the columns
                                                              sheets a_conditions, a_pooled
  b  excess core features (observed - expected) by condition, AID+ women:
     single feature vs two or more, the latter split by coded DLB/PD
                                                              sheet b_excess
  c  first recorded AID diagnosis in years relative to the AD diagnosis (0),
     per condition in panel a's order (median, IQR; negative, AID first; line
     at 0): the workbook's AID-to-AD gaps (positive, AID first) negated in the
     drawing code; x for a condition with fewer than 10 dated women; the
     x-axis drawn as a timeline, with an arrowhead at its right end
                                                              sheet c_aid_dx_timing
  d  possible mimics: IPW odds ratio in women (95% CI) of >= 1 and >= 2 core
     features in all women (reference) and after removing the AID+ women
     of the conditions in each cumulative set (narrow, wider, all reported);
     faint lines, the reference values                       sheet d_possible_mimics_ipw
Extended data (120 mm wide; workbook extended_data_aid_condition_combinations
.xlsx): SIR of each core-feature combination by condition, AID+ women
(color, log scale), with the number of women (cell text).

On-figure text is limited to axis labels, tick labels, panel letters and
legends; counts, estimates and definitions are in the caption. Journal
artwork rules (Nature research journals): width <= 180 mm, sans-serif text
5-7 pt, panel letters 8 pt bold, RGB, vector PDF with editable text (heatmap
and color bar as vector cells). When a caption file is given, the titles and
captions go to .docx files beside the figures.

Run from the data folder, after aid_condition_analysis.py.
"""

import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.legend import Legend  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import render_utils as ru  # noqa: E402

# =============================================================================
# PATHS (relative to the data folder the script is run from)
# =============================================================================
SOURCE_DATA_DIR = ""  # folder of the source-data workbooks (.xlsx) written by aid_condition_analysis.py
FIGURES_DIR = ""  # folder for the figure files (.png, .pdf) written here
CAPTIONS_PATH = ""  # optional JSON with titles and captions; blank = no caption files

WORKBOOK = "figure_aid_condition_spread"
STEM = "figure_aid_condition_spread"
EXTENDED_WORKBOOK = "extended_data_aid_condition_combinations"
EXTENDED_STEM = "extended_data_aid_condition_combinations"

# =============================================================================
# STYLE
# =============================================================================
MM = 1 / 25.4
FIG_W, FIG_H = 180 * MM, 210 * MM
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
INK, INK2, GRID, REF = "#0b0b0b", "#52514e", "#d9d8d4", "#6f6e69"
FEMALES = CAT[0]
SINGLE, MULTI, MULTI_LBD = "#9b9a95", "#86b6ef", "#1c5cab"
M = ru.MINUS
LW = 0.8
SIZE_LEVELS = [8.0, 18.0, 32.0, 52.0]     # marker areas (pt^2) of the quartile tiers of women
X_LEFT = 0.01
plt.rcParams.update({
    "font.size": 6, "axes.titlesize": 6, "axes.labelsize": 6, "xtick.labelsize": 6,
    "ytick.labelsize": 6, "legend.fontsize": 5.5, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK, "axes.linewidth": LW,
    "xtick.major.width": LW, "ytick.major.width": LW, "lines.linewidth": LW,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "pdf.fonttype": 42, "ps.fonttype": 42})

# Panel a membership columns: (grouping, header, filled class, half-filled class or None,
# open class or None); any other value (not classified) is left blank
MEMBERSHIP = [("serology", "Seropositive", "seropositive", None, "seronegative"),
              ("systemic", "Systemic", "systemic", None, "organ_specific"),
              ("interferon", "Type I IFN", "ifn_core", None, "ifn_other"),
              ("pd_risk", "PD risk raised", "pd_raised", "pd_mixed", "pd_not_raised"),
              ("pleiotropy", "PD pleiotropy", "pleiotropic_strong", "pleiotropic_weak", None)]
NOT_CLASSIFIED = "not_classified"   # aid_condition_analysis.py UNASSIGNED: drawn blank
COMBO_ORDER = ["none", "P", "F", "V", "F+P", "V+P", "F+V", "F+V+P"]
COMBO_TICK = {"none": "None", "P": "Park.", "F": "Fluct.\ncog.", "V": "Vis.\nhall.",
              "F+P": "Park. +\nFluct. cog.", "V+P": "Park. +\nVis. hall.",
              "F+V": "Fluct. cog.\n+ Vis. hall.", "F+V+P": "All three"}
SET_TICK = {"reference": "All\nwomen", "narrow": "Narrow", "wider": "+ Wider",
            "all_reported": "+ All\nreported"}
SET_CAPTION = {"narrow": "narrow", "wider": "+ wider", "all_reported": "+ all reported"}
D_OUTCOMES = [("core_1plus", FEMALES, "o", "≥ 1 core feature"),
              ("core_2plus", CAT[6], "D", "≥ 2 core features")]


# =============================================================================
# HELPERS
# =============================================================================
def workbook(name):
    path = os.path.join(SOURCE_DATA_DIR, name + ".xlsx")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path}: run aid_condition_analysis.py first")
    return pd.read_excel(path, sheet_name=None)


def parameter(wb, key):
    p = wb["parameters"].set_index("parameter")["value"]
    if key not in p.index:
        raise KeyError(f"parameters: {key} missing")
    return p[key]


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=2, width=LW)


def letter(fig, ax, s, x_fig):
    """Panel letter at a fixed figure x, just above its axes (8 pt bold)."""
    y = ax.get_position().y1 + 4 / (fig.get_size_inches()[1] * 72)
    fig.text(x_fig, y, s, fontsize=8, fontweight="bold", va="bottom", ha="left")


def save(fig, stem):
    for ext in ("pdf", "png"):
        ru.save_figure(fig, os.path.join(FIGURES_DIR, f"{stem}.{ext}"), dpi=300, format=ext)
    plt.close(fig)


def size_edges(n):
    """Quartile edges of the carrier counts (the size tiers of the other figures)."""
    e = [float(np.percentile(n, q)) for q in (25, 50, 75)]
    for i in range(1, 3):
        e[i] = max(e[i], e[i - 1] + 1e-6)
    return e


def size_of(n, edges):
    for i, e in enumerate(edges):
        if n <= e:
            return SIZE_LEVELS[i]
    return SIZE_LEVELS[-1]


def size_labels(edges, nmin, nmax):
    b = [int(nmin)] + [int(np.floor(e)) for e in edges] + [int(nmax)]
    return [f"{b[0]}{ru.DASH}{b[1]}"] + [f"{b[i] + 1}{ru.DASH}{b[i + 1]}" for i in range(1, 4)]


def log_ticks(lo, hi, candidates=(0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8)):
    return [t for t in candidates if lo <= t <= hi]


# =============================================================================
# AID CONDITION FIGURE
# =============================================================================
def draw_figure(wb):
    A, P = wb["a_conditions"], wb["a_pooled"].iloc[0]
    C, T, D = wb["b_excess"], wb["c_aid_dx_timing"], wb["d_possible_mimics_ipw"]
    min_dated = int(parameter(wb, "min_group_n"))
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    outer = fig.add_gridspec(3, 1, height_ratios=[2.0, 1.3, 1.9], hspace=0.72,
                             left=0.20, right=0.985, top=0.935, bottom=0.075)
    info = {}

    # a: forest and membership matrix
    ga = outer[0].subgridspec(1, 2, width_ratios=[2.2, 1.0], wspace=0.04)
    ax = fig.add_subplot(ga[0])
    axm = fig.add_subplot(ga[1], sharey=ax)
    est = A[A["estimated"] == 1].sort_values("rate_ratio")
    t = pd.concat([A[A["estimated"] != 1].sort_values("row_label"), est]).reset_index(drop=True)
    yv = np.arange(len(t))
    edges = size_edges(t["n_female"])
    ax.axvspan(P["prediction_low"], P["prediction_high"], color=FEMALES, alpha=0.10, lw=0)
    ax.axvline(P["pooled"], color=FEMALES, lw=1.0)
    ax.axvline(1, color=INK2, lw=LW, ls="--")
    ok = (t["estimated"] == 1).to_numpy()
    ax.hlines(yv[ok], t["ci_low"][ok], t["ci_high"][ok], color=INK2, lw=LW)
    ax.scatter(t["rate_ratio"][ok], yv[ok], s=[size_of(n, edges) for n in t["n_female"][ok]],
               color=INK, zorder=3, lw=0)
    ax.scatter(np.ones((~ok).sum()), yv[~ok], marker="x", s=14, color=INK, lw=0.9, zorder=3)
    ax.set_xscale("log")
    ax.minorticks_off()
    ax.set_xlim(min(0.8, float(np.nanmin(t["ci_low"])) * 0.92),
                max(2.2, float(np.nanmax(t["ci_high"])) * 1.08))
    tk = log_ticks(*ax.get_xlim())
    ax.set_xticks(tk, [f"{v:g}" for v in tk])
    ax.set_yticks(yv, list(t["row_label"]))
    ax.set_ylim(-0.7, len(t) - 0.3)
    ax.set_xlabel("Core features relative to expected (×)")
    ax.text(1, 1.01, f"as in AID{M} women", transform=ax.get_xaxis_transform(), ha="center",
            va="bottom", fontsize=5, color=INK2)
    style(ax)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    labels = size_labels(edges, t["n_female"].min(), t["n_female"].max())
    ax.legend([Line2D([], [], ls="none", marker="o", markersize=np.sqrt(s), color=INK, mew=0)
               for s in SIZE_LEVELS], labels, title="Women, n", title_fontsize=5.5,
              loc="upper left", bbox_to_anchor=(0.0, -0.17), frameon=False, handletextpad=0.2,
              borderaxespad=0, ncol=4, columnspacing=1.0)
    info["size_tiers"] = ", ".join(labels)
    for j, (g, head, fill, shade, opn) in enumerate(MEMBERSHIP):
        unknown = set(t["class_" + g]) - {fill, shade, opn, NOT_CLASSIFIED}
        if unknown:
            raise KeyError(f"panel a, {g}: classes not in MEMBERSHIP {sorted(unknown)}")
        for i, v in enumerate(t["class_" + g]):
            if v == fill:
                axm.scatter(j, i, s=26, color=INK, lw=0)
            elif shade is not None and v == shade:
                axm.plot([j], [i], ls="none", marker="o", markersize=float(np.sqrt(26)),
                         fillstyle="left", markerfacecolor=INK, markerfacecoloralt="white",
                         markeredgecolor=INK, markeredgewidth=0.8)
            elif opn is not None and v == opn:
                axm.scatter(j, i, s=26, facecolor="white", edgecolor=INK, lw=0.8)
    axm.set_xlim(-0.6, len(MEMBERSHIP) + 0.6)
    for j in range(len(MEMBERSHIP)):
        axm.axvline(j, color=GRID, lw=0.5, zorder=0)
    axm.xaxis.tick_top()
    axm.set_xticks(range(len(MEMBERSHIP)), [m[1] for m in MEMBERSHIP], rotation=60, ha="left",
                   rotation_mode="anchor")
    axm.tick_params(axis="x", length=0, pad=1)
    axm.tick_params(axis="y", left=False, labelleft=False)
    for s in axm.spines.values():
        s.set_visible(False)
    # key to the membership columns, below them: one row each for all groupings,
    # PD risk and PD pleiotropy, left-aligned on a common edge (right edge of the
    # widest row at the columns' right edge)
    km = dict(ls="none", marker="o", markersize=float(np.sqrt(26)) * 0.85,
              markeredgecolor=INK, markeredgewidth=0.8)
    k_fill = Line2D([], [], markerfacecolor=INK, **km)
    k_half = Line2D([], [], markerfacecolor=INK, fillstyle="left", markerfacecoloralt="white",
                    **km)
    k_open = Line2D([], [], markerfacecolor="white", **km)
    k_none = Line2D([], [], ls="none")
    key_rows = [([k_fill, k_open, k_none],
                 ["class named", "contrasting class", "(blank, not classified)"]),
                ([k_none, k_fill, k_half, k_open],
                 ["PD risk:", "raised", "limited or conflicting", "not raised"]),
                ([k_none, k_fill, k_half],
                 ["PD pleiotropy:", "moderate to strong", "weak"])]
    key_y = [-0.17 - 0.075 * r for r in range(len(key_rows))]
    keys = []
    for (hs, ls), yk in zip(key_rows, key_y):
        leg = Legend(axm, hs, ls, loc="upper left", bbox_to_anchor=(0.0, yk), ncol=len(hs),
                     frameon=False, handlelength=0.9, handletextpad=0.25,
                     columnspacing=0.8, borderaxespad=0, borderpad=0)
        axm.add_artist(leg)
        keys.append(leg)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    to_ax = axm.transAxes.inverted()
    widest = max(to_ax.transform(lg.get_window_extent(rend))[1][0]
                 - to_ax.transform(lg.get_window_extent(rend))[0][0] for lg in keys)
    for lg, yk in zip(keys, key_y):
        lg.set_bbox_to_anchor((1.0 - widest, yk), transform=axm.transAxes)
    letter(fig, ax, "a", X_LEFT)

    # b: excess by condition (full width)
    ax = fig.add_subplot(outer[1])
    e = C.sort_values("total", ascending=False).reset_index(drop=True)
    xv = np.arange(len(e))
    ax.bar(xv - 0.2, e["single"], width=0.38, color=SINGLE, lw=0, label="Single feature")
    ax.bar(xv + 0.2, e["multi_coded_lbd"], width=0.38, color=MULTI_LBD, lw=0,
           label="Two or more, coded DLB/PD")
    same_sign = (e["multi_coded_lbd"] > 0) == (e["multi_no_coded_lbd"] > 0)
    ax.bar(xv + 0.2, e["multi_no_coded_lbd"], bottom=np.where(same_sign, e["multi_coded_lbd"], 0),
           width=0.38, color=MULTI, lw=0, label="Two or more, no coded DLB/PD")
    ax.axhline(0, color=INK2, lw=LW)
    ax.set_xticks(xv, list(e["row_label"]), rotation=50, ha="right", rotation_mode="anchor",
                  fontsize=5.5)
    ax.set_xlim(-0.7, len(e) - 0.3)
    ax.set_ylabel(f"Excess core features,\nAID+ women (O {M} E)")
    ax.legend(frameon=False, loc="upper right", borderaxespad=0.1)
    style(ax)
    ax.tick_params(axis="x", length=0)
    pos = ax.get_position()
    ax.set_position([0.075, pos.y0, 0.985 - 0.075, pos.height])
    letter(fig, ax, "b", X_LEFT)

    # c: first recorded AID diagnosis in years relative to the AD diagnosis (panel a's
    # rows). The workbook holds the AID-to-AD gap (positive, AID first); it is negated
    # here, so negative values are AIDs diagnosed before AD and the IQR runs -Q3 to -Q1
    gc = outer[2].subgridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.38)
    ax = fig.add_subplot(gc[0])
    tm = T.set_index("condition").reindex(t["condition"]).reset_index()
    shown = (tm["n_dated"] >= min_dated).to_numpy()
    info["c_not_shown"] = bool((~shown).any())
    ax.axvline(0, color=INK2, lw=LW, ls="--")
    ax.hlines(yv[shown], -tm["aid_to_ad_q3_yrs"][shown], -tm["aid_to_ad_q1_yrs"][shown],
              color=INK2, lw=LW)
    ax.scatter(-tm["aid_to_ad_median_yrs"][shown], yv[shown], s=12, color=INK, lw=0, zorder=3)
    ax.scatter(np.zeros((~shown).sum()), yv[~shown], marker="x", s=14, color=INK, lw=0.9,
               zorder=3)
    ax.set_yticks(yv, list(t["row_label"]))
    ax.set_ylim(-0.7, len(t) - 0.3)
    ax.set_xlabel("AID diagnosis, years relative to AD diagnosis")
    ax.text(0, 1.01, "AD diagnosis", transform=ax.get_xaxis_transform(), ha="center",
            va="bottom", fontsize=5, color=INK2)
    style(ax)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    # x-axis as a timeline: arrowhead at the right end of the bottom spine (later times)
    ax.plot(1, 0, marker=">", ms=3.5, mew=0, color=INK2, transform=ax.transAxes,
            clip_on=False, zorder=5)
    letter(fig, ax, "c", X_LEFT)

    # d: possible mimics, IPW odds ratio in women with the AID+ women of each set removed
    ax = fig.add_subplot(gc[1])
    order = [s for s in SET_TICK if s in set(D["set"])]
    for off, (y, col, mk, lab) in zip((-0.12, 0.12), D_OUTCOMES):
        Dd = D[D["outcome"] == y].set_index("set").loc[order]
        ax.axhline(Dd.loc["reference", "odds_ratio"], color=col, lw=0.5, alpha=0.35, zorder=0)
        xx = np.arange(len(Dd)) + off
        ax.vlines(xx, Dd["ci_low"], Dd["ci_high"], color=col, lw=LW, zorder=3)
        ax.scatter(xx, Dd["odds_ratio"], s=12, marker=mk, color=col, zorder=4, label=lab, lw=0)
    ax.axhline(1, color=INK2, lw=LW, ls="--")
    ax.set_yscale("log")
    ax.minorticks_off()
    ax.set_ylim(min(0.8, float(D["ci_low"].min()) * 0.92), float(D["ci_high"].max()) * 1.25)
    tk = log_ticks(*ax.get_ylim())
    ax.set_yticks(tk, [f"{v:g}" for v in tk])
    ax.set_xticks(range(len(order)), [SET_TICK[s] for s in order], fontsize=5.5)
    ax.set_xlim(-0.5, len(order) - 0.5)
    ax.set_xlabel("AID+ women with possible mimics removed")
    ax.set_ylabel(f"Odds ratio, AID+ vs AID{M} women (95% CI)")
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2,
              handletextpad=0.3, borderaxespad=0.1, columnspacing=1.0)
    style(ax)
    ax.tick_params(axis="x", length=0)
    letter(fig, ax, "d", ax.get_position().x0 - 0.075)
    save(fig, STEM)
    return info


# =============================================================================
# EXTENDED DATA FIGURE
# =============================================================================
def draw_extended(wb):
    d = wb["combinations"]
    rows = sorted(d["row_label"].unique(), key=ru.sort_key)
    obs = d.pivot(index="row_label", columns="combination", values="observed").loc[rows, COMBO_ORDER]
    sir = d.pivot(index="row_label", columns="combination", values="sir").loc[rows, COMBO_ORDER]
    Mx = np.log2(np.where((obs > 0) & (sir > 0), sir, np.nan))
    cmap = LinearSegmentedColormap.from_list("div", ["#184f95", "#86b6ef", "#f0efec", "#ec835a",
                                                     "#b8322f"])
    fig, ax = plt.subplots(figsize=(120 * MM, min(0.2 * len(rows) + 1.3, 208 * MM)),
                           layout="constrained")
    im = ax.pcolormesh(np.arange(len(COMBO_ORDER) + 1) - 0.5, np.arange(len(rows) + 1) - 0.5,
                       np.ma.masked_invalid(Mx), cmap=cmap, vmin=-3, vmax=3, edgecolors="white",
                       linewidth=0.4)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    for i in range(len(rows)):
        for j in range(len(COMBO_ORDER)):
            ax.text(j, i, ru.fmt_count(obs.iat[i, j]), ha="center", va="center", fontsize=5.5,
                    color=INK)
    ax.set_xticks(range(len(COMBO_ORDER)), [COMBO_TICK[c] for c in COMBO_ORDER], fontsize=5.5)
    ax.set_yticks(range(len(rows)), rows, fontsize=6)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, shrink=0.6, ticks=[-3, -2, -1, 0, 1, 2, 3])
    cbar.solids.set_rasterized(False)
    cbar.ax.set_yticklabels(["0.125", "0.25", "0.5", "1", "2", "4", "8"])
    cbar.set_label(f"SIR, AID+ vs AID{M} women (log scale)", fontsize=6)
    cbar.ax.tick_params(labelsize=6)
    save(fig, EXTENDED_STEM)
    carriers = d.drop_duplicates("row_label").set_index("row_label").loc[rows, "n_female"]
    return {"carriers": "; ".join(f"{r}, {ru.fmt_count(n)}" for r, n in carriers.items())}


# =============================================================================
# CAPTIONS
# =============================================================================
def figure_caption(cap, wb, info):
    c = cap["figure"]
    P = wb["a_pooled"].iloc[0]
    A = wb["a_conditions"]
    n = {"n_female": ru.fmt_count(parameter(wb, "n_aid_pos_female")),
         "n_conditions": ru.fmt_count(parameter(wb, "n_conditions_shown")),
         "n_other": ru.fmt_count(parameter(wb, "n_other_conditions")),
         "n_other_female": ru.fmt_count(parameter(wb, "n_females_other_conditions")),
         "min_group": int(parameter(wb, "min_group_n")),
         "min_events": int(parameter(wb, "min_condition_events")),
         "k": ru.fmt_count(P["k"]), "size_tiers": info["size_tiers"]}
    D = wb["d_possible_mimics_ipw"].drop_duplicates("set").set_index("set")
    sets = [s for s in SET_CAPTION if s in D.index]
    n["d_removed"] = "; ".join(
        f"{SET_CAPTION[s]}, {ru.fmt_count(D.loc[s, 'aid_pos_female_removed'])}" for s in sets)
    by = {w: [SET_CAPTION[s] for s in sets if D.loc[s, "weights"] == w] for w in ("reuse", "refit")}
    unknown = set(D.loc[sets, "weights"]) - set(by)
    if unknown:
        raise KeyError(f"d_possible_mimics_ipw: weights {sorted(unknown)}")
    n["d_weights"] = "; ".join(f"{word} for {ru.join_and(by[w])}" for w, word in
                               (("reuse", "stored"), ("refit", "refitted")) if by[w])
    out = [ru.fill(c["intro"], n), c["standardization"], ru.fill(c["a"], n)]
    if (A["estimated"] != 1).any():
        out.append(ru.fill(c["a_not_estimated"], n))
    out += [c["a_groupings"], ru.fill(c["b"], n), c["c"]]
    if info["c_not_shown"]:
        out.append(ru.fill(c["c_not_shown"], n))
    out += [ru.fill(c["d"], n), c["error_bars"], c["abbreviations"]]
    return out


def extended_caption(cap, wb, info):
    c = cap["extended"]
    return [ru.fill(c["body"], {"carriers": info["carriers"],
                                "n_female": ru.fmt_count(parameter(wb, "n_aid_pos_female"))}),
            c["color"], c["abbreviations"]]


def main():
    cap = ru.load_captions(CAPTIONS_PATH, __file__)
    print(f"  Titles and captions: {'written' if cap else 'none (no caption file set)'}")
    wb = workbook(WORKBOOK)
    info = draw_figure(wb)
    wbx = workbook(EXTENDED_WORKBOOK)
    infox = draw_extended(wbx)
    if cap:
        ru.write_caption(os.path.join(FIGURES_DIR, STEM + "_caption.docx"),
                         cap["figure"]["title"], figure_caption(cap, wb, info))
        ru.write_caption(os.path.join(FIGURES_DIR, EXTENDED_STEM + "_caption.docx"),
                         cap["extended"]["title"], extended_caption(cap, wbx, infox))


if __name__ == "__main__":
    main()