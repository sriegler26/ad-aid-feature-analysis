"""
dlb_feature_specificity_figure.py
=================================
Draws the DLB feature specificity figure (.png, .pdf) from the source-data
workbook written by dlb_core_analysis.py. Fits no model and computes no
statistic: every plotted value (odds ratios, CIs, P and Q values, the sex-test
ratio and Q, discriminant coefficients, survival curves, bootstrap bands, the
coded-DLB reference curve) is read from the workbook. Numbers printed on the
figure are formatted by render_utils.

Layout:
- a: forest of the AID+ versus AID- odds ratio of the tested features, three
  series (female, all patients; female, coded DLB excluded; male, all
  patients). Rows: every tested feature with an estimate in at least one
  series (the others are named in the caption); the core features first,
  then the supportive features, then the cognitive-profile features
  (visuospatial and executive impairment, part of the essential feature,
  dementia, in McKeith 2017), as the workbook column mckeith_class gives
  them, each group under its own heading and
  ordered by the odds ratio in women (descending). Significance symbols from Q;
  an estimate without a Q (below the per-group floor) is drawn at reduced
  opacity, with a dagger when its unadjusted P < 0.05. The sex test (ratio
  female / male and its Q) in the left gutter for features with Q below
  SEX_Q_THRESH. The panel height follows the number of rows at a fixed row
  pitch (A_ROW_MM), so the figure grows with the feature set; above
  MAX_FIG_H_MM the pitch shrinks to fit (printed).
- b: discriminant coefficient (coded DLB versus AD alone, multivariable) against
  the odds ratio in women, labels placed by a collision solver; core features
  filled, supportive features half-filled, cognitive-profile features open; a panel note when the discriminant model
  was not fitted.
- c: share of women with score >= x, AID+ and AID-, with bootstrap bands; left
  all patients (with the coded-DLB reference curve, both sexes), right coded
  DLB excluded. The x-axis runs from 0 to score_max, with ticks at multiples
  of one core feature's points (score_core_points), so each tick is the
  lowest score with that many core features; both are read from the workbook.
  Thin gray vertical lines join the AID- and AID+ curves at 1, 2, 3 and 4 core
  features (scores core_points x k), in both panels, each labelled with the
  difference AID+ minus AID- in percentage points (whole number, signed),
  taken from the two plotted curve values.

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
import matplotlib.ticker as mt
from matplotlib.lines import Line2D

import render_utils as ru

warnings.filterwarnings("ignore")

# =============================================================================
# PATHS (relative to the data folder the script is run from)
# =============================================================================
SOURCE_DATA_DIR = ""  # folder of the source-data workbooks (.xlsx) written by dlb_core_analysis.py
FIGURES_DIR = ""  # folder for the figure files written here
CAPTIONS_PATH = ""  # optional JSON with the title and caption; blank = no caption file

FIGURE_FILE = "figure_dlb_feature_specificity"

# =============================================================================
# DRAWING SETTINGS
# =============================================================================
MM = 1 / 25.4
MAX_W = 180 * MM
DPI = 300

# Vertical layout in mm
A_ROW_MM = 7.3            # panel a, one feature row
A_HEADER_ROWS = 0.5       # panel a, each group heading, in rows
BOTTOM_ROW_MM = 41.1      # panels b and c
TOP_MM = 11.9             # above panel a (legend, tag)
BOTTOM_MM = 16.15         # below panels b and c (axis labels)
GAP_MM = 20.6             # between panel a and panels b / c
MAX_FIG_H_MM = 240.0      # tallest figure allowed
B_C_WIDTHS = (1.25, 1.45)  # panel b : panel c

C_F_POS = "#C0392B"   # female, all patients: filled red diamond
C_AMBER = "#D4813A"   # female, coded DLB excluded: open amber diamond
C_M_POS = "#1A5276"   # male, all patients: open blue square
C_TEXT = "#333333"
C_HET = "#6E6E6E"     # gutter annotation: neutral gray
C_CD_POS = "#C0392B"  # panel c: AID+
C_CD_NEG = "#8A8A8A"  # panel c: AID-
C_GAP = C_HET         # panel c: gap lines between the AID+ and AID- curves
GAP_LW = 0.5
GAP_FS = 4.9          # panel c: gap labels (AID+ minus AID-, percentage points)

M = ru.MINUS
SIG_LEVELS = [(0.001, "***"), (0.01, "**"), (0.05, "*")]   # on Q
SIG_NS = ""
SIG_UNCORR = "†"   # unadjusted P < 0.05 where no Q exists
SIG_ALPHA = 0.05        # Q below this: full opacity
SEX_Q_THRESH = 0.05     # the sex test is shown in the gutter for Q below this

SERIES = [("female_all", "Women, all patients"),
          ("female_dlb_excluded", "Women, coded DLB excluded"),
          ("male_all", "Men, all patients")]
CD_ARMS = [("all", "Women, all patients"),
           ("dlb_excluded", "Women, coded DLB excluded")]
# McKeith class of each tested feature (workbook column mckeith_class), in panel
# order; visuospatial and executive impairment are the cognitive profile (McKeith
# 2017: under the essential feature, dementia, not among the supportive features)
CLASSES = [("core", "Core features"), ("supportive", "Supportive features"),
           ("cognitive_profile", "Cognitive profile")]


def check_classes(df, sheet):
    """Stop on an unknown class, or on a workbook written before the cognitive
    profile was its own class (no cognitive_profile row)."""
    unknown = set(df["mckeith_class"]) - {k for k, _ in CLASSES}
    if unknown:
        raise KeyError(f"panel {sheet}: unknown McKeith class {sorted(unknown)}")
    if not (df["mckeith_class"] == "cognitive_profile").any():
        raise KeyError(f"sheet {sheet}: no cognitive_profile row; the workbook predates the "
                       f"cognitive-profile class (rerun dlb_core_analysis.py or use the "
                       f"patched source data)")

SHORT_LABEL = {
    "fluct": "Fluct. cog.",
    "vis_hall": "Vis. hall.",
    "park_all_mckeith": "Parkinsonism",
    "rbd_any": "RBD",
    "ortho": "Ortho.",
    "vis_seq": "Visuospatial",
    "aud_hall": "Aud. hall.",
    "incont": "Incont.",
    "hyper": "Hypersomnia",
    "exec_seq": "Exec.",
    "syncope": "Syncope",
    "constipation": "Constip.",
    "pi_mckeith": "Post. instab.",
    "hyposmia": "Hyposmia",
    "systematized_delusions": "Syst. delus.",
    "neuro_sens": "Neurol. sens.",
    "transient_unresponsiveness": "Trans. unresp.",
}

B_LEADER_MIN_PTS = 8.0
B_LABEL_RADII = (9.0, 13.0, 18.0, 24.0, 30.0, 36.0)
CD_LOG_X = False
# Panel c x ticks: multiples of score_core_points up to score_max (workbook parameters)

# =============================================================================
# STYLE
# =============================================================================
try:
    import scienceplots  # noqa: F401
    plt.style.use(["nature"])
except (ImportError, OSError):
    pass

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica Neue", "Helvetica", "DejaVu Sans"],
    "font.size": 6, "axes.titlesize": 6, "axes.labelsize": 6.5,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 5.5,
    "axes.linewidth": 1.0, "xtick.major.width": 1.0, "ytick.major.width": 1.0,
    "xtick.major.size": 2.0, "ytick.major.size": 2.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.0, "patch.linewidth": 1.0,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
})

TXT_Q = r"$\mathit{Q}$"


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


def isnan(x):
    return x is None or (isinstance(x, float) and np.isnan(x))


def gap_marks(wb, core_pts, smax):
    """Panel c gap-line scores: the lowest score with 1, 2, ... core features
    (core_pts x k). k runs to score_n_core_features when the workbook has it,
    otherwise to the largest multiple of core_pts within score_max."""
    p = wb["parameters"].set_index("parameter")["value"]
    if "score_n_core_features" in p.index:
        n_core = int(p["score_n_core_features"])
    else:
        n_core = smax // core_pts
    return [core_pts * k for k in range(1, n_core + 1) if core_pts * k <= smax]


# =============================================================================
# HELPERS
# =============================================================================
def save_fig(fig, name):
    for ext in ("pdf", "png"):
        ru.save_figure(fig, os.path.join(FIGURES_DIR, f"{name}.{ext}"), dpi=DPI,
                       facecolor="white")
    plt.close(fig)


def strip_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def clip(v, lo, hi):
    return min(max(v, lo), hi)


# -- Panel b label placement ---------------------------------------------------
def _seg_rect_hit(p0, p1, rect, pad=0.0):
    """True if segment p0-p1 intersects an axis-aligned rect (x0,y0,x1,y1)."""
    x0, y0, x1, y1 = rect
    x0 -= pad; y0 -= pad; x1 += pad; y1 += pad
    # Liang-Barsky
    dx = p1[0] - p0[0]; dy = p1[1] - p0[1]
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, p0[0] - x0), (dx, x1 - p0[0]),
                 (-dy, p0[1] - y0), (dy, y1 - p0[1])):
        if p == 0:
            if q < 0:
                return False
        else:
            r = q / p
            if p < 0:
                if r > t1:
                    return False
                t0 = max(t0, r)
            else:
                if r < t0:
                    return False
                t1 = min(t1, r)
    return t0 <= t1


def _segs_cross(a0, a1, b0, b1):
    def o(p, q, r):
        return ((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))
    d1, d2 = o(a0, a1, b0), o(a0, a1, b1)
    d3, d4 = o(b0, b1, a0), o(b0, b1, a1)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def _overlap(r1, r2, pad=0.0):
    return not (r1[2] + pad <= r2[0] or r2[2] + pad <= r1[0] or
                r1[3] + pad <= r2[1] or r2[3] + pad <= r1[1])


def place_leader_labels(ax, xs, ys, texts, *, fontsize=4.9, color="#333333",
                        leader_color="#8A8A8A", leader_lw=0.35,
                        marker_pad=3.0, label_pad=1.5,
                        radii=(9.0, 13.0, 18.0, 24.0, 30.0),
                        n_angles=24, zorder=6, seed_angle=25.0, min_leader_pts=8.0,
                        cluster_tol=7.0, marker_diam=None):
    """Place each label near its own point without collisions (display space:
    candidate positions on rings around each point, scored against the other
    points, the labels and leaders already placed and the axes boundary).
    Points whose markers overlap share one leader carrying both names.
    Returns (results, merged labels, the highest placement cost: 500 or more
    means a label could not avoid a collision)."""
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    dpi = fig.dpi
    p2px = dpi / 72.0

    pts = np.array([ax.transData.transform((x, y)) for x, y in zip(xs, ys)])
    ax_box = ax.get_window_extent()
    bounds = (ax_box.x0, ax_box.y0, ax_box.x1, ax_box.y1)

    sizes = []
    for t in texts:
        tmp = ax.text(0, 0, t, fontsize=fontsize, transform=None)
        bb = tmp.get_window_extent(renderer=r)
        sizes.append((bb.width, bb.height))
        tmp.remove()

    d = np.hypot(pts[:, None, 0] - pts[None, :, 0],
                 pts[:, None, 1] - pts[None, :, 1])
    np.fill_diagonal(d, np.inf)
    crowd = (1.0 / np.maximum(d, 1.0)).sum(axis=1)
    order = sorted(range(len(texts)), key=lambda i: (-crowd[i], pts[i, 0], pts[i, 1]))

    tol = cluster_tol * p2px
    parent = list(range(len(texts)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            if d[i, j] < tol:
                parent[find(i)] = find(j)
    clusters = {}
    for i in range(len(texts)):
        clusters.setdefault(find(i), []).append(i)

    angles = np.deg2rad(seed_angle + np.arange(n_angles) * 360.0 / n_angles)
    placed_boxes, placed_leaders = [], []
    results = [None] * len(texts)
    worst = 0.0

    def block_cost(boxes, leaders, members):
        c = 0.0
        for box, (p0, tip) in zip(boxes, leaders):
            if (box[0] < bounds[0] or box[2] > bounds[2] or
                    box[1] < bounds[1] or box[3] > bounds[3]):
                c += 1500.0
            c += 0.6 * np.hypot(tip[0] - p0[0], tip[1] - p0[1]) / p2px
            for j in range(len(texts)):
                mk = (pts[j, 0] - marker_pad * p2px, pts[j, 1] - marker_pad * p2px,
                      pts[j, 0] + marker_pad * p2px, pts[j, 1] + marker_pad * p2px)
                if _overlap(box, mk):
                    c += 900.0
                if j not in members and _seg_rect_hit(p0, tip, mk):
                    c += 800.0
        return c

    merged_text = {}
    skip = set()
    if marker_diam is not None:
        for key, members in list(clusters.items()):
            if len(members) < 2:
                continue
            sub = [i for i in members]
            if max(d[i, j] for i in sub for j in sub if i != j) < marker_diam * p2px:
                members_sorted = sorted(sub, key=lambda i: (-pts[i, 1], pts[i, 0]))
                anchor = members_sorted[0]
                merged_text[anchor] = "\n".join(texts[i] for i in members_sorted)
                skip.update(members_sorted[1:])
                clusters[key] = [anchor]

    for i, t in merged_text.items():
        tmp = ax.text(0, 0, t, fontsize=fontsize, transform=None,
                      linespacing=1.15)
        bb = tmp.get_window_extent(renderer=r)
        sizes[i] = (bb.width, bb.height)
        tmp.remove()

    label_of = {i: merged_text.get(i, texts[i]) for i in range(len(texts))}

    for members in clusters.values():
        if len(members) < 2:
            continue
        members = sorted(members, key=lambda i: (-pts[i, 1], pts[i, 0]))
        cx = float(np.mean([pts[i, 0] for i in members]))
        cy = float(np.mean([pts[i, 1] for i in members]))
        bh = max(sizes[i][1] for i in members)
        step = bh + label_pad * p2px
        best_block = None
        for side in (1, -1):
            for rad in radii:
                if rad < cluster_tol + 4.0:
                    continue
                boxes, leaders, offs = [], [], []
                for s_i, i in enumerate(members):
                    ly = cy + ((len(members) - 1) / 2.0 - s_i) * step
                    lx = cx + side * rad * p2px
                    ha = "left" if side > 0 else "right"
                    bx0 = lx if side > 0 else lx - sizes[i][0]
                    box = (bx0, ly - sizes[i][1] / 2.0,
                           bx0 + sizes[i][0], ly + sizes[i][1] / 2.0)
                    tip = (lx, ly)
                    boxes.append(box)
                    leaders.append(((pts[i, 0], pts[i, 1]), tip))
                    offs.append((lx - pts[i, 0], ly - pts[i, 1], ha))
                c = block_cost(boxes, leaders, set(members))
                if best_block is None or c < best_block[0]:
                    best_block = (c, boxes, leaders, offs)
        cst, boxes, leaders, offs = best_block
        worst = max(worst, cst)
        for s_i, i in enumerate(members):
            placed_boxes.append(boxes[s_i])
            placed_leaders.append(leaders[s_i])
            dx, dy, ha = offs[s_i]
            results[i] = (dx / p2px, dy / p2px, ha)

    for i in order:
        if results[i] is not None or i in skip:
            continue
        px, py = pts[i]
        w, h = sizes[i]
        best = None
        for rad in radii:
            for a in angles:
                dx = np.cos(a) * rad * p2px
                dy = np.sin(a) * rad * p2px
                ha = "left" if dx > 2 else ("right" if dx < -2 else "center")
                ax_ = {"left": 0.0, "center": -0.5, "right": -1.0}[ha]
                bx0 = px + dx + ax_ * w
                by0 = py + dy - h / 2.0
                box = (bx0, by0, bx0 + w, by0 + h)

                cost = rad * 0.6
                cost += 3.0 * abs(np.sin(a))

                if (box[0] < bounds[0] or box[2] > bounds[2] or
                        box[1] < bounds[1] or box[3] > bounds[3]):
                    cost += 1500.0

                for j in range(len(texts)):
                    mk = (pts[j, 0] - marker_pad * p2px, pts[j, 1] - marker_pad * p2px,
                          pts[j, 0] + marker_pad * p2px, pts[j, 1] + marker_pad * p2px)
                    if _overlap(box, mk):
                        cost += 900.0
                for b in placed_boxes:
                    if _overlap(box, b, pad=label_pad * p2px):
                        cost += 900.0

                tip = (bx0 if dx > 0 else (box[2] if dx < 0 else px),
                       py + dy)
                for j in range(len(texts)):
                    if j == i:
                        continue
                    mk = (pts[j, 0] - marker_pad * p2px, pts[j, 1] - marker_pad * p2px,
                          pts[j, 0] + marker_pad * p2px, pts[j, 1] + marker_pad * p2px)
                    if _seg_rect_hit((px, py), tip, mk):
                        cost += 800.0
                for b in placed_boxes:
                    if _seg_rect_hit((px, py), tip, b):
                        cost += 600.0
                for L in placed_leaders:
                    if _segs_cross((px, py), tip, L[0], L[1]):
                        cost += 500.0

                if best is None or cost < best[0]:
                    best = (cost, dx, dy, ha, box, tip)

        cst, dx, dy, ha, box, tip = best
        worst = max(worst, cst)
        placed_boxes.append(box)
        placed_leaders.append(((px, py), tip))
        results[i] = (dx / p2px, dy / p2px, ha)

    for i in range(len(texts)):
        if i in skip:
            continue
        dx, dy, ha = results[i]
        show_leader = np.hypot(dx, dy) > min_leader_pts
        arrow = dict(arrowstyle="-", lw=leader_lw, color=leader_color,
                     shrinkA=0.5, shrinkB=1.4,
                     connectionstyle="arc3,rad=0.0") if show_leader else None
        ax.annotate(label_of[i], (xs[i], ys[i]), textcoords="offset points",
                    xytext=(dx, dy), fontsize=fontsize, color=color,
                    ha=ha, va="center", linespacing=1.15,
                    arrowprops=arrow, zorder=zorder)
    return results, merged_text, worst


def panel_note(ax, text):
    """A panel with nothing to draw: the reason, centered, no axes."""
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=6, color=C_HET,
            transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


# =============================================================================
# DATA
# =============================================================================
def panel_a_rows(wb):
    """Panel a rows: core, supportive and cognitive-profile features, each
    ordered by the odds ratio in women (descending); a feature with no estimate
    in any series is left out and returned by name for the caption."""
    a = wb["a"].copy()
    check_classes(a, "a")
    keys = {"female_all": "f", "female_dlb_excluded": "fx", "male_all": "m"}
    out = pd.DataFrame({"col": a["feature"], "label": a["row_label"],
                        "mckeith_class": a["mckeith_class"]})
    for s, k in keys.items():
        fitted = a[f"status_{s}"] == "fitted"
        for src, dst in (("odds_ratio", "or"), ("ci_low", "lo"), ("ci_high", "hi"),
                         ("p", "p"), ("q", "q")):
            out[f"{dst}_{k}"] = a[f"{src}_{s}"].where(fitted).astype(float)
        out[f"floor_{k}"] = fitted & (a[f"fdr_role_{s}"] != "tested")
    tested = a["sex_status"] == "tested"
    out["sex_ratio"] = a["sex_ratio"].where(tested).astype(float)
    out["sex_q"] = a["sex_q"].where(tested).astype(float)
    shown = out[["or_f", "or_fx", "or_m"]].notna().any(axis=1)
    dropped = list(out.loc[~shown, "label"])
    out = out[shown].copy()
    out["_class"] = out["mckeith_class"].map({k: i for i, (k, _) in enumerate(CLASSES)})
    out = out.sort_values(["_class", "or_f"], ascending=[True, False], na_position="last",
                          kind="mergesort").drop(columns="_class")
    return out.reset_index(drop=True), dropped


def panel_b_rows(wb):
    b = wb["b"]
    missing = sorted(set(b["feature"]) - set(SHORT_LABEL))
    if missing:
        raise KeyError(f"no short panel b label for: {', '.join(missing)}")
    check_classes(b, "b")
    f = b[(b["status"] == "fitted") & b["coef"].notna() & b["odds_ratio_female_all"].notna()]
    return pd.DataFrame({"col": f["feature"].values, "label": f["row_label"].values,
                         "short": [SHORT_LABEL[c] for c in f["feature"]],
                         "cls": f["mckeith_class"].values,
                         "beta": f["coef"].astype(float).values,
                         "or_f": f["odds_ratio_female_all"].astype(float).values}), len(b)


def panel_c_data(wb):
    cur, summ = wb["c_curves"], wb["c_summary"].set_index("arm")
    cd = {}
    for arm, lbl in CD_ARMS:
        d = cur[cur["arm"] == arm].sort_values("score")
        tested = arm in summ.index and summ.loc[arm, "status"] == "tested" and len(d)
        cd[arm] = dict(label=lbl, ok=bool(tested), grid=d["score"].to_numpy(float),
                       s_pos=d["share_aid_pos"].to_numpy(float),
                       s_neg=d["share_aid_neg"].to_numpy(float),
                       lo_p=d["ci_low_aid_pos"].to_numpy(float),
                       hi_p=d["ci_high_aid_pos"].to_numpy(float),
                       lo_n=d["ci_low_aid_neg"].to_numpy(float),
                       hi_n=d["ci_high_aid_neg"].to_numpy(float),
                       reason=None if tested else (summ.loc[arm, "reason"] if arm in summ.index
                                                   else "no patients"))
    ref = wb["c_reference"].sort_values("score")
    s_dlb = (ref["share_coded_dlb"].to_numpy(float), ref["score"].to_numpy(float)) \
        if int(parameter(wb, "coded_dlb_reference_drawn")) == 1 and len(ref) else None
    return cd, s_dlb


def figure_height(n_rows, n_headers):
    """Figure height (mm) and panel a row pitch (mm) for n_rows feature rows and
    n_headers group headings; the pitch shrinks only above MAX_FIG_H_MM."""
    fixed = TOP_MM + GAP_MM + BOTTOM_ROW_MM + BOTTOM_MM
    units = n_rows + A_HEADER_ROWS * n_headers
    pitch = A_ROW_MM
    if fixed + units * pitch > MAX_FIG_H_MM:
        pitch = (MAX_FIG_H_MM - fixed) / units
        print(f"  Panel a: {n_rows} rows exceed {MAX_FIG_H_MM:g} mm at {A_ROW_MM:g} mm per row; "
              f"row pitch reduced to {pitch:.2f} mm")
    return fixed + units * pitch, pitch


# =============================================================================
# FIGURE
# =============================================================================
def draw(wb):
    df_A, dropped = panel_a_rows(wb)
    df_B, n_b = panel_b_rows(wb)
    CD, S_DLB = panel_c_data(wb)
    SMAX = int(parameter(wb, "score_max"))
    CORE_PTS = int(parameter(wb, "score_core_points"))
    if CORE_PTS < 1:
        raise ValueError(f"parameters: score_core_points {CORE_PTS} must be at least 1")
    C_TICKS = list(range(0, SMAX + 1, CORE_PTS))     # lowest score with 0, 1, 2, ... core features
    print(f"  Panel c: x-axis 0-{SMAX}, ticks at {', '.join(str(t) for t in C_TICKS)} "
          f"(multiples of {CORE_PTS}, one core feature)")
    C_MARKS = gap_marks(wb, CORE_PTS, SMAX)
    print(f"  Panel c: AID+ / AID{M} gap lines at {', '.join(str(m) for m in C_MARKS)} "
          f"(1-{len(C_MARKS)} core features)")
    n_feats = len(df_A)
    groups = [(k, lbl) for k, lbl in CLASSES if (df_A["mckeith_class"] == k).any()]

    print("  Panel a order (core, supportive, cognitive profile; odds ratio in women "
          "descending): "
          + ", ".join(df_A["label"]))
    if dropped:
        print("  Panel a: not shown (no estimate in any series): " + ", ".join(dropped))
    n_gutter = int((df_A["sex_q"] < SEX_Q_THRESH).sum())
    print(f"  Panel a gutter: sex test shown for {n_gutter} of {n_feats} features "
          f"(Q < {SEX_Q_THRESH:g})")

    fig_h_mm, pitch_mm = figure_height(n_feats, len(groups))
    a_mm = fig_h_mm - TOP_MM - GAP_MM - BOTTOM_ROW_MM - BOTTOM_MM
    print(f"  Figure height {fig_h_mm:.1f} mm; panel a {a_mm:.1f} mm "
          f"({pitch_mm:.2f} mm per row)")
    fig = plt.figure(figsize=(MAX_W, fig_h_mm * MM))
    mean_h = (a_mm + BOTTOM_ROW_MM) / 2
    gs_main = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[a_mm, BOTTOM_ROW_MM],
                                left=0.225, right=0.985, top=1 - TOP_MM / fig_h_mm,
                                bottom=BOTTOM_MM / fig_h_mm, hspace=GAP_MM / mean_h)
    ax_A = fig.add_subplot(gs_main[0])
    gs_bot = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_main[1],
                                              width_ratios=list(B_C_WIDTHS),
                                              wspace=0.26)
    ax_B = fig.add_subplot(gs_bot[0])
    gs_c = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_bot[1],
                                            wspace=0.10)
    ax_C1 = fig.add_subplot(gs_c[0])
    ax_C2 = fig.add_subplot(gs_c[1], sharey=ax_C1)

    # ---- Panel a ----
    X_MIN_SPAN = (0.5, 2.6)
    X_CAP = (0.4, 3.0)
    X_PAD = 1.04
    X_TICK_POOL = [0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]
    SPACING = 3.9
    Y_F, Y_FX, Y_M = 0.80, 0.0, -0.80
    SIG_FS = 5.6
    SIG_DX = 3.0

    _lims = pd.concat([df_A[c] for c in ("lo_f", "hi_f", "lo_fx", "hi_fx",
                                         "lo_m", "hi_m")]).dropna()
    _lims = _lims[_lims > 0]
    if len(_lims):
        X_LEFT = max(min(X_MIN_SPAN[0], float(_lims.min()) / X_PAD), X_CAP[0])
        X_RIGHT = min(max(X_MIN_SPAN[1], float(_lims.max()) * X_PAD), X_CAP[1])
    else:
        X_LEFT, X_RIGHT = X_MIN_SPAN
    X_TICKS = [t for t in X_TICK_POOL if X_LEFT <= t <= X_RIGHT]
    _n_lo = int((df_A[["lo_f", "lo_fx", "lo_m"]] < X_LEFT).sum().sum())
    _n_hi = int((df_A[["hi_f", "hi_fx", "hi_m"]] > X_RIGHT).sum().sum())
    if _n_lo or _n_hi:
        print(f"  Panel a: intervals truncated at the axis: {_n_lo} lower, {_n_hi} upper "
              "(drawn with arrowheads)")

    GUT_X = -0.02
    GUT_DY_LBL = 0.42
    GUT_DY_HET = -0.72
    GUT_FS_LBL = 6.5
    GUT_FS_HET = 4.9
    GUT_FS_HEAD = 6.5

    def sigsym(q, p):
        if not isnan(q):
            for thr, sym in SIG_LEVELS:
                if q < thr:
                    return sym
            return SIG_NS
        if not isnan(p) and p < 0.05:
            return SIG_UNCORR
        return ""

    def sexfmt(r):
        """Gutter text: the ratio female / male and its Q, for Q below the threshold."""
        if isnan(r["sex_q"]) or isnan(r["sex_ratio"]) or r["sex_q"] >= SEX_Q_THRESH:
            return None
        q = ru.fmt_p(r["sex_q"])
        q_txt = f" {q[0]} {q[1:]}" if q.startswith("<") else f" = {q}"
        return f"ratio {ru.fmt_ratio(r['sex_ratio'])},  {TXT_Q}{q_txt}"

    def arrow_low(y, lo, color, alpha, ms, zorder):
        """Arrowhead at the left axis limit for a CI that extends beyond it."""
        if not np.isnan(lo) and lo < X_LEFT:
            ax_A.plot(X_LEFT, y, marker="<", ms=ms, color=color, alpha=alpha, zorder=zorder,
                      clip_on=False)

    def draw_sig(y, hi, sym, color, alpha):
        if not sym:
            return
        x = clip(hi, X_LEFT, X_RIGHT) if not np.isnan(hi) else X_RIGHT
        dx = SIG_DX + (3.0 if (not np.isnan(hi) and hi > X_RIGHT) else 0.0)
        ax_A.annotate(sym, (x, y), textcoords="offset points", xytext=(dx, -0.4),
                      ha="left", va="center", fontsize=SIG_FS, color=color,
                      alpha=alpha, annotation_clip=False, zorder=6)

    # y positions, top to bottom: a heading (half a row) above each group
    y_of, heads, y = {}, [], 0.0
    for k, lbl in groups:
        heads.append((y, lbl))                 # top of the heading band
        y -= A_HEADER_ROWS * SPACING
        for i in df_A.index[df_A["mckeith_class"] == k]:
            y -= SPACING
            y_of[i] = y + SPACING / 2
    y_bottom = y

    for i, r in df_A.iterrows():
        y_c = y_of[i]
        y_f, y_fx, y_m = y_c + Y_F, y_c + Y_FX, y_c + Y_M

        # female, all patients (solid line, filled diamond)
        if not np.isnan(r["or_f"]):
            al_f = 1.0 if (not np.isnan(r["q_f"]) and r["q_f"] < SIG_ALPHA) else 0.38
            if not np.isnan(r["lo_f"]):
                ax_A.plot([clip(r["lo_f"], X_LEFT, X_RIGHT), clip(r["hi_f"], X_LEFT, X_RIGHT)],
                          [y_f, y_f], color=C_F_POS, lw=1.1, alpha=al_f,
                          solid_capstyle="round", zorder=3)
                if r["hi_f"] > X_RIGHT:
                    ax_A.plot(X_RIGHT, y_f, marker=">", ms=2.6, color=C_F_POS,
                              alpha=al_f, zorder=3, clip_on=False)
                arrow_low(y_f, r["lo_f"], C_F_POS, al_f, 2.6, 3)
            ax_A.scatter(r["or_f"], y_f, marker="D", s=16, color=C_F_POS,
                         alpha=al_f, zorder=5, linewidths=0)
            draw_sig(y_f, r["hi_f"], sigsym(r["q_f"], r["p_f"]), C_F_POS, al_f)

        # female, coded DLB excluded (dashed line, open amber diamond)
        if not np.isnan(r["or_fx"]):
            al_fx = 1.0 if (not np.isnan(r["q_fx"]) and r["q_fx"] < SIG_ALPHA) else 0.38
            ax_A.plot([clip(r["lo_fx"], X_LEFT, X_RIGHT), clip(r["hi_fx"], X_LEFT, X_RIGHT)],
                      [y_fx, y_fx], color=C_AMBER, lw=0.9, ls="--", alpha=al_fx, zorder=2)
            if r["hi_fx"] > X_RIGHT:
                ax_A.plot(X_RIGHT, y_fx, marker=">", ms=2.4, color=C_AMBER,
                          alpha=al_fx, zorder=2, clip_on=False)
            arrow_low(y_fx, r["lo_fx"], C_AMBER, al_fx, 2.4, 2)
            ax_A.scatter(r["or_fx"], y_fx, marker="D", s=13, facecolor="white",
                         edgecolor=C_AMBER, linewidths=0.9, alpha=al_fx, zorder=4)
            draw_sig(y_fx, r["hi_fx"], sigsym(r["q_fx"], r["p_fx"]), C_AMBER, al_fx)

        # male, all patients (open blue square)
        if not np.isnan(r["or_m"]):
            al_m = 0.9 if (not np.isnan(r["q_m"]) and r["q_m"] < SIG_ALPHA) else 0.38
            ax_A.plot([clip(r["lo_m"], X_LEFT, X_RIGHT), clip(r["hi_m"], X_LEFT, X_RIGHT)],
                      [y_m, y_m], color=C_M_POS, lw=1.0, alpha=al_m, zorder=3)
            if r["hi_m"] > X_RIGHT:
                ax_A.plot(X_RIGHT, y_m, marker=">", ms=2.4, color=C_M_POS,
                          alpha=al_m, zorder=3, clip_on=False)
            arrow_low(y_m, r["lo_m"], C_M_POS, al_m, 2.4, 3)
            ax_A.scatter(r["or_m"], y_m, marker="s", s=13, facecolor="white",
                         edgecolor=C_M_POS, linewidths=1.0, alpha=al_m, zorder=5)
            draw_sig(y_m, r["hi_m"], sigsym(r["q_m"], r["p_m"]), C_M_POS, al_m)

        # left gutter: feature name, and the sex test beneath it
        sex_txt = sexfmt(r)
        if sex_txt is None:
            ax_A.text(GUT_X, y_c, r["label"], va="center", ha="right",
                      transform=ax_A.get_yaxis_transform(), fontsize=GUT_FS_LBL)
        else:
            ax_A.text(GUT_X, y_c + GUT_DY_LBL, r["label"], va="center", ha="right",
                      transform=ax_A.get_yaxis_transform(), fontsize=GUT_FS_LBL)
            ax_A.text(GUT_X, y_c + GUT_DY_HET, sex_txt, va="center", ha="right",
                      transform=ax_A.get_yaxis_transform(), fontsize=GUT_FS_HET,
                      color=C_HET)

        # light rule below each row; none below a group's last row
        nxt = i + 1
        if nxt < n_feats and df_A.loc[nxt, "mckeith_class"] == r["mckeith_class"]:
            ax_A.axhline(y_c - SPACING / 2, color="#EBEBEB", lw=0.9, zorder=0)

    # group headings in the gutter, a darker rule above each group but the first
    for j, (y_top, lbl) in enumerate(heads):
        ax_A.text(GUT_X, y_top - A_HEADER_ROWS * SPACING / 2, lbl, va="center", ha="right",
                  fontweight="bold", transform=ax_A.get_yaxis_transform(),
                  fontsize=GUT_FS_HEAD, color=C_TEXT)
        if j:
            ax_A.axhline(y_top, color="#BDBDBD", lw=0.9, zorder=0)

    ax_A.axvline(1.0, color="#555555", lw=0.9, ls=":", zorder=1)
    ax_A.set_xscale("log")
    ax_A.set_xlim(X_LEFT, X_RIGHT)
    ax_A.set_xticks(X_TICKS)
    ax_A.xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: "{:g}".format(v)))
    ax_A.xaxis.set_minor_locator(mt.NullLocator())
    ax_A.set_yticks([])
    ax_A.set_xlabel(f"Odds ratio, AID+ versus AID{M}", fontsize=6.5)
    ax_A.set_ylim(y_bottom, 0.0)
    strip_axes(ax_A)

    leg_A = [
        Line2D([0], [0], color=C_F_POS, lw=1.1, marker="D", ms=3.8,
               markerfacecolor=C_F_POS, markeredgewidth=0, label=SERIES[0][1]),
        Line2D([0], [0], color=C_AMBER, lw=0.9, ls="--", marker="D", ms=3.4,
               markerfacecolor="white", markeredgecolor=C_AMBER, label=SERIES[1][1]),
        Line2D([0], [0], color=C_M_POS, lw=1.0, marker="s", ms=3.4,
               markerfacecolor="white", markeredgecolor=C_M_POS, label=SERIES[2][1]),
    ]
    ax_A.legend(handles=leg_A, loc="lower left", bbox_to_anchor=(0.0, 1.005),
                ncol=3, frameon=False, fontsize=5.6,
                handletextpad=0.45, handlelength=1.5, columnspacing=1.6,
                borderaxespad=0.0)

    # ---- Panel b ----
    merged = {}
    if len(df_B) >= 2:
        _b_lo_pre, _b_hi_pre = df_B["beta"].min(), df_B["beta"].max()
        _b_span_pre = (_b_hi_pre - _b_lo_pre) or 1.0
        B_MARKER_S = 20
        B_MARKER_DIAM = float(np.sqrt(B_MARKER_S))
        # filled, core; half-filled, supportive; open, cognitive profile
        cls = df_B["cls"].to_numpy()
        core, cog = cls == "core", cls == "cognitive_profile"
        sup = ~(core | cog)
        ax_B.scatter(df_B.loc[core, "beta"], df_B.loc[core, "or_f"], s=B_MARKER_S,
                     color=C_F_POS, zorder=5, edgecolor="white", linewidths=0.4)
        if sup.any():
            ax_B.plot(df_B.loc[sup, "beta"], df_B.loc[sup, "or_f"], ls="none", marker="o",
                      markersize=float(np.sqrt(B_MARKER_S - 2)), fillstyle="left",
                      markerfacecolor=C_F_POS, markerfacecoloralt="white",
                      markeredgecolor=C_F_POS, markeredgewidth=0.8, zorder=5)
        ax_B.scatter(df_B.loc[cog, "beta"], df_B.loc[cog, "or_f"], s=B_MARKER_S - 2,
                     facecolor="white", edgecolor=C_F_POS, zorder=5, linewidths=0.8)
        ax_B.set_xlim(_b_lo_pre - 0.36 * _b_span_pre, _b_hi_pre + 0.12 * _b_span_pre)
        ax_B.margins(y=0.16)
        ax_B.set_ylim(ax_B.get_ylim())
        _, merged, worst = place_leader_labels(
            ax_B, df_B["beta"].values, df_B["or_f"].values,
            list(df_B["short"]), fontsize=4.9, color=C_TEXT,
            leader_color=C_HET, radii=B_LABEL_RADII,
            min_leader_pts=B_LEADER_MIN_PTS,
            marker_diam=B_MARKER_DIAM)
        for t in merged.values():
            print("  Panel b: overlapping markers share one leader: " + " / ".join(t.split("\n")))
        print(f"  Panel b: {len(df_B)} features; highest label placement cost {worst:.0f}"
              + ("  (a label could not avoid a collision: check the figure)" if worst >= 500
                 else ""))
        ax_B.axhline(1.0, color="#555555", lw=0.8, ls=":", zorder=1)
        ax_B.set_xlabel("Discriminant β, DLB versus AD alone", fontsize=6.5)
        ax_B.set_ylabel(f"Odds ratio in women,\nAID+ vs AID{M}", fontsize=6.5)
        strip_axes(ax_B)
    else:
        panel_note(ax_B, "Discriminant model not fitted")
        ax_B.set_xlabel("Discriminant β, DLB versus AD alone", fontsize=6.5)
        print("  Panel b: discriminant model not fitted (panel note drawn)")

    # ---- Panel c ----
    def cd_x(v):
        return np.log1p(v) if CD_LOG_X else v

    def draw_cd(ax, arm, show_dlb_ref):
        r = CD[arm]
        ax.set_title(r["label"], fontsize=6, pad=2.5)
        if not r["ok"]:
            ax.text(0.5, 0.5, "Not drawn", ha="center", va="center", fontsize=6,
                    color=C_HET, transform=ax.transAxes)
        else:
            x_cd = cd_x(r["grid"])
            ax.fill_between(x_cd, 100 * r["lo_n"], 100 * r["hi_n"],
                            color=C_CD_NEG, alpha=0.18, lw=0, zorder=2)
            ax.fill_between(x_cd, 100 * r["lo_p"], 100 * r["hi_p"],
                            color=C_CD_POS, alpha=0.18, lw=0, zorder=2)
            if show_dlb_ref and S_DLB is not None:
                ax.plot(cd_x(S_DLB[1]), 100 * S_DLB[0], color=C_TEXT, lw=0.8, ls=":",
                        alpha=0.75, zorder=3)
            ax.plot(x_cd, 100 * r["s_neg"], color=C_CD_NEG, lw=1.1, zorder=4)
            ax.plot(x_cd, 100 * r["s_pos"], color=C_CD_POS, lw=1.3, zorder=5)
            # gap lines: AID- curve to AID+ curve at each core-feature mark, the
            # values read from the plotted arrays (linear between grid points, as drawn)
            for mk in C_MARKS:
                if not (r["grid"][0] <= mk <= r["grid"][-1]):
                    print(f"  Panel c ({arm}): score {mk} outside the curve range; no gap line")
                    continue
                y_n = 100 * float(np.interp(mk, r["grid"], r["s_neg"]))
                y_p = 100 * float(np.interp(mk, r["grid"], r["s_pos"]))
                if np.isnan(y_n) or np.isnan(y_p):
                    print(f"  Panel c ({arm}): score {mk} has no curve value; no gap line")
                    continue
                ax.plot([cd_x(mk)] * 2, [y_n, y_p], color=C_GAP, lw=GAP_LW,
                        solid_capstyle="butt", zorder=4.5)
                # label: AID+ minus AID-, percentage points, whole number, signed;
                # just above and right of the line's upper end (clear of the
                # falling curves)
                diff = ru.fmt_num(y_p - y_n, 0, plus=True)
                ax.annotate(diff, (cd_x(mk), max(y_n, y_p)), textcoords="offset points",
                            xytext=(1.5, 2.5), ha="left", va="bottom", fontsize=GAP_FS,
                            color=C_GAP, zorder=6)
                print(f"  Panel c ({arm}): score {mk}: AID+ {ru.fmt_pct(y_p)}, "
                      f"AID{M} {ru.fmt_pct(y_n)}, difference {diff} percentage points")
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_locator(mt.MultipleLocator(25))
        ax.set_xticks(cd_x(np.array(C_TICKS, dtype=float)))
        ax.set_xticklabels([str(t) for t in C_TICKS])
        ax.xaxis.set_minor_locator(mt.NullLocator())
        ax.set_xlim(0, cd_x(max(SMAX, 1)))
        strip_axes(ax)

    draw_cd(ax_C1, "all", show_dlb_ref=True)
    draw_cd(ax_C2, "dlb_excluded", show_dlb_ref=False)
    ax_C1.set_ylabel("Patients with\nscore ≥ x  (%)", fontsize=6.5, labelpad=2.0)
    ax_C2.tick_params(axis="y", labelleft=False)

    p1, p2 = ax_C1.get_position(), ax_C2.get_position()
    xm = (p1.x0 + p2.x1) / 2
    y = p1.y0 - 0.052 * 170 / fig_h_mm
    try:
        fig.canvas.draw()
        rend = fig.canvas.get_renderer()
        if len(df_B) >= 2:
            # aligned with panel b's x label
            bb = ax_B.xaxis.label.get_window_extent(renderer=rend)
            y = fig.transFigure.inverted().transform(bb)[0][1]
            va = "bottom"
        else:
            # panel b has no tick labels: placed below panel c's own tick labels
            bb = ax_C1.xaxis.get_tightbbox(rend)
            y = fig.transFigure.inverted().transform(bb)[0][1] - 0.008 * 170 / fig_h_mm
            va = "top"
    except Exception:
        va = "top"
    fig.text(xm, y, "DLB feature-burden score, x", fontsize=6.5, ha="center", va=va)

    leg_CD = [Line2D([0], [0], color=C_CD_POS, lw=1.3, label="AID+"),
              Line2D([0], [0], color=C_CD_NEG, lw=1.1, label=f"AID{M}")]
    if S_DLB is not None:
        leg_CD.append(Line2D([0], [0], color=C_TEXT, lw=0.8, ls=":", alpha=0.75,
                             label="Coded DLB (ref.)"))
    ax_C1.legend(handles=leg_CD, loc="upper right", frameon=False, fontsize=5.0,
                 handlelength=1.5, handletextpad=0.4, borderaxespad=0.2,
                 labelspacing=0.3)

    # ---- Panel tags (one shared pad, in mm so they do not move with the height) ----
    TAG_PAD_X = 0.030
    TAG_PAD_Y_MM = 2.04
    TAG_A_EXTRA_MM = 3.74

    def panel_tag(ax, letter, extra_mm=0.0):
        bb = ax.get_position()
        fig.text(bb.x0 - TAG_PAD_X, bb.y1 + (TAG_PAD_Y_MM + extra_mm) / fig_h_mm, letter,
                 fontsize=8, fontweight="bold", va="bottom", ha="left")

    panel_tag(ax_A, "a", extra_mm=TAG_A_EXTRA_MM)
    panel_tag(ax_B, "b")
    panel_tag(ax_C1, "c")

    save_fig(fig, FIGURE_FILE)
    full = dict(zip(df_B["short"], df_B["label"])) if len(df_B) else {}
    merged_full = [[full[t] for t in m.split("\n")] for m in merged.values()]
    return dict(n_gutter=n_gutter, n_feats=n_feats, merged=merged_full, b_fitted=len(df_B) >= 2,
                ref_drawn=S_DLB is not None, dropped=dropped,
                no_q=bool(df_A[["floor_f", "floor_fx", "floor_m"]].any().any()))


# =============================================================================
# CAPTION
# =============================================================================
def lower_label(x):
    """A row label inside a caption sentence: first letter lowercased, except a
    label that starts with an abbreviation or a proper name (REM, McKeith)."""
    return x if x.split()[0].isupper() or x.startswith("REM") else x[0].lower() + x[1:]


def captions(cap, wb, info):
    c = cap["caption"]
    summ = wb["c_summary"].set_index("arm")
    n = lambda key: ru.fmt_count(parameter(wb, key))
    out = [ru.fill(c["body"], {})]
    out.append(ru.fill(c["a"], {"k": info["n_feats"],
                                "aid_pos_female": n("n_aid_pos_female"),
                                "aid_neg_female": n("n_aid_neg_female"),
                                "aid_pos_male": n("n_aid_pos_male"),
                                "aid_neg_male": n("n_aid_neg_male")}))
    if info["dropped"]:
        out.append(ru.fill(c["a_not_shown"], {"features": ru.join_and(
            lower_label(x) for x in info["dropped"])}))
    out.append(c["a_symbols"])
    if info["no_q"]:
        out.append(ru.fill(c["a_no_q"], {"group_floor": int(parameter(wb, "group_floor"))}))
    out.append(ru.fill(c["a_sex"], {"threshold": ru.fmt_p(SEX_Q_THRESH)}))
    out.append(ru.fill(c["b"], {"group_floor": int(parameter(wb, "group_floor"))})
               if info["b_fitted"] else c["b_not_fitted"])
    if info["merged"]:
        out.append(ru.fill(c["b_merged"], {"features": "; ".join(
            ru.join_and(lower_label(x) for x in m) for m in info["merged"])}))
    arm_n = {}
    for arm in ("all", "dlb_excluded"):
        arm_n[f"pos_{arm}"] = ru.fmt_count(summ.loc[arm, "n_aid_pos"]) if arm in summ.index else ru.DASH
        arm_n[f"neg_{arm}"] = ru.fmt_count(summ.loc[arm, "n_aid_neg"]) if arm in summ.index else ru.DASH
    out.append(ru.fill(c["c"], {**arm_n, "resamples": n("bootstrap_resamples"),
                                "core_points": int(parameter(wb, "score_core_points")),
                                "supportive_cap": int(parameter(wb, "score_supportive_cap"))}))
    if info["ref_drawn"]:
        out.append(ru.fill(c["c_reference"], {"n_dlb": n("n_coded_dlb_reference")}))
    else:
        out.append(ru.fill(c["c_reference_none"], {
            "n_dlb": n("n_coded_dlb_reference"),
            "min_n": int(parameter(wb, "coded_dlb_reference_min_n"))}))
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