"""
inflammation_tier_sensitivity_tables.py
=======================================
Renders the five inflammation tier tables (.docx) from the source-data
workbooks written by inflammation_tier_sensitivity.py. Fits no model and
computes no statistic: every number comes from the workbooks, formatted once
by render_utils.

Tables
------
- Inflammation tier frequency: per sex and tier, the AID+ vs AID- odds ratio
  of a tier mention (95% CI), Q, then the patients with the tier per arm.
- Inflammation tier and symptom prevalence: per tier, domains and categories;
  the AID+ vs AID- odds ratio (95% CI) within patients with and within
  patients without the tier, the ratio of the two (95% CI) and Q (the AID x
  tier interaction), then the patients with the outcome documented in the
  four AID x tier groups.
- Inflammation tier and age at first documentation: the same, in years (the
  AID+ minus AID- difference in each stratum and the difference of the two),
  per domain.
- Inflammation tier and symptom sequence: per tier and expected-order pair,
  within patients with and within patients without the tier the share
  inverted in AID+ and in AID- and the AID+ vs AID- odds ratio (exact 95%
  CI), then the ratio of the two odds ratios and Q (exact test of the
  interaction).
- Inflammation tier and cognitive scores: the layout of the timing table, in
  points per year (the AID+ minus AID- difference in the annual change in
  each stratum and the difference of the two), per score (MMSE, MoCA), then
  the patients in the four AID x tier groups.
Results come first, then counts; every CI is inline. Every row is shown; "–"
marks a model not fitted or a comparison not tested, "NE" (sequence: "not
estimable") an odds ratio that is not estimable. Rows are
alphabetical.

Captions: the titles and caption wording are read from an optional caption
file (CAPTIONS_PATH); this script fills in the numbers and picks the
sentences that apply. With no file set, the tables are written without them.

Run from the data folder, after inflammation_tier_sensitivity.py.
"""

import os

import pandas as pd

import render_utils as ru

# =============================================================================
# PATHS (relative to the data folder the script is run from)
# =============================================================================
SOURCE_DATA_DIR = ""  # folder of the source-data workbooks (.xlsx) written by inflammation_tier_sensitivity.py
TABLES_DIR = ""  # folder for the .docx tables written here
CAPTIONS_PATH = ""  # optional JSON with titles and captions; blank = bare tables

FREQUENCY_FILE = "table_inflammation_tier_frequency"
PREVALENCE_FILE = "table_inflammation_tier_prevalence"
TIMING_FILE = "table_inflammation_tier_timing"
SEQUENCE_FILE = "table_inflammation_tier_sequence"
COGNITIVE_FILE = "table_inflammation_tier_cognitive_scores"

M = ru.MINUS
TIERS = ["inflammation_record", "lab_result", "flare"]
TIER_LABELS = {"inflammation_record": "Inflammation record", "lab_result": "Lab result",
               "flare": "Flare"}
WITH, WITHOUT, INTERACTION = ("aid_pos_vs_aid_neg_with_tier", "aid_pos_vs_aid_neg_without_tier",
                              "interaction")
CELLS = ["aid_pos_exposed", "aid_pos_unexposed", "aid_neg_exposed", "aid_neg_unexposed"]
CELL_HEADERS = ["AID+ with the tier", "AID+ without", f"AID{M} with the tier", f"AID{M} without"]
STRATA = [("exposed", "With the tier"), ("unexposed", "Without the tier")]
NOT_ESTIMABLE = "not estimable"


def workbook(name):
    path = os.path.join(SOURCE_DATA_DIR, name + ".xlsx")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path}: run inflammation_tier_sensitivity.py first")
    return pd.read_excel(path, sheet_name=None)


def parameter(wb, key):
    p = wb["parameters"].set_index("parameter")["value"]
    if key not in p.index:
        raise KeyError(f"parameters: {key} missing")
    return p[key]


def tier_order():
    return sorted(TIERS, key=lambda t: ru.sort_key(TIER_LABELS[t]))


# =============================================================================
# CELLS
# =============================================================================
def q_cell(r):
    if r["status"] != "fitted" or r["fdr_role"] != "tested":
        return ru.DASH
    return ru.fmt_p(r["q"])


def estimate_cell(r, ratio):
    if r["status"] != "fitted":
        return ru.DASH
    if ratio:
        return ru.fmt_ratio_ci(r["odds_ratio"], r["ci_low"], r["ci_high"])
    return ru.fmt_coef_ci(r["coef"], r["ci_low"], r["ci_high"])


def estimate_rows(df, ratio, counts):
    """One table row per outcome (df: one tier and level): AID+ vs AID- with
    the tier, without it, the interaction and its Q, then the counts (from the
    interaction row; they are the same on the three rows)."""
    rows = []
    inter = df[df["estimate"] == INTERACTION]
    for _, ri in sorted(inter.iterrows(), key=lambda x: ru.sort_key(x[1]["row_label"])):
        same = df[df["code"] == ri["code"]].set_index("estimate")
        rows.append({"kind": "data", "values": [ri["row_label"]]
                     + [estimate_cell(same.loc[e], ratio) for e in (WITH, WITHOUT, INTERACTION)]
                     + [q_cell(ri)] + counts(ri)})
    return rows


def share_cell(k, n):
    """"x.x% (k/n)"; "–" for an empty group."""
    if pd.isna(n) or int(n) == 0:
        return ru.DASH
    return f"{ru.fmt_pct(100 * float(k) / float(n))} ({ru.fmt_count_of(k, n)})"


def stratum_cells(r, s):
    """AID+ and AID- inverted shares and the AID+ vs AID- OR of one stratum."""
    if r[f"status_{s}"] != "tested":
        orv = ru.DASH
    elif int(r[f"or_estimable_{s}"]):
        orv = ru.fmt_ratio_ci(r[f"odds_ratio_{s}"], r[f"ci_low_{s}"], r[f"ci_high_{s}"])
    else:
        orv = NOT_ESTIMABLE
    return [share_cell(r[f"inverted_aid_pos_{s}"], r[f"n_aid_pos_{s}"]),
            share_cell(r[f"inverted_aid_neg_{s}"], r[f"n_aid_neg_{s}"]), orv]


def modification_rows(df):
    rows = []
    for _, r in sorted(df.iterrows(), key=lambda x: ru.sort_key(x[1]["row_label"])):
        if r["status"] != "tested":
            res = [ru.DASH, ru.DASH]
        else:
            res = [ru.fmt_ratio(r["or_ratio"]) if int(r["ratio_estimable"]) else NOT_ESTIMABLE,
                   ru.fmt_p(r["q"])]
        rows.append({"kind": "data", "values": [r["row_label"]] + stratum_cells(r, "exposed")
                     + stratum_cells(r, "unexposed") + res})
    return rows


# =============================================================================
# TABLES
# =============================================================================
def frequency_table(cap):
    wb = workbook(FREQUENCY_FILE)
    fr = wb["frequency"]
    rows = []
    for sex, lbl in (("Female", "Women"), ("Male", "Men")):
        rows.append({"kind": "header", "label": lbl})
        f = fr[fr["sex"] == sex]
        for _, r in sorted(f.iterrows(), key=lambda x: ru.sort_key(x[1]["row_label"])):
            if r["status"] != "tested":
                res = [ru.DASH, ru.DASH]
            elif r["or_estimable"] != 1:
                res = ["NE", ru.fmt_p(r["q"])]
            else:
                res = [ru.fmt_ratio_ci(r["odds_ratio"], r["ci_low"], r["ci_high"]),
                       ru.fmt_p(r["q"])]
            rows.append({"kind": "data", "values": [r["row_label"]] + res
                         + [ru.fmt_count(r["with_tier_aid_pos"]),
                            ru.fmt_count(r["with_tier_aid_neg"])]})

    def captions():
        c, t = cap["common"], cap["frequency"]
        v = {k: ru.fmt_count(parameter(wb, k)) for k in
             ("n_aid_pos_female", "n_aid_pos_male", "n_aid_neg_female", "n_aid_neg_male")}
        fam = int(parameter(wb, "fdr_family_size"))
        out = [ru.fill(t["body"], v), c["tiers"], c["ceiling"], t["counts"], c["ci"],
               ru.fill(t["fdr"], {"family": fam}) if fam else t["fdr_none"]]
        if (fr["status"] == "tested").any() and (fr.loc[fr["status"] == "tested",
                                                        "or_estimable"] != 1).any():
            out.append(t["not_estimable"])
        out.append(ru.fill(t["legend"], {"min_group": int(parameter(wb, "min_group_n"))}))
        return out
    write(FREQUENCY_FILE, ["Tier", "Odds ratio\n(95% CI)", "*Q*", "AID+ with the tier, *n*",
                           f"AID{M} with the tier, *n*"], rows, cap, "frequency", captions)


def group_sizes_text(cap, wb):
    """Group sizes per tier: "inflammation record, AID+ with the tier n = …"."""
    parts = []
    for x in tier_order():
        sizes = {lbl: ru.fmt_count(parameter(wb, f"n_{lbl}_{x}")) for lbl in CELLS}
        parts.append(ru.fill(cap["common"]["group_sizes_item"],
                             {"tier": TIER_LABELS[x].lower(), **sizes}))
    return ru.fill(cap["common"]["group_sizes"], {"items": "; ".join(parts)})


def secondary_sentence(cap, wb, key, fmt):
    sd = wb["secondary_definition"]
    if sd.empty:
        return []
    hits = sd[(sd["became_significant"] == 1) | (sd["lost_significance"] == 1)
              | (sd["change_ge_small"] == 1)]
    if hits.empty:
        return []
    items = []
    for _, r in sorted(hits.iterrows(), key=lambda x: (ru.sort_key(x[1]["row_label"]),
                                                       ru.sort_key(TIER_LABELS[x[1]["tier"]]))):
        crossed = (ru.fill(cap["common"]["secondary_crossed"],
                           {"q_threshold": ru.fmt_p(parameter(wb, "q_threshold"))})
                   if r["became_significant"] or r["lost_significance"] else "")
        items.append(ru.fill(cap[key]["secondary_item"], {
            "domain": r["row_label"], "tier": TIER_LABELS[r["tier"]].lower(),
            "est_primary": fmt(r["estimate_primary"]), "est_secondary": fmt(r["estimate_secondary"]),
            "q_primary": ru.p_phrase("Q", r["q_primary"]),
            "q_secondary": ru.p_phrase("Q", r["q_secondary"]), "crossed": crossed}))
    return [ru.fill(cap["common"]["secondary_definition"], {"changes": ru.join_and(items)})]


def family_text(cap, wb, key, levels):
    """The tested interactions per tier; levels: [(parameter suffix, unit)]."""
    parts, total = [], 0
    for x in tier_order():
        sizes = [int(parameter(wb, f"fdr_family_size_{x}{suffix}")) for suffix, _ in levels]
        total += sum(sizes)
        parts.append(f"{TIER_LABELS[x].lower()}, "
                     + " and ".join(f"{n} {unit}" for n, (_, unit) in zip(sizes, levels)))
    if not total:
        return cap[key]["fdr_none"]
    return ru.fill(cap[key]["fdr"], {"families": "; ".join(parts)})


def prevalence_table(cap):
    wb = workbook(PREVALENCE_FILE)
    df = wb["estimates"]
    rows = []
    counts = lambda r: [ru.fmt_count(r[f"events_{c}"]) for c in CELLS]
    for x in tier_order():
        for level, lbl in (("domain", "domains"), ("category", "categories")):
            part = df[(df["tier"] == x) & (df["level"] == level)]
            rows.append({"kind": "header", "label": f"{TIER_LABELS[x]}: {lbl}"})
            rows += estimate_rows(part, True, counts)

    def captions():
        c, t = cap["common"], cap["prevalence"]
        out = [ru.fill(t["body"], {"n_aid_pos": ru.fmt_count(parameter(wb, "n_aid_pos")),
                                   "n_aid_neg": ru.fmt_count(parameter(wb, "n_aid_neg"))}),
               group_sizes_text(cap, wb), t["question"], t["ratio"], t["model"], c["tiers"],
               c["ceiling"], c["circular"], t["other"], t["documented"], c["ci"],
               family_text(cap, wb, "prevalence", [("_domain", "domains"),
                                                   ("_category", "categories")])]
        out += secondary_sentence(cap, wb, "prevalence", ru.fmt_ratio)
        out.append(ru.fill(t["legend"], {"min_events": int(parameter(wb, "min_events_per_term")),
                                         "min_group": int(parameter(wb, "min_group_n"))}))
        return out
    headers = (["Domain or category (domain)"]
               + [f"{w}: odds ratio\n(95% CI)" for _, w in STRATA]
               + ["Ratio of odds ratios\n(95% CI)", "*Q*"]
               + [f"{h}\ndocumented, *n*" for h in CELL_HEADERS])
    write(PREVALENCE_FILE, headers, rows, cap, "prevalence", captions)


def timing_table(cap):
    wb = workbook(TIMING_FILE)
    df = wb["estimates"]
    rows = []
    counts = lambda r: [ru.fmt_count(r[f"n_{c}"]) for c in CELLS]
    for x in tier_order():
        rows.append({"kind": "header", "label": TIER_LABELS[x]})
        rows += estimate_rows(df[df["tier"] == x], False, counts)

    def captions():
        c, t = cap["common"], cap["timing"]
        out = [ru.fill(t["body"], {"n_aid_pos": ru.fmt_count(parameter(wb, "n_aid_pos")),
                                   "n_aid_neg": ru.fmt_count(parameter(wb, "n_aid_neg"))}),
               t["question"], t["ratio"], t["model"], c["tiers"], t["exposure"],
               c["circular_domain"], t["documented"], c["ci"],
               family_text(cap, wb, "timing", [("", "domains")])]
        out += secondary_sentence(cap, wb, "timing", ru.fmt_coef)
        out.append(ru.fill(t["legend"], {"min_events": int(parameter(wb, "min_events_per_term")),
                                         "min_group": int(parameter(wb, "min_group_n"))}))
        return out
    headers = (["Domain"]
               + [f"{w}: difference,\nyears (95% CI)" for _, w in STRATA]
               + ["Difference of\ndifferences, years\n(95% CI)", "*Q*"]
               + [f"{h}\ndocumented, *n*" for h in CELL_HEADERS])
    write(TIMING_FILE, headers, rows, cap, "timing", captions)


def sequence_table(cap):
    wb = workbook(SEQUENCE_FILE)
    df = wb["pairs"]
    rows = []
    for x in tier_order():
        rows.append({"kind": "header", "label": TIER_LABELS[x]})
        rows += modification_rows(df[df["tier"] == x])

    def captions():
        c, t = cap["common"], cap["sequence"]
        fams = [(x, int(parameter(wb, f"fdr_family_size_{x}"))) for x in tier_order()]
        fdr = (ru.fill(t["fdr"], {"families": "; ".join(
            f"{TIER_LABELS[x].lower()}, {n}" for x, n in fams)})
            if sum(n for _, n in fams) else t["fdr_none"])
        out = [ru.fill(t["body"], {"n_aid_pos": ru.fmt_count(parameter(wb, "n_aid_pos")),
                                   "n_aid_neg": ru.fmt_count(parameter(wb, "n_aid_neg"))}),
               t["question"], t["ratio"], c["tiers"], t["exposure"], t["pairs"],
               ru.fill(t["sample"], {"yrs": int(parameter(wb, "predx_span_floor_yrs"))}),
               t["counts"], c["ci"], fdr]
        if (df["status"] != "tested").any() or any(
                (df[f"status_{s}"] != "tested").any() for s, _ in STRATA):
            out.append(ru.fill(t["legend"], {"min_group": int(parameter(wb, "min_group_n"))}))
        if any(((df[f"status_{s}"] == "tested") & (df[f"or_estimable_{s}"] == 0)).any()
               for s, _ in STRATA):
            out.append(t["not_estimable"])
        return out
    headers = (["Expected order"]
               + [h for _, w in STRATA for h in (f"{w}: AID+ inverted,\n% (*n*/*N*)",
                                                 f"{w}: AID{M} inverted,\n% (*n*/*N*)",
                                                 f"{w}: odds ratio\n(95% CI)")]
               + ["Ratio of odds ratios", "*Q*"])
    write(SEQUENCE_FILE, headers, rows, cap, "sequence", captions)


def cognitive_scores_table(cap):
    wb = workbook(COGNITIVE_FILE)
    df = wb["estimates"]
    rows = []
    counts = lambda r: [ru.fmt_count(r[f"n_{c}"]) for c in CELLS]
    for x in tier_order():
        rows.append({"kind": "header", "label": TIER_LABELS[x]})
        rows += estimate_rows(df[df["tier"] == x], False, counts)

    def captions():
        c, t = cap["common"], cap["cognitive_scores"]
        out = [ru.fill(t["body"], {"n_aid_pos": ru.fmt_count(parameter(wb, "n_scored_aid_pos")),
                                   "n_aid_neg": ru.fmt_count(parameter(wb, "n_scored_aid_neg"))}),
               t["question"], t["ratio"],
               ru.fill(t["model"], {"gap": ru.fmt_num(parameter(wb, "min_slope_gap_yrs"), 2)}),
               c["tiers"], t["exposure"], t["scores_used"], t["counts"], c["ci"],
               family_text(cap, wb, "cognitive_scores", [("", "scores")])]
        out.append(ru.fill(t["legend"], {"min_events": int(parameter(wb, "min_events_per_term")),
                                         "min_group": int(parameter(wb, "min_group_n"))}))
        return out
    headers = (["Score"]
               + [f"{w}: difference,\npoints per year (95% CI)" for _, w in STRATA]
               + ["Difference of\ndifferences, points\nper year (95% CI)", "*Q*"]
               + [f"{h}, *n*" for h in CELL_HEADERS])
    write(COGNITIVE_FILE, headers, rows, cap, "cognitive_scores", captions)


# nine-column tables: a narrower label column and padding keep the inline CIs,
# the Q values and the count headers on one line per word
WIDE = {"first_width": 2.3, "pad": 2}


def write(name, headers, rows, cap, key, captions):
    title = cap[key]["title"] if cap else None
    ru.write_table(os.path.join(TABLES_DIR, name + ".docx"), headers, rows, title,
                   captions() if cap else (), **(WIDE if len(headers) > 5 else {}))


def main():
    cap = ru.load_captions(CAPTIONS_PATH, __file__)
    print(f"  Titles and captions: {'written' if cap else 'none (no caption file set)'}")
    frequency_table(cap)
    prevalence_table(cap)
    timing_table(cap)
    sequence_table(cap)
    cognitive_scores_table(cap)


if __name__ == "__main__":
    main()