"""
cognitive_score_comparison_tables.py
====================================
Renders the two cognitive score tables (.docx) from the source-data workbooks
written by cognitive_score_comparison.py. Fits no model and computes no
statistic: every number comes from the workbooks, formatted once by
render_utils.

Tables
------
- Cognitive scores: annual change per score (MMSE, MoCA), the AID+ minus
  AID- difference (95% CI), P, then the patients per arm.
- Cognitive scores and medication exposure: per class (corticosteroids,
  synthetic DMARDs and immunosuppressants, antimalarials, biologic DMARDs)
  and score, the annual-change
  AID x exposure interaction (95% CI), Q (within class), then the patients
  in the four AID x exposure groups. Classes ordered by their number of
  fitted rows, most first (ties alphabetical), as in the medication tables.
Results come first, then counts; every CI is inline. Every row is shown; "–"
marks a model not fitted. P values of the scores table are not
corrected for multiple comparisons (two scores).

Captions: the titles and caption wording are read from an optional caption
file (CAPTIONS_PATH); this script fills in the numbers and picks the
sentences that apply. With no file set, the tables are written without them.

Run from the data folder, after cognitive_score_comparison.py.
"""

import os

import pandas as pd

import render_utils as ru

# =============================================================================
# PATHS (relative to the data folder the script is run from)
# =============================================================================
SOURCE_DATA_DIR = ""  # folder of the source-data workbooks (.xlsx) written by cognitive_score_comparison.py
TABLES_DIR = ""  # folder for the .docx tables written here
CAPTIONS_PATH = ""  # optional JSON with titles and captions; blank = bare tables

SCORES_FILE = "table_cognitive_scores"
MEDICATION_FILE = "table_cognitive_scores_medication"

M = ru.MINUS
MEASURES = [("annual_change", "Annual change, points per year")]
MED_MEASURES = {"annual_change": "annual change, points per year"}
CLASSES = ["on_cs", "on_dmard", "on_antimalarial", "on_biologic"]
CLASS_LABELS = {"on_cs": "Corticosteroids", "on_dmard": "Synthetic DMARDs and immunosuppressants",
                "on_antimalarial": "Antimalarials", "on_biologic": "Biologic DMARDs"}
CELLS = ["aid_pos_exposed", "aid_pos_unexposed", "aid_neg_exposed", "aid_neg_unexposed"]
CELL_HEADERS = ["AID+ exposed, *n*", "AID+ unexposed, *n*",
                f"AID{M} exposed, *n*", f"AID{M} unexposed, *n*"]


def workbook(name):
    path = os.path.join(SOURCE_DATA_DIR, name + ".xlsx")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path}: run cognitive_score_comparison.py first")
    return pd.read_excel(path, sheet_name=None)


def parameter(wb, key):
    p = wb["parameters"].set_index("parameter")["value"]
    if key not in p.index:
        raise KeyError(f"parameters: {key} missing")
    return p[key]


def coef_pair(r):
    if r["status"] != "fitted":
        return [ru.DASH, ru.DASH]
    return [ru.fmt_coef_ci(r["coef"], r["ci_low"], r["ci_high"]), ru.fmt_p(r["p"])]


def scores_table(cap):
    wb = workbook(SCORES_FILE)
    sc = wb["scores"]
    rows = []
    for measure, label in MEASURES:
        rows.append({"kind": "header", "label": label})
        for _, r in sorted(sc[sc["measure"] == measure].iterrows(),
                           key=lambda x: ru.sort_key(x[1]["score_type"])):
            rows.append({"kind": "data", "values": [r["score_type"]] + coef_pair(r)
                         + [ru.fmt_count(r["n_aid_pos"]), ru.fmt_count(r["n_aid_neg"])]})

    def captions():
        t = cap["scores"]
        return [ru.fill(t["body"], {
                    "n_aid_pos": ru.fmt_count(parameter(wb, "n_scored_aid_pos")),
                    "n_aid_neg": ru.fmt_count(parameter(wb, "n_scored_aid_neg"))}),
                cap["common"]["scores_used"],
                ru.fill(t["model"], {"gap": ru.fmt_num(parameter(wb, "min_slope_gap_yrs"), 2)}),
                t["counts"], cap["common"]["ci"], t["p"],
                ru.fill(t["legend"], {"min_events": int(parameter(wb, "min_events_per_term"))})]
    write(SCORES_FILE, ["Measure and score", f"Difference, AID+ minus AID{M}\n(95% CI)", "*P*",
                        "AID+, *n*", f"AID{M}, *n*"], rows, cap, "scores", captions)


def class_order(df):
    """Classes by number of fitted rows, most first; ties alphabetical."""
    fitted = df[df["status"] == "fitted"].groupby("drug_class").size()
    return sorted(CLASSES, key=lambda c: (-int(fitted.get(c, 0)), ru.sort_key(CLASS_LABELS[c])))


def interaction_cells(r):
    if r["status"] != "fitted":
        return [ru.DASH, ru.DASH]
    return [ru.fmt_coef_ci(r["coef"], r["ci_low"], r["ci_high"]), ru.fmt_p(r["q"])]


def medication_table(cap):
    wb = workbook(MEDICATION_FILE)
    df = wb["interactions"]
    order = class_order(df)
    rows = []
    for c in order:
        rows.append({"kind": "header", "label": CLASS_LABELS[c]})
        part = df[df["drug_class"] == c]
        for measure, _ in MEASURES:
            m = part[part["measure"] == measure]
            for _, r in sorted(m.iterrows(), key=lambda x: ru.sort_key(x[1]["score_type"])):
                rows.append({"kind": "data", "values": [f"{r['score_type']} {MED_MEASURES[measure]}"]
                             + interaction_cells(r) + [ru.fmt_count(r[f"n_{x}"]) for x in CELLS]})

    def captions():
        t, cm = cap["medication"], cap["common"]
        fams = [(c, int(parameter(wb, f"fdr_family_size_{c}"))) for c in order]
        fdr = (ru.fill(t["fdr"], {"families": "; ".join(
            f"{dict(cm['class_names'])[c]}, {n}" for c, n in fams)})
            if sum(n for _, n in fams) else t["fdr_none"])
        out = [ru.fill(t["body"], {
                   "n_aid_pos": ru.fmt_count(parameter(wb, "n_scored_aid_pos")),
                   "n_aid_neg": ru.fmt_count(parameter(wb, "n_scored_aid_neg"))}),
               t["interaction"], ru.fill(t["model"], {"gap": ru.fmt_num(
                   parameter(wb, "min_slope_gap_yrs"), 2)}),
               t["exposure"], t["classes"], cm["scores_used"], t["counts"], cm["ci"], fdr]
        if (df["status"] != "fitted").any():
            out.append(ru.fill(t["legend"], {"min_events": int(parameter(wb, "min_events_per_term")),
                                             "min_group": int(parameter(wb, "min_group_n"))}))
        return out
    write(MEDICATION_FILE, ["Class, score and measure", "Interaction\n(95% CI)", "*Q*"]
          + CELL_HEADERS, rows, cap, "medication", captions)


# seven-column table: a narrower label column and padding keep the inline CIs
# and the count headers on one line per word
WIDE = {"first_width": 2.3, "pad": 2}


def write(name, headers, rows, cap, key, captions):
    title = cap[key]["title"] if cap else None
    ru.write_table(os.path.join(TABLES_DIR, name + ".docx"), headers, rows, title,
                   captions() if cap else (), **(WIDE if len(headers) > 5 else {}))


def main():
    cap = ru.load_captions(CAPTIONS_PATH, __file__)
    print(f"  Titles and captions: {'written' if cap else 'none (no caption file set)'}")
    scores_table(cap)
    medication_table(cap)


if __name__ == "__main__":
    main()