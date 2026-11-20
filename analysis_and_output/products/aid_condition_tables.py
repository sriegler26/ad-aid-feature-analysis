"""
aid_condition_tables.py
=======================
Renders the AID condition tables (.docx) from the source-data workbooks
written by aid_condition_analysis.py:
  - the published groupings: per class against the contrasting class, the
    rate ratio and its Q, then the female carriers and observed / expected
    core features in each; contrasts that could not be estimated are named in the
    caption rather than shown as empty rows;
  - the possible-mimic pairs: per condition and core feature it has been
    reported to produce on its own, grouped by the cumulative set of possible
    mimics it first enters (narrow, wider, all reported; the sets whose
    carriers are removed in panel d of the AID condition figure), the female
    carriers, the feature
    observed / expected, the
    feature as the only core feature (observed / expected), with other core
    features, and with a coded DLB or PD diagnosis.
The per-condition SIRs are shown in the AID condition figure and its extended data figure
(drawn by aid_condition_figure.py); aid_condition_analysis.py also prints them.
Fits no model and computes no statistic: every number comes from the
workbooks, formatted once by render_utils. Rows: groupings in their
fixed order; possible-mimic pairs by set, then alphabetical. Captions: the titles
and caption wording are read from an optional caption file (CAPTIONS_PATH);
this script fills in the numbers and picks the sentences that apply. With
no file set, the tables are written without them.

Run from the data folder, after aid_condition_analysis.py.
"""

import os

import pandas as pd

import render_utils as ru

# =============================================================================
# PATHS (relative to the data folder the script is run from)
# =============================================================================
SOURCE_DATA_DIR = ""  # folder of the source-data workbooks (.xlsx) written by aid_condition_analysis.py
TABLES_DIR = ""  # folder for the .docx tables written here
CAPTIONS_PATH = ""  # optional JSON with titles and captions; blank = bare tables

MIMIC_TABLE = "table_aid_condition_mimic_pairs"
GROUPING_TABLE = "table_aid_condition_groupings"
WIDE = {"first_width": 2.0, "pad": 1}
TIER_ORDER = ["narrow", "wider", "reported", "check"]
TIER_HEADERS = {"narrow": "Narrow set", "wider": "Added in the wider set",
                "reported": "Added in the all-reported set",
                "check": "Check (in the all-reported set)"}
FEATURE_SHORT = {"F": "fluctuating cognition", "V": "visual hallucinations",
                 "P": "parkinsonism"}
# Groupings with contrasts, in table order (PD risk and PD pleiotropy are descriptive
# only in aid_condition_analysis.py and have no contrast rows)
GROUPING_ORDER = ["serology", "systemic", "interferon", "genetic_cluster", "continuum"]


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


def obs_exp(o, e):
    return f"{ru.fmt_count(o)} / {ru.fmt_num(e, 2)}"


def lower_first(s):
    return s if s[:2].isupper() else s[:1].lower() + s[1:]   # keeps "PD ..."


# =============================================================================
# GROUPING TABLE
# =============================================================================
def grouping_table(cap):
    wb = workbook(GROUPING_TABLE)
    df, gone = wb["contrasts"], wb["not_shown"]
    unknown = sorted(set(df["grouping"]) - set(GROUPING_ORDER))
    if unknown:
        raise KeyError(f"grouping table: groupings not in GROUPING_ORDER {unknown}")
    rows = []
    for g in GROUPING_ORDER:
        d = df[df["grouping"] == g]
        if not len(d):
            continue
        rows.append({"kind": "header", "label": d["grouping_label"].iloc[0]})
        for _, r in d.iterrows():
            rows.append({"kind": "data", "values": [
                f"{r['class_label']} vs {lower_first(r['comparator_label'])}",
                ru.fmt_ratio_ci(r["ratio"], r["ci_low"], r["ci_high"]), ru.fmt_p(r["q_ratio"]),
                f"{ru.fmt_count(r['n_1'])} / {ru.fmt_count(r['n_0'])}",
                obs_exp(r["o_1"], r["e_1"]), obs_exp(r["o_0"], r["e_0"])]})

    def captions():
        c = cap["groupings"]
        out = [ru.fill(c["body"], {"n_female": ru.fmt_count(parameter(wb, "n_aid_pos_female"))}),
               c["groupings"], c["sources"], c["method"],
               ru.fill(c["fdr"], {"k": ru.fmt_count(len(df))})]
        if len(gone):
            listed = "; ".join(f"{r['class_label']} ({ru.fmt_count(r['n_1'])})"
                               for _, r in gone.iterrows())
            out.append(ru.fill(c["not_shown"], {"classes": listed,
                                                "min_group": int(parameter(wb, "min_group_n")),
                                                "min_events": int(parameter(wb,
                                                                            "min_events_per_term"))}))
        out.append(c["abbreviations"])
        return out
    headers = ["Class vs contrasting class", "Rate ratio\n(95% CI)", "*Q*",
               "Women, *n*\n(class / contrast)",
               "Core features, class\nobserved / expected",
               "Core features, contrast\nobserved / expected"]
    title = cap["groupings"]["title"] if cap else None
    ru.write_table(os.path.join(TABLES_DIR, GROUPING_TABLE + ".docx"), headers, rows, title,
                   captions() if cap else (), **WIDE)


# =============================================================================
# MIMIC-PAIR TABLE
# =============================================================================
def mimic_table(cap):
    wb = workbook(MIMIC_TABLE)
    df = wb["pairs"]
    absent = df[df["status"] == "no carriers"]
    rows = []
    for tier in TIER_ORDER:
        d = df[(df["tier"] == tier) & (df["status"] != "no carriers")]
        if not len(d):
            continue
        rows.append({"kind": "header", "label": TIER_HEADERS[tier]})
        for _, r in sorted(d.iterrows(), key=lambda x: (ru.sort_key(x[1]["row_label"]),
                                                       x[1]["feature"])):
            rows.append({"kind": "data", "values": [
                f"{r['row_label']}, {FEATURE_SHORT[r['feature']]}", ru.fmt_count(r["n_female"]),
                obs_exp(r["o"], r["e"]),
                obs_exp(r["n_feature_alone"], r["n_feature_alone_expected"]),
                ru.fmt_count(r["n_feature_with_other_core"]),
                ru.fmt_count(r["n_feature_coded_lbd"])]})

    def captions():
        c = cap["mimic"]
        out = [ru.fill(c["body"], {"n_female": ru.fmt_count(parameter(wb, "n_aid_pos_female"))}),
               c["tiers"], c["columns"]]
        if (df["status"] == "fewer than 10 female carriers").any():
            out.append(c["few"])
        if len(absent):
            out.append(ru.fill(c["absent"], {"pairs": "; ".join(
                f"{r['row_label']}, {FEATURE_SHORT[r['feature']]}" for _, r in absent.iterrows())}))
        out.append(c["abbreviations"])
        return out
    headers = ["Condition, feature", "Women, *n*", "Feature\nobserved / expected",
               "Feature as the only core\nfeature, observed / expected",
               "Feature with other\ncore features, *n*", "Feature and coded\nDLB or PD, *n*"]
    title = cap["mimic"]["title"] if cap else None
    ru.write_table(os.path.join(TABLES_DIR, MIMIC_TABLE + ".docx"), headers, rows, title,
                   captions() if cap else (), **WIDE)


def main():
    cap = ru.load_captions(CAPTIONS_PATH, __file__)
    print(f"  Titles and captions: {'written' if cap else 'none (no caption file set)'}")
    grouping_table(cap)
    mimic_table(cap)


if __name__ == "__main__":
    main()