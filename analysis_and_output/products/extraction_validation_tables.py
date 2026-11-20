"""
extraction_validation_tables.py
===============================
Renders the three tables (.docx) of the symptom extraction validation from
their source-data workbooks:
  - table_extraction_validation: name and category precision and recall
    (primary annotator), overall, then category precision by category;
  - table_extraction_validation_by_aid_sex: the same three measures by AID
    status and sex, with the AID+ vs AID- and female vs male tests;
  - table_extraction_validation_second_annotator: each annotator's estimate
    on the shared items, observed agreement (precision) or positive specific
    agreement and the Jaccard index (recall), and Gwet's AC1.
The workbooks hold aggregate counts from the validation (entered by hand from
the validation run; no item-level data) and every displayed value as a
formula of those counts. This script fits no model and computes no statistic:
every number comes from the workbooks, formatted once by render_utils.
Rows: categories alphabetical, Other last. Captions: the titles and caption
wording are read from an optional caption file (CAPTIONS_PATH); this script
fills in the numbers. With no file set, the tables are written without them.

Run from the data folder. The three workbooks must be in SOURCE_DATA_DIR,
with their formula values saved (open and save them once in Excel or
LibreOffice if they were edited).
"""

import os

import pandas as pd

import render_utils as ru

# =============================================================================
# PATHS (relative to the data folder the script is run from)
# =============================================================================
SOURCE_DATA_DIR = ""  # folder of the three validation workbooks (.xlsx)
TABLES_DIR = ""  # folder for the .docx tables written here
CAPTIONS_PATH = ""  # optional JSON with titles and captions; blank = bare tables

PRECISION_FILE = "table_extraction_validation"
BY_GROUP_FILE = "table_extraction_validation_by_aid_sex"
ANNOTATOR_FILE = "table_extraction_validation_second_annotator"

M = ru.MINUS
NA = "n/a"
MEASURES = [("name_precision", "NAME PRECISION"), ("category_precision", "CATEGORY PRECISION"),
            ("recall", "RECALL")]
SEXES = [("Female", "Women"), ("Male", "Men"), ("All", "All")]
AIDS = ["AID+", "AID-", "All"]
WIDE = {"first_width": 2.6, "pad": 1}


def workbook(name, numeric):
    """The workbook as {sheet: DataFrame}. numeric: {sheet: [columns]} that
    must hold a value in every row (a formula saved without its value reads
    as empty)."""
    path = os.path.join(SOURCE_DATA_DIR, name + ".xlsx")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path}: the validation workbook is missing")
    wb = pd.read_excel(path, sheet_name=None)
    for sheet, cols in numeric.items():
        if sheet not in wb:
            raise KeyError(f"{name}: sheet {sheet} missing")
        df = wb[sheet]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(f"{name}, {sheet}: columns missing {missing}")
        empty = [c for c in cols if df[c].isna().any()]
        if empty:
            raise ValueError(f"{name}, {sheet}: no saved value in {empty}; open the workbook in "
                             f"Excel or LibreOffice and save it, so its formula values are stored")
    return wb


def parameter(wb, key):
    p = wb["parameters"].set_index("parameter")["value"]
    if key not in p.index:
        raise KeyError(f"parameters: {key} missing")
    return p[key]


def pct_ci(est, low, high):
    """A percentage with its 95% CI, 1 decimal: "x.x (lo–hi)"."""
    return f"{ru.fmt_num(est, 1)} ({ru.fmt_num(low, 1)}{ru.DASH}{ru.fmt_num(high, 1)})"


def ac1_ci(est, low, high):
    return f"{ru.fmt_num(est, 3)} ({ru.fmt_num(low, 3)}{ru.DASH}{ru.fmt_num(high, 3)})"


def one(df, **keys):
    m = pd.Series(True, index=df.index)
    for k, v in keys.items():
        m &= df[k] == v
    if int(m.sum()) != 1:
        raise ValueError(f"expected one row for {keys}, found {int(m.sum())}")
    return df[m].iloc[0]


# =============================================================================
# PRECISION AND RECALL (PRIMARY ANNOTATOR)
# =============================================================================
def precision_table(cap):
    cols = ["correct", "total", "pct", "ci_low", "ci_high"]
    wb = workbook(PRECISION_FILE, {"overall": cols, "category_precision": cols,
                                   "parameters": ["value"]})
    if float(parameter(wb, "check_category_correct_sum")) != 0:
        raise ValueError("category_precision does not add up to the overall category precision")
    ov, cp = wb["overall"], wb["category_precision"]
    rows = [{"kind": "header", "label": "OVERALL"}]
    for m in ("name_precision", "category_precision", "recall"):
        r = one(ov, measure=m)
        rows.append({"kind": "data", "values": [r["row_label"],
                                                pct_ci(r["pct"], r["ci_low"], r["ci_high"])]})
    rows.append({"kind": "header", "label": "CATEGORY PRECISION BY CATEGORY"})
    for _, r in sorted(cp.iterrows(), key=lambda x: ru.sort_key(x[1]["row_label"])):
        rows.append({"kind": "data", "values": [r["row_label"],
                                                pct_ci(r["pct"], r["ci_low"], r["ci_high"])]})
    caps = ()
    if cap:
        t = cap["precision"]
        caps = [t["body"],
                ru.fill(t["sample"], {"per_category": int(parameter(wb, "n_per_category")),
                                      "n_categories": int(parameter(wb, "n_categories"))}),
                t["definitions"], t["ci"], t["order"],
                ru.fill(t["n"], {"n_extractions": ru.fmt_count(parameter(wb, "n_extractions")),
                                 "per_category": int(parameter(wb, "n_per_category")),
                                 "n_true": ru.fmt_count(parameter(wb, "n_true_symptoms")),
                                 "n_notes": int(parameter(wb, "n_recall_notes"))})]
    ru.write_table(os.path.join(TABLES_DIR, PRECISION_FILE + ".docx"),
                   ["Measure / category", "% (95% CI)"], rows,
                   cap["precision"]["title"] if cap else None, caps, **WIDE)


# =============================================================================
# BY AID STATUS AND SEX
# =============================================================================
def by_group_table(cap):
    wb = workbook(BY_GROUP_FILE, {"cells": ["correct", "total", "pct", "ci_low", "ci_high"],
                                  "tests": ["p_aid", "p_sex"]})
    cells, tests = wb["cells"], wb["tests"]
    rows = []
    for m, label in MEASURES:
        rows.append({"kind": "header", "label": label})
        t = one(tests, measure=m)
        for i, (sex, sex_label) in enumerate(SEXES):
            vals = [sex_label]
            for a in AIDS:
                r = one(cells, measure=m, sex=sex, aid=a)
                vals.append(pct_ci(r["pct"], r["ci_low"], r["ci_high"]))
            # one test per measure, on the block's first row
            vals += [ru.fmt_p(t["p_aid"]), ru.fmt_p(t["p_sex"])] if i == 0 else ["", ""]
            rows.append({"kind": "data", "values": vals})
    caps = ()
    if cap:
        t = cap["by_aid_sex"]

        def sizes(m):
            return {f"{a}_{s}": ru.fmt_count(one(cells, measure=m, sex=sx, aid=ax)["total"])
                    for ax, a in (("AID+", "pos"), ("AID-", "neg"))
                    for sx, s in (("Female", "female"), ("Male", "male"))}
        if sizes("name_precision") != sizes("category_precision"):
            raise ValueError("name and category precision have different group sizes")
        caps = [t["body"], t["tests"], t["ci"], t["pooled"],
                ru.fill(t["n_precision"], sizes("name_precision")),
                ru.fill(t["n_recall"], sizes("recall"))]
    headers = ["Sex", "AID+, % (95% CI)", f"AID{M}, % (95% CI)", "All patients, % (95% CI)",
               f"*P*, AID+ vs AID{M}", "*P*, women vs men"]
    ru.write_table(os.path.join(TABLES_DIR, BY_GROUP_FILE + ".docx"), headers, rows,
                   cap["by_aid_sex"]["title"] if cap else None, caps, first_width=1.2, pad=1)


# =============================================================================
# SECOND ANNOTATOR AND AGREEMENT
# =============================================================================
def annotator_table(cap):
    pcols = ["n", "primary_pct", "primary_ci_low", "primary_ci_high", "secondary_pct",
             "secondary_ci_low", "secondary_ci_high", "agreement_pct", "ac1", "ac1_ci_low",
             "ac1_ci_high"]
    rcols = ["identified_primary", "identified_secondary", "identified_both", "identified_either",
             "primary_pct", "primary_ci_low", "primary_ci_high", "secondary_pct",
             "secondary_ci_low", "secondary_ci_high", "positive_specific_agreement_pct",
             "jaccard_pct"]
    wb = workbook(ANNOTATOR_FILE, {"precision": pcols, "recall": rcols, "parameters": ["value"]})
    pr, rc = wb["precision"], wb["recall"]
    if pr["n"].nunique() != 1:
        raise ValueError("name and category precision were judged on different numbers of items")
    rows = []
    for m in ("name_precision", "category_precision"):
        r = one(pr, measure=m)
        rows.append({"kind": "data", "values": [
            r["row_label"], pct_ci(r["primary_pct"], r["primary_ci_low"], r["primary_ci_high"]),
            pct_ci(r["secondary_pct"], r["secondary_ci_low"], r["secondary_ci_high"]),
            ru.fmt_num(r["agreement_pct"], 1), NA,
            ac1_ci(r["ac1"], r["ac1_ci_low"], r["ac1_ci_high"])]})
    if len(rc) != 1:
        raise ValueError("recall: expected one row")
    r = rc.iloc[0]
    rows.append({"kind": "data", "values": [
        r["row_label"], pct_ci(r["primary_pct"], r["primary_ci_low"], r["primary_ci_high"]),
        pct_ci(r["secondary_pct"], r["secondary_ci_low"], r["secondary_ci_high"]),
        ru.fmt_num(r["positive_specific_agreement_pct"], 1), ru.fmt_num(r["jaccard_pct"], 1), NA]})
    caps = ()
    if cap:
        t = cap["second_annotator"]
        caps = [t["body"],
                ru.fill(t["sample"], {"per_category": int(parameter(wb, "n_per_category")),
                                      "n_notes": int(parameter(wb, "n_recall_notes")),
                                      "n_notes_all": int(parameter(wb,
                                                                   "n_recall_notes_primary_sample"))}),
                t["agreement"], t["jaccard"], t["ac1"], t["ci"],
                ru.fill(t["n"], {"n": ru.fmt_count(pr["n"].iloc[0]),
                                 "primary": ru.fmt_count(r["identified_primary"]),
                                 "secondary": ru.fmt_count(r["identified_secondary"]),
                                 "both": ru.fmt_count(r["identified_both"]),
                                 "either": ru.fmt_count(r["identified_either"])})]
    headers = ["Measure", "Primary annotator, % (95% CI)", "Secondary annotator, % (95% CI)",
               "Agreement, %", "Jaccard index, %", "Gwet’s AC1 (95% CI)"]
    ru.write_table(os.path.join(TABLES_DIR, ANNOTATOR_FILE + ".docx"), headers, rows,
                   cap["second_annotator"]["title"] if cap else None, caps,
                   first_width=1.6, pad=1)


def main():
    cap = ru.load_captions(CAPTIONS_PATH, __file__)
    print(f"  Titles and captions: {'written' if cap else 'none (no caption file set)'}")
    precision_table(cap)
    by_group_table(cap)
    annotator_table(cap)


if __name__ == "__main__":
    main()