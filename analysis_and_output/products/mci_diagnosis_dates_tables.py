"""
mci_diagnosis_dates_tables.py
=============================
Renders the table of coded mild cognitive impairment (MCI) and Alzheimer's
disease (AD) diagnosis dates (.docx) from the source-data workbook written by
mci_diagnosis_dates.py. Fits no model and computes no statistic: every number
comes from the workbook, formatted once by render_utils.

Rows: all patients, then AID+ and AID- within female and within male patients.
Columns: the gap (AD minus MCI date) as median (95% CI) and Q1-Q3, the sign
test and Wilcoxon P, then the number of patients with a dated MCI diagnosis.
Captions: the titles and caption wording are read from an optional caption
file (CAPTIONS_PATH); this script fills in the numbers. With no file set, the
table is written without them.

Run from the data folder, after mci_diagnosis_dates.py.
"""

import os

import pandas as pd

import render_utils as ru

# =============================================================================
# PATHS (relative to the data folder the script is run from)
# =============================================================================
SOURCE_DATA_DIR = ""  # folder of the source-data workbooks (.xlsx) written by mci_diagnosis_dates.py
TABLES_DIR = ""  # folder for the .docx tables written here
CAPTIONS_PATH = ""  # optional JSON with titles and captions; blank = bare tables

TABLE_FILE = "table_mci_ad_diagnosis_dates"
M = ru.MINUS
LABELS = {"all": "All patients", "aid_pos_female": "AID+ women",
          "aid_neg_female": f"AID{M} women", "aid_pos_male": "AID+ men",
          "aid_neg_male": f"AID{M} men"}


def main():
    cap = ru.load_captions(CAPTIONS_PATH, __file__)
    print(f"  Titles and captions: {'written' if cap else 'none (no caption file set)'}")
    path = os.path.join(SOURCE_DATA_DIR, TABLE_FILE + ".xlsx")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path}: run mci_diagnosis_dates.py first")
    wb = pd.read_excel(path, sheet_name=None)
    g = wb["groups"]
    p = wb["parameters"].set_index("parameter")["value"]
    rows = []
    for _, r in g.iterrows():
        tested = r["status"] == "tested"
        has = r["n_mci_dated"] > 0
        rows.append({"kind": "data", "values": [
            LABELS[r["group"]],
            ru.fmt_coef_ci(r["gap_median_yrs"], r["gap_median_ci_low"], r["gap_median_ci_high"])
            if has else ru.DASH,
            f"{ru.fmt_num(r['gap_q1_yrs'], 2)} to {ru.fmt_num(r['gap_q3_yrs'], 2)}" if has
            else ru.DASH,
            ru.fmt_p(r["p_sign"]) if tested else ru.DASH,
            ru.fmt_p(r["p_wilcoxon"]) if tested else ru.DASH,
            ru.fmt_count(r["n_mci_dated"])]})
    headers = ["Group", "Gap, median\n(95% CI), years", "Gap, Q1 to Q3,\nyears",
               "*P*, sign test", "*P*, Wilcoxon", "MCI diagnosis\nwith date, *n*"]
    caps = ()
    if cap:
        t = cap["table"]
        caps = [t["body"], t["mci"], t["gap"], t["tests"], t["comparisons"],
                ru.fill(t["dash"], {"min_group": int(p["min_group_n"])}), t["ne"],
                ru.fill(t["undated"], {"n_undated": ru.fmt_count(p["n_mci_coded_undated"])}),
                t["abbreviations"]]
    ru.write_table(os.path.join(TABLES_DIR, TABLE_FILE + ".docx"), headers, rows,
                   cap["table"]["title"] if cap else None, caps, first_width=1.3, pad=0)


if __name__ == "__main__":
    main()