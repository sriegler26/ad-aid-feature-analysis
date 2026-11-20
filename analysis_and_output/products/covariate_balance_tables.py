"""
covariate_balance_tables.py
===========================
Renders the covariate balance tables (.docx) from the source-data workbooks
written by propensity_score.py: the standardized mean difference (SMD) of
each covariate between AID+ and AID- before and after propensity-score
weighting and its change, for all patients and within each sex. The tables
give the precise values behind the propensity-score balance figure.
Computes no statistic: every number comes from the workbooks, formatted once
by render_utils.

Rows are alphabetical by their readable label; the two covariates of
the propensity-score model are marked with a dagger. Group sizes are in the
caption. Female before male.

Captions: the titles and caption wording are read from an optional caption
file (CAPTIONS_PATH); with no file set, the tables are written without them.

Run from the data folder, after propensity_score.py.
"""

import os

import pandas as pd

import render_utils as ru

# =============================================================================
# PATHS (relative to the data folder the script is run from)
# =============================================================================
SOURCE_DATA_DIR = ""  # folder with the propensity_score.py workbooks (table_covariate_balance_weighting.xlsx, table_covariate_balance_weighting_by_sex.xlsx)
TABLES_DIR = ""  # folder for the .docx tables written here
CAPTIONS_PATH = ""  # optional JSON with titles and captions; blank = bare tables

ALL_PATIENTS = "table_covariate_balance_weighting"
BY_SEX = "table_covariate_balance_weighting_by_sex"
BY_DESIGN = "Balanced by design"
MARK = "†"
# Covariate labels as shown in the tables (the workbooks keep their own wording)
SHOWN_LABEL = {"Female sex": "Sex (women)"}
CAPTIONS = ru.load_captions(CAPTIONS_PATH, __file__)


def read(name, sheet):
    return pd.read_excel(os.path.join(SOURCE_DATA_DIR, name + ".xlsx"), sheet_name=sheet)


def label(r):
    return SHOWN_LABEL.get(r["row_label"], r["row_label"]) + (MARK if r["role"] == BY_DESIGN else "")


def cells(r):
    return [ru.fmt_smd(r["smd_unweighted"]), ru.fmt_smd(r["smd_weighted"]),
            ru.fmt_smd(r["smd_change"])]


def ordered(df):
    return sorted((r for _, r in df.iterrows()), key=lambda r: ru.sort_key(label(r)))


def caption(key, values):
    if not CAPTIONS:
        return None, ()
    c, common = CAPTIONS[key], CAPTIONS["common"]
    return c["title"], [ru.fill(c["body"], values), common["smd"], common["weights"],
                        common["rows"], c.get("extra", ""), common["threshold"],
                        common["figure"]]


def main():
    a = read(ALL_PATIENTS, "all_patients")
    rows = [{"kind": "data", "values": [label(r)] + cells(r)} for r in ordered(a)]
    title, cap = caption("all_patients", {"n_aid_pos": ru.fmt_count(a["n_aid_pos"].iloc[0]),
                                          "n_aid_neg": ru.fmt_count(a["n_aid_neg"].iloc[0])})
    ru.write_table(os.path.join(TABLES_DIR, ALL_PATIENTS + ".docx"),
                   ["Covariate", "SMD, unweighted", "SMD, weighted", "Change"], rows, title, [s for s in cap if s])

    s = read(BY_SEX, "by_sex")
    f, m = ordered(s[s["sex"] == "Female"]), ordered(s[s["sex"] == "Male"])
    if [label(r) for r in f] != [label(r) for r in m]:
        raise RuntimeError("the two sex strata have different rows")
    rows = [{"kind": "data", "values": [label(rf)] + cells(rf) + cells(rm)}
            for rf, rm in zip(f, m)]
    n = lambda sx, c: ru.fmt_count(s.loc[s["sex"] == sx, c].iloc[0])
    title, cap = caption("by_sex", {"n_aid_pos_female": n("Female", "n_aid_pos"),
                                    "n_aid_neg_female": n("Female", "n_aid_neg"),
                                    "n_aid_pos_male": n("Male", "n_aid_pos"),
                                    "n_aid_neg_male": n("Male", "n_aid_neg")})
    ru.write_table(os.path.join(TABLES_DIR, BY_SEX + ".docx"),
                   ["Covariate"] + [f"{sx}, {h}" for sx in ("Women", "Men")
                                    for h in ("SMD, unweighted", "SMD, weighted", "change")],
                   rows, title, [x for x in cap if x])


if __name__ == "__main__":
    main()