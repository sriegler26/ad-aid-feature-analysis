"""
symptom_prevalence_tables.py
============================
Renders the four symptom prevalence tables (.docx) from the source-data
workbooks written by symptom_prevalence.py. Runs no
model and computes no statistic: every number comes from the workbooks and
is formatted once by render_utils.

Tables (TABLES_DIR)
-------------------
  table_symptom_prevalence_weighted.docx         domains and categories, IPW-weighted
  table_symptom_prevalence_unweighted.docx       domains and categories, unweighted
  table_known_sex_differences.docx               does AID change a known sex difference
  table_symptom_prevalence_cns_excluded.docx     domains after excluding CNS-primary AID

Columns: the results (odds ratio with its 95% CI inline, Q) first, then the
number of patients with the symptom documented in each group (documented, n;
the group sizes are in the caption). Every domain and category is shown with its
counts; a row without a test shows "–". The categories come before the
domains; within each, rows are alphabetical by their readable label, Other
last. The positive control keeps its alphabetical place, marked; it is
not in the FDR family, so its Q cell reads "NA (P ...)" with its unadjusted P
value. The known sex differences table gives Q only (no P column), for every
row: all patients, AID+, AID- and the ratio.

Captions: the titles and caption wording are read from an optional caption
file (CAPTIONS_PATH); this script fills in the numbers and picks the
sentences that apply. With no file set, the tables are written without them.

Run from the data folder, after symptom_prevalence.py.
"""

import os

import pandas as pd

import render_utils as ru

# =============================================================================
# PATHS (relative to the data folder the script is run from)
# =============================================================================
SOURCE_DATA_DIR = ""  # folder of the source-data workbooks (.xlsx) written by symptom_prevalence.py
TABLES_DIR = ""  # folder for the .docx tables written here
CAPTIONS_PATH = ""  # optional JSON with titles and captions; blank = bare tables

WEIGHTED_FILE = "table_symptom_prevalence_weighted"
UNWEIGHTED_FILE = "table_symptom_prevalence_unweighted"
SEX_DIFFERENCES_FILE = "table_known_sex_differences"
CNS_FILE = "table_symptom_prevalence_cns_excluded"

MARK = "†"
M = ru.MINUS
ARM_CELLS = ["events_aid_pos", "events_aid_neg"]
ARM_HEADERS = ["AID+\ndocumented, *n*", f"AID{M}\ndocumented, *n*"]
RESULT_HEADERS = ["Odds ratio\n(95% CI)", "*Q*"]


def workbook(name):
    path = os.path.join(SOURCE_DATA_DIR, name + ".xlsx")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path}: run symptom_prevalence.py first")
    return pd.read_excel(path, sheet_name=None)


def parameter(wb, key):
    p = wb["parameters"].set_index("parameter")["value"]
    if key not in p.index:
        raise KeyError(f"parameters: {key} missing")
    return p[key]


def has_parameter(wb, key):
    return key in set(wb["parameters"]["parameter"])


def first(df, col):
    return int(df[col].iloc[0])


# =============================================================================
# CELLS
# =============================================================================
def label(r):
    return r["row_label"] + (MARK if r["fdr_role"] == "positive_control" else "")


def result_cells(r):
    """Odds ratio with its CI inline, and the Q column: Q for tested rows,
    "NA (P ...)" with the unadjusted P for the positive control, "–" for rows
    not fitted or not tested."""
    if r["status"] != "fitted":
        return [ru.DASH, ru.DASH]
    if r["fdr_role"] == "positive_control":
        q = f"NA ({ru.p_phrase('P', r['p'])})"
    elif r["fdr_role"] == "not_tested":
        q = ru.DASH
    else:
        q = ru.fmt_p(r["q"])
    return [ru.fmt_ratio_ci(r["odds_ratio"], r["ci_low"], r["ci_high"]), q]


def model_rows(df, cells):
    rows = []
    for _, r in sorted(df.iterrows(), key=lambda x: ru.sort_key(x[1]["row_label"])):
        rows.append({"kind": "data", "values": [label(r)] + result_cells(r)
                     + [ru.fmt_count(r[c]) for c in cells]})
    return rows


def prevalence_rows(wb, cells):
    return ([{"kind": "header", "label": "Categories"}] + model_rows(wb["categories"], cells)
            + [{"kind": "header", "label": "Domains"}] + model_rows(wb["domains"], cells))


# =============================================================================
# CAPTION PIECES
# =============================================================================
def group_values(wb):
    d = wb["domains"]
    return {k: ru.fmt_count(first(d, k)) for k in
            ("n_aid_pos", "n_aid_neg", "n_aid_pos_female", "n_aid_pos_male",
             "n_aid_neg_female", "n_aid_neg_male")}


def fdr_sentence(cap, wb):
    """The Q sentence for the families this table has; a family with no tested
    row gets its own wording instead of "0 tested"."""
    c = cap["common"]
    n_dom = int(parameter(wb, "fdr_family_size_domains"))
    n_cat = (int(parameter(wb, "fdr_family_size_categories"))
             if has_parameter(wb, "fdr_family_size_categories") else None)
    v = {"family_domains": ru.fmt_count(n_dom),
         "family_categories": ru.fmt_count(n_cat or 0)}
    if n_cat is None:
        key = "fdr_domains" if n_dom else "fdr_none_domains"
    elif n_dom and n_cat:
        key = "fdr_both"
    elif n_dom:
        key = "fdr_domains_no_categories"
    elif n_cat:
        key = "fdr_categories_no_domains"
    else:
        key = "fdr_none"
    return ru.fill(c[key], v)


def secondary_sentence(cap, sheet, measure, wb):
    """One sentence when the secondary domain definition moves an odds ratio by
    at least the smaller threshold or moves Q across the threshold; else none."""
    if sheet.empty:
        return None
    q_thr = parameter(wb, "q_threshold")
    hit = sheet[(sheet["or_change_ge_small"] == 1) | (sheet["became_significant"] == 1)
                | (sheet["lost_significance"] == 1)]
    if hit.empty:
        return None
    items = []
    for _, r in sorted(hit.iterrows(), key=lambda x: ru.sort_key(x[1]["row_label"])):
        crossed = (r["became_significant"] == 1) or (r["lost_significance"] == 1)
        items.append(ru.fill(cap["common"]["secondary_item"], {
            "measure": measure, "domain": r["row_label"],
            "or_primary": ru.fmt_ratio(r["odds_ratio_primary"]),
            "or_secondary": ru.fmt_ratio(r["odds_ratio_secondary"]),
            "q_primary": ru.p_phrase("Q", r["q_primary"]),
            "q_secondary": ru.p_phrase("Q", r["q_secondary"]),
            "crossed": ru.fill(cap["common"]["secondary_crossed"],
                               {"q_threshold": ru.fmt_p(q_thr)}) if crossed else ""}))
    return ru.fill(cap["common"]["secondary_definition"], {"changes": "; ".join(items)})


def prevalence_captions(cap, key, wb):
    """Sentences in order: what the table shows (with group sizes), the model,
    the ceiling, the symbols and the not-tested legend, then the secondary-
    definition sentence when it applies."""
    c, t = cap["common"], cap[key]
    v = group_values(wb)
    v.update(min_events=ru.fmt_count(parameter(wb, "min_events_per_term")),
             min_group=ru.fmt_count(parameter(wb, "min_group_n")),
             weighting=c["weighting"], ceiling=c["age_ceiling"])
    parts = [t["body"], c["documented"], c["ci"], fdr_sentence(cap, wb),
             c["positive_control"], c["other"], t["legend"]]
    parts = [ru.fill(p, v) for p in parts]
    s = secondary_sentence(cap, wb["secondary_definition"], t["measure"], wb)
    return parts + ([s] if s else [])


# =============================================================================
# TABLES
# =============================================================================
def prevalence_table(cap, key, name, cells, headers):
    wb = workbook(name)
    write(name, ["Domain or category (domain)"] + RESULT_HEADERS + headers,
          prevalence_rows(wb, cells), cap, key,
          (lambda: prevalence_captions(cap, key, wb)))


def sex_differences_table(cap):
    wb = workbook(SEX_DIFFERENCES_FILE)
    sd = wb["sex_differences"]
    rows = []
    group_text = {"all_patients": "All patients, women vs men",
                  "aid_pos": "AID+ patients, women vs men",
                  "aid_neg": f"AID{M} patients, women vs men",
                  "ratio": f"Ratio, AID+ to AID{M}"}
    for code in sorted(sd["code"].unique(), key=lambda c: ru.sort_key(
            sd.loc[sd["code"] == c, "row_label"].iloc[0])):
        g = sd[sd["code"] == code].set_index("contrast")
        higher = {"Female": "women", "Male": "men"}[g["expected_higher"].iloc[0]]
        rows.append({"kind": "header",
                     "label": f"{g['row_label'].iloc[0]} (reported to be more common in {higher})"})
        for contrast in ("all_patients", "aid_pos", "aid_neg", "ratio"):
            r = g.loc[contrast]
            fitted = r["status"] == "fitted"
            is_ratio = contrast == "ratio"
            rows.append({"kind": "data", "values": [
                group_text[contrast],
                ru.fmt_ratio_ci(r["odds_ratio"], r["ci_low"], r["ci_high"]) if fitted
                else ru.DASH,
                ru.fmt_p(r["q"]) if fitted else ru.DASH,
                "n/a" if is_ratio else ru.fmt_count(r["events_female"]),
                "n/a" if is_ratio else ru.fmt_count(r["events_male"])]})

    def captions():
        c, t = cap["common"], cap["sex_differences"]
        v = {}
        for contrast, a in (("aid_pos", "aid_pos"), ("aid_neg", "aid_neg")):
            r = sd[sd["contrast"] == contrast].iloc[0]
            v[f"n_{a}_female"] = ru.fmt_count(r["n_female"])
            v[f"n_{a}_male"] = ru.fmt_count(r["n_male"])
        v.update(min_events=ru.fmt_count(parameter(wb, "min_events_per_term")),
                 min_group=ru.fmt_count(parameter(wb, "min_group_n")),
                 q_threshold=ru.fmt_p(parameter(wb, "q_threshold")),
                 weighting=c["weighting"], ceiling=c["age_ceiling"])
        tested = sd[sd["q"].notna()]
        q_key = "q" if len(tested) else "q_none"
        parts = [t["body"], t["documented"], c["ci"], t[q_key]]
        if len(tested):
            names = {}
            for fam in ("all_patients", "aid_neg"):
                rep = sd[(sd["contrast"] == fam) & (sd["reproduced"] == 1)]
                labels = sorted(rep["row_label"], key=ru.sort_key)
                names[fam] = (ru.join_and([l.lower() for l in labels]) if labels
                              else t["reproduced_none"])
            parts.append(ru.fill(t["reproduced"], {**v, "all_patients": names["all_patients"],
                                                   "aid_neg": names["aid_neg"]}))
        parts.append(t["legend"])
        return [ru.fill(p, v) for p in parts]
    write(SEX_DIFFERENCES_FILE,
          ["Symptom and comparison", "Odds ratio\n(95% CI)", "*Q*",
           "Women\ndocumented, *n*", "Men\ndocumented, *n*"], rows, cap, "sex_differences",
          captions)


def cns_table(cap):
    wb = workbook(CNS_FILE)
    d = wb["domains"]
    rows = model_rows(d, ARM_CELLS)

    def captions():
        c, t = cap["common"], cap["cns"]
        v = group_values(wb)
        v.update(n_excluded=ru.fmt_count(parameter(wb, "n_excluded_cns_primary")),
                 min_events=ru.fmt_count(parameter(wb, "min_events_per_term")),
                 weighting=c["weighting"], ceiling=c["age_ceiling"])
        b = wb["balance"]
        allb = b[b["stratum"] == "All"].set_index("variable")
        v.update(smd_age=ru.fmt_coef(allb.loc["age_at_entry", "smd_weighted"]),
                 smd_notes=ru.fmt_coef(allb.loc["log_note_count_predx", "smd_weighted"]),
                 smd_age_all=ru.fmt_coef(allb.loc["age_at_entry", "smd_weighted_all_patients"]),
                 smd_notes_all=ru.fmt_coef(allb.loc["log_note_count_predx",
                                                    "smd_weighted_all_patients"]))
        parts = [t["body"], t["balance"], t["documented"], c["ci"],
                 fdr_sentence(cap, wb), t["positive_control"], t["legend"]]
        parts = [ru.fill(p, v) for p in parts]
        s = secondary_sentence(cap, wb["secondary_definition"], t["measure"], wb)
        return parts + ([s] if s else [])
    write(CNS_FILE, ["Domain"] + RESULT_HEADERS + ARM_HEADERS, rows, cap, "cns", captions)


def write(name, headers, rows, cap, key, captions):
    title = cap[key]["title"] if cap else None
    ru.write_table(os.path.join(TABLES_DIR, name + ".docx"), headers, rows, title,
                   captions() if cap else ())


def main():
    cap = ru.load_captions(CAPTIONS_PATH, __file__)
    print(f"  Titles and captions: {'written' if cap else 'none (no caption file set)'}")
    prevalence_table(cap, "weighted", WEIGHTED_FILE, ARM_CELLS, ARM_HEADERS)
    prevalence_table(cap, "unweighted", UNWEIGHTED_FILE, ARM_CELLS, ARM_HEADERS)
    sex_differences_table(cap)
    cns_table(cap)


if __name__ == "__main__":
    main()