"""
symptom_timing_sequence_tables.py
=================================
Renders the symptom timing and sequence tables (.docx) from the source-data
workbooks written by symptom_timing_sequence.py. Fits no model and computes
no statistic: every number comes from the workbooks, formatted once by
render_utils.

Tables
------
- Age at first documentation: the AID+ minus AID- difference in years per
  category and per domain (95% CI), Q, then documented n for AID+ and AID-;
  a Categories section, then a Domains section (each its own FDR family).
- Sequence inversion: per expected-order pair (patients with at least 5
  years of notes before the AD diagnosis), the share inverted in AID+ and in
  AID- (% (inverted/both documented)), the Fisher odds ratio (exact 95% CI)
  and Q.
- Sequence inversion by sex: the same, within female and within male
  patients (one group of rows per sex).
Every CI is inline after its estimate, in parentheses (no CI column). Every
row is shown; "–" marks a model or test not run; an odds ratio of 0
or infinity is printed "not estimable" (its Q stands). Rows are alphabetical
by their readable label; the positive control keeps its place, marked,
with its P value in the Q column.

Captions: the titles and caption wording are read from an optional caption
file (CAPTIONS_PATH); this script fills in the numbers and picks the
sentences that apply. With no file set, the tables are written without them.

Run from the data folder, after symptom_timing_sequence.py.
"""

import os

import pandas as pd

import render_utils as ru

# =============================================================================
# PATHS (relative to the data folder the script is run from)
# =============================================================================
SOURCE_DATA_DIR = ""  # folder of the source-data workbooks (.xlsx) written by symptom_timing_sequence.py
TABLES_DIR = ""  # folder for the .docx tables written here
CAPTIONS_PATH = ""  # optional JSON with titles and captions; blank = bare tables

FIRST_AGE_FILE = "table_age_at_first_documentation"
INVERSION_FILE = "table_symptom_sequence_inversion"
INVERSION_BY_SEX_FILE = "table_symptom_sequence_inversion_by_sex"

MARK = "†"
M = ru.MINUS
ARM_HEADERS = ["AID+\ndocumented, *n*", f"AID{M}\ndocumented, *n*"]


def workbook(name):
    path = os.path.join(SOURCE_DATA_DIR, name + ".xlsx")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path}: run symptom_timing_sequence.py first")
    return pd.read_excel(path, sheet_name=None)


def parameter(wb, key):
    p = wb["parameters"].set_index("parameter")["value"]
    if key not in p.index:
        raise KeyError(f"parameters: {key} missing")
    return p[key]


def group_sizes(wb):
    return {k: ru.fmt_count(parameter(wb, k)) for k in
            ("n_aid_pos", "n_aid_neg", "n_aid_pos_female", "n_aid_pos_male",
             "n_aid_neg_female", "n_aid_neg_male")}


# =============================================================================
# CELLS
# =============================================================================
def label(r):
    return r["row_label"] + (MARK if r["fdr_role"] == "positive_control" else "")


def q_cell(r):
    """Q for tested rows, the unadjusted P for the positive control, "–" for
    rows not fitted."""
    if r["status"] != "fitted":
        return ru.DASH
    if r["fdr_role"] == "positive_control":
        return ru.fmt_p(r["p"])
    return ru.fmt_p(r["q"])


def coef_cells(r):
    if r["status"] != "fitted":
        return [ru.DASH, ru.DASH]
    return [ru.fmt_coef_ci(r["coef"], r["ci_low"], r["ci_high"]), q_cell(r)]


NOT_ESTIMABLE = "not estimable"
SEXES = [("female", "Women"), ("male", "Men")]


def fisher_cells(r):
    """Odds ratio (95% CI) and Q of a Fisher row; "–" when not tested."""
    if r["status"] != "tested":
        return [ru.DASH, ru.DASH]
    orv = (ru.fmt_ratio_ci(r["odds_ratio"], r["ci_low"], r["ci_high"])
           if int(r["or_estimable"]) else NOT_ESTIMABLE)
    return [orv, ru.fmt_p(r["q"])]


def share_cell(k, n):
    """"x.x% (k/n)"; "–" for an empty group."""
    if pd.isna(n) or int(n) == 0:
        return ru.DASH
    return f"{ru.fmt_pct(100 * float(k) / float(n))} ({ru.fmt_count_of(k, n)})"


def sorted_rows(df):
    return df.assign(_k=[ru.sort_key(x) for x in df["row_label"]]).sort_values("_k")


# =============================================================================
# TABLES
# =============================================================================
def first_age_rows(df):
    rows = []
    for _, r in sorted_rows(df).iterrows():
        counts = [ru.fmt_count(r["n_aid_pos"]), ru.fmt_count(r["n_aid_neg"])]
        rows.append({"kind": "data", "values": [label(r)] + coef_cells(r) + counts})
    return rows


def first_age_sections(wb):
    """Categories, then domains, as in the weighted prevalence table."""
    return ([{"kind": "header", "label": "Categories"}] + first_age_rows(wb["categories"])
            + [{"kind": "header", "label": "Domains"}] + first_age_rows(wb["domains"]))


def inversion_data_rows(f):
    return [{"kind": "data", "values": [r["row_label"]]
             + [share_cell(r["inverted_aid_pos"], r["n_aid_pos"]),
                share_cell(r["inverted_aid_neg"], r["n_aid_neg"])] + fisher_cells(r)}
            for _, r in sorted_rows(f).iterrows()]


def inversion_rows(df, by_sex):
    """All patients: the pair rows; by sex: a group of rows per sex."""
    if not by_sex:
        return inversion_data_rows(df)
    rows = []
    for sx, words in SEXES:
        g = df[df["sex"] == sx]
        if g.empty:
            continue
        rows.append({"kind": "header", "label": words})
        rows += inversion_data_rows(g)
    return rows


# =============================================================================
# CAPTIONS
# =============================================================================
def family_text(cap, key, size, values):
    return ru.fill(cap[key]["fdr"] if size else cap[key]["fdr_none"], values)


def secondary_sentence(cap, wb, key):
    sd = wb["secondary_definition"]
    if sd.empty:
        return []
    hits = sd[(sd["became_significant"] == 1) | (sd["lost_significance"] == 1)
              | (sd["change_ge_small"] == 1)]
    if hits.empty:
        return []
    q_thr = parameter(wb, "q_threshold")
    items = []
    for _, r in hits.sort_values("row_label").iterrows():
        crossed = (ru.fill(cap["common"]["secondary_crossed"], {"q_threshold": q_thr})
                   if r["became_significant"] or r["lost_significance"] else "")
        items.append(ru.fill(cap["common"]["secondary_item"], {
            "measure": cap[key]["measure"], "domain": r["row_label"],
            "est_primary": ru.fmt_coef(r["coef_primary"]),
            "est_secondary": ru.fmt_coef(r["coef_secondary"]),
            "q_primary": ru.p_phrase("Q", r["q_primary"]),
            "q_secondary": ru.p_phrase("Q", r["q_secondary"]), "crossed": crossed}))
    return [ru.fill(cap["common"]["secondary_definition"], {"changes": ru.join_and(items)})]


def level_family_text(cap, key, n_dom, n_cat):
    """The Q sentence for the two families (domains, categories); a family
    with no tested row gets its own wording instead of "0 tested"."""
    t = cap[key]
    k = ("fdr_both" if n_dom and n_cat else "fdr_domains_no_categories" if n_dom
         else "fdr_categories_no_domains" if n_cat else "fdr_none")
    return ru.fill(t[k], {"family_domains": ru.fmt_count(n_dom),
                          "family_categories": ru.fmt_count(n_cat)})


def control_fitted(df):
    return ((df["fdr_role"] == "positive_control") & (df["status"] == "fitted")).any()


def first_age_captions(cap, wb, key):
    c, sizes = cap["common"], group_sizes(wb)
    out = [ru.fill(cap[key]["body"], {**sizes, "span": c["span"],
                                      "first_documentation": c["first_documentation"]}),
           cap[key]["documented"], cap[key]["other"], c["ci"],
           level_family_text(cap, key, int(parameter(wb, "fdr_family_size_domains")),
                             int(parameter(wb, "fdr_family_size_categories")))]
    if control_fitted(wb["categories"]):
        out.append(c["positive_control_categories"])
    elif control_fitted(wb["domains"]):
        out.append(c["positive_control"])
    runs = (int(parameter(wb, "linearity_models_run"))
            + int(parameter(wb, "linearity_models_run_categories")))
    if runs:
        largest = max(v for v in (parameter(wb, "linearity_largest_shift_years"),
                                  parameter(wb, "linearity_largest_shift_years_categories"))
                      if pd.notna(v))
        out.append(ru.fill(c["linearity"], {"measure": cap[key]["measure"],
                                            "largest": ru.fmt_coef(largest)}))
    out += secondary_sentence(cap, wb, key)
    out.append(ru.fill(cap[key]["legend"], {
        "min_events": int(parameter(wb, "min_events_per_term")),
        "min_group": int(parameter(wb, "min_group_n"))}))
    return out


def inversion_captions(cap, wb, key, by_sex):
    c, sizes = cap["common"], group_sizes(wb)
    fam = ({f"family_{sx}": int(parameter(wb, f"fdr_family_size_{sx}")) for sx, _ in SEXES}
           if by_sex else {"family": int(parameter(wb, "fdr_family_size"))})
    pair_sample = ru.fill(c["pair_sample"], {"yrs": int(parameter(wb, "predx_span_floor_yrs"))})
    return [ru.fill(cap[key]["body"], {**sizes, "pairs": c["pairs"], "pair_sample": pair_sample}),
            cap[key]["counts"], c["ci"],
            family_text(cap, key, sum(fam.values()), fam),
            ru.fill(cap[key]["legend"], {"min_group": int(parameter(wb, "min_group_n"))})] \
        + ([cap[key]["not_estimable"]] if not_estimable(wb["pairs"]).any() else [])


def not_estimable(df):
    return (df["status"] == "tested") & (df["or_estimable"] == 0)


# =============================================================================
# MAIN
# =============================================================================
def main():
    cap = ru.load_captions(CAPTIONS_PATH, __file__)

    def title_of(key):
        return cap[key]["title"] if cap else None

    wb = workbook(FIRST_AGE_FILE)
    headers = ["Domain or category (domain)", "Difference, years\n(95% CI)", "*Q*"] + ARM_HEADERS
    ru.write_table(os.path.join(TABLES_DIR, FIRST_AGE_FILE + ".docx"), headers,
                   first_age_sections(wb), title_of("first_age"),
                   first_age_captions(cap, wb, "first_age") if cap else ())

    headers = ["Expected order", "AID+ inverted,\n% (*n*/*N*)", f"AID{M} inverted,\n% (*n*/*N*)",
               "Odds ratio\n(95% CI)", "*Q*"]
    for key, name, by_sex in (("inversion", INVERSION_FILE, False),
                              ("inversion_by_sex", INVERSION_BY_SEX_FILE, True)):
        wb = workbook(name)
        ru.write_table(os.path.join(TABLES_DIR, name + ".docx"), headers,
                       inversion_rows(wb["pairs"], by_sex), title_of(key),
                       inversion_captions(cap, wb, key, by_sex) if cap else ())


if __name__ == "__main__":
    main()