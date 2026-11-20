"""
cognitive_score_comparison.py
=============================
Does the annual change in MMSE and MoCA scores differ between patients with
and without autoimmune disease (AID), and does exposure to a medication used
to treat autoimmune disease change the AID difference?

Computes every statistic and writes the source data for the two cognitive
score tables (one workbook per table) and a checks workbook. The tables are
rendered by cognitive_score_comparison_tables.py; nothing here formats a
number.

Scores
------
Used here: absolute MMSE and MoCA scores with a value, of high or medium
confidence, not a change or sub-score row, with a measurement date
(score_date). Age at the score = (score_date - birth_date) in whole days /
365.25. The rows removed by each filter are printed every run. The filters
and the annual change below are analysis_utils.absolute_cognitive_scores and
analysis_utils.annual_change, shared with inflammation_tier_sensitivity.py.

Analyses
--------
- Scored vs unscored patients: the documentation measures (age at first note,
  documentation span, total, pre- and post-diagnosis note counts), Mann-Whitney
  U within each AID x sex group (checks).
- Annual change: per patient, (last - first score) / years between them,
  for patients with two or more scores at least 0.25 years apart; OLS
  slope ~ aid_status + age_c + female, HC3 (age_c = age at the first score,
  centered).
  P values are not corrected for multiple comparisons (no FDR family: two
  scores).
- Medication: each of the four classes used to treat
  autoimmune disease (corticosteroids, conventional DMARDs, antimalarials,
  biologics), on annual change. Models A (the annual-change model), B
  (+ exposed) and C (+ exposed + aid_status:exposed), HC3. Exposed: a dated
  record of the class before the patient's last score. Undated records do
  not count. The Model C interaction is the table's result; Q:
  Benjamini-Hochberg within each class over its tested rows (2 scores).
  The same models with exposure at any time, dated or not, are a check (no Q).
- Annual change by group and Mann-Whitney U within sex (checks).
Gate: analysis_utils.model_floor_reason, continuous outcomes (10 patients per
term; both AID groups and both sexes). Medication interaction: also 10
patients in each AID x exposure group.
Linearity: the centered age squared added; flagged at a shift of 0.5
points per year; for the medication models, on the Model C interaction.

Inputs: the patient table, the note table, the cognitive score table and the
medication exposure table (see PATHS).
"""

import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import mannwhitneyu, norm, t as t_dist

import analysis_utils as au

warnings.filterwarnings("ignore")   # after the imports: statsmodels resets its own filters

# =============================================================================
# PATHS (relative to the data folder the script is run from)
# =============================================================================
COHORT_PATIENTS_PATH = ""  # patient table: AID status, sex, key dates (birth, AD diagnosis, death) and analysis-population flag
COHORT_NOTES_PATH = ""  # note table: one row per clinical note, with its date and specialty
COGNITIVE_SCORES_PATH = ""  # cognitive score table: one row per documented score, with its type, value, date and confidence
MEDICATION_RECORDS_PATH = ""  # medication exposure table: one row per record of a drug class, with its date when known
SOURCE_DATA_DIR = ""  # folder for the source-data workbooks written here

SCORES_FILE = "table_cognitive_scores.xlsx"
MEDICATION_FILE = "table_cognitive_scores_medication.xlsx"
CHECKS_FILE = "checks_cognitive_scores.xlsx"

# =============================================================================
# SETTINGS
# =============================================================================
SCORE_TYPES = au.COGNITIVE_SCORE_TYPES
MIN_SLOPE_GAP_YRS = au.MIN_SLOPE_GAP_YRS   # first-to-last score gap for an annual change
MED_CLASSES = ["on_cs", "on_dmard", "on_antimalarial", "on_biologic"]
GROUPS = [(1, 1, "AID+ female"), (1, 0, "AID+ male"), (0, 1, "AID- female"), (0, 0, "AID- male")]
CELLS = [(1, 1, "aid_pos_exposed"), (1, 0, "aid_pos_unexposed"),
         (0, 1, "aid_neg_exposed"), (0, 0, "aid_neg_unexposed")]
DOC_MEASURES = [("age_at_entry", "Age at first note, years"),
                ("documentation_span_yrs", "Documentation span, years"),
                ("note_count", "Total note count"),
                ("note_count_predx", "Pre-diagnosis note count"),
                ("note_count_postdx", "Post-diagnosis note count")]
MEASURE_LABELS = {"annual_change": "Annual change"}
NAN = float("nan")

FORMULA_SLOPE = "slope ~ aid_status + age_c + female"
MED_FORMULAS = {"annual_change": {"A": FORMULA_SLOPE, "B": FORMULA_SLOPE + " + exposed",
                                  "C": FORMULA_SLOPE + " + exposed + aid_status:exposed"}}
MED_DEFINITIONS = {"annual_change": "before_last_score"}


def n_terms(formula):
    return len([t for t in formula.split("~", 1)[1].split("+") if t.strip()])


def empty_fit():
    return {"coef": NAN, "se": NAN, "crit": NAN, "ci_low": NAN, "ci_high": NAN, "p": NAN}


def term_values(mod, term):
    if term not in mod.params.index:
        return empty_fit()
    ci = mod.conf_int()
    crit = (t_dist.ppf(0.975, mod.df_resid) if getattr(mod, "use_t", False)
            else norm.ppf(0.975))
    return {"coef": float(mod.params[term]), "se": float(mod.bse[term]), "crit": float(crit),
            "ci_low": float(ci.loc[term, 0]), "ci_high": float(ci.loc[term, 1]),
            "p": float(mod.pvalues[term])}


def fit_ols(formula, d, cluster=None):
    kw = ({"cov_type": "cluster", "cov_kwds": {"groups": d[cluster]}} if cluster
          else {"cov_type": "HC3"})
    return smf.ols(formula, data=d).fit(**kw)


def check_ci(df, label):
    f = df[(df["status"] == "fitted") & df["coef"].notna()]
    if not (np.allclose(f["ci_low"], f["coef"] - f["crit"] * f["se"], rtol=1e-9, atol=1e-9)
            and np.allclose(f["ci_high"], f["coef"] + f["crit"] * f["se"], rtol=1e-9, atol=1e-9)):
        raise RuntimeError(f"{label}: CI does not equal coef -/+ crit x SE")


def linearity_row(key, formula, d, term, n_gate, cluster=None):
    """The model refitted with age_c squared added (same rows); the quadratic
    model must meet the model floor (model_floor_reason) with one more term
    (n_gate: the rows the gate counts). Flag at a shift of 0.5 points per year."""
    row = {**key, "status": "not_run", "reason": "", "coef_linear": NAN, "coef_quadratic": NAN,
           "shift_points": NAN, "coef_age_squared": NAN, "flag": 0}
    fq = formula + " + age_c_sq"
    why = au.model_floor_reason(n_gate["y"], n_gate["aid_status"], n_terms(fq),
                                sex=n_gate["female"], continuous=True)
    if why:
        row["reason"] = why
        return row
    try:
        ml, mq = fit_ols(formula, d, cluster), fit_ols(fq, d, cluster)
    except Exception as e:
        row["reason"] = f"fit error: {type(e).__name__}"
        return row
    if term not in ml.params.index or term not in mq.params.index:
        row["reason"] = "term not estimated"
        return row
    bl, bq = float(ml.params[term]), float(mq.params[term])
    row.update(status="run", coef_linear=bl, coef_quadratic=bq, shift_points=abs(bq - bl),
               coef_age_squared=float(mq.params.get("age_c_sq", NAN)),
               flag=int(abs(bq - bl) >= au.LINEARITY_SHIFT_POINTS))
    return row


def mann_whitney(a, b):
    a, b = pd.Series(a).dropna().astype(float), pd.Series(b).dropna().astype(float)
    why = au.group_minimum_reason(len(a), len(b))
    if why:
        return "not_tested", why, NAN
    return "tested", "", float(mannwhitneyu(a, b, alternative="two-sided")[1])


def fmt_console(v, spec):
    return "n/a" if pd.isna(v) else format(v, spec)


# =============================================================================
# SECTION 1 -- POPULATION AND SCORES
# =============================================================================
au.section("SECTION 1 -- POPULATION AND SCORES")
pop, notes = au.load_population(COHORT_PATIENTS_PATH, COHORT_NOTES_PATH)
scores, filter_rows = au.absolute_cognitive_scores(
    au.load_cognitive_scores(COGNITIVE_SCORES_PATH, pop, notes), pop)

# -----------------------------------------------------------------------------
au.sub("Scored vs unscored patients: documentation measures (Mann-Whitney U; checks)")
# -----------------------------------------------------------------------------
scored = set(scores["person_id"].astype(int))
scored_rows = []
for a, s, lbl in GROUPS:
    g = pop[(pop["aid_status"] == a) & (pop["female"] == s)]
    gs, gu = g[g["person_id"].astype(int).isin(scored)], g[~g["person_id"].astype(int).isin(scored)]
    print(f"\n  {lbl}: scored {len(gs):,} of {len(g):,}")
    for col, label in DOC_MEASURES:
        st, why, p = mann_whitney(gs[col], gu[col])
        scored_rows.append({"group": lbl, "measure": col, "row_label": label,
                            "n_scored": len(gs), "n_unscored": len(gu),
                            "median_scored": gs[col].median(), "median_unscored": gu[col].median(),
                            "status": st, "reason": why, "p": p})
        print(f"    {label:<30} scored {fmt_console(gs[col].median(), '.1f'):>7}  unscored "
              f"{fmt_console(gu[col].median(), '.1f'):>7}  "
              + (f"P={p:.4f}" if st == "tested" else f"not tested: {why}"))

# =============================================================================
# SECTION 2 -- ANNUAL CHANGE
# =============================================================================
au.section("SECTION 2 -- ANNUAL CHANGE")
score_rows, lin_rows = [], []
slopes = {}
for st in SCORE_TYPES:
    s = au.annual_change(scores, st)   # annual change, first to last score
    slopes[st] = s
    why = au.model_floor_reason(s["slope"], s["aid_status"], n_terms(FORMULA_SLOPE),
                                sex=s["female"], continuous=True)
    r = {"score_type": st, "measure": "annual_change",
         "row_label": f"{st}, {MEASURE_LABELS['annual_change']}", "n_rows": len(s),
         "n_aid_pos": int((s["aid_status"] == 1).sum()),
         "n_aid_neg": int((s["aid_status"] == 0).sum()),
         "status": "not_fitted", "reason": why, **empty_fit()}
    if not why:
        try:
            mod = fit_ols(FORMULA_SLOPE, s)
            r.update(status="fitted", reason="", **term_values(mod, "aid_status"))
            lin_rows.append(linearity_row(
                {"analysis": "annual_change", "score_type": st, "term": "AID"}, FORMULA_SLOPE, s,
                "aid_status", {"y": s["slope"], "aid_status": s["aid_status"],
                               "female": s["female"]}))
        except Exception as e:
            r.update(status="failed", reason=f"fit error: {type(e).__name__}")
    score_rows.append(r)

score_df = pd.DataFrame(score_rows)
check_ci(score_df, "score models")
for _, r in score_df.iterrows():
    print(f"  {r['score_type']:<5} {r['measure']:<14} n patients AID+ {r['n_aid_pos']:,} "
          f"AID- {r['n_aid_neg']:,}  "
          + (f"AID+ minus AID- {r['coef']:+.2f} [{r['ci_low']:+.2f}, {r['ci_high']:+.2f}] "
             f"P={r['p']:.4f}"
             if r["status"] == "fitted" else f"{r['status']}: {r['reason']}"))

# =============================================================================
# SECTION 3 -- MEDICATION EXPOSURE AND ANNUAL CHANGE
# =============================================================================
au.section("SECTION 3 -- MEDICATION EXPOSURE AND ANNUAL CHANGE")
print("  A: the annual-change model; B: + exposed; C: + exposed + aid_status:exposed (HC3).")
print("  Exposed = a dated record of the class before the last score. Check: any record,")
print("  dated or not.")
meds = au.load_medication_records(MEDICATION_RECORDS_PATH, pop)
med_ever, med_first_dated = {}, {}
for c in MED_CLASSES:
    m = meds[meds["drug_class"] == c]
    med_ever[c] = set(m["person_id"].astype(int))
    med_first_dated[c] = m.dropna(subset=["exposure_date"]).groupby("person_id")["exposure_date"].min()
    au.count_line(f"population patients with a record of {c} (any)", len(med_ever[c]), len(pop))


def exposure(c, definition, d, date_col):
    if definition == "ever":
        return d["person_id"].astype(int).isin(med_ever[c]).astype(int)
    first = pd.to_datetime(d["person_id"].map(med_first_dated[c]))
    return (first < d[date_col]).fillna(False).astype(int)


def med_gate(measure, d, y, terms):
    """The model floor on patients (MIN_EVENTS_PER_TERM per model term); 10
    patients in each AID x exposure group."""
    return au.model_floor_reason(d[y], d["aid_status"], terms, sex=d["female"],
                                 continuous=True, modifier=d["exposed"])


def med_model_rows(c, measure, definition, st, d):
    d = d.copy()
    y, date_col, cluster = "slope", "last_score_date", None
    d["exposed"] = exposure(c, definition, d, date_col)
    formulas = MED_FORMULAS[measure]
    why = med_gate(measure, d, y, n_terms(formulas["C"]))
    counts = {f"n_{lbl}": int(d.loc[(d["aid_status"] == a) & (d["exposed"] == e),
                                    "person_id"].nunique()) for a, e, lbl in CELLS}
    fits, status, reason = {}, "not_fitted", why
    if not why:
        try:
            fits = {m: fit_ols(f, d, cluster) for m, f in formulas.items()}
            status, reason = "fitted", ""
            if any(t not in fits[m].params.index for m, t in
                   (("A", "aid_status"), ("B", "exposed"), ("C", "aid_status:exposed"))):
                fits, status, reason = {}, "failed", "term not estimated"
        except Exception as e:
            fits, status, reason = {}, "failed", f"fit error: {type(e).__name__}"
    rows = []
    for m, t, role in (("A", "aid_status", "AID"), ("B", "aid_status", "AID"),
                       ("B", "exposed", "exposure"), ("C", "aid_status:exposed", "interaction")):
        rows.append({"drug_class": c, "measure": measure, "exposure_definition": definition,
                     "score_type": st, "row_label": f"{st}, {MEASURE_LABELS[measure]}",
                     "model": m, "term": role, "n_rows": len(d),
                     "n_patients": int(d["person_id"].nunique()), **counts, "status": status,
                     "reason": reason, **(term_values(fits[m], t) if fits else empty_fit())})
    lins = []
    if status == "fitted" and definition != "ever":
        lins.append(linearity_row({"analysis": f"{c}_{measure}", "score_type": st,
                                   "term": "interaction (model C)"},
                                  formulas["C"], d, "aid_status:exposed",
                                  {"y": d.drop_duplicates("person_id")[y],
                                   "aid_status": d.drop_duplicates("person_id")["aid_status"],
                                   "female": d.drop_duplicates("person_id")["female"]},
                                  cluster=cluster))
    return rows, lins


med_rows = []
for c in MED_CLASSES:
    for measure, data in (("annual_change", slopes),):
        for definition in (MED_DEFINITIONS[measure], "ever"):
            for st in SCORE_TYPES:
                rows, lins = med_model_rows(c, measure, definition, st, data[st])
                med_rows += rows
                lin_rows += lins
med_df = pd.DataFrame(med_rows)
check_ci(med_df, "medication models")
med_df["table"] = ((med_df["exposure_definition"] != "ever") & (med_df["model"] == "C")).astype(int)
med_df["q"] = NAN
med_df["fdr_family_size"] = 0
for c in MED_CLASSES:
    tested = med_df.index[(med_df["table"] == 1) & (med_df["drug_class"] == c)
                          & (med_df["status"] == "fitted") & med_df["p"].notna()]
    if len(tested):
        med_df.loc[tested, "q"] = au.bh_fdr(med_df.loc[tested, "p"].values)
        med_df.loc[tested, "fdr_family_size"] = len(tested)
for _, r in med_df[med_df["model"] == "C"].iterrows():
    print(f"  {r['drug_class']:<16} {r['measure']:<14} {r['exposure_definition']:<18} "
          f"{r['score_type']:<5} n={int(r['n_patients']):,}  "
          + (f"interaction {r['coef']:+.3f} [{r['ci_low']:+.2f}, {r['ci_high']:+.2f}] "
             f"P={r['p']:.4f}" + (f" Q={r['q']:.4f}" if r["table"] else " (check, no Q)")
             if r["status"] == "fitted" else f"{r['status']}: {r['reason']}"))
lin_df = pd.DataFrame(lin_rows, columns=["analysis", "score_type", "term", "status", "reason",
                                         "coef_linear", "coef_quadratic", "shift_points",
                                         "coef_age_squared", "flag"])
run = lin_df[lin_df["status"] == "run"]
print(f"\n  Linearity (age squared): {len(run)} of {len(lin_df)} run; flagged (shift >= "
      f"{au.LINEARITY_SHIFT_POINTS} points): {int(run['flag'].sum())}")

# -----------------------------------------------------------------------------
au.sub("Annual change by group; Mann-Whitney U, AID+ vs AID-, within sex (checks)")
# -----------------------------------------------------------------------------
group_rows, test_rows = [], []
for st, s in slopes.items():
    print(f"\n  {st}: {len(s):,} patients"
          + (f" (median gap {s['gap_yrs'].median():.1f} yr)" if len(s) else ""))
    for a, sx, lbl in GROUPS:
        v = s.loc[(s["aid_status"] == a) & (s["female"] == sx), "slope"]
        shown = len(v) >= au.MIN_GROUP_N
        group_rows.append({"score_type": st, "group": lbl, "n": len(v),
                           "median": v.median() if shown else NAN,
                           "q1": v.quantile(.25) if shown else NAN,
                           "q3": v.quantile(.75) if shown else NAN})
        print(f"  {lbl:<12} n={len(v):>5}  " + (
            f"median {v.median():+.2f}  IQR [{v.quantile(.25):+.2f}, {v.quantile(.75):+.2f}]"
            if shown else f"not shown (fewer than {au.MIN_GROUP_N})"))
    for sx, sex in ((1, "Female"), (0, "Male")):
        ap = s.loc[(s["aid_status"] == 1) & (s["female"] == sx), "slope"]
        an = s.loc[(s["aid_status"] == 0) & (s["female"] == sx), "slope"]
        status, why, p = mann_whitney(ap, an)
        test_rows.append({"score_type": st, "sex": sex, "n_aid_pos": len(ap),
                          "median_aid_pos": ap.median(), "n_aid_neg": len(an),
                          "median_aid_neg": an.median(), "status": status, "reason": why,
                          "p": p})
        print(f"  {sex:<6} " + (f"AID+ {ap.median():+.2f}/yr (n={len(ap):,})  AID- "
                                f"{an.median():+.2f}/yr (n={len(an):,})  P={p:.4f}"
                                if status == "tested" else f"not tested: {why}"))

# =============================================================================
# SECTION 4 -- SOURCE DATA
# =============================================================================
au.section("SECTION 4 -- SOURCE DATA")
FIT_COLUMNS = ["status", "reason", "coef", "se", "crit", "ci_low", "ci_high", "p"]


def parameters(extra=()):
    rows = [("min_events_per_term", au.MIN_EVENTS_PER_TERM), ("min_group_n", au.MIN_GROUP_N),
            ("min_slope_gap_yrs", MIN_SLOPE_GAP_YRS),
            ("linearity_shift_points", au.LINEARITY_SHIFT_POINTS),
            ("n_aid_pos", int(pop["aid_status"].sum())),
            ("n_aid_neg", int((pop["aid_status"] == 0).sum())),
            ("n_scored_aid_pos", int(pop.loc[pop["aid_status"] == 1, "person_id"].astype(int)
                                     .isin(scored).sum())),
            ("n_scored_aid_neg", int(pop.loc[pop["aid_status"] == 0, "person_id"].astype(int)
                                     .isin(scored).sum())),
            ("score_rows_kept", int(len(scores)))]
    return pd.DataFrame(rows + list(extra), columns=["parameter", "value"])


au.write_workbook(os.path.join(SOURCE_DATA_DIR, SCORES_FILE), {
    "scores": score_df[["score_type", "measure", "row_label", "n_rows", "n_aid_pos",
                        "n_aid_neg"] + FIT_COLUMNS],
    "parameters": parameters(),
})
MED_ID = ["drug_class", "measure", "exposure_definition", "score_type", "row_label"]
MED_COUNTS = ["n_rows", "n_patients"] + [f"n_{lbl}" for _, _, lbl in CELLS]
table_med = med_df[med_df["table"] == 1]
au.write_workbook(os.path.join(SOURCE_DATA_DIR, MEDICATION_FILE), {
    "interactions": table_med[MED_ID + MED_COUNTS + FIT_COLUMNS + ["q", "fdr_family_size"]],
    "parameters": parameters(
        [(f"fdr_family_size_{c}", int(table_med.loc[table_med["drug_class"] == c,
                                                     "fdr_family_size"].max()))
         for c in MED_CLASSES]
        + [(f"n_population_with_record_{c}", len(med_ever[c])) for c in MED_CLASSES]),
})
au.write_workbook(os.path.join(SOURCE_DATA_DIR, CHECKS_FILE), {
    "filters": pd.DataFrame(filter_rows),
    "scored_vs_unscored": pd.DataFrame(scored_rows),
    "medication_models": med_df[MED_ID + ["model", "term"] + MED_COUNTS + FIT_COLUMNS
                                + ["table", "q", "fdr_family_size"]],
    "linearity": lin_df,
    "annual_change_by_group": pd.DataFrame(group_rows),
    "annual_change_tests": pd.DataFrame(test_rows),
    "parameters": parameters(),
})
print("\n" + au.SEP + "\n  COMPLETE\n" + au.SEP)
