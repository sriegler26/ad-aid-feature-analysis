"""
symptom_timing_sequence.py
==========================
Does autoimmune disease (AID) change when symptoms of Alzheimer's disease (AD)
are first documented in the clinical notes, and does it change the order in
which they are first documented?

Computes every statistic and writes the source data for the timing and
sequence tables (one workbook per table, in SOURCE_DATA_DIR) and a checks
workbook. The tables are rendered from these workbooks by the table
script; nothing here formats a number.

Population and first documentation
----------------------------------
- Analysis population: patients with at least one clinical note.
- First documentation of a domain (or category): the earliest dated symptom
  record of it; age at it in whole days / 365.25. Records on undated notes
  are not used; "other" is not used. No age ceiling: it applies to presence
  analyses only.
- Pre-diagnosis documentation span: AD diagnosis date minus first dated
  note, in years, never clipped. It is negative for a patient whose first
  note follows the diagnosis; that value carries real information (entry
  after diagnosis, and by how much), and unclipped, age at first note + span
  = age at AD diagnosis for every patient. It is used only as an adjustment
  covariate, centered within each model's sample.
- Age at first note (age at the first dated note): a symptom cannot be
  documented before the record begins, so a patient first seen younger can
  be documented younger. Adjustment covariate, centered within each model's
  sample. With the span beside it, age at AD diagnosis is held fixed too.
- Before the diagnosis = strictly before the AD diagnosis date.

Timing models (domains; one row per patient with the domain documented)
-----------------------------------------------------------------------
first_age ~ aid_status + female + age_at_entry_c + predx_span_c
The AID coefficient is the estimand. No AID x sex term is fitted
(sex-specific results are reported by stratification, not by interaction
tests). Outcome: age at first
documentation (OLS with HC3 standard errors: the table; median regression:
a check). Years from first documentation to the AD diagnosis is not modeled:
with both covariates its AID coefficient is exactly minus that of age at
first documentation. Unweighted: the outcome exists only for patients
with the domain documented. Gate: the shared model floor for continuous
outcomes (10 patients per term, both AID groups and both sexes present).
FDR:
Benjamini-Hochberg per outcome and domain definition over the
domains; the positive control (systemic inflammation autoimmune) is shown
with its P value, outside the family. Linearity: every fitted model
refitted with the squares of both centered covariates added; flagged at a
coefficient shift of 0.5 years or more (not a test).

Timing models (categories; one row per patient with the category documented)
---------------------------------------------------------------------------
The same pooled model (OLS with HC3, unweighted, covariates centered within
each sample), gate and linearity check, fitted to every category except the
unassigned one ("other"). FDR: Benjamini-Hochberg over the tested
categories, a family of its own beside the domains; the categories of the
systemic inflammation autoimmune domain are positive controls, shown with
their P values outside the family. The secondary domain definition applies
to domains only. In the checks workbook, `level` (domain / category) tells
the two apart.

AD-core onset
-------------
The first documentation of any of episodic memory, executive function and
global cognition (analysis_utils.AD_CORE_DOMAINS). The combined target is
fitted with the timing model (a single test, so Q = P); each AD-core
domain on its own is the age-at-first-documentation model of that domain,
printed beside it. Floor check for every target: patients with a span of at
least 5 years, the same model (covariates centered within that sample); not
in any FDR family. Descriptive: Mann-Whitney tests, quartiles
per AID x sex group, time from onset to diagnosis, and which AD-core domain
set the onset.

Sequence inversion (category pairs; analysis_utils.SEQUENCE_PAIRS)
-----------------------------------------------------------------
Each pair is (expected first, expected later) in typical amnestic AD. A
patient's order is inverted when the later-expected category is first
documented strictly before the expected-first one; a same-day first
documentation (tie) counts as not inverted. Patients with both categories
documented and a span of at least 5 years.
Test: Fisher's exact 2x2 comparison of the share inverted, AID+ vs AID-,
unadjusted (conditional MLE odds ratio, exact 95% CI), in all patients (the
table) and within female and within male patients (the by-sex table).
Gate: two_by_two_reason (at least 10 patients in each AID group compared;
not tested when no patient, or every patient, is inverted in both). The odds
ratio is not estimable when an AID group has no inverted (or only inverted)
patients; P is still reported. FDR: BH over the tested pairs, separately for
all patients, female and male patients. Check (no Q): all patients with ties
excluded.

Secondary domain definition
---------------------------
Sleep disturbance under autonomic instead of somatic. Every domain-level
timing model is also fitted under it; for each domain whose first
documentation differs, the change in the estimate and whether Q crossed
0.05 is printed and written (thresholds 0.25 and 0.5 years).

Inputs: the patient table, the note table and the symptom table (paths below).
"""

import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import mannwhitneyu, norm, t as t_dist
from statsmodels.tools.sm_exceptions import IterationLimitWarning

import analysis_utils as au

warnings.filterwarnings("ignore")   # after the imports: statsmodels resets its own filters

# =============================================================================
# PATHS (relative to the data folder the script is run from)
# =============================================================================
COHORT_PATIENTS_PATH = ""  # patient table: AID status, sex, key dates (birth, AD diagnosis, death) and analysis-population flag
COHORT_NOTES_PATH = ""  # note table: one row per clinical note, with its date and whether it is a cognitive-specialty note
SYMPTOM_RECORDS_PATH = ""  # symptom table: one row per documented symptom, with its category
SOURCE_DATA_DIR = ""  # folder for the source-data workbooks written here

FIRST_AGE_FILE = "table_age_at_first_documentation.xlsx"
INVERSION_FILE = "table_symptom_sequence_inversion.xlsx"
INVERSION_BY_SEX_FILE = "table_symptom_sequence_inversion_by_sex.xlsx"
CHECKS_FILE = "checks_symptom_timing_sequence.xlsx"

# =============================================================================
# SETTINGS
# =============================================================================
Q_THRESHOLD = 0.05                 # secondary-definition check: Q crossing
ESTIMATE_CHANGE_SMALL_YRS = 0.25   # secondary-definition check: |change|
ESTIMATE_CHANGE_LARGE_YRS = 0.5
MEAN_MEDIAN_FLAG_YRS = 0.5         # mean vs median AID coefficient (console)

# timing analyses: name -> (model kind, outcome column)
TIMING_ANALYSES = [("first_age_mean", "ols", "first_age"),
                   ("first_age_median", "median", "first_age")]
# adjustment covariates of every timing model: (centered column, source column)
COVARIATES = [("age_at_entry_c", "age_at_entry"),
              ("predx_span_c", "predx_documentation_span_yrs")]
COV_TERMS = " + ".join(c for c, _ in COVARIATES)
DEFINITIONS = ["primary", "secondary"]
TERM_AID = "aid_status"
INVERSION_FLOOR_YRS = au.PREDX_SPAN_FLOOR_5_YRS
AD_CORE_FLOORS = [au.PREDX_SPAN_FLOOR_5_YRS]

GROUPS = [(1, 1, "AID+ female"), (1, 0, "AID+ male"), (0, 1, "AID- female"), (0, 0, "AID- male")]
GROUP_COUNT_COLUMNS = ["n_aid_pos", "n_aid_neg", "n_aid_pos_female", "n_aid_pos_male",
                       "n_aid_neg_female", "n_aid_neg_male"]
FIT_COLUMNS = ["n_model", "status", "reason", "coef", "se", "crit", "ci_low", "ci_high",
               "p", "q", "fdr_family_size"]
NAN = float("nan")


def n_terms(formula):
    return len([t for t in formula.split("~", 1)[1].split("+") if t.strip()])


def group_counts(d):
    out = {"n_aid_pos": int((d["aid_status"] == 1).sum()),
           "n_aid_neg": int((d["aid_status"] == 0).sum())}
    for a, arm in ((1, "aid_pos"), (0, "aid_neg")):
        for s, sx in ((1, "female"), (0, "male")):
            out[f"n_{arm}_{sx}"] = int(((d["aid_status"] == a) & (d["female"] == s)).sum())
    return out


def empty_fit():
    return {"coef": NAN, "se": NAN, "crit": NAN, "ci_low": NAN, "ci_high": NAN, "p": NAN}


def term_values(mod, term):
    """coef, SE, the critical value of the 95% CI (t when the fit uses t, z
    otherwise), CI and P of one term; NaN when the term is absent."""
    if term not in mod.params.index:
        return empty_fit()
    ci = mod.conf_int()
    crit = (t_dist.ppf(0.975, mod.df_resid) if getattr(mod, "use_t", False)
            else norm.ppf(0.975))
    return {"coef": float(mod.params[term]), "se": float(mod.bse[term]), "crit": float(crit),
            "ci_low": float(ci.loc[term, 0]), "ci_high": float(ci.loc[term, 1]),
            "p": float(mod.pvalues[term])}


def fdr_role(code, level="domain"):
    controls = (au.POSITIVE_CONTROL_DOMAINS if level == "domain"
                else au.POSITIVE_CONTROL_CATEGORIES)
    return "positive_control" if code in controls else "tested"


# =============================================================================
# TIMING MODEL FITS (OLS with HC3, or median regression)
# =============================================================================
QUANTREG_ITERATION_LIMIT = []   # median-regression fits that reached the iteration limit


def fit_linear(kind, formula, d):
    if kind == "ols":
        return smf.ols(formula, data=d).fit(cov_type="HC3")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", IterationLimitWarning)
        mod = smf.quantreg(formula, data=d).fit(q=0.5)
    if any(issubclass(w.category, IterationLimitWarning) for w in caught):
        QUANTREG_ITERATION_LIMIT.append(formula)
    return mod


def fit_timing(kind, y, d):
    """The timing model of one outcome on one sample (covariates: COVARIATES,
    centered by center()). Returns (row, fit); fit is None when not fitted."""
    fp = f"{y} ~ aid_status + female + {COV_TERMS}"
    why = au.model_floor_reason(d[y], d["aid_status"], n_terms(fp), sex=d["female"],
                                continuous=True)
    out = {"n_model": len(d), **group_counts(d),
           "status": "fitted" if not why else "not_fitted", "reason": why, **empty_fit(),
           "coef_female": NAN, "p_female": NAN, "coef_age_at_entry": NAN,
           "p_age_at_entry": NAN, "coef_span": NAN, "p_span": NAN, "r_squared": NAN}
    if why:
        return out, None
    try:
        mp = fit_linear(kind, fp, d)
    except Exception as e:
        out.update(status="failed", reason=f"fit error: {type(e).__name__}")
        return out, None
    fem = term_values(mp, "female")
    ent, spn = term_values(mp, "age_at_entry_c"), term_values(mp, "predx_span_c")
    out.update(term_values(mp, TERM_AID), coef_female=fem["coef"], p_female=fem["p"],
               coef_age_at_entry=ent["coef"], p_age_at_entry=ent["p"],
               coef_span=spn["coef"], p_span=spn["p"],
               r_squared=mp.rsquared if kind == "ols" else getattr(mp, "prsquared", NAN))
    if out["status"] == "fitted" and pd.isna(out["coef"]):
        out.update(status="failed", reason="term not estimated")
    return out, mp


def linearity_row(kind, y, d, fit):
    """The model refitted with the squares of both centered covariates added
    (one refit, same rows). The quadratic model must meet the same floor with
    the two more terms."""
    fq = (f"{y} ~ aid_status + female + " + COV_TERMS + " + "
          + " + ".join(f"{c}_sq" for c, _ in COVARIATES))
    row = {"status": "not_run", "reason": "", "coef_linear": NAN, "coef_quadratic": NAN,
           "shift_years": NAN, "shift_pct": NAN, "coef_age_at_entry_squared": NAN,
           "coef_span_squared": NAN, "flag": 0}
    why = au.model_floor_reason(d[y], d["aid_status"], n_terms(fq),
                                sex=d["female"], continuous=True)
    if why:
        row["reason"] = why
        return row
    try:
        mq = fit_linear(kind, fq, d)
    except Exception as e:
        row["reason"] = f"fit error: {type(e).__name__}"
        return row
    bl, bq = fit.params.get(TERM_AID, NAN), mq.params.get(TERM_AID, NAN)
    if pd.isna(bl) or pd.isna(bq):
        row["reason"] = "term not estimated"
        return row
    shift = abs(bq - bl)
    row.update(status="run", coef_linear=float(bl), coef_quadratic=float(bq),
               shift_years=float(shift),
               shift_pct=float(shift / abs(bl) * 100) if bl != 0 else NAN,
               coef_age_at_entry_squared=float(mq.params.get("age_at_entry_c_sq", NAN)),
               coef_span_squared=float(mq.params.get("predx_span_c_sq", NAN)),
               flag=int(shift >= au.LINEARITY_SHIFT_YEARS))
    return row


def add_fdr(rows, family_keys):
    """BH q within each family over fitted, tested rows."""
    df = pd.DataFrame(rows)
    df["q"] = NAN
    df["fdr_family_size"] = 0
    for _, idx in df.groupby(family_keys).groups.items():
        m = df.loc[idx]
        tested = m.index[(m["status"] == "fitted") & (m["fdr_role"] == "tested")
                         & m["p"].notna()]
        if len(tested):
            df.loc[tested, "q"] = au.bh_fdr(df.loc[tested, "p"].values)
            df.loc[tested, "fdr_family_size"] = len(tested)
    return df


def fmt_console(v, spec):
    return "n/a" if pd.isna(v) else format(v, spec)


def print_timing(df, title):
    print(f"\n  {title}")
    print(f"  {'':<34} {'model':<12} {'n':>5} {'coef (yr)':>10} {'95% CI':>17} "
          f"{'P':>8} {'Q':>8}  status")
    for _, r in df.sort_values(["model", "code"], ascending=[False, True]).iterrows():
        if r["status"] == "fitted":
            ci = f"[{r['ci_low']:+.2f}, {r['ci_high']:+.2f}]"
            tag = "" if r["fdr_role"] == "tested" else f"[{r['fdr_role']}]"
            print(f"  {r['code']:<34} {r['model']:<12} {r['n_model']:>5} {r['coef']:>+10.2f} "
                  f"{ci:>17} {r['p']:>8.4f} {fmt_console(r['q'], '.4f'):>8}  {tag}")
        else:
            print(f"  {r['code']:<34} {r['model']:<12} {r['n_model']:>5} "
                  f"{r['status']}: {r['reason']}")


# =============================================================================
# SECTION 1 -- POPULATION AND FIRST DOCUMENTATION
# =============================================================================
au.section("SECTION 1 -- POPULATION AND FIRST DOCUMENTATION")
pop, notes = au.load_population(COHORT_PATIENTS_PATH, COHORT_NOTES_PATH)
POP_COLUMNS = ["person_id", "aid_status", "female", "birth_date", "ad_diagnosis_date",
               "age_at_dx", "age_at_entry", "predx_documentation_span_yrs"]
span = pop["predx_documentation_span_yrs"]
print("\n  Patients whose first dated note follows the AD diagnosis (negative span):")
cohort_rows = []
for a, s, lbl in GROUPS:
    g = (pop["aid_status"] == a) & (pop["female"] == s)
    au.count_line(lbl, (g & (span < 0)).sum(), g.sum())
    cohort_rows.append({"group": lbl, "n": int(g.sum()),
                        "n_first_note_after_diagnosis": int((g & (span < 0)).sum()),
                        "span_median": NAN, "span_q1": NAN, "span_q3": NAN})
print("  Pre-diagnosis documentation span, years, median [IQR]:")
for a, lbl in ((1, "AID+"), (0, "AID-")):
    w = span[pop["aid_status"] == a].dropna()
    print(f"    {lbl:<5}: {w.median():.2f} [{w.quantile(.25):.2f}, {w.quantile(.75):.2f}]  "
          f"n={len(w):,}")
    cohort_rows.append({"group": lbl, "n": int((pop["aid_status"] == a).sum()),
                        "n_first_note_after_diagnosis": int((w < 0).sum()),
                        "span_median": w.median(), "span_q1": w.quantile(.25),
                        "span_q3": w.quantile(.75)})

rec = au.load_symptom_records(SYMPTOM_RECORDS_PATH, pop, notes)
dated = rec[rec["note_date"].notna()]
au.count_line("dated records of the unassigned category (not used)",
              (dated["category"].isin(au.UNASSIGNED_CATEGORIES)).sum(), len(dated))


def first_doc_frame(level, definition="primary"):
    f = au.first_documentation(rec, pop, level, definition)
    f = f.merge(pop[POP_COLUMNS[1:] + ["person_id"]], on="person_id", how="left",
                validate="many_to_one")
    # years from first documentation to the AD diagnosis (positive: before it)
    f["years_to_diagnosis"] = f["age_at_dx"] - f["first_age"]
    f["before_diagnosis"] = au.is_pre_diagnosis(f["first_date"], f["ad_diagnosis_date"])
    return f


first_dom = {d: first_doc_frame("domain", d) for d in DEFINITIONS}
first_cat = first_doc_frame("category")
au.count_line("patient x domain first documentations (primary definition)",
              len(first_dom["primary"]))
au.count_line("patients with any domain documented", first_dom["primary"]["person_id"].nunique(),
              len(pop))

# =============================================================================
# SECTION 2 -- TIMING MODELS (DOMAINS)
# =============================================================================
au.section("SECTION 2 -- TIMING MODELS (DOMAINS)")
print(f"  first_age ~ aid_status + female + {COV_TERMS}")
print("  age_at_entry_c: age at the first dated note; predx_span_c: pre-diagnosis")
print("  documentation span; both centered within each sample")


def center(d):
    """The centered covariates (and their squares) of one model sample."""
    d = d.copy()
    for name, source in COVARIATES:
        d[name] = d[source] - d[source].mean()
        d[name + "_sq"] = d[name] ** 2
    return d


def sample(frame, code):
    return center(frame[frame["code"] == code])


timing_rows, lin_rows = [], []
for analysis, kind, y in TIMING_ANALYSES:
    for defn in DEFINITIONS:
        for code in sorted(au.DOMAIN_DEFINITIONS[defn]):
            d = sample(first_dom[defn], code)
            res, fit = fit_timing(kind, y, d)
            timing_rows.append({"analysis": analysis, "level": "domain", "definition": defn,
                                "code": code, "row_label": au.DOMAIN_LABELS[code],
                                "fdr_role": fdr_role(code), "model": "pooled", "term": TERM_AID,
                                **res})
            if res["status"] == "fitted" and defn == "primary":
                lin_rows.append({"analysis": analysis, "level": "domain", "code": code,
                                 "model": "pooled", "term": TERM_AID,
                                 **linearity_row(kind, y, d, fit)})

# categories: the pooled OLS model, every category but the unassigned one; written
# after every other row, so the domain and AD-core rows keep their places
CATEGORIES = sorted(c for c in au.SCHEMA_CATEGORIES if c not in au.UNASSIGNED_CATEGORIES)
CATEGORY_DOMAIN = au.category_domain("primary")
cat_rows, cat_lin_rows = [], []
for code in CATEGORIES:
    d = sample(first_cat, code)
    res, fit = fit_timing("ols", "first_age", d)
    cat_rows.append({"analysis": "first_age_mean", "level": "category", "definition": "primary",
                     "code": code, "domain": CATEGORY_DOMAIN[code],
                     "row_label": au.category_row_label(code),
                     "fdr_role": fdr_role(code, "category"), "model": "pooled",
                     "term": TERM_AID, **res})
    if res["status"] == "fitted":
        cat_lin_rows.append({"analysis": "first_age_mean", "level": "category", "code": code,
                             "model": "pooled", "term": TERM_AID,
                             **linearity_row("ols", "first_age", d, fit)})

# =============================================================================
# SECTION 3 -- AD-CORE ONSET
# =============================================================================
au.section("SECTION 3 -- AD-CORE ONSET")
print(f"  AD-core domains: {', '.join(au.AD_CORE_DOMAINS)} (earliest first documentation "
      f"of any of them)")
core = au.ad_core_onset(first_dom["primary"]).merge(pop[POP_COLUMNS], on="person_id",
                                                    how="left", validate="one_to_one")
core["code"] = "combined"
core["years_to_diagnosis"] = core["age_at_dx"] - core["first_age"]
core["before_diagnosis"] = au.is_pre_diagnosis(core["first_date"], core["ad_diagnosis_date"])
au.count_line("patients with an AD-core onset", len(core), len(pop))

d = sample(core, "combined")
res, fit = fit_timing("ols", "first_age", d)
timing_rows.append({"analysis": "ad_core_onset", "level": "domain", "definition": "primary",
                    "code": "combined", "row_label": "AD core (combined)", "fdr_role": "tested",
                    "model": "pooled", "term": TERM_AID, **res})
if res["status"] == "fitted":
    lin_rows.append({"analysis": "ad_core_onset", "level": "domain", "code": "combined",
                     "model": "pooled", "term": TERM_AID,
                     **linearity_row("ols", "first_age", d, fit)})

core_targets = [(c, sample(first_dom["primary"], c)) for c in au.AD_CORE_DOMAINS] \
    + [("combined", core)]
for yrs in AD_CORE_FLOORS:
    for code, frame in core_targets:
        fl = center(frame[frame["predx_documentation_span_yrs"] >= yrs])
        res_f, _ = fit_timing("ols", "first_age", fl)
        timing_rows.append({"analysis": f"ad_core_floor_{yrs}yr", "level": "domain",
                            "definition": "primary", "code": code,
                            "row_label": au.DOMAIN_LABELS.get(code, "AD core (combined)"),
                            "fdr_role": "check", "model": "pooled", "term": TERM_AID, **res_f})

# FDR families: analysis x level x definition x term (categories: a family of their own)
timing_df = add_fdr(timing_rows + cat_rows, ["analysis", "level", "definition", "term"])
f = timing_df[timing_df["status"] == "fitted"]
for col, want in (("ci_low", f["coef"] - f["crit"] * f["se"]),
                  ("ci_high", f["coef"] + f["crit"] * f["se"])):
    if not np.allclose(f[col], want, rtol=1e-9, atol=1e-9):
        raise RuntimeError(f"timing models: {col} does not equal coef -/+ crit x SE")
linearity_df = pd.DataFrame(lin_rows + cat_lin_rows, columns=[
    "analysis", "level", "code", "model", "term", "status", "reason", "coef_linear",
    "coef_quadratic", "shift_years", "shift_pct", "coef_age_at_entry_squared",
    "coef_span_squared", "flag"])


def rows_of(analysis, definition="primary", level="domain"):
    return timing_df[(timing_df["analysis"] == analysis)
                     & (timing_df["definition"] == definition)
                     & (timing_df["level"] == level)]


for analysis, _, _ in TIMING_ANALYSES:
    print_timing(rows_of(analysis), f"{analysis} (primary definition)")
    if analysis == "first_age_mean":
        print_timing(rows_of(analysis, level="category"),
                     f"{analysis}, categories (own FDR family)")

au.sub("3a. AD-core onset: combined target (Q = P, a single test) and each domain")
print_timing(rows_of("ad_core_onset"), "combined target")
print_timing(rows_of("first_age_mean")[rows_of("first_age_mean")["code"].isin(au.AD_CORE_DOMAINS)],
             "each AD-core domain (the age-at-first-documentation model)")
for yrs in AD_CORE_FLOORS:
    print_timing(rows_of(f"ad_core_floor_{yrs}yr"),
                 f"floor check: span >= {yrs} yr, same model")

def print_linearity(df):
    run = df[df["status"] == "run"]
    print(f"  {len(run)} of {len(df)} run")
    for _, r in df.sort_values("shift_years", ascending=False, na_position="last").iterrows():
        if r["status"] == "run":
            print(f"  {r['analysis']:<24} {r['code']:<34} {r['model']:<12} "
                  f"{r['coef_linear']:>+8.3f} -> {r['coef_quadratic']:>+8.3f}  "
                  f"shift {r['shift_years']:.3f} yr{'  <-- flagged' if r['flag'] else ''}")
        else:
            print(f"  {r['analysis']:<24} {r['code']:<34} {r['model']:<12} "
                  f"not run: {r['reason']}")


au.sub("Linearity (age at first note squared and span squared added; flag at a shift >= "
       f"{au.LINEARITY_SHIFT_YEARS} yr; not a test)")
print_linearity(linearity_df[linearity_df["level"] == "domain"])
au.sub("Linearity, categories (the same refit and flag)")
print_linearity(linearity_df[linearity_df["level"] == "category"])

au.count_line("median-regression fits that reached the iteration limit (1,000; "
              "estimates kept)", len(QUANTREG_ITERATION_LIMIT))
au.sub("Mean (OLS) vs median AID coefficient by domain "
       f"(flag at a difference >= {MEAN_MEDIAN_FLAG_YRS} yr)")
mean_ = rows_of("first_age_mean")
median_ = rows_of("first_age_median")
mm = mean_[mean_["model"] == "pooled"][["code", "row_label", "status", "coef"]].merge(
    median_[median_["model"] == "pooled"][["code", "status", "coef"]],
    on="code", suffixes=("_mean", "_median"))
mm["difference"] = mm["coef_mean"] - mm["coef_median"]
mm["flag"] = (mm["difference"].abs() >= MEAN_MEDIAN_FLAG_YRS).astype(int)
for _, r in mm.sort_values("difference", key=abs, ascending=False).iterrows():
    print(f"  {r['code']:<34} {fmt_console(r['coef_mean'], '+.2f'):>8} "
          f"{fmt_console(r['coef_median'], '+.2f'):>8} {fmt_console(r['difference'], '+.2f'):>8}"
          f"{'  <-- check distribution' if r['flag'] else ''}")

# -----------------------------------------------------------------------------
au.sub("Descriptive (console)")
# -----------------------------------------------------------------------------
def describe(series, label):
    s = pd.to_numeric(pd.Series(series), errors="coerce").dropna()
    if len(s) < au.MIN_GROUP_N:
        return f"  {label:<40}  n={len(s)} -- below the floor of {au.MIN_GROUP_N}"
    q = s.quantile([0.10, 0.25, 0.50, 0.75, 0.90])
    return (f"  {label:<40}  n={len(s):>5,}  med={s.median():>6.1f}  mean={s.mean():>6.1f}  "
            f"sd={s.std():>5.1f}  p10={q[0.10]:>5.1f}  p25={q[0.25]:>5.1f}  "
            f"p75={q[0.75]:>5.1f}  p90={q[0.90]:>5.1f}")


def mann_whitney(a, b, label):
    a, b = pd.Series(a).dropna(), pd.Series(b).dropna()
    if len(a) < au.MIN_GROUP_N or len(b) < au.MIN_GROUP_N:
        print(f"  {label}: not tested (fewer than {au.MIN_GROUP_N} patients in a group)")
        return
    p = mannwhitneyu(a.astype(float), b.astype(float), alternative="two-sided")[1]
    print(f"  {label}: AID+ median {a.median():.2f} (n={len(a):,})  AID- median "
          f"{b.median():.2f} (n={len(b):,})  difference {a.median() - b.median():+.2f}  "
          f"P={p:.6f}")


print("\n  2b. Share first documented before the AD diagnosis, and median years before")
print("      it among those (descriptive; not adjusted for the span)")
for code in sorted(au.DOMAINS_PRIMARY):
    sub_ = first_dom["primary"][first_dom["primary"]["code"] == code]
    parts = []
    for a, s, lbl in GROUPS:
        g = sub_[(sub_["aid_status"] == a) & (sub_["female"] == s)]
        if len(g) < au.MIN_GROUP_N:
            parts = None
            print(f"  {code:<34} not shown: {lbl} has {len(g)} patients "
                  f"(fewer than {au.MIN_GROUP_N})")
            break
        before = g[g["before_diagnosis"]]["years_to_diagnosis"]
        med = before.median() if len(before) >= au.MIN_GROUP_N else NAN
        parts.append(f"{lbl} {g['before_diagnosis'].mean() * 100:5.1f}% "
                     f"{fmt_console(med, '+.2f')} yr")
    if parts:
        print(f"  {code:<34} " + " | ".join(parts))

print("\n  3a. AD-core onset, each target")
for code, frame in core_targets:
    print(f"\n  --- {code} ---")
    mann_whitney(frame.loc[frame["aid_status"] == 1, "first_age"],
                 frame.loc[frame["aid_status"] == 0, "first_age"], "all patients, Mann-Whitney")
    for s, sx in ((1, "female"), (0, "male")):
        g = frame[frame["female"] == s]
        mann_whitney(g.loc[g["aid_status"] == 1, "first_age"],
                     g.loc[g["aid_status"] == 0, "first_age"], f"{sx}, Mann-Whitney")
    # same comparison, patients with a pre-diagnosis span of at least the floor
    span_floor = au.PREDX_SPAN_FLOOR_5_YRS
    floor_frame = frame[frame["predx_documentation_span_yrs"] >= span_floor]
    mann_whitney(floor_frame.loc[floor_frame["aid_status"] == 1, "first_age"],
                 floor_frame.loc[floor_frame["aid_status"] == 0, "first_age"],
                 f"floor >= {span_floor} yr, all patients, Mann-Whitney")
    for s, sx in ((1, "female"), (0, "male")):
        g = floor_frame[floor_frame["female"] == s]
        mann_whitney(g.loc[g["aid_status"] == 1, "first_age"],
                     g.loc[g["aid_status"] == 0, "first_age"],
                     f"floor >= {span_floor} yr, {sx}, Mann-Whitney")
    for a, s, lbl in GROUPS:
        print(describe(frame.loc[(frame["aid_status"] == a) & (frame["female"] == s),
                                 "first_age"], f"  {lbl}, age at onset"))

print("\n  3b. Years from AD-core onset to the AD diagnosis (positive: onset before it)")
for a, s, lbl in GROUPS:
    g = core[(core["aid_status"] == a) & (core["female"] == s)]
    share = g["before_diagnosis"].mean() * 100 if len(g) >= au.MIN_GROUP_N else NAN
    print(describe(g["years_to_diagnosis"], f"  {lbl}")
          + f"  [before the diagnosis: {fmt_console(share, '.1f')}%]")
for s, sx in ((1, "female"), (0, "male")):
    g = core[core["female"] == s]
    mann_whitney(g.loc[g["aid_status"] == 1, "years_to_diagnosis"],
                 g.loc[g["aid_status"] == 0, "years_to_diagnosis"], f"{sx}, Mann-Whitney")

print("\n  3c. Subjective cognitive burden first documented before the AD diagnosis")
scb = first_dom["primary"][first_dom["primary"]["code"] == "subjective_cognitive_burden"]
for a, s, lbl in GROUPS:
    g = scb[(scb["aid_status"] == a) & (scb["female"] == s)]
    if len(g) < au.MIN_GROUP_N:
        print(f"  {lbl:<12} not shown: {len(g)} patients (fewer than {au.MIN_GROUP_N})")
        continue
    print(f"  {lbl:<12} n={len(g):>4}  before the diagnosis {g['before_diagnosis'].mean() * 100:.1f}%")

print("\n  3d. AD-core domain that set the onset (same-day ties: first in the list)")
first_domain_rows = []
for a, s, lbl in GROUPS:
    g = core[(core["aid_status"] == a) & (core["female"] == s)]
    if len(g) < au.MIN_GROUP_N:
        print(f"  {lbl:<12} not shown: {len(g)} patients (fewer than {au.MIN_GROUP_N})")
        continue
    vc = g["first_ad_core_domain"].value_counts()
    print(f"  {lbl:<12} n={len(g)}: " + ", ".join(
        f"{dom} {cnt} ({cnt / len(g) * 100:.1f}%)" for dom, cnt in vc.items()))
    for dom in au.AD_CORE_DOMAINS:
        first_domain_rows.append({"group": lbl, "n": len(g), "domain": dom,
                                  "n_first": int(vc.get(dom, 0))})

# =============================================================================
# SECTION 4 -- SEQUENCE INVERSION
# =============================================================================
au.section("SECTION 4 -- SEQUENCE INVERSION")
print("  Inverted: the later-expected category first documented strictly before the")
print("  expected-first one; same-day first documentation counts as not inverted.")
print("  Fisher's exact test, AID+ vs AID- (unadjusted): all patients (table) and")
print("  within female and within male patients (by-sex table)")
SEXES = [("all", None), ("female", 1), ("male", 0)]


def pair_frame(first, second, yrs):
    el = first_cat[first_cat["predx_documentation_span_yrs"] >= yrs]
    a = el[el["code"] == first][["person_id", "first_age"]].rename(columns={"first_age": "age_first"})
    b = el[el["code"] == second][["person_id", "first_age"]].rename(columns={"first_age": "age_second"})
    d = a.merge(b, on="person_id", how="inner").merge(
        pop[["person_id", "aid_status", "female"]], on="person_id", how="left",
        validate="one_to_one")
    gap = d["age_second"] - d["age_first"]
    d["inverted"] = (gap < 0).astype(int)
    d["tied"] = (gap == 0).astype(int)
    return d


def fisher_row(key, g):
    """Fisher's exact test of inverted, AID+ vs AID-, in one group of patients.
    Gate: two_by_two_reason (at least MIN_GROUP_N patients in each AID group;
    not every patient, or no patient, inverted in both). The OR is not
    estimable (0 or infinite, with an open CI) when an AID group has no
    inverted, or only inverted, patients: odds_ratio and its CI are then
    blank, or_estimable = 0, and P stands."""
    pos, neg = g[g["aid_status"] == 1], g[g["aid_status"] == 0]
    ip, ineg = int(pos["inverted"].sum()), int(neg["inverted"].sum())
    r = {**key, "n_aid_pos": len(pos), "inverted_aid_pos": ip,
         "tied_aid_pos": int(pos["tied"].sum()), "n_aid_neg": len(neg),
         "inverted_aid_neg": ineg, "tied_aid_neg": int(neg["tied"].sum()),
         "status": "tested", "reason": "", "or_estimable": 0,
         "odds_ratio": NAN, "ci_low": NAN, "ci_high": NAN, "p": NAN}
    why = au.two_by_two_reason(ip, len(pos), ineg, len(neg), "an inverted order")
    if why:
        r.update(status="not_tested", reason=why)
        return r
    orv, lo, hi, p = au.exact_2x2(ip, len(pos) - ip, ineg, len(neg) - ineg)
    r["p"] = p
    if np.isfinite(orv) and orv > 0 and np.isfinite(hi) and lo > 0:
        r.update(or_estimable=1, odds_ratio=orv, ci_low=lo, ci_high=hi)
    return r


def bh_within(df, keys):
    """BH Q within each family (keys) over the tested rows."""
    df = df.copy()
    df["q"] = NAN
    df["fdr_family_size"] = 0
    for _, idx in df.groupby(keys).groups.items():
        t = df.loc[idx]
        t = t.index[(t["status"] == "tested") & t["p"].notna()]
        if len(t):
            df.loc[t, "q"] = au.bh_fdr(df.loc[t, "p"].values)
            df.loc[t, "fdr_family_size"] = len(t)
    return df


inv_rows, ties_rows = [], []
for first, second in au.SEQUENCE_PAIRS:
    d = pair_frame(first, second, INVERSION_FLOOR_YRS)
    key = {"floor_yrs": INVERSION_FLOOR_YRS, "first": first, "second": second,
           "row_label": au.sequence_pair_label(first, second), "fdr_role": "tested"}
    for sx, sel in SEXES:
        g = d if sel is None else d[d["female"] == sel]
        inv_rows.append(fisher_row({**key, "sex": sx}, g))
    # check: all patients, ties excluded (same test; not in any FDR family)
    ties_rows.append(fisher_row({**key, "fdr_role": "check", "sex": "all"},
                                d[d["tied"] == 0]))

# FDR: per sex (all, female, male), over the tested pairs
inv_df = bh_within(pd.DataFrame(inv_rows), ["sex"])
ties_df = pd.DataFrame(ties_rows).assign(q=NAN, fdr_family_size=0)


def print_inversion(df, title):
    print(f"\n  {title}")
    for _, r in df.iterrows():
        cnt = (f"AID+ {r['inverted_aid_pos']}/{r['n_aid_pos']} (tied {r['tied_aid_pos']})  "
               f"AID- {r['inverted_aid_neg']}/{r['n_aid_neg']} (tied {r['tied_aid_neg']})")
        if r["status"] != "tested":
            res = f"not tested: {r['reason']}"
        else:
            orv = (f"OR {r['odds_ratio']:.2f} [{r['ci_low']:.2f}-{r['ci_high']:.2f}]"
                   if r["or_estimable"] else "OR not estimable")
            res = f"{orv}  P={r['p']:.4f}  Q={fmt_console(r['q'], '.4f')}"
        print(f"  {r['row_label']:<46} {r['sex']:<6} {res}  [{cnt}]")


print(f"  Patients with both categories documented and a span of at least "
      f"{INVERSION_FLOOR_YRS} years")
print_inversion(inv_df[inv_df["sex"] == "all"], "All patients (table)")
print_inversion(inv_df[inv_df["sex"] != "all"].sort_values(["sex", "row_label"]),
                "Within female and within male patients (by-sex table; own FDR family per sex)")
print_inversion(ties_df, "Check: ties excluded (all patients)")

# =============================================================================
# SECTION 5 -- SECONDARY DOMAIN DEFINITION CHECK
# =============================================================================
au.section("SECTION 5 -- SECONDARY DOMAIN DEFINITION CHECK")
key_cols = ["person_id", "code", "first_date"]
differs = []
for code in sorted(au.DOMAINS_PRIMARY):
    a = first_dom["primary"].loc[first_dom["primary"]["code"] == code, key_cols]
    b = first_dom["secondary"].loc[first_dom["secondary"]["code"] == code, key_cols]
    if not a.reset_index(drop=True).equals(b.reset_index(drop=True)):
        differs.append(code)
print(f"  Domains whose first documentation differs between the definitions: "
      f"{', '.join(differs) if differs else 'none'}")
CHECK_COLUMNS = ["analysis", "model", "code", "row_label", "status_primary",
                 "status_secondary", "coef_primary", "coef_secondary", "estimate_change",
                 "q_primary", "q_secondary", "became_significant", "lost_significance",
                 "change_ge_small", "change_ge_large"]
check_rows = []
for analysis, _, _ in TIMING_ANALYSES:
    for model in ("pooled",):
        for code in differs:
            sel = ((timing_df["analysis"] == analysis) & (timing_df["model"] == model)
                   & (timing_df["level"] == "domain") & (timing_df["code"] == code))
            p_ = timing_df[sel & (timing_df["definition"] == "primary")].iloc[0]
            s_ = timing_df[sel & (timing_df["definition"] == "secondary")].iloc[0]
            both = p_["status"] == "fitted" and s_["status"] == "fitted"
            change = s_["coef"] - p_["coef"] if both else NAN
            sig_p = pd.notna(p_["q"]) and p_["q"] < Q_THRESHOLD
            sig_s = pd.notna(s_["q"]) and s_["q"] < Q_THRESHOLD
            r = {"analysis": analysis, "model": model, "code": code,
                 "row_label": au.DOMAIN_LABELS[code], "status_primary": p_["status"],
                 "status_secondary": s_["status"], "coef_primary": p_["coef"],
                 "coef_secondary": s_["coef"], "estimate_change": change,
                 "q_primary": p_["q"], "q_secondary": s_["q"],
                 "became_significant": int(both and sig_s and not sig_p),
                 "lost_significance": int(both and sig_p and not sig_s),
                 "change_ge_small": int(both and abs(change) >= ESTIMATE_CHANGE_SMALL_YRS),
                 "change_ge_large": int(both and abs(change) >= ESTIMATE_CHANGE_LARGE_YRS)}
            check_rows.append(r)
            crossed = r["became_significant"] or r["lost_significance"]
            print(f"  {analysis:<24} {model:<12} {code:<10} coef "
                  f"{fmt_console(p_['coef'], '+.2f')} -> {fmt_console(s_['coef'], '+.2f')}  "
                  f"Q {fmt_console(p_['q'], '.4f')} -> {fmt_console(s_['q'], '.4f')}  "
                  f"[{'Q crossed ' + str(Q_THRESHOLD) if crossed else 'Q side unchanged'}; "
                  f"|change| >= {ESTIMATE_CHANGE_LARGE_YRS} yr: "
                  f"{'yes' if r['change_ge_large'] else 'no'}, >= {ESTIMATE_CHANGE_SMALL_YRS} yr: "
                  f"{'yes' if r['change_ge_small'] else 'no'}]")
check_df = pd.DataFrame(check_rows, columns=CHECK_COLUMNS)

# =============================================================================
# SECTION 6 -- SOURCE DATA
# =============================================================================
au.section("SECTION 6 -- SOURCE DATA")
TIMING_SHEET = ["code", "row_label", "fdr_role"] + GROUP_COUNT_COLUMNS + FIT_COLUMNS
CATEGORY_SHEET = ["code", "domain", "row_label", "fdr_role"] + GROUP_COUNT_COLUMNS + FIT_COLUMNS
INV_SHEET = ["floor_yrs", "first", "second", "row_label", "fdr_role", "sex",
             "n_aid_pos", "inverted_aid_pos", "tied_aid_pos", "n_aid_neg", "inverted_aid_neg",
             "tied_aid_neg", "status", "reason", "or_estimable", "odds_ratio", "ci_low",
             "ci_high", "p", "q", "fdr_family_size"]


def family_size(df):
    s = df.loc[df["q"].notna(), "fdr_family_size"]
    return int(s.iloc[0]) if len(s) else 0


def parameters(extra=()):
    rows = [("min_events_per_term", au.MIN_EVENTS_PER_TERM), ("min_group_n", au.MIN_GROUP_N),
            ("predx_span_floor_yrs", au.PREDX_SPAN_FLOOR_5_YRS),
            ("linearity_shift_years", au.LINEARITY_SHIFT_YEARS),
            ("q_threshold", Q_THRESHOLD),
            ("estimate_change_small_yrs", ESTIMATE_CHANGE_SMALL_YRS),
            ("estimate_change_large_yrs", ESTIMATE_CHANGE_LARGE_YRS),
            ("n_aid_pos", int(pop["aid_status"].sum())),
            ("n_aid_neg", int((pop["aid_status"] == 0).sum()))]
    for a, s, lbl in GROUPS:
        arm = "aid_pos" if a else "aid_neg"
        rows.append((f"n_{arm}_{'female' if s else 'male'}",
                     int(((pop["aid_status"] == a) & (pop["female"] == s)).sum())))
    return pd.DataFrame(rows + list(extra), columns=["parameter", "value"])


first_age = rows_of("first_age_mean")
first_age_cat = rows_of("first_age_mean", level="category")
lin_1a = linearity_df[(linearity_df["analysis"] == "first_age_mean")
                      & (linearity_df["level"] == "domain")]
lm = lin_1a[lin_1a["status"] == "run"]
lin_cat = linearity_df[(linearity_df["analysis"] == "first_age_mean")
                       & (linearity_df["level"] == "category")]
lmc = lin_cat[lin_cat["status"] == "run"]
au.write_workbook(os.path.join(SOURCE_DATA_DIR, FIRST_AGE_FILE), {
    "domains": first_age[TIMING_SHEET],
    "categories": first_age_cat[CATEGORY_SHEET],
    "secondary_definition": check_df[check_df["analysis"] == "first_age_mean"],
    "parameters": parameters([
        ("fdr_family_size_domains", family_size(first_age)),
        ("linearity_models_run", len(lm)),
        ("linearity_largest_shift_years", lm["shift_years"].max() if len(lm) else NAN),
        ("linearity_flagged", int(lm["flag"].sum()) if len(lm) else 0),
        ("fdr_family_size_categories", family_size(first_age_cat)),
        ("linearity_models_run_categories", len(lmc)),
        ("linearity_largest_shift_years_categories",
         lmc["shift_years"].max() if len(lmc) else NAN),
        ("linearity_flagged_categories", int(lmc["flag"].sum()) if len(lmc) else 0)]),
})

all_inv = inv_df[inv_df["sex"] == "all"]
sex_inv = inv_df[inv_df["sex"] != "all"]
au.write_workbook(os.path.join(SOURCE_DATA_DIR, INVERSION_FILE), {
    "pairs": all_inv[INV_SHEET],
    "parameters": parameters([("fdr_family_size", family_size(all_inv))]),
})
au.write_workbook(os.path.join(SOURCE_DATA_DIR, INVERSION_BY_SEX_FILE), {
    "pairs": sex_inv[INV_SHEET],
    "parameters": parameters(
        [(f"fdr_family_size_{sx}", family_size(sex_inv[sex_inv["sex"] == sx]))
         for sx in ("female", "male")]),
})

timing_out = timing_df[["analysis", "level", "definition", "model", "term"] + TIMING_SHEET
                       + ["coef_female", "p_female", "coef_age_at_entry", "p_age_at_entry",
                          "coef_span", "p_span", "r_squared"]]
au.write_workbook(os.path.join(SOURCE_DATA_DIR, CHECKS_FILE), {
    "timing_models": timing_out,
    "mean_vs_median": mm,
    "linearity": linearity_df,
    "secondary_definition": check_df,
    "inversion_ties_excluded": ties_df[INV_SHEET],
    "ad_core_first_domain": pd.DataFrame(first_domain_rows,
                                         columns=["group", "n", "domain", "n_first"]),
    "cohort": pd.DataFrame(cohort_rows),
    "parameters": parameters([("mean_median_flag_yrs", MEAN_MEDIAN_FLAG_YRS),
                              ("median_fits_at_iteration_limit",
                               len(QUANTREG_ITERATION_LIMIT))]),
})
print("\n" + au.SEP + "\n  COMPLETE\n" + au.SEP)