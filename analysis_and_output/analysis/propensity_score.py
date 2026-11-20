"""
propensity_score.py
===================
Propensity-score (PS) documentation-balance weights. Runs before the other
analysis scripts; every weighted analysis reads the weights this script writes.

INPUTS
------
  The patient table (AID status, sex, key dates, analysis-population flag
  and coded DLB) and the note table (one row per clinical note, with its date
  and whether it is a cognitive specialty note).

OUTPUTS
-------
  The PS weights (PS_WEIGHTS_PATH): one trimmed weight per patient of the
  analysis population.
  source_data/figure_propensity_score_balance.xlsx
      sheets a, c, e: love plots (all patients, female, male): per covariate,
          n, mean and SD per arm, unweighted and weighted, and both SMDs;
      sheets b, d, f: PS density per arm on one shared grid;
      common_support: per density panel, n and PS range per arm and the
          overlap bounds;
      parameters: the density settings (bandwidth factor, grid points, grid
          padding), the trim percentile per tail and the balance threshold,
          for the figure caption.
  source_data/table_covariate_balance_weighting.xlsx (all_patients) and
  source_data/table_covariate_balance_weighting_by_sex.xlsx (by_sex)
      SMD before and after weighting, its change and the balance label, for
          the seven covariates of the figure (six within each sex); one
          workbook per table.
  source_data/table_propensity_score_weighting_checks.xlsx
      model_coefficients, model_fit, vif, nested_lrt_models,
      nested_lrt_tests, weight_distribution, ess_by_sex, sex_composition,
      common_support, balance_female_dlb_excluded, dlb_removal_female.

PS SPECIFICATION
----------------
  logit P(AID+) = (age_at_entry + log_note_count_predx) * female

  Two documentation covariates, with sex-specific coefficients on both:
    age_at_entry         age at the first dated note (when the patient
                         entered the record);
    log_note_count_predx log(1 + dated notes strictly before the AD
                         diagnosis date).
  The female main effect is present only because the interactions require
  it (marginality); it does not adjust for sex. The model is fitted once,
  pooled over both sexes, so one trim threshold and one weight scale apply
  throughout.

  Two models must not be conflated. This one, the documentation model,
  estimates P(AID+ | how much opportunity the health system had to
  document the patient) and produces weights. The outcome models in the
  analysis scripts estimate the AID contrast, sex-stratified or with an
  AID x sex interaction. The sex finding lives in the outcome models;
  nothing here adjusts it away. Sex-specific coefficients make the
  ascertainment correction equally accurate in both sexes, so a residual
  AID x sex difference cannot be explained as an ascertainment artifact.

  What the weights correct. This is not inverse-probability weighting for a
  causal exposure: AID status is not an intervention and has no index date.
  The weights correct for differential ascertainment. AID+ patients enter
  the record earlier and are documented more densely, so a symptom is more
  likely to be recorded in an AID+ chart whether or not it is more likely to
  be present. Weighting equalizes documentation opportunity, so that a
  residual difference is attributable to the phenotype rather than to the
  number of chances to write it down.

  Inclusion criterion. A covariate enters if it indexes documentation
  opportunity, and is excluded if it indexes disease severity: balancing on
  severity would remove the signal under study. Sex is admitted only as the
  variable the coefficients may vary by, because the estimand is reported
  within each sex and so requires balance within each sex; no other
  covariate is a reporting stratum.

  Excluded, with reasons:
    age_at_dx          marks when AD was diagnosed, not how much opportunity
                       there was to document a symptom before it; its
                       balance is reported.
    note_count         lifetime count; includes post-diagnosis notes, i.e.
                       severity and care-pathway volume.
    note_count_postdx,
    documentation_span_yrs   include post-diagnosis documentation; reported
                       as balance checks only.

  Why the note count is pre-diagnosis. Documentation after the AD diagnosis
  date is driven by the diagnosis itself (work-up, referral, follow-up) and
  by ongoing AID care, so it mixes severity with care pathway. Notes dated
  strictly before the diagnosis date measure the opportunity an early
  symptom had to be captured. A note dated on the diagnosis day is
  post-diagnosis. A patient first seen on or after the diagnosis date
  has 0 pre-diagnosis notes, so log_note_count_predx = 0 encodes "no
  pre-diagnosis documentation"; these patients are kept and counted.

WEIGHTS
-------
  Stabilized: AID+ w = P(AID+) / PS, AID- w = P(AID-) / (1 - PS), with
  P(AID+) the marginal AID+ proportion of the population.
  Trimmed: the stabilized weights clipped at the 1st and 99th percentiles of
  the pooled distribution (both arms together); the same threshold pair
  applies to both arms. This is a fixed percentile trim, not a balance
  search. Because AID- is the large majority, the threshold is set mostly by
  the AID- distribution and AID+ is clipped more often; the clip rates are
  printed and written.

BALANCE
-------
  |weighted SMD| < 0.10 balanced, 0.10 to < 0.20 moderate, >= 0.20 severe
  (a rule of thumb, Austin; not an inferential threshold). Continuous rows
  use the pooled-SD SMD, the binary sex row the proportion-based SMD.
  Balance is checked pooled and within each sex: the estimand is reported by
  sex, so balance must hold within each sex, not only overall (Green and
  Stuart 2014).

  Balance after removing coded DLB (check). The analyses
  of female patients with a coded DLB diagnosis excluded reuse these weights
  (the model is not refitted in the smaller sample). The check recomputes
  the female balance rows in the female patients without coded DLB with the
  same weights, and describes the removed patients per arm: number, share of
  the arm's weight, mean weight of removed and kept patients, and the
  effective sample size before and after. Male patients have no exclusion.

  The nested likelihood-ratio tests compare the model with documentation
  covariates only (main), with a sex main effect added (+sex) and with the
  sex interactions (full, in use). Full vs +sex (2 df) asks whether the
  documentation coefficients differ by sex. If it is not significant, the
  specification still stands on within-sex balance, a design objective. AIC
  and BIC may
  disagree: BIC penalizes log(n) per parameter and targets a true model,
  AIC targets predictive calibration, the relevant aim for a nuisance model.

  VIF is computed on the two covariates only; interaction terms are
  collinear with their components by construction.

Raises on: a missing input or column; a patient of the population
without a dated note (no age at first note, hence no weight); a negative
age at first note; non-convergence; a null PS.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2, gaussian_kde, pearsonr, pointbiserialr
from statsmodels.stats.outliers_influence import variance_inflation_factor

import analysis_utils as au

# =============================================================================
# PATHS (relative to the data folder the script is run from)
# =============================================================================
COHORT_PATIENTS_PATH = ""  # patient table: AID status, sex, key dates (birth, AD diagnosis, death), analysis-population flag and coded DLB
COHORT_NOTES_PATH = ""  # note table: one row per clinical note, with its date (may be missing) and whether it is a cognitive specialty note
PS_WEIGHTS_PATH = ""  # PS weights written here: one trimmed weight per analysis patient
SOURCE_DATA_DIR = ""  # folder for the source-data workbooks written here

FIGURE_BALANCE_FILE = "figure_propensity_score_balance.xlsx"
TABLE_BALANCE_FILE = "table_covariate_balance_weighting.xlsx"
TABLE_BALANCE_BY_SEX_FILE = "table_covariate_balance_weighting_by_sex.xlsx"
TABLE_CHECKS_FILE = "table_propensity_score_weighting_checks.xlsx"

# =============================================================================
# SETTINGS
# =============================================================================
PS_COVARIATES = ["age_at_entry", "log_note_count_predx"]
PS_FORMULA_MAIN = "aid_status ~ age_at_entry + log_note_count_predx"
PS_FORMULA_SEX = "aid_status ~ age_at_entry + log_note_count_predx + female"
PS_FORMULA = "aid_status ~ (age_at_entry + log_note_count_predx) * female"
TRIM_PCT = 1                      # percentile per tail, pooled distribution
BALANCE_THRESHOLD = 0.10
KDE_BW = 0.15
KDE_GRID_POINTS = 500
KDE_GRID_PAD = 0.01               # grid = pooled PS range widened by this on each side
BINARY_ROWS = {"female"}

# (variable, figure label, table label, role), top to bottom.
BALANCE_ROWS = [
    ("age_at_entry", "Age at first note", "Age at first note",
     "Balanced by design"),
    ("log_note_count_predx", "Log pre-Dx note count",
     "Log pre-diagnosis note count", "Balanced by design"),
    ("log_note_count_postdx", "Log post-Dx note count",
     "Log post-diagnosis note count", "Balance check"),
    ("log_note_count", "Log total note count", "Log total note count",
     "Balance check"),
    ("documentation_span_yrs", "Documentation span", "Documentation span",
     "Balance check"),
    ("age_at_dx", "Age at AD diagnosis", "Age at AD diagnosis",
     "Balance check"),
    ("female", "Sex", "Female sex", "Balance check"),
]

TERM_LABELS = {
    "Intercept": "Intercept",
    "age_at_entry": "Age at first note (per year)",
    "log_note_count_predx": "Log pre-diagnosis note count",
    "female": "Female (main effect required by the interactions)",
    "age_at_entry:female": "Age at first note x female",
    "log_note_count_predx:female": "Log pre-diagnosis note count x female",
}
MODEL_LABELS = {
    "main": "Documentation covariates only",
    "sex": "Documentation covariates + sex main effect",
    "full": "Documentation covariates x sex (used)",
}
COMPARISONS = [
    ("full_vs_main", "main", "full",
     "Sex anywhere in the model"),
    ("full_vs_sex", "sex", "full",
     "Sex-specific documentation coefficients"),
    ("sex_vs_main", "main", "sex",
     "Sex main effect only"),
]

# =============================================================================
# LOAD
# =============================================================================
au.section("LOADING DATA")
pts, notes = au.load_population(COHORT_PATIENTS_PATH, COHORT_NOTES_PATH,
                                extra_columns=("coded_dementia_with_lewy_bodies",))

no_dated = pts["note_count_dated"] == 0
au.count_line("population patients without a dated note (no age at first note)",
              no_dated.sum(), len(pts))
if no_dated.any():
    raise ValueError(f"{int(no_dated.sum()):,} population patients have no dated note, "
                     f"so no age at first note and no PS weight")
neg_age = pts["age_at_entry"] < 0
au.count_line("first dated note before birth", neg_age.sum(), len(pts))
if neg_age.any():
    raise ValueError(f"{int(neg_age.sum()):,} patients have a first note before birth")
if (pts["note_count_predx"] < 0).any():
    raise ValueError("negative pre-diagnosis note count")

pts["log_note_count"] = np.log1p(pts["note_count"])
pts["log_note_count_predx"] = np.log1p(pts["note_count_predx"])
pts["log_note_count_postdx"] = np.log1p(pts["note_count_postdx"])
pts["log_cog_notes"] = np.log1p(pts["cog_spec_note_count"])
pts["coded_dlb"] = pts["coded_dementia_with_lewy_bodies"]

# =============================================================================
# CANDIDATE VARIABLES (console only)
# =============================================================================
au.section("CANDIDATE VARIABLES")
CONTINUOUS_VARS = [
    "age_at_dx", "age_at_entry", "documentation_span_yrs", "note_count",
    "log_note_count", "note_count_predx", "log_note_count_predx",
    "note_count_postdx", "log_note_count_postdx", "cog_spec_note_count",
    "log_cog_notes",
]
print(f"  {'Variable':<26} {'n':>7} {'mean':>9} {'median':>9} {'SD':>9} "
      f"{'min':>9} {'max':>9}")
for v in CONTINUOUS_VARS:
    s = pts[v].dropna()
    print(f"  {v:<26} {len(s):>7,} {s.mean():>9.2f} {s.median():>9.2f} "
          f"{s.std():>9.2f} {s.min():>9.2f} {s.max():>9.2f}")


def _pearson(x, y):
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(d) < 10:
        return float("nan")
    return pearsonr(d["x"], d["y"])[0]


def _pointbiserial(b, c):
    d = pd.DataFrame({"b": b, "c": c}).dropna()
    if len(d) < 10 or d["b"].nunique() < 2:
        return float("nan")
    return pointbiserialr(d["b"], d["c"])[0]


au.sub("Pairwise Pearson r, |r| > 0.3")
r_matrix = pd.DataFrame(np.nan, index=CONTINUOUS_VARS, columns=CONTINUOUS_VARS)
for i, v1 in enumerate(CONTINUOUS_VARS):
    for j, v2 in enumerate(CONTINUOUS_VARS):
        if i == j:
            r_matrix.loc[v1, v2] = 1.0
        elif j > i:
            r = _pearson(pts[v1], pts[v2])
            r_matrix.loc[v1, v2] = r
            r_matrix.loc[v2, v1] = r
for i, v1 in enumerate(CONTINUOUS_VARS):
    for j, v2 in enumerate(CONTINUOUS_VARS):
        r = r_matrix.loc[v1, v2]
        if j > i and not pd.isna(r) and abs(r) > 0.3:
            print(f"  {v1:<26} {v2:<26} {r:+.3f}")

au.sub("Screening: SMD vs AID status, largest |r| with another candidate")
CLINICAL_VARS = {"note_count", "log_note_count", "cog_spec_note_count", "log_cog_notes"}
POSTDX_VARS = {"note_count_postdx", "log_note_count_postdx", "documentation_span_yrs"}
for v in CONTINUOUS_VARS:
    s = au.smd(pts[v], pts["aid_status"])
    others = r_matrix.loc[v].drop(v).abs()
    max_r = float(others.max())
    if v in CLINICAL_VARS:
        decision = "excluded (clinical)"
    elif v in POSTDX_VARS:
        decision = "excluded (post-diagnosis)"
    elif abs(s) < 0.1:
        decision = "low priority (balanced)"
    elif max_r > 0.7:
        decision = "conditional (collinear)"
    elif max_r > 0.5:
        decision = "conditional (moderately collinear)"
    else:
        decision = "candidate"
    print(f"  {v:<26} SMD {s:+.3f}  r(female) {_pointbiserial(pts['female'], pts[v]):+.3f}  "
          f"max|r| {max_r:.3f}  {decision}")

au.section("DATA QUALITY")
post_entrant = au.is_post_diagnosis(pts["first_entry_date"], pts["ad_diagnosis_date"])
for lvl in (1, 0):
    m = pts["aid_status"] == lvl
    au.count_line(f"first dated note on or after the AD diagnosis date, "
                  f"{au.AID_LABELS[lvl]}", (post_entrant & m).sum(), m.sum())

# =============================================================================
# PS FIT
# =============================================================================
au.section("PROPENSITY SCORE FIT")
print(f"  Model: {PS_FORMULA}")
ps_data = pts[["aid_status"] + PS_COVARIATES + ["female"]]
if ps_data.isna().any().any():
    raise ValueError("missing PS covariate values in the population")


def fit_logit(formula):
    m = smf.logit(formula, data=ps_data).fit(disp=0)
    if not m.converged:
        raise RuntimeError(f"PS model did not converge: {formula}")
    return m


ps_model = fit_logit(PS_FORMULA)
print(f"  n = {int(ps_model.nobs):,}; log-likelihood {ps_model.llf:.2f}; "
      f"McFadden pseudo R2 {ps_model.prsquared:.4f}")
coef_rows = []
for term in ps_model.params.index:
    b, se, p = ps_model.params[term], ps_model.bse[term], ps_model.pvalues[term]
    coef_rows.append({"term": term, "term_label": TERM_LABELS[term], "coef": b,
                      "se": se, "crit": ps_model.tvalues[term], "p": p})
    print(f"  {term:<30} coef {b:+.4f}  SE {se:.4f}  z {ps_model.tvalues[term]:+.3f}  "
          f"P {p:.4f}")

pts["ps_score"] = ps_model.predict(pts)
if pts["ps_score"].isna().any():
    raise ValueError(f"{int(pts['ps_score'].isna().sum()):,} patients with a null PS")
p_aid_pos = pts["aid_status"].mean()
p_aid_neg = 1 - p_aid_pos
pts["ps_weight_raw"] = np.where(pts["aid_status"] == 1, 1.0 / pts["ps_score"],
                                1.0 / (1.0 - pts["ps_score"]))
pts["ps_weight_stabilized"] = np.where(pts["aid_status"] == 1,
                                       p_aid_pos / pts["ps_score"],
                                       p_aid_neg / (1.0 - pts["ps_score"]))
trim_low = pts["ps_weight_stabilized"].quantile(TRIM_PCT / 100.0)
trim_high = pts["ps_weight_stabilized"].quantile(1 - TRIM_PCT / 100.0)
pts["ps_weight_trimmed"] = pts["ps_weight_stabilized"].clip(lower=trim_low, upper=trim_high)
n_trim_low = int((pts["ps_weight_stabilized"] < trim_low).sum())
n_trim_high = int((pts["ps_weight_stabilized"] > trim_high).sum())
pts["clipped"] = ~np.isclose(pts["ps_weight_stabilized"], pts["ps_weight_trimmed"],
                             atol=1e-12)
print(f"\n  Marginal P(AID+) {p_aid_pos:.4f}")
print(f"  Trim: {TRIM_PCT}st/{100 - TRIM_PCT}th percentile of the pooled stabilized "
      f"weights, lo {trim_low:.4f}, hi {trim_high:.4f}")
print(f"  Trimmed at lower bound {n_trim_low:,}, upper bound {n_trim_high:,}")
for lvl in (1, 0):
    m = pts["aid_status"] == lvl
    print(f"  {au.AID_LABELS[lvl]} clip rate: {int(pts.loc[m, 'clipped'].sum()):,}/"
          f"{int(m.sum()):,} ({100 * pts.loc[m, 'clipped'].mean():.1f}%)")

au.sub("Nested likelihood-ratio tests")
models = {"main": fit_logit(PS_FORMULA_MAIN), "sex": fit_logit(PS_FORMULA_SEX),
          "full": ps_model}
lrt_model_rows = []
for key, m in models.items():
    lrt_model_rows.append({"model": key, "model_label": MODEL_LABELS[key],
                           "k": int(m.df_model) + 1, "loglik": m.llf,
                           "aic": m.aic, "bic": m.bic})
    print(f"  {MODEL_LABELS[key]:<46} k {int(m.df_model) + 1}  logLik {m.llf:.3f}  "
          f"AIC {m.aic:.2f}  BIC {m.bic:.2f}")
lrt_test_rows = []
for key, a, b, label in COMPARISONS:
    lr = 2 * (models[b].llf - models[a].llf)
    df_ = int(models[b].df_model - models[a].df_model)
    p = float(1 - chi2.cdf(lr, df_))
    lrt_test_rows.append({"comparison": key, "comparison_label": label, "df": df_,
                          "chi2": lr, "p": p})
    print(f"  {label:<46} df {df_}  chi2 {lr:.3f}  P {p:.4f}")
p_sex_specific = [r["p"] for r in lrt_test_rows if r["comparison"] == "full_vs_sex"][0]
if p_sex_specific < 0.05:
    print("  Full vs +sex is significant: the documentation coefficients differ by sex.")
else:
    print("  Full vs +sex is not significant: the specification rests on within-sex "
          "balance, not on a sex-specific documentation gradient.")

au.sub("VIF, PS covariates only")
vif_data = sm.add_constant(pts[PS_COVARIATES].copy())
vif_rows = []
for i, col in enumerate(vif_data.columns):
    if col == "const":
        continue
    v = variance_inflation_factor(vif_data.values, i)
    label = "high" if v > 10 else ("elevated" if v > 5 else "low")
    vif_rows.append({"term": col, "term_label": TERM_LABELS[col], "vif": v, "label": label})
    print(f"  {col:<26} VIF {v:.3f}  {label}")

# =============================================================================
# BALANCE
# =============================================================================
au.section("BALANCE: SMD BEFORE AND AFTER WEIGHTING")


def smd_unweighted_row(df, col):
    if col in BINARY_ROWS:
        return au.smd_proportion(df[col], df["aid_status"])
    return au.smd(df[col], df["aid_status"])


def smd_weighted_row(df, col):
    if col in BINARY_ROWS:
        return au.smd_proportion_weighted(df, col)
    return au.smd_weighted(df, col)


def balance_rows(df, rows):
    out = []
    for var, fig_label, tab_label, role in rows:
        u, w = smd_unweighted_row(df, var), smd_weighted_row(df, var)
        out.append({"variable": var, "row_label": tab_label, "figure_label": fig_label,
                    "role": role, "smd_unweighted": u, "smd_weighted": w,
                    "smd_change": w - u, "balance": au.balance_label(w)})
    return out


SEX_ROWS = [r for r in BALANCE_ROWS if r[0] not in BINARY_ROWS]
STRATA = [("All patients", pts, BALANCE_ROWS)] + [
    (au.SEX_LABELS[s], pts[pts["female"] == s], SEX_ROWS) for s in (1, 0)]
balance = {}
for name, df, rows in STRATA:
    balance[name] = balance_rows(df, rows)
    print(f"\n  {name}")
    for r in balance[name]:
        print(f"    {r['row_label']:<34} unweighted {r['smd_unweighted']:+.3f}  "
              f"weighted {r['smd_weighted']:+.3f}  change {r['smd_change']:+.3f}  "
              f"{r['balance']}")
for name, _, _ in STRATA:
    for r in balance[name]:
        if r["role"] == "Balanced by design" and abs(r["smd_weighted"]) >= BALANCE_THRESHOLD:
            print(f"  [NOTE] {name}: PS covariate {r['variable']} has |weighted SMD| "
                  f">= {BALANCE_THRESHOLD}")

au.sub("Effective sample size and clipping")
ess_rows = []
for s in (1, 0):
    for lvl in (1, 0):
        g = pts[(pts["female"] == s) & (pts["aid_status"] == lvl)]
        if len(g) == 0:
            continue
        ess = au.effective_sample_size(g["ps_weight_trimmed"])
        nc = int(g["clipped"].sum())
        ess_rows.append({"sex": au.SEX_LABELS[s], "arm": au.AID_LABELS[lvl], "n": len(g),
                         "ess": ess, "ess_pct": 100 * ess / len(g), "n_clipped": nc,
                         "pct_clipped": 100 * nc / len(g)})
        print(f"  {au.SEX_LABELS[s]:<7} {au.AID_LABELS[lvl]:<5} n {len(g):>6,}  "
              f"ESS {ess:>9.1f} ({100 * ess / len(g):.1f}%)  clipped {nc:,} "
              f"({100 * nc / len(g):.1f}%)")
ess_arm = {lvl: au.effective_sample_size(pts.loc[pts["aid_status"] == lvl,
                                                 "ps_weight_trimmed"]) for lvl in (1, 0)}
n_arm = {lvl: int((pts["aid_status"] == lvl).sum()) for lvl in (1, 0)}

au.sub("Weighted sex composition by arm")
source_pct_female = 100 * pts["female"].mean()
composition_rows = []
for lvl in (1, 0):
    g = pts[pts["aid_status"] == lvl]
    fu = 100 * g["female"].mean()
    fw = 100 * np.average(g["female"], weights=g["ps_weight_trimmed"])
    composition_rows.append({"arm": au.AID_LABELS[lvl], "n": len(g),
                             "pct_female_unweighted": fu, "pct_female_weighted": fw,
                             "pct_female_population": source_pct_female})
    print(f"  {au.AID_LABELS[lvl]:<5} n {len(g):,}  % female unweighted {fu:.2f}  "
          f"weighted {fw:.2f}")
print(f"  Population: {source_pct_female:.2f}% female")

# =============================================================================
# BALANCE AFTER REMOVING CODED DLB (FEMALE PATIENTS; CHECK)
# =============================================================================
au.section("BALANCE AFTER REMOVING CODED DLB: FEMALE PATIENTS, SAME WEIGHTS (CHECK)")
print("  The weights are not refitted: the coded-DLB-excluded analyses reuse them.")
fem_all = pts[pts["female"] == 1]
fem_kept = fem_all[fem_all["coded_dlb"] != 1]
dlb_balance = []
for (_, _, tab_label, role), r_all, r_kept in zip(
        SEX_ROWS, balance["Female"], balance_rows(fem_kept, SEX_ROWS)):
    dlb_balance.append({
        "variable": r_kept["variable"], "row_label": tab_label, "role": role,
        "n_aid_pos": int((fem_kept["aid_status"] == 1).sum()),
        "n_aid_neg": int((fem_kept["aid_status"] == 0).sum()),
        "smd_unweighted": r_kept["smd_unweighted"], "smd_weighted": r_kept["smd_weighted"],
        "smd_change": r_kept["smd_change"], "balance": r_kept["balance"],
        "smd_weighted_female_all": r_all["smd_weighted"],
        "smd_weighted_shift": r_kept["smd_weighted"] - r_all["smd_weighted"]})
dlb_balance = pd.DataFrame(dlb_balance)
for _, r in dlb_balance.iterrows():
    print(f"    {r['row_label']:<34} unweighted {r['smd_unweighted']:+.3f}  weighted "
          f"{r['smd_weighted']:+.3f} (all female patients {r['smd_weighted_female_all']:+.3f})  "
          f"{r['balance']}")
    if r["role"] == "Balanced by design" and abs(r["smd_weighted"]) >= BALANCE_THRESHOLD:
        print(f"  [NOTE] Female, coded DLB excluded: PS covariate {r['variable']} has "
              f"|weighted SMD| >= {BALANCE_THRESHOLD}")
dlb_removal = []
for lvl in (1, 0):
    g = fem_all[fem_all["aid_status"] == lvl]
    rm = g[g["coded_dlb"] == 1]
    kp = g[g["coded_dlb"] != 1]
    w_all = g["ps_weight_trimmed"].sum()
    ess_b = au.effective_sample_size(g["ps_weight_trimmed"])
    ess_a = au.effective_sample_size(kp["ps_weight_trimmed"])
    dlb_removal.append({
        "arm": au.AID_LABELS[lvl], "n": len(g), "n_removed": len(rm), "n_kept": len(kp),
        "pct_removed": 100 * len(rm) / len(g) if len(g) else float("nan"),
        "pct_weight_removed": 100 * rm["ps_weight_trimmed"].sum() / w_all
        if w_all > 0 else float("nan"),
        "mean_weight_removed": rm["ps_weight_trimmed"].mean() if len(rm) else float("nan"),
        "mean_weight_kept": kp["ps_weight_trimmed"].mean() if len(kp) else float("nan"),
        "n_clipped_removed": int(rm["clipped"].sum()),
        "ess_before": ess_b, "ess_after": ess_a,
        "ess_pct_after": 100 * ess_a / len(kp) if len(kp) else float("nan")})
dlb_removal = pd.DataFrame(dlb_removal)
for _, r in dlb_removal.iterrows():
    print(f"  {r['arm']:<5} removed {r['n_removed']:,}/{r['n']:,} ({r['pct_removed']:.1f}%; "
          f"{r['pct_weight_removed']:.1f}% of the weight)  mean weight removed "
          f"{r['mean_weight_removed']:.3f}, kept {r['mean_weight_kept']:.3f}  ESS "
          f"{r['ess_before']:.1f} -> {r['ess_after']:.1f}")

# =============================================================================
# WEIGHTS
# =============================================================================
au.section("SAVING THE WEIGHTS")
ps_out = pts[["person_id", "ps_weight_trimmed"]].copy()
ps_out["person_id"] = ps_out["person_id"].astype("Int64")
au.write_parquet(ps_out, PS_WEIGHTS_PATH)
w = ps_out["ps_weight_trimmed"]
print(f"  Saved: {PS_WEIGHTS_PATH}  ({len(ps_out):,} rows)")
print(f"  Trimmed weight min {w.min():.4f}  median {w.median():.4f}  "
      f"max {w.max():.4f}  mean {w.mean():.4f}")

# =============================================================================
# SOURCE DATA
# =============================================================================
au.section("SOURCE DATA")

LOVE_COLUMNS = [
    "row_label", "variable", "role", "n_aid_pos", "n_aid_neg",
    "mean_aid_pos_unweighted", "sd_aid_pos_unweighted",
    "mean_aid_neg_unweighted", "sd_aid_neg_unweighted", "smd_unweighted",
    "mean_aid_pos_weighted", "sd_aid_pos_weighted",
    "mean_aid_neg_weighted", "sd_aid_neg_weighted", "smd_weighted",
]


def arm_summary(df, col):
    """n, mean and SD per arm, with the variance conventions of the SMDs."""
    binary = col in BINARY_ROWS
    out = {}
    for lvl, tag in ((1, "aid_pos"), (0, "aid_neg")):
        g = df[df["aid_status"] == lvl]
        a = g[col].astype(float)
        out[f"n_{tag}"] = int(len(a))
        m = a.mean()
        out[f"mean_{tag}_unweighted"] = m
        out[f"sd_{tag}_unweighted"] = np.sqrt(m * (1 - m)) if binary else a.std(ddof=1)
        mw = np.average(a, weights=g["ps_weight_trimmed"])
        out[f"mean_{tag}_weighted"] = mw
        out[f"sd_{tag}_weighted"] = (
            np.sqrt(mw * (1 - mw)) if binary
            else np.sqrt(np.average((a - mw) ** 2, weights=g["ps_weight_trimmed"])))
    return out


def smd_from_summary(r, kind):
    num = r[f"mean_aid_pos_{kind}"] - r[f"mean_aid_neg_{kind}"]
    den = np.sqrt((r[f"sd_aid_pos_{kind}"] ** 2 + r[f"sd_aid_neg_{kind}"] ** 2) / 2)
    return num / den if den > 0 else float("nan")


def love_sheet(df, rows):
    recs = []
    for r in rows:
        rec = {"row_label": r["figure_label"], "variable": r["variable"],
               "role": r["role"], **arm_summary(df, r["variable"]),
               "smd_unweighted": r["smd_unweighted"], "smd_weighted": r["smd_weighted"]}
        for kind in ("unweighted", "weighted"):
            if not np.isclose(smd_from_summary(rec, kind), rec[f"smd_{kind}"],
                              rtol=0, atol=1e-9):
                raise RuntimeError(f"{r['variable']}: {kind} SMD re-derived from the "
                                   f"stored means and SDs does not match")
        recs.append(rec)
    return pd.DataFrame(recs)[LOVE_COLUMNS]


figure_sheets = {
    "a": love_sheet(pts, balance["All patients"]),
    "c": love_sheet(pts[pts["female"] == 1], balance["Female"]),
    "e": love_sheet(pts[pts["female"] == 0], balance["Male"]),
}
grid = np.linspace(max(0.0, float(pts["ps_score"].min()) - KDE_GRID_PAD),
                   float(pts["ps_score"].max()) + KDE_GRID_PAD, KDE_GRID_POINTS)
support_rows = []
for panel, stratum, df in (("b", "All patients", pts),
                           ("d", "Females", pts[pts["female"] == 1]),
                           ("f", "Males", pts[pts["female"] == 0])):
    sp = df.loc[df["aid_status"] == 1, "ps_score"].to_numpy(float)
    sn = df.loc[df["aid_status"] == 0, "ps_score"].to_numpy(float)
    figure_sheets[panel] = pd.DataFrame({
        "ps_score": grid,
        "density_aid_pos": gaussian_kde(sp, bw_method=KDE_BW)(grid),
        "density_aid_neg": gaussian_kde(sn, bw_method=KDE_BW)(grid),
    })
    support_rows.append({
        "panel": panel, "stratum": stratum, "n_aid_pos": int(len(sp)),
        "n_aid_neg": int(len(sn)), "ps_min_aid_pos": sp.min(), "ps_max_aid_pos": sp.max(),
        "ps_min_aid_neg": sn.min(), "ps_max_aid_neg": sn.max(),
        "common_support_lower": max(sp.min(), sn.min()),
        "common_support_upper": min(sp.max(), sn.max()),
    })
    print(f"  Common support, {stratum}: [{max(sp.min(), sn.min()):.4f}, "
          f"{min(sp.max(), sn.max()):.4f}]")
common_support = pd.DataFrame(support_rows)
figure_sheets["common_support"] = common_support
figure_sheets["parameters"] = pd.DataFrame([
    {"parameter": "kde_bandwidth_factor", "value": KDE_BW},
    {"parameter": "kde_grid_points", "value": KDE_GRID_POINTS},
    {"parameter": "kde_grid_padding", "value": KDE_GRID_PAD},
    {"parameter": "trim_percentile_per_tail", "value": TRIM_PCT},
    {"parameter": "balance_threshold", "value": BALANCE_THRESHOLD},
])
au.write_workbook(os.path.join(SOURCE_DATA_DIR, FIGURE_BALANCE_FILE),
                  {k: figure_sheets[k] for k in ("a", "b", "c", "d", "e", "f",
                                                 "common_support", "parameters")})

TABLE_COLUMNS = ["row_label", "role", "n_aid_pos", "n_aid_neg", "smd_unweighted",
                 "smd_weighted", "smd_change", "balance"]


def table_sheet(name, df):
    recs = []
    for r in balance[name]:
        recs.append({**r, "n_aid_pos": int((df["aid_status"] == 1).sum()),
                     "n_aid_neg": int((df["aid_status"] == 0).sum())})
    return pd.DataFrame(recs)[TABLE_COLUMNS]


by_sex = []
for s in (1, 0):
    t = table_sheet(au.SEX_LABELS[s], pts[pts["female"] == s])
    t.insert(0, "sex", au.SEX_LABELS[s])
    by_sex.append(t)
au.write_workbook(os.path.join(SOURCE_DATA_DIR, TABLE_BALANCE_FILE), {
    "all_patients": table_sheet("All patients", pts),
})
au.write_workbook(os.path.join(SOURCE_DATA_DIR, TABLE_BALANCE_BY_SEX_FILE), {
    "by_sex": pd.concat(by_sex, ignore_index=True),
})

model_fit = pd.DataFrame([
    {"metric": "n", "value": int(ps_model.nobs)},
    {"metric": "n_aid_pos", "value": n_arm[1]},
    {"metric": "n_aid_neg", "value": n_arm[0]},
    {"metric": "log_likelihood", "value": ps_model.llf},
    {"metric": "pseudo_r2_mcfadden", "value": ps_model.prsquared},
    {"metric": "converged", "value": int(ps_model.converged)},
    {"metric": "marginal_p_aid_pos", "value": p_aid_pos},
])
weight_distribution = pd.DataFrame(
    [{"metric": f"stabilized_weight_p{int(q * 100)}",
      "value": pts["ps_weight_stabilized"].quantile(q)}
     for q in (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)] + [
        {"metric": "trim_percentile_per_tail", "value": TRIM_PCT},
        {"metric": "trim_threshold_low", "value": trim_low},
        {"metric": "trim_threshold_high", "value": trim_high},
        {"metric": "n_trimmed_low", "value": n_trim_low},
        {"metric": "n_trimmed_high", "value": n_trim_high},
        {"metric": "ess_aid_pos", "value": ess_arm[1]},
        {"metric": "ess_aid_neg", "value": ess_arm[0]},
        {"metric": "n_aid_pos", "value": n_arm[1]},
        {"metric": "n_aid_neg", "value": n_arm[0]},
    ])
au.write_workbook(os.path.join(SOURCE_DATA_DIR, TABLE_CHECKS_FILE), {
    "model_coefficients": pd.DataFrame(coef_rows),
    "model_fit": model_fit,
    "vif": pd.DataFrame(vif_rows),
    "nested_lrt_models": pd.DataFrame(lrt_model_rows),
    "nested_lrt_tests": pd.DataFrame(lrt_test_rows),
    "weight_distribution": weight_distribution,
    "ess_by_sex": pd.DataFrame(ess_rows),
    "sex_composition": pd.DataFrame(composition_rows),
    "common_support": common_support,
    "balance_female_dlb_excluded": dlb_balance,
    "dlb_removal_female": dlb_removal,
})

print("\n" + au.SEP + "\n  COMPLETE\n" + au.SEP)