"""
dlb_core_analysis.py
====================
Do patients with Alzheimer's disease (AD) and autoimmune disease (AID+) have
more documented features of dementia with Lewy bodies (DLB; McKeith 2017)
than AID- patients, within each sex; does it hold after removing coded Lewy
body diagnoses; do the elevated features concentrate on the strongest DLB
discriminators; and do the core features co-occur and cluster around the
diagnosis?

Computes every statistic and writes the source data for the DLB tables and
the two DLB figures (one workbook per product) and a checks workbook. Tables
and figures are drawn by the table and figure scripts; nothing here formats a number.

Features
--------
First date per patient and feature = the earliest analyzed mention
(analysis_utils.dlb_feature_first_dates; the
parkinsonism composites are the earliest over their components). A feature
counts as present when its first date is at or below the patient's age
ceiling (the 90th percentile of age at last note among AID- patients of
the same sex); every presence flag and window below uses these dates. RBD
(has_rbd_any) combines confirmed and probable RBD (probable RBD counted
only without confirmed RBD, determined after the ceiling).
Windows (through analysis_utils): parkinsonism (McKeith) = parkinsonism
(bradykinesia, rigidity, rest tremor or parkinsonism documented as such) first
documented no earlier than 1 year before the AD diagnosis date; postural
instability (McKeith) = postural instability first documented in the same
window; visuospatial and executive function, pre/peri-diagnosis = first
documented by 1 year after the AD diagnosis date.

Tested features (POOL): the McKeith 2017 core features
(fluctuating cognition, visual hallucinations, REM sleep behavior disorder
(RBD), parkinsonism (McKeith)), the supportive features analyzed here
(orthostatic hypotension, syncope, urinary incontinence, constipation,
hypersomnia, hyposmia, auditory hallucinations, systematized delusions,
postural instability (McKeith), neuroleptic sensitivity, transient
unresponsiveness) and the cognitive profile (visuospatial and executive
function pre/peri-diagnosis; McKeith 2017 places these under the essential
feature, dementia, not among the supportive features). The class is written
to the source data (mckeith_class) and used in no analysis.
Per-group floor: a feature enters an FDR family only
with >= MIN_EVENTS_PER_TERM x GROUP_FLOOR_TERMS patients in the rarer outcome
class within each compared group (AID+ and AID- of the block; coded DLB and AD
only for the discriminant; analysis_utils.group_floor_reason). Below it the
estimate is still fitted and written (fdr_role "below_group_floor", no Q), so
every table records every feature. The floor decides membership of the
multivariable discriminant too; the script stops if a core feature fails it,
because dropping a core feature would change every coefficient.

Analyses
--------
- DLB features: the tested pool, AID+ vs AID-, per sex and arm:
  female, all patients; female, coded DLB excluded; female, coded DLB and PD
  excluded; male, all patients. Logistic regression
  feature ~ aid_status + age_at_dx weighted by ps_weight_trimmed (binomial
  GLM, var_weights, HC0). BH within each sex x arm over the features that
  meet the per-group floor.
  The table workbook holds the arms in TABLE_ARMS (female all, female coded
  DLB excluded, male all); the coded DLB and PD excluded arm is a check
  (checks sheet features_dlb_pd_excluded). Each arm is its
  own BH family, so dropping it from the table changes no other Q.
  Parkinsonism components are described only (counts).
- Sex: the ratio of the AID odds ratios, female / male, of the tested features
  (all patients): exp(b_female - b_male), SE sqrt(se_w^2 + se_m^2), the Wald
  test of the AID x sex interaction of the fully sex-stratified model (=
  Cochran's Q, 1 df). Gate: both strata fitted and the four-group rule. BH
  over the features that also meet the per-group floor in both sexes.
- Co-occurrence of the core triad (parkinsonism, fluctuating cognition,
  visual hallucinations): pairs and the triple, Wilson intervals, weighted
  odds ratio, BH within sex x arm over the four; among combination-positive
  AID+ female patients, how many of the features were first documented by 1 year after
  the AD diagnosis date; RBD by the number of core features.
- Co-occurrence beyond chance (log only): within AID+ female
  patients (all; coded DLB excluded), observed patients with each pair and
  the triad against the count expected if the features were independent:
  each feature permuted independently within strata of total note count x
  documentation span (5 x 5 bins cut within the group; strata under 10
  patients merged into their note-count neighbour), 10,000 permutations,
  fixed seed; the expected count is the permutation mean, with a 2.5-97.5
  percentile interval and a two-sided P = (k + 1) / (B + 1); plus the
  expected count under unstratified independence. No FDR; a set with fewer
  than 10 observed patients is skipped.
- Timing composition, co-occurrence figure panel b (log only): among AID+
  female patients with each pair and the triad, without vs with coded DLB,
  the all / some / none composition of the features first documented by 1
  year after the AD diagnosis (the figure workbook's b sheet, built in this
  section and written in section 9) by the Fisher-Freeman-Halton exact test, and the number in the
  window by Mann-Whitney U; BH across the combinations per test. A
  combination with fewer than 10 patients in either group is skipped.
- Discriminant: coded DLB vs AD only (no coded DLB, PD or vascular dementia),
  unweighted logistic regression on the tested features that meet the
  per-group floor in coded DLB and in AD only (multivariable) and on each
  feature alone (bivariable); BH within each model type over the features that
  meet the floor; in-sample and 5-fold cross-validated AUC.
- DLB feature-burden score: SCORE_CORE_POINTS x core features (fluctuating
  cognition, visual hallucinations, RBD, parkinsonism (McKeith)) + the other
  tested features (supportive and cognitive-profile; named "supportive" in
  the score settings), each counted, up to
  SCORE_SUPPORTIVE_CAP. The three settings are derived from the lists: the
  cap is the number of supportive features (all counted), one core feature is
  worth the cap + 1 (so no number of supportive features reaches one core
  feature), and the maximum SCORE_MAX = core points x the number of core
  features + the cap (14, 13 and 69 with 4 core and 13 supportive features).
  The curves run over 0..SCORE_MAX. Share of female patients scoring >= x,
  AID+ and AID-, all patients and coded DLB excluded, with percentile
  bootstrap bands; the coded DLB reference curve (both sexes); Mann-Whitney U.
- McKeith features not used: prevalence by AID x sex group and the fixed,
  literature-based reason each is left out (NOT_USED).
- Every analyzed feature must be documented in at least one patient; the
  script stops otherwise (a feature absent from the input would read as zero).
Gates: analysis_utils.model_floor_reason (models), two_by_two_reason (Fisher),
group_minimum_reason (Mann-Whitney U). Described only (checks): age at first
visuospatial and executive documentation, and the visuospatial timing
relative to the AD diagnosis.

Inputs: the patient table, the note table, the DLB feature table, the symptom
table and the propensity-score weights (paths below).
"""

import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.optimize import brentq
from scipy.stats import chi2, fisher_exact, mannwhitneyu, norm
from scipy.stats.contingency import odds_ratio
from statsmodels.genmod.families import Binomial

import analysis_utils as au

warnings.filterwarnings("ignore")   # after the imports: statsmodels resets its own filters

# =============================================================================
# PATHS (relative to the data folder the script is run from)
# =============================================================================
COHORT_PATIENTS_PATH = ""  # patient table: AID status, sex, key dates (birth, AD diagnosis, death), analysis-population flag and coded DLB, Parkinson's disease and vascular dementia diagnoses
COHORT_NOTES_PATH = ""  # note table: one row per clinical note, with its date and whether it is a cognitive-specialty note
DLB_MENTIONS_PATH = ""  # DLB feature table: one row per documented DLB feature, with its date
SYMPTOM_RECORDS_PATH = ""  # symptom table: one row per documented symptom, with its category
PS_WEIGHTS_PATH = ""  # propensity-score weights, one per analysis patient
SOURCE_DATA_DIR = ""  # folder for the source-data workbooks written here
COEFFICIENTS_OUT_PATH = ""  # written here: the multivariable discriminant coefficient of each tested feature (blank when not fitted); read by aid_condition_analysis.py

FEATURES_FILE = "table_dlb_features.xlsx"
SEX_FILE = "table_dlb_sex_heterogeneity.xlsx"
NOT_USED_FILE = "table_dlb_features_not_used.xlsx"
DISCRIMINANT_FILE = "table_dlb_discriminant_coefficients.xlsx"
FIGURE_SPECIFICITY_FILE = "figure_dlb_feature_specificity.xlsx"
FIGURE_COOCCURRENCE_FILE = "figure_dlb_core_feature_cooccurrence.xlsx"
CHECKS_FILE = "checks_dlb_core.xlsx"

# =============================================================================
# SETTINGS
# =============================================================================
WCOL = "ps_weight_trimmed"
MCKEITH_WINDOW_YRS = au.DLB_MCKEITH_WINDOW_YRS    # parkinsonism from 1 yr before AD dx
PRE_PERI_WINDOW_YRS = au.DLB_PRE_PERI_WINDOW_YRS  # first documented by 1 yr after AD dx
TOST_MARGINS = [1.2, 1.3, 1.5]
# The feature-burden score settings (SCORE_CORE_POINTS, SCORE_SUPPORTIVE_CAP,
# SCORE_MAX) are derived from the feature lists, after SCORE_SUPPORTIVE below.
N_BOOT = 2000
BOOT_SEED = 20260813
SCORE_REFERENCE_MIN_N = 20
SCORE_SHARES_PRINTED = (10, 20)   # console only: shares at score >= x; one and two core features' points are added
COOC_CHANCE_PERMUTATIONS = 10000   # co-occurrence beyond chance (log only): permutations
COOC_CHANCE_SEED = 20261104        # its seed (AID+ female all patients; + 1 coded DLB excluded)
COOC_CHANCE_QUANTILES = 5          # bins of note count and of documentation span (5 x 5 strata)
CV_FOLDS = 5
CV_SEED = 0
GROUP_FLOOR_TERMS = 2      # per-group floor = MIN_EVENTS_PER_TERM x the feature model's 2 terms
NAN = float("nan")

# Every DLB feature with a first date in the patient table.
FEATURES = au.DLB_FLAG_FEATURES
# The tested pool: (flag column, readable label). The multivariable discriminant
# keeps this order for the features that meet the per-group floor.
POOL = [("has_fluct", "Fluctuating cognition"),
        ("has_vis_hall", "Visual hallucinations"),
        ("has_park_all_mckeith", "Parkinsonism (McKeith)"),
        ("has_rbd_any", "REM sleep behavior disorder (RBD)"),
        ("has_ortho", "Orthostatic hypotension"),
        ("has_vis_seq", "Visuospatial, pre/peri-diagnosis"),
        ("has_aud_hall", "Auditory hallucinations"),
        ("has_incont", "Urinary incontinence"),
        ("has_hyper", "Hypersomnia"),
        ("has_exec_seq", "Executive function, pre/peri-diagnosis"),
        ("has_syncope", "Syncope"),
        ("has_constipation", "Constipation"),
        ("has_pi_mckeith", "Postural instability (McKeith)"),
        ("has_hyposmia", "Hyposmia"),
        ("has_systematized_delusions", "Systematized delusions"),
        ("has_neuro_sens", "Neuroleptic sensitivity"),
        ("has_transient_unresponsiveness", "Transient unresponsiveness")]
POOL_COLS = [c for c, _ in POOL]
# McKeith 2017 core features
CORE_FEATURES = ["has_fluct", "has_vis_hall", "has_rbd_any", "has_park_all_mckeith"]
# Visuospatial and executive function: the cognitive profile (McKeith 2017 places them
# under the essential feature, dementia, not among the supportive features)
COGNITIVE_PROFILE_FEATURES = ["has_vis_seq", "has_exec_seq"]
# The class of each tested feature, written to the source data (mckeith_class); a label
# only: no analysis uses it (the score uses SCORE_CORE and SCORE_SUPPORTIVE below)
MCKEITH_CLASS = {c: "core" if c in CORE_FEATURES
                 else "cognitive_profile" if c in COGNITIVE_PROFILE_FEATURES
                 else "supportive" for c in POOL_COLS}
# Parkinsonism variants and components, described only
DESCRIBED = [("has_park_all_mckeith", "Parkinsonism (McKeith)"),
             ("has_park_core_mckeith", "Parkinsonism (McKeith), without standalone terms"),
             ("has_park_all_ever", "Parkinsonism, any time"),
             ("has_park_all_pdd_type", "Parkinsonism, first documented more than 1 year "
                                       "before AD diagnosis"),
             ("has_brady_or_rigid_mckeith", "Bradykinesia or rigidity (McKeith)"),
             ("has_brady_mckeith", "Bradykinesia (McKeith)"),
             ("has_rigid_mckeith", "Rigidity (McKeith)"),
             ("has_tremor_mckeith", "Rest tremor (McKeith)")]
LABEL = dict(POOL + DESCRIBED)
# Arms: (code, columns whose 1 excludes the patient); female all, male "all" only
ARMS = [("all", []), ("dlb_excluded", ["coded_dlb"]),
        ("dlb_pd_excluded", ["coded_dlb", "coded_pd"])]
# (sex, arm) pairs in the DLB features table workbook; the rest go to the checks
TABLE_ARMS = [("Female", "all"), ("Female", "dlb_excluded"), ("Male", "all")]
SEXES = [(1, "Female"), (0, "Male")]
CORE_TRIAD = ["has_park_all_mckeith", "has_fluct", "has_vis_hall"]
# Co-occurrence: (set, code, columns)
COMBOS = [("primary", "park_fluct", ["has_park_all_mckeith", "has_fluct"]),
          ("primary", "park_vis_hall", ["has_park_all_mckeith", "has_vis_hall"]),
          ("primary", "fluct_vis_hall", ["has_fluct", "has_vis_hall"]),
          ("primary", "triad", ["has_park_all_mckeith", "has_fluct", "has_vis_hall"]),
          ("context", "brady_rigid_fluct", ["has_brady_or_rigid_mckeith", "has_fluct"]),
          ("context", "brady_rigid_vis_hall", ["has_brady_or_rigid_mckeith", "has_vis_hall"]),
          ("context", "brady_rigid_triad",
           ["has_brady_or_rigid_mckeith", "has_fluct", "has_vis_hall"])]
COMBO_LABEL = {"park_fluct": "Parkinsonism and fluctuating cognition",
               "park_vis_hall": "Parkinsonism and visual hallucinations",
               "fluct_vis_hall": "Fluctuating cognition and visual hallucinations",
               "triad": "All three (parkinsonism, fluctuating cognition, visual hallucinations)",
               "brady_rigid_fluct": "Bradykinesia or rigidity and fluctuating cognition",
               "brady_rigid_vis_hall": "Bradykinesia or rigidity and visual hallucinations",
               "brady_rigid_triad": "All three (bradykinesia or rigidity, fluctuating "
                                    "cognition, visual hallucinations)"}
# The window proxy of each core feature in the co-occurrence timing
PRE_PERI = {"has_park_all_mckeith": "has_park_all_preperi",
            "has_brady_or_rigid_mckeith": "has_brady_or_rigid_preperi",
            "has_fluct": "has_fluct_preperi", "has_vis_hall": "has_vis_hall_preperi"}
SCORE_CORE = CORE_FEATURES
SCORE_SUPPORTIVE = [c for c in POOL_COLS if c not in CORE_FEATURES]
# Feature-burden score, from the lists: every supportive
# feature counts (cap = their number), one core feature outweighs all of them
# together (core points = cap + 1), and the maximum follows.
SCORE_SUPPORTIVE_CAP = len(SCORE_SUPPORTIVE)
SCORE_CORE_POINTS = SCORE_SUPPORTIVE_CAP + 1
SCORE_MAX = SCORE_CORE_POINTS * len(SCORE_CORE) + SCORE_SUPPORTIVE_CAP
# McKeith features not used: (code, label, source, feature or category, reason).
# The reasons are fixed from the literature (the table script holds their wording):
#   common_in_ad        anxiety, apathy, depression: common in AD (Zhao 2016)
#   common_in_dementia  repeated falls: common in dementia generally (Allan
#                       2009); overlaps the tested parkinsonism, postural
#                       instability, orthostatic hypotension and syncope
NOT_USED = [("repeated_falls", "Repeated falls", "mentions", "repeated_falls",
             "common_in_dementia"),
            ("apathy", "Apathy", "category", "apathy", "common_in_ad"),
            ("anxiety", "Anxiety", "category", "anxiety", "common_in_ad"),
            ("depression", "Depression", "category", "depression", "common_in_ad")]
NOT_USED_REASONS = ["common_in_ad", "common_in_dementia"]
GROUPS = [(1, 1, "aid_pos_female"), (0, 1, "aid_neg_female"), (1, 0, "aid_pos_male"),
          (0, 0, "aid_neg_male")]
GROUP_LABEL = {"aid_pos_female": "AID+ female", "aid_neg_female": "AID- female",
               "aid_pos_male": "AID+ male", "aid_neg_male": "AID- male"}


def arms_for(female):
    """Female: every arm; male: all patients only (no exclusion arm)."""
    return ARMS if female == 1 else ARMS[:1]


def apply_arm(d, cols):
    keep = pd.Series(True, index=d.index)
    for c in cols:
        keep &= d[c] != 1
    return d[keep]


def fmt_console(v, spec):
    return "n/a" if pd.isna(v) else format(v, spec)


def bh_by(df, family_cols, eligible):
    """BH q within each family over the eligible rows."""
    df = df.copy()
    df["q"] = NAN
    df["fdr_family_size"] = 0
    if not len(df):
        return df
    for _, idx in df.groupby(family_cols, dropna=False, sort=False).groups.items():
        t = [i for i in idx if eligible.loc[i] and pd.notna(df.loc[i, "p"])]
        if t:
            df.loc[t, "q"] = au.bh_fdr(df.loc[t, "p"].values)
            df.loc[t, "fdr_family_size"] = len(t)
    return df


def check_ci(df, label):
    f = df[(df["status"] == "fitted") & df["coef"].notna()]
    lo, hi = np.exp(f["coef"] - f["crit"] * f["se"]), np.exp(f["coef"] + f["crit"] * f["se"])
    if not (np.allclose(f["ci_low"], lo, rtol=1e-7) and np.allclose(f["ci_high"], hi, rtol=1e-7)):
        raise RuntimeError(f"{label}: CI does not equal exp(coef -/+ crit x SE)")


def exact_2x2(a, b, c, d):
    """Conditional MLE odds ratio, exact 95% CI, Fisher's exact P of
    [[a, b], [c, d]] (rows AID+ / AID-, columns yes / no)."""
    table = [[int(a), int(b)], [int(c), int(d)]]
    res = odds_ratio(table, kind="conditional")
    ci = res.confidence_interval(0.95)
    return float(res.statistic), float(ci.low), float(ci.high), float(fisher_exact(table)[1])


def exact_row(d, y, event):
    """AID+ vs AID- on the 0/1 column y: counts, gate (two_by_two_reason) and,
    when tested, the exact 2x2 values."""
    pos, neg = d[d["aid_status"] == 1], d[d["aid_status"] == 0]
    e1, e2 = int(pos[y].sum()), int(neg[y].sum())
    r = {"events_aid_pos": e1, "n_aid_pos": len(pos), "events_aid_neg": e2,
         "n_aid_neg": len(neg), "status": "", "reason": "", "odds_ratio": NAN, "ci_low": NAN,
         "ci_high": NAN, "or_estimable": 0, "p": NAN}
    why = au.two_by_two_reason(e1, len(pos), e2, len(neg), event)
    if why:
        r.update(status="not_tested", reason=why)
        return r
    orv, lo, hi, p = exact_2x2(e1, len(pos) - e1, e2, len(neg) - e2)
    r.update(status="tested", p=p)
    if np.isfinite(orv) and orv > 0:
        r.update(odds_ratio=orv, ci_low=lo, ci_high=hi, or_estimable=1)
    return r


def weighted_or(d, y):
    """PS-weighted logistic regression y ~ aid_status + age_at_dx (binomial
    GLM, var_weights, HC0), the AID term. Gate: model_floor_reason, 2 terms."""
    d = d[["aid_status", "age_at_dx", WCOL]].assign(outcome=d[y].astype(int).values)
    pos, neg = d[d["aid_status"] == 1], d[d["aid_status"] == 0]
    out = {"events_aid_pos": int(pos["outcome"].sum()), "n_aid_pos": len(pos),
           "events_aid_neg": int(neg["outcome"].sum()), "n_aid_neg": len(neg),
           "n_model": len(d), "status": "", "reason": "", "coef": NAN, "se": NAN, "crit": NAN,
           "odds_ratio": NAN, "ci_low": NAN, "ci_high": NAN, "p": NAN}
    why = au.model_floor_reason(d["outcome"], d["aid_status"], 2)
    if why:
        out.update(status="not_fitted", reason=why)
        return out
    try:
        m = smf.glm("outcome ~ aid_status + age_at_dx", data=d, family=Binomial(),
                    var_weights=d[WCOL]).fit(cov_type="HC0")
    except Exception as e:
        out.update(status="failed", reason=f"fit error: {type(e).__name__}")
        return out
    ci = m.conf_int().loc["aid_status"]
    out.update(status="fitted", coef=float(m.params["aid_status"]),
               se=float(m.bse["aid_status"]), crit=float(norm.ppf(0.975)),
               odds_ratio=float(np.exp(m.params["aid_status"])),
               ci_low=float(np.exp(ci[0])), ci_high=float(np.exp(ci[1])),
               p=float(m.pvalues["aid_status"]))
    return out


def group_floor(d, c, group_col, labels):
    """analysis_utils.group_floor_reason of feature c across the two values of
    group_col (1, 0) in d: "" when the feature may enter an FDR family."""
    return au.group_floor_reason(d[c], d[group_col], GROUP_FLOOR_TERMS, labels)


def unpenalized_logistic(max_iter):
    """scikit-learn logistic regression without penalty (penalty=None before
    1.8, C=inf from 1.8; the same fit)."""
    import sklearn
    from sklearn.linear_model import LogisticRegression
    major, minor = (int(x) for x in sklearn.__version__.split(".")[:2])
    if (major, minor) >= (1, 8):
        return LogisticRegression(C=np.inf, max_iter=max_iter)
    return LogisticRegression(penalty=None, max_iter=max_iter)


# =============================================================================
# SECTION 1 -- POPULATION, FIRST DATES, AGE CEILING, FLAGS
# =============================================================================
au.section("SECTION 1 -- POPULATION, FEATURE FIRST DATES, AGE CEILING, FLAGS")
pop, notes = au.load_population(COHORT_PATIENTS_PATH, COHORT_NOTES_PATH,
                                extra_columns=("coded_dementia_with_lewy_bodies",
                                               "coded_parkinsons_disease",
                                               "coded_vascular_dementia"))
pop = au.load_ps_weights(PS_WEIGHTS_PATH, pop)
pop = pop.sort_values("person_id", kind="mergesort").reset_index(drop=True)
pop["coded_dlb"] = pop["coded_dementia_with_lewy_bodies"]
pop["coded_pd"] = pop["coded_parkinsons_disease"]
pop["coded_vd"] = pop["coded_vascular_dementia"]
pop["sex"] = pop["female"].map(au.SEX_LABELS)

mentions = au.load_dlb_feature_mentions(DLB_MENTIONS_PATH, pop, notes)
pop, ceil = au.dlb_patient_flags(pop, mentions)
# A feature that no patient has would otherwise enter every table as zero
# patients.
empty = [c for c in POOL_COLS + ["has_" + n[3] for n in NOT_USED if n[2] == "mentions"]
         if pop[c].sum() == 0]
if empty:
    raise RuntimeError(f"No patient has {empty}: check the "
                       f"input data")


def group_mask(a, s):
    return (pop["aid_status"] == a) & (pop["female"] == s)


au.sub("Patients whose only documentation of a feature is past the age ceiling (removed / "
       "with the feature before the ceiling)")
ceiling_rows = []
print("  {:<28}".format("Feature") + "".join(f"{GROUP_LABEL[g]:>16}" for _, _, g in GROUPS))
# RBD is reported once, as the aggregate of confirmed and probable RBD
pop["d_rbd_any"] = pop[["d_rbd_conf", "d_rbd_prob"]].min(axis=1)
pop["t_rbd_any"] = pop[["t_rbd_conf", "t_rbd_prob"]].min(axis=1)
for f in [x for x in FEATURES if x not in ("rbd_conf", "rbd_prob")] + ["rbd_any"]:
    line = f"  {f:<28}"
    for a, s, g in GROUPS:
        m = group_mask(a, s)
        raw = int(pop.loc[m, "d_" + f].notna().sum())
        lost = int((pop.loc[m, "d_" + f].notna() & pop.loc[m, "t_" + f].isna()).sum())
        ceiling_rows.append({"feature": f, "group": g, "n_group": int(m.sum()),
                             "n_with_feature_before_ceiling": raw, "n_removed_by_ceiling": lost})
        line += f"{f'{lost:,}/{raw:,}':>16}"
    print(line)

count_rows = []
for c in POOL_COLS + [c for c, _ in DESCRIBED[1:]]:
    for a, s, g in GROUPS:
        m = group_mask(a, s)
        count_rows.append({"feature": c[4:], "row_label": LABEL[c], "group": g,
                           "events": int(pop.loc[m, c].sum()), "n": int(m.sum())})
for a, s, g in GROUPS:
    m = group_mask(a, s)
    print(f"  {GROUP_LABEL[g]:<11} n = {int(m.sum()):,}; coded DLB {int(pop.loc[m, 'coded_dlb'].sum()):,}"
          f", coded PD {int(pop.loc[m, 'coded_pd'].sum()):,}, coded vascular dementia "
          f"{int(pop.loc[m, 'coded_vd'].sum()):,}")

# =============================================================================
# SECTION 2 -- DLB FEATURES, AID+ VS AID- (WEIGHTED), PER SEX AND ARM
# =============================================================================
au.section("SECTION 2 -- DLB FEATURES: AID+ VS AID- (PS-WEIGHTED LOGISTIC), PER SEX AND ARM")
print("  feature ~ aid_status + age_at_dx, weights ps_weight_trimmed (HC0). BH within sex x arm")
print(f"  over the {len(POOL)} tested features that meet the per-group floor "
      f"({au.MIN_EVENTS_PER_TERM * GROUP_FLOOR_TERMS} in the rarer class among AID+ and among AID-);")
print("  below it: fitted and written, no Q.")
feat_rows, desc_rows, peri_rows = [], [], []
for s, sex in SEXES:
    for arm, cols in arms_for(s):
        d = apply_arm(pop[pop["female"] == s], cols)
        for c, lbl in POOL:
            gf = group_floor(d, c, "aid_status", ("AID+", "AID-"))
            feat_rows.append({"sex": sex, "arm": arm, "feature": c[4:], "row_label": lbl,
                              "mckeith_class": MCKEITH_CLASS[c],
                              "fdr_role": "below_group_floor" if gf else "tested",
                              "group_floor_reason": gf, **weighted_or(d, c)})
        for c, lbl in DESCRIBED:
            pos, neg = d[d["aid_status"] == 1], d[d["aid_status"] == 0]
            desc_rows.append({"sex": sex, "arm": arm, "feature": c[4:], "row_label": lbl,
                              "events_aid_pos": int(pos[c].sum()), "n_aid_pos": len(pos),
                              "events_aid_neg": int(neg[c].sum()), "n_aid_neg": len(neg)})
        for c, lbl in (("has_vis_hall_preperi", "Visual hallucinations, pre/peri-diagnosis"),
                       ("has_fluct_preperi", "Fluctuating cognition, pre/peri-diagnosis")):
            peri_rows.append({"sex": sex, "arm": arm, "feature": c[4:], "row_label": lbl,
                              **weighted_or(d, c)})
features = pd.DataFrame(feat_rows)
features = bh_by(features, ["sex", "arm"],
                 (features["fdr_role"] == "tested") & (features["status"] == "fitted"))
check_ci(features, "DLB features")
peri = pd.DataFrame(peri_rows)
check_ci(peri, "pre/peri-diagnosis")
descriptive = pd.DataFrame(desc_rows)
for (sex, arm), g in features.groupby(["sex", "arm"], sort=False):
    print(f"\n  {sex} | {arm}  (AID+ n = {g['n_aid_pos'].iloc[0]:,}, AID- n = "
          f"{g['n_aid_neg'].iloc[0]:,})"
          + ("" if (sex, arm) in TABLE_ARMS else "  [check only, not in the table]"))
    for _, r in g.iterrows():
        est = (f"OR {r['odds_ratio']:.2f} [{r['ci_low']:.2f}-{r['ci_high']:.2f}]"
               if r["status"] == "fitted" else f"{r['status']}: {r['reason']}")
        q = "Q=" + fmt_console(r["q"], ".4f") if r["fdr_role"] == "tested" \
            else f"no Q: {r['group_floor_reason']}"
        print(f"  {r['row_label']:<42} {r['events_aid_pos']:>6,} {r['events_aid_neg']:>6,}  "
              f"{est:<40} {q}")
au.sub("Parkinsonism and RBD, described (patients with the feature, AID+ / AID-)")
for _, r in descriptive.iterrows():
    print(f"  {r['sex']:<7} {r['arm']:<16} {r['row_label']:<62} "
          f"{r['events_aid_pos']:>6,}/{r['n_aid_pos']:<6,} {r['events_aid_neg']:>6,}/"
          f"{r['n_aid_neg']:,}")

# =============================================================================
# SECTION 3 -- SEX: RATIO OF THE AID ODDS RATIOS, FEMALE / MALE
# =============================================================================
au.section("SECTION 3 -- SEX: RATIO OF THE AID ODDS RATIOS, FEMALE / MALE (ALL PATIENTS)")
print("  ratio = exp(b_female - b_male), SE = sqrt(se_female^2 + se_male^2): the Wald test of the")
print("  AID x sex interaction of the fully sex-stratified model (= Cochran's Q, 1 df).")


def tost_p(beta, se, margin):
    if pd.isna(beta) or pd.isna(se) or se <= 0:
        return NAN
    lm = np.log(margin)
    return float(max(1 - norm.cdf((beta + lm) / se), 1 - norm.cdf((lm - beta) / se)))


def min_detectable_or(n_pos, n_neg, p0, alpha=0.05, power=0.80):
    if n_pos < au.MIN_GROUP_N or n_neg < au.MIN_GROUP_N or pd.isna(p0) or p0 <= 0 or p0 >= 1:
        return NAN
    za = norm.ppf(1 - alpha / 2)
    nh = 2.0 / (1.0 / n_pos + 1.0 / n_neg)

    def power_gap(o):
        o1 = p0 / (1 - p0) * o
        p1 = o1 / (1 + o1)
        pb = (p0 + p1) / 2
        se0 = np.sqrt(2 * pb * (1 - pb) / nh)
        se1 = np.sqrt(p0 * (1 - p0) / nh + p1 * (1 - p1) / nh)
        return norm.cdf((abs(p1 - p0) - za * se0) / se1) - power

    try:
        return float(brentq(power_gap, 1.001, 50.0))
    except ValueError:
        return NAN


base = features[features["arm"] == "all"].set_index(["sex", "feature"])
sex_rows, sex_extra = [], []
crit = float(norm.ppf(0.975))
for c, lbl in POOL:
    f, m = base.loc[("Female", c[4:])], base.loc[("Male", c[4:])]
    rec = {"feature": c[4:], "row_label": lbl, "mckeith_class": MCKEITH_CLASS[c]}
    for a, s, g in GROUPS:
        gm = group_mask(a, s)
        rec[f"events_{g}"] = int(pop.loc[gm, c].sum())
        rec[f"n_{g}"] = int(gm.sum())
    for sx, r in (("female", f), ("male", m)):
        rec.update({f"status_{sx}": r["status"], f"odds_ratio_{sx}": r["odds_ratio"],
                    f"ci_low_{sx}": r["ci_low"], f"ci_high_{sx}": r["ci_high"]})
    why = au.model_floor_reason(pop[c], pop["aid_status"], 5, female=pop["female"])
    if f["status"] != "fitted" or m["status"] != "fitted":
        why = "the female or the male model was not fitted"
    extra = {"feature": c[4:], "row_label": lbl, "cochran_q": NAN, "i2": NAN}
    if why:
        rec.update(status="not_tested", reason=why, coef=NAN, se=NAN, crit=NAN, ratio=NAN,
                   ci_low=NAN, ci_high=NAN, p=NAN)
    else:
        delta, se = f["coef"] - m["coef"], float(np.sqrt(f["se"] ** 2 + m["se"] ** 2))
        w = np.array([1 / f["se"] ** 2, 1 / m["se"] ** 2])
        yv = np.array([f["coef"], m["coef"]])
        ybar = (w * yv).sum() / w.sum()
        q_stat = float((w * (yv - ybar) ** 2).sum())
        rec.update(status="tested", reason="", coef=delta, se=se, crit=crit,
                   ratio=float(np.exp(delta)), ci_low=float(np.exp(delta - crit * se)),
                   ci_high=float(np.exp(delta + crit * se)), p=float(1 - chi2.cdf(q_stat, 1)))
        extra.update(cochran_q=q_stat, i2=max(0.0, (q_stat - 1) / q_stat) * 100
                     if q_stat > 0 else 0.0)
    for mg in TOST_MARGINS:
        extra[f"tost_p_male_{mg:g}"] = tost_p(m["coef"], m["se"], mg) \
            if m["status"] == "fitted" else NAN
    extra["male_min_detectable_or"] = min_detectable_or(
        m["n_aid_pos"], m["n_aid_neg"], 100 * (m["events_aid_neg"] / m["n_aid_neg"]) / 100
        if m["n_aid_neg"] else NAN)
    in_floor = f["fdr_role"] == "tested" and m["fdr_role"] == "tested"
    rec["fdr_role"] = "tested" if in_floor else "below_group_floor"
    rec["group_floor_reason"] = "" if in_floor else "; ".join(
        f"{sx}: {r['group_floor_reason']}" for sx, r in (("female", f), ("male", m))
        if r["group_floor_reason"])
    sex_rows.append(rec)
    sex_extra.append(extra)
sex_het = pd.DataFrame(sex_rows)
sex_het = bh_by(sex_het.assign(_all=1), ["_all"],
                (sex_het["status"] == "tested") & (sex_het["fdr_role"] == "tested")).drop(
    columns="_all")
sex_extra = pd.DataFrame(sex_extra)
for _, r in sex_het.iterrows():
    print(f"  {r['row_label']:<42} female OR {fmt_console(r['odds_ratio_female'], '.2f')}  "
          f"male OR {fmt_console(r['odds_ratio_male'], '.2f')}  "
          + (f"ratio {r['ratio']:.2f} [{r['ci_low']:.2f}-{r['ci_high']:.2f}] "
             f"P={r['p']:.4f} Q={fmt_console(r['q'], '.4f')}" if r["status"] == "tested"
             else f"not tested: {r['reason']}")
          + (f"  [no Q: {r['group_floor_reason']}]" if r["status"] == "tested"
             and r["fdr_role"] != "tested" else ""))

# =============================================================================
# SECTION 4 -- CO-OCCURRENCE OF THE CORE FEATURES; TIMING; RBD
# =============================================================================
au.section("SECTION 4 -- CO-OCCURRENCE OF THE CORE FEATURES, THEIR TIMING, AND RBD")
co_rows, co_or_rows, timing_rows = [], [], []
for s, sex in SEXES:
    for arm, cols in arms_for(s):
        d = apply_arm(pop[pop["female"] == s], cols).copy()
        for st, code, cs in COMBOS:
            d["_combo"] = d[cs].eq(1).all(axis=1).astype(int)
            r = weighted_or(d, "_combo")
            co_or_rows.append({"sex": sex, "arm": arm, "set": st, "combination": code,
                               "row_label": COMBO_LABEL[code], **r})
            for a, aid in ((1, "AID+"), (0, "AID-")):
                g = d[d["aid_status"] == a]
                k = int(g["_combo"].sum())
                pct, lo, hi = au.wilson_ci(k, len(g))
                co_rows.append({"sex": sex, "arm": arm, "set": st, "combination": code,
                                "row_label": COMBO_LABEL[code], "aid": aid, "events": k,
                                "n": len(g), "pct": pct, "ci_low": lo, "ci_high": hi})
            co = d[d["_combo"] == 1].copy()
            npre = sum(co[PRE_PERI[c]] for c in cs) if len(co) else pd.Series(dtype=int)
            co["_all_pre"] = (npre == len(cs)).astype(int)
            rec = {"sex": sex, "arm": arm, "set": st, "combination": code,
                   "row_label": COMBO_LABEL[code], "n_features": len(cs)}
            for a, aid in ((1, "aid_pos"), (0, "aid_neg")):
                msk = co["aid_status"] == a
                for j in range(len(cs) + 1):
                    rec[f"n_{j}_window_{aid}"] = int((npre[msk] == j).sum()) if msk.any() else 0
            t = exact_row(co, "_all_pre", "every feature in the window")
            rec.update({f"combo_{k}": v for k, v in t.items() if k.startswith(("n_", "events_"))})
            rec.update({k: v for k, v in t.items() if not k.startswith(("n_", "events_"))})
            timing_rows.append(rec)
cooccur = pd.DataFrame(co_rows)
cooccur_or = pd.DataFrame(co_or_rows)
cooccur_or = bh_by(cooccur_or, ["sex", "arm"],
                   (cooccur_or["set"] == "primary") & (cooccur_or["status"] == "fitted"))
check_ci(cooccur_or, "co-occurrence")
co_timing = pd.DataFrame(timing_rows)
co_timing = bh_by(co_timing, ["sex", "arm"],
                  (co_timing["set"] == "primary") & (co_timing["status"] == "tested"))
for _, r in cooccur_or[cooccur_or["set"] == "primary"].iterrows():
    print(f"  {r['sex']:<7} {r['arm']:<16} {r['row_label']:<72} "
          + (f"OR {r['odds_ratio']:.2f} [{r['ci_low']:.2f}-{r['ci_high']:.2f}] "
             f"Q={fmt_console(r['q'], '.4f')}" if r["status"] == "fitted"
             else f"{r['status']}: {r['reason']}"))

pop["_core_count"] = pop[CORE_TRIAD].sum(axis=1)
pop["_four"] = (pop[CORE_TRIAD].eq(1).all(axis=1) & (pop["has_rbd_any"] == 1)).astype(int)
by_count, four, split = [], [], []
for s, sex in SEXES:
    for arm, cols in arms_for(s):
        d = apply_arm(pop[pop["female"] == s], cols)
        for b in range(4):
            by_count.append({"sex": sex, "arm": arm, "core_feature_count": b,
                             **exact_row(d[d["_core_count"] == b], "has_rbd_any", "RBD")})
        four.append({"sex": sex, "arm": arm, **weighted_or(d, "_four")})
        tri = d[d["_core_count"] == 3]
        split.append({"sex": sex, "arm": arm, **exact_row(tri, "has_rbd_any", "RBD")})
by_count = pd.DataFrame(by_count)
by_count = bh_by(by_count, ["sex", "arm"], by_count["status"] == "tested")
four = pd.DataFrame(four)
check_ci(four, "RBD and the core triad")
split = pd.DataFrame(split)
print("\n  RBD by the number of core features (AID+ / AID-):")
for _, r in by_count.iterrows():
    print(f"  {r['sex']:<7} {r['arm']:<16} {r['core_feature_count']}  "
          f"{r['events_aid_pos']:>4}/{r['n_aid_pos']:<6} {r['events_aid_neg']:>4}/"
          f"{r['n_aid_neg']:<6} " + (f"P={r['p']:.4f}" if r["status"] == "tested"
                                     else f"not tested: {r['reason']}"))

# =============================================================================
# SECTION 4b -- CORE-FEATURE CO-OCCURRENCE BEYOND CHANCE, AID+ FEMALE PATIENTS (LOG ONLY)
# =============================================================================
au.section("SECTION 4b -- CO-OCCURRENCE OF THE CORE FEATURES BEYOND CHANCE, AID+ FEMALE "
           "PATIENTS (LOG ONLY)")
print(f"  Observed = patients with every feature of the set. Expected = mean count over "
      f"{COOC_CHANCE_PERMUTATIONS:,} permutations,")
print(f"  each feature permuted independently within strata of total note count x documentation "
      f"span ({COOC_CHANCE_QUANTILES} x {COOC_CHANCE_QUANTILES}")
print(f"  quantile bins cut within the group; a stratum under {au.MIN_GROUP_N} patients is merged "
      f"into its neighbour on note count).")
print("  Interval = 2.5th-97.5th percentile of the permuted counts. P (two-sided) = (k + 1) / "
      "(B + 1), k = permutations at least")
print("  as far from the expected count as the observed. Plain = expected under unstratified "
      "independence (n x product of the")
print(f"  prevalences). No FDR. A set with fewer than {au.MIN_GROUP_N} observed patients is skipped.")


def chance_bin_text(g):
    return "+".join(str(b + 1) for b in g)


def chance_strata(d):
    """Stratum label per patient of d: COOC_CHANCE_QUANTILES bins of total note count and of
    documentation span, cut within d (ties broken by row order, patients sorted by ID).
    Within each span bin, a stratum under au.MIN_GROUP_N patients is merged into its
    neighbour on note count (the next higher bin; the next lower for the top bin), repeated
    until none is left or the span bin is one stratum. Merges are printed."""
    q = COOC_CHANCE_QUANTILES
    n_no_span = int(d["documentation_span_yrs"].isna().sum())
    if n_no_span:
        print(f"    {n_no_span:,} patients without a documentation span (no dated note) "
              f"counted as span 0")
    nb = pd.qcut(d["note_count"].rank(method="first"), q, labels=False).to_numpy()
    sb = pd.qcut(d["documentation_span_yrs"].fillna(0.0).rank(method="first"), q,
                 labels=False).to_numpy()
    label = np.empty(len(d), dtype=object)
    for s in range(q):
        in_span = sb == s
        bins = [[b] for b in range(q)]

        def size(g):
            return int((in_span & np.isin(nb, g)).sum())

        while len(bins) > 1:
            small = [i for i, g in enumerate(bins) if size(g) < au.MIN_GROUP_N]
            if not small:
                break
            i = small[0]
            j = i + 1 if i + 1 < len(bins) else i - 1
            print(f"    merged: span bin {s + 1}, note-count bin {chance_bin_text(bins[i])} "
                  f"({size(bins[i])} patients) into note-count bin {chance_bin_text(bins[j])} "
                  f"({size(bins[j])} patients)")
            lo, hi = min(i, j), max(i, j)
            bins[lo:hi + 1] = [sorted(bins[lo] + bins[hi])]
        if len(bins) == 1 and size(bins[0]) < au.MIN_GROUP_N:
            print(f"    span bin {s + 1}: {size(bins[0])} patients in all, kept as one stratum")
        for g in bins:
            label[in_span & np.isin(nb, g)] = f"span {s + 1}, notes {chance_bin_text(g)}"
    sizes = pd.Series(label).value_counts()
    print(f"    {len(sizes)} strata; patients per stratum {int(sizes.min())}-{int(sizes.max())}")
    return label


def chance_cooccurrence(d, group_label, seed):
    """Observed and permutation-expected co-occurrence of the primary sets in d (log only)."""
    au.sub(f"{group_label} (n = {len(d):,})")
    if len(d) < COOC_CHANCE_QUANTILES * COOC_CHANCE_QUANTILES:
        print(f"  not tested: {len(d):,} patients, fewer than one per stratum")
        return
    label = chance_strata(d)
    rng = np.random.default_rng(seed)
    B = COOC_CHANCE_PERMUTATIONS
    vals = {f: d[f].to_numpy(dtype=np.int8) for f in CORE_TRIAD}
    perm = {f: np.empty((B, len(d)), dtype=np.int8) for f in CORE_TRIAD}
    strata = [np.flatnonzero(label == s) for s in sorted(set(label))]
    for f in CORE_TRIAD:
        for idx in strata:
            order = np.argsort(rng.random((B, len(idx))), axis=1)
            perm[f][:, idx] = vals[f][idx][order]
    for f in CORE_TRIAD:
        print(f"    {LABEL[f]:<24} {int(vals[f].sum()):>5,} patients ({100 * vals[f].mean():.1f}%)")
    for st, code, cs in COMBOS:
        if st != "primary":
            continue
        obs = int(np.all([vals[c] == 1 for c in cs], axis=0).sum())
        plain = len(d) * float(np.prod([vals[c].mean() for c in cs]))
        if obs < au.MIN_GROUP_N:
            print(f"  {COMBO_LABEL[code]}: skipped, {obs} observed patients (fewer than "
                  f"{au.MIN_GROUP_N})")
            continue
        joint = perm[cs[0]] == 1
        for c in cs[1:]:
            joint &= perm[c] == 1
        cnt = joint.sum(axis=1)
        exp = float(cnt.mean())
        lo, hi = np.percentile(cnt, [2.5, 97.5])
        k = int((np.abs(cnt - exp) >= abs(obs - exp) - 1e-9).sum())
        p = (k + 1) / (B + 1)
        print(f"  {COMBO_LABEL[code]}")
        print(f"    observed {obs:,}; expected {exp:.1f} [{lo:.0f}-{hi:.0f}]; O/E "
              f"{obs / exp if exp > 0 else NAN:.2f}; P = {p:.4g}   plain independence "
              f"{plain:.1f} (O/E {obs / plain if plain > 0 else NAN:.2f})")


aidpos_female_all = pop[(pop["female"] == 1) & (pop["aid_status"] == 1)]
for gi, (grp_label, d) in enumerate((
        ("AID+ female, all patients", aidpos_female_all),
        ("AID+ female, coded DLB excluded",
         aidpos_female_all[aidpos_female_all["coded_dlb"] != 1]))):
    chance_cooccurrence(d, grp_label, COOC_CHANCE_SEED + gi)

# =============================================================================
# SECTION 4c -- TIMING COMPOSITION (CO-OCCURRENCE PANEL b), AID+ FEMALE PATIENTS WITHOUT VS WITH CODED DLB
#               (LOG ONLY)
# =============================================================================
au.section("SECTION 4c -- TIMING COMPOSITION (CO-OCCURRENCE PANEL b): AID+ FEMALE PATIENTS WITHOUT VS WITH "
           "CODED DLB (LOG ONLY)")
print(f"  Among AID+ female patients with each combination: how many of its features were first "
      f"documented by {PRE_PERI_WINDOW_YRS:g} year")
print("  after the AD diagnosis date (all / some / none), without vs with coded DLB. Tests: the "
      "2 x 3 composition by the")
print("  Fisher-Freeman-Halton exact test (full enumeration); the number of features in the window "
      "by Mann-Whitney U.")
print(f"  BH across the combinations, per test. A combination with fewer than {au.MIN_GROUP_N} "
      f"patients in either group is skipped.")
print("  The counts are the figure workbook's b sheet (built here, written in section 9).")


def ffh_exact_2x3(table):
    """Fisher-Freeman-Halton exact P of a 2 x 3 table: with both margins fixed, the sum of
    the probabilities of every table no more probable than the observed one (full
    enumeration of the first row)."""
    from scipy.special import gammaln

    def lchoose(a, b):
        return gammaln(a + 1) - gammaln(b + 1) - gammaln(a - b + 1)

    t = np.asarray(table, dtype=int)
    r1 = int(t[0].sum())
    c1, c2, c3 = (int(x) for x in t.sum(axis=0))
    n = c1 + c2 + c3
    a1 = np.arange(0, min(c1, r1) + 1)[:, None]
    a2 = np.arange(0, min(c2, r1) + 1)[None, :]
    a3 = r1 - a1 - a2
    ok = (a3 >= 0) & (a3 <= c3)
    lp = (lchoose(c1, a1) + lchoose(c2, a2) + lchoose(c3, np.clip(a3, 0, c3))
          - lchoose(n, r1))[ok]
    lp_obs = (lchoose(c1, t[0, 0]) + lchoose(c2, t[0, 1]) + lchoose(c3, t[0, 2])
              - lchoose(n, r1))
    return float(min(1.0, np.exp(lp[lp <= lp_obs + 1e-7]).sum()))


# Panel b of the co-occurrence figure, built once here and written by section 9 (sheet b):
# per group and combination, the AID+ female patients with every feature of the combination,
# by how many of its features were first documented in the window. The second dict keeps
# each patient's number for the Mann-Whitney test.
TIMING_LABEL = {"dlb_excluded": "without coded DLB", "coded_dlb": "with coded DLB"}
fig3b, fig3b_npre = [], {}
for grp, d in (("dlb_excluded", aidpos_female_all[aidpos_female_all["coded_dlb"] == 0]),
               ("coded_dlb", aidpos_female_all[aidpos_female_all["coded_dlb"] == 1])):
    for st, code, cs in COMBOS:
        if st != "primary":
            continue
        co = d[d[cs].eq(1).all(axis=1)]
        npre = sum(co[PRE_PERI[c]] for c in cs) if len(co) else pd.Series(dtype=int)
        fig3b.append({"group": grp, "combination": code, "row_label": COMBO_LABEL[code],
                      "n_features": len(cs), "n": len(co),
                      "all_in_window": int((npre == len(cs)).sum()),
                      "some_in_window": int(((npre > 0) & (npre < len(cs))).sum()),
                      "none_in_window": int((npre == 0).sum())})
        fig3b_npre[(grp, code)] = npre.to_numpy(dtype=int)
FIG3B_ROW = {(r["group"], r["combination"]): r for r in fig3b}

timing_test = []
for st, code, cs in COMBOS:
    if st != "primary":
        continue
    rec = {"code": code, "n_features": len(cs), "status": "tested", "reason": "",
           "p_ffh": NAN, "u": NAN, "p_mw": NAN}
    small = [lbl for grp, lbl in TIMING_LABEL.items()
             if FIG3B_ROW[(grp, code)]["n"] < au.MIN_GROUP_N]
    if small:
        rec.update(status="not_tested",
                   reason=f"fewer than {au.MIN_GROUP_N} patients {' and '.join(small)}")
    else:
        rec["p_ffh"] = ffh_exact_2x3([[FIG3B_ROW[(grp, code)][k] for k in
                                       ("all_in_window", "some_in_window", "none_in_window")]
                                      for grp in TIMING_LABEL])
        a_, b_ = fig3b_npre[("dlb_excluded", code)], fig3b_npre[("coded_dlb", code)]
        if np.unique(np.concatenate([a_, b_])).size == 1:
            rec.update(u=float(len(a_) * len(b_)) / 2, p_mw=1.0)
        else:
            u, p = mannwhitneyu(a_, b_, alternative="two-sided")
            rec.update(u=float(u), p_mw=float(p))
    timing_test.append(rec)
timing_test = pd.DataFrame(timing_test)
for p_col, q_col in (("p_ffh", "q_ffh"), ("p_mw", "q_mw")):
    timing_test[q_col] = NAN
    t = timing_test["status"] == "tested"
    if t.any():
        timing_test.loc[t, q_col] = au.bh_fdr(timing_test.loc[t, p_col].values)
for _, r in timing_test.iterrows():
    print(f"\n  {COMBO_LABEL[r['code']]}")
    for grp, lbl in TIMING_LABEL.items():
        row = FIG3B_ROW[(grp, r["code"])]
        n_ = row["n"]
        pc = (lambda x: f"{100 * x / n_:.1f}%" if n_ else "n/a")
        print(f"    {lbl:<17} n = {n_:>5,}: all {row['all_in_window']:>4,} "
              f"({pc(row['all_in_window'])}), some {row['some_in_window']:>4,} "
              f"({pc(row['some_in_window'])}), none {row['none_in_window']:>4,} "
              f"({pc(row['none_in_window'])})")
    if r["status"] != "tested":
        print(f"    skipped: {r['reason']}")
    else:
        print(f"    Fisher-Freeman-Halton P = {r['p_ffh']:.4g} (Q = {r['q_ffh']:.4g}); "
              f"Mann-Whitney U = {r['u']:.1f}, P = {r['p_mw']:.4g} (Q = {r['q_mw']:.4g})")

# =============================================================================
# SECTION 5 -- PROFILE SIMILARITY TO CODED DLB (CHECK)
# =============================================================================
au.section("SECTION 5 -- FEATURE-PREVALENCE PROFILE SIMILARITY TO CODED DLB (CHECK)")
prof_prev, prof_sim = [], []
for s, sex in SEXES:
    d = pop[pop["female"] == s]
    groups = [("AID+", d[d["aid_status"] == 1]), ("AID-", d[d["aid_status"] == 0]),
              ("AD only", d[(d["coded_dlb"] == 0) & (d["coded_pd"] == 0)]),
              ("coded DLB", d[d["coded_dlb"] == 1])]
    vec = {g: np.array([x[c].mean() if len(x) else NAN for c in POOL_COLS]) for g, x in groups}
    for i, c in enumerate(POOL_COLS):
        for g, x in groups:
            prof_prev.append({"sex": sex, "group": g, "n": len(x), "feature": c[4:],
                              "row_label": LABEL[c], "events": int(x[c].sum()),
                              "prevalence_pct": 100 * vec[g][i]})
    ref = vec["coded DLB"]
    for g in ("AID+", "AID-", "AD only"):
        a, ok = vec[g], ~(np.isnan(vec[g]) | np.isnan(ref))
        na, nb = np.linalg.norm(a[ok]), np.linalg.norm(ref[ok])
        cos = float(a[ok] @ ref[ok] / (na * nb)) if na and nb else NAN
        dist = float(np.linalg.norm(a[ok] - ref[ok])) if ok.any() else NAN
        prof_sim.append({"sex": sex, "group": g, "n": len(dict(groups)[g]),
                         "cosine_similarity_to_coded_dlb": cos,
                         "euclidean_distance_to_coded_dlb": dist})
        print(f"  {sex:<7} {g:<8} cosine {fmt_console(cos, '.4f')}  Euclidean "
              f"{fmt_console(dist, '.4f')}")
prof_prev, prof_sim = pd.DataFrame(prof_prev), pd.DataFrame(prof_sim)

# =============================================================================
# SECTION 6 -- DISCRIMINANT: CODED DLB VS AD ONLY
# =============================================================================
au.section("SECTION 6 -- DISCRIMINANT: CODED DLB VS AD ONLY (UNWEIGHTED LOGISTIC)")
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

dlb = pop[pop["coded_dlb"] == 1]
ad_only = pop[(pop["coded_dlb"] == 0) & (pop["coded_pd"] == 0) & (pop["coded_vd"] == 0)]
train = pd.concat([dlb.assign(_y=1), ad_only.assign(_y=0)], ignore_index=True)
y = train["_y"].astype(float)
print(f"  Training set: coded DLB n = {len(dlb):,}; AD only (no coded DLB, PD or vascular "
      f"dementia) n = {len(ad_only):,}")


DISC_FLOOR = {c: group_floor(train, c, "_y", ("coded DLB", "AD only")) for c in POOL_COLS}
MODEL_COLS = [c for c in POOL_COLS if not DISC_FLOOR[c]]
core_out = [LABEL[c] for c in CORE_FEATURES if DISC_FLOOR[c]]
if core_out:
    raise RuntimeError(f"Core feature(s) below the per-group floor in the discriminant: "
                       f"{core_out}. Leaving a core feature out changes every coefficient; "
                       f"decide the model before running.")
for c in POOL_COLS:
    if DISC_FLOOR[c]:
        print(f"  Not in the multivariable model (per-group floor): {LABEL[c]}: {DISC_FLOOR[c]}")


def logit_rows(cols, model_type):
    """One unweighted logistic regression of coded DLB on `cols`; rows per term."""
    rows = []
    base_r = {"model": model_type, "n_model": len(train), "n_dlb": len(dlb),
              "n_ad_only": len(ad_only)}
    why = au.model_floor_reason(y, None, len(cols))
    mod = None
    if not why:
        try:
            mod = sm.Logit(y, sm.add_constant(train[cols].astype(float))).fit(disp=False)
            if not mod.mle_retvals.get("converged", True):
                why, mod = "did not converge", None
        except Exception as e:
            why = f"fit error: {type(e).__name__}"
    for c in cols:
        r = {**base_r, "feature": c[4:], "row_label": LABEL[c],
             "mckeith_class": MCKEITH_CLASS[c],
             "fdr_role": "below_group_floor" if DISC_FLOOR[c] else "tested",
             "group_floor_reason": DISC_FLOOR[c],
             "events_dlb": int(dlb[c].sum()), "events_ad_only": int(ad_only[c].sum())}
        if mod is None or not np.isfinite(mod.params.get(c, NAN)) \
                or not np.isfinite(mod.bse.get(c, NAN)):
            r.update(status="not_fitted" if why and "fit error" not in why
                     and why != "did not converge" else "failed",
                     reason=why or "term not estimated", coef=NAN, se=NAN, crit=NAN,
                     odds_ratio=NAN, ci_low=NAN, ci_high=NAN, p=NAN)
        else:
            ci = mod.conf_int()
            r.update(status="fitted", reason="", coef=float(mod.params[c]),
                     se=float(mod.bse[c]), crit=float(norm.ppf(0.975)),
                     odds_ratio=float(np.exp(mod.params[c])),
                     ci_low=float(np.exp(ci.loc[c, 0])), ci_high=float(np.exp(ci.loc[c, 1])),
                     p=float(mod.pvalues[c]))
        rows.append(r)
    return rows, mod


multi_rows, joint = logit_rows(MODEL_COLS, "multivariable")
for c in POOL_COLS:             # every feature gets a row; those left out are not in the model
    if c not in MODEL_COLS:
        multi_rows.append({"model": "multivariable", "n_model": len(train), "n_dlb": len(dlb),
                           "n_ad_only": len(ad_only), "feature": c[4:], "row_label": LABEL[c],
                           "mckeith_class": MCKEITH_CLASS[c], "fdr_role": "below_group_floor",
                           "group_floor_reason": DISC_FLOOR[c],
                           "events_dlb": int(dlb[c].sum()),
                           "events_ad_only": int(ad_only[c].sum()), "status": "not_in_model",
                           "reason": "below the per-group floor: " + DISC_FLOOR[c], "coef": NAN,
                           "se": NAN, "crit": NAN, "odds_ratio": NAN, "ci_low": NAN,
                           "ci_high": NAN, "p": NAN})
multi = pd.DataFrame(multi_rows).set_index("feature").loc[[c[4:] for c in POOL_COLS]] \
    .reset_index()
multi = bh_by(multi.assign(_f=1), ["_f"], multi["status"] == "fitted").drop(columns="_f")
check_ci(multi, "discriminant, multivariable")
biv = []
for c in POOL_COLS:
    biv += logit_rows([c], "bivariable")[0]
biv = pd.DataFrame(biv)
biv = bh_by(biv.assign(_f=1), ["_f"],
            (biv["status"] == "fitted") & (biv["fdr_role"] == "tested")).drop(columns="_f")
check_ci(biv, "discriminant, bivariable")
for (_, a), (_, b) in zip(multi.iterrows(), biv.iterrows()):
    print(f"  {a['row_label']:<42} multivariable b {fmt_console(a['coef'], '+.3f')} "
          f"Q={fmt_console(a['q'], '.4f')}   bivariable b {fmt_console(b['coef'], '+.3f')} "
          f"Q={fmt_console(b['q'], '.4f')}")

fit_rows, score_rows = [], []
if joint is not None:
    X = sm.add_constant(train[MODEL_COLS].astype(float))
    prob = joint.predict(X)
    auc = roc_auc_score(y, prob)
    fpr, tpr, thr = roc_curve(y, prob)
    j = int(np.argmax(tpr - fpr))
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=CV_SEED)
    aucs = []
    for tr, te in cv.split(train[MODEL_COLS], y):
        clf = unpenalized_logistic(2000)
        clf.fit(train[MODEL_COLS].iloc[tr].astype(float), y.iloc[tr])
        aucs.append(roc_auc_score(y.iloc[te], clf.predict_proba(
            train[MODEL_COLS].iloc[te].astype(float))[:, 1]))
    fit_rows = [("n_dlb", len(dlb)), ("n_ad_only", len(ad_only)), ("auc_in_sample", auc),
                ("youden_threshold", float(thr[j])), ("sensitivity", float(tpr[j])),
                ("specificity", float(1 - fpr[j])), ("cv_auc_mean", float(np.mean(aucs))),
                ("cv_auc_min", float(min(aucs))), ("cv_auc_max", float(max(aucs))),
                ("cv_folds", CV_FOLDS), ("cv_seed", CV_SEED),
                ("n_features_in_model", len(MODEL_COLS))]
    print(f"\n  In-sample AUC {auc:.3f}; Youden threshold {thr[j]:.3f} (sensitivity "
          f"{tpr[j]:.3f}, specificity {1 - fpr[j]:.3f}); {CV_FOLDS}-fold CV AUC "
          f"{np.mean(aucs):.3f} ({min(aucs):.3f}-{max(aucs):.3f})")
    allx = sm.add_constant(pop[MODEL_COLS].astype(float), has_constant="add")[X.columns]
    score = joint.predict(allx)
    sgroups = {g: group_mask(a, s) for a, s, g in GROUPS}
    sgroups["coded_dlb"] = pop["coded_dlb"] == 1
    sgroups["ad_only"] = (pop["coded_dlb"] == 0) & (pop["coded_pd"] == 0) & (pop["coded_vd"] == 0)
    for g, msk in sgroups.items():
        sc = score[msk]
        score_rows.append({"group": g, "n": len(sc), "median": sc.median(),
                           "q1": sc.quantile(.25), "q3": sc.quantile(.75), "mean": sc.mean(),
                           "comparison": "", "status": "", "reason": "", "statistic": NAN,
                           "p": NAN})
    for s, sex in SEXES:
        a_, b_ = score[group_mask(1, s)], score[group_mask(0, s)]
        why = au.group_minimum_reason(len(a_), len(b_))
        u, p = (NAN, NAN) if why else mannwhitneyu(a_, b_, alternative="two-sided")
        score_rows.append({"group": "", "comparison": f"AID+ vs AID-, {sex}",
                           "status": "not_tested" if why else "tested", "reason": why,
                           "statistic": float(u), "p": float(p)})
else:
    fit_rows = [("n_dlb", len(dlb)), ("n_ad_only", len(ad_only))]
    print("  Multivariable model not fitted: " + str(multi["reason"].iloc[0]))
disc_fit = pd.DataFrame(fit_rows, columns=["metric", "value"])
disc_scores = pd.DataFrame(score_rows)

# =============================================================================
# SECTION 7 -- DLB FEATURE-BURDEN SCORE (FIGURE: SHARE OF FEMALE PATIENTS AT OR ABOVE EACH SCORE)
# =============================================================================
au.section("SECTION 7 -- DLB FEATURE-BURDEN SCORE")
if SCORE_SUPPORTIVE_CAP >= SCORE_CORE_POINTS:
    raise RuntimeError("score settings: the supportive cap reaches one core feature's points")
pop["core_n"] = pop[SCORE_CORE].sum(axis=1)
pop["supportive_n"] = pop[SCORE_SUPPORTIVE].sum(axis=1)
pop["dlb_score"] = SCORE_CORE_POINTS * pop["core_n"] + np.minimum(pop["supportive_n"],
                                                                  SCORE_SUPPORTIVE_CAP)
pop["mckeith_tier"] = np.where(pop["core_n"] >= 2, "two_or_more_core",
                               np.where(pop["core_n"] == 1, "one_core", "no_core"))
if (np.minimum(pop["supportive_n"], SCORE_SUPPORTIVE_CAP) >= SCORE_CORE_POINTS).any():
    raise RuntimeError("supportive points reach one core point")
smax_observed = int(pop["dlb_score"].max())
if smax_observed > SCORE_MAX or int(pop["dlb_score"].min()) < 0:
    raise RuntimeError(f"a feature-burden score lies outside 0-{SCORE_MAX}")
grid = np.arange(0, SCORE_MAX + 1)
print(f"  Score = {SCORE_CORE_POINTS} x core features ({len(SCORE_CORE)}) + supportive features "
      f"({len(SCORE_SUPPORTIVE)}, counted up to {SCORE_SUPPORTIVE_CAP}); maximum {SCORE_MAX} "
      f"(curve grid 0-{SCORE_MAX}); highest observed {smax_observed}")


def survival(vals):
    v = np.asarray(vals, dtype=float)
    return np.array([(v >= g).mean() for g in grid])


def boot_survival(vals, seed):
    rng = np.random.default_rng(seed)
    v = np.asarray(vals, dtype=float)
    out = np.empty((N_BOOT, len(grid)))
    for b in range(N_BOOT):
        s_ = v[rng.integers(0, len(v), len(v))]
        out[b] = np.array([(s_ >= g).mean() for g in grid])
    return np.percentile(out, 2.5, axis=0), np.percentile(out, 97.5, axis=0)


def boot_diff(a, b, seed):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    out = np.empty((N_BOOT, len(grid)))
    for i in range(N_BOOT):
        sa = a[rng.integers(0, len(a), len(a))]
        sb = b[rng.integers(0, len(b), len(b))]
        out[i] = (np.array([(sa >= g).mean() for g in grid])
                  - np.array([(sb >= g).mean() for g in grid]))
    return np.percentile(out, 2.5, axis=0), np.percentile(out, 97.5, axis=0)


curves, summary = [], []
female_patients = pop[pop["female"] == 1]
for arm, cols in ARMS[:2]:
    d = apply_arm(female_patients, cols)
    pos = d.loc[d["aid_status"] == 1, "dlb_score"].values
    neg = d.loc[d["aid_status"] == 0, "dlb_score"].values
    why = au.group_minimum_reason(len(pos), len(neg))
    if why:
        summary.append({"arm": arm, "n_aid_pos": len(pos), "n_aid_neg": len(neg),
                        "status": "not_tested", "reason": why, "statistic": NAN, "p": NAN})
        print(f"  {arm}: not drawn: {why}")
        continue
    s_pos, s_neg = survival(pos), survival(neg)
    lo_p, hi_p = boot_survival(pos, BOOT_SEED)
    lo_n, hi_n = boot_survival(neg, BOOT_SEED)
    dlo, dhi = boot_diff(pos, neg, BOOT_SEED + 1)
    u, p = mannwhitneyu(pos, neg, alternative="two-sided")
    summary.append({"arm": arm, "n_aid_pos": len(pos), "n_aid_neg": len(neg),
                    "status": "tested", "reason": "", "statistic": float(u), "p": float(p)})
    curves.append(pd.DataFrame({"arm": arm, "score": grid, "share_aid_pos": s_pos,
                                "share_aid_neg": s_neg, "ci_low_aid_pos": lo_p,
                                "ci_high_aid_pos": hi_p, "ci_low_aid_neg": lo_n,
                                "ci_high_aid_neg": hi_n, "difference": s_pos - s_neg,
                                "difference_ci_low": dlo, "difference_ci_high": dhi}))
    print(f"  Female, {arm}: AID+ n = {len(pos):,}, AID- n = {len(neg):,}; Mann-Whitney "
          f"P = {p:.4g}")
curves = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame(
    columns=["arm", "score", "share_aid_pos", "share_aid_neg", "ci_low_aid_pos",
             "ci_high_aid_pos", "ci_low_aid_neg", "ci_high_aid_neg", "difference",
             "difference_ci_low", "difference_ci_high"])
for r in summary:
    print(f"  c_summary  {r['arm']:<13} AID+ n = {r['n_aid_pos']:,}, AID- n = {r['n_aid_neg']:,}; "
          f"{r['status']}{': ' + r['reason'] if r['reason'] else ''}; Mann-Whitney U = "
          f"{fmt_console(r['statistic'], '.1f')}, P = {fmt_console(r['p'], '.4g')}")
share_at = sorted({x for x in SCORE_SHARES_PRINTED + (SCORE_CORE_POINTS, 2 * SCORE_CORE_POINTS)
                   if 0 <= x <= SCORE_MAX})
for arm in curves["arm"].unique():
    c_ = curves[curves["arm"] == arm].set_index("score")
    for x in share_at:
        r = c_.loc[x]
        print(f"  Share >= {x:<3} {arm:<13} AID+ {100 * r['share_aid_pos']:5.1f}% "
              f"[{100 * r['ci_low_aid_pos']:.1f}-{100 * r['ci_high_aid_pos']:.1f}]  AID- "
              f"{100 * r['share_aid_neg']:5.1f}% [{100 * r['ci_low_aid_neg']:.1f}-"
              f"{100 * r['ci_high_aid_neg']:.1f}]  difference {100 * r['difference']:+.1f} "
              f"[{100 * r['difference_ci_low']:+.1f} to {100 * r['difference_ci_high']:+.1f}] "
              f"percentage points")
ref_scores = pop.loc[pop["coded_dlb"] == 1, "dlb_score"].values
ref_shown = int(len(ref_scores) >= SCORE_REFERENCE_MIN_N)
reference = pd.DataFrame({"score": grid, "share_coded_dlb": survival(ref_scores)
                          if ref_shown else np.full(len(grid), NAN)})
print(f"  Coded DLB reference (both sexes): n = {len(ref_scores):,} "
      f"({'drawn' if ref_shown else 'not drawn, fewer than ' + str(SCORE_REFERENCE_MIN_N)})")

dist_rows, tier_rows = [], []
for arm, cols in ARMS:
    d = apply_arm(female_patients, cols)
    for a, aid in ((1, "AID+"), (0, "AID-")):
        v = d.loc[d["aid_status"] == a, "dlb_score"]
        dist_rows.append({"sex": "Female", "arm": arm, "aid": aid, "n": len(v),
                          "median": v.median(), "q1": v.quantile(.25), "q3": v.quantile(.75),
                          "p90": v.quantile(.90), "max": v.max()})
        for t in ("no_core", "one_core", "two_or_more_core"):
            tier_rows.append({"sex": "Female", "arm": arm, "aid": aid, "tier": t,
                              "n": int((d.loc[d["aid_status"] == a, "mckeith_tier"] == t).sum()),
                              "n_group": len(v)})
v = pd.Series(ref_scores, dtype=float)
dist_rows.append({"sex": "both", "arm": "coded_dlb", "aid": "both", "n": len(v),
                  "median": v.median(), "q1": v.quantile(.25), "q3": v.quantile(.75),
                  "p90": v.quantile(.90), "max": v.max()})
score_dist, score_tiers = pd.DataFrame(dist_rows), pd.DataFrame(tier_rows)

# =============================================================================
# SECTION 8 -- McKEITH FEATURES NOT USED; DESCRIPTIVE TIMING
# =============================================================================
au.section("SECTION 8 -- McKEITH FEATURES NOT USED IN THE TESTED POOL")
rec = au.load_symptom_records(SYMPTOM_RECORDS_PATH, pop, notes)
dated = rec[rec["note_date"].notna()].merge(pop[["person_id", "birth_date"]], on="person_id",
                                            how="left", validate="many_to_one")
in_ceiling = dated[au.within_age_ceiling(au.years_between(dated["note_date"], dated["birth_date"]),
                                         dated["person_id"].map(ceil["ceiling_by_person"]))]
not_used = []
female_pop = pop["female"] == 1
for code, lbl, src, key, reason in NOT_USED:
    flag = pop["has_" + key] if src == "mentions" else \
        pop["person_id"].isin(in_ceiling.loc[in_ceiling["category"] == key, "person_id"]).astype(int)
    r = {"code": code, "row_label": lbl, "source": src}
    for a, s, g in GROUPS:
        gm = group_mask(a, s)
        r[f"events_{g}"] = int(flag[gm].sum())
        r[f"n_{g}"] = int(gm.sum())
    r["events"], r["n"] = int(flag.sum()), len(pop)
    if r["events"] == 0:
        raise RuntimeError(f"No patient has {lbl} ({src} {key}): check the input data")
    r["share"] = r["events"] / r["n"] if r["n"] else NAN
    if reason not in NOT_USED_REASONS:
        raise ValueError(f"NOT_USED: unknown reason {reason}")
    r["reason_not_used"] = reason
    not_used.append(r)
    print(f"  {lbl:<34} {r['events']:>6,}/{r['n']:,} ({100 * r['share']:.1f}%)  {reason}")
not_used = pd.DataFrame(not_used)

au.sub("Age at first visuospatial / executive documentation (no ceiling; described only)")
age_rows = []
for s, sex in SEXES:
    for f, lbl in (("visuospatial", "Visuospatial"), ("executive", "Executive function")):
        d = pop[(pop["female"] == s) & pop["d_" + f].notna()]
        fa = au.years_between(d["d_" + f], d["birth_date"])
        for a, aid in ((1, "AID+"), (0, "AID-")):
            v = fa[d["aid_status"] == a]
            age_rows.append({"sex": sex, "domain": f, "row_label": lbl, "aid": aid, "n": len(v),
                             "median": v.median(), "q1": v.quantile(.25),
                             "q3": v.quantile(.75), "mean": v.mean()})
            print(f"  {sex:<7} {lbl:<19} {aid:<5} n = {len(v):>6,}  median "
                  f"{fmt_console(v.median(), '.2f')} ({fmt_console(v.quantile(.25), '.2f')}-"
                  f"{fmt_console(v.quantile(.75), '.2f')})")
age_first = pd.DataFrame(age_rows)

au.sub("First visuospatial documentation relative to the AD diagnosis, years (described)")
vis_rows = []
for s, sex in SEXES:
    for arm, cols in arms_for(s):
        d = apply_arm(pop[(pop["female"] == s) & pop["d_visuospatial"].notna()], cols)
        yrs = au.years_between(d["d_visuospatial"], d["ad_diagnosis_date"])
        for a, aid in ((1, "AID+"), (0, "AID-")):
            v = yrs[d["aid_status"] == a]
            vis_rows.append({"sex": sex, "arm": arm, "aid": aid, "n": len(v),
                             "median": v.median(), "q1": v.quantile(.25), "q3": v.quantile(.75)})
            print(f"  {sex:<7} {arm:<16} {aid:<5} n = {len(v):>6,}  median "
                  f"{fmt_console(v.median(), '+.2f')} ({fmt_console(v.quantile(.25), '+.2f')}"
                  f" to {fmt_console(v.quantile(.75), '+.2f')})")
vis_timing = pd.DataFrame(vis_rows)

au.sub("Hyposmia and RBD by sex (all patients)")
prev14 = []
for c, lbl in (("has_hyposmia", "Hyposmia"), ("has_rbd_any", "REM sleep behavior disorder (RBD)")):
    for s, sex in SEXES:
        v = pop.loc[pop["female"] == s, c]
        pct, lo, hi = au.wilson_ci(int(v.sum()), len(v))
        prev14.append({"feature": c[4:], "row_label": lbl, "sex": sex, "events": int(v.sum()),
                       "n": len(v), "pct": pct, "ci_low": lo, "ci_high": hi})
        print(f"  {lbl:<9} {sex:<7} {int(v.sum()):>6,}/{len(v):,} {pct:.1f}% "
              f"[{lo:.1f}-{hi:.1f}]")
prev14 = pd.DataFrame(prev14)

# =============================================================================
# SECTION 9 -- SOURCE DATA
# =============================================================================
au.section("SECTION 9 -- SOURCE DATA")


def arm_sizes(pairs=None):
    """n per AID group of each sex x arm; pairs limits to those (sex, arm) pairs."""
    rows = []
    for s, sex in SEXES:
        for arm, cols in arms_for(s):
            if pairs is not None and (sex, arm) not in pairs:
                continue
            d = apply_arm(pop[pop["female"] == s], cols)
            for a, aid in ((1, "aid_pos"), (0, "aid_neg")):
                rows.append((f"n_{aid}_{sex.lower()}_{arm}", int((d["aid_status"] == a).sum())))
    return rows


def parameters(extra=()):
    rows = [("min_events_per_term", au.MIN_EVENTS_PER_TERM), ("min_group_n", au.MIN_GROUP_N),
            ("age_ceiling_quantile", au.AGE_CEILING_QUANTILE),
            ("age_ceiling_female", ceil["ceiling_female"]),
            ("age_ceiling_male", ceil["ceiling_male"]),
            ("age_ceiling_n_ref_female", ceil["n_ref_female"]),
            ("age_ceiling_n_ref_male", ceil["n_ref_male"]),
            ("mckeith_window_yrs", MCKEITH_WINDOW_YRS),
            ("pre_peri_window_yrs", PRE_PERI_WINDOW_YRS),
            ("group_floor", au.MIN_EVENTS_PER_TERM * GROUP_FLOOR_TERMS),
            ("n_tested_features", len(POOL)),
            ("n_aid_pos", int(pop["aid_status"].sum())),
            ("n_aid_neg", int((pop["aid_status"] == 0).sum()))]
    for a, s, g in GROUPS:
        rows.append((f"n_{g}", int(group_mask(a, s).sum())))
    return pd.DataFrame(rows + list(extra), columns=["parameter", "value"])


FIT = ["n_model", "status", "reason", "coef", "se", "crit", "odds_ratio", "ci_low", "ci_high",
       "p", "q", "fdr_family_size"]
COUNTS = ["events_aid_pos", "n_aid_pos", "events_aid_neg", "n_aid_neg"]
FEATURE_COLUMNS = (["sex", "arm", "feature", "row_label", "mckeith_class", "fdr_role",
                    "group_floor_reason"] + COUNTS + FIT)
in_table = pd.Series([(s, a) in TABLE_ARMS for s, a in zip(features["sex"], features["arm"])],
                     index=features.index)
au.write_workbook(os.path.join(SOURCE_DATA_DIR, FEATURES_FILE), {
    "features": features.loc[in_table, FEATURE_COLUMNS],
    "parameters": parameters(arm_sizes(TABLE_ARMS) + [
        (f"fdr_family_size_{sex.lower()}_{arm}",
         int(features.loc[(features["sex"] == sex) & (features["arm"] == arm),
                          "fdr_family_size"].max()))
        for sex, arm in TABLE_ARMS]),
})
SEX_COUNTS = [f"{k}_{g}" for _, _, g in GROUPS for k in ("events", "n")]
au.write_workbook(os.path.join(SOURCE_DATA_DIR, SEX_FILE), {
    "sex_heterogeneity": sex_het[
        ["feature", "row_label", "mckeith_class", "fdr_role", "group_floor_reason"] + SEX_COUNTS
        + [f"{k}_{sx}" for sx in ("female", "male")
           for k in ("status", "odds_ratio", "ci_low", "ci_high")]
        + ["status", "reason", "coef", "se", "crit", "ratio", "ci_low", "ci_high", "p", "q",
           "fdr_family_size"]],
    "parameters": parameters([("fdr_family_size", int(sex_het["fdr_family_size"].max()))]),
})
au.write_workbook(os.path.join(SOURCE_DATA_DIR, NOT_USED_FILE), {
    "features": not_used,
    "parameters": parameters(),
})
for t_ in (multi, biv):     # beta's own 95% CI (the table shows beta and the odds ratio)
    t_["coef_ci_low"] = t_["coef"] - t_["crit"] * t_["se"]
    t_["coef_ci_high"] = t_["coef"] + t_["crit"] * t_["se"]
DISC = ["model", "feature", "row_label", "mckeith_class", "fdr_role", "group_floor_reason",
        "n_model", "n_dlb", "n_ad_only", "events_dlb", "events_ad_only"] + FIT[1:5] \
    + ["coef_ci_low", "coef_ci_high"] + FIT[5:]
au.write_workbook(os.path.join(SOURCE_DATA_DIR, DISCRIMINANT_FILE), {
    "multivariable": multi[DISC],
    "bivariable": biv[DISC],
    "fit": disc_fit,
    "parameters": parameters([("fdr_family_size_multivariable",
                               int(multi["fdr_family_size"].max())),
                              ("fdr_family_size_bivariable", int(biv["fdr_family_size"].max())),
                              ("n_features_in_model", len(MODEL_COLS))]),
})

# ---- Figure: feature specificity ---------------------------------------------
SERIES = [("female_all", "Female", "all"), ("female_dlb_excluded", "Female", "dlb_excluded"),
          ("male_all", "Male", "all")]
fig_a = []
for c, lbl in POOL:
    r = {"feature": c[4:], "row_label": lbl, "mckeith_class": MCKEITH_CLASS[c]}
    for key, sex, arm in SERIES:
        f = features[(features["sex"] == sex) & (features["arm"] == arm)
                     & (features["feature"] == c[4:])].iloc[0]
        for k in ("status", "fdr_role", "odds_ratio", "ci_low", "ci_high", "p", "q"):
            r[f"{k}_{key}"] = f[k]
    h = sex_het[sex_het["feature"] == c[4:]].iloc[0]
    r.update({"sex_status": h["status"], "sex_fdr_role": h["fdr_role"],
              "sex_ratio": h["ratio"], "sex_ci_low": h["ci_low"],
              "sex_ci_high": h["ci_high"], "sex_p": h["p"], "sex_q": h["q"]})
    fig_a.append(r)
fig_b = [{"feature": c[4:], "row_label": lbl, "mckeith_class": MCKEITH_CLASS[c],
          "status": multi.loc[multi["feature"] == c[4:], "status"].iloc[0],
          "coef": multi.loc[multi["feature"] == c[4:], "coef"].iloc[0],
          "odds_ratio_female_all": features.loc[
              (features["sex"] == "Female") & (features["arm"] == "all")
              & (features["feature"] == c[4:]), "odds_ratio"].iloc[0]} for c, lbl in POOL]
SCORE_PARAMETERS = [("score_core_points", SCORE_CORE_POINTS),
                    ("score_supportive_cap", SCORE_SUPPORTIVE_CAP),
                    ("score_max", SCORE_MAX), ("score_max_observed", smax_observed),
                    ("score_n_core_features", len(SCORE_CORE)),
                    ("score_n_supportive_features", len(SCORE_SUPPORTIVE))]
print("  Score parameters: " + ", ".join(f"{k} {v}" for k, v in SCORE_PARAMETERS))
au.write_workbook(os.path.join(SOURCE_DATA_DIR, FIGURE_SPECIFICITY_FILE), {
    "a": pd.DataFrame(fig_a),
    "b": pd.DataFrame(fig_b),
    "c_curves": curves,
    "c_reference": reference,
    "c_summary": pd.DataFrame(summary),
    "parameters": parameters(SCORE_PARAMETERS + [
                              ("bootstrap_resamples", N_BOOT),
                              ("bootstrap_seed", BOOT_SEED),
                              ("n_coded_dlb_reference", len(ref_scores)),
                              ("coded_dlb_reference_drawn", ref_shown),
                              ("coded_dlb_reference_min_n", SCORE_REFERENCE_MIN_N)]),
})

# ---- Figure: co-occurrence ----------------------------------------------------
PRIMARY = [code for st, code, _ in COMBOS if st == "primary"]
fig3a = cooccur[cooccur["combination"].isin(PRIMARY)]
fig3a_or = cooccur_or[cooccur_or["combination"].isin(PRIMARY)][
    ["sex", "arm", "combination", "row_label", "status", "reason", "odds_ratio", "ci_low",
     "ci_high", "p", "q", "fdr_family_size"]]
aidpos_female = pop[(pop["female"] == 1) & (pop["aid_status"] == 1)]
fig3c, fold = [], NAN
for arm, cols in ARMS[:2]:
    d = apply_arm(aidpos_female, cols)
    for burden, sel in (("0_1", d["_core_count"] <= 1), ("2_3", d["_core_count"] >= 2)):
        k, n = int(d.loc[sel, "has_rbd_any"].sum()), int(sel.sum())
        pct, lo, hi = au.wilson_ci(k, n)
        fig3c.append({"arm": arm, "core_burden": burden, "events": k, "n": n, "pct": pct,
                      "ci_low": lo, "ci_high": hi})
fig3c = pd.DataFrame(fig3c)
ex = fig3c[fig3c["arm"] == "dlb_excluded"].set_index("core_burden")["pct"]
if ex.get("0_1", NAN) > 0 and not np.isnan(ex.get("2_3", NAN)):
    fold = float(ex["2_3"] / ex["0_1"])
bk, bn = int(aidpos_female["has_rbd_any"].sum()), len(aidpos_female)
bpct, blo, bhi = au.wilson_ci(bk, bn)
au.write_workbook(os.path.join(SOURCE_DATA_DIR, FIGURE_COOCCURRENCE_FILE), {
    "a": fig3a,
    "a_odds_ratios": fig3a_or,
    "b": pd.DataFrame(fig3b),
    "c": fig3c,
    "c_baseline": pd.DataFrame([{"events": bk, "n": bn, "pct": bpct, "ci_low": blo,
                                 "ci_high": bhi}]),
    "parameters": parameters([("pre_peri_window_yrs_figure", PRE_PERI_WINDOW_YRS),
                              ("rbd_fold_dlb_excluded", fold)]),
})

coef_out = pd.DataFrame({"feature": POOL_COLS,
                         "coefficient": multi.set_index("feature").loc[
                             [c[4:] for c in POOL_COLS], "coef"].astype(float).to_numpy()})
au.write_parquet(coef_out, COEFFICIENTS_OUT_PATH)
print(f"  Saved: {COEFFICIENTS_OUT_PATH}  ({len(coef_out)} features)")
au.write_workbook(os.path.join(SOURCE_DATA_DIR, CHECKS_FILE), {
    "age_ceiling": pd.DataFrame(ceiling_rows),
    "feature_counts": pd.DataFrame(count_rows),
    "features_dlb_pd_excluded": features.loc[~in_table, FEATURE_COLUMNS],
    "described_features": descriptive,
    "pre_peri_diagnosis": peri,
    "cooccurrence": cooccur,
    "cooccurrence_odds_ratios": cooccur_or,
    "cooccurrence_timing": co_timing,
    "rbd_by_core_count": by_count,
    "rbd_four_features": four,
    "rbd_all_three_core_features": split,
    "sex_heterogeneity_extras": sex_extra,
    "profile_prevalence": prof_prev,
    "profile_similarity": prof_sim,
    "discriminant_scores": disc_scores,
    "score_distribution": score_dist,
    "score_tiers": score_tiers,
    "age_first_documentation": age_first,
    "visuospatial_timing": vis_timing,
    "hyposmia_rbd_by_sex": prev14,
    "parameters": parameters(arm_sizes() + [(f"tost_margin_{i}", m)
                                            for i, m in enumerate(TOST_MARGINS, 1)]),
})
print("\n" + au.SEP + "\n  COMPLETE\n" + au.SEP)