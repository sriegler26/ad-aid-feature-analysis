"""
cohort_characteristics.py
=========================
Cohort tables: who is in the analysis, and who is left out. Writes the source
data of five tables; cohort_characteristics_tables.py renders them.

INPUTS
------
  The patient table (AID status, sex, key dates, analysis-population flag,
  self-reported race and ethnicity, coded DLB, PD and vascular dementia) and
  the note table (see PATHS).

OUTPUTS (source_data/)
----------------------
  table_1.xlsx                                   cohort characteristics, analysis population
  table_cohort_characteristics_by_sex.xlsx       the same within female and male patients
  table_cohort_characteristics_no_notes.xlsx     patients without notes, and
                                                 the share without notes by AID
  table_cohort_characteristics_no_notes_by_sex.xlsx
  table_lewy_body_diagnoses.xlsx                 coded DLB, PD and Lewy body
                                                 disease, AID+ vs AID-, within sex

THE COHORTS
-----------
  Analysis population: patients with at least one clinical note. Patients
  without notes: the others; they have no note dates, so only demographic
  rows exist for them. All patients: both.

CHARACTERISTICS (the cohort table and its three companions)
--------------------------------------------------
  Demographics: sex (the overall tables only), age at AD diagnosis, death
  documented, self-reported race, ethnicity. Documentation (analysis
  population only): total note count (every note, undated included) and age
  at first note (age at the first dated note).
  Continuous rows: median, Q1 and Q3 per arm (with mean and SD), written only
  when the arm has at least MIN_SUMMARY_N values, so no summary of fewer
  than five patients is published. Categorical rows: the count and the group
  size per arm (every count is written, n >= 1).
  Race and ethnicity: raw values are grouped by the two maps below; an
  unmapped value counts as "Other" and is printed every run with its count.
  The by-sex tables list, in both sexes, every category present in that
  cohort, so the two strata have the same rows.

  P values are descriptive and corrected for nothing (no FDR family):
    continuous            Mann-Whitney U, two-sided;
    two categories        Fisher's exact test (sex, death; one rule for every
                          2x2);
    race, ethnicity       chi-square over every category with at least one
                          patient, unknown included, no continuity
                          correction.
  The gate is the shared one (analysis_utils): a 2x2 needs >= 10 patients per
  group and an event that varies (two_by_two_reason); Mann-Whitney U and
  the chi-square test need >= 10 patients (non-missing values) per
  group (group_minimum_reason). A row not tested keeps its counts and
  summaries and carries the reason.
  SMD: standardized mean difference, AID+ minus AID-, unweighted.
  Continuous: pooled SD (sample variances); categories: proportion-based,
  each category against all others; sex and death: proportion-based.

  Share without notes (no-note workbook): patients without notes as a share
  of all patients, AID+ vs AID-, overall and within each sex: Fisher's exact
  test with the conditional maximum-likelihood odds ratio and its exact 95%
  CI, same gate; and the proportion-based SMD.

LEWY BODY DIAGNOSES
-------------------
  Coded dementia with Lewy bodies (DLB), coded Parkinson's disease (PD) and
  Lewy body disease (LBD) = DLB or PD (a patient with both counts once).
  Compared AID+ vs AID- within female and within male patients only (no
  pooled comparison), in two cohorts: the analysis population (with notes)
  and the whole cohort (all patients). Each comparison: Fisher's exact test,
  conditional maximum-likelihood odds ratio, exact 95% CI; gate
  two_by_two_reason.
  Does restricting to the analysis population inflate the association?
  The ratio of the two odds ratios (analysis population / whole cohort),
  within each sex: above 1, the restriction inflates the AID association;
  at or below 1, it does not. The analysis population is part of the whole
  cohort, so the two estimates are not independent; the CI and P come from a
  nonparametric bootstrap over the patients of the whole cohort
  (BOOTSTRAP_RESAMPLES resamples, fixed seed): each resample recomputes both
  odds ratios, percentile 95% CI of the ratio, two-sided P = 2 x the smaller
  tail share of log(ratio) at 0, as (k + 1) / (B + 1). Run only when both
  odds ratios are tested and estimable; not reported when more than 1% of the
  resamples give an odds ratio of 0 or infinity.
  No FDR correction.

  Printed to the console only (no workbook, no table row):
  - Coded vascular dementia (VD), AID+ vs AID- within female and within male
    patients, in both cohorts, as for DLB/PD/LBD (Fisher's exact test,
    conditional ML OR, exact 95% CI, gate two_by_two_reason).
  - Female-vs-male ratio of odds ratios (Altman-Bland) for DLB, PD, LBD and
    VD, analysis population only: OR_female / OR_male; each log-OR SE from
    its exact CI, (ln hi - ln lo) / (2 x 1.959964); z = (ln OR_f - ln OR_m) /
    sqrt(SE_f^2 + SE_m^2); 95% CI exp(d +/- 1.959964 SE); two-sided normal P.
    Run only when both odds ratios are tested and estimable.
  - Specificity of the sex difference to DLB against VD: theta =
    ln(ratio_DLB) - ln(ratio_VD), analysis population. DLB and VD share
    patients, so the CI and P come from a nonparametric bootstrap
    (SPECIFICITY_RESAMPLES resamples, SPECIFICITY_SEED) of the patients of
    the analysis population within each sex: each resample recomputes the
    four conditional odds ratios (DLB and VD, female and male), percentile
    95% CI of theta, two-sided P = 2 x the smaller tail share at 0, as
    (k + 1) / (B + 1). Not reported when more than
    BOOTSTRAP_MAX_NOT_ESTIMABLE of the resamples give an odds ratio of 0 or
    infinity.

Every count, P value and SMD is computed here; the table script only
formats. Raises on a missing input or column, a flag other than 0/1,
a missing birth or AD diagnosis date, a duplicate key, or an
analysis-population flag that disagrees with the notes.
"""

import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu, norm
from scipy.stats.contingency import odds_ratio

import analysis_utils as au

warnings.filterwarnings("ignore")

# =============================================================================
# PATHS (relative to the data folder the script is run from)
# =============================================================================
COHORT_PATIENTS_PATH = ""  # patient table: AID status, sex, key dates (birth, AD diagnosis, death), analysis-population flag, self-reported race and ethnicity, coded DLB, PD and vascular dementia
COHORT_NOTES_PATH = ""  # note table: one row per clinical note, with its date and specialty
SOURCE_DATA_DIR = ""  # folder for the source-data workbooks written here

TABLE_1_FILE = "table_1.xlsx"
BY_SEX_FILE = "table_cohort_characteristics_by_sex.xlsx"
NO_NOTES_FILE = "table_cohort_characteristics_no_notes.xlsx"
NO_NOTES_BY_SEX_FILE = "table_cohort_characteristics_no_notes_by_sex.xlsx"
LEWY_BODY_FILE = "table_lewy_body_diagnoses.xlsx"

# =============================================================================
# SETTINGS
# =============================================================================
MIN_SUMMARY_N = 5       # median / quartiles / mean / SD written from 5 values up
NAN = float("nan")
EXTRA_COLUMNS = ["first_race", "ethnicity", "coded_dementia_with_lewy_bodies",
                 "coded_parkinsons_disease", "coded_vascular_dementia"]

# Raw value -> readable category. Null is "Unknown/not reported"; a value not
# in the map is "Other" (printed every run).
UNKNOWN = "Unknown/not reported"
RACE_MAP = {
    "White": "White",
    "Asian": "Asian",
    "Other": "Other",
    "Black or African American": "Black or African American",
    "Native Hawaiian or Other Pacific Islander": "Native Hawaiian or Pacific Islander",
    "Native Hawaiian": "Native Hawaiian or Pacific Islander",
    "Other Pacific Islander": "Native Hawaiian or Pacific Islander",
    "Native American or Alaska Native": "Native American or Alaska Native",
    "Declined": UNKNOWN,
    "Unknown/Declined": UNKNOWN,
    "Unknown": UNKNOWN,
}
ETHNICITY_MAP = {
    "Not Hispanic or Latino": "Not Hispanic or Latino",
    "Hispanic or Latino": "Hispanic or Latino",
    "*Unspecified": UNKNOWN,
    "Unknown/Declined": UNKNOWN,
    "Unknown": UNKNOWN,
    "Declined": UNKNOWN,
}
CATEGORICAL = [("first_race", "Self-reported race", RACE_MAP),
               ("ethnicity", "Ethnicity", ETHNICITY_MAP)]

ROW_COLUMNS = [
    "section", "row_type", "variable", "row_label",
    "count_aid_pos", "n_aid_pos", "count_aid_neg", "n_aid_neg",
    "n_nonmissing_aid_pos", "median_aid_pos", "q1_aid_pos", "q3_aid_pos",
    "mean_aid_pos", "sd_aid_pos",
    "n_nonmissing_aid_neg", "median_aid_neg", "q1_aid_neg", "q3_aid_neg",
    "mean_aid_neg", "sd_aid_neg",
    "test", "statistic", "status", "reason", "p", "smd",
]

DIAGNOSES = [("dlb", "Dementia with Lewy bodies", "component"),
             ("pd", "Parkinson's disease", "component"),
             ("lbd", "Lewy body disease (DLB or PD)", "combined")]
DIAGNOSIS_COLUMNS = {"dlb": "coded_dementia_with_lewy_bodies",
                     "pd": "coded_parkinsons_disease", "lbd": "coded_lewy_body_disease",
                     "vd": "coded_vascular_dementia"}   # vd: log only, not in DIAGNOSES
COHORTS = [("with_notes", "Analysis population (with notes)"),
           ("all", "Whole cohort (all patients)")]
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20261026
BOOTSTRAP_MAX_NOT_ESTIMABLE = 0.01   # share of resamples with an OR of 0 or infinity
# Log only: female-vs-male ratio of ORs, and its DLB-vs-VD specificity test
Z_95 = 1.959964                      # two-sided 95% normal quantile
RATIO_DIAGNOSES = ["dlb", "pd", "lbd", "vd"]
SPECIFICITY_RESAMPLES = 2000
SPECIFICITY_SEED = 20261104          # distinct from BOOTSTRAP_SEED


# =============================================================================
# TESTS
# =============================================================================
def exact_2x2(e1, n1, e2, n2):
    """Fisher's exact test of e1/n1 vs e2/n2 with the conditional maximum-
    likelihood odds ratio and its exact 95% CI. An odds ratio of 0 or
    infinity (a zero cell) is not estimable: written empty, with
    or_estimable = 0; the P value stands."""
    table = [[int(e1), int(n1 - e1)], [int(e2), int(n2 - e2)]]
    res = odds_ratio(table, kind="conditional")
    ci = res.confidence_interval(0.95)
    orv, lo, hi = float(res.statistic), float(ci.low), float(ci.high)
    est = int(np.isfinite(orv) and orv > 0)
    if not est:
        orv = lo = hi = NAN
    return {"odds_ratio": orv, "ci_low": lo if np.isfinite(lo) else NAN,
            "ci_high": hi if np.isfinite(hi) else NAN, "or_estimable": est,
            "p": float(fisher_exact(table)[1])}


def smd_counts(e1, n1, e2, n2):
    """Proportion-based SMD of e1/n1 vs e2/n2 (NaN for an empty group)."""
    if n1 == 0 or n2 == 0:
        return NAN
    var = np.r_[np.ones(int(e1)), np.zeros(int(n1 - e1)),
                np.ones(int(e2)), np.zeros(int(n2 - e2))]
    grp = np.r_[np.ones(int(n1)), np.zeros(int(n2))]
    return au.smd_proportion(var, grp, min_n=1)


def row(**kw):
    r = {c: NAN for c in ROW_COLUMNS}
    r.update(test="", status="", reason="")
    r.update(kw)
    return r


def summary(series, tag):
    s = pd.to_numeric(series, errors="coerce").dropna()
    out = {f"n_nonmissing_{tag}": int(len(s))}
    if len(s) >= MIN_SUMMARY_N:
        out.update({f"median_{tag}": s.median(), f"q1_{tag}": s.quantile(0.25),
                    f"q3_{tag}": s.quantile(0.75), f"mean_{tag}": s.mean(),
                    f"sd_{tag}": s.std(ddof=1)})
    return out, s


def continuous_row(section_, variable, label, pos, neg):
    sp, a = summary(pos[variable], "aid_pos")
    sn, b = summary(neg[variable], "aid_neg")
    r = row(section=section_, row_type="continuous", variable=variable, row_label=label,
            n_aid_pos=len(pos), n_aid_neg=len(neg), **sp, **sn,
            test="Mann-Whitney U", smd=au.smd_from_groups(a, b))
    why = au.group_minimum_reason(len(a), len(b))
    if why:
        r.update(status="not_tested", reason=why)
    else:
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        r.update(status="tested", statistic=float(u), p=float(p))
    return r


def binary_test(r, e1, n1, e2, n2, event):
    why = au.two_by_two_reason(e1, n1, e2, n2, event)
    if why:
        r.update(status="not_tested", reason=why)
    else:
        r.update(status="tested", p=exact_2x2(e1, n1, e2, n2)["p"])
    return r


def binary_row(section_, variable, label, pos, neg, event):
    e1, e2 = int(pos[variable].sum()), int(neg[variable].sum())
    r = row(section=section_, row_type="binary", variable=variable, row_label=label,
            count_aid_pos=e1, n_aid_pos=len(pos), count_aid_neg=e2, n_aid_neg=len(neg),
            test="Fisher exact", smd=smd_counts(e1, len(pos), e2, len(neg)))
    return binary_test(r, e1, len(pos), e2, len(neg), event)


def sex_rows(section_, pos, neg):
    e1, e2 = int(pos["female"].sum()), int(neg["female"].sum())
    head = row(section=section_, row_type="category_header", variable="female",
               row_label="Sex", n_aid_pos=len(pos), n_aid_neg=len(neg),
               test="Fisher exact", smd=smd_counts(e1, len(pos), e2, len(neg)))
    binary_test(head, e1, len(pos), e2, len(neg), "female sex")
    return [head,
            row(section=section_, row_type="category", variable="female",
                row_label="Female", count_aid_pos=e1, n_aid_pos=len(pos),
                count_aid_neg=e2, n_aid_neg=len(neg)),
            row(section=section_, row_type="category", variable="female",
                row_label="Male", count_aid_pos=len(pos) - e1, n_aid_pos=len(pos),
                count_aid_neg=len(neg) - e2, n_aid_neg=len(neg))]


def category_rows(section_, variable, label, pos, neg, categories):
    cp = pos[variable + "_category"].value_counts()
    cn = neg[variable + "_category"].value_counts()
    head = row(section=section_, row_type="category_header", variable=variable,
               row_label=label, n_aid_pos=len(pos), n_aid_neg=len(neg), test="Chi-square")
    present = [c for c in categories if cp.get(c, 0) + cn.get(c, 0) > 0]
    why = au.group_minimum_reason(len(pos), len(neg))
    if not why and len(present) < 2:
        why = "fewer than two categories present"
    if why:
        head.update(status="not_tested", reason=why)
    else:
        ct = np.array([[cp.get(c, 0) for c in present], [cn.get(c, 0) for c in present]])
        chi2, p, dof, _ = chi2_contingency(ct, correction=False)
        head.update(status="tested", statistic=float(chi2), p=float(p),
                    test=f"Chi-square (df = {int(dof)})")
    out = [head]
    for c in categories:
        e1, e2 = int(cp.get(c, 0)), int(cn.get(c, 0))
        out.append(row(section=section_, row_type="category", variable=variable,
                       row_label=c, count_aid_pos=e1, n_aid_pos=len(pos),
                       count_aid_neg=e2, n_aid_neg=len(neg),
                       smd=smd_counts(e1, len(pos), e2, len(neg))))
    return out


def characteristics(group, categories, include_sex, documentation):
    """The rows of one comparison, AID+ vs AID-, within `group`."""
    pos, neg = group[group["aid_status"] == 1], group[group["aid_status"] == 0]
    rows = []
    if include_sex:
        rows += sex_rows("demographics", pos, neg)
    rows.append(continuous_row("demographics", "age_at_dx", "Age at AD diagnosis, years",
                               pos, neg))
    rows.append(binary_row("demographics", "death_flag", "Death documented", pos, neg,
                           "a documented death"))
    for variable, label, _ in CATEGORICAL:
        rows += category_rows("demographics", variable, label, pos, neg,
                              categories[variable])
    if documentation:
        rows.append(continuous_row("documentation", "note_count", "Total note count",
                                   pos, neg))
        rows.append(continuous_row("documentation", "age_at_entry",
                                   "Age at first note, years", pos, neg))
    return pd.DataFrame(rows)[ROW_COLUMNS]


def categories_of(group):
    return {v: sorted(group[v + "_category"].unique()) for v, _, _ in CATEGORICAL}


def by_sex(group, include_sex_row, documentation):
    """Both sexes with the categories present in `group`, so the strata align."""
    cats = categories_of(group)
    parts = []
    for s in (1, 0):
        t = characteristics(group[group["female"] == s], cats, include_sex_row,
                            documentation)
        t.insert(0, "sex", au.SEX_LABELS[s])
        parts.append(t)
    return pd.concat(parts, ignore_index=True)


def group_sizes(group, by_sex_):
    out = [("n_aid_pos", int((group["aid_status"] == 1).sum())),
           ("n_aid_neg", int((group["aid_status"] == 0).sum()))]
    if by_sex_:
        for a, tag in ((1, "aid_pos"), (0, "aid_neg")):
            for s, sx in ((1, "female"), (0, "male")):
                out.append((f"n_{tag}_{sx}", int(((group["aid_status"] == a)
                                                  & (group["female"] == s)).sum())))
    return out


def parameters(group, by_sex_, extra=()):
    rows = [("min_group_n", au.MIN_GROUP_N), ("min_summary_n", MIN_SUMMARY_N)]
    rows += group_sizes(group, by_sex_) + list(extra)
    return pd.DataFrame(rows, columns=["parameter", "value"])


def print_rows(title, df):
    au.sub(title)
    for r in df.itertuples(index=False):
        where = f"{r.sex:<7} " if "sex" in df.columns else ""
        if r.row_type == "continuous":
            vals = (f"AID+ {r.median_aid_pos:.2f} ({r.q1_aid_pos:.2f}-{r.q3_aid_pos:.2f}) "
                    f"n {r.n_nonmissing_aid_pos}  AID- {r.median_aid_neg:.2f} "
                    f"({r.q1_aid_neg:.2f}-{r.q3_aid_neg:.2f}) n {r.n_nonmissing_aid_neg}")
        elif r.row_type in ("binary", "category"):
            vals = (f"AID+ {int(r.count_aid_pos)}/{int(r.n_aid_pos)}  "
                    f"AID- {int(r.count_aid_neg)}/{int(r.n_aid_neg)}")
        else:
            vals = ""
        test = (f"P {r.p:.4g}" if r.status == "tested"
                else (f"not tested: {r.reason}" if r.status else ""))
        smd = f"SMD {r.smd:+.3f}" if not pd.isna(r.smd) else ""
        print(f"  {where}{r.row_label[:34]:<34} {vals}  {test}  {smd}")


# =============================================================================
# LOAD
# =============================================================================
au.section("LOADING DATA")
cohort, notes = au.load_cohort(COHORT_PATIENTS_PATH, COHORT_NOTES_PATH, EXTRA_COLUMNS)

for variable, label, mapping in CATEGORICAL:
    raw = cohort[variable]
    cohort[variable + "_category"] = raw.map(mapping).where(raw.notna(), UNKNOWN)
    unmapped = raw.notna() & ~raw.isin(list(mapping))
    cohort.loc[unmapped, variable + "_category"] = "Other"
    au.count_line(f"{label.lower()}: missing values (counted as {UNKNOWN})",
                  raw.isna().sum(), len(raw))
    au.count_line(f"{label.lower()}: values not in the map (counted as Other)",
                  unmapped.sum(), len(raw))
    for value, k in raw[unmapped].value_counts().sort_index().items():
        print(f"      {value!r}: {k:,}")

with_notes = cohort[cohort["in_population"] == 1]
without_notes = cohort[cohort["in_population"] == 0]
for name, g in (("with notes", with_notes), ("without notes", without_notes)):
    print(f"  Patients {name}: {len(g):,} (AID+ {int(g['aid_status'].sum()):,}, "
          f"AID- {int((g['aid_status'] == 0).sum()):,}; female {int(g['female'].sum()):,}, "
          f"male {int((g['female'] == 0).sum()):,})")
if len(without_notes) == 0:
    print("  [NOTE] no patient without notes: the no-note tables hold counts of 0 "
          "and no test.")

# =============================================================================
# CHARACTERISTICS
# =============================================================================
au.section("CHARACTERISTICS")
table_1 = characteristics(with_notes, categories_of(with_notes), True, True)
table_by_sex = by_sex(with_notes, False, True)
no_notes = characteristics(without_notes, categories_of(without_notes), True, False)
no_notes_by_sex = by_sex(without_notes, False, False)
print_rows("Cohort characteristics: analysis population, AID+ vs AID-", table_1)
print_rows("Analysis population by sex", table_by_sex)
print_rows("Patients without notes", no_notes)
print_rows("Patients without notes by sex", no_notes_by_sex)

au.sub("Share of patients without notes, AID+ vs AID-")
exclusion_rows = []
for stratum, g in (("all", cohort), ("Female", cohort[cohort["female"] == 1]),
                   ("Male", cohort[cohort["female"] == 0])):
    pos, neg = g[g["aid_status"] == 1], g[g["aid_status"] == 0]
    e1, e2 = int((pos["in_population"] == 0).sum()), int((neg["in_population"] == 0).sum())
    r = {"stratum": stratum, "without_notes_aid_pos": e1, "n_aid_pos": len(pos),
         "without_notes_aid_neg": e2, "n_aid_neg": len(neg), "status": "", "reason": "",
         "odds_ratio": NAN, "ci_low": NAN, "ci_high": NAN, "or_estimable": NAN, "p": NAN,
         "smd": smd_counts(e1, len(pos), e2, len(neg))}
    why = au.two_by_two_reason(e1, len(pos), e2, len(neg), "no notes")
    if why:
        r.update(status="not_tested", reason=why)
    else:
        r.update(status="tested", **exact_2x2(e1, len(pos), e2, len(neg)))
    exclusion_rows.append(r)
    res = (f"OR {r['odds_ratio']:.3f} ({r['ci_low']:.3f}-{r['ci_high']:.3f})  P {r['p']:.4g}"
           if r["status"] == "tested" else f"not tested: {r['reason']}")
    print(f"  {stratum:<7} AID+ {e1:,}/{len(pos):,}  AID- {e2:,}/{len(neg):,}  {res}  "
          f"SMD {r['smd']:+.3f}")
exclusion = pd.DataFrame(exclusion_rows)

# =============================================================================
# LEWY BODY DIAGNOSES
# =============================================================================
au.section("CODED LEWY BODY DIAGNOSES, AID+ vs AID-, WITHIN SEX")
cohort["coded_lewy_body_disease"] = ((cohort["coded_dementia_with_lewy_bodies"] == 1)
                                     | (cohort["coded_parkinsons_disease"] == 1)).astype(int)
au.count_line("patients with both coded DLB and coded PD (counted once in LBD)",
              ((cohort["coded_dementia_with_lewy_bodies"] == 1)
               & (cohort["coded_parkinsons_disease"] == 1)).sum(), len(cohort))
cohort_groups = {"with_notes": cohort["in_population"] == 1,
                 "all": pd.Series(True, index=cohort.index)}


def conditional_or(e1, n1, e2, n2):
    """Conditional maximum-likelihood odds ratio of e1/n1 vs e2/n2 (NaN when
    0 or infinite)."""
    v = float(odds_ratio([[int(e1), int(n1 - e1)], [int(e2), int(n2 - e2)]],
                         kind="conditional").statistic)
    return v if np.isfinite(v) and v > 0 else NAN


def bootstrap_restriction(g, col, rng):
    """Bootstrap of OR(analysis population) / OR(whole cohort) over the
    patients of g (one sex, every patient). Resampling patients with
    replacement equals a multinomial draw over the eight AID x diagnosis x
    notes cells, which is what is drawn here."""
    cells = g.groupby(["aid_status", col, "in_population"]).size()
    keys = [(a, d, w) for a in (1, 0) for d in (1, 0) for w in (1, 0)]
    counts = np.array([cells.get(k, 0) for k in keys], dtype=float)
    draws = rng.multinomial(len(g), counts / counts.sum(), size=BOOTSTRAP_RESAMPLES)
    idx = {k: i for i, k in enumerate(keys)}
    ratios = np.full(BOOTSTRAP_RESAMPLES, np.nan)
    for b, d in enumerate(draws):
        ep_w, np_w = d[idx[(1, 1, 1)]], d[idx[(1, 1, 1)]] + d[idx[(1, 0, 1)]]
        en_w, nn_w = d[idx[(0, 1, 1)]], d[idx[(0, 1, 1)]] + d[idx[(0, 0, 1)]]
        ep_a = ep_w + d[idx[(1, 1, 0)]]
        np_a = np_w + d[idx[(1, 1, 0)]] + d[idx[(1, 0, 0)]]
        en_a = en_w + d[idx[(0, 1, 0)]]
        nn_a = nn_w + d[idx[(0, 1, 0)]] + d[idx[(0, 0, 0)]]
        if min(np_w, nn_w) == 0:
            continue
        ratios[b] = conditional_or(ep_w, np_w, en_w, nn_w) / conditional_or(ep_a, np_a, en_a, nn_a)
    return ratios


dx_rows, ratio_rows = [], []
rng = np.random.default_rng(BOOTSTRAP_SEED)
for s in (1, 0):
    sx = au.SEX_LABELS[s]
    for code, label, role in DIAGNOSES:
        col = DIAGNOSIS_COLUMNS[code]
        rows = {}
        for ckey, clabel in COHORTS:
            g = cohort[(cohort["female"] == s) & cohort_groups[ckey]]
            pos, neg = g[g["aid_status"] == 1], g[g["aid_status"] == 0]
            e1, e2 = int(pos[col].sum()), int(neg[col].sum())
            r = {"sex": sx, "diagnosis": code, "cohort": ckey, "row_label": label,
                 "cohort_label": clabel, "role": role, "events_aid_pos": e1,
                 "n_aid_pos": len(pos), "events_aid_neg": e2, "n_aid_neg": len(neg),
                 "status": "", "reason": "", "odds_ratio": NAN, "ci_low": NAN,
                 "ci_high": NAN, "or_estimable": NAN, "p": NAN}
            why = au.two_by_two_reason(e1, len(pos), e2, len(neg), f"coded {code.upper()}")
            if why:
                r.update(status="not_tested", reason=why)
            else:
                r.update(status="tested", **exact_2x2(e1, len(pos), e2, len(neg)))
            dx_rows.append(r)
            rows[ckey] = r
            res = (f"OR {r['odds_ratio']:.3f} ({r['ci_low']:.3f}-{r['ci_high']:.3f})  "
                   f"P {r['p']:.4g}" if r["status"] == "tested"
                   else f"not tested: {r['reason']}")
            print(f"  {sx:<7} {code.upper():<4} {ckey:<11} AID+ {e1:,}/{len(pos):,}  "
                  f"AID- {e2:,}/{len(neg):,}  {res}")

        # does restricting to the analysis population inflate the association?
        w, a_ = rows["with_notes"], rows["all"]
        r = {"sex": sx, "diagnosis": code, "row_label": label, "role": role,
             "odds_ratio_with_notes": w["odds_ratio"], "odds_ratio_all": a_["odds_ratio"],
             "ratio": NAN, "ci_low": NAN, "ci_high": NAN, "p": NAN, "status": "",
             "reason": "", "n_resamples": BOOTSTRAP_RESAMPLES,
             "n_resamples_not_estimable": NAN}
        ok = all(x["status"] == "tested" and x["or_estimable"] == 1 for x in (w, a_))
        if not ok:
            r.update(status="not_tested", reason="an odds ratio was not tested or not "
                                                 "estimable in one of the two cohorts")
        else:
            g = cohort[cohort["female"] == s]
            ratios = bootstrap_restriction(g, col, rng)
            bad = int(np.isnan(ratios).sum())
            r["n_resamples_not_estimable"] = bad
            r["ratio"] = w["odds_ratio"] / a_["odds_ratio"]
            if bad > BOOTSTRAP_MAX_NOT_ESTIMABLE * BOOTSTRAP_RESAMPLES:
                r.update(status="not_tested",
                         reason=f"an odds ratio of 0 or infinity in {bad} of "
                                f"{BOOTSTRAP_RESAMPLES} bootstrap resamples")
            else:
                lr = np.log(ratios[~np.isnan(ratios)])
                k = len(lr)
                lo, hi = np.exp(np.percentile(lr, [2.5, 97.5]))
                p = min(1.0, 2 * (min((lr <= 0).sum(), (lr >= 0).sum()) + 1) / (k + 1))
                r.update(status="tested", ci_low=lo, ci_high=hi, p=p)
        ratio_rows.append(r)
        res = (f"ratio {r['ratio']:.3f} ({r['ci_low']:.3f}-{r['ci_high']:.3f})  "
               f"P {r['p']:.4g}  resamples not estimable {r['n_resamples_not_estimable']}"
               if r["status"] == "tested" else f"not tested: {r['reason']}")
        print(f"  {sx:<7} {code.upper():<4} analysis population / whole cohort: {res}")
diagnoses = pd.DataFrame(dx_rows)
restriction = pd.DataFrame(ratio_rows)

# --- Console only from here to SOURCE DATA: nothing below enters a workbook --
VD_COLUMN = DIAGNOSIS_COLUMNS["vd"]
if VD_COLUMN not in cohort.columns:
    raise KeyError(f"patient table: missing column {VD_COLUMN}")

au.sub("Coded vascular dementia (VD), AID+ vs AID- (log only, not in the table)")
vd_rows = []
for s in (1, 0):
    sx = au.SEX_LABELS[s]
    for ckey, clabel in COHORTS:
        g = cohort[(cohort["female"] == s) & cohort_groups[ckey]]
        pos, neg = g[g["aid_status"] == 1], g[g["aid_status"] == 0]
        e1, e2 = int(pos[VD_COLUMN].sum()), int(neg[VD_COLUMN].sum())
        r = {"sex": sx, "diagnosis": "vd", "cohort": ckey, "events_aid_pos": e1,
             "n_aid_pos": len(pos), "events_aid_neg": e2, "n_aid_neg": len(neg),
             "status": "", "reason": "", "odds_ratio": NAN, "ci_low": NAN,
             "ci_high": NAN, "or_estimable": NAN, "p": NAN}
        why = au.two_by_two_reason(e1, len(pos), e2, len(neg), "coded VD")
        if why:
            r.update(status="not_tested", reason=why)
        else:
            r.update(status="tested", **exact_2x2(e1, len(pos), e2, len(neg)))
        vd_rows.append(r)
        res = (f"OR {r['odds_ratio']:.3f} ({r['ci_low']:.3f}-{r['ci_high']:.3f})  "
               f"P {r['p']:.4g}" if r["status"] == "tested"
               else f"not tested: {r['reason']}")
        print(f"  {sx:<7} {'VD':<4} {ckey:<11} AID+ {e1:,}/{len(pos):,}  "
              f"AID- {e2:,}/{len(neg):,}  {res}")

# the analysis-population OR of each diagnosis and sex: {(code, sex label): row}
pop_or = {(r["diagnosis"], r["sex"]): r for r in dx_rows + vd_rows
          if r["cohort"] == "with_notes"}

au.sub("Female-vs-male ratio of odds ratios (Altman-Bland), analysis population "
       "(log only)")
sex_ratio = {}
for code in RATIO_DIAGNOSES:
    f_, m_ = pop_or[(code, au.SEX_LABELS[1])], pop_or[(code, au.SEX_LABELS[0])]
    ok = all(x["status"] == "tested" and x["or_estimable"] == 1
             and np.isfinite(x["ci_low"]) and np.isfinite(x["ci_high"])
             and x["ci_low"] > 0 for x in (f_, m_))
    if not ok:
        sex_ratio[code] = None
        print(f"  {code.upper():<4} female / male: not tested: an odds ratio was not tested "
              f"or not estimable in one of the two sexes")
        continue
    se_f = (np.log(f_["ci_high"]) - np.log(f_["ci_low"])) / (2 * Z_95)
    se_m = (np.log(m_["ci_high"]) - np.log(m_["ci_low"])) / (2 * Z_95)
    d = np.log(f_["odds_ratio"]) - np.log(m_["odds_ratio"])
    se = float(np.sqrt(se_f ** 2 + se_m ** 2))
    z = d / se
    p = float(2 * norm.sf(abs(z)))
    sex_ratio[code] = {"log_ratio": d, "se": se}
    print(f"  {code.upper():<4} female / male: ratio {np.exp(d):.3f} "
          f"({np.exp(d - Z_95 * se):.3f}-{np.exp(d + Z_95 * se):.3f})  z {z:.3f}  P {p:.4g}")


def bootstrap_specificity(g, rng):
    """Bootstrap of theta = ln(OR_f/OR_m)_DLB - ln(OR_f/OR_m)_VD over the
    patients of g (the analysis population), resampled within each sex.
    Resampling a sex's patients with replacement equals a multinomial draw
    over its eight AID x DLB x VD cells, which is what is drawn here (DLB and
    VD of the same patient stay together). NaN where an OR is 0 or infinite
    or an AID group is empty."""
    keys = [(a, d, v) for a in (1, 0) for d in (1, 0) for v in (1, 0)]
    aid = np.array([k[0] == 1 for k in keys])
    has = {"dlb": np.array([k[1] == 1 for k in keys]),
           "vd": np.array([k[2] == 1 for k in keys])}
    theta = np.zeros(SPECIFICITY_RESAMPLES)
    for s, sex_sign in ((1, 1.0), (0, -1.0)):
        gs = g[g["female"] == s]
        cells = gs.groupby(["aid_status", DIAGNOSIS_COLUMNS["dlb"], VD_COLUMN]).size()
        counts = np.array([cells.get(k, 0) for k in keys], dtype=float)
        draws = rng.multinomial(len(gs), counts / counts.sum(), size=SPECIFICITY_RESAMPLES)
        n1, n2 = draws[:, aid].sum(axis=1), draws[:, ~aid].sum(axis=1)
        for code, dx_sign in (("dlb", 1.0), ("vd", -1.0)):
            e1 = draws[:, aid & has[code]].sum(axis=1)
            e2 = draws[:, ~aid & has[code]].sum(axis=1)
            for b in range(SPECIFICITY_RESAMPLES):
                if np.isnan(theta[b]):
                    continue
                if min(n1[b], n2[b]) == 0:
                    theta[b] = NAN
                    continue
                theta[b] += sex_sign * dx_sign * np.log(conditional_or(e1[b], n1[b],
                                                                       e2[b], n2[b]))
    return theta


au.sub("Specificity of the sex difference: DLB vs VD, analysis population (log only)")
if sex_ratio["dlb"] is None or sex_ratio["vd"] is None:
    print("  theta = ln(ratio DLB) - ln(ratio VD): not tested: the female / male ratio "
          "of DLB or VD was not tested")
else:
    theta_obs = sex_ratio["dlb"]["log_ratio"] - sex_ratio["vd"]["log_ratio"]
    thetas = bootstrap_specificity(cohort[cohort_groups["with_notes"]],
                                   np.random.default_rng(SPECIFICITY_SEED))
    bad = int(np.isnan(thetas).sum())
    if bad > BOOTSTRAP_MAX_NOT_ESTIMABLE * SPECIFICITY_RESAMPLES:
        print(f"  theta = ln(ratio DLB) - ln(ratio VD): {theta_obs:.3f}  not tested: an "
              f"odds ratio of 0 or infinity in {bad} of {SPECIFICITY_RESAMPLES} bootstrap "
              f"resamples")
    else:
        t = thetas[~np.isnan(thetas)]
        k = len(t)
        lo, hi = np.percentile(t, [2.5, 97.5])
        p = min(1.0, 2 * (min((t <= 0).sum(), (t >= 0).sum()) + 1) / (k + 1))
        print(f"  theta = ln(ratio DLB) - ln(ratio VD): {theta_obs:.3f} ({lo:.3f} to "
              f"{hi:.3f})  exp(theta) {np.exp(theta_obs):.3f} "
              f"({np.exp(lo):.3f}-{np.exp(hi):.3f})  P {p:.4g}  "
              f"resamples not estimable {bad} of {SPECIFICITY_RESAMPLES}")

# =============================================================================
# SOURCE DATA
# =============================================================================
au.section("SOURCE DATA")
unmapped_counts = [(f"{v}_values_not_in_map",
                    int((cohort[v].notna() & ~cohort[v].isin(list(m))).sum()))
                   for v, _, m in CATEGORICAL]
au.write_workbook(os.path.join(SOURCE_DATA_DIR, TABLE_1_FILE), {
    "characteristics": table_1,
    "parameters": parameters(with_notes, False, unmapped_counts),
})
au.write_workbook(os.path.join(SOURCE_DATA_DIR, BY_SEX_FILE), {
    "characteristics": table_by_sex,
    "parameters": parameters(with_notes, True, unmapped_counts),
})
au.write_workbook(os.path.join(SOURCE_DATA_DIR, NO_NOTES_FILE), {
    "characteristics": no_notes,
    "share_without_notes": exclusion,
    "parameters": parameters(without_notes, False, unmapped_counts
                             + [("n_all_patients", len(cohort)),
                                ("n_with_notes", len(with_notes))]),
})
au.write_workbook(os.path.join(SOURCE_DATA_DIR, NO_NOTES_BY_SEX_FILE), {
    "characteristics": no_notes_by_sex,
    "share_without_notes": exclusion[exclusion["stratum"] != "all"].reset_index(drop=True),
    "parameters": parameters(without_notes, True, unmapped_counts
                             + [("n_all_patients", len(cohort)),
                                ("n_with_notes", len(with_notes))]),
})
dx_params = [("min_group_n", au.MIN_GROUP_N),
             ("bootstrap_resamples", BOOTSTRAP_RESAMPLES),
             ("bootstrap_seed", BOOTSTRAP_SEED),
             ("bootstrap_max_not_estimable", BOOTSTRAP_MAX_NOT_ESTIMABLE)]
for s, sx in ((1, "female"), (0, "male")):
    for ckey, _ in COHORTS:
        g = cohort[(cohort["female"] == s) & cohort_groups[ckey]]
        dx_params += [(f"n_aid_pos_{sx}_{ckey}", int((g["aid_status"] == 1).sum())),
                      (f"n_aid_neg_{sx}_{ckey}", int((g["aid_status"] == 0).sum()))]
dx_params.append(("n_both_dlb_and_pd", int(((cohort["coded_dementia_with_lewy_bodies"] == 1)
                                             & (cohort["coded_parkinsons_disease"] == 1)).sum())))
au.write_workbook(os.path.join(SOURCE_DATA_DIR, LEWY_BODY_FILE), {
    "diagnoses": diagnoses,
    "restriction_ratio": restriction,
    "parameters": pd.DataFrame(dx_params, columns=["parameter", "value"]),
})

print("\n" + au.SEP + "\n  COMPLETE\n" + au.SEP)