"""
inflammation_tier_sensitivity.py
================================
Does documented systemic inflammation compress the cognitive timeline in
patients with autoimmune disease (AID), so that symptoms of Alzheimer's
disease (AD) are documented more often, earlier, or in a different order?

Every table asks whether the AID+ vs AID- difference is modified by the
tier: AID+ vs AID- within patients
with an inflammation-tier mention and within patients without one, and the
AID x tier interaction. Prevalence and timing take all three from one model
per outcome with AID status, the tier and their interaction (Model C); the
sequence table from exact 2x2 tests (see Sequence below). The two design
contrasts,
  (1) AID+ with a tier mention vs AID+ without one;
  (2) AID+ with a mention vs AID- without one,
leave out AID- patients with the tier; they are written to the checks
workbook from the same Model C, with Models A (AID only) and B (+ tier).

Computes every statistic and writes the source data for the inflammation
tier tables (one workbook per table) and a checks workbook. The tables are
rendered by inflammation_tier_sensitivity_tables.py; nothing here formats a number.

Tiers
-----
In the tier records, each record of the systemic inflammation autoimmune,
arthralgia and myalgia categories carries the lab-result and flare tiers; the
inflammation-record tier is any record of the systemic inflammation
autoimmune category. A patient has a tier when at least one dated record
carries it. Records on undated notes are not used.
- Presence analyses (frequency, prevalence): tier and symptom records at or
  below the patient's age ceiling.
- Timing and order: a patient counts as exposed when a tier mention is dated
  on or before the first documentation of the domain (or of the earlier of
  the two categories of a pair): unlike the other date comparisons, a mention
  on the same day counts, because co-occurrence is enough to show exposure.
  No age ceiling, as for the outcomes.

Analyses
--------
- Tier patients and overlap (checks).
- Frequency: is each tier documented more often in AID+ than AID-, within
  each sex? Fisher's exact test, conditional MLE odds ratio and exact CI; gate
  two_by_two_reason; BH over the six tier x sex rows.
- Prevalence: logit has_x ~ aid_status + female + age_at_dx + tier +
  aid_status:tier, weighted by ps_weight_trimmed (var_weights, HC0), per
  domain and category. Outcomes built from the tier records (systemic
  inflammation autoimmune and its categories) are not analyzed. "Other" is
  fitted for information only (no P or Q).
- Timing: OLS HC3, first_age ~ aid_status + female + age_at_entry_c +
  predx_span_c + tier + aid_status:tier, among patients with the domain
  documented; and the AD-core onset (analysis_utils.ad_core_onset; the
  interaction a single test, checks only). age_at_entry_c is age at the
  first dated note (a symptom cannot be documented before the record begins) and
  predx_span_c the pre-diagnosis documentation span, both centered within
  each sample; together they also hold age at AD diagnosis fixed.
- Sequence: is the AID+ vs AID- difference in an inverted order of first
  documentation modified by the tier? Per expected-order pair
  (analysis_utils.SEQUENCE_PAIRS), among patients with both categories
  documented and a pre-diagnosis documentation span of at least 5 years:
  AID+ vs AID- within patients with the tier and within patients without it
  (Fisher's exact test, conditional MLE odds ratio, exact 95% CI), the ratio
  of the two odds ratios and the exact conditional test of the AID x tier
  interaction (analysis_utils.effect_modification_row); unadjusted. Tested
  when both strata pass two_by_two_reason. Same-day first documentation
  (tie) counts as not inverted; tie counts and a ties-excluded check are
  written, as in symptom_timing_sequence.py.
- Cognitive scores: is the AID+ vs AID- difference in the annual change of
  MMSE and MoCA scores modified by the tier? The annual change is the one of
  cognitive_score_comparison.py (analysis_utils.absolute_cognitive_scores
  and analysis_utils.annual_change): per patient, (last - first score) /
  years between them, two or more scores at least 0.25 years apart. OLS
  HC3, slope ~ aid_status + age_c + female + tier + aid_status:tier (age_c =
  age at the first score, centered): the medication models of
  cognitive_score_comparison.py with the tier in place of the class.
  Exposed: a tier mention dated on or before the patient's last score (a
  mention on the same day counts, as for timing); undated mentions do not
  count. No age ceiling.
Estimates (prevalence, timing, cognitive scores), from Model C: AID+ vs AID-
without the tier (the AID term), with the tier (AID term + interaction), and
the interaction (the ratio of the two odds ratios; for timing the difference
of the two differences, in years; for cognitive scores in points per year).
Each is a Wald linear combination of Model C's coefficients (covariance from
the fit); only the interaction is tested. The design contrasts (checks) are
Wald combinations of the same fit (prevalence, timing).
Gate (prevalence, timing, cognitive scores): analysis_utils.model_floor_reason on Model C (10 per term, rarer class
for 0/1 outcomes; both AID groups and both sexes; 10 in the rarer outcome
class, or 10 patients for timing and cognitive scores, in each AID x tier
group); Models A and B are fitted only with C. FDR: the interaction, BH
within each tier x level x definition (prevalence), tier x definition
(timing) or tier (cognitive scores, over MMSE and MoCA); tier over the
tested pairs (sequence); the checks keep their own families (each design
contrast, and each tested term of Models A, B and C, per tier x level x
definition for prevalence and per tier x definition for timing; the Model C
interaction per tier for cognitive scores).
Linearity: age at AD diagnosis squared (prevalence; OR shift >= 20%
with |OR - 1| >= 0.05), the squares of both centered timing covariates in
one refit (timing; shift >= 0.5 years) or the centered age at the first
score squared (cognitive scores; shift >= 0.5 points per year) added to
Models A and C; compared: the AID term of A and the three estimates of C.
Secondary domain definition: every domain-level result refitted under
it; the change in the interaction is printed and written.

Inputs: the patient table, the note table, the symptom records, the
inflammation-tier records, the propensity-score weights and the cognitive
score table (see PATHS).
"""

import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import fisher_exact, norm, t as t_dist
from scipy.stats.contingency import odds_ratio
from statsmodels.genmod.families import Binomial

import analysis_utils as au

warnings.filterwarnings("ignore")   # after the imports: statsmodels resets its own filters

# =============================================================================
# PATHS (relative to the data folder the script is run from)
# =============================================================================
COHORT_PATIENTS_PATH = ""  # patient table: AID status, sex, key dates (birth, AD diagnosis, death) and analysis-population flag
COHORT_NOTES_PATH = ""  # note table: one row per clinical note, with its date and specialty
SYMPTOM_RECORDS_PATH = ""  # symptom records: one row per documented symptom, with its note and symptom category
TIER_RECORDS_PATH = ""  # inflammation-tier records: one row per record of the three tier categories, with its note and tiers
PS_WEIGHTS_PATH = ""  # propensity-score weights, one per analysis patient
COGNITIVE_SCORES_PATH = ""  # cognitive score table: one row per documented score, with its type, value, date and confidence
SOURCE_DATA_DIR = ""  # folder for the source-data workbooks written here

FREQUENCY_FILE = "table_inflammation_tier_frequency.xlsx"
PREVALENCE_FILE = "table_inflammation_tier_prevalence.xlsx"
TIMING_FILE = "table_inflammation_tier_timing.xlsx"
SEQUENCE_FILE = "table_inflammation_tier_sequence.xlsx"
COGNITIVE_FILE = "table_inflammation_tier_cognitive_scores.xlsx"
CHECKS_FILE = "checks_inflammation_tiers.xlsx"

# =============================================================================
# SETTINGS
# =============================================================================
Q_THRESHOLD = 0.05                 # secondary-definition check: Q crossing
OR_CHANGE_SMALL = 0.10             # secondary-definition check: |OR change|
OR_CHANGE_LARGE = 0.20
ESTIMATE_CHANGE_SMALL_YRS = 0.25   # secondary-definition check: |change|, timing
ESTIMATE_CHANGE_LARGE_YRS = 0.5
WCOL = "ps_weight_trimmed"
DEFINITIONS = ["primary", "secondary"]
TIER_NAMES = [x for _, x, _ in au.TIERS]
TIER_LABELS = {x: lbl for _, x, lbl in au.TIERS}
# Outcomes built from the tier records: not analyzed against the tiers.
CIRCULAR_DOMAINS = {"systemic_inflammation_autoimmune"}
CIRCULAR_CATEGORIES = set(au.TIER_CATEGORIES)
SEQUENCE_FLOOR_YRS = au.PREDX_SPAN_FLOOR_5_YRS
GROUPS = [(1, 1, "AID+ female"), (1, 0, "AID+ male"), (0, 1, "AID- female"), (0, 0, "AID- male")]
CELLS = [(1, 1, "aid_pos_exposed"), (1, 0, "aid_pos_unexposed"),
         (0, 1, "aid_neg_exposed"), (0, 0, "aid_neg_unexposed")]
# The estimates of the prevalence and timing tables, from Model C; only the
# interaction is tested.
ESTIMATES = [("aid_pos_vs_aid_neg_with_tier", "AID+ vs AID- with the tier"),
             ("aid_pos_vs_aid_neg_without_tier", "AID+ vs AID- without the tier"),
             ("interaction", "AID x tier interaction")]
INTERACTION = "interaction"
TESTED = [e for e in ESTIMATES if e[0] == INTERACTION]
# The two design contrasts, from the same Model C (checks).
CONTRASTS = [("aid_pos_tier_vs_aid_pos_no_tier", "AID+ with the tier vs AID+ without"),
             ("aid_pos_tier_vs_aid_neg_no_tier", "AID+ with the tier vs AID- without")]
NAN = float("nan")


def n_terms(formula):
    return len([t for t in formula.split("~", 1)[1].split("+") if t.strip()])


def empty_fit():
    return {"coef": NAN, "se": NAN, "crit": NAN, "ci_low": NAN, "ci_high": NAN, "p": NAN}


def crit_of(mod):
    """Critical value of the 95% CI: t when the fit uses t, z otherwise."""
    return float(t_dist.ppf(0.975, mod.df_resid) if getattr(mod, "use_t", False)
                 else norm.ppf(0.975))


def term_values(mod, term):
    if term not in mod.params.index:
        return empty_fit()
    ci = mod.conf_int()
    return {"coef": float(mod.params[term]), "se": float(mod.bse[term]), "crit": crit_of(mod),
            "ci_low": float(ci.loc[term, 0]), "ci_high": float(ci.loc[term, 1]),
            "p": float(mod.pvalues[term])}


def linear_combination(mod, terms):
    """Wald estimate of the sum of `terms` (covariance from the fit)."""
    b, v = mod.params, mod.cov_params()
    coef = float(sum(b[t] for t in terms))
    var = float(sum(v.loc[a, c] for a in terms for c in terms))
    se = float(np.sqrt(var))
    crit = crit_of(mod)
    if getattr(mod, "use_t", False):
        p = float(2 * t_dist.sf(abs(coef / se), mod.df_resid))
    else:
        p = float(2 * norm.sf(abs(coef / se)))
    return {"coef": coef, "se": se, "crit": crit, "ci_low": coef - crit * se,
            "ci_high": coef + crit * se, "p": p}


def estimate_terms(x):
    """Model C terms whose sum is each estimate of the tables."""
    return {"aid_pos_vs_aid_neg_with_tier": ["aid_status", f"aid_status:{x}"],
            "aid_pos_vs_aid_neg_without_tier": ["aid_status"],
            "interaction": [f"aid_status:{x}"]}


def contrast_terms(x):
    """Model C terms whose sum is each design contrast."""
    return {"aid_pos_tier_vs_aid_pos_no_tier": [x, f"aid_status:{x}"],
            "aid_pos_tier_vs_aid_neg_no_tier": ["aid_status", x, f"aid_status:{x}"]}


def cell_counts(d, y, x, binary):
    out = {}
    for a, e, lbl in CELLS:
        c = d[(d["aid_status"] == a) & (d[x] == e)]
        if binary:
            out["events_" + lbl] = int(c[y].sum())
        out["n_" + lbl] = len(c)
    out["n_aid_pos"] = int((d["aid_status"] == 1).sum())
    out["n_aid_neg"] = int((d["aid_status"] == 0).sum())
    return out


def bh_by(df, family_cols, eligible):
    """BH q within each family over the eligible rows."""
    df = df.copy()
    df["q"] = NAN
    df["fdr_family_size"] = 0
    if not len(df):
        return df
    for _, idx in df.groupby(family_cols).groups.items():
        m = df.loc[idx]
        t = m.index[eligible.loc[idx] & m["p"].notna()]
        if len(t):
            df.loc[t, "q"] = au.bh_fdr(df.loc[t, "p"].values)
            df.loc[t, "fdr_family_size"] = len(t)
    return df


def exp_ratio_columns(df):
    """Odds ratio and CI from coef, CI on the log scale (logistic rows)."""
    df = df.copy()
    df["odds_ratio"] = np.exp(df["coef"])
    df["ci_low"] = np.exp(df["ci_low"])
    df["ci_high"] = np.exp(df["ci_high"])
    return df


def check_ci(df, label, ratio):
    f = df[(df["status"] == "fitted") & df["coef"].notna()]
    lo, hi = f["coef"] - f["crit"] * f["se"], f["coef"] + f["crit"] * f["se"]
    if ratio:
        lo, hi = np.exp(lo), np.exp(hi)
    if not (np.allclose(f["ci_low"], lo, rtol=1e-9, atol=1e-9)
            and np.allclose(f["ci_high"], hi, rtol=1e-9, atol=1e-9)):
        raise RuntimeError(f"{label}: CI does not equal coef -/+ crit x SE")


def fmt_console(v, spec):
    return "n/a" if pd.isna(v) else format(v, spec)


# =============================================================================
# MODEL SETS (A, B, C), ESTIMATES AND CONTRASTS
# =============================================================================
def fit_one(kind, formula, d):
    if kind == "ols":
        return smf.ols(formula, data=d).fit(cov_type="HC3")
    if kind == "logit_weighted":
        return smf.glm(formula, data=d, family=Binomial(),
                       var_weights=d[WCOL]).fit(cov_type="HC0", disp=False)
    mod = smf.logit(formula, data=d).fit(disp=0)
    if not mod.mle_retvals.get("converged", True):
        raise RuntimeError("did not converge")
    return mod


def model_set(kind, y, base_rhs, d, x, why):
    """Models A (base), B (+ tier), C (+ tier + AID x tier) when the gate
    passed (why == ""). A fit error in any model, or a term not estimated,
    marks the whole set failed. Returns (status, reason, {model: fit})."""
    if why:
        return "not_fitted", why, {}
    forms = {"A": f"{y} ~ {base_rhs}", "B": f"{y} ~ {base_rhs} + {x}",
             "C": f"{y} ~ {base_rhs} + {x} + aid_status:{x}"}
    fits = {}
    try:
        for m, f in forms.items():
            fits[m] = fit_one(kind, f, d)
    except Exception as e:
        reason = str(e) if str(e) == "did not converge" else f"fit error: {type(e).__name__}"
        return "failed", reason, {}
    needed = {"A": ["aid_status"], "B": ["aid_status", x], "C": ["aid_status", x, f"aid_status:{x}"]}
    for m, terms in needed.items():
        if any(t not in fits[m].params.index or not np.isfinite(fits[m].bse[t]) for t in terms):
            return "failed", "term not estimated", {}
    return "fitted", "", fits


# (model, term name in the fit, term role)
MODEL_TERMS = [("A", "aid_status", "AID"), ("B", "aid_status", "AID"),
               ("B", "{x}", "exposure"), ("C", "aid_status:{x}", "interaction")]


def model_rows(key, status, reason, fits, x, n_model, counts):
    rows = []
    for m, t, role in MODEL_TERMS:
        t = t.format(x=x)
        vals = term_values(fits[m], t) if status == "fitted" else empty_fit()
        rows.append({**key, "model": m, "term": role, "n_model": n_model, "status": status,
                     "reason": reason, **counts, **vals})
    return rows


def combination_rows(key, status, reason, fits, n_model, counts, items, terms, col):
    """One row per Wald combination of Model C terms: items [(code, label)],
    terms {code: [term, ...]}; the code goes to column `col`, the label to
    `col`_label."""
    rows = []
    for code, label in items:
        vals = (linear_combination(fits["C"], terms[code]) if status == "fitted"
                else empty_fit())
        rows.append({**key, col: code, f"{col}_label": label, "n_model": n_model,
                     "status": status, "reason": reason, **counts, **vals})
    return rows


def linearity_rows(key, kind, y, base_rhs, d, x, fits, sq, ratio, sex_col="female",
                   shift_threshold=au.LINEARITY_SHIFT_YEARS):
    """Models A and C refitted with `sq` added (same rows); compares the AID
    term of A and the three estimates of C. `sq` is one squared column
    or a list of them (added together in one refit; each coefficient written
    as coef_<column>). The quadratic model must meet the events-per-term floor
    with the added terms counted. A linear model is flagged at a shift of
    shift_threshold (years by default; points per year for cognitive scores)."""
    sqs = [sq] if isinstance(sq, str) else list(sq)
    out = []
    targets = [("A", "AID", ["aid_status"])] \
        + [("C", code, terms) for code, terms in estimate_terms(x).items()]
    forms = {"A": f"{y} ~ {base_rhs}", "C": f"{y} ~ {base_rhs} + {x} + aid_status:{x}"}
    quad = {}
    for m in ("A", "C"):
        fq = forms[m] + " + " + " + ".join(sqs)
        why = au.model_floor_reason(d[y], d["aid_status"], n_terms(fq), sex=d[sex_col],
                                    continuous=not ratio)
        if why:
            quad[m] = (None, why)
            continue
        try:
            quad[m] = (fit_one(kind, fq, d), "")
        except Exception as e:
            quad[m] = (None, f"fit error: {type(e).__name__}")
    for m, target, terms in targets:
        row = {**key, "model": m, "term": target, "status": "not_run", "reason": "",
               "estimate_linear": NAN, "estimate_quadratic": NAN, "shift": NAN,
               "shift_pct": NAN, "coef_squared": NAN, "flag": 0}
        mq, why = quad[m]
        if mq is None:
            row["reason"] = why
            out.append(row)
            continue
        if any(t not in mq.params.index for t in terms):
            row["reason"] = "term not estimated"
            out.append(row)
            continue
        bl = sum(float(fits[m].params[t]) for t in terms)
        bq = sum(float(mq.params[t]) for t in terms)
        el, eq = (np.exp(bl), np.exp(bq)) if ratio else (bl, bq)
        if ratio:
            shift_pct = abs(eq - el) / abs(el) * 100
            flag = int(shift_pct >= au.LINEARITY_OR_SHIFT * 100
                       and abs(el - 1) >= au.LINEARITY_OR_MIN_DIFF)
            shift = abs(eq - el)
        else:
            shift = abs(eq - el)
            shift_pct = shift / abs(el) * 100 if el != 0 else NAN
            flag = int(shift >= shift_threshold)
        row.update(status="run", estimate_linear=el, estimate_quadratic=eq, shift=shift,
                   shift_pct=shift_pct, flag=flag,
                   coef_squared=float(mq.params.get(sqs[0], NAN)) if len(sqs) == 1 else NAN)
        if len(sqs) > 1:
            row.update({f"coef_{s}": float(mq.params.get(s, NAN)) for s in sqs})
        out.append(row)
    return out


LIN_VALUE_COLUMNS = ["model", "term", "status", "reason", "estimate_linear",
                     "estimate_quadratic", "shift", "shift_pct", "coef_squared", "flag"]


def linearity_frame(rows):
    """Linearity rows with fixed columns (also when no model was fitted)."""
    df = pd.DataFrame(rows)
    for c in LIN_VALUE_COLUMNS:
        if c not in df.columns:
            df[c] = NAN
    df["flag"] = df["flag"].fillna(0).astype(int)
    return df


def print_tested(df, key_col, title, ratio, col, items):
    """Per tier and tested item (code, label): rows fitted and rows with
    Q < 0.05; then the outcomes not fitted."""
    print(f"\n  {title}")
    for x in TIER_NAMES:
        for code, label in items:
            m = df[(df["tier"] == x) & (df[col] == code)]
            f = m[m["status"] == "fitted"]
            sig = f[f["q"] < Q_THRESHOLD]
            est = (lambda r: f"OR {r['odds_ratio']:.2f}") if ratio else \
                (lambda r: f"{r['coef']:+.2f} yr")
            print(f"  {x:<20} {label:<36} fitted {len(f):>3} of {len(m):>3}; Q < 0.05: "
                  + (", ".join(f"{r[key_col]} ({est(r)}, Q={r['q']:.4f})"
                               for _, r in sig.iterrows()) or "none"))
        nt = df[(df["tier"] == x) & (df[col] == items[0][0]) & (df["status"] != "fitted")]
        for _, r in nt.iterrows():
            print(f"      not fitted: {r[key_col]:<34} {r['status']}: {r['reason']}")


# =============================================================================
# SECTION 1 -- POPULATION, AGE CEILING, TIERS PER PATIENT
# =============================================================================
au.section("SECTION 1 -- POPULATION, AGE CEILING, TIERS PER PATIENT")
pop, notes = au.load_population(COHORT_PATIENTS_PATH, COHORT_NOTES_PATH)
pop = au.load_ps_weights(PS_WEIGHTS_PATH, pop)
pop["age_at_dx_sq"] = pop["age_at_dx"] ** 2

ceil = au.p90_age_ceiling(pop)
ceiling_by_person = ceil["ceiling_by_person"]
print(f"\n  Age ceiling, female (yr): {ceil['ceiling_female']:.2f}  "
      f"(AID- reference n = {ceil['n_ref_female']:,})")
print(f"  Age ceiling, male (yr)  : {ceil['ceiling_male']:.2f}  "
      f"(AID- reference n = {ceil['n_ref_male']:,})")
past_all = ~au.within_age_ceiling(pop["age_at_entry"], pop["person_id"].map(ceiling_by_person))
print("  Patients whose first dated note is past their ceiling (no record counts in the "
      "presence analyses):")
for a, s, lbl in GROUPS:
    g = (pop["aid_status"] == a) & (pop["female"] == s)
    au.count_line(lbl, (past_all & g).sum(), g.sum())

rec = au.load_symptom_records(SYMPTOM_RECORDS_PATH, pop, notes)
dated = rec[rec["note_date"].notna()].merge(pop[["person_id", "birth_date"]], on="person_id",
                                            how="left", validate="many_to_one")
dated["age_at_record"] = au.years_between(dated["note_date"], dated["birth_date"])
in_ceiling = dated[au.within_age_ceiling(dated["age_at_record"],
                                         dated["person_id"].map(ceiling_by_person))]
au.count_line("dated symptom records within the age ceiling (presence analyses)",
              len(in_ceiling), len(dated))

tiers = au.load_tier_records(TIER_RECORDS_PATH, pop, notes)
tiers = tiers[tiers["note_date"].notna()].merge(pop[["person_id", "birth_date"]],
                                                on="person_id", how="left",
                                                validate="many_to_one")
tiers["age_at_record"] = au.years_between(tiers["note_date"], tiers["birth_date"])
tier_presence = tiers[au.within_age_ceiling(tiers["age_at_record"],
                                            tiers["person_id"].map(ceiling_by_person))]
au.count_line("dated tier records past the age ceiling (not in the presence analyses)",
              len(tiers) - len(tier_presence), len(tiers))
for col, x, _ in au.TIERS:
    pop[x] = pop["person_id"].isin(tier_presence.loc[tier_presence[col] == 1, "person_id"]) \
        .astype(int)
# earliest dated mention of each tier, any age (timing and order)
TIER_FIRST = {x: tiers[tiers[col] == 1].groupby("person_id")["note_date"].min().to_dict()
              for col, x, _ in au.TIERS}


def tier_first_date(person_ids, x):
    """Earliest dated mention of tier x per patient (NaT when none)."""
    return pd.to_datetime(person_ids.map(TIER_FIRST[x]))


print("\n  Patients with each tier (a mention within the ceiling):")
tier_patient_rows = []
for a, s, lbl in GROUPS + [(1, None, "AID+"), (0, None, "AID-")]:
    g = pop[(pop["aid_status"] == a) & ((pop["female"] == s) if s is not None else True)]
    tier_patient_rows.append({"group": lbl, "n": len(g),
                              **{f"n_{x}": int(g[x].sum()) for x in TIER_NAMES}})
for x in TIER_NAMES:
    print(f"    {TIER_LABELS[x]:<20} {int(pop[x].sum()):>6,} of {len(pop):,}")

au.sub("Tier overlap (patients with both tiers)")
overlap_rows = []
print("  " + " " * 20 + "".join(f"{TIER_LABELS[x]:>22}" for x in TIER_NAMES))
for xa in TIER_NAMES:
    vals = [int(((pop[xa] == 1) & (pop[xb] == 1)).sum()) for xb in TIER_NAMES]
    print(f"  {TIER_LABELS[xa]:<20}" + "".join(f"{v:>22,}" for v in vals))
    overlap_rows += [{"tier_a": xa, "tier_b": xb, "n_patients": v}
                     for xb, v in zip(TIER_NAMES, vals)]

# =============================================================================
# SECTION 2 -- TIER FREQUENCY, AID+ VS AID-, WITHIN SEX
# =============================================================================
au.section("SECTION 2 -- TIER FREQUENCY, AID+ VS AID-, WITHIN SEX")
print("  Fisher's exact test; conditional MLE odds ratio, exact 95% CI.")


def exact_2x2(a, b, c, d):
    table = [[int(a), int(b)], [int(c), int(d)]]
    res = odds_ratio(table, kind="conditional")
    ci = res.confidence_interval(0.95)
    return float(res.statistic), float(ci.low), float(ci.high), float(fisher_exact(table)[1])


freq_rows = []
for x in TIER_NAMES:
    for sx, sel in (("Female", 1), ("Male", 0)):
        ap = pop[(pop["aid_status"] == 1) & (pop["female"] == sel)]
        an = pop[(pop["aid_status"] == 0) & (pop["female"] == sel)]
        a, c = int(ap[x].sum()), int(an[x].sum())
        r = {"tier": x, "row_label": TIER_LABELS[x], "sex": sx,
             "with_tier_aid_pos": a, "n_aid_pos": len(ap),
             "with_tier_aid_neg": c, "n_aid_neg": len(an),
             "odds_ratio": NAN, "ci_low": NAN, "ci_high": NAN, "or_estimable": 0, "p": NAN}
        why = au.two_by_two_reason(a, len(ap), c, len(an),
                                   f"a mention of the {TIER_LABELS[x].lower()} tier")
        if why:
            r.update(status="not_tested", reason=why)
        else:
            orv, lo, hi, p = exact_2x2(a, len(ap) - a, c, len(an) - c)
            r.update(status="tested", reason="", p=p)
            if np.isfinite(orv) and orv > 0:
                r.update(odds_ratio=orv, ci_low=lo, ci_high=hi, or_estimable=1)
        freq_rows.append(r)
freq_df = pd.DataFrame(freq_rows)
freq_df = bh_by(freq_df.assign(_f=0), ["_f"], freq_df["status"] == "tested").drop(columns="_f")
for _, r in freq_df.iterrows():
    res = (f"OR {fmt_console(r['odds_ratio'], '.2f')} [{fmt_console(r['ci_low'], '.2f')}-"
           f"{fmt_console(r['ci_high'], '.2f')}]  P={r['p']:.4f}  Q={r['q']:.4f}"
           if r["status"] == "tested" else f"not tested: {r['reason']}")
    print(f"  {r['tier']:<20} {r['sex']:<7} AID+ {r['with_tier_aid_pos']}/{r['n_aid_pos']}  "
          f"AID- {r['with_tier_aid_neg']}/{r['n_aid_neg']}  {res}")

# =============================================================================
# SECTION 3 -- PREVALENCE
# =============================================================================
au.section("SECTION 3 -- TIER AND SYMPTOM PREVALENCE")
print("  Weighted logit (binomial GLM, HC0): has_x ~ aid_status + female + age_at_dx")
print("  [+ tier [+ aid_status:tier]]; symptom and tier records within the age ceiling.")
c2d = {d: au.category_domain(d) for d in DEFINITIONS}
cat_ids = {c: set(in_ceiling.loc[in_ceiling["category"] == c, "person_id"])
           for c in au.SCHEMA_CATEGORIES if c not in CIRCULAR_CATEGORIES}
dom_ids = {defn: {dom: set(in_ceiling.loc[in_ceiling["category"].map(c2d[defn]) == dom,
                                          "person_id"])
                  for dom in sorted(au.DOMAIN_DEFINITIONS[defn]) if dom not in CIRCULAR_DOMAINS}
           for defn in DEFINITIONS}
PREV_BASE = "aid_status + female + age_at_dx"

prev_model_rows, prev_est_rows, prev_contrast_rows, lin_prev = [], [], [], []
for x in TIER_NAMES:
    outcomes = [("category", "primary", c, ids) for c, ids in sorted(cat_ids.items())] \
        + [("domain", defn, dom, ids) for defn in DEFINITIONS
           for dom, ids in dom_ids[defn].items()]
    for level, defn, code, ids in outcomes:
        d = pop[["person_id", "aid_status", "female", "age_at_dx", "age_at_dx_sq", WCOL, x]].copy()
        d["has_x"] = d["person_id"].isin(ids).astype(int)
        fC = f"has_x ~ {PREV_BASE} + {x} + aid_status:{x}"
        why = au.model_floor_reason(d["has_x"], d["aid_status"], n_terms(fC), sex=d["female"],
                                    modifier=d[x])
        status, reason, fits = model_set("logit_weighted", "has_x", PREV_BASE, d, x, why)
        if level == "category":
            label, dom = au.category_row_label(code), c2d["primary"][code]
            role = "not_tested" if code in au.NOT_TESTED_CATEGORIES else "tested"
        else:
            label, dom, role = au.DOMAIN_LABELS[code], code, "tested"
        key = {"tier": x, "level": level, "definition": defn, "code": code, "domain": dom,
               "row_label": label, "fdr_role": role}
        counts = cell_counts(d, "has_x", x, True)
        prev_model_rows += model_rows(key, status, reason, fits, x, len(d), counts)
        prev_est_rows += combination_rows(key, status, reason, fits, len(d), counts, ESTIMATES,
                                          estimate_terms(x), "estimate")
        prev_contrast_rows += combination_rows(key, status, reason, fits, len(d), counts,
                                               CONTRASTS, contrast_terms(x), "contrast")
        if defn == "primary":
            if status == "fitted":
                lin_prev += linearity_rows(key, "logit_weighted", "has_x", PREV_BASE, d, x, fits,
                                           "age_at_dx_sq", ratio=True)
            else:
                lin_prev.append({**key, "model": "", "term": "", "status": "not_run",
                                 "reason": f"models not fitted: {reason}"})

prev_models = pd.DataFrame(prev_model_rows)
prev_models.loc[prev_models["fdr_role"] == "not_tested", "p"] = NAN
elig = ((prev_models["status"] == "fitted") & (prev_models["fdr_role"] == "tested")
        & (((prev_models["model"] == "A") & (prev_models["term"] == "AID"))
           | ((prev_models["model"] == "B") & (prev_models["term"] == "exposure"))
           | ((prev_models["model"] == "C") & (prev_models["term"] == "interaction"))))
prev_models = exp_ratio_columns(bh_by(prev_models, ["tier", "level", "definition", "model",
                                                    "term"], elig))
check_ci(prev_models, "prevalence models", True)
prev = pd.DataFrame(prev_est_rows)
prev.loc[prev["fdr_role"] == "not_tested", "p"] = NAN
prev = exp_ratio_columns(bh_by(prev, ["tier", "level", "definition", "estimate"],
                               (prev["status"] == "fitted") & (prev["fdr_role"] == "tested")
                               & (prev["estimate"] == INTERACTION)))
check_ci(prev, "prevalence estimates", True)
prev_con = pd.DataFrame(prev_contrast_rows)
prev_con.loc[prev_con["fdr_role"] == "not_tested", "p"] = NAN
prev_con = exp_ratio_columns(bh_by(prev_con, ["tier", "level", "definition", "contrast"],
                                   (prev_con["status"] == "fitted")
                                   & (prev_con["fdr_role"] == "tested")))
check_ci(prev_con, "prevalence contrasts", True)
lin_prev = linearity_frame(lin_prev)
print_tested(prev[(prev["definition"] == "primary") & (prev["level"] == "domain")], "code",
             "Domains (primary definition)", True, "estimate", TESTED)
print_tested(prev[prev["level"] == "category"], "code", "Categories", True, "estimate", TESTED)
print_tested(prev_con[(prev_con["definition"] == "primary") & (prev_con["level"] == "domain")],
             "code", "Design contrasts, domains (primary definition; checks)", True, "contrast",
             CONTRASTS)
run = lin_prev[lin_prev["status"] == "run"]
print(f"\n  Linearity (age at AD diagnosis squared): {len(run)} of {len(lin_prev)} run; flagged "
      f"(OR shift >= 20% and |OR - 1| >= 0.05): {int(run['flag'].sum())}"
      + (f"; largest shift {run['shift_pct'].max():.1f}%" if len(run) else ""))

# =============================================================================
# SECTION 4 -- TIMING (AGE AT FIRST DOCUMENTATION) AND AD-CORE ONSET
# =============================================================================
au.section("SECTION 4 -- TIER AND AGE AT FIRST DOCUMENTATION")
print("  OLS, HC3: first_age ~ aid_status + female + age_at_entry_c + predx_span_c [+ tier")
print("  [+ aid_status:tier]], among patients with the domain documented; age_at_entry_c =")
print("  age at the first dated note, predx_span_c = pre-diagnosis documentation span, both")
print("  centered within each sample. Exposed: a tier mention dated on or before the")
print("  domain's first documentation.")
POP_COLUMNS = ["person_id", "aid_status", "female", "age_at_entry",
               "predx_documentation_span_yrs"]
# adjustment covariates of the timing models: (centered column, source column)
TIME_COVARIATES = [("age_at_entry_c", "age_at_entry"),
                   ("predx_span_c", "predx_documentation_span_yrs")]
TIME_BASE = "aid_status + female + " + " + ".join(c for c, _ in TIME_COVARIATES)
TIME_SQUARES = [f"{c}_sq" for c, _ in TIME_COVARIATES]


def exposure_on_or_before(person_ids, dates, x):
    # exposure: a tier mention dated on or before the first documentation (cognitive
    # scores: on or before the last score)
    return (tier_first_date(person_ids, x) <= dates).astype(int)


def timing_sample(frame):
    d = frame.merge(pop[POP_COLUMNS], on="person_id", how="left", validate="many_to_one")
    for name, source in TIME_COVARIATES:
        d[name] = d[source] - d[source].mean()
        d[name + "_sq"] = d[name] ** 2
    return d


first_dom = {defn: au.first_documentation(rec, pop, "domain", defn) for defn in DEFINITIONS}
core = au.ad_core_onset(first_dom["primary"])
au.count_line("patients with an AD-core onset", len(core), len(pop))
time_model_rows, time_est_rows, time_contrast_rows, lin_time = [], [], [], []
for x in TIER_NAMES:
    targets = [("domain", defn, code, first_dom[defn][first_dom[defn]["code"] == code])
               for defn in DEFINITIONS for code in sorted(au.DOMAIN_DEFINITIONS[defn])
               if code not in CIRCULAR_DOMAINS] \
        + [("ad_core_onset", "primary", "combined", core)]
    for analysis, defn, code, frame in targets:
        d = timing_sample(frame[["person_id", "first_date", "first_age"]])
        d[x] = exposure_on_or_before(d["person_id"], d["first_date"], x)
        fC = f"first_age ~ {TIME_BASE} + {x} + aid_status:{x}"
        why = au.model_floor_reason(d["first_age"], d["aid_status"], n_terms(fC),
                                    sex=d["female"], continuous=True, modifier=d[x])
        status, reason, fits = model_set("ols", "first_age", TIME_BASE, d, x, why)
        key = {"tier": x, "analysis": analysis, "definition": defn, "code": code,
               "row_label": au.DOMAIN_LABELS.get(code, "AD core (combined)"),
               "fdr_role": "tested"}
        counts = cell_counts(d, "first_age", x, False)
        time_model_rows += model_rows(key, status, reason, fits, x, len(d), counts)
        time_est_rows += combination_rows(key, status, reason, fits, len(d), counts, ESTIMATES,
                                          estimate_terms(x), "estimate")
        time_contrast_rows += combination_rows(key, status, reason, fits, len(d), counts,
                                               CONTRASTS, contrast_terms(x), "contrast")
        if defn == "primary":
            if status == "fitted":
                lin_time += linearity_rows(key, "ols", "first_age", TIME_BASE, d, x, fits,
                                           TIME_SQUARES, ratio=False)
            else:
                lin_time.append({**key, "model": "", "term": "", "status": "not_run",
                                 "reason": f"models not fitted: {reason}"})

time_models = pd.DataFrame(time_model_rows)
time_models = bh_by(time_models, ["tier", "analysis", "definition", "model", "term"],
                    (time_models["status"] == "fitted") & (time_models["model"] == "C"))
check_ci(time_models, "timing models", False)
timing = pd.DataFrame(time_est_rows)
timing = bh_by(timing, ["tier", "analysis", "definition", "estimate"],
               (timing["status"] == "fitted") & (timing["estimate"] == INTERACTION))
check_ci(timing, "timing estimates", False)
timing_con = pd.DataFrame(time_contrast_rows)
timing_con = bh_by(timing_con, ["tier", "analysis", "definition", "contrast"],
                   timing_con["status"] == "fitted")
check_ci(timing_con, "timing contrasts", False)
lin_time = linearity_frame(lin_time)
print_tested(timing[(timing["analysis"] == "domain") & (timing["definition"] == "primary")],
             "code", "Domains (primary definition)", False, "estimate", TESTED)
print_tested(timing_con[(timing_con["analysis"] == "domain")
                        & (timing_con["definition"] == "primary")],
             "code", "Design contrasts, domains (primary definition; checks)", False, "contrast",
             CONTRASTS)
au.sub("AD-core onset (combined target; the interaction a single test, Q = P; checks)")
for _, r in timing[timing["analysis"] == "ad_core_onset"].iterrows():
    print(f"  {r['tier']:<20} {r['estimate_label']:<36} "
          + (f"{r['coef']:+.2f} yr [{r['ci_low']:+.2f}, {r['ci_high']:+.2f}] P={r['p']:.4f}"
             if r["status"] == "fitted" else f"{r['status']}: {r['reason']}"))
run = lin_time[lin_time["status"] == "run"]
print(f"\n  Linearity (age at first note squared and span squared): {len(run)} of "
      f"{len(lin_time)} run; flagged (shift >= "
      f"{au.LINEARITY_SHIFT_YEARS} yr): {int(run['flag'].sum())}"
      + (f"; largest shift {run['shift'].max():.3f} yr" if len(run) else ""))

# =============================================================================
# SECTION 5 -- SEQUENCE (EXPECTED ORDER OF FIRST DOCUMENTATION)
# =============================================================================
au.section("SECTION 5 -- TIER AND THE EXPECTED ORDER OF FIRST DOCUMENTATION")
print("  Is the AID+ vs AID- difference in an inverted order modified by the tier? Per tier and")
print("  pair: AID+ vs AID- within patients with and within patients without a tier mention")
print("  (Fisher's exact test, conditional MLE odds ratio, exact 95% CI), the ratio of the two")
print("  odds ratios and the exact conditional test of the AID x tier interaction (unadjusted).")
print("  Inverted: the later-expected category first documented strictly before the first;")
print("  same day = not inverted (tie). With the tier: a mention dated on or before the earlier")
print(f"  of the two first documentations. Span >= {SEQUENCE_FLOOR_YRS} yr.")
first_cat = au.first_documentation(rec, pop, "category").merge(
    pop[POP_COLUMNS], on="person_id", how="left", validate="many_to_one")


def pair_frame(first, second, yrs, x):
    el = first_cat[first_cat["predx_documentation_span_yrs"] >= yrs]
    a = el[el["code"] == first][["person_id", "first_age", "first_date"]].rename(
        columns={"first_age": "age_first", "first_date": "date_first"})
    b = el[el["code"] == second][["person_id", "first_age", "first_date"]].rename(
        columns={"first_age": "age_second", "first_date": "date_second"})
    d = a.merge(b, on="person_id", how="inner").merge(
        pop[["person_id", "aid_status", "female"]], on="person_id", how="left",
        validate="one_to_one")
    d[x] = exposure_on_or_before(d["person_id"], d[["date_first", "date_second"]].min(axis=1), x)
    gap = d["age_second"] - d["age_first"]
    d["inverted"] = (gap < 0).astype(int)
    d["tied"] = (gap == 0).astype(int)
    return d


def stratum_counts(d, x):
    """Per tier stratum (exposed = with the tier): patients, inverted and tied,
    AID+ and AID-."""
    out = {}
    for s, e in (("exposed", 1), ("unexposed", 0)):
        for a, arm in ((1, "aid_pos"), (0, "aid_neg")):
            g = d[(d[x] == e) & (d["aid_status"] == a)]
            out.update({f"n_{arm}_{s}": len(g), f"inverted_{arm}_{s}": int(g["inverted"].sum()),
                        f"tied_{arm}_{s}": int(g["tied"].sum())})
    return out


def modification_row(key, d, x):
    c = stratum_counts(d, x)
    strata = [(c[f"inverted_aid_pos_{s}"], c[f"n_aid_pos_{s}"], c[f"inverted_aid_neg_{s}"],
               c[f"n_aid_neg_{s}"]) for s in au.EFFECT_MODIFICATION_STRATA]
    return {**key, **c, **au.effect_modification_row(*strata, "an inverted order")}


seq_rows = []
for x in TIER_NAMES:
    for first, second in au.SEQUENCE_PAIRS:
        d = pair_frame(first, second, SEQUENCE_FLOOR_YRS, x)
        key = {"tier": x, "floor_yrs": SEQUENCE_FLOOR_YRS, "first": first, "second": second,
               "row_label": au.sequence_pair_label(first, second)}
        seq_rows.append(modification_row({**key, "fdr_role": "tested", "sample": "all"}, d, x))
        # check: ties excluded (same test; not in any FDR family)
        seq_rows.append(modification_row({**key, "fdr_role": "check", "sample": "ties_excluded"},
                                         d[d["tied"] == 0], x))
seq_all = pd.DataFrame(seq_rows)
# FDR: per tier, over the tested pairs
seq_all = bh_by(seq_all, ["tier", "sample"],
                (seq_all["status"] == "tested") & (seq_all["fdr_role"] == "tested"))
seq_all.loc[seq_all["fdr_role"] == "check", ["q", "fdr_family_size"]] = [NAN, 0]
sequence = seq_all[seq_all["fdr_role"] == "tested"]
seq_checks = seq_all[seq_all["fdr_role"] == "check"]


def print_modification(df):
    for _, r in df.iterrows():
        cnt = "  ".join(f"{s}: AID+ {r[f'inverted_aid_pos_{s}']}/{r[f'n_aid_pos_{s}']} "
                        f"AID- {r[f'inverted_aid_neg_{s}']}/{r[f'n_aid_neg_{s}']} "
                        + (f"OR {fmt_console(r[f'odds_ratio_{s}'], '.2f')}"
                           if r[f"status_{s}"] == "tested" else "not tested")
                        for s in au.EFFECT_MODIFICATION_STRATA)
        res = (f"ratio {fmt_console(r['or_ratio'], '.2f')} P={r['p']:.4f} "
               f"Q={fmt_console(r['q'], '.4f')}" if r["status"] == "tested"
               else f"not tested: {r['reason']}")
        print(f"  {r['tier']:<20} {r['row_label']:<46} {res}\n      [{cnt}]")


print_modification(sequence)
au.sub("Check: ties excluded (same test, no Q)")
print_modification(seq_checks)

# =============================================================================
# SECTION 6 -- SECONDARY DOMAIN DEFINITION CHECK
# =============================================================================
au.section("SECTION 6 -- SECONDARY DOMAIN DEFINITION CHECK")
differs_prev = [dom for dom in sorted(dom_ids["primary"])
                if dom_ids["primary"][dom] != dom_ids["secondary"][dom]]
key_cols = ["person_id", "code", "first_date"]
differs_time = []
for dom in sorted(au.DOMAINS_PRIMARY):
    if dom in CIRCULAR_DOMAINS:
        continue
    a = first_dom["primary"].loc[first_dom["primary"]["code"] == dom, key_cols]
    b = first_dom["secondary"].loc[first_dom["secondary"]["code"] == dom, key_cols]
    if not a.reset_index(drop=True).equals(b.reset_index(drop=True)):
        differs_time.append(dom)
print(f"  Domains whose patient flags differ (prevalence): {', '.join(differs_prev) or 'none'}")
print(f"  Domains whose first documentation differs (timing): {', '.join(differs_time) or 'none'}")
SEC_COLUMNS = ["analysis", "tier", "estimate", "code", "row_label", "status_primary",
               "status_secondary", "estimate_primary", "estimate_secondary", "estimate_change",
               "q_primary", "q_secondary", "became_significant", "lost_significance",
               "change_ge_small", "change_ge_large"]


def definition_check(df, analysis, differs, est_col, small, large):
    """Primary vs secondary definition for the tested estimate (the
    interaction) of every domain whose outcome differs."""
    rows = []
    for x in TIER_NAMES:
        for code, _ in TESTED:
            for dom in differs:
                sel = (df["tier"] == x) & (df["estimate"] == code) & (df["code"] == dom)
                p_ = df[sel & (df["definition"] == "primary")].iloc[0]
                s_ = df[sel & (df["definition"] == "secondary")].iloc[0]
                both = p_["status"] == "fitted" and s_["status"] == "fitted"
                change = s_[est_col] - p_[est_col] if both else NAN
                sig_p = pd.notna(p_["q"]) and p_["q"] < Q_THRESHOLD
                sig_s = pd.notna(s_["q"]) and s_["q"] < Q_THRESHOLD
                r = {"analysis": analysis, "tier": x, "estimate": code, "code": dom,
                     "row_label": au.DOMAIN_LABELS[dom], "status_primary": p_["status"],
                     "status_secondary": s_["status"], "estimate_primary": p_[est_col],
                     "estimate_secondary": s_[est_col], "estimate_change": change,
                     "q_primary": p_["q"], "q_secondary": s_["q"],
                     "became_significant": int(both and sig_s and not sig_p),
                     "lost_significance": int(both and sig_p and not sig_s),
                     "change_ge_small": int(both and abs(change) >= small),
                     "change_ge_large": int(both and abs(change) >= large)}
                rows.append(r)
                crossed = r["became_significant"] or r["lost_significance"]
                print(f"  {analysis:<10} {x:<20} {code:<12} {dom:<10} "
                      f"{fmt_console(p_[est_col], '.2f')} -> {fmt_console(s_[est_col], '.2f')}  "
                      f"Q {fmt_console(p_['q'], '.4f')} -> {fmt_console(s_['q'], '.4f')}  "
                      f"[{'Q crossed ' + str(Q_THRESHOLD) if crossed else 'Q side unchanged'}; "
                      f"|change| >= {large}: {'yes' if r['change_ge_large'] else 'no'}, "
                      f">= {small}: {'yes' if r['change_ge_small'] else 'no'}]")
    return pd.DataFrame(rows, columns=SEC_COLUMNS)


sec_prev = definition_check(prev[prev["level"] == "domain"], "prevalence", differs_prev,
                            "odds_ratio", OR_CHANGE_SMALL, OR_CHANGE_LARGE)
sec_time = definition_check(timing[timing["analysis"] == "domain"], "timing", differs_time,
                            "coef", ESTIMATE_CHANGE_SMALL_YRS, ESTIMATE_CHANGE_LARGE_YRS)

# =============================================================================
# SECTION 7 -- TIER AND THE ANNUAL CHANGE OF COGNITIVE SCORES
# =============================================================================
au.section("SECTION 7 -- TIER AND THE ANNUAL CHANGE OF COGNITIVE SCORES")
print("  OLS, HC3: slope ~ aid_status + age_c + female [+ tier [+ aid_status:tier]], per score;")
print("  slope = (last - first score) / years between them (analysis_utils.annual_change, as")
print("  in cognitive_score_comparison.py), age_c = age at the first score, centered.")
print("  Exposed: a tier mention dated on or before the patient's last score.")
scores, _ = au.absolute_cognitive_scores(
    au.load_cognitive_scores(COGNITIVE_SCORES_PATH, pop, notes), pop)
scored = set(scores["person_id"].astype(int))
SLOPE_BASE = "aid_status + age_c + female"
slopes = {st: au.annual_change(scores, st) for st in au.COGNITIVE_SCORE_TYPES}
for st, s in slopes.items():
    au.count_line(f"patients with an annual change of {st}", len(s), len(pop))
cog_model_rows, cog_est_rows, lin_cog = [], [], []
for x in TIER_NAMES:
    for st in au.COGNITIVE_SCORE_TYPES:
        d = slopes[st].copy()
        d[x] = exposure_on_or_before(d["person_id"], d["last_score_date"], x)
        fC = f"slope ~ {SLOPE_BASE} + {x} + aid_status:{x}"
        why = au.model_floor_reason(d["slope"], d["aid_status"], n_terms(fC),
                                    sex=d["female"], continuous=True, modifier=d[x])
        status, reason, fits = model_set("ols", "slope", SLOPE_BASE, d, x, why)
        key = {"tier": x, "code": st, "row_label": st, "fdr_role": "tested"}
        counts = cell_counts(d, "slope", x, False)
        cog_model_rows += model_rows(key, status, reason, fits, x, len(d), counts)
        cog_est_rows += combination_rows(key, status, reason, fits, len(d), counts, ESTIMATES,
                                         estimate_terms(x), "estimate")
        if status == "fitted":
            lin_cog += linearity_rows(key, "ols", "slope", SLOPE_BASE, d, x, fits, "age_c_sq",
                                      ratio=False, shift_threshold=au.LINEARITY_SHIFT_POINTS)
        else:
            lin_cog.append({**key, "model": "", "term": "", "status": "not_run",
                            "reason": f"models not fitted: {reason}"})

cog_models = pd.DataFrame(cog_model_rows)
cog_models = bh_by(cog_models, ["tier", "model", "term"],
                   (cog_models["status"] == "fitted") & (cog_models["model"] == "C"))
check_ci(cog_models, "cognitive score models", False)
cognitive = pd.DataFrame(cog_est_rows)
cognitive = bh_by(cognitive, ["tier", "estimate"],
                  (cognitive["status"] == "fitted") & (cognitive["estimate"] == INTERACTION))
check_ci(cognitive, "cognitive score estimates", False)
lin_cog = linearity_frame(lin_cog)
for _, r in cognitive.iterrows():
    print(f"  {r['tier']:<20} {r['code']:<5} {r['estimate_label']:<30} "
          + (f"{r['coef']:+.2f} points/yr [{r['ci_low']:+.2f}, {r['ci_high']:+.2f}]"
             + (f" P={r['p']:.4f} Q={r['q']:.4f}" if r["estimate"] == INTERACTION else "")
             if r["status"] == "fitted" else f"{r['status']}: {r['reason']}"))
run = lin_cog[lin_cog["status"] == "run"]
print(f"\n  Linearity (age at the first score squared): {len(run)} of {len(lin_cog)} run; "
      f"flagged (shift >= {au.LINEARITY_SHIFT_POINTS} points/yr): {int(run['flag'].sum())}"
      + (f"; largest shift {run['shift'].max():.3f}" if len(run) else ""))

# =============================================================================
# SECTION 8 -- SOURCE DATA
# =============================================================================
au.section("SECTION 8 -- SOURCE DATA")
FIT_COLUMNS = ["n_model", "status", "reason", "coef", "se", "crit", "ci_low", "ci_high", "p",
               "q", "fdr_family_size"]
CELL_N = [f"n_{lbl}" for _, _, lbl in CELLS]
CELL_EVENTS = [f"events_{lbl}" for _, _, lbl in CELLS]
CELL_TIED = [f"tied_{lbl}" for _, _, lbl in CELLS]


def family_size(df):
    s = df.loc[df["q"].notna(), "fdr_family_size"]
    return int(s.iloc[0]) if len(s) else 0


def parameters(extra=()):
    rows = [("min_events_per_term", au.MIN_EVENTS_PER_TERM), ("min_group_n", au.MIN_GROUP_N),
            ("age_ceiling_quantile", au.AGE_CEILING_QUANTILE),
            ("age_ceiling_female", ceil["ceiling_female"]),
            ("age_ceiling_male", ceil["ceiling_male"]),
            ("age_ceiling_n_ref_female", ceil["n_ref_female"]),
            ("age_ceiling_n_ref_male", ceil["n_ref_male"]),
            ("q_threshold", Q_THRESHOLD),
            ("n_aid_pos", int(pop["aid_status"].sum())),
            ("n_aid_neg", int((pop["aid_status"] == 0).sum()))]
    for a, s, lbl in GROUPS:
        rows.append((f"n_{'aid_pos' if a else 'aid_neg'}_{'female' if s else 'male'}",
                     int(((pop["aid_status"] == a) & (pop["female"] == s)).sum())))
    for x in TIER_NAMES:
        for a, e, lbl in CELLS:
            rows.append((f"n_{lbl}_{x}", int(((pop["aid_status"] == a) & (pop[x] == e)).sum())))
    return pd.DataFrame(rows + list(extra), columns=["parameter", "value"])


au.write_workbook(os.path.join(SOURCE_DATA_DIR, FREQUENCY_FILE), {
    "frequency": freq_df[["tier", "row_label", "sex", "with_tier_aid_pos", "n_aid_pos",
                          "with_tier_aid_neg", "n_aid_neg", "status", "reason", "odds_ratio",
                          "ci_low", "ci_high", "or_estimable", "p", "q", "fdr_family_size"]],
    "parameters": parameters([("fdr_family_size", family_size(freq_df))]),
})

PREV_SHEET = ["tier", "level", "code", "domain", "row_label", "fdr_role", "estimate",
              "estimate_label"] + CELL_EVENTS + CELL_N + ["n_aid_pos", "n_aid_neg"] \
    + FIT_COLUMNS[:6] + ["odds_ratio"] + FIT_COLUMNS[6:]
prev_primary = prev[prev["definition"] == "primary"]
au.write_workbook(os.path.join(SOURCE_DATA_DIR, PREVALENCE_FILE), {
    "estimates": prev_primary[PREV_SHEET],
    "secondary_definition": sec_prev,
    "parameters": parameters(
        [(f"fdr_family_size_{x}_{lvl}",
          family_size(prev_primary[(prev_primary["tier"] == x) & (prev_primary["level"] == lvl)
                                   & (prev_primary["estimate"] == INTERACTION)]))
         for x in TIER_NAMES for lvl in ("domain", "category")]
        + [("or_change_small", OR_CHANGE_SMALL), ("or_change_large", OR_CHANGE_LARGE)]),
})

TIME_SHEET = ["tier", "code", "row_label", "fdr_role", "estimate", "estimate_label"] \
    + CELL_N + ["n_aid_pos", "n_aid_neg"] + FIT_COLUMNS
time_primary = timing[(timing["definition"] == "primary") & (timing["analysis"] == "domain")]
au.write_workbook(os.path.join(SOURCE_DATA_DIR, TIMING_FILE), {
    "estimates": time_primary[TIME_SHEET],
    "secondary_definition": sec_time,
    "parameters": parameters(
        [(f"fdr_family_size_{x}",
          family_size(time_primary[(time_primary["tier"] == x)
                                   & (time_primary["estimate"] == INTERACTION)]))
         for x in TIER_NAMES]
        + [("estimate_change_small_yrs", ESTIMATE_CHANGE_SMALL_YRS),
           ("estimate_change_large_yrs", ESTIMATE_CHANGE_LARGE_YRS),
           ("linearity_shift_years", au.LINEARITY_SHIFT_YEARS)]),
})

STRATUM_COLUMNS = [f"{k}_{s}" for s in au.EFFECT_MODIFICATION_STRATA for k in (
    "n_aid_pos", "inverted_aid_pos", "tied_aid_pos", "n_aid_neg", "inverted_aid_neg",
    "tied_aid_neg", "status", "reason", "or_estimable", "odds_ratio", "ci_low", "ci_high", "p")]
SEQ_SHEET = ["tier", "floor_yrs", "first", "second", "row_label", "fdr_role"] \
    + STRATUM_COLUMNS + ["status", "reason", "ratio_estimable", "or_ratio", "p", "q",
                         "fdr_family_size"]
au.write_workbook(os.path.join(SOURCE_DATA_DIR, SEQUENCE_FILE), {
    "pairs": sequence[SEQ_SHEET],
    "parameters": parameters(
        [(f"fdr_family_size_{x}", family_size(sequence[sequence["tier"] == x]))
         for x in TIER_NAMES]
        + [("predx_span_floor_yrs", SEQUENCE_FLOOR_YRS)]),
})

au.write_workbook(os.path.join(SOURCE_DATA_DIR, COGNITIVE_FILE), {
    "estimates": cognitive[TIME_SHEET],
    "parameters": parameters(
        [(f"fdr_family_size_{x}",
          family_size(cognitive[(cognitive["tier"] == x)
                                & (cognitive["estimate"] == INTERACTION)]))
         for x in TIER_NAMES]
        + [("min_slope_gap_yrs", au.MIN_SLOPE_GAP_YRS),
           ("linearity_shift_points", au.LINEARITY_SHIFT_POINTS),
           ("n_scored_aid_pos", int(pop.loc[pop["aid_status"] == 1, "person_id"].astype(int)
                                    .isin(scored).sum())),
           ("n_scored_aid_neg", int(pop.loc[pop["aid_status"] == 0, "person_id"].astype(int)
                                    .isin(scored).sum())),
           ("score_rows_kept", int(len(scores)))]),
})

au.write_workbook(os.path.join(SOURCE_DATA_DIR, CHECKS_FILE), {
    "tier_patients": pd.DataFrame(tier_patient_rows),
    "tier_overlap": pd.DataFrame(overlap_rows),
    "prevalence_models": prev_models,
    "prevalence_contrasts": prev_con,
    "prevalence_secondary": prev[prev["definition"] == "secondary"],
    "linearity_prevalence": lin_prev,
    "timing_models": time_models,
    "timing_contrasts": timing_con,
    "timing_secondary_ad_core": timing[(timing["definition"] == "secondary")
                                       | (timing["analysis"] == "ad_core_onset")],
    "linearity_timing": lin_time,
    "sequence_checks": seq_checks[["sample"] + SEQ_SHEET],
    "cognitive_score_models": cog_models,
    "linearity_cognitive_scores": lin_cog,
    "parameters": parameters([
        ("tier_records_dated", int(len(tiers))),
        ("tier_records_past_ceiling", int(len(tiers) - len(tier_presence))),
        ("linearity_or_shift", au.LINEARITY_OR_SHIFT),
        ("linearity_or_min_diff", au.LINEARITY_OR_MIN_DIFF),
        ("linearity_shift_years", au.LINEARITY_SHIFT_YEARS)]),
})
print("\n" + au.SEP + "\n  COMPLETE\n" + au.SEP)
