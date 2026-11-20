"""
medication_analysis.py
======================
Does exposure to the medications used to treat autoimmune disease (AID)
change the AID+ vs AID- difference in the documented symptoms of Alzheimer's
disease (AD)? For each of corticosteroids, synthetic DMARDs and
immunosuppressants, antimalarials and biologic DMARDs, the tables report
whether the AID association differs between
exposed and unexposed patients: the interaction (the AID odds ratio among
exposed patients divided by the AID odds ratio among unexposed patients; for
age, the AID difference among exposed minus that among unexposed). Outcomes:
documentation of each symptom domain and category (prevalence) and age at
first documentation of each domain and category (timing), from one model
per outcome with AID status, exposure and their interaction (Model C); and
inversion of the expected order of first documentation (sequence), from the
AID+ vs AID- odds ratios within each exposure stratum and an exact test of
their interaction.

Computes every statistic and writes the source data for the medication tables
(one workbook per table) and a checks workbook. The tables are rendered by
medication_tables.py; nothing here formats a number.

Exposure (medication records, by drug class)
--------------------------------------------
- Prevalence: exposed = a record dated at or below the patient's age ceiling,
  or an undated record. Records dated past the ceiling do not count; an
  undated record cannot be placed, so it counts, as in "ever exposed".
- Timing and sequence: exposed = a record dated strictly before the first
  documentation of the domain or category (sequence: of the earlier of the two
  categories); undated records do not count. A drug cannot change the date of
  a symptom documented before it, and a symptom recorded on the day of the
  record is unlikely to be caused by it, the reasoning used for the diagnosis
  day. Exposure
  at any time ("ever", any record, dated or not) is refitted as a check.
- Exposure checks: any record (ever), and a first dated record strictly before
  the AD diagnosis date, by is_pre_diagnosis.
NSAIDs, multiple sclerosis disease-modifying therapies, cholinesterase
inhibitors, NMDA receptor antagonists and anti-amyloid monoclonal antibodies
are described only (checks; no table).

Models
------
- Prevalence: logit has_x ~ aid_status + female + age_at_dx [+ exposure
  [+ aid_status:exposure]], weighted by ps_weight_trimmed (binomial GLM,
  var_weights, HC0), per domain (both definitions) and category; symptom
  records at or below the age ceiling. Corticosteroids: Models A (AID),
  B (+ exposure) and C at the category level, the set marked failed when any
  of the three does not fit; other classes and domains: Model C.
- Timing: OLS HC3, first_age ~ aid_status + female [+ exposure
  [+ aid_status:exposure]] + age_at_entry_c + predx_span_c, among patients
  with the domain documented, and the same for every category except the
  unassigned one ("other"), among patients with the category documented
  (corticosteroids: A, B, C; others: C); and the AD-core onset
  (analysis_utils.ad_core_onset; checks, a single test).
  age_at_entry_c is age at the first dated note (a symptom cannot be
  documented before the record begins) and predx_span_c the pre-diagnosis
  documentation span, both centered within each sample; together they
  also hold age at AD diagnosis fixed.
- Sequence: is the AID+ vs AID- difference in an inverted order of first
  documentation modified by exposure? Per expected-order pair
  (analysis_utils.SEQUENCE_PAIRS), among patients with both categories
  documented and a span of at least 5 years: AID+ vs AID- within exposed and
  within unexposed patients (Fisher's exact test, conditional MLE odds ratio,
  exact 95% CI), the ratio of the two odds ratios, and the exact conditional
  test of the AID x exposure interaction (analysis_utils.
  effect_modification_row); unadjusted. Tested when both strata pass
  two_by_two_reason. A same-day first documentation (tie) counts as not
  inverted; tie counts, a ties-excluded check and exposure "ever" (check) are
  written.
Gate (prevalence, timing): analysis_utils.model_floor_reason on Model C (10
per term on the rarer outcome class, or patients for timing; both AID
groups, the outcome varying within each; both sexes; each AID x exposure
group with 10 in its rarer outcome class, or 10 patients). FDR: BH
within each class x level x definition x model x term (prevalence), class x
analysis x level (domain, category) x definition x exposure definition x
model x term (timing), over the tested rows: Model A AID, Model B exposure,
Model C interaction (timing: Model C only); sequence, class over the tested
pairs.
The positive control (systemic inflammation autoimmune and its categories)
shows its P value outside the family; "other" is fitted for information only
(no P or Q). Linearity: age at AD diagnosis squared (prevalence; OR shift
>= 20% with |OR - 1| >= 0.05) or the squares of both centered timing
covariates in one refit (timing; shift >= 0.5 years) added to Models A and C.
Secondary domain definition: every
domain-level interaction refitted under it; the change is printed and
written.

Checks (no table): corticosteroid exposure before the diagnosis vs never;
exposure by AID and sex group and AID+ vs AID- within each sex (Fisher's exact
test); documentation measures, corticosteroid exposed vs unexposed (Mann-
Whitney U); AD pharmacotherapy; medications within AID+.

Inputs: the patient table, the note table, the symptom records, the
propensity-score weights and the medication records (see PATHS).
"""

import os
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import mannwhitneyu, norm, t as t_dist
from statsmodels.genmod.families import Binomial

import analysis_utils as au

warnings.filterwarnings("ignore")   # after the imports: statsmodels resets its own filters

# =============================================================================
# PATHS (relative to the data folder the script is run from)
# =============================================================================
COHORT_PATIENTS_PATH = ""  # patient table: AID status, sex, key dates (birth, AD diagnosis, death) and analysis-population flag
COHORT_NOTES_PATH = ""  # note table: one row per clinical note, with its date and specialty
SYMPTOM_RECORDS_PATH = ""  # symptom records: one row per documented symptom, with its note and symptom category
MEDICATION_RECORDS_PATH = ""  # medication records: one row per exposure record, with its drug class and date
PS_WEIGHTS_PATH = ""  # propensity-score weights, one per analysis patient
SOURCE_DATA_DIR = ""  # folder for the source-data workbooks written here

PREVALENCE_FILE = "table_medication_prevalence.xlsx"
TIMING_FILE = "table_medication_timing.xlsx"
SEQUENCE_FILE = "table_medication_sequence.xlsx"
CHECKS_FILE = "checks_medication.xlsx"

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
# Classes with interaction models: (drug class, readable label)
MODEL_CLASSES = [("on_cs", "Corticosteroids"), ("on_dmard", "Synthetic DMARDs and immunosuppressants"),
                 ("on_antimalarial", "Antimalarials"), ("on_biologic", "Biologic DMARDs")]
FULL_CLASS = "on_cs"               # Models A, B and C (categories, timing)
CLASS_LABELS = {"on_cs": "Corticosteroids", "on_dmard": "Synthetic DMARDs and immunosuppressants",
                "on_biologic": "Biologic DMARDs", "on_antimalarial": "Antimalarials",
                "on_ms_dmt": "Multiple sclerosis disease-modifying therapies",
                "on_nsaid": "Nonsteroidal anti-inflammatory drugs",
                "on_chei": "Cholinesterase inhibitors", "on_nmda_antagonist": "NMDA receptor antagonists",
                "on_anti_amyloid": "Anti-amyloid monoclonal antibodies"}
CLASS_LABELS_LOWER = {c: (lbl if lbl.split()[0].isupper() else lbl[0].lower() + lbl[1:])
                      for c, lbl in CLASS_LABELS.items()}   # in running text
# Classes described in the checks (exposure by group, AID+ vs AID- within sex)
DESCRIBED_CLASSES = ["on_cs", "on_dmard", "on_biologic", "on_antimalarial", "on_ms_dmt",
                     "on_nsaid", "on_chei", "on_nmda_antagonist", "on_anti_amyloid"]
# Exposure definitions of the timing and sequence models
EXPOSURE_BEFORE = "before_first_documentation"   # tables
EXPOSURE_EVER = "ever"                           # check
SEQUENCE_FLOOR_YRS = au.PREDX_SPAN_FLOOR_5_YRS
GROUPS = [(1, 1, "AID+ female"), (1, 0, "AID+ male"), (0, 1, "AID- female"), (0, 0, "AID- male")]
CELLS = [(1, 1, "aid_pos_exposed"), (1, 0, "aid_pos_unexposed"),
         (0, 1, "aid_neg_exposed"), (0, 0, "aid_neg_unexposed")]
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
    for _, idx in df.groupby(family_cols, dropna=False).groups.items():
        m = df.loc[idx]
        t = m.index[eligible.loc[idx] & m["p"].notna()]
        if len(t):
            df.loc[t, "q"] = au.bh_fdr(df.loc[t, "p"].values)
            df.loc[t, "fdr_family_size"] = len(t)
    return df


def exp_ratio_columns(df):
    """Odds ratio and CI from coef and the CI on the log scale (logistic rows)."""
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


def two_by_two_row(e1, n1, e2, n2, event):
    """Gate (two_by_two_reason) and, when tested, the exact 2x2 values."""
    r = {"status": "", "reason": "", "odds_ratio": NAN, "ci_low": NAN, "ci_high": NAN,
         "or_estimable": 0, "p": NAN}
    why = au.two_by_two_reason(e1, n1, e2, n2, event)
    if why:
        r.update(status="not_tested", reason=why)
        return r
    orv, lo, hi, p = au.exact_2x2(e1, n1 - e1, e2, n2 - e2)
    r.update(status="tested", p=p)
    if np.isfinite(orv) and orv > 0:
        r.update(odds_ratio=orv, ci_low=lo, ci_high=hi, or_estimable=1)
    return r


# =============================================================================
# MODEL SETS (A, B, C)
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


# (model, term name in the fit, term role); Model C reports its interaction
MODEL_TERMS = {"A": [("aid_status", "AID")],
               "B": [("aid_status", "AID"), ("{x}", "exposure")],
               "C": [("aid_status:{x}", "interaction")]}


def formulas(y, base_rhs, x):
    """Models A, B, C. base_rhs = (terms before the exposure, terms after it):
    the continuous adjustment covariates come last in every model."""
    pre, post = base_rhs
    tail = f" + {post}" if post else ""
    return {"A": f"{y} ~ {pre}{tail}", "B": f"{y} ~ {pre} + {x}{tail}",
            "C": f"{y} ~ {pre} + {x} + aid_status:{x}{tail}"}


def model_set(kind, y, base_rhs, d, x, why, full, set_rule):
    """Fit Models A, B and C (full) or C alone, when the gate passed (why ==
    ""). Returns {model: (status, reason, fit)}. With set_rule, a model that
    does not fit, or a reported term not estimated, marks the whole set
    failed (corticosteroid prevalence categories); without it, each
    model keeps its own status (timing)."""
    models = ["A", "B", "C"] if full else ["C"]
    forms = formulas(y, base_rhs, x)
    if why:
        return {m: ("not_fitted", why, None) for m in models}
    out = {}
    for m in models:
        try:
            mod = fit_one(kind, forms[m], d)
        except Exception as e:
            reason = str(e) if str(e) == "did not converge" else f"fit error: {type(e).__name__}"
            out[m] = ("failed", reason, None)
            continue
        terms = [t.format(x=x) for t, _ in MODEL_TERMS[m]]
        if any(t not in mod.params.index or not np.isfinite(mod.params[t])
               or not np.isfinite(mod.bse[t]) for t in terms):
            out[m] = ("failed", "term not estimated", None)
        else:
            out[m] = ("fitted", "", mod)
    if set_rule and any(s != "fitted" for s, _, _ in out.values()):
        first_reason = next(r for s, r, _ in out.values() if s != "fitted")
        out = {m: ("failed", r if s != "fitted" else f"another model in the set: {first_reason}",
                   None) for m, (s, r, _) in out.items()}
    return out


def model_rows(key, fits, x, n_model, counts):
    rows = []
    for m, (status, reason, mod) in fits.items():
        for t, role in MODEL_TERMS[m]:
            t = t.format(x=x)
            vals = term_values(mod, t) if status == "fitted" else empty_fit()
            rows.append({**key, "model": m, "term": role, "n_model": n_model, "status": status,
                         "reason": reason, **counts, **vals})
    return rows


def linearity_rows(key, kind, y, base_rhs, d, x, fits, sq, ratio, sex_col="female"):
    """Models A (AID term) and C (interaction) refitted with `sq` added (same
    rows), when the model was fitted. `sq` is one squared column or a
    list of them (added together in one refit; each coefficient written as
    coef_<column>). The quadratic model must meet the events-per-term floor
    with the added terms counted."""
    sqs = [sq] if isinstance(sq, str) else list(sq)
    out = []
    forms = formulas(y, base_rhs, x)
    terms = {"A": ("aid_status", "AID"), "C": (f"aid_status:{x}", "interaction")}
    for m in ("A", "C"):
        if m not in fits:
            continue
        status, reason, mod = fits[m]
        term, role = terms[m]
        row = {**key, "model": m, "term": role, "status": "not_run", "reason": "",
               "estimate_linear": NAN, "estimate_quadratic": NAN, "shift": NAN,
               "shift_pct": NAN, "coef_squared": NAN, "flag": 0}
        if status != "fitted":
            row["reason"] = f"model not fitted: {reason}"
            out.append(row)
            continue
        fq = forms[m] + " + " + " + ".join(sqs)
        why = au.model_floor_reason(d[y], d["aid_status"], n_terms(fq), sex=d[sex_col],
                                    continuous=not ratio)
        if why:
            row["reason"] = why
            out.append(row)
            continue
        try:
            mq = fit_one(kind, fq, d)
        except Exception as e:
            row["reason"] = f"fit error: {type(e).__name__}"
            out.append(row)
            continue
        if term not in mq.params.index or not np.isfinite(mq.params[term]):
            row["reason"] = "term not estimated"
            out.append(row)
            continue
        bl, bq = float(mod.params[term]), float(mq.params[term])
        if ratio:
            el, eq = np.exp(bl), np.exp(bq)
            shift_pct = abs(eq - el) / abs(el) * 100
            flag = int(shift_pct >= au.LINEARITY_OR_SHIFT * 100
                       and abs(el - 1) >= au.LINEARITY_OR_MIN_DIFF)
        else:
            el, eq = bl, bq
            shift_pct = abs(eq - el) / abs(el) * 100 if el != 0 else NAN
            flag = int(abs(eq - el) >= au.LINEARITY_SHIFT_YEARS)
        row.update(status="run", estimate_linear=el, estimate_quadratic=eq, shift=abs(eq - el),
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


# =============================================================================
# SECTION 1 -- POPULATION, AGE CEILING, EXPOSURE PER PATIENT
# =============================================================================
au.section("SECTION 1 -- POPULATION, AGE CEILING, EXPOSURE PER PATIENT")
pop, notes = au.load_population(COHORT_PATIENTS_PATH, COHORT_NOTES_PATH)
pop = au.load_ps_weights(PS_WEIGHTS_PATH, pop)
pop["age_at_dx_sq"] = pop["age_at_dx"] ** 2
after = pop["predx_documentation_span_yrs"] < 0
print("  First dated note after the AD diagnosis (negative pre-diagnosis documentation span):")
for a, lbl in ((1, "AID+"), (0, "AID-")):
    au.count_line(lbl, (after & (pop["aid_status"] == a)).sum(), (pop["aid_status"] == a).sum())

ceil = au.p90_age_ceiling(pop)
ceiling_by_person = ceil["ceiling_by_person"]
print(f"\n  Age ceiling, female (yr): {ceil['ceiling_female']:.2f}  "
      f"(AID- reference n = {ceil['n_ref_female']:,})")
print(f"  Age ceiling, male (yr)  : {ceil['ceiling_male']:.2f}  "
      f"(AID- reference n = {ceil['n_ref_male']:,})")
past_all = ~au.within_age_ceiling(pop["age_at_entry"], pop["person_id"].map(ceiling_by_person))
print("  Patients whose first dated note is past their ceiling (no symptom record counts in "
      "the presence analyses):")
ceiling_rows = []
for a, s, lbl in GROUPS:
    g = (pop["aid_status"] == a) & (pop["female"] == s)
    au.count_line(lbl, (past_all & g).sum(), g.sum())
    ceiling_rows.append({"group": lbl, "aid_status": a, "female": s, "n_patients": int(g.sum()),
                         "n_past_ceiling": int((past_all & g).sum())})

med = au.load_medication_records(MEDICATION_RECORDS_PATH, pop)
med = med.merge(pop[["person_id", "birth_date"]], on="person_id", how="left",
                validate="many_to_one")
med["age_at_record"] = au.years_between(med["exposure_date"], med["birth_date"])
dated_med = med[med["exposure_date"].notna()]
med["counts_in_presence"] = (med["exposure_date"].isna()
                             | au.within_age_ceiling(med["age_at_record"],
                                                     med["person_id"].map(ceiling_by_person)))
au.count_line("medication records dated past the age ceiling (not in the presence analyses)",
              (~med["counts_in_presence"]).sum(), len(med))
# earliest dated record of each class, per patient (timing and order)
FIRST_DATED = {c: dated_med[dated_med["drug_class"] == c].groupby("person_id")["exposure_date"]
               .min().to_dict() for c in DESCRIBED_CLASSES}

exposure_summary = []
print(f"\n  {'Class':<16} {'ever':>7} {'dated':>7} {'undated':>8} {'before':>7} {'on/after':>9} "
      f"{'presence':>9}   (before / on or after the AD diagnosis date: first dated record)")
for c in DESCRIBED_CLASSES:
    rc = med[med["drug_class"] == c]
    pop[f"ever_{c}"] = pop["person_id"].isin(rc["person_id"]).astype(int)
    pop[c] = pop["person_id"].isin(rc.loc[rc["counts_in_presence"], "person_id"]).astype(int)
    first = pd.to_datetime(pop["person_id"].map(FIRST_DATED[c]))
    before = au.is_pre_diagnosis(first, pop["ad_diagnosis_date"])
    on_after = au.is_post_diagnosis(first, pop["ad_diagnosis_date"])
    pop[f"before_dx_{c}"] = before.astype(int)
    pop[f"timing_{c}"] = np.select(
        [pop[f"ever_{c}"] == 0, first.isna(), before, on_after],
        ["never", "undated_only", "before_diagnosis", "on_or_after_diagnosis"], "never")
    tc = pop[f"timing_{c}"].value_counts()
    lost = int(((pop[f"ever_{c}"] == 1) & (pop[c] == 0)).sum())
    exposure_summary.append((c, lost))
    print(f"  {c:<16} {int(pop[f'ever_{c}'].sum()):>7,} "
          f"{int(pop[f'ever_{c}'].sum()) - tc.get('undated_only', 0):>7,} "
          f"{tc.get('undated_only', 0):>8,} {tc.get('before_diagnosis', 0):>7,} "
          f"{tc.get('on_or_after_diagnosis', 0):>9,} {int(pop[c].sum()):>9,}")
print("  Patients exposed ever but not in the presence analyses (every record dated past the "
      "ceiling):")
for c, lost in exposure_summary:
    au.count_line(c, lost)

rec = au.load_symptom_records(SYMPTOM_RECORDS_PATH, pop, notes)
dated = rec[rec["note_date"].notna()].merge(pop[["person_id", "birth_date"]], on="person_id",
                                            how="left", validate="many_to_one")
dated["age_at_record"] = au.years_between(dated["note_date"], dated["birth_date"])
in_ceiling = dated[au.within_age_ceiling(dated["age_at_record"],
                                         dated["person_id"].map(ceiling_by_person))]
au.count_line("dated symptom records within the age ceiling (presence analyses)",
              len(in_ceiling), len(dated))


def exposed_before(person_ids, dates, c):
    """1 when the patient's earliest dated record of class c is strictly before
    `dates` (the first documentation), else 0; undated records do not count."""
    return (pd.to_datetime(person_ids.map(FIRST_DATED[c])) < dates).astype(int)


# =============================================================================
# SECTION 2 -- PREVALENCE
# =============================================================================
au.section("SECTION 2 -- MEDICATION AND SYMPTOM PREVALENCE")
print("  Weighted logit (binomial GLM, HC0): has_x ~ aid_status + female + age_at_dx")
print("  [+ exposure [+ aid_status:exposure]]; symptom and medication records within the age")
print("  ceiling (undated medication records count).")
c2d = {d: au.category_domain(d) for d in DEFINITIONS}
cat_ids = {c: set(in_ceiling.loc[in_ceiling["category"] == c, "person_id"])
           for c in au.SCHEMA_CATEGORIES}
dom_ids = {defn: {dom: set(in_ceiling.loc[in_ceiling["category"].map(c2d[defn]) == dom,
                                          "person_id"])
                  for dom in sorted(au.DOMAIN_DEFINITIONS[defn])}
           for defn in DEFINITIONS}
PREV_BASE = ("aid_status + female", "age_at_dx")

prev_rows, lin_prev = [], []
for x, _ in MODEL_CLASSES:
    outcomes = [("domain", defn, dom, ids) for defn in DEFINITIONS
                for dom, ids in dom_ids[defn].items()] \
        + [("category", "primary", c, ids) for c, ids in sorted(cat_ids.items())]
    for level, defn, code, ids in outcomes:
        full = x == FULL_CLASS and level == "category"
        d = pop[["person_id", "aid_status", "female", "age_at_dx", "age_at_dx_sq", WCOL, x]].copy()
        d["has_x"] = d["person_id"].isin(ids).astype(int)
        fC = formulas("has_x", PREV_BASE, x)["C"]
        why = au.model_floor_reason(d["has_x"], d["aid_status"], n_terms(fC), sex=d["female"],
                                    modifier=d[x])
        fits = model_set("logit_weighted", "has_x", PREV_BASE, d, x, why, full, set_rule=full)
        if level == "category":
            label, dom = au.category_row_label(code), c2d["primary"][code]
            role = ("not_tested" if code in au.NOT_TESTED_CATEGORIES else
                    "positive_control" if code in au.POSITIVE_CONTROL_CATEGORIES else "tested")
        else:
            label, dom = au.DOMAIN_LABELS[code], code
            role = "positive_control" if code in au.POSITIVE_CONTROL_DOMAINS else "tested"
        key = {"drug_class": x, "level": level, "definition": defn, "code": code, "domain": dom,
               "row_label": label, "fdr_role": role}
        prev_rows += model_rows(key, fits, x, len(d), cell_counts(d, "has_x", x, True))
        if defn == "primary":
            lin_prev += linearity_rows(key, "logit_weighted", "has_x", PREV_BASE, d, x, fits,
                                       "age_at_dx_sq", ratio=True)
    print(f"  {x}: fitted")

prev_models = pd.DataFrame(prev_rows)
prev_models.loc[prev_models["fdr_role"] == "not_tested", "p"] = NAN
TESTED_TERMS = {("A", "AID"), ("B", "exposure"), ("C", "interaction")}
elig = ((prev_models["status"] == "fitted") & (prev_models["fdr_role"] == "tested")
        & pd.Series([(m, t) in TESTED_TERMS for m, t in zip(prev_models["model"],
                                                             prev_models["term"])],
                    index=prev_models.index))
prev_models = exp_ratio_columns(bh_by(prev_models, ["drug_class", "level", "definition", "model",
                                                    "term"], elig))
check_ci(prev_models, "prevalence models", True)
lin_prev = linearity_frame(lin_prev)
prev = prev_models[(prev_models["model"] == "C") & (prev_models["term"] == "interaction")]


def print_interactions(df, title, ratio):
    print(f"\n  {title}")
    for x, _ in MODEL_CLASSES:
        m = df[df["drug_class"] == x]
        f = m[m["status"] == "fitted"]
        sig = f[f["q"] < Q_THRESHOLD]
        est = (lambda r: f"OR {r['odds_ratio']:.2f}") if ratio else \
            (lambda r: f"{r['coef']:+.2f} yr")
        key = "row_label"
        print(f"  {x:<16} fitted {len(f):>3} of {len(m):>3}; Q < 0.05: "
              + (", ".join(f"{r[key]} ({est(r)}, Q={r['q']:.4f})" for _, r in sig.iterrows())
                 or "none"))


p1 = prev[prev["definition"] == "primary"]
print_interactions(p1[p1["level"] == "domain"], "Interaction, domains (primary definition)", True)
print_interactions(p1[p1["level"] == "category"], "Interaction, categories", True)
run = lin_prev[lin_prev["status"] == "run"]
print(f"\n  Linearity (age at AD diagnosis squared): {len(run)} of {len(lin_prev)} run; flagged "
      f"(OR shift >= 20% and |OR - 1| >= 0.05): {int(run['flag'].sum())}"
      + (f"; largest shift {run['shift_pct'].max():.1f}%" if len(run) else ""))

# ---- corticosteroids before the diagnosis vs never (check) --------------------
au.sub("Check: corticosteroid exposure before the AD diagnosis date vs never (categories)")
predx_pop = pop[pop["timing_on_cs"].isin(["before_diagnosis", "never"])].copy()
au.count_line("patients with a first corticosteroid record before the diagnosis or none",
              len(predx_pop), len(pop))
predx_rows = []
XP = "before_dx_on_cs"
for code in au.SCHEMA_CATEGORIES:
    d = predx_pop[["person_id", "aid_status", "female", "age_at_dx", WCOL, XP]].copy()
    d["has_x"] = d["person_id"].isin(cat_ids[code]).astype(int)
    fC = formulas("has_x", PREV_BASE, XP)["C"]
    why = au.model_floor_reason(d["has_x"], d["aid_status"], n_terms(fC), sex=d["female"],
                                modifier=d[XP])
    fits = model_set("logit_weighted", "has_x", PREV_BASE, d, XP, why, False, set_rule=False)
    role = ("not_tested" if code in au.NOT_TESTED_CATEGORIES else
            "positive_control" if code in au.POSITIVE_CONTROL_CATEGORIES else "tested")
    predx_rows += model_rows({"drug_class": "on_cs", "exposure_definition": "before_diagnosis",
                              "level": "category", "code": code,
                              "row_label": au.category_row_label(code), "fdr_role": role},
                             fits, XP, len(d), cell_counts(d, "has_x", XP, True))
predx_cs = exp_ratio_columns(pd.DataFrame(predx_rows))
predx_cs.loc[predx_cs["fdr_role"] == "not_tested", "p"] = NAN
check_ci(predx_cs, "corticosteroids before the diagnosis", True)
print(f"  fitted {int((predx_cs['status'] == 'fitted').sum())} of {len(predx_cs)} categories "
      f"(interaction, uncorrected; written to the checks)")

# =============================================================================
# SECTION 3 -- TIMING (AGE AT FIRST DOCUMENTATION) AND AD-CORE ONSET
# =============================================================================
au.section("SECTION 3 -- MEDICATION AND AGE AT FIRST DOCUMENTATION")
print("  OLS, HC3: first_age ~ aid_status + female [+ exposure [+ aid_status:exposure]]")
print("  + age_at_entry_c + predx_span_c, among patients with the domain documented;")
print("  age_at_entry_c = age at the first dated note, predx_span_c = pre-diagnosis")
print("  documentation span, both centered within each sample. Exposed: a record dated")
print("  strictly before the domain's first documentation (check: any record, ever).")
POP_COLUMNS = ["person_id", "aid_status", "female", "age_at_entry",
               "predx_documentation_span_yrs"]
# adjustment covariates of the timing models: (centered column, source column)
TIME_COVARIATES = [("age_at_entry_c", "age_at_entry"),
                   ("predx_span_c", "predx_documentation_span_yrs")]
TIME_BASE = ("aid_status + female", " + ".join(c for c, _ in TIME_COVARIATES))
TIME_SQUARES = [f"{c}_sq" for c, _ in TIME_COVARIATES]


def timing_sample(frame):
    d = frame.merge(pop[POP_COLUMNS], on="person_id", how="left", validate="many_to_one")
    for name, source in TIME_COVARIATES:
        d[name] = d[source] - d[source].mean()
        d[name + "_sq"] = d[name] ** 2
    return d


def with_exposure(d, x, exposure, date_col):
    d = d.copy()
    if exposure == EXPOSURE_BEFORE:
        d[x] = exposed_before(d["person_id"], d[date_col], x)
    else:
        d[x] = d["person_id"].map(pop.set_index("person_id")[f"ever_{x}"]).astype(int)
    return d


def timing_target(x, full, analysis, level, defn, code, domain, label, role, frame):
    """Model C (corticosteroids: A, B, C) of one target under each exposure
    definition ("ever" only for primary-definition domains and categories),
    and the linearity rows of the same targets. Returns (rows, linearity rows)."""
    rows, lin = [], []
    base = timing_sample(frame[["person_id", "first_date", "first_age"]])
    for exposure in (EXPOSURE_BEFORE, EXPOSURE_EVER):
        if exposure == EXPOSURE_EVER and (analysis not in ("domain", "category")
                                          or defn != "primary"):
            continue
        d = with_exposure(base, x, exposure, "first_date")
        fC = formulas("first_age", TIME_BASE, x)["C"]
        why = au.model_floor_reason(d["first_age"], d["aid_status"], n_terms(fC),
                                    sex=d["female"], continuous=True, modifier=d[x])
        fits = model_set("ols", "first_age", TIME_BASE, d, x, why, full, set_rule=False)
        key = {"drug_class": x, "analysis": analysis, "level": level, "definition": defn,
               "exposure_definition": exposure, "code": code, "domain": domain,
               "row_label": label, "fdr_role": role}
        rows += model_rows(key, fits, x, len(d), cell_counts(d, "first_age", x, False))
        if analysis in ("domain", "category") and defn == "primary":
            lin += linearity_rows(key, "ols", "first_age", TIME_BASE, d, x, fits,
                                  TIME_SQUARES, ratio=False)
    return rows, lin


first_dom = {defn: au.first_documentation(rec, pop, "domain", defn) for defn in DEFINITIONS}
first_cat_doc = au.first_documentation(rec, pop, "category")
core = au.ad_core_onset(first_dom["primary"])
au.count_line("patients with an AD-core onset", len(core), len(pop))
time_rows, lin_time = [], []
for x, _ in MODEL_CLASSES:
    full = x == FULL_CLASS
    targets = [("domain", defn, code, first_dom[defn][first_dom[defn]["code"] == code])
               for defn in DEFINITIONS for code in sorted(au.DOMAIN_DEFINITIONS[defn])] \
        + [("ad_core_onset", "primary", "combined", core)]
    for analysis, defn, code, frame in targets:
        role = "positive_control" if code in au.POSITIVE_CONTROL_DOMAINS else "tested"
        rows, lin = timing_target(x, full, analysis, "domain", defn, code,
                                  code if analysis == "domain" else NAN,
                                  au.DOMAIN_LABELS.get(code, "AD core (combined)"), role, frame)
        time_rows += rows
        lin_time += lin
    print(f"  {x}: fitted")

# categories (every category but the unassigned one), after every domain row
TIME_CATEGORIES = sorted(c for c in au.SCHEMA_CATEGORIES if c not in au.UNASSIGNED_CATEGORIES)
for x, _ in MODEL_CLASSES:
    full = x == FULL_CLASS
    for code in TIME_CATEGORIES:
        role = "positive_control" if code in au.POSITIVE_CONTROL_CATEGORIES else "tested"
        rows, lin = timing_target(x, full, "category", "category", "primary", code,
                                  c2d["primary"][code], au.category_row_label(code), role,
                                  first_cat_doc[first_cat_doc["code"] == code])
        time_rows += rows
        lin_time += lin
    print(f"  {x}: fitted (categories)")

time_models = pd.DataFrame(time_rows)
time_models = bh_by(time_models, ["drug_class", "analysis", "level", "definition",
                                  "exposure_definition", "model", "term"],
                    (time_models["status"] == "fitted") & (time_models["model"] == "C")
                    & (time_models["fdr_role"] == "tested"))
check_ci(time_models, "timing models", False)
lin_time = linearity_frame(lin_time)
timing = time_models[(time_models["model"] == "C")
                     & time_models["analysis"].isin(["domain", "category"])
                     & (time_models["exposure_definition"] == EXPOSURE_BEFORE)]
print_interactions(timing[(timing["definition"] == "primary") & (timing["level"] == "domain")],
                   "Interaction, domains (primary definition; exposure before first documentation)",
                   False)
print_interactions(timing[timing["level"] == "category"],
                   "Interaction, categories (exposure before first documentation)", False)
au.sub("AD-core onset (combined target; a single test per class, Q = P; checks)")
for _, r in time_models[(time_models["analysis"] == "ad_core_onset")
                        & (time_models["model"] == "C")].iterrows():
    print(f"  {r['drug_class']:<16} "
          + (f"{r['coef']:+.2f} yr [{r['ci_low']:+.2f}, {r['ci_high']:+.2f}] P={r['p']:.4f}"
             if r["status"] == "fitted" else f"{r['status']}: {r['reason']}"))
for level, words in (("domain", ""), ("category", ", categories")):
    lin_level = lin_time[lin_time["level"] == level]
    run = lin_level[lin_level["status"] == "run"]
    print(f"\n  Linearity{words} (age at first note squared and span squared): {len(run)} of "
          f"{len(lin_level)} run; flagged (shift >= "
          f"{au.LINEARITY_SHIFT_YEARS} yr): {int(run['flag'].sum())}"
          + (f"; largest shift {run['shift'].max():.3f} yr" if len(run) else ""))

# =============================================================================
# SECTION 4 -- SEQUENCE (EXPECTED ORDER OF FIRST DOCUMENTATION)
# =============================================================================
au.section("SECTION 4 -- MEDICATION AND THE EXPECTED ORDER OF FIRST DOCUMENTATION")
print("  Is the AID+ vs AID- difference in an inverted order modified by exposure? Per class")
print("  and pair: AID+ vs AID- within exposed and within unexposed patients (Fisher's exact")
print("  test, conditional MLE odds ratio, exact 95% CI), the ratio of the two odds ratios and")
print("  the exact conditional test of the AID x exposure interaction (unadjusted).")
print("  Inverted: the later-expected category first documented strictly before the first;")
print("  same day = not inverted (tie). Exposed: a record dated strictly before the earlier of")
print(f"  the two first documentations (check: any record, ever). Span >= {SEQUENCE_FLOOR_YRS} yr.")
first_cat = first_cat_doc.merge(pop[POP_COLUMNS], on="person_id", how="left",
                                validate="many_to_one")


def pair_frame(first, second, yrs):
    el = first_cat[first_cat["predx_documentation_span_yrs"] >= yrs]
    a = el[el["code"] == first][["person_id", "first_age", "first_date"]].rename(
        columns={"first_age": "age_first", "first_date": "date_first"})
    b = el[el["code"] == second][["person_id", "first_age", "first_date"]].rename(
        columns={"first_age": "age_second", "first_date": "date_second"})
    d = a.merge(b, on="person_id", how="inner").merge(
        pop[["person_id", "aid_status", "female"]], on="person_id", how="left",
        validate="one_to_one")
    d["earlier_date"] = d[["date_first", "date_second"]].min(axis=1)
    gap = d["age_second"] - d["age_first"]
    d["inverted"] = (gap < 0).astype(int)
    d["tied"] = (gap == 0).astype(int)
    return d


def stratum_counts(d, x):
    """Per exposure stratum: patients, inverted and tied, AID+ and AID-."""
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


PAIR_FRAMES = {(f, s): pair_frame(f, s, SEQUENCE_FLOOR_YRS) for f, s in au.SEQUENCE_PAIRS}
seq_rows = []
for x, _ in MODEL_CLASSES:
    for first, second in au.SEQUENCE_PAIRS:
        for exposure in (EXPOSURE_BEFORE, EXPOSURE_EVER):
            d = with_exposure(PAIR_FRAMES[(first, second)], x, exposure, "earlier_date")
            key = {"drug_class": x, "floor_yrs": SEQUENCE_FLOOR_YRS, "first": first,
                   "second": second, "row_label": au.sequence_pair_label(first, second),
                   "exposure_definition": exposure}
            seq_rows.append(modification_row({**key, "fdr_role": "tested", "sample": "all"},
                                             d, x))
            if exposure == EXPOSURE_BEFORE:
                # check: ties excluded (same test; not in any FDR family)
                seq_rows.append(modification_row({**key, "fdr_role": "check",
                                                  "sample": "ties_excluded"},
                                                 d[d["tied"] == 0], x))
seq_all = pd.DataFrame(seq_rows)
# FDR: per class and exposure definition, over the tested pairs
seq_all = bh_by(seq_all, ["drug_class", "exposure_definition", "sample"],
                (seq_all["status"] == "tested") & (seq_all["fdr_role"] == "tested"))
seq_all.loc[seq_all["fdr_role"] == "check", ["q", "fdr_family_size"]] = [NAN, 0]
sequence = seq_all[(seq_all["fdr_role"] == "tested")
                   & (seq_all["exposure_definition"] == EXPOSURE_BEFORE)]
seq_checks = seq_all[(seq_all["fdr_role"] == "check")
                     | (seq_all["exposure_definition"] == EXPOSURE_EVER)]


def print_modification(df, by):
    for _, r in df.iterrows():
        cnt = "  ".join(f"{s}: AID+ {r[f'inverted_aid_pos_{s}']}/{r[f'n_aid_pos_{s}']} "
                        f"AID- {r[f'inverted_aid_neg_{s}']}/{r[f'n_aid_neg_{s}']} "
                        + (f"OR {fmt_console(r[f'odds_ratio_{s}'], '.2f')}"
                           if r[f"status_{s}"] == "tested" else "not tested")
                        for s in au.EFFECT_MODIFICATION_STRATA)
        res = (f"ratio {fmt_console(r['or_ratio'], '.2f')} P={r['p']:.4f} "
               f"Q={fmt_console(r['q'], '.4f')}" if r["status"] == "tested"
               else f"not tested: {r['reason']}")
        print(f"  {r[by]:<20} {r['row_label']:<46} {res}\n      [{cnt}]")


print_modification(sequence, "drug_class")
au.sub("Checks: ties excluded (no Q); exposure ever (same test, own FDR family)")
print_modification(seq_checks[seq_checks["fdr_role"] == "check"], "drug_class")
print_modification(seq_checks[seq_checks["exposure_definition"] == EXPOSURE_EVER], "drug_class")

# =============================================================================
# SECTION 5 -- SECONDARY DOMAIN DEFINITION CHECK
# =============================================================================
au.section("SECTION 5 -- SECONDARY DOMAIN DEFINITION CHECK")
differs_prev = [dom for dom in sorted(dom_ids["primary"])
                if dom_ids["primary"][dom] != dom_ids["secondary"][dom]]
key_cols = ["person_id", "code", "first_date"]
differs_time = []
for dom in sorted(au.DOMAINS_PRIMARY):
    a = first_dom["primary"].loc[first_dom["primary"]["code"] == dom, key_cols]
    b = first_dom["secondary"].loc[first_dom["secondary"]["code"] == dom, key_cols]
    if not a.reset_index(drop=True).equals(b.reset_index(drop=True)):
        differs_time.append(dom)
print(f"  Domains whose patient flags differ (prevalence): {', '.join(differs_prev) or 'none'}")
print(f"  Domains whose first documentation differs (timing): {', '.join(differs_time) or 'none'}")
SEC_COLUMNS = ["analysis", "drug_class", "code", "row_label", "status_primary",
               "status_secondary", "estimate_primary", "estimate_secondary", "estimate_change",
               "q_primary", "q_secondary", "became_significant", "lost_significance",
               "change_ge_small", "change_ge_large"]


def definition_check(df, analysis, differs, est_col, small, large):
    rows = []
    for x, _ in MODEL_CLASSES:
        for dom in differs:
            sel = (df["drug_class"] == x) & (df["code"] == dom)
            p_ = df[sel & (df["definition"] == "primary")].iloc[0]
            s_ = df[sel & (df["definition"] == "secondary")].iloc[0]
            both = p_["status"] == "fitted" and s_["status"] == "fitted"
            change = s_[est_col] - p_[est_col] if both else NAN
            sig_p = pd.notna(p_["q"]) and p_["q"] < Q_THRESHOLD
            sig_s = pd.notna(s_["q"]) and s_["q"] < Q_THRESHOLD
            r = {"analysis": analysis, "drug_class": x, "code": dom,
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
            print(f"  {analysis:<10} {x:<16} {dom:<10} "
                  f"{fmt_console(p_[est_col], '.2f')} -> {fmt_console(s_[est_col], '.2f')}  "
                  f"Q {fmt_console(p_['q'], '.4f')} -> {fmt_console(s_['q'], '.4f')}  "
                  f"[{'Q crossed ' + str(Q_THRESHOLD) if crossed else 'Q side unchanged'}; "
                  f"|change| >= {large}: {'yes' if r['change_ge_large'] else 'no'}, "
                  f">= {small}: {'yes' if r['change_ge_small'] else 'no'}]")
    return pd.DataFrame(rows, columns=SEC_COLUMNS)


sec_prev = definition_check(prev[prev["level"] == "domain"], "prevalence", differs_prev,
                            "odds_ratio", OR_CHANGE_SMALL, OR_CHANGE_LARGE)
sec_time = definition_check(timing[timing["level"] == "domain"], "timing", differs_time,
                            "coef", ESTIMATE_CHANGE_SMALL_YRS, ESTIMATE_CHANGE_LARGE_YRS)

# =============================================================================
# SECTION 6 -- EXPOSURE DESCRIPTION (CHECKS)
# =============================================================================
au.section("SECTION 6 -- EXPOSURE BY GROUP; AID+ VS AID- WITHIN SEX")


def grp(a, s=None):
    return pop[(pop["aid_status"] == a) & ((pop["female"] == s) if s is not None else True)]


by_group = []
for c in DESCRIBED_CLASSES:
    for a, s, lbl in GROUPS + [(1, None, "AID+"), (0, None, "AID-")]:
        g = grp(a, s)
        tc = g[f"timing_{c}"].value_counts()
        by_group.append({"drug_class": c, "row_label": CLASS_LABELS[c], "group": lbl,
                         "n": len(g), "ever": int(g[f"ever_{c}"].sum()),
                         "before_diagnosis": tc.get("before_diagnosis", 0),
                         "on_or_after_diagnosis": tc.get("on_or_after_diagnosis", 0),
                         "undated_only": tc.get("undated_only", 0), "never": tc.get("never", 0),
                         "within_ceiling_or_undated": int(g[c].sum())})
by_group = pd.DataFrame(by_group)

print("  AID+ vs AID- within sex: Fisher's exact test; conditional MLE odds ratio, exact 95% CI.")
WITHIN_SEX = [(f"ever_{c}", c, "ever") for c in DESCRIBED_CLASSES] \
    + [(f"before_dx_{c}", c, "before_diagnosis") for c in ("on_cs", "on_dmard", "on_biologic")]
within_sex = []
for col, c, measure in WITHIN_SEX:
    for sex, s in (("Female", 1), ("Male", 0)):
        ap, an = grp(1, s), grp(0, s)
        e1, e2 = int(ap[col].sum()), int(an[col].sum())
        r = {"drug_class": c, "measure": measure,
             "row_label": CLASS_LABELS[c] + (" (first record before the AD diagnosis date)"
                                             if measure == "before_diagnosis" else ""),
             "sex": sex, "with_aid_pos": e1, "n_aid_pos": len(ap), "with_aid_neg": e2,
             "n_aid_neg": len(an), **two_by_two_row(e1, len(ap), e2, len(an), "a record")}
        within_sex.append(r)
        print(f"  {c:<16} {measure:<17} {sex:<7} AID+ {e1}/{len(ap)}  AID- {e2}/{len(an)}  "
              + (f"OR {fmt_console(r['odds_ratio'], '.2f')} P={r['p']:.4f}"
                 if r["status"] == "tested" else f"not tested: {r['reason']}"))
within_sex = pd.DataFrame(within_sex)

au.sub("Documentation measures: corticosteroid exposed (ever) vs unexposed (Mann-Whitney U)")
DOC_MEASURES = [("age_at_entry", "Age at first note, years"),
                ("documentation_span_yrs", "Documentation span, years"),
                ("note_count", "Total note count"),
                ("note_count_predx", "Pre-diagnosis note count"),
                ("note_count_postdx", "Post-diagnosis note count")]
doc_rows = []
ex_, un_ = pop[pop["ever_on_cs"] == 1], pop[pop["ever_on_cs"] == 0]
for col, label in DOC_MEASURES:
    a, b = ex_[col].dropna(), un_[col].dropna()
    why = au.group_minimum_reason(len(a), len(b))
    p = NAN if why else float(mannwhitneyu(a, b, alternative="two-sided")[1])
    doc_rows.append({"measure": col, "row_label": label, "n_exposed": len(a),
                     "median_exposed": a.median(), "mean_exposed": a.mean(),
                     "n_unexposed": len(b), "median_unexposed": b.median(),
                     "mean_unexposed": b.mean(), "status": "not_tested" if why else "tested",
                     "reason": why, "p": p})
    print(f"  {label:<28} exposed {len(a):>6,} median {fmt_console(a.median(), '.1f'):>7}  "
          f"unexposed {len(b):>6,} median {fmt_console(b.median(), '.1f'):>7}  "
          + (f"P={p:.4f}" if not why else f"not tested: {why}"))
doc_df = pd.DataFrame(doc_rows)

au.sub("AD pharmacotherapy by AID x sex group")
ad_rows = []
for a, s, lbl in GROUPS:
    g = grp(a, s)
    ch, nmda = int(g["ever_on_chei"].sum()), int(g["ever_on_nmda_antagonist"].sum())
    either = int(((g["ever_on_chei"] == 1) | (g["ever_on_nmda_antagonist"] == 1)).sum())
    ad_rows.append({"group": lbl, "n": len(g), "cholinesterase_inhibitor": ch,
                    "nmda_receptor_antagonist": nmda, "either": either})
    print(f"  {lbl:<11} n {len(g):>6,}  ChEI {ch:>6,}  NMDA receptor antagonist {nmda:>6,}  "
          f"either {either:>6,}")
ad_df = pd.DataFrame(ad_rows)

au.sub("Medications within AID+ (descriptive, pooled across sex)")
aid_pos = grp(1)
pos_rows = [{"row_label": CLASS_LABELS[c], "item": f"ever_{c}",
             "n": int(aid_pos[f"ever_{c}"].sum()), "n_aid_pos": len(aid_pos)}
            for c in ("on_cs", "on_dmard", "on_biologic", "on_antimalarial")]
pos_rows.insert(1, {"row_label": "Corticosteroids (first record before the AD diagnosis date)",
                    "item": "before_dx_on_cs", "n": int(aid_pos["before_dx_on_cs"].sum()),
                    "n_aid_pos": len(aid_pos)})
for c1, c2 in combinations(["on_cs", "on_dmard", "on_biologic", "on_antimalarial"], 2):
    both = int(((aid_pos[f"ever_{c1}"] == 1) & (aid_pos[f"ever_{c2}"] == 1)).sum())
    pos_rows.append({"row_label": f"{CLASS_LABELS[c1]} and {CLASS_LABELS_LOWER[c2]}",
                     "item": f"ever_{c1}+ever_{c2}", "n": both, "n_aid_pos": len(aid_pos)})
pos_df = pd.DataFrame(pos_rows)
for _, r in pos_df.iterrows():
    print(f"  {r['row_label']:<62} {r['n']:>6,} of {r['n_aid_pos']:,}")

# =============================================================================
# SECTION 7 -- SOURCE DATA
# =============================================================================
au.section("SECTION 7 -- SOURCE DATA")
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
    return pd.DataFrame(rows + list(extra), columns=["parameter", "value"])


def group_sizes(col_fmt):
    """Patients in each AID x exposure group per class (col_fmt: the exposure
    column of class c)."""
    out = []
    for x, _ in MODEL_CLASSES:
        for a, e, lbl in CELLS:
            out.append((f"n_{lbl}_{x}", int(((pop["aid_status"] == a)
                                              & (pop[col_fmt.format(c=x)] == e)).sum())))
    return out


def rows_not_shown(df, keys):
    out = []
    for x, _ in MODEL_CLASSES:
        m = df[df["drug_class"] == x]
        for k, sel in keys:
            mm = m[sel(m)]
            out.append((f"n_rows_{x}{k}", len(mm)))
            out.append((f"n_rows_not_fitted_{x}{k}", int((mm["status"] != "fitted").sum())))
    return out


PREV_SHEET = ["drug_class", "level", "code", "domain", "row_label", "fdr_role"] + CELL_EVENTS \
    + CELL_N + ["n_aid_pos", "n_aid_neg"] + FIT_COLUMNS[:6] + ["odds_ratio"] + FIT_COLUMNS[6:]
prev_primary = prev[prev["definition"] == "primary"]
au.write_workbook(os.path.join(SOURCE_DATA_DIR, PREVALENCE_FILE), {
    "interactions": prev_primary[PREV_SHEET],
    "secondary_definition": sec_prev,
    "parameters": parameters(
        group_sizes("{c}")
        + [(f"fdr_family_size_{x}_{lvl}",
            family_size(prev_primary[(prev_primary["drug_class"] == x)
                                     & (prev_primary["level"] == lvl)]))
           for x, _ in MODEL_CLASSES for lvl in ("domain", "category")]
        + rows_not_shown(prev_primary, [("_domain", lambda m: m["level"] == "domain"),
                                        ("_category", lambda m: m["level"] == "category")])
        + [("or_change_small", OR_CHANGE_SMALL), ("or_change_large", OR_CHANGE_LARGE)]),
})

TIME_SHEET = ["drug_class", "level", "code", "domain", "row_label", "fdr_role"] + CELL_N \
    + ["n_aid_pos", "n_aid_neg"] + FIT_COLUMNS
time_primary = timing[timing["definition"] == "primary"]
time_dom = time_primary[time_primary["level"] == "domain"]
time_cat = time_primary[time_primary["level"] == "category"]
au.write_workbook(os.path.join(SOURCE_DATA_DIR, TIMING_FILE), {
    "interactions": time_primary[TIME_SHEET],
    "secondary_definition": sec_time,
    "parameters": parameters(
        [(f"fdr_family_size_{x}", family_size(time_dom[time_dom["drug_class"] == x]))
         for x, _ in MODEL_CLASSES]
        + rows_not_shown(time_dom, [("", lambda m: m["code"].notna())])
        + [("estimate_change_small_yrs", ESTIMATE_CHANGE_SMALL_YRS),
           ("estimate_change_large_yrs", ESTIMATE_CHANGE_LARGE_YRS),
           ("linearity_shift_years", au.LINEARITY_SHIFT_YEARS)]
        + [(f"fdr_family_size_{x}_category",
            family_size(time_cat[time_cat["drug_class"] == x])) for x, _ in MODEL_CLASSES]
        + rows_not_shown(time_cat, [("_category", lambda m: m["code"].notna())])),
})

STRATUM_COLUMNS = [f"{k}_{s}" for s in au.EFFECT_MODIFICATION_STRATA for k in (
    "n_aid_pos", "inverted_aid_pos", "tied_aid_pos", "n_aid_neg", "inverted_aid_neg",
    "tied_aid_neg", "status", "reason", "or_estimable", "odds_ratio", "ci_low", "ci_high", "p")]
SEQ_SHEET = ["drug_class", "floor_yrs", "first", "second", "row_label", "fdr_role"] \
    + STRATUM_COLUMNS + ["status", "reason", "ratio_estimable", "or_ratio", "p", "q",
                         "fdr_family_size"]
au.write_workbook(os.path.join(SOURCE_DATA_DIR, SEQUENCE_FILE), {
    "pairs": sequence[SEQ_SHEET],
    "parameters": parameters(
        [(f"fdr_family_size_{x}", family_size(sequence[sequence["drug_class"] == x]))
         for x, _ in MODEL_CLASSES]
        + [(f"n_rows_not_tested_{x}",
            int((sequence[sequence["drug_class"] == x]["status"] != "tested").sum()))
           for x, _ in MODEL_CLASSES]
        + [("predx_span_floor_yrs", SEQUENCE_FLOOR_YRS)]),
})

au.write_workbook(os.path.join(SOURCE_DATA_DIR, CHECKS_FILE), {
    "prevalence_models": prev_models,
    "prevalence_secondary": prev[prev["definition"] == "secondary"],
    "linearity_prevalence": lin_prev,
    "corticosteroids_before_dx": predx_cs,
    "timing_models": time_models,
    "linearity_timing": lin_time,
    "sequence_checks": seq_checks[["exposure_definition", "sample"] + SEQ_SHEET],
    "exposure_by_group": by_group,
    "exposure_within_sex": within_sex,
    "documentation_measures": doc_df,
    "ad_pharmacotherapy": ad_df,
    "aid_pos_medications": pos_df,
    "age_ceiling": pd.DataFrame(ceiling_rows),
    "parameters": parameters(
        group_sizes("{c}") + [(f"n_ever_{lbl}_{x}", int(((pop["aid_status"] == a)
                                                         & (pop[f"ever_{x}"] == e)).sum()))
                              for x, _ in MODEL_CLASSES for a, e, lbl in CELLS]
        + [("medication_records", int(len(med))),
           ("medication_records_undated", int(med["exposure_date"].isna().sum())),
           ("medication_records_past_ceiling", int((~med["counts_in_presence"]).sum())),
           ("linearity_or_shift", au.LINEARITY_OR_SHIFT),
           ("linearity_or_min_diff", au.LINEARITY_OR_MIN_DIFF),
           ("linearity_shift_years", au.LINEARITY_SHIFT_YEARS)]),
})
print("\n" + au.SEP + "\n  COMPLETE\n" + au.SEP)