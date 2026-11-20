"""
symptom_prevalence.py
=====================
Does autoimmune disease (AID) change which symptom domains and categories
are documented in the clinical notes of patients with Alzheimer's disease
(AD)?

Computes every statistic and writes the source data for the prevalence
tables (one workbook per table, in SOURCE_DATA_DIR). The tables are rendered
from these workbooks by symptom_prevalence_tables.py; nothing here formats a
number.

Population and outcome
----------------------
- Analysis population: patients with at least one clinical note.
- A patient has a category when at least one symptom record of that category
  sits on a dated note at or below the patient's age ceiling: the
  sex-specific 90th percentile of age at last note among AID- patients.
  Records past the ceiling, and records on undated notes, are not counted;
  the patient stays in the denominator.
- A domain is present when any of its categories is present. Domains are
  analyzed under the primary definition and the secondary definition (sleep
  disturbance counted under autonomic instead of somatic); categories do not
  depend on the definition.

Models (one row per outcome and model)
--------------------------------------
- unweighted  : logit has_x ~ aid_status + female + age_at_dx
- weighted    : the same, weighted by the trimmed propensity-score weights
The prevalence models have no AID x sex interaction term.
Binomial GLM, logit link, HC0 sandwich standard errors; the weights enter as
var_weights (equal to the hand-computed IPW sandwich). The odds ratio and CI
come from the coefficient and SE (exp(coef -/+ crit x se)), checked.
A model is fitted only when the shared floor allows it (model_floor_reason:
10 events per term in the rarer outcome class, both AID groups present with
the outcome varying in each, both sexes present).
FDR: Benjamini-Hochberg within each family: domains per definition and
model; categories per model; the CNS-primary exclusion per definition. The
positive control (systemic inflammation autoimmune and its categories) is
shown with its P value, outside every family. "Other" is fitted for
information only: no P or Q.

Further analyses
----------------
- Linearity: the weighted model with age_at_dx squared added; the flag is
  an AID odds-ratio shift of at least 20% with |OR - 1| >= 0.05. Not a test.
- Known sex differences (aggression higher in male patients; depression and delusion
  higher in female patients): does the cohort reproduce them, and does AID change them?
  Female-vs-male ORs in all patients (pooled weighted model) and in AID- and
  AID+ (weighted model with an AID x sex term, fitted here for this table
  only), and the AID+ / AID- ratio (the test of change). BH within all patients,
  within AID-, within AID+ and within the ratios, each over the three
  symptoms. Each row records whether its OR points the reported way.
- CNS-primary AID excluded: the weighted model per domain after excluding
  AID+ patients whose AID primarily affects the central nervous system; the
  full-cohort PS weights are kept, and the balance of the two PS covariates
  is recomputed in the restricted cohort.
- Secondary domain definition: for each domain whose patient flags differ
  between the definitions, the change in odds ratio and whether Q crossed
  0.05, per model (printed; written for the caption).

Inputs: the patient, note, symptom-record, AID-subtype and propensity-score
weight tables (see PATHS). Run after propensity_score.py.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm
from statsmodels.genmod.families import Binomial

import analysis_utils as au

# =============================================================================
# PATHS (relative to the data folder the script is run from)
# =============================================================================
COHORT_PATIENTS_PATH = ""  # patient table: AID status, sex, key dates (birth, AD diagnosis, death), analysis-population flag and coded diagnoses (DLB, vascular dementia, PD, mixed dementia, hypertension)
COHORT_NOTES_PATH = ""  # note table: one row per clinical note, with its date (may be missing) and whether it is a cognitive specialty note
SYMPTOM_RECORDS_PATH = ""  # symptom table: one row per documented symptom, with its note and its category (56-category schema)
AID_SUBTYPES_PATH = ""  # AID-subtype table: whether each patient has an AID that primarily affects the central nervous system
PS_WEIGHTS_PATH = ""  # propensity-score weights, one per analysis patient (written by propensity_score.py)
SOURCE_DATA_DIR = ""  # folder for the source-data workbooks written here

WEIGHTED_FILE = "table_symptom_prevalence_weighted.xlsx"
UNWEIGHTED_FILE = "table_symptom_prevalence_unweighted.xlsx"
SEX_DIFFERENCES_FILE = "table_known_sex_differences.xlsx"
CNS_FILE = "table_symptom_prevalence_cns_excluded.xlsx"
CHECKS_FILE = "checks_symptom_prevalence.xlsx"

# =============================================================================
# SETTINGS
# =============================================================================
CRIT = float(norm.ppf(0.975))      # critical value of the 95% CI (z)
Q_THRESHOLD = 0.05                 # secondary-definition check: Q crossing
OR_CHANGE_SMALL = 0.10             # secondary-definition check: |OR change|
OR_CHANGE_LARGE = 0.20
WCOL = "ps_weight_trimmed"

FORMULA_POOLED = "has_x ~ aid_status + female + age_at_dx"
# the AID x sex model is used only by the known sex differences (Section 4)
FORMULA_INTERACTION = "has_x ~ aid_status + female + aid_status:female + age_at_dx"
FORMULA_QUADRATIC = FORMULA_POOLED + " + age_at_dx_sq"
TERM_AID = "aid_status"
TERM_INTERACTION = "aid_status:female"

# (model, formula, reported term, weighted, interaction floor)
MODELS = [("unweighted", FORMULA_POOLED, TERM_AID, False, False),
          ("weighted", FORMULA_POOLED, TERM_AID, True, False)]
DEFINITIONS = ["primary", "secondary"]

# Known sex differences: (category, sex reported to have more)
SEX_DIFFERENCE_ROWS = [("aggression", "Male"), ("depression", "Female"),
                       ("delusion", "Female")]

BALANCE_VARIABLES = [("age_at_entry", "Age at first note"),
                     ("log_note_count_predx", "Log pre-diagnosis note count")]

GROUPS = [(1, 1, "AID+ female"), (1, 0, "AID+ male"), (0, 1, "AID- female"), (0, 0, "AID- male")]
COUNT_COLUMNS = ["events_aid_pos", "n_aid_pos", "events_aid_neg", "n_aid_neg",
                 "events_aid_pos_female", "n_aid_pos_female",
                 "events_aid_pos_male", "n_aid_pos_male",
                 "events_aid_neg_female", "n_aid_neg_female",
                 "events_aid_neg_male", "n_aid_neg_male"]
FIT_COLUMNS = ["n_model", "status", "reason", "coef", "se", "crit", "odds_ratio",
               "ci_low", "ci_high", "p", "fdr_role", "q", "fdr_family_size"]
NAN = float("nan")


def n_terms(formula):
    return len([t for t in formula.split("~", 1)[1].split("+") if t.strip()])


# =============================================================================
# MODEL FITS
# =============================================================================
def fit_logit(formula, d, terms, weighted, interaction):
    """Fits one GLM when the floor allows it. Returns {"status", "reason",
    "n_model"} plus, for each term in `terms`, coef / se / odds_ratio / CI / p."""
    out = {"n_model": len(d), "status": "not_fitted", "reason": ""}
    for t in terms:
        out[t] = {"coef": NAN, "se": NAN, "odds_ratio": NAN, "ci_low": NAN,
                  "ci_high": NAN, "p": NAN}
    why = au.model_floor_reason(d["has_x"], d["aid_status"], n_terms(formula),
                                female=d["female"] if interaction else None,
                                sex=d["female"])
    if why:
        out["reason"] = why
        return out
    try:
        kw = {"var_weights": d[WCOL]} if weighted else {}
        mod = smf.glm(formula, data=d, family=Binomial(), **kw).fit(cov_type="HC0", disp=False)
    except Exception as e:
        out.update(status="failed", reason=f"fit error: {type(e).__name__}")
        return out
    for t in terms:
        if t not in mod.params.index:
            out.update(status="failed", reason=f"term {t} not estimated")
            return out
    for t in terms:
        b, se = float(mod.params[t]), float(mod.bse[t])
        ci = mod.conf_int().loc[t]
        out[t] = {"coef": b, "se": se, "odds_ratio": float(np.exp(b)),
                  "ci_low": float(np.exp(ci[0])), "ci_high": float(np.exp(ci[1])),
                  "p": float(mod.pvalues[t])}
    out["status"] = "fitted"
    return out


def model_row(d, model, formula, term, weighted, interaction):
    f = fit_logit(formula, d, [term], weighted, interaction)
    return {"model": model, "n_model": f["n_model"], "status": f["status"],
            "reason": f["reason"], "crit": CRIT, **f[term]}


def counts(d):
    out = {}
    for arm, a in (("aid_pos", 1), ("aid_neg", 0)):
        g = d[d["aid_status"] == a]
        out[f"events_{arm}"] = int(g["has_x"].sum())
        out[f"n_{arm}"] = len(g)
        for sx, lbl in ((1, "female"), (0, "male")):
            gs = g[g["female"] == sx]
            out[f"events_{arm}_{lbl}"] = int(gs["has_x"].sum())
            out[f"n_{arm}_{lbl}"] = len(gs)
    return out


def fdr_role(code, level):
    if level == "category" and code in au.NOT_TESTED_CATEGORIES:
        return "not_tested"
    controls = au.POSITIVE_CONTROL_DOMAINS if level == "domain" else au.POSITIVE_CONTROL_CATEGORIES
    return "positive_control" if code in controls else "tested"


def add_fdr(df, family_cols):
    """BH q within each family over fitted, tested rows; P removed from rows
    that are not tested ("other")."""
    df = df.copy()
    df.loc[df["fdr_role"] == "not_tested", "p"] = NAN
    df["q"] = NAN
    df["fdr_family_size"] = 0
    for _, idx in df.groupby(family_cols).groups.items():
        m = df.loc[idx]
        tested = m.index[(m["status"] == "fitted") & (m["fdr_role"] == "tested")
                         & m["p"].notna()]
        if len(tested):
            df.loc[tested, "q"] = au.bh_fdr(df.loc[tested, "p"].values)
            df.loc[tested, "fdr_family_size"] = len(tested)
    return df


def check_plugins(df, label):
    f = df[df["status"] == "fitted"]
    for col, val in (("odds_ratio", np.exp(f["coef"])),
                     ("ci_low", np.exp(f["coef"] - f["crit"] * f["se"])),
                     ("ci_high", np.exp(f["coef"] + f["crit"] * f["se"]))):
        if not np.allclose(f[col], val, rtol=1e-9, atol=0):
            raise RuntimeError(f"{label}: {col} does not match coef / se")


def print_models(df, label):
    for model in ("weighted", "unweighted"):
        m = df[df["model"] == model].sort_values("q", na_position="last")
        if m.empty:
            continue
        print(f"\n  {label} -- {model}")
        print(f"  {'':<38} {'OR':>7} {'95% CI':>15} {'P':>9} {'Q':>8}  status")
        for _, r in m.iterrows():
            fitted = r["status"] == "fitted"
            ci = f"[{r['ci_low']:.2f}-{r['ci_high']:.2f}]" if fitted else ""
            q = f"{r['q']:.4f}" if pd.notna(r["q"]) else "n/a"
            tag = "" if fitted else f"{r['status']}: {r['reason']}"
            if fitted and r["fdr_role"] != "tested":
                tag += f"  [{r['fdr_role']}]"
            print(f"  {r['code']:<38} {r['odds_ratio']:>7.2f} {ci:>15} {r['p']:>9.4f} "
                  f"{q:>8}  {tag}")


def flag_columns(denom, definition, domain):
    return [f"cat__{c}" for c in au.DOMAIN_DEFINITIONS[definition][domain]]


def outcome_frame(denom, definition=None, domain=None, category=None):
    if category is not None:
        y = denom[f"cat__{category}"]
    else:
        y = denom[flag_columns(denom, definition, domain)].max(axis=1)
    return denom.assign(has_x=y.astype(int))


# =============================================================================
# SECTION 1 -- POPULATION, WEIGHTS, AGE CEILING
# =============================================================================
au.section("SECTION 1 -- POPULATION, WEIGHTS, AGE CEILING")
pop, notes = au.load_population(
    COHORT_PATIENTS_PATH, COHORT_NOTES_PATH,
    extra_columns=["coded_dementia_with_lewy_bodies", "coded_vascular_dementia",
                   "coded_parkinsons_disease", "coded_mixed_dementia",
                   "coded_hypertension"])
pop = au.load_ps_weights(PS_WEIGHTS_PATH, pop)
pop = au.load_aid_subtypes(AID_SUBTYPES_PATH, pop)
pop["age_at_dx_sq"] = pop["age_at_dx"] ** 2
pop["log_note_count_predx"] = np.log1p(pop["note_count_predx"])

ceil = au.p90_age_ceiling(pop)
print(f"\n  Age ceiling, female (yr): {ceil['ceiling_female']:.2f}  "
      f"(AID- reference n = {ceil['n_ref_female']:,})")
print(f"  Age ceiling, male (yr)  : {ceil['ceiling_male']:.2f}  "
      f"(AID- reference n = {ceil['n_ref_male']:,})")
past_all = ~au.within_age_ceiling(pop["age_at_entry"],
                                  pop["person_id"].map(ceil["ceiling_by_person"]))
print("\n  Patients whose first dated note is past their ceiling (no record counts):")
past_rows = []
for aid, sx, lbl in GROUPS:
    g = (pop["aid_status"] == aid) & (pop["female"] == sx)
    au.count_line(lbl, (past_all & g).sum(), g.sum())
    past_rows.append({"group": lbl, "aid_status": aid, "female": sx,
                      "n_patients": int(g.sum()), "n_past_ceiling": int((past_all & g).sum())})

# -----------------------------------------------------------------------------
au.sub("1a. Characteristics by AID x sex (console)")
# -----------------------------------------------------------------------------
def describe(series, label):
    s = pd.to_numeric(pd.Series(series), errors="coerce").dropna()
    if len(s) < au.MIN_GROUP_N:
        return f"  {label:<40}  n={len(s)} -- below the floor of {au.MIN_GROUP_N}"
    q = s.quantile([0.10, 0.25, 0.50, 0.75, 0.90])
    return (f"  {label:<40}  n={len(s):>5,}  med={s.median():>7.1f}  mean={s.mean():>7.1f}  "
            f"sd={s.std():>6.1f}  p10={q[0.10]:>6.1f}  p25={q[0.25]:>5.1f}  "
            f"p75={q[0.75]:>5.1f}  p90={q[0.90]:>5.1f}")


for aid, sx, lbl in GROUPS:
    g = pop[(pop["aid_status"] == aid) & (pop["female"] == sx)]
    print(f"\n  {lbl}  (n={len(g):,})")
    for col, name in [("age_at_dx", "Age at AD diagnosis (yr)"),
                      ("age_at_entry", "Age at first note (yr)"),
                      ("age_at_last_note", "Age at last note (yr)"),
                      ("documentation_span_yrs", "Documentation span (yr)"),
                      ("note_count", "Total note count"),
                      ("note_count_predx", "Pre-diagnosis note count"),
                      ("note_count_postdx", "Post-diagnosis note count"),
                      ("cog_spec_note_count", "Cognitive specialty note count")]:
        print(describe(g[col], f"    {name}"))
    for flag, name in [("death_flag", "Death documented"),
                       ("has_cns_primary_aid", "CNS-primary AID"),
                       ("coded_dementia_with_lewy_bodies", "Coded DLB"),
                       ("coded_vascular_dementia", "Coded VD"),
                       ("coded_parkinsons_disease", "Coded PD"),
                       ("coded_mixed_dementia", "Coded mixed dementia"),
                       ("coded_hypertension", "Hypertension (any documented)")]:
        print(f"    {name:<35} {g[flag].mean() * 100:.1f}%")

# =============================================================================
# SECTION 2 -- SYMPTOM RECORDS -> PRESENCE FLAGS
# =============================================================================
au.section("SECTION 2 -- SYMPTOM RECORDS -> PRESENCE FLAGS")
rec = au.load_symptom_records(SYMPTOM_RECORDS_PATH, pop, notes)
any_record_ids = set(rec["person_id"].astype(int))
dated = rec[rec["note_date"].notna()].merge(pop[["person_id", "birth_date"]],
                                            on="person_id", how="left", validate="many_to_one")
dated["age_at_record"] = au.years_between(dated["note_date"], dated["birth_date"])
keep = au.within_age_ceiling(dated["age_at_record"],
                             dated["person_id"].map(ceil["ceiling_by_person"]))
au.count_line("dated records removed by the age ceiling", (~keep).sum(), len(dated))
kept = dated[keep]
au.count_line("records counted", len(kept))

cat_flags = (kept.groupby(["person_id", "category"]).size().unstack(fill_value=0)
             .clip(upper=1).reindex(columns=au.SCHEMA_CATEGORIES, fill_value=0)
             .astype(int).add_prefix("cat__"))
denom = pop.set_index("person_id").join(cat_flags, how="left")
CAT_COLS = [f"cat__{c}" for c in au.SCHEMA_CATEGORIES]
denom[CAT_COLS] = denom[CAT_COLS].fillna(0).astype(int)
denom = denom.reset_index()
if len(denom) != len(pop) or denom["person_id"].duplicated().any():
    raise RuntimeError("the denominator is not one row per population patient")

# -----------------------------------------------------------------------------
au.sub("1b. Denominator and record reconciliation (console)")
# -----------------------------------------------------------------------------
mapped = [f"cat__{c}" for c in au.SCHEMA_CATEGORIES if c not in au.UNASSIGNED_CATEGORIES]
any_raw = denom["person_id"].astype(int).isin(any_record_ids)
any_mapped = denom[mapped].max(axis=1) == 1
print(f"  {'Group':<14} {'n_total':>8} {'zero_raw':>9} {'raw_only':>9} {'mapped':>8}")
for aid, sx, lbl in GROUPS + [(None, None, "Total")]:
    g = (pd.Series(True, index=denom.index) if aid is None
         else (denom["aid_status"] == aid) & (denom["female"] == sx))
    print(f"  {lbl:<14} {int(g.sum()):>8,} {int((g & ~any_raw).sum()):>9,} "
          f"{int((g & any_raw & ~any_mapped).sum()):>9,} {int((g & any_mapped).sum()):>8,}")
print("    zero_raw: no symptom record at all; raw_only: records exist but none is a "
      "domain-mapped, dated record within the ceiling; mapped: at least one such record.")

# =============================================================================
# SECTION 3 -- DOMAIN AND CATEGORY MODELS, LINEARITY
# =============================================================================
au.section("SECTION 3 -- DOMAIN AND CATEGORY MODELS")


def linearity_row(d):
    lin = fit_logit(FORMULA_POOLED, d, [TERM_AID], True, False)
    if lin["status"] != "fitted":
        return {"status": "not_run", "reason": f"main model not fitted: {lin['reason']}",
                "odds_ratio_linear": NAN, "odds_ratio_quadratic": NAN,
                "or_shift_pct": NAN, "coef_age_squared": NAN, "flag": 0}
    quad = fit_logit(FORMULA_QUADRATIC, d, [TERM_AID, "age_at_dx_sq"], True, False)
    if quad["status"] != "fitted":
        return {"status": "not_run", "reason": f"quadratic model not fitted: {quad['reason']}",
                "odds_ratio_linear": lin[TERM_AID]["odds_ratio"], "odds_ratio_quadratic": NAN,
                "or_shift_pct": NAN, "coef_age_squared": NAN, "flag": 0}
    o1, o2 = lin[TERM_AID]["odds_ratio"], quad[TERM_AID]["odds_ratio"]
    shift = abs(o2 - o1) / o1
    flag = int(shift >= au.LINEARITY_OR_SHIFT and abs(o1 - 1) >= au.LINEARITY_OR_MIN_DIFF)
    return {"status": "run", "reason": "", "odds_ratio_linear": o1,
            "odds_ratio_quadratic": o2, "or_shift_pct": shift * 100,
            "coef_age_squared": quad["age_at_dx_sq"]["coef"], "flag": flag}


au.sub("3a. Domains (primary and secondary definitions)")
dom_rows, lin_rows = [], []
for defn in DEFINITIONS:
    for dom in sorted(au.DOMAIN_DEFINITIONS[defn]):
        d = outcome_frame(denom, definition=defn, domain=dom)
        base = {"definition": defn, "code": dom, "row_label": au.DOMAIN_LABELS[dom],
                "fdr_role": fdr_role(dom, "domain"), **counts(d)}
        for model, formula, term, weighted, inter in MODELS:
            dom_rows.append({**base, **model_row(d, model, formula, term, weighted, inter)})
        lin_rows.append({"level": "domain", "definition": defn, "code": dom,
                         "row_label": au.DOMAIN_LABELS[dom], **linearity_row(d)})
domain_df = add_fdr(pd.DataFrame(dom_rows), ["definition", "model"])
check_plugins(domain_df, "domain models")

au.sub("3b. Categories")
c2d = au.category_domain("primary")
cat_rows = []
for cat in au.SCHEMA_CATEGORIES:
    d = outcome_frame(denom, category=cat)
    base = {"code": cat, "domain": c2d[cat], "row_label": au.category_row_label(cat),
            "fdr_role": fdr_role(cat, "category"), **counts(d)}
    for model, formula, term, weighted, inter in MODELS:
        cat_rows.append({**base, **model_row(d, model, formula, term, weighted, inter)})
    lin_rows.append({"level": "category", "definition": "", "code": cat,
                     "row_label": au.category_row_label(cat), **linearity_row(d)})
category_df = add_fdr(pd.DataFrame(cat_rows), ["model"])
check_plugins(category_df, "category models")
linearity_df = pd.DataFrame(lin_rows)

for defn in DEFINITIONS:
    print_models(domain_df[domain_df["definition"] == defn], f"Domains, {defn} definition")
print_models(category_df, "Categories")

run = linearity_df[linearity_df["status"] == "run"]
print(f"\n  Linearity check (weighted model + age at AD diagnosis squared): "
      f"{len(run)} of {len(linearity_df)} run, {len(linearity_df) - len(run)} not run")
flagged = run[run["flag"] == 1]
print(f"    flagged (AID OR shift >= {au.LINEARITY_OR_SHIFT:.0%} and |OR - 1| >= "
      f"{au.LINEARITY_OR_MIN_DIFF}): {len(flagged)}"
      + ("" if flagged.empty else " -- " + "; ".join(
          f"{r['level']} {r['definition']} {r['code']} ({r['or_shift_pct']:.1f}%)".replace("  ", " ")
          for _, r in flagged.sort_values("or_shift_pct", ascending=False).iterrows())))
if len(run):
    print(f"    largest shift: {run['or_shift_pct'].max():.1f}%")

# =============================================================================
# SECTION 4 -- KNOWN SEX DIFFERENCES
# =============================================================================
au.section("SECTION 4 -- KNOWN SEX DIFFERENCES")
# Does the cohort reproduce a sex difference reported in the literature, and
# does AID change it? Female-vs-male ORs on one scale:
#   all_patients : exp(b_female) of the pooled weighted model (the weighted
#                  prevalence model, adjusted for AID and age at AD diagnosis)
#   aid_neg      : exp(b_female) of the weighted interaction model
#   aid_pos      : exp(b_female + b_interaction) of the same model
#   ratio        : exp(b_interaction) = aid_pos / aid_neg, the test of change
# BH within each of all_patients, aid_neg, aid_pos and ratio over the three
# symptoms. Each row records whether its OR points the way the literature
# reports.
Q_FAMILIES = ["all_patients", "aid_neg", "aid_pos", "ratio"]


def contrast_result(coef, var):
    se = float(np.sqrt(var))
    return {"coef": float(coef), "se": se, "odds_ratio": float(np.exp(coef)),
            "ci_low": float(np.exp(coef - CRIT * se)), "ci_high": float(np.exp(coef + CRIT * se)),
            "p": float(2 * norm.sf(abs(coef / se)))}


def weighted_fit(formula, d, interaction):
    why = au.model_floor_reason(d["has_x"], d["aid_status"], n_terms(formula),
                                female=d["female"] if interaction else None,
                                sex=d["female"])
    if why:
        return "not_fitted", why, None
    try:
        mod = smf.glm(formula, data=d, family=Binomial(),
                      var_weights=d[WCOL]).fit(cov_type="HC0", disp=False)
    except Exception as e:
        return "failed", f"fit error: {type(e).__name__}", None
    return "fitted", "", mod


sex_rows = []
for cat, expected in SEX_DIFFERENCE_ROWS:
    d = outcome_frame(denom, category=cat)
    st_p, why_p, pooled = weighted_fit(FORMULA_POOLED, d, False)
    st_i, why_i, inter = weighted_fit(FORMULA_INTERACTION, d, True)
    res = {}
    if pooled is not None:
        res["all_patients"] = contrast_result(pooled.params["female"],
                                              pooled.cov_params().loc["female", "female"])
    if inter is not None:
        b, v = inter.params, inter.cov_params()
        f, i = "female", TERM_INTERACTION
        res["aid_neg"] = contrast_result(b[f], v.loc[f, f])
        res["aid_pos"] = contrast_result(b[f] + b[i], v.loc[f, f] + v.loc[i, i] + 2 * v.loc[f, i])
        res["ratio"] = contrast_result(b[i], v.loc[i, i])
    for contrast, aid in (("all_patients", "all"), ("aid_pos", 1), ("aid_neg", 0),
                          ("ratio", None)):
        status, why = (st_p, why_p) if contrast == "all_patients" else (st_i, why_i)
        r = {"code": cat, "row_label": au.CATEGORY_LABELS[cat], "expected_higher": expected,
             "contrast": contrast, "n_model": len(d), "status": status, "reason": why,
             "crit": CRIT, "coef": NAN, "se": NAN, "odds_ratio": NAN, "ci_low": NAN,
             "ci_high": NAN, "p": NAN,
             "events_female": NAN, "n_female": NAN, "events_male": NAN, "n_male": NAN}
        if aid is not None:
            g = d if aid == "all" else d[d["aid_status"] == aid]
            for sx, lbl in ((1, "female"), (0, "male")):
                gs = g[g["female"] == sx]
                r[f"events_{lbl}"], r[f"n_{lbl}"] = int(gs["has_x"].sum()), len(gs)
        if contrast in res:
            r.update(res[contrast])
        sex_rows.append(r)
sex_df = pd.DataFrame(sex_rows)
sex_df["q"] = NAN
sex_df["fdr_family_size"] = 0
for fam in Q_FAMILIES:
    idx = sex_df.index[(sex_df["contrast"] == fam) & (sex_df["status"] == "fitted")]
    if len(idx):
        sex_df.loc[idx, "q"] = au.bh_fdr(sex_df.loc[idx, "p"].values)
        sex_df.loc[idx, "fdr_family_size"] = len(idx)
want_female = sex_df["expected_higher"] == "Female"
sex_df["in_reported_direction"] = (
    (sex_df["contrast"] != "ratio") & (sex_df["status"] == "fitted")
    & np.where(want_female, sex_df["odds_ratio"] > 1, sex_df["odds_ratio"] < 1)).astype(int)
sex_df["reproduced"] = ((sex_df["in_reported_direction"] == 1)
                        & (sex_df["q"] < Q_THRESHOLD)).astype(int)
check_plugins(sex_df, "known sex differences")
for _, r in sex_df.iterrows():
    tag = (f"OR {r['odds_ratio']:.2f} [{r['ci_low']:.2f}-{r['ci_high']:.2f}]  P={r['p']:.4f}"
           + (f"  Q={r['q']:.4f}" if pd.notna(r["q"]) else "")
           + ("" if r["contrast"] == "ratio" else
              f"  reported direction: {'yes' if r['in_reported_direction'] else 'no'}")
           if r["status"] == "fitted" else f"{r['status']}: {r['reason']}")
    counts_txt = ("" if r["contrast"] == "ratio" else
                  f"female {int(r['events_female'])}/{int(r['n_female'])}  "
                  f"male {int(r['events_male'])}/{int(r['n_male'])}  ")
    print(f"  {r['code']:<11} {r['contrast']:<13} {counts_txt}{tag}")
for fam in ("all_patients", "aid_neg", "aid_pos"):
    rep = sex_df[(sex_df["contrast"] == fam) & (sex_df["reproduced"] == 1)]["code"].tolist()
    print(f"  Known difference reproduced (Q < {Q_THRESHOLD}, reported direction), "
          f"{fam}: {', '.join(rep) if rep else 'none'}")

# =============================================================================
# SECTION 5 -- CNS-PRIMARY AID EXCLUSION
# =============================================================================
au.section("SECTION 5 -- CNS-PRIMARY AID EXCLUSION")
restricted = denom[denom["has_cns_primary_aid"] == 0].copy()
au.count_line("AID+ patients excluded (CNS-primary AID)", len(denom) - len(restricted),
              int(denom["aid_status"].sum()))
print(f"  Remaining: AID+ {int(restricted['aid_status'].sum()):,}  "
      f"AID- {int((restricted['aid_status'] == 0).sum()):,}")

cns_rows = []
for defn in DEFINITIONS:
    for dom in sorted(au.DOMAIN_DEFINITIONS[defn]):
        d = outcome_frame(restricted, definition=defn, domain=dom)
        cns_rows.append({"definition": defn, "code": dom, "row_label": au.DOMAIN_LABELS[dom],
                         "fdr_role": fdr_role(dom, "domain"), **counts(d),
                         **model_row(d, "weighted", FORMULA_POOLED, TERM_AID, True, False)})
cns_df = add_fdr(pd.DataFrame(cns_rows), ["definition"])
check_plugins(cns_df, "CNS-primary exclusion")
full_w = domain_df[domain_df["model"] == "weighted"].set_index(["definition", "code"])
for c in ("status", "odds_ratio", "ci_low", "ci_high", "p", "q"):
    cns_df[f"all_patients_{c}"] = [full_w.loc[(a, b), c] for a, b in
                                   zip(cns_df["definition"], cns_df["code"])]
for defn in DEFINITIONS:
    print(f"\n  CNS-primary AID excluded, {defn} definition (weighted)")
    for _, r in cns_df[cns_df["definition"] == defn].iterrows():
        q = f"{r['q']:.4f}" if pd.notna(r["q"]) else "n/a"
        tag = "" if r["status"] == "fitted" else f"{r['status']}: {r['reason']}"
        print(f"  {r['code']:<38} {r['odds_ratio']:>7.2f} [{r['ci_low']:.2f}-{r['ci_high']:.2f}] "
              f"P={r['p']:.4f} Q={q}  (all patients {r['all_patients_odds_ratio']:.2f})  {tag}")


def arm_summary(g, col):
    u = pd.to_numeric(g[col], errors="coerce").dropna()
    w = g[[col, WCOL]].apply(pd.to_numeric, errors="coerce").dropna()
    mw = np.average(w[col], weights=w[WCOL])
    return (len(u), u.mean(), u.std(ddof=1), mw,
            np.sqrt(np.average((w[col] - mw) ** 2, weights=w[WCOL])))


balance_rows = []
for stratum, sel in (("All", None), ("Female", 1), ("Male", 0)):
    g = restricted if sel is None else restricted[restricted["female"] == sel]
    g_all = denom if sel is None else denom[denom["female"] == sel]
    gp, gn = g[g["aid_status"] == 1], g[g["aid_status"] == 0]
    for col, label in BALANCE_VARIABLES:
        npos, mp, sp, mwp, swp = arm_summary(gp, col)
        nneg, mn, sn, mwn, swn = arm_summary(gn, col)
        r = {"stratum": stratum, "variable": col, "row_label": label,
             "n_aid_pos": npos, "n_aid_neg": nneg,
             "mean_aid_pos_unweighted": mp, "sd_aid_pos_unweighted": sp,
             "mean_aid_neg_unweighted": mn, "sd_aid_neg_unweighted": sn,
             "smd_unweighted": au.smd_from_groups(gp[col], gn[col]),
             "mean_aid_pos_weighted": mwp, "sd_aid_pos_weighted": swp,
             "mean_aid_neg_weighted": mwn, "sd_aid_neg_weighted": swn,
             "smd_weighted": au.smd_weighted_from_groups(gp, gn, col),
             "smd_weighted_all_patients": au.smd_weighted_from_groups(
                 g_all[g_all["aid_status"] == 1], g_all[g_all["aid_status"] == 0], col)}
        for kind in ("unweighted", "weighted"):
            num = r[f"mean_aid_pos_{kind}"] - r[f"mean_aid_neg_{kind}"]
            den = np.sqrt((r[f"sd_aid_pos_{kind}"] ** 2 + r[f"sd_aid_neg_{kind}"] ** 2) / 2)
            if not np.isclose(num / den, r[f"smd_{kind}"], rtol=0, atol=1e-9):
                raise RuntimeError(f"CNS balance {stratum} {col}: {kind} SMD mismatch")
        r["balance"] = au.balance_label(r["smd_weighted"])
        r["balance_all_patients"] = au.balance_label(r["smd_weighted_all_patients"])
        balance_rows.append(r)
        print(f"  Balance, {stratum:<6} {label:<30} SMD unweighted {r['smd_unweighted']:+.3f}  "
              f"weighted {r['smd_weighted']:+.3f}  (all patients, weighted "
              f"{r['smd_weighted_all_patients']:+.3f})")
balance_df = pd.DataFrame(balance_rows)
same = (balance_df["balance"] == balance_df["balance_all_patients"]).all()
print(f"  Balance band unchanged after exclusion in every row: {'yes' if same else 'no'}")

# =============================================================================
# SECTION 6 -- SECONDARY DOMAIN DEFINITION CHECK
# =============================================================================
au.section("SECTION 6 -- SECONDARY DOMAIN DEFINITION CHECK")
differs = []
for dom in sorted(au.DOMAINS_PRIMARY):
    a = outcome_frame(denom, "primary", dom)["has_x"]
    b = outcome_frame(denom, "secondary", dom)["has_x"]
    if (a != b).any():
        differs.append(dom)
print(f"  Domains whose patient flags differ between the definitions: "
      f"{', '.join(differs) if differs else 'none'}")


def definition_check(df, model_label):
    rows = []
    for dom in differs:
        p_ = df[(df["definition"] == "primary") & (df["code"] == dom)].iloc[0]
        s_ = df[(df["definition"] == "secondary") & (df["code"] == dom)].iloc[0]
        both = p_["status"] == "fitted" and s_["status"] == "fitted"
        change = s_["odds_ratio"] - p_["odds_ratio"] if both else NAN
        sig_p = pd.notna(p_["q"]) and p_["q"] < Q_THRESHOLD
        sig_s = pd.notna(s_["q"]) and s_["q"] < Q_THRESHOLD
        r = {"model": model_label, "code": dom, "row_label": au.DOMAIN_LABELS[dom],
             "status_primary": p_["status"], "status_secondary": s_["status"],
             "odds_ratio_primary": p_["odds_ratio"], "odds_ratio_secondary": s_["odds_ratio"],
             "or_change": change, "q_primary": p_["q"], "q_secondary": s_["q"],
             "became_significant": int(both and sig_s and not sig_p),
             "lost_significance": int(both and sig_p and not sig_s),
             "or_change_ge_small": int(both and abs(change) >= OR_CHANGE_SMALL),
             "or_change_ge_large": int(both and abs(change) >= OR_CHANGE_LARGE)}
        rows.append(r)
        print(f"  {model_label:<12} {dom:<12} OR {p_['odds_ratio']:.2f} -> {s_['odds_ratio']:.2f}"
              f"  Q {p_['q']:.4f} -> {s_['q']:.4f}  "
              f"[{'Q crossed ' + str(Q_THRESHOLD) if r['became_significant'] or r['lost_significance'] else 'Q side unchanged'}; "
              f"|OR change| >= {OR_CHANGE_LARGE}: {'yes' if r['or_change_ge_large'] else 'no'}, "
              f">= {OR_CHANGE_SMALL}: {'yes' if r['or_change_ge_small'] else 'no'}]")
    return pd.DataFrame(rows, columns=["model", "code", "row_label", "status_primary",
                                       "status_secondary", "odds_ratio_primary",
                                       "odds_ratio_secondary", "or_change", "q_primary",
                                       "q_secondary", "became_significant",
                                       "lost_significance", "or_change_ge_small",
                                       "or_change_ge_large"])


check_by_model = {m: definition_check(domain_df[domain_df["model"] == m], m)
                  for m in ("weighted", "unweighted")}
check_by_model["cns_excluded"] = definition_check(cns_df, "cns_excluded")

# =============================================================================
# SECTION 7 -- SOURCE DATA
# =============================================================================
au.section("SECTION 7 -- SOURCE DATA")
RESULT_COLUMNS = [c for c in FIT_COLUMNS if c != "fdr_role"]
DOMAIN_SHEET = ["code", "row_label", "fdr_role"] + COUNT_COLUMNS + RESULT_COLUMNS
CATEGORY_SHEET = ["code", "domain", "row_label", "fdr_role"] + COUNT_COLUMNS + RESULT_COLUMNS


def parameters(extra=()):
    rows = [("min_events_per_term", au.MIN_EVENTS_PER_TERM), ("min_group_n", au.MIN_GROUP_N),
            ("age_ceiling_quantile", au.AGE_CEILING_QUANTILE),
            ("age_ceiling_female", ceil["ceiling_female"]),
            ("age_ceiling_male", ceil["ceiling_male"]),
            ("age_ceiling_n_ref_female", ceil["n_ref_female"]),
            ("age_ceiling_n_ref_male", ceil["n_ref_male"]),
            ("q_threshold", Q_THRESHOLD), ("or_change_small", OR_CHANGE_SMALL),
            ("or_change_large", OR_CHANGE_LARGE)] + list(extra)
    return pd.DataFrame(rows, columns=["parameter", "value"])


def family_sizes(dom, cat=None):
    def size(df):
        s = df.loc[df["q"].notna(), "fdr_family_size"]
        return int(s.iloc[0]) if len(s) else 0
    out = [("fdr_family_size_domains", size(dom))]
    if cat is not None:
        out.append(("fdr_family_size_categories", size(cat)))
    return out


for model, path in (("weighted", WEIGHTED_FILE), ("unweighted", UNWEIGHTED_FILE)):
    dom = domain_df[(domain_df["model"] == model) & (domain_df["definition"] == "primary")]
    cat = category_df[category_df["model"] == model]
    au.write_workbook(os.path.join(SOURCE_DATA_DIR, path), {
        "domains": dom[DOMAIN_SHEET],
        "categories": cat[CATEGORY_SHEET],
        "secondary_definition": check_by_model[model],
        "parameters": parameters(family_sizes(dom, cat)),
    })

au.write_workbook(os.path.join(SOURCE_DATA_DIR, SEX_DIFFERENCES_FILE), {
    "sex_differences": sex_df[["code", "row_label", "expected_higher", "contrast",
                               "events_female", "n_female", "events_male", "n_male",
                               "n_model", "status", "reason", "coef", "se", "crit",
                               "odds_ratio", "ci_low", "ci_high", "p", "q",
                               "fdr_family_size", "in_reported_direction", "reproduced"]],
    "parameters": parameters(),
})

cns_primary = cns_df[cns_df["definition"] == "primary"]
au.write_workbook(os.path.join(SOURCE_DATA_DIR, CNS_FILE), {
    "domains": cns_primary[DOMAIN_SHEET],
    "balance": balance_df,
    "secondary_definition": check_by_model["cns_excluded"],
    "parameters": parameters(family_sizes(cns_primary) + [
        ("n_excluded_cns_primary", int(len(denom) - len(restricted))),
        ("n_aid_pos_all_patients", int(denom["aid_status"].sum())),
        ("balance_band_unchanged", int(same))]),
})

au.write_workbook(os.path.join(SOURCE_DATA_DIR, CHECKS_FILE), {
    "linearity": linearity_df,
    "secondary_definition_domains": domain_df[domain_df["definition"] == "secondary"][
        ["model"] + DOMAIN_SHEET],
    "secondary_definition_cns": cns_df[cns_df["definition"] == "secondary"][DOMAIN_SHEET],
    "age_ceiling": pd.DataFrame(past_rows),
    "parameters": parameters([("records_removed_by_ceiling", int((~keep).sum())),
                              ("records_counted", int(len(kept))),
                              ("records_on_undated_notes", int(rec["note_date"].isna().sum()))]),
})
print("\n" + au.SEP + "\n  COMPLETE\n" + au.SEP)