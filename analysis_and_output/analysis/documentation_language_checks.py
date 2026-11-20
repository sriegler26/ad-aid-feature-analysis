"""
documentation_language_checks.py
================================
Checks of how symptoms are written about in the clinical notes of patients
with and without autoimmune disease (AID): change and severity language
(worsening, improvement, a severe level), and explicit relative-time language
(statements that place a symptom a stated time before the note). Source data
only: two checks workbooks, no rendered table.

Change and severity language
----------------------------
Each domain-mapped symptom record carries whether it states worsening,
improvement, and the highest severity level stated. Records on undated notes
are not used; records past the patient's age ceiling (the sex-specific 90th
percentile of age at last note among AID- patients) are removed, as in every
presence analysis.
- Patients with at least one worsening / improvement / severe record, AID+
  vs AID-, within each sex: overall (denominator: patients with at least one
  kept record) and by domain (denominator: patients with the domain
  documented). Fisher's exact test, conditional MLE odds ratio, exact CI;
  gate two_by_two_reason. BH within each measure x level x sex (and
  definition); the systemic inflammation autoimmune domain is the positive
  control (P shown, outside the family). Secondary domain definition: the
  domain rows refitted under it; the change is printed and written. These
  unweighted rows are kept as the check.
- The same patients and denominators, propensity-score weighted: within
  each sex, logit has_measure ~ aid_status + age_at_dx (no sex term), a
  binomial GLM with ps_weight_trimmed as var_weights and HC0 sandwich
  standard errors; gate model_floor_reason. OR, 95% CI, P; BH within each
  measure x level x sex x definition, the positive control outside the
  family with its P. Secondary-definition check as above. Linearity check:
  each weighted model refitted with age_at_dx squared added; flagged when
  the AID odds ratio shifts by at least 20% with |OR - 1| >= 0.05 (not a
  test).
- Per-patient share of worsening and of severe records, Mann-Whitney U
  within sex; and the PS-weighted mean share by AID group within sex
  (descriptive, no test).
- Share of records before and after the AD diagnosis in the cognitive
  domains (before = strictly before the diagnosis day; descriptive, shown at
  >= 10 records).

Explicit relative-time language
-------------------------------
Each such record carries its kind and the implied offset from the note date
in whole days. Used here: dated records of a domain-mapped category.
deviation = offset / 365.25 years; age implied = ((note date - birth date)
in days + offset) / 365.25. First match per patient: deviation by kind.
Domains with >= 10 first-match patients in both AID groups. Matches on the
domain's own first documentation: AID+ vs AID- age at the note and age
implied, Mann-Whitney U (gate group_minimum_reason).

Inputs: the patient table, the note table, the symptom record table, the
change-language record table, the explicit-time record table and the
propensity-score weights (see PATHS).
"""

import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import fisher_exact, mannwhitneyu
from scipy.stats.contingency import odds_ratio
from statsmodels.genmod.families import Binomial

import analysis_utils as au

warnings.filterwarnings("ignore")

# =============================================================================
# PATHS (relative to the data folder the script is run from)
# =============================================================================
COHORT_PATIENTS_PATH = ""  # patient table: AID status, sex, key dates (birth, AD diagnosis, death) and analysis-population flag
COHORT_NOTES_PATH = ""  # note table: one row per clinical note, with its date and specialty
SYMPTOM_RECORDS_PATH = ""  # symptom record table: one row per documented symptom, with its note and symptom category
CHANGE_RECORDS_PATH = ""  # change-language record table: per symptom record, whether it states worsening or improvement, and its severity level
EXPLICIT_TIME_PATH = ""  # explicit-time record table: per symptom record with relative-time language, its kind and the implied offset from the note date
PS_WEIGHTS_PATH = ""  # propensity-score weight table: one row per analysis-population patient, with its trimmed weight (ps_weight_trimmed)
SOURCE_DATA_DIR = ""  # folder for the source-data workbooks written here

CHANGE_FILE = "checks_change_language.xlsx"
EXPLICIT_FILE = "checks_explicit_time.xlsx"

# =============================================================================
# SETTINGS
# =============================================================================
Q_THRESHOLD = 0.05                 # secondary-definition check: Q crossing
OR_CHANGE_SMALL = 0.10             # secondary-definition check: |OR change|
OR_CHANGE_LARGE = 0.20
DEFINITIONS = ["primary", "secondary"]
MEASURES = [("worsening", "a worsening record"), ("improvement", "an improvement record"),
            ("severe", "a severe record")]
COGNITIVE_DOMAINS = {
    "episodic_memory", "semantic_language", "working_memory_attention",
    "executive_function", "visuospatial", "orientation_reality_monitoring",
    "global_cognition", "self_awareness", "subjective_cognitive_burden",
}
NAN = float("nan")

# Weighted model (within sex, so no sex term)
WCOL = "ps_weight_trimmed"
FORMULA_WEIGHTED = "has_x ~ aid_status + age_at_dx"
N_TERMS_WEIGHTED = 2
TERM_AID = "aid_status"
FORMULA_QUADRATIC = FORMULA_WEIGHTED + " + age_at_dx_sq"
N_TERMS_QUADRATIC = 3


def exact_2x2(a, b, c, d):
    table = [[int(a), int(b)], [int(c), int(d)]]
    res = odds_ratio(table, kind="conditional")
    ci = res.confidence_interval(0.95)
    return float(res.statistic), float(ci.low), float(ci.high), float(fisher_exact(table)[1])


def mann_whitney(a, b):
    a, b = pd.Series(a).dropna().astype(float), pd.Series(b).dropna().astype(float)
    why = au.group_minimum_reason(len(a), len(b))
    if why:
        return "not_tested", why, NAN
    return "tested", "", float(mannwhitneyu(a, b, alternative="two-sided")[1])


def fmt_console(v, spec):
    return "n/a" if pd.isna(v) else format(v, spec)


def weighted_logit(d):
    """PS-weighted logit of has_x on AID and age at AD diagnosis (binomial GLM,
    var_weights, HC0), when model_floor_reason allows it. Returns the result
    columns of a patient_level row for the AID term."""
    out = {"odds_ratio": NAN, "ci_low": NAN, "ci_high": NAN, "or_estimable": 0, "p": NAN,
           "status": "not_fitted", "reason": ""}
    why = au.model_floor_reason(d["has_x"], d["aid_status"], N_TERMS_WEIGHTED)
    if why:
        out["reason"] = why
        return out
    try:
        mod = smf.glm(FORMULA_WEIGHTED, data=d, family=Binomial(),
                      var_weights=d[WCOL]).fit(cov_type="HC0", disp=False)
    except Exception as e:
        out.update(status="failed", reason=f"fit error: {type(e).__name__}")
        return out
    if TERM_AID not in mod.params.index:
        out.update(status="failed", reason=f"term {TERM_AID} not estimated")
        return out
    ci = mod.conf_int().loc[TERM_AID]
    orv, lo, hi = (float(np.exp(v)) for v in (mod.params[TERM_AID], ci[0], ci[1]))
    out.update(status="fitted", p=float(mod.pvalues[TERM_AID]))
    if np.isfinite([orv, lo, hi]).all() and orv > 0:
        out.update(odds_ratio=orv, ci_low=lo, ci_high=hi, or_estimable=1)
    return out


def linearity_row(d, lin):
    """Linearity check of one weighted model: refit with age_at_dx squared
    added; flag an AID odds-ratio shift of at least LINEARITY_OR_SHIFT with
    |OR - 1| >= LINEARITY_OR_MIN_DIFF. Not a test. The quadratic model needs
    the floor for its three terms. lin: the weighted_logit result."""
    out = {"status": "not_run", "reason": "", "odds_ratio_linear": NAN,
           "odds_ratio_quadratic": NAN, "or_shift_pct": NAN, "coef_age_squared": NAN, "flag": 0}
    if lin["status"] != "fitted":
        out["reason"] = f"main model not fitted: {lin['reason']}"
        return out
    if lin["or_estimable"] != 1:
        out["reason"] = "main model odds ratio not estimable"
        return out
    why = au.model_floor_reason(d["has_x"], d["aid_status"], N_TERMS_QUADRATIC)
    if why:
        out.update(reason=f"quadratic model not fitted: {why}", odds_ratio_linear=lin["odds_ratio"])
        return out
    try:
        mod = smf.glm(FORMULA_QUADRATIC, data=d, family=Binomial(),
                      var_weights=d[WCOL]).fit(cov_type="HC0", disp=False)
        o2 = float(np.exp(mod.params[TERM_AID]))
        c2 = float(mod.params["age_at_dx_sq"])
    except Exception as e:
        out.update(reason=f"quadratic model not fitted: fit error: {type(e).__name__}",
                   odds_ratio_linear=lin["odds_ratio"])
        return out
    o1 = lin["odds_ratio"]
    shift = abs(o2 - o1) / o1
    out.update(status="run", odds_ratio_linear=o1, odds_ratio_quadratic=o2,
               or_shift_pct=shift * 100, coef_age_squared=c2,
               flag=int(shift >= au.LINEARITY_OR_SHIFT and abs(o1 - 1) >= au.LINEARITY_OR_MIN_DIFF))
    return out


def definition_check_rows(df, ok_status, domains):
    """Secondary-definition check of one patient-level sheet: one row per
    measure x sex x domain in `domains` (columns of secondary_definition)."""
    out = []
    for meas, _ in MEASURES:
        for sex in ("Female", "Male"):
            for dom in domains:
                sel = (df["level"] == "domain") & (df["measure"] == meas) & (df["sex"] == sex) \
                    & (df["domain"] == dom)
                p_ = df[sel & (df["definition"] == "primary")].iloc[0]
                s_ = df[sel & (df["definition"] == "secondary")].iloc[0]
                both = (p_["status"] == ok_status and s_["status"] == ok_status
                        and p_["or_estimable"] == 1 and s_["or_estimable"] == 1)
                change = s_["odds_ratio"] - p_["odds_ratio"] if both else NAN
                sig_p = pd.notna(p_["q"]) and p_["q"] < Q_THRESHOLD
                sig_s = pd.notna(s_["q"]) and s_["q"] < Q_THRESHOLD
                out.append({"measure": meas, "sex": sex, "domain": dom,
                            "row_label": au.DOMAIN_LABELS[dom],
                            "status_primary": p_["status"], "status_secondary": s_["status"],
                            "odds_ratio_primary": p_["odds_ratio"],
                            "odds_ratio_secondary": s_["odds_ratio"], "or_change": change,
                            "q_primary": p_["q"], "q_secondary": s_["q"],
                            "became_significant": int(sig_s and not sig_p),
                            "lost_significance": int(sig_p and not sig_s),
                            "or_change_ge_small": int(both and abs(change) >= OR_CHANGE_SMALL),
                            "or_change_ge_large": int(both and abs(change) >= OR_CHANGE_LARGE)})
                print(f"  {meas:<12} {sex:<7} {dom:<10} OR "
                      f"{fmt_console(p_['odds_ratio'], '.2f')} -> "
                      f"{fmt_console(s_['odds_ratio'], '.2f')}  Q {fmt_console(p_['q'], '.4f')}"
                      f" -> {fmt_console(s_['q'], '.4f')}")
    return out


# =============================================================================
# SECTION 1 -- POPULATION AND CHANGE-LANGUAGE RECORDS
# =============================================================================
au.section("SECTION 1 -- POPULATION AND CHANGE-LANGUAGE RECORDS")
pop, notes = au.load_population(COHORT_PATIENTS_PATH, COHORT_NOTES_PATH)
pop = au.load_ps_weights(PS_WEIGHTS_PATH, pop)
ceil = au.p90_age_ceiling(pop)
print(f"\n  Age ceiling, female (yr): {ceil['ceiling_female']:.2f}; male: {ceil['ceiling_male']:.2f}")
cr = au.load_change_language_records(CHANGE_RECORDS_PATH, pop, notes)
cr = cr[cr["note_date"].notna()].merge(
    pop[["person_id", "aid_status", "female", "birth_date", "ad_diagnosis_date"]],
    on="person_id", how="left", validate="many_to_one")
cr["severe"] = (cr["severity_level"] == "severe").astype(int)
cr["age_at_record"] = au.years_between(cr["note_date"], cr["birth_date"])
keep = au.within_age_ceiling(cr["age_at_record"], cr["person_id"].map(ceil["ceiling_by_person"]))
au.count_line("dated records past the age ceiling (removed)", (~keep).sum(), len(cr))
cr = cr[keep].copy()
for defn in DEFINITIONS:
    cr[f"domain_{defn}"] = cr["category"].map(au.category_domain(defn))
print(f"  Records kept: {len(cr):,} ({cr['person_id'].nunique():,} patients)")
print("  Severity level of kept records: " + "  ".join(
    f"{l} {int((cr['severity_level'] == l).sum()):,}" for l in au.SEVERITY_LEVELS))

# =============================================================================
# SECTION 2 -- PATIENT LEVEL, AID+ VS AID-, WITHIN SEX
# =============================================================================
au.section("SECTION 2 -- PATIENTS WITH A WORSENING / IMPROVEMENT / SEVERE RECORD")
print("  Denominator: patients with at least one kept record (overall) or with the domain")
print("  documented (by domain). Fisher's exact test, conditional MLE OR, exact 95% CI.")
cols = [m for m, _ in MEASURES]
pt_any = cr.groupby("person_id")[cols].max().reset_index()
labels = pop[["person_id", "aid_status", "female"]]
rows = []
targets = [("overall", "primary", "all", pt_any)]
for defn in DEFINITIONS:
    pt_dom = cr.groupby(["person_id", f"domain_{defn}"])[cols].max().reset_index() \
        .rename(columns={f"domain_{defn}": "domain"})
    for dom in sorted(au.DOMAIN_DEFINITIONS[defn]):
        targets.append(("domain", defn, dom, pt_dom[pt_dom["domain"] == dom]))
for level, defn, dom, base in targets:
    base = base.merge(labels, on="person_id", how="left", validate="many_to_one")
    for meas, event in MEASURES:
        for sex, sel in (("Female", 1), ("Male", 0)):
            b = base[base["female"] == sel]
            ap, an = b[b["aid_status"] == 1], b[b["aid_status"] == 0]
            a, c = int(ap[meas].sum()), int(an[meas].sum())
            r = {"measure": meas, "level": level, "definition": defn, "domain": dom,
                 "row_label": "All records" if dom == "all" else au.DOMAIN_LABELS[dom],
                 "fdr_role": ("positive_control" if dom in au.POSITIVE_CONTROL_DOMAINS
                              else "tested"),
                 "sex": sex, "with_aid_pos": a, "n_aid_pos": len(ap), "with_aid_neg": c,
                 "n_aid_neg": len(an), "odds_ratio": NAN, "ci_low": NAN, "ci_high": NAN,
                 "or_estimable": 0, "p": NAN}
            why = au.two_by_two_reason(a, len(ap), c, len(an), event)
            if why:
                r.update(status="not_tested", reason=why)
            else:
                orv, lo, hi, p = exact_2x2(a, len(ap) - a, c, len(an) - c)
                r.update(status="tested", reason="", p=p)
                if np.isfinite(orv) and orv > 0:
                    r.update(odds_ratio=orv, ci_low=lo, ci_high=hi, or_estimable=1)
            rows.append(r)
chg = pd.DataFrame(rows)
chg["q"] = NAN
chg["fdr_family_size"] = 0
for _, idx in chg.groupby(["measure", "level", "definition", "sex"]).groups.items():
    m = chg.loc[idx]
    t = m.index[(m["status"] == "tested") & (m["fdr_role"] == "tested")]
    if len(t):
        chg.loc[t, "q"] = au.bh_fdr(chg.loc[t, "p"].values)
        chg.loc[t, "fdr_family_size"] = len(t)
for _, r in chg[chg["level"] == "overall"].iterrows():
    print(f"  {r['measure']:<12} {r['sex']:<7} AID+ {r['with_aid_pos']:,}/{r['n_aid_pos']:,}  "
          f"AID- {r['with_aid_neg']:,}/{r['n_aid_neg']:,}  "
          + (f"OR {fmt_console(r['odds_ratio'], '.2f')} [{fmt_console(r['ci_low'], '.2f')}-"
             f"{fmt_console(r['ci_high'], '.2f')}]  P={r['p']:.4f}  Q={r['q']:.4f}"
             if r["status"] == "tested" else f"not tested: {r['reason']}"))
print("\n  By domain, primary definition (Q < 0.05 listed; rows not tested with the reason):")
for meas, _ in MEASURES:
    for sex in ("Female", "Male"):
        m = chg[(chg["level"] == "domain") & (chg["definition"] == "primary")
                & (chg["measure"] == meas) & (chg["sex"] == sex)]
        f = m[m["status"] == "tested"]
        print(f"  {meas:<12} {sex:<7} tested {len(f):>3} of {len(m):>3}; Q < 0.05: "
              + (", ".join(f"{r['domain']} (OR {fmt_console(r['odds_ratio'], '.2f')}, "
                           f"Q={r['q']:.4f})" for _, r in f[f["q"] < Q_THRESHOLD].iterrows())
                 or "none"))
        for _, r in m[m["status"] != "tested"].iterrows():
            print(f"      not tested: {r['domain']:<34} {r['reason']}")

au.sub("Secondary domain definition check (domain rows whose patients differ)")
differs = []
for dom in sorted(au.DOMAINS_PRIMARY):
    a = set(cr.loc[cr["domain_primary"] == dom, "person_id"])
    b = set(cr.loc[cr["domain_secondary"] == dom, "person_id"])
    if a != b or not cr.loc[cr["domain_primary"] == dom].index.equals(
            cr.loc[cr["domain_secondary"] == dom].index):
        differs.append(dom)
print(f"  Domains whose records differ between the definitions: {', '.join(differs) or 'none'}")
sec_rows = []
for meas, _ in MEASURES:
    for sex in ("Female", "Male"):
        for dom in differs:
            sel = (chg["level"] == "domain") & (chg["measure"] == meas) & (chg["sex"] == sex) \
                & (chg["domain"] == dom)
            p_ = chg[sel & (chg["definition"] == "primary")].iloc[0]
            s_ = chg[sel & (chg["definition"] == "secondary")].iloc[0]
            both = (p_["status"] == "tested" and s_["status"] == "tested"
                    and p_["or_estimable"] == 1 and s_["or_estimable"] == 1)
            change = s_["odds_ratio"] - p_["odds_ratio"] if both else NAN
            sig_p = pd.notna(p_["q"]) and p_["q"] < Q_THRESHOLD
            sig_s = pd.notna(s_["q"]) and s_["q"] < Q_THRESHOLD
            r = {"measure": meas, "sex": sex, "domain": dom, "row_label": au.DOMAIN_LABELS[dom],
                 "status_primary": p_["status"], "status_secondary": s_["status"],
                 "odds_ratio_primary": p_["odds_ratio"], "odds_ratio_secondary": s_["odds_ratio"],
                 "or_change": change, "q_primary": p_["q"], "q_secondary": s_["q"],
                 "became_significant": int(sig_s and not sig_p),
                 "lost_significance": int(sig_p and not sig_s),
                 "or_change_ge_small": int(both and abs(change) >= OR_CHANGE_SMALL),
                 "or_change_ge_large": int(both and abs(change) >= OR_CHANGE_LARGE)}
            sec_rows.append(r)
            print(f"  {meas:<12} {sex:<7} {dom:<10} OR {fmt_console(p_['odds_ratio'], '.2f')} -> "
                  f"{fmt_console(s_['odds_ratio'], '.2f')}  Q {fmt_console(p_['q'], '.4f')} -> "
                  f"{fmt_console(s_['q'], '.4f')}")
sec_df = pd.DataFrame(sec_rows, columns=[
    "measure", "sex", "domain", "row_label", "status_primary", "status_secondary",
    "odds_ratio_primary", "odds_ratio_secondary", "or_change", "q_primary", "q_secondary",
    "became_significant", "lost_significance", "or_change_ge_small", "or_change_ge_large"])

au.sub("Propensity-score weighted: has_measure ~ aid_status + age_at_dx, within sex")
print("  Binomial GLM, ps_weight_trimmed as var_weights, HC0 SE; same patients and")
print("  denominators as the Fisher rows above (kept as the check).")
wlabels = pop[["person_id", "aid_status", "female", "age_at_dx", WCOL]]
wrows, lin_rows = [], []
for level, defn, dom, base in targets:
    base = base.merge(wlabels, on="person_id", how="left", validate="many_to_one")
    for meas, _ in MEASURES:
        for sex, sel in (("Female", 1), ("Male", 0)):
            b = base[base["female"] == sel]
            ap, an = b[b["aid_status"] == 1], b[b["aid_status"] == 0]
            r = {"measure": meas, "level": level, "definition": defn, "domain": dom,
                 "row_label": "All records" if dom == "all" else au.DOMAIN_LABELS[dom],
                 "fdr_role": ("positive_control" if dom in au.POSITIVE_CONTROL_DOMAINS
                              else "tested"),
                 "sex": sex, "with_aid_pos": int(ap[meas].sum()), "n_aid_pos": len(ap),
                 "with_aid_neg": int(an[meas].sum()), "n_aid_neg": len(an)}
            d = b.assign(has_x=b[meas].astype(int), age_at_dx_sq=b["age_at_dx"] ** 2)
            fit = weighted_logit(d)
            r.update(fit)
            wrows.append(r)
            lin_rows.append({k: r[k] for k in ("measure", "level", "definition", "domain",
                                               "row_label", "sex")} | linearity_row(d, fit))
wchg = pd.DataFrame(wrows)
key = ["measure", "level", "definition", "domain", "sex"]
count_cols = ["with_aid_pos", "n_aid_pos", "with_aid_neg", "n_aid_neg"]
if not wchg[key + count_cols].equals(chg[key + count_cols]):
    raise RuntimeError("weighted rows: patients or counts differ from the Fisher rows")
wchg["q"] = NAN
wchg["fdr_family_size"] = 0
for _, idx in wchg.groupby(["measure", "level", "definition", "sex"]).groups.items():
    m = wchg.loc[idx]
    t = m.index[(m["status"] == "fitted") & (m["fdr_role"] == "tested")]
    if len(t):
        wchg.loc[t, "q"] = au.bh_fdr(wchg.loc[t, "p"].values)
        wchg.loc[t, "fdr_family_size"] = len(t)
wchg = wchg[list(chg.columns)]
for _, r in wchg[wchg["level"] == "overall"].iterrows():
    print(f"  {r['measure']:<12} {r['sex']:<7} AID+ {r['with_aid_pos']:,}/{r['n_aid_pos']:,}  "
          f"AID- {r['with_aid_neg']:,}/{r['n_aid_neg']:,}  "
          + (f"OR {fmt_console(r['odds_ratio'], '.2f')} [{fmt_console(r['ci_low'], '.2f')}-"
             f"{fmt_console(r['ci_high'], '.2f')}]  P={r['p']:.4f}  Q={r['q']:.4f}"
             if r["status"] == "fitted" else f"{r['status']}: {r['reason']}"))
print(f"\n  By domain, weighted: domains with Q < {Q_THRESHOLD} (of those fitted in the family;"
      f" primary listed):")
for meas, _ in MEASURES:
    for sex in ("Female", "Male"):
        parts = []
        for defn in DEFINITIONS:
            m = wchg[(wchg["level"] == "domain") & (wchg["definition"] == defn)
                     & (wchg["measure"] == meas) & (wchg["sex"] == sex)]
            f = m[(m["status"] == "fitted") & (m["fdr_role"] == "tested")]
            sig = f[f["q"] < Q_THRESHOLD]
            names = [f"{r['domain']} (OR {fmt_console(r['odds_ratio'], '.2f')})"
                     for _, r in sig.iterrows()] if defn == "primary" else []
            parts.append(f"{defn} {len(sig)} of {len(f)}"
                         + (f" ({', '.join(names)})" if names else ""))
        print(f"  {meas:<12} {sex:<7} " + "; ".join(parts))
nf = wchg[wchg["status"] != "fitted"]
au.count_line("weighted rows not fitted (reason in the workbook)", len(nf), len(wchg))

lin_df = pd.DataFrame(lin_rows)
run = lin_df[lin_df["status"] == "run"]
flagged = run[run["flag"] == 1]
print(f"\n  Linearity check (weighted model + age at AD diagnosis squared): {len(run)} of "
      f"{len(lin_df)} run, {len(lin_df) - len(run)} not run")
print(f"    flagged (AID OR shift >= {au.LINEARITY_OR_SHIFT:.0%} and |OR - 1| >= "
      f"{au.LINEARITY_OR_MIN_DIFF}): {len(flagged)}"
      + ("" if flagged.empty else " -- " + "; ".join(
          f"{r['measure']} {r['sex']} {r['definition']} {r['domain']} ({r['or_shift_pct']:.1f}%)"
          for _, r in flagged.sort_values("or_shift_pct", ascending=False).iterrows())))
if len(run):
    print(f"    largest shift: {run['or_shift_pct'].max():.1f}%")

au.sub("Secondary domain definition check, weighted models")
wsec_df = pd.DataFrame(definition_check_rows(wchg, "fitted", differs), columns=sec_df.columns)

# =============================================================================
# SECTION 3 -- SHARES OF RECORDS
# =============================================================================
au.section("SECTION 3 -- SHARES OF RECORDS")
au.sub("Per-patient share of worsening and of severe records (Mann-Whitney U, within sex)")
pt_rate = cr.groupby("person_id")[["worsening", "severe"]].mean().mul(100).reset_index() \
    .merge(labels, on="person_id", how="left", validate="one_to_one")
share_rows = []
for meas in ("worsening", "severe"):
    for sex, sel in (("Female", 1), ("Male", 0)):
        b = pt_rate[pt_rate["female"] == sel]
        ap, an = b.loc[b["aid_status"] == 1, meas], b.loc[b["aid_status"] == 0, meas]
        status, why, p = mann_whitney(ap, an)
        share_rows.append({"measure": meas, "sex": sex, "n_aid_pos": len(ap),
                           "median_aid_pos": ap.median(), "n_aid_neg": len(an),
                           "median_aid_neg": an.median(), "status": status, "reason": why,
                           "p": p})
        print(f"  {meas:<10} {sex:<7} AID+ median {fmt_console(ap.median(), '.1f')}% "
              f"(n={len(ap):,})  AID- median {fmt_console(an.median(), '.1f')}% (n={len(an):,})  "
              + (f"P={p:.4f}" if status == "tested" else f"not tested: {why}"))

au.sub("Per-patient share of worsening and of severe records, PS-weighted mean (descriptive)")
wpt_rate = pt_rate.merge(pop[["person_id", WCOL]], on="person_id", how="left",
                         validate="one_to_one")
wshare_rows = []
for meas in ("worsening", "severe"):
    for sex, sel in (("Female", 1), ("Male", 0)):
        b = wpt_rate[wpt_rate["female"] == sel]
        r = {"measure": meas, "sex": sex}
        for arm, a in (("aid_pos", 1), ("aid_neg", 0)):
            g = b[b["aid_status"] == a]
            r[f"n_{arm}"] = len(g)
            r[f"weighted_mean_{arm}"] = (float(np.average(g[meas], weights=g[WCOL]))
                                         if len(g) else NAN)
        r["weighted_difference"] = r["weighted_mean_aid_pos"] - r["weighted_mean_aid_neg"]
        wshare_rows.append(r)
        print(f"  {meas:<10} {sex:<7} AID+ {fmt_console(r['weighted_mean_aid_pos'], '.1f')}% "
              f"(n={r['n_aid_pos']:,})  AID- {fmt_console(r['weighted_mean_aid_neg'], '.1f')}% "
              f"(n={r['n_aid_neg']:,})  difference "
              f"{fmt_console(r['weighted_difference'], '+.1f')} points")

au.sub("Before vs after the AD diagnosis, cognitive domains (share of records)")
cr["before_diagnosis"] = au.is_pre_diagnosis(cr["note_date"], cr["ad_diagnosis_date"]).astype(int)
sc = cr[cr["domain_primary"].isin(COGNITIVE_DOMAINS)]
ba_rows = []
for a, lbl in ((1, "AID+"), (0, "AID-")):
    pre = sc[(sc["aid_status"] == a) & (sc["before_diagnosis"] == 1)]
    post = sc[(sc["aid_status"] == a) & (sc["before_diagnosis"] == 0)]
    shown = len(pre) >= au.MIN_GROUP_N and len(post) >= au.MIN_GROUP_N
    r = {"group": lbl, "n_records_before": len(pre), "n_records_after": len(post)}
    for meas in ("worsening", "improvement", "severe"):
        r[f"pct_{meas}_before"] = 100 * pre[meas].mean() if shown else NAN
        r[f"pct_{meas}_after"] = 100 * post[meas].mean() if shown else NAN
    ba_rows.append(r)
    print(f"  {lbl:<5} " + ("before / after: " + "  ".join(
        f"{meas} {r[f'pct_{meas}_before']:.1f}% / {r[f'pct_{meas}_after']:.1f}%"
        for meas in ("worsening", "improvement", "severe")) if shown
        else f"fewer than {au.MIN_GROUP_N} records before or after the diagnosis"))

# =============================================================================
# SECTION 4 -- EXPLICIT RELATIVE-TIME LANGUAGE
# =============================================================================
au.section("SECTION 4 -- EXPLICIT RELATIVE-TIME LANGUAGE")
c2d = au.category_domain("primary")
rec = au.load_symptom_records(SYMPTOM_RECORDS_PATH, pop, notes)
sym = rec[rec["note_date"].notna() & (rec["category"].map(c2d) != "unassigned")].copy()
sym["domain"] = sym["category"].map(c2d)
sym = sym.merge(pop[["person_id", "birth_date"]], on="person_id", how="left",
                validate="many_to_one")
ex = au.load_explicit_time_records(EXPLICIT_TIME_PATH, pop, notes)
matched = ex[ex["note_date"].notna() & (ex["category"].map(c2d) != "unassigned")].copy()
matched["domain"] = matched["category"].map(c2d)
matched = matched.merge(pop[["person_id", "aid_status", "birth_date"]], on="person_id",
                        how="left", validate="many_to_one")
days = (matched["note_date"] - matched["birth_date"]).dt.days
matched["age_at_record"] = days / au.DAYS_PER_YEAR
matched["deviation_yrs"] = matched["implied_offset_days"].astype(float) / au.DAYS_PER_YEAR
matched["age_implied"] = (days + matched["implied_offset_days"].astype(float)) / au.DAYS_PER_YEAR
n_rows, n_pts = len(matched), matched["person_id"].nunique()
print(f"  Records with explicit relative-time language: {n_rows:,} of {len(sym):,} dated "
      f"domain records ({100 * n_rows / max(len(sym), 1):.2f}%)")
print(f"  Patients with at least one such record: {n_pts:,} of {len(pop):,}")
summary_rows = [{"records_matched": n_rows, "records_dated_domain": len(sym),
                 "patients_matched": n_pts, "patients": len(pop)}]
dev_rows, cov_rows, conf_rows = [], [], []
if n_rows:
    first_match = (matched.sort_values(["person_id", "note_date"])
                   .groupby("person_id").first().reset_index())
    print("\n  Deviation of the implied date from the note date, first match per patient (years):")
    for kind in au.EXPLICIT_TIME_KINDS + [None]:
        g = first_match["deviation_yrs"] if kind is None else \
            first_match.loc[first_match["kind"] == kind, "deviation_yrs"]
        if len(g):
            dev_rows.append({"kind": kind or "all", "n": len(g), "mean": g.mean(),
                             "median": g.median(), "sd": g.std(), "q1": g.quantile(.25),
                             "q3": g.quantile(.75)})
            print(f"  {kind or 'all':<12} n={len(g):>6,}  mean {g.mean():.2f}  median "
                  f"{g.median():.2f}  sd {fmt_console(g.std(), '.2f')}")
    cov = first_match.groupby(["domain", "aid_status"]).size().unstack(fill_value=0) \
        .reindex(columns=[1, 0], fill_value=0)
    for dom, r in cov.iterrows():
        cov_rows.append({"domain": dom, "row_label": au.DOMAIN_LABELS[dom],
                         "n_aid_pos": int(r[1]), "n_aid_neg": int(r[0]),
                         "both_at_least_min": int(r[1] >= au.MIN_GROUP_N
                                                  and r[0] >= au.MIN_GROUP_N)})
    print(f"\n  Domains with >= {au.MIN_GROUP_N} first-match patients in both AID groups: "
          f"{sum(r['both_at_least_min'] for r in cov_rows)} of {len(cov_rows)}")

    au.sub("Matches on the domain's own first documentation")
    dom_first = (sym.groupby(["person_id", "domain"])["note_date"].min()
                 .rename("first_doc_date").reset_index())
    conf = (matched.sort_values(["person_id", "domain", "note_date"])
            .groupby(["person_id", "domain"]).first().reset_index()
            .merge(dom_first, on=["person_id", "domain"], how="left"))
    conf = conf[conf["note_date"] == conf["first_doc_date"]]
    print(f"  (patient, domain) pairs whose first explicit-time match is the domain's first "
          f"documentation: {len(conf):,}")
    for dom in sorted(conf["domain"].unique()):
        dd = conf[conf["domain"] == dom]
        ap, an = dd[dd["aid_status"] == 1], dd[dd["aid_status"] == 0]
        st_nd, why, p_nd = mann_whitney(ap["age_at_record"], an["age_at_record"])
        _, _, p_im = mann_whitney(ap["age_implied"], an["age_implied"])
        conf_rows.append({"domain": dom, "row_label": au.DOMAIN_LABELS[dom],
                          "n_aid_pos": len(ap), "n_aid_neg": len(an),
                          "median_age_note_aid_pos": ap["age_at_record"].median(),
                          "median_age_note_aid_neg": an["age_at_record"].median(),
                          "median_age_implied_aid_pos": ap["age_implied"].median(),
                          "median_age_implied_aid_neg": an["age_implied"].median(),
                          "status": st_nd, "reason": why, "p_age_note": p_nd,
                          "p_age_implied": p_im})
        if st_nd == "tested":
            print(f"  {dom:<32} AID+ {len(ap):>5,} AID- {len(an):>5,}  gap (note) "
                  f"{ap['age_at_record'].median() - an['age_at_record'].median():+.2f} yr "
                  f"P={p_nd:.4f}  gap (implied) "
                  f"{ap['age_implied'].median() - an['age_implied'].median():+.2f} yr "
                  f"P={p_im:.4f}")

# =============================================================================
# SECTION 5 -- SOURCE DATA (checks)
# =============================================================================
au.section("SECTION 5 -- SOURCE DATA")


def parameters(extra=()):
    rows = [("min_group_n", au.MIN_GROUP_N), ("age_ceiling_female", ceil["ceiling_female"]),
            ("age_ceiling_male", ceil["ceiling_male"]), ("q_threshold", Q_THRESHOLD),
            ("or_change_small", OR_CHANGE_SMALL), ("or_change_large", OR_CHANGE_LARGE),
            ("n_aid_pos", int(pop["aid_status"].sum())),
            ("n_aid_neg", int((pop["aid_status"] == 0).sum()))]
    return pd.DataFrame(rows + list(extra), columns=["parameter", "value"])


au.write_workbook(os.path.join(SOURCE_DATA_DIR, CHANGE_FILE), {
    "patient_level": chg,
    "patient_level_weighted": wchg,
    "secondary_definition": sec_df,
    "secondary_definition_weighted": wsec_df,
    "linearity_weighted": lin_df,
    "record_shares": pd.DataFrame(share_rows),
    "weighted_record_shares": pd.DataFrame(wshare_rows, columns=[
        "measure", "sex", "n_aid_pos", "weighted_mean_aid_pos", "n_aid_neg",
        "weighted_mean_aid_neg", "weighted_difference"]),
    "before_after_diagnosis": pd.DataFrame(ba_rows),
    "parameters": parameters([("records_kept", int(len(cr)))]),
})
au.write_workbook(os.path.join(SOURCE_DATA_DIR, EXPLICIT_FILE), {
    "summary": pd.DataFrame(summary_rows),
    "deviation_by_kind": pd.DataFrame(dev_rows, columns=["kind", "n", "mean", "median", "sd",
                                                         "q1", "q3"]),
    "domain_coverage": pd.DataFrame(cov_rows, columns=["domain", "row_label", "n_aid_pos",
                                                       "n_aid_neg", "both_at_least_min"]),
    "first_documentation_matches": pd.DataFrame(conf_rows),
    "parameters": parameters(),
})
print("\n" + au.SEP + "\n  COMPLETE\n" + au.SEP)