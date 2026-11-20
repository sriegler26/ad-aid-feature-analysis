"""
dlb_feature_timing.py
=====================
When are the DLB core features and RBD first documented in female patients
with AD, AID+ against AID-? Early documentation, around the AD diagnosis and
not at older ages than in AID- patients, fits a DLB signal; documentation
concentrated years after the diagnosis would fit late-stage AD or long
follow-up instead.

Computes every statistic and writes the source data for one table (two
panels) and one checks workbook. The table is drawn by
dlb_feature_timing_tables.py; nothing here formats a number.

Population and features
-----------------------
Female patients of the analysis population with PS weights, coded DLB
included. Features come from the shared analysis_utils.dlb_patient_flags,
so they are the DLB core analysis's and the counts of patients with a
feature equal its events: parkinsonism (McKeith),
fluctuating cognition, visual hallucinations, and RBD (confirmed, or probable
without confirmed; has_rbd_any). A feature counts when its first analyzed
date is at or below the patient's own-sex age ceiling, computed on the
whole population, both sexes, as in every DLB analysis; parkinsonism
(McKeith) only when first documented no earlier than 1 year before the AD
diagnosis date. The date of a feature is its first analyzed date within the
ceiling; for RBD the date of the confirmed record when there is one, else of
the probable record. Groups: AID+ and AID-, PS-weighted. Ages are whole days
/ 365.25 from the birth date.

Part A: timing among the patients with the feature
--------------------------------------------------
Per feature and AID group: patients with the feature; median (Q1, Q3) age at
first documentation; median (Q1, Q3) years from the AD diagnosis to first
documentation (negative = before). The diagnosis day counts as after the
diagnosis.
A group's summary needs >= 10 patients with the feature
(group_minimum_reason). AID+ minus AID- differences in the median age and in
the median years from the diagnosis, with percentile bootstrap 95% CIs and
two-sided bootstrap P values, need >= 10 patients with the feature in each
group.

Part B: first-documentation rate per person-year, before and from 5 years
after the AD diagnosis
-------------------------------------------------------------------------
A patient is at risk from the first dated note (left truncation) to the
earliest of the last dated note, the age ceiling and death (end of
observation), or to the first documentation (the event). A documentation at
the first note, or up to 30 days before it, counts one day after entry.
Parkinsonism can be an event only from 1 year before the AD diagnosis, so its
time at risk starts at the later of the first note and that date, and
patients whose parkinsonism is first documented earlier (never an event) are
left out and counted. A documentation dated after the end of observation
(after death) is not an event: the patient is censored at the end of
observation and counted. Two windows of years from the AD diagnosis: before 5
years after it (< 5, the pre-diagnosis years included) and from 5 years after
it (>= 5); [lo, hi), a documentation at the middle of its day. Rates per 100
person-years observed, PS-weighted; per window the rate ratio AID+ / AID-
with a percentile bootstrap 95% CI and two-sided bootstrap P, estimated when
each group has >= 10 patients with time in the window and >= 10 events in it
(rate_ratio_floor_reason); and the ratio of the two rate ratios (>= 5 / < 5,
above 1: the AID+ excess is larger late), estimated when both are. Longer
follow-up adds person-years as well as events, so it cannot by itself raise a
rate.

Bootstrap: 2,000 draws, patients resampled within each AID group (Part A:
the patients with the feature), weights held fixed; P = 2 x the smaller share
of draws on either side of 0 (log scale for ratios), as (k + 1) / (B + 1),
capped at 1. No result is adjusted for multiple comparisons. Dates are first
documentation, not onset.

Inputs: the patient table, the note table, the DLB feature table and the
propensity-score weights (see PATHS).
"""

import os

import numpy as np
import pandas as pd

import analysis_utils as au

# =============================================================================
# PATHS (relative to the data folder the script is run from)
# =============================================================================
COHORT_PATIENTS_PATH = ""  # patient table: AID status, sex, key dates (birth, AD diagnosis, death) and analysis-population flag
COHORT_NOTES_PATH = ""  # note table: one row per clinical note, with its date and specialty
DLB_MENTIONS_PATH = ""  # DLB feature table: one row per documented feature, with its date
PS_WEIGHTS_PATH = ""  # propensity-score weights, one per analysis patient
SOURCE_DATA_DIR = ""  # folder for the source-data workbooks written here

TABLE_FILE = "table_dlb_feature_timing.xlsx"
CHECKS_FILE = "checks_dlb_feature_timing.xlsx"

# =============================================================================
# SETTINGS
# =============================================================================
NAN = float("nan")
N_BOOT = 2000                         # bootstrap draws (every percentile CI and P)
SEED_RATES, SEED_TIMING = 2, 3
DAY = 1 / au.DAYS_PER_YEAR
EV_MID = 0.5 * DAY                    # a documentation sits at the middle of its day
MCKEITH_START_DAYS = int(np.floor(au.DLB_MCKEITH_WINDOW_YRS * au.DAYS_PER_YEAR))
TINY_GAP_YRS = 30 * DAY               # documentation up to 30 days before entry
LATE_YRS = 5.0                        # the window split: years after the AD diagnosis
EDGES = np.array([-100.0, LATE_YRS, 100.0])
WLAB = ["<5", ">=5"]

# (code, label, presence flag); the date column is built below
FEATURES = [("park", "Parkinsonism (McKeith)", "has_park_all_mckeith"),
            ("fluct", "Fluctuating cognition", "has_fluct"),
            ("vh", "Visual hallucinations", "has_vis_hall"),
            ("rbd", "REM sleep behavior disorder (RBD)", "has_rbd_any")]
GROUPS = [("aid_pos", "AID+"), ("aid_neg", "AID-")]

ST_ANALYZED = "analyzed"
ST_MISSING = "missing age at first note, end of observation, age at diagnosis or weight"
ST_PDD = "parkinsonism first documented more than 1 year before the AD diagnosis"
ST_NO_FU = "no follow-up after the start of risk, feature not documented"
ST_ENDS_PRE = "observation ends before the start of risk"
ST_EV_PRE = "feature dated more than 30 days before the start of risk"
EXCL_REASONS = [ST_PDD, ST_NO_FU, ST_ENDS_PRE, ST_EV_PRE, ST_MISSING]


# =============================================================================
# HELPERS
# =============================================================================
def wquant(x, w, q):
    """Weighted quantile (midpoint cumulative weights, linear interpolation)."""
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    m = ~(np.isnan(x) | np.isnan(w)) & (w > 0)
    x, w = x[m], w[m]
    if len(x) == 0:
        return NAN
    o = np.argsort(x, kind="mergesort")
    x, w = x[o], w[o]
    c = (np.cumsum(w) - 0.5 * w) / w.sum()
    return float(np.interp(q, c, x))


def boot_p(draws, null=0.0):
    """Two-sided bootstrap P of the null value: 2 x the smaller share of draws
    at or beyond it on either side, as (k + 1) / (B + 1), capped at 1. NaN
    draws are left out."""
    d = np.asarray(draws, float)
    d = d[~np.isnan(d)]
    if len(d) == 0:
        return NAN
    lo = ((d <= null).sum() + 1) / (len(d) + 1)
    hi = ((d >= null).sum() + 1) / (len(d) + 1)
    return float(min(1.0, 2 * min(lo, hi)))


# =============================================================================
# SECTION 1 -- POPULATION, WEIGHTS, FEATURES
# =============================================================================
au.section("SECTION 1 -- POPULATION, WEIGHTS, FEATURES AND THE AGE CEILING")
pop, notes = au.load_population(COHORT_PATIENTS_PATH, COHORT_NOTES_PATH)
pop = au.load_ps_weights(PS_WEIGHTS_PATH, pop)
pop = pop.sort_values("person_id", kind="mergesort").reset_index(drop=True)
pop["w"] = pop["ps_weight_trimmed"].astype(float)
mentions = au.load_dlb_feature_mentions(DLB_MENTIONS_PATH, pop, notes)
pop, CEIL = au.dlb_patient_flags(pop, mentions)   # ceilings from both sexes
pop["ceiling"] = pop["person_id"].map(CEIL["ceiling_by_person"]).astype(float)
pop["age_at_death"] = au.years_between(pop["death_date"], pop["birth_date"])
au.count_line("male patients (not analyzed here; used for the male age ceiling only)",
              (pop["female"] == 0).sum(), len(pop))
fem = pop[pop["female"] == 1].copy()
if fem.empty:
    raise ValueError("no female patients in the analysis population")
dx = fem["ad_diagnosis_date"]

# Dates of the features: the analyzed first date within the ceiling; RBD from
# the record that defines it (confirmed, else probable).
fem["date_park"] = fem["t_park_all"]
fem["date_fluct"] = fem["t_fluct"]
fem["date_vh"] = fem["t_vis_hall"]
fem["date_rbd"] = fem["t_rbd_conf"].where(fem["has_rbd_conf"] == 1, fem["t_rbd_prob"])
for code, label, flag in FEATURES:
    fem["f_" + code] = fem[flag].astype(int)
    fem["date_" + code] = fem["date_" + code].where(fem["f_" + code] == 1)
    fem["yfd_" + code] = au.years_between(fem["date_" + code], dx)
    fem["age_" + code] = au.years_between(fem["date_" + code], fem["birth_date"])
    undated = int(((fem["f_" + code] == 1) & fem["date_" + code].isna()).sum())
    if undated:
        raise RuntimeError(f"{label}: {undated:,} patients with the feature but no date")

# Start of risk: the first dated note; for parkinsonism the later of that and
# 1 year before the AD diagnosis date (no dated note: no start of risk).
fem["entry_age"] = fem["age_at_entry"]
mck_start = dx - pd.Timedelta(days=MCKEITH_START_DAYS)
fem["entry_age_park"] = au.years_between(
    pd.concat([fem["first_entry_date"], mck_start], axis=1).max(axis=1, skipna=False),
    fem["birth_date"])
fem["park_before_window"] = fem["has_park_all_pdd_type"]

GROUP_MASK = {"aid_pos": fem["aid_status"] == 1, "aid_neg": fem["aid_status"] == 0}
for g, lbl in GROUPS:
    au.count_line(f"female patients, {lbl}", GROUP_MASK[g].sum(), len(fem))
au.count_line("female patients with parkinsonism first documented more than 1 year before "
              "the AD diagnosis (not counted as parkinsonism (McKeith))",
              fem["park_before_window"].sum(), len(fem))
au.count_line("female patients with no dated note (no time at risk)",
              fem["entry_age"].isna().sum(), len(fem))
for code, label, flag in FEATURES:
    au.count_line(f"female patients with {label}", fem["f_" + code].sum(), len(fem))


# =============================================================================
# SECTION 2 -- PART A: TIMING AMONG THE PATIENTS WITH THE FEATURE
# =============================================================================
au.section("SECTION 2 -- PART A: AGE AND TIMING OF FIRST DOCUMENTATION")


def summary(d, w, code):
    """Summary of the patients of d with the feature (weights w)."""
    a, t = d["age_" + code].to_numpy(float), d["yfd_" + code].to_numpy(float)
    n = len(d)
    why = au.group_minimum_reason(n)
    r = dict(n=n, status="not_estimated" if why else "estimated", reason=why)
    if why:
        return r
    r.update(med_age=wquant(a, w, .5), q1_age=wquant(a, w, .25), q3_age=wquant(a, w, .75),
             med_yrs_from_dx=wquant(t, w, .5), q1_yrs_from_dx=wquant(t, w, .25),
             q3_yrs_from_dx=wquant(t, w, .75))
    return r


def median_diff_boot(p, n, wp, wn, seed):
    """AID+ minus AID- difference in weighted medians: percentile bootstrap CI
    and two-sided bootstrap P (patients resampled within group, weights
    fixed)."""
    est = wquant(p, wp, .5) - wquant(n, wn, .5)
    rng = np.random.default_rng(seed)
    b = np.empty(N_BOOT)
    for i in range(N_BOOT):
        ip, ineg = rng.integers(0, len(p), len(p)), rng.integers(0, len(n), len(n))
        b[i] = wquant(p[ip], wp[ip], .5) - wquant(n[ineg], wn[ineg], .5)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return est, float(lo), float(hi), boot_p(b)


def part_a(weighted):
    timing, diffs = [], []
    for code, label, flag in FEATURES:
        pos = {}
        for g, glbl in GROUPS:
            d = fem[GROUP_MASK[g] & (fem["f_" + code] == 1)]
            w = d["w"].to_numpy(float) if weighted else np.ones(len(d))
            pos[g] = (d, w)
            timing.append(dict(feature=code, group=g,
                               weighting="weighted" if weighted else "unweighted",
                               n_group=int(GROUP_MASK[g].sum()), **summary(d, w, code)))
        (dp, wp), (dn, wn) = pos["aid_pos"], pos["aid_neg"]
        why = au.group_minimum_reason(len(dp), len(dn))
        r = dict(feature=code, n_aid_pos=len(dp), n_aid_neg=len(dn),
                 status="not_estimated" if why else "estimated", reason=why)
        if not why:
            for k, col in (("age", "age_"), ("yrs_from_dx", "yfd_")):
                est, lo, hi, p = median_diff_boot(dp[col + code].to_numpy(float),
                                                  dn[col + code].to_numpy(float), wp, wn,
                                                  SEED_TIMING)
                r.update({f"diff_med_{k}": est, f"diff_med_{k}_ci_low": lo,
                          f"diff_med_{k}_ci_high": hi, f"p_diff_med_{k}": p})
        diffs.append(r)
    return pd.DataFrame(timing), pd.DataFrame(diffs)


timing, diffs = part_a(True)
timing_u, diffs_u = part_a(False)
for _, r in timing.iterrows():
    if r["status"] == "estimated":
        print(f"  {r['feature']:<6} {r['group']:<8} n {r['n']:>5}  age {r['med_age']:5.1f}  "
              f"yrs from dx {r['med_yrs_from_dx']:+6.2f}")
    else:
        print(f"  {r['feature']:<6} {r['group']:<8} n {r['n']:>5}  {r['reason']}")
for _, r in diffs.iterrows():
    if r["status"] == "estimated":
        print(f"  {r['feature']:<6} AID+ - AID-: median age {r['diff_med_age']:+.2f} "
              f"({r['diff_med_age_ci_low']:+.2f} to {r['diff_med_age_ci_high']:+.2f}) "
              f"P={r['p_diff_med_age']:.3f}; median yrs from dx "
              f"{r['diff_med_yrs_from_dx']:+.2f} ({r['diff_med_yrs_from_dx_ci_low']:+.2f} to "
              f"{r['diff_med_yrs_from_dx_ci_high']:+.2f}) P={r['p_diff_med_yrs_from_dx']:.3f}")
    else:
        print(f"  {r['feature']:<6} AID+ - AID-: {r['reason']}")


# =============================================================================
# SECTION 3 -- TIME AT RISK
# =============================================================================
def surv_table(d, code):
    """Entry, exit, event and end of observation (ages) for each patient of d
    and feature code; returns (analyzed rows, status per patient of d)."""
    t_age = d["age_" + code]
    death_age = np.where(d["death_flag"] == 1, d["age_at_death"].fillna(np.inf), np.inf)
    censor = np.nanmin(np.column_stack([d["age_at_last_note"].astype(float),
                                        d["ceiling"].astype(float), death_age]), axis=1)
    # documentation after the end of observation (after death) is not an event
    late = (d["f_" + code] == 1) & (t_age > censor)
    ev = (d["f_" + code] == 1) & t_age.notna() & ~late
    entry = (d["entry_age_park"] if code == "park" else d["entry_age"]).astype(float)
    full = pd.DataFrame({"person_id": d["person_id"].astype(int), "aid": d["aid_status"],
                         "w": d["w"], "dx": d["age_at_dx"].astype(float), "entry": entry,
                         "exit": np.where(ev, t_age, censor).astype(float),
                         "event": ev.astype(int), "obs_end": censor.astype(float),
                         "late": late.astype(int)},
                        index=d.index)
    miss = full[["entry", "exit", "dx", "w"]].isna().any(axis=1)
    pdd = (d["park_before_window"] == 1) if code == "park" else pd.Series(False, index=d.index)
    out = full[~miss & ~pdd].copy()
    gap = out["entry"] - out["exit"]
    tiny = (((gap > 0) | ((gap == 0) & (out["event"] == 1))) & (gap <= TINY_GAP_YRS))
    out.loc[tiny, "exit"] = out.loc[tiny, "entry"] + DAY
    bad = out["exit"] <= out["entry"]
    status = pd.Series(ST_ANALYZED, index=d.index, dtype=object)
    status[miss] = ST_MISSING
    status[pdd & ~miss] = ST_PDD
    b = out[bad]
    status[b.index[(b["event"] == 1).to_numpy()]] = ST_EV_PRE
    status[b.index[((b["event"] == 0) & (b["exit"] == b["entry"])).to_numpy()]] = ST_NO_FU
    status[b.index[((b["event"] == 0) & (b["exit"] < b["entry"])).to_numpy()]] = ST_ENDS_PRE
    st = out[~bad].copy()
    # window analyses: a documentation sits at the middle of its day
    st["exit_w"] = st["exit"] + np.where(st["event"] == 1, EV_MID, 0.0)
    return st, status


# =============================================================================
# SECTION 4 -- PART B: RATES BEFORE AND FROM 5 YEARS AFTER THE DIAGNOSIS
# =============================================================================
au.section("SECTION 4 -- PART B: FIRST-DOCUMENTATION RATE BEFORE AND FROM 5 YEARS AFTER THE "
           "DIAGNOSIS")


def window_mats(st):
    """Person-years and events per window of years from diagnosis ([lo, hi);
    a documentation at the middle of its day)."""
    e = (st["entry"] - st["dx"]).to_numpy()
    x = (st["exit_w"] - st["dx"]).to_numpy()
    ev = st["event"].to_numpy() == 1
    lo, hi = EDGES[:-1], EDGES[1:]
    pt = np.clip(np.minimum(x[:, None], hi) - np.maximum(e[:, None], lo), 0, None)
    E = (ev[:, None] & (x[:, None] >= lo) & (x[:, None] < hi)).astype(float)
    return pt, E


def rates(pt, E, w):
    num, den = (w[:, None] * E).sum(0), (w[:, None] * pt).sum(0)
    return np.divide(num, den, out=np.full(len(num), np.nan), where=den > 0)


def rr_boot(pt, E, w, g1, g0, seed):
    """Rate ratio g1 / g0 per window with its bootstrap draws of the log rate
    ratio, and the gate per window (rate_ratio_floor_reason: patients with
    time in the window and events, per group)."""
    r1, r0 = rates(pt[g1], E[g1], w[g1]), rates(pt[g0], E[g0], w[g0])
    rng = np.random.default_rng(seed)
    i1, i0 = np.where(g1)[0], np.where(g0)[0]
    boots = []
    for _ in range(N_BOOT):
        b1, b0 = rng.choice(i1, len(i1)), rng.choice(i0, len(i0))
        with np.errstate(invalid="ignore", divide="ignore"):
            boots.append(np.log(rates(pt[b1], E[b1], w[b1]) / rates(pt[b0], E[b0], w[b0])))
    b = np.array(boots)
    b[~np.isfinite(b)] = np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        rr = r1 / r0
    why = [au.rate_ratio_floor_reason(int((pt[g1, k] > 0).sum()), int((pt[g0, k] > 0).sum()),
                                      int(E[g1, k].sum()), int(E[g0, k].sum()),
                                      labels=("AID+", "AID-")) for k in range(pt.shape[1])]
    return r1, r0, rr, b, why


def pct_ci(draws):
    """exp of the 2.5th and 97.5th percentiles of log-scale draws (NaN left out)."""
    d = np.asarray(draws, float)
    d = d[~np.isnan(d)]
    if len(d) == 0:
        return NAN, NAN
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(np.exp(lo)), float(np.exp(hi))


ST_CACHE, recon = {}, []
for code, label, flag in FEATURES:
    st, status = surv_table(fem, code)
    ST_CACHE[code] = st
    for g, glbl in GROUPS:
        s_ = status[GROUP_MASK[g]]
        rr_ = dict(feature=code, group=g, n_group=len(s_),
                   n_excluded=int((s_ != ST_ANALYZED).sum()),
                   n_analyzed=int((s_ == ST_ANALYZED).sum()),
                   n_events=int(st.loc[st.index.isin(s_.index), "event"].sum()),
                   n_documented_after_end=int(st.loc[st.index.isin(s_.index), "late"].sum()))
        rr_.update({f"excluded: {x}": int((s_ == x).sum()) for x in EXCL_REASONS})
        recon.append(rr_)
recon = pd.DataFrame(recon)


def part_b(weighted):
    rows, ratios = [], []
    for code, label, flag in FEATURES:
        st = ST_CACHE[code]
        w = st["w"].to_numpy(float) if weighted else np.ones(len(st))
        aid = st["aid"].to_numpy() == 1
        pt, E = window_mats(st)
        r1, r0, rr, b, why = rr_boot(pt, E, w, aid, ~aid, SEED_RATES)
        for k, wl in enumerate(WLAB):
            lo, hi = pct_ci(b[:, k])
            rows.append(dict(feature=code, window_yrs_from_dx=wl,
                             weighting="weighted" if weighted else "unweighted",
                             n_aid_pos=int((pt[aid, k] > 0).sum()),
                             n_aid_neg=int((pt[~aid, k] > 0).sum()),
                             events_aid_pos=int(E[aid, k].sum()),
                             events_aid_neg=int(E[~aid, k].sum()),
                             py_aid_pos=float(pt[aid, k].sum()), py_aid_neg=float(pt[~aid, k].sum()),
                             rate100_aid_pos=100 * r1[k], rate100_aid_neg=100 * r0[k],
                             status="not_estimated" if why[k] else "estimated", reason=why[k],
                             rate_ratio=NAN if why[k] else rr[k],
                             ci_low=NAN if why[k] else lo, ci_high=NAN if why[k] else hi,
                             p=NAN if why[k] else boot_p(b[:, k])))
        # ratio of the rate ratios, late / early
        both = not (why[0] or why[1])
        rd = b[:, 1] - b[:, 0]
        lo, hi = pct_ci(rd)
        ratios.append(dict(feature=code, weighting="weighted" if weighted else "unweighted",
                           status="estimated" if both else "not_estimated",
                           reason="" if both else "a window's rate ratio was not estimated",
                           ratio_of_rate_ratios=rr[1] / rr[0] if both else NAN,
                           ci_low=lo if both else NAN, ci_high=hi if both else NAN,
                           p=boot_p(rd) if both else NAN))
    return pd.DataFrame(rows), pd.DataFrame(ratios)


window, ratio = part_b(True)
window_u, ratio_u = part_b(False)
for _, r in window.iterrows():
    rr_txt = (f"RR {r['rate_ratio']:.2f} ({r['ci_low']:.2f}-{r['ci_high']:.2f}) P={r['p']:.3f}"
              if r["status"] == "estimated" else r["reason"])
    print(f"  {r['feature']:<6} {r['window_yrs_from_dx']:>4}  events {r['events_aid_pos']:>4}/"
          f"{r['events_aid_neg']:<5} rate/100 py {r['rate100_aid_pos']:6.2f} / "
          f"{r['rate100_aid_neg']:6.2f}  {rr_txt}")
for _, r in ratio.iterrows():
    print(f"  {r['feature']:<6} ratio of rate ratios (>=5 / <5): "
          + (f"{r['ratio_of_rate_ratios']:.2f} ({r['ci_low']:.2f}-{r['ci_high']:.2f}) "
             f"P={r['p']:.3f}" if r["status"] == "estimated" else r["reason"]))


# =============================================================================
# SECTION 5 -- SOURCE DATA
# =============================================================================
au.section("SECTION 5 -- SOURCE DATA")
params = pd.DataFrame([
    ("n_population", len(pop)), ("n_female", len(fem)),
    ("ceiling_female", CEIL["ceiling_female"]), ("ceiling_male", CEIL["ceiling_male"]),
    ("n_boot", N_BOOT), ("seed_rates", SEED_RATES), ("seed_timing", SEED_TIMING),
    ("min_group_n", au.MIN_GROUP_N), ("min_events_per_term", au.MIN_EVENTS_PER_TERM),
    ("mckeith_window_yrs", au.DLB_MCKEITH_WINDOW_YRS), ("late_window_yrs", LATE_YRS),
    ("tiny_gap_days", 30), ("windows_yrs_from_dx", ", ".join(WLAB)),
    ("n_park_before_window", int(fem["park_before_window"].sum()))],
    columns=["parameter", "value"])
params["value"] = params["value"].astype(str)
labels = pd.DataFrame({"feature": [c for c, _, _ in FEATURES],
                       "row_label": [lbl for _, lbl, _ in FEATURES]})
groups_df = pd.DataFrame(GROUPS, columns=["group", "group_label"])
au.write_workbook(os.path.join(SOURCE_DATA_DIR, TABLE_FILE), {
    "timing": timing, "differences": diffs, "window_rates": window,
    "rate_ratio_change": ratio, "features": labels, "groups": groups_df,
    "parameters": params})
au.write_workbook(os.path.join(SOURCE_DATA_DIR, CHECKS_FILE), {
    "reconciliation": recon, "timing_unweighted": timing_u,
    "differences_unweighted": diffs_u, "window_rates_unweighted": window_u,
    "rate_ratio_change_unweighted": ratio_u, "parameters": params})
print("\n" + au.SEP + "\n  COMPLETE\n" + au.SEP)
