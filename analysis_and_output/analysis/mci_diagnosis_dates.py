"""
mci_diagnosis_dates.py
======================
Do the coded mild cognitive impairment (MCI) and Alzheimer's disease (AD)
diagnosis dates differ, in all patients and in AID+ and AID- patients within
each sex? The DLB analyses anchor their windows on the AD
diagnosis date (parkinsonism from 1 year before it; pre/peri-diagnosis by 1
year after it); if MCI is coded differently in AID+ patients, those windows
would not mean the same in both groups.

Computes every statistic and writes the source data for the MCI vs AD table
and a checks workbook. The table is drawn by mci_diagnosis_dates_tables.py;
nothing here formats a number.

MCI: a coded MCI diagnosis; the MCI date is the earliest date of that
diagnosis.

Analyses (analysis population)
------------------------------
- Groups: all patients; AID+ and AID- within female and within male patients.
  Per group: patients, coded MCI, dated MCI; among patients with both dates,
  MCI before / on / after the AD diagnosis date (before = strictly before
  the diagnosis day; is_pre_diagnosis / is_post_diagnosis), the gap
  (AD - MCI, years) as median (Q1-Q3) with a percentile bootstrap 95% CI of
  the median.
- Do the dates differ (the main comparison): the exact sign test
  (MCI before vs after the AD date; same-day pairs left out), the primary
  test, since it assumes nothing about the shape of the gaps; the Wilcoxon
  signed-rank test of the gap beside it. Per group; gate: group_minimum_reason
  (>= 10 patients with both dates). P values are reported without correction.
- Checks: Pearson correlation of age at MCI and age at AD diagnosis per group.

Inputs: the patient table, the note table and the coded diagnosis table
(see PATHS).
"""

import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import binomtest, pearsonr, wilcoxon

import analysis_utils as au

warnings.filterwarnings("ignore")

# =============================================================================
# PATHS (relative to the data folder the script is run from)
# =============================================================================
COHORT_PATIENTS_PATH = ""  # patient table: AID status, sex, key dates (birth, AD diagnosis, death) and analysis-population flag
COHORT_NOTES_PATH = ""  # note table: one row per clinical note, with its date and specialty
DIAGNOSIS_FLAGS_PATH = ""  # coded diagnosis table: per patient, whether MCI and related diagnoses are coded, with their first dates
SOURCE_DATA_DIR = ""  # folder for the source-data workbooks written here

TABLE_FILE = "table_mci_ad_diagnosis_dates.xlsx"
CHECKS_FILE = "checks_mci_diagnosis_dates.xlsx"

# =============================================================================
# SETTINGS
# =============================================================================
N_BOOT = 2000
BOOT_SEED = 20260805
NAN = float("nan")
GROUPS = [("all", "All patients", None, None),
          ("aid_pos_female", "AID+ female", 1, 1), ("aid_neg_female", "AID- female", 0, 1),
          ("aid_pos_male", "AID+ male", 1, 0), ("aid_neg_male", "AID- male", 0, 0)]


def boot_ci_median(x, seed):
    """Percentile bootstrap 95% CI of the median (N_BOOT resamples of the
    patients, one generator per group)."""
    x = np.asarray(x, dtype=float)
    if len(x) < 3:
        return NAN, NAN
    rng = np.random.default_rng(seed)
    boots = [np.median(rng.choice(x, size=len(x), replace=True)) for _ in range(N_BOOT)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


# =============================================================================
# SECTION 1 -- POPULATION AND DATES
# =============================================================================
au.section("SECTION 1 -- POPULATION, CODED MCI AND THE TWO DATES")
pop, notes = au.load_population(COHORT_PATIENTS_PATH, COHORT_NOTES_PATH)
pop = au.load_dlb_diagnosis_flags(DIAGNOSIS_FLAGS_PATH, pop)
pop = pop.sort_values("person_id", kind="mergesort").reset_index(drop=True)
dx = pop["ad_diagnosis_date"]
mci = pop["mci_first_date"]
pop["mci_dated"] = mci.notna().astype(int)
pop["mci_before"] = au.is_pre_diagnosis(mci, dx).astype(int)
pop["mci_after"] = au.is_post_diagnosis(mci, dx + pd.Timedelta(days=1)).astype(int)
pop["mci_same_day"] = (au.is_post_diagnosis(mci, dx) & (pop["mci_after"] == 0)).astype(int)
pop["gap_yrs"] = au.years_between(dx, mci)              # AD - MCI; > 0 = MCI first
pop["age_at_mci"] = au.years_between(mci, pop["birth_date"])
au.count_line("patients with a coded MCI diagnosis", pop["mci_any"].sum(), len(pop))
au.count_line("... with a dated MCI diagnosis", pop["mci_dated"].sum(), len(pop))
au.count_line("coded MCI without a date (not in the date comparisons)",
              int(((pop["mci_any"] == 1) & (pop["mci_dated"] == 0)).sum()), len(pop))
if (pop["mci_before"] + pop["mci_after"] + pop["mci_same_day"] != pop["mci_dated"]).any():
    raise RuntimeError("MCI before / on / after does not partition the dated MCI diagnoses")


def group_frame(a, s):
    d = pop
    if s is not None:
        d = d[(d["aid_status"] == a) & (d["female"] == s)]
    return d


# =============================================================================
# SECTION 2 -- PER GROUP: DO THE DATES DIFFER?
# =============================================================================
au.section("SECTION 2 -- MCI VS AD DIAGNOSIS DATE PER GROUP (SIGN TEST PRIMARY)")
rows, corr_rows = [], []
for k, (code, label, a, s) in enumerate(GROUPS):
    d = group_frame(a, s)
    both = d[d["mci_dated"] == 1]
    g = both["gap_yrs"].to_numpy(dtype=float)
    nb, na_, ns = int(both["mci_before"].sum()), int(both["mci_after"].sum()), \
        int(both["mci_same_day"].sum())
    r = {"group": code, "row_label": label, "n": len(d), "n_mci_coded": int(d["mci_any"].sum()),
         "n_mci_dated": len(both), "n_mci_before": nb, "n_mci_same_day": ns,
         "n_mci_after": na_, "gap_median_yrs": NAN, "gap_q1_yrs": NAN, "gap_q3_yrs": NAN,
         "gap_median_ci_low": NAN, "gap_median_ci_high": NAN,
         "pct_mci_before": 100 * nb / len(both) if len(both) else NAN,
         "status": "", "reason": "", "p_sign": NAN, "p_wilcoxon": NAN}
    if len(both):
        r.update(gap_median_yrs=float(np.median(g)), gap_q1_yrs=float(np.percentile(g, 25)),
                 gap_q3_yrs=float(np.percentile(g, 75)))
        lo, hi = boot_ci_median(g, BOOT_SEED)
        r.update(gap_median_ci_low=lo, gap_median_ci_high=hi)
    why = au.group_minimum_reason(len(both))
    if why:
        r.update(status="not_tested", reason=why)
    else:
        r.update(status="tested",
                 p_sign=float(binomtest(nb, nb + na_, 0.5).pvalue) if nb + na_ else NAN,
                 p_wilcoxon=float(wilcoxon(g).pvalue) if (g != 0).any() else NAN)
    rows.append(r)
    cr = {"group": code, "row_label": label, "n": len(both), "status": "", "reason": "",
          "pearson_r": NAN, "p": NAN, "age_at_dx_median": NAN, "age_at_mci_median": NAN}
    why = au.group_minimum_reason(len(both))
    if len(both):
        cr.update(age_at_dx_median=float(both["age_at_dx"].median()),
                  age_at_mci_median=float(both["age_at_mci"].median()))
    if why:
        cr.update(status="not_tested", reason=why)
    else:
        rr, pp = pearsonr(both["age_at_dx"], both["age_at_mci"])
        cr.update(status="tested", pearson_r=float(rr), p=float(pp))
    corr_rows.append(cr)
    print(f"  {label:<14} n {len(d):>6,}; MCI coded {r['n_mci_coded']:,}, dated {len(both):,}: "
          f"before {nb:,}, same day {ns:,}, after {na_:,}; median gap "
          f"{r['gap_median_yrs']:.2f} yr; sign P {r['p_sign']:.4g}, Wilcoxon P "
          f"{r['p_wilcoxon']:.4g} {r['reason']}")
groups = pd.DataFrame(rows)
corr = pd.DataFrame(corr_rows)

# =============================================================================
# SECTION 3 -- SOURCE DATA
# =============================================================================
au.section("SECTION 3 -- SOURCE DATA")
params = pd.DataFrame([("n_population", len(pop)), ("n_boot", N_BOOT), ("boot_seed", BOOT_SEED),
                       ("min_group_n", au.MIN_GROUP_N),
                       ("n_mci_coded_undated", int(((pop["mci_any"] == 1)
                                                    & (pop["mci_dated"] == 0)).sum()))],
                      columns=["parameter", "value"])
au.write_workbook(os.path.join(SOURCE_DATA_DIR, TABLE_FILE),
                  {"groups": groups, "parameters": params})
au.write_workbook(os.path.join(SOURCE_DATA_DIR, CHECKS_FILE),
                  {"age_correlation": corr, "parameters": params})
print("\n" + au.SEP + "\n  COMPLETE\n" + au.SEP)
