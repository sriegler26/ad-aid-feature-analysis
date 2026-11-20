"""
aid_condition_analysis.py
=========================
Where does the excess of documented core features of dementia with Lewy
bodies (DLB) in female patients with Alzheimer's disease (AD) and autoimmune
disease (AID+) come from? Is it spread across the coded autoimmune conditions
or concentrated in a few; does it sit in patients with several core features,
often with a coded DLB or Parkinson's disease (PD) diagnosis, or in patients
with one feature; when was the AID first recorded relative to the AD diagnosis;
does the AID+ excess survive removing the carriers of conditions reported to
produce a core feature (possible mimics); do published groupings of the conditions (serology,
systemic vs organ-specific, type I interferon, genetic cluster, immunological
continuum) carry more of it; and how do the conditions sit against published PD
risk and PD pleiotropy (descriptive only)? In female patients with Sjogren's
syndrome, are the features documented in the episode that diagnosed it?

Computes every statistic and writes the source data for the AID condition
figure, the extended data figure of core-feature combinations by
condition, the grouping table, the mimic-pair table and the Sjogren's timing
figure (one workbook per product), and a checks workbook. Tables and figures
are drawn by the table and figure scripts; nothing here formats a number. The
per-condition SIRs and RoRs are printed to the console and kept in the checks
workbook.

Estimators
----------
A comparison of AID+ with AID- patients, including an AID+ subset defined by
removing patients, uses the inverse-probability-weighted (IPW) model of the main
analysis (analysis_utils.ipw_female_fit: female patients, y ~ aid_status +
age_at_dx, binomial, var_weights = the trimmed PS weight, HC0). Indirect standardization
(SIR) is used only for single conditions or groups of conditions; no SIR pooled
over all AID+ patients is reported as an effect.

Method: indirect standardization
--------------------------------
Expected counts come from AID- patients of the same sex (analysis_utils:
standardization_strata, expected_counts): strata = quintiles of the
pre-diagnosis documentation span x quintiles of the pre-diagnosis note count
x AD-age bands, cut on the AID- patients of each sex; a stratum without AID-
support takes the overall same-sex rate (counted). The PS weights are not
used (fitted for the AID+ vs AID- contrast, not for condition subgroups).
SIR = observed / expected with an exact (Garwood) CI and a two-sided exact
Poisson P; RoR = SIR female / SIR male by offset Poisson regression with the
HC0 variance. Gates: a SIR needs >= 10 carriers of that sex; a ratio of two
groups >= 10 patients and >= 10 observed events in each
(analysis_utils.rate_ratio_floor_reason).

Outcomes: core features = parkinsonism (McKeith) + fluctuating cognition +
visual hallucinations (0-3 per patient), from analysis_utils.dlb_patient_flags
(age ceiling, windows); >= 1 and >= 2 core features; the exact combination
of the three (8 classes); coded DLB or PD. Units: each coded condition, with
ulcerative colitis, Crohn's disease and inflammatory bowel disease as one
unit. Conditions shown: >= 10 female carriers (one list for every display).

Products
--------
- Per-condition SIRs (console and checks; all carriers, isolated carriers):
  per unit the carriers, observed and expected per sex, SIR female, SIR male,
  RoR. BH within each arm, separately for the female SIR, the male SIR
  and the RoR.
- AID condition figure, panels in narrative order:
  a  spread: per condition the females' mutually adjusted rate ratio of core
     features (offset Poisson on all females, log E offset, HC0; one 0/1
     term per condition with >= 5 core features among its female carriers,
     a "smaller conditions" term, and a "per extra condition" term), the
     DerSimonian-Laird pool, tau, I2 and the 95% prediction interval; the
     conditions' membership of five published groupings (serology, organ
     involvement, type I interferon: two classes; PD risk: raised, limited
     or conflicting evidence, not raised; PD pleiotropy: moderate to strong, weak; units outside a
     map are not classified);
  b  excess core features (observed - expected), AID+ females, split exactly
     into single-feature and multi-feature combinations (expected values are
     stratum means, so the combination excesses add up to the total), each
     female split 1/k over her k conditions; the multi-feature excess split by
     coded DLB/PD; bootstrap CIs of the totals;
  c  per condition (>= 10 female carriers), years from the first recorded AID
     diagnosis to the AD diagnosis in female carriers (median, IQR; positive =
     AID first), n dated, n and % with the AID diagnosis after the AD
     diagnosis (the female rows of the aid_dx_timing check);
  d  possible mimics, IPW: the female IPW odds ratio of >= 1 and >= 2 core
     features in all females and after removing the AID+ carriers (both
     sexes) of the conditions in each cumulative set of possible mimics
     (narrow, wider, all reported; analysis_utils.AID_MIMIC_SETS); weights
     reused or refitted by analysis_utils.ipw_weights_rule (both designs in
     the checks workbook).
- Extended data: core-feature combinations by condition, AID+ females:
  observed, expected, SIR.
- Possible-mimic pair table: per pair the set it enters, female carriers,
  observed and expected of the feature, the feature alone (observed and
  expected) or with other core features, with coded DLB/PD; the parkinsonism
  components in the source data and console.
- Grouping table: per published grouping, a class against the contrasting
  class (disjoint female carriers), observed / expected core features in
  each and their rate ratio; contrasts not estimated are listed apart (for
  the caption). Rate ratios of >= 2 core features and coded DLB/PD in the
  source data and console; BH within each measure. PD risk and PD pleiotropy
  (DESCRIPTIVE_ONLY) get class SIRs (class_sirs) and no contrast.
- Sjogren's figure: gap = first counted core-feature date minus
  the Sjogren's date; concurrent = within 1 year either side.
- Checks: per-condition SIRs and RoRs, condition-level checks, the sex
  differential across conditions, possible-mimic groups, Sjogren's gap
  checks; co-occurrence of >= 2 core features against independence by group
  (sheet cooccurrence_vs_independence); both weight designs of
  panel d with balance and weight change (sheet possible_mimics_ipw_designs);
  and (section 11; console and checks workbook only): the spread model refit
  without pemphigoid carriers, sheet pemphigoid_removed; AID diagnosis
  (earliest per patient and unit) against the AD diagnosis and the first
  counted core feature per condition and sex, sheet aid_dx_timing (its female
  rows are panel c); pemphigoid gap (first counted core feature minus the
  pemphigoid date, Sjogren's method) per sex, sheet pemphigoid_timing; the sex
  pattern per condition (console).

Inputs: the patient table, the note table, the DLB feature table, the coded
AID conditions, the medication exposure records, the propensity-score weights
(propensity_score.py), and the discriminant coefficients written by
dlb_core_analysis.py (run it first).
"""

import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.stats import norm
from scipy.stats.contingency import odds_ratio

import analysis_utils as au

warnings.filterwarnings("ignore")   # after the imports: statsmodels resets its own filters

# =============================================================================
# PATHS (relative to the data folder the script is run from)
# =============================================================================
COHORT_PATIENTS_PATH = ""  # patient table: AID status, sex, key dates (birth, AD diagnosis, death), analysis-population flag and coded DLB and PD diagnoses
COHORT_NOTES_PATH = ""  # note table: one row per clinical note, with its date
DLB_MENTIONS_PATH = ""  # DLB feature table: one row per documented feature, with its date
AID_CONDITIONS_PATH = ""  # coded AID conditions: one row per patient and condition, with the diagnosis date
MEDICATION_RECORDS_PATH = ""  # medication exposure records: drug class and exposure date
COEFFICIENTS_PATH = ""  # discriminant coefficients written by dlb_core_analysis.py, one per tested DLB feature
PS_WEIGHTS_PATH = ""  # propensity-score weights written by propensity_score.py, one per population patient
SOURCE_DATA_DIR = ""  # folder for the source-data workbooks written here

TABLE_MIMIC_FILE = "table_aid_condition_mimic_pairs.xlsx"
TABLE_GROUPING_FILE = "table_aid_condition_groupings.xlsx"
FIGURE_FILE = "figure_aid_condition_spread.xlsx"
EXTENDED_FILE = "extended_data_aid_condition_combinations.xlsx"
FIGURE_SJOGRENS_FILE = "figure_sjogrens_feature_timing.xlsx"
CHECKS_FILE = "checks_aid_conditions.xlsx"

# =============================================================================
# SETTINGS
# =============================================================================
NAN = float("nan")
CRIT = float(norm.ppf(0.975))
SEXES = [(1, "female"), (0, "male")]
ARMS = ["all_carriers", "isolated"]
IBD_UNIT = au.AID_IBD_UNIT
SJOGRENS = "sjogrens syndrome"
CELIAC = "celiac disease"
PEMPHIGOID = "pemphigoid"
MIN_IND_EVENTS = 5            # core features among a condition's female carriers for its own term
N_BOOT_EXCESS = 2000
N_BOOT_COOC = 1000
SPREAD_SEED = 20261102
CONCURRENT_YRS = 1
GAP_BIN_YRS = 0.5
N_BOOT = 2000
N_PERM_SJOGRENS = 5000
SJOGRENS_SEED = 20260804
PEMPHIGOID_SEED = 20261105    # pemphigoid gap bootstrap (check): + 1 female, + 0 male
TOST_MARGINS = [1.25, 1.50, 2.00]
MDE_POWER = 0.80
PASSTHROUGH_LOGROR_FRAC = 0.50
PASSTHROUGH_MAX_ITER = 3

CORE = "core"
CORE_FEATURES = [h for h, _, _ in au.DLB_CORE_BURDEN]
FEAT = {"F": "has_fluct", "V": "has_vis_hall", "P": "has_park_all_mckeith"}
FEAT_LABEL = {"F": "Fluctuating cognition", "V": "Visual hallucinations",
              "P": "Parkinsonism (McKeith)"}
PARK_COMPONENTS = [("has_tremor_mckeith", "rest_tremor"), ("has_brady_mckeith", "bradykinesia"),
                   ("has_rigid_mckeith", "rigidity")]
# Exact combinations of the three core features (F fluctuations, V visual hallucinations,
# P parkinsonism); display order and labels as the co-occurrence figure.
COMBOS = ["none", "P", "F", "V", "F+P", "V+P", "F+V", "F+V+P"]
COMBO_K = {c: (0 if c == "none" else c.count("+") + 1) for c in COMBOS}
COMBO_LABELS = {"none": "None", "P": "Park.", "F": "Fluct. cog.", "V": "Vis. hall.",
                "F+P": "Park. + Fluct. cog.", "V+P": "Park. + Vis. hall.",
                "F+V": "Fluct. cog. + Vis. hall.", "F+V+P": "All three"}
# The tested DLB features are the rows of the coefficient file written by
# dlb_core_analysis.py (read in section 1), so this script follows its feature set.
NONSEQ_EXCLUDE = ["has_vis_seq", "has_exec_seq"]
FEATURE_LABELS = {"has_fluct": "Fluctuating cognition", "has_vis_hall": "Visual hallucinations",
                  "has_park_all_mckeith": "Parkinsonism (McKeith)",
                  "has_ortho": "Orthostatic hypotension",
                  "has_vis_seq": "Visuospatial, pre/peri-diagnosis",
                  "has_aud_hall": "Auditory hallucinations", "has_incont": "Urinary incontinence",
                  "has_hyper": "Hypersomnia",
                  "has_exec_seq": "Executive function, pre/peri-diagnosis",
                  "has_syncope": "Syncope", "has_constipation": "Constipation",
                  "has_rbd_any": "REM sleep behavior disorder (RBD)",
                  "has_pi_mckeith": "Postural instability (McKeith)",
                  "has_systematized_delusions": "Systematized delusions",
                  "has_neuro_sens": "Neuroleptic sensitivity",
                  "has_transient_unresponsiveness": "Transient unresponsiveness",
                  "has_park_all_ever": "Parkinsonism, any time",
                  "has_brady_or_rigid_mckeith": "Bradykinesia or rigidity (McKeith)",
                  "has_hyposmia": "Hyposmia", "coded_dlb": "Coded DLB diagnosis",
                  "coded_pd": "Coded PD diagnosis", "coded_lbd": "Coded DLB or PD diagnosis"}
BLOCKS = {"core_mckeith": ["has_fluct", "has_vis_hall", "has_park_all_mckeith"],
          "autonomic": ["has_ortho", "has_incont", "has_syncope", "has_constipation"],
          "cognitive_seq": ["has_vis_seq", "has_exec_seq"],
          "other_supportive": ["has_aud_hall", "has_hyper"],
          "context": ["has_rbd_any", "has_park_all_ever", "has_brady_or_rigid_mckeith",
                      "has_hyposmia"],
          "diagnosis": ["coded_dlb", "coded_pd"]}
NEGATIVE_CONTROL = ["has_ortho", "has_incont", "has_syncope", "has_constipation", "has_vis_seq",
                    "has_exec_seq", "has_aud_hall"]
OUTCOME_LABELS = {"core": "Core features", "core_rbd": "Core and RBD burden",
                  "core_1plus": "1 or more core features", "core_2plus": "2 or more core features",
                  "core_3plus": "All 3 core features", "core_rbd_1plus": "Core and RBD, 1 or more",
                  "core_rbd_2plus": "Core and RBD, 2 or more",
                  "core_rbd_3plus": "Core and RBD, 3 or more",
                  "core_rbd_4plus": "Core and RBD, all 4", "autonomic": "Autonomic burden",
                  "cognitive_seq": "Cognitive-sequence burden",
                  "nonspecific": "Non-specific burden (negative control)",
                  "core_beta": "Core features, weighted by the discriminant coefficients",
                  "nonseq_beta": "Tested features without the sequence features, weighted by the "
                                 "discriminant coefficients", **FEATURE_LABELS}
OUTCOME_FAMILIES = ([(b, BLOCKS[b]) for b in ("core_mckeith", "autonomic", "cognitive_seq",
                                              "other_supportive")]
                    + [("pooled_core", ["core"]), ("pooled_autonomic", ["autonomic"]),
                       ("pooled_cognitive_seq", ["cognitive_seq"]),
                       ("threshold_core", ["core_1plus", "core_2plus", "core_3plus"]),
                       ("core_rbd", ["core_rbd"]),
                       ("core_rbd_threshold", ["core_rbd_1plus", "core_rbd_2plus",
                                               "core_rbd_3plus", "core_rbd_4plus"]),
                       ("negative_control", ["nonspecific"]),
                       ("context", BLOCKS["context"]), ("diagnosis", BLOCKS["diagnosis"])])
NOT_IN_FDR = {"context", "diagnosis", "negative_control"}

# Published groupings of the conditions (prespecified; IBD as one unit). A unit not in a
# grouping's map takes its default class, or is not classified.
SEROPOSITIVE = ["rheumatoid arthritis", "systemic lupus erythematosus", "sjogrens syndrome",
                "scleroderma", "dermatomyositis", "polymyositis",
                "mixed connective tissue disease", "myasthenia gravis", "neuromyelitis optica",
                "type 1 diabetes", "graves disease", "hashimotos thyroiditis", "celiac disease",
                "primary biliary cholangitis", "autoimmune hepatitis", "pernicious anemia",
                "pemphigus", "pemphigoid", "addisons disease"]
ANCA = ["granulomatosis with polyangiitis", "anca-associated vasculitis",
        "microscopic polyangiitis", "eosinophilic granulomatosis"]


def _m(label, units):
    return {u: label for u in units}


GROUPINGS = {
    "serology": {"label": "Serology", "map": {
        **_m("seronegative", ["psoriasis", "psoriatic arthritis", "ankylosing spondylitis",
                              IBD_UNIT]), **_m("seropositive", SEROPOSITIVE)}},
    "systemic": {"label": "Organ involvement", "map": {
        **_m("systemic", ["rheumatoid arthritis", "scleroderma", "systemic lupus erythematosus",
                          "sjogrens syndrome", "polymyositis", "dermatomyositis"]),
        **_m("organ_specific", ["type 1 diabetes", "multiple sclerosis", "graves disease",
                                "hashimotos thyroiditis", "pernicious anemia",
                                "addisons disease", "vitiligo", "myasthenia gravis"])}},
    "interferon": {"label": "Type I interferon", "map": _m(
        "ifn_core", ["systemic lupus erythematosus", "sjogrens syndrome", "scleroderma",
                     "dermatomyositis"]), "default": "ifn_other"},
    "genetic_cluster": {"label": "Genetic cluster", "map": {
        **_m("connective_tissue", ["systemic lupus erythematosus", "sjogrens syndrome",
                                   "dermatomyositis", "polymyositis"]),
        **_m("ra_gpa", ["rheumatoid arthritis", "granulomatosis with polyangiitis"]),
        **_m("hepatic", ["autoimmune hepatitis", "primary biliary cholangitis"]),
        **_m("ibd", [IBD_UNIT, "primary sclerosing cholangitis"]),
        **_m("celiac", ["celiac disease", "dermatitis herpetiformis"]),
        **_m("endocrine", ["type 1 diabetes", "addisons disease", "hashimotos thyroiditis",
                           "graves disease"]),
        **_m("psoriatic", ["psoriasis", "psoriatic arthritis"]),
        "multiple sclerosis": "ms", "myasthenia gravis": "mg",
        "ankylosing spondylitis": "as", "sarcoidosis": "sarcoidosis"}},
    "continuum": {"label": "Immunological continuum", "map": {
        **_m("autoimmune", ["rheumatoid arthritis", "systemic lupus erythematosus",
                            "sjogrens syndrome", "type 1 diabetes"] + ANCA),
        **_m("mixed_mhc1", ["ankylosing spondylitis", "psoriatic arthritis", "behcets disease"]),
        IBD_UNIT: "autoinflammatory"}},
    # PD risk (descriptive only: panel a membership and class SIRs; no contrast).
    # Classification evidence:
    # published meta-analyses and four multi-disease population studies: Sweden (Li X,
    # Sundquist 2012; SIRs), Denmark (Rugbjerg 2009; case-control), Taiwan (Chang 2018),
    # UK Biobank (Ma 2024). Raised: meta-analysis CI excludes 1 above, or significantly
    # raised in >= 2 of the four studies and lowered in none. Limited or conflicting
    # evidence: significantly raised in exactly one study, or studies disagree. Not raised:
    # data exist, no significant raised result. Not classified: no data (no default class).
    #   Raised
    #     pemphigoid              Li 2023 meta OR 2.67 (2.15-3.31), 17 studies
    #     IBD                     Li HX 2023 meta RR 1.17 (1.03-1.33), 14 studies
    #     Sjogren's               Li 2023 meta OR 1.61 (1.24-2.09); Taiwan HR 1.56 (1.35-1.79)
    #     Graves'                 Li 2023 meta OR 1.45 (1.24-1.70); Sweden SIR 1.63 (1.39-1.90)
    #     psoriasis               Ungprasert 2016 meta RR 1.38 (1.15-1.66)
    #     ankylosing spondylitis  Luo 2025 meta OR 1.55 (1.31-1.83)
    #     Behcet's                He 2022 meta RR 1.93 (1.07-3.49)
    #     multiple sclerosis      Li 2023 meta OR 2.02 (0.87-4.70); Sweden SIR 1.66
    #                             (1.05-2.50); UK Biobank HR 2.49 (1.55-4.02) (>= 2 studies)
    #   Limited or conflicting evidence
    #     rheumatoid arthritis    Li 2023 meta 0.79 (0.61-1.03); Taiwan HR 1.14 (1.03-1.28)
    #                             raised; Denmark ~30% lower (systemic-involvement subgroup,
    #                             primarily RA); Sweden 1.07 ns (studies disagree)
    #     polymyalgia rheumatica  He 2022 meta 1.13 (0.91-1.40), 2 studies; Sweden SIR 1.25
    #                             (1.01-1.53) (one study)
    #     Hashimoto's             Sweden SIR 2.40 (1.86-3.06) (one study)
    #     pernicious anemia       Sweden SIR 1.49 (1.13-1.94) (one study)
    #   Not raised
    #     SLE                     Li 2023 meta 0.82 (0.66-1.03); Sweden 1.00 ns; Taiwan ns
    #     celiac disease          Li 2023 meta 1.16 (0.79-1.69); Sweden 1.01 ns
    #     type 1 diabetes         Sweden 3.03 (0.29-11.14), ns
    #     sarcoidosis             Sweden 1.33 (0.91-1.88), ns
    #     scleroderma             Sweden 0.91 (0.47-1.60), ns
    #     dermatomyositis,
    #     polymyositis            Sweden polymyositis/dermatomyositis 0.87 ns
    #     GPA                     Sweden 1.15 (0.90-1.45) ns
    #   Not classified: giant cell arteritis, lichen planus, alopecia areata, vitiligo and
    #   every other unit (no meta-analysis, no estimate in the four studies).
    "pd_risk": {"label": "PD risk", "map": {
        **_m("pd_raised", ["pemphigoid", IBD_UNIT, "sjogrens syndrome", "graves disease",
                           "psoriasis", "ankylosing spondylitis", "behcets disease",
                           "multiple sclerosis"]),
        **_m("pd_mixed", ["rheumatoid arthritis", "polymyalgia rheumatica",
                          "hashimotos thyroiditis", "pernicious anemia"]),
        **_m("pd_not_raised", ["systemic lupus erythematosus", "celiac disease",
                               "type 1 diabetes", "sarcoidosis", "scleroderma",
                               "dermatomyositis", "polymyositis",
                               "granulomatosis with polyangiitis"])}},
    # PD pleiotropy (descriptive only), Witoelar 2017 (JAMA Neurol) wording: strong/moderate
    # enrichment for Crohn's disease, ulcerative colitis and rheumatoid arthritis; weak for
    # type 1 diabetes, celiac disease, psoriasis and multiple sclerosis. Every other unit was
    # not studied: not classified (no default class).
    "pleiotropy": {"label": "PD pleiotropy", "map": {
        **_m("pleiotropic_strong", [IBD_UNIT, "rheumatoid arthritis"]),
        **_m("pleiotropic_weak", ["type 1 diabetes", "celiac disease", "psoriasis",
                                  "multiple sclerosis"])}},
}
CLASS_LABELS = {
    "seropositive": "Seropositive", "seronegative": "Seronegative", "systemic": "Systemic",
    "organ_specific": "Organ-specific", "ifn_core": "Type I interferon core",
    "ifn_other": "Other conditions", "connective_tissue": "Connective tissue",
    "ra_gpa": "Rheumatoid arthritis / GPA", "hepatic": "Hepatic", "ibd": "Inflammatory bowel",
    "celiac": "Celiac", "endocrine": "Endocrine", "psoriatic": "Psoriatic",
    "ms": "Multiple sclerosis", "mg": "Myasthenia gravis", "as": "Ankylosing spondylitis",
    "sarcoidosis": "Sarcoidosis", "autoimmune": "Autoimmune", "mixed_mhc1": "Mixed (MHC class I)",
    "autoinflammatory": "Autoinflammatory", "pd_raised": "Raised PD risk",
    "pd_mixed": "Limited or conflicting evidence on PD risk",
    "pd_not_raised": "PD risk not raised",
    "pleiotropic_strong": "Pleiotropy with PD, moderate to strong",
    "pleiotropic_weak": "Pleiotropy with PD, weak"}
# Two-class groupings: (class, contrasting class); the others: each class vs the rest,
# except the descriptive-only groupings, which get class SIRs and no contrast.
TWO_CLASS = {"serology": ("seropositive", "seronegative"),
             "systemic": ("systemic", "organ_specific"), "interferon": ("ifn_core", "ifn_other")}
DESCRIPTIVE_ONLY = ["pd_risk", "pleiotropy"]
PLEIOTROPY_CLASSES = ["pleiotropic_strong", "pleiotropic_weak"]
# Figure panel a: membership columns (filled / shaded / open / blank; drawn by the figure script).
MEMBERSHIP = ["serology", "systemic", "interferon", "pd_risk", "pleiotropy"]
UNASSIGNED = "not_classified"
GROUPING_MEASURES = [("core", "Core features"), ("core_2plus", "2 or more core features"),
                     ("coded_lbd", "Coded DLB or PD")]


def class_of(g, unit):
    return GROUPINGS[g]["map"].get(unit, GROUPINGS[g].get("default", UNASSIGNED))


# Possible-mimic pairs and their cumulative sets: analysis_utils.AID_MIMIC_PAIRS and
# AID_MIMIC_SETS (with their definitions).
MIMIC_TIER_LABELS = {"narrow": "Narrow", "wider": "Wider", "reported": "All reported",
                     "check": "Check"}
MIMIC_CONDITIONS = sorted({c for c, _, _ in au.AID_MIMIC_PAIRS})
# Explicit co-member stripping of the decomposition check.
COMORBID_STRIP = {"systemic lupus erythematosus": ["rheumatoid arthritis", "sjogrens syndrome"],
                  "psoriatic arthritis": ["psoriasis"], "pemphigus": ["pemphigoid"],
                  "anca-associated vasculitis": ["granulomatosis with polyangiitis"]}
CONDITION_LABELS = {
    IBD_UNIT: "Inflammatory bowel disease", "sjogrens syndrome": "Sjögren's syndrome",
    "crohns disease": "Crohn's disease", "hashimotos thyroiditis": "Hashimoto's thyroiditis",
    "graves disease": "Graves' disease", "addisons disease": "Addison's disease",
    "behcets disease": "Behçet's disease", "cns vasculitis": "CNS vasculitis",
    "anca-associated vasculitis": "ANCA-associated vasculitis",
    "type 1 diabetes": "Type 1 diabetes"}


def condition_label(unit):
    return CONDITION_LABELS.get(unit, unit[:1].upper() + unit[1:])


# =============================================================================
# SMALL HELPERS
# =============================================================================
def mde_ratio(e):
    """Minimum detectable SIR at alpha 0.05 and 80% power given E."""
    if not (np.isfinite(e) and e > 0):
        return NAN
    return float(np.exp((CRIT + norm.ppf(MDE_POWER)) / np.sqrt(e)))


def tost_inside(est, lo, hi, margin):
    """1 when the CI lies inside [1/margin, margin] (equivalence), else 0."""
    if not (np.isfinite(est) and np.isfinite(lo) and np.isfinite(hi)) or lo <= 0:
        return NAN
    return int(lo >= 1.0 / margin and hi <= margin)


def exact_2x2(a, b, c, d):
    """Conditional MLE odds ratio, exact 95% CI and two-sided Fisher P."""
    t = [[int(a), int(b)], [int(c), int(d)]]
    res = odds_ratio(t, kind="conditional")
    ci = res.confidence_interval(0.95)
    return float(res.statistic), float(ci.low), float(ci.high), float(stats.fisher_exact(t)[1])


def bh_rows(df, family_cols, p_col, q_col, eligible):
    df[q_col] = NAN
    df[q_col.replace("q_", "fdr_family_size_")] = 0
    if not len(df):
        return df
    for _, idx in df.groupby(family_cols, dropna=False, sort=False).groups.items():
        t = [i for i in idx if eligible.loc[i] and pd.notna(df.loc[i, p_col])]
        if t:
            df.loc[t, q_col] = au.bh_fdr(df.loc[t, p_col].values)
            df.loc[t, q_col.replace("q_", "fdr_family_size_")] = len(t)
    return df


def dl_pool(betas, ses):
    """DerSimonian-Laird random-effects pool with tau, I2 and the 95% prediction
    interval (t with k - 2 degrees of freedom; log scale)."""
    b, s = np.asarray(betas, dtype=float), np.asarray(ses, dtype=float)
    ok = np.isfinite(b) & np.isfinite(s) & (s > 0)
    b, s = b[ok], s[ok]
    k = len(b)
    out = dict(k=k, pooled=NAN, se=NAN, lo=NAN, hi=NAN, p=NAN, Q=NAN, p_Q=NAN, I2=NAN,
               tau2=NAN, pred_lo=NAN, pred_hi=NAN)
    if k == 0:
        return out
    w = 1.0 / s ** 2
    fe = float((w * b).sum() / w.sum())
    Q = float((w * (b - fe) ** 2).sum())
    if k > 1:
        C = float(w.sum() - (w ** 2).sum() / w.sum())
        tau2 = max(0.0, (Q - (k - 1)) / C) if C > 0 else 0.0
        p_Q = float(stats.chi2.sf(Q, k - 1))
        I2 = float(max(0.0, (Q - (k - 1)) / Q) * 100) if Q > 0 else 0.0
    else:
        tau2, p_Q, I2 = 0.0, NAN, NAN
    wr = 1.0 / (s ** 2 + tau2)
    pooled = float((wr * b).sum() / wr.sum())
    se = float(np.sqrt(1.0 / wr.sum()))
    out.update(pooled=pooled, se=se, lo=pooled - CRIT * se, hi=pooled + CRIT * se,
               p=float(2 * norm.sf(abs(pooled / se))), Q=Q, p_Q=p_Q, I2=I2, tau2=tau2)
    if k > 2:
        half = float(stats.t.ppf(0.975, k - 2) * np.sqrt(tau2 + se ** 2))
        out.update(pred_lo=pooled - half, pred_hi=pooled + half)
    return out


def ratio_row(y, e, g, labels):
    """Offset Poisson rate ratio of group 1 vs group 0 (HC0), gated."""
    y, e, g = (np.asarray(v, dtype=float) for v in (y, e, g))
    n1, n0 = int((g == 1).sum()), int((g == 0).sum())
    why = au.rate_ratio_floor_reason(n1, n0, y[g == 1].sum(), y[g == 0].sum(), labels)
    coef, se = au.log_rate_ratio_hc0(y, e, g) if not why else (NAN, NAN)
    if not why and not (np.isfinite(coef) and np.isfinite(se) and se > 0):
        why = "no expected events in a group"
    est = not why
    return {"n_1": n1, "n_0": n0, "o_1": float(y[g == 1].sum()), "e_1": float(e[g == 1].sum()),
            "o_0": float(y[g == 0].sum()), "e_0": float(e[g == 0].sum()),
            "status": "estimated" if est else "not_estimated", "reason": why,
            "coef": coef if est else NAN, "se": se if est else NAN,
            "ratio": float(np.exp(coef)) if est else NAN,
            "ci_low": float(np.exp(coef - CRIT * se)) if est else NAN,
            "ci_high": float(np.exp(coef + CRIT * se)) if est else NAN,
            "p": float(2 * norm.sf(abs(coef / se))) if est else NAN}


def sir_cols(o, e, prefix=""):
    v = au.sir_values(o, e)
    return {f"{prefix}o": o, f"{prefix}e": e, f"{prefix}sir": v["sir"],
            f"{prefix}sir_ci_low": v["ci_low"], f"{prefix}sir_ci_high": v["ci_high"],
            f"{prefix}p": v["p"]}


# =============================================================================
# SECTION 1 -- POPULATION, DLB FLAGS, CONDITIONS, STRATA, OUTCOMES
# =============================================================================
au.section("SECTION 1 -- POPULATION, DLB FLAGS, CONDITIONS, STRATA, OUTCOMES")
pop, cond, UNITS, ceil, strata_diag = au.aid_condition_population(
    COHORT_PATIENTS_PATH, COHORT_NOTES_PATH, DLB_MENTIONS_PATH, AID_CONDITIONS_PATH)
pop = au.add_ps_model_columns(au.load_ps_weights(PS_WEIGHTS_PATH, pop))
pop["coded_lbd"] = ((pop["coded_dlb"] == 1) | (pop["coded_pd"] == 1)).astype(int)
pop["autonomic"] = pop[BLOCKS["autonomic"]].sum(axis=1)
pop["cognitive_seq"] = pop[BLOCKS["cognitive_seq"]].sum(axis=1)
pop["nonspecific"] = pop[NEGATIVE_CONTROL].sum(axis=1)
code = np.full(len(pop), "none", dtype=object)
for c in COMBOS[1:]:
    parts = c.split("+")
    m = np.ones(len(pop), dtype=bool)
    for f in "FVP":
        m &= (pop[FEAT[f]].to_numpy() == 1) == (f in parts)
    code[m] = c
pop["combo"] = code
for c in COMBOS:
    pop["cb__" + c] = (pop["combo"] == c).astype(int)
coefs = au.load_dlb_discriminant_coefficients(COEFFICIENTS_PATH)
POOL_FEATURES = list(coefs)                     # every tested feature, NaN if not in the model
missing = [f for f in POOL_FEATURES if f not in pop.columns]
if missing:
    raise KeyError(f"dlb_discriminant_coefficients: features without a patient flag {missing}; "
                   f"analysis_utils and dlb_core_analysis.py must define the same features")
IN_MODEL = [f for f in POOL_FEATURES if np.isfinite(coefs[f])]
BETA_OUTCOMES = []
if all(f in IN_MODEL for f in CORE_FEATURES):
    pop["core_beta"] = sum(pop[f] * coefs[f] for f in CORE_FEATURES)
    pop["nonseq_beta"] = sum(pop[f] * coefs[f] for f in IN_MODEL if f not in NONSEQ_EXCLUDE)
    BETA_OUTCOMES = ["core_beta", "nonseq_beta"]
print(f"  Discriminant coefficients: {len(POOL_FEATURES)} tested features, {len(IN_MODEL)} in "
      f"the model; weighted outcomes {'built' if BETA_OUTCOMES else 'not built (a core feature is not in the model)'}")
EXPECTED_OUTCOMES = sorted(set(
    ["core", "core_rbd", "autonomic", "cognitive_seq", "nonspecific", "coded_lbd"]
    + [f"core_{k}plus" for k in (1, 2, 3)] + [f"core_rbd_{k}plus" for k in (1, 2, 3, 4)]
    + POOL_FEATURES + [f for v in BLOCKS.values() for f in v] + BETA_OUTCOMES
    + ["cb__" + c for c in COMBOS]))
FALLBACK = {}
for o in EXPECTED_OUTCOMES:
    pop["e__" + o], fb = au.expected_counts(pop, o)
    FALLBACK[o] = int((fb & (pop["aid_status"] == 1)).sum())
nodlb_ref = pop["coded_dlb"] == 0
nolbd_ref = pop["coded_lbd"] == 0
for o in ["core"] + BLOCKS["core_mckeith"]:
    pop["e_nodlb__" + o], _ = au.expected_counts(pop, o, reference=nodlb_ref)
print(f"  AID+ patients priced at the overall same-sex rate (stratum without AID- support), "
      f"core features: {FALLBACK['core']:,}")

AIDPOS = pop["aid_status"] == 1
FEMALE = pop["female"] == 1
WF = AIDPOS & FEMALE
ISOLATED = pop["n_aids"] == 1
N_AID_POS = {1: int(WF.sum()), 0: int((AIDPOS & ~FEMALE).sum())}


def carriers(u):
    return AIDPOS & (pop["cond__" + u] == 1)


def arm_mask(mask, arm):
    return mask & ISOLATED if arm == "isolated" else mask


def ror_row(mask, outcome, exclude_dlb=False):
    """SIRs and RoR (sir_ror_row) of the AID+ patients in mask."""
    d = pop[mask & AIDPOS]
    e = "e__" + outcome
    if exclude_dlb:
        d = d[d["coded_dlb"] == 0]
        e = "e_nodlb__" + outcome
    return au.sir_ror_row(d["female"], d[outcome], d[e])


NF = {u: int((carriers(u) & FEMALE).sum()) for u in UNITS}
CONDITIONS = sorted([u for u in UNITS if NF[u] >= au.MIN_GROUP_N],
                    key=lambda u: condition_label(u).lower())
OTHER_UNITS = [u for u in UNITS if u not in CONDITIONS]
OTHER_MASK = WF & (pop[[f"cond__{u}" for u in OTHER_UNITS]].sum(axis=1) > 0) \
    if OTHER_UNITS else pd.Series(False, index=pop.index)
print(f"  AID+ females {N_AID_POS[1]:,}; conditions with >= {au.MIN_GROUP_N} female carriers: "
      f"{len(CONDITIONS)}; other conditions: {len(OTHER_UNITS)} ({int(OTHER_MASK.sum()):,} "
      f"females)")


# =============================================================================
# SECTION 2 -- CONDITION TABLES (ALL CARRIERS; ISOLATED CARRIERS)
# =============================================================================
au.section("SECTION 2 -- PER CONDITION: SIR FEMALE, SIR MALE, RoR (CORE FEATURES)")
cond_rows = []
for arm in ARMS:
    for u in UNITS:
        m = arm_mask(carriers(u), arm)
        if not m.any():
            continue
        r = ror_row(m, CORE)
        cond_rows.append({"arm": arm, "condition": u, "row_label": condition_label(u),
                          "pleiotropic": int(class_of("pleiotropy", u) in PLEIOTROPY_CLASSES),
                          "mimic": int(u in MIMIC_CONDITIONS), "n_carriers": int(m.sum()), **r,
                          "mde_female": mde_ratio(r["e_female"]),
                          "mde_male": mde_ratio(r["e_male"])})
conds = pd.DataFrame(cond_rows)
conds = bh_rows(conds, ["arm"], "p_sir_female", "q_sir_female",
                conds["status_sir_female"] == "estimated")
conds = bh_rows(conds, ["arm"], "p_sir_male", "q_sir_male",
                conds["status_sir_male"] == "estimated")
conds = bh_rows(conds, ["arm"], "p_ror", "q_ror", conds["status_ror"] == "estimated")
for arm in ARMS:
    d = conds[conds["arm"] == arm]
    print(f"  {arm}: {len(d)} units with a carrier; SIR female estimated "
          f"{int((d['status_sir_female'] == 'estimated').sum())}, RoR estimated "
          f"{int((d['status_ror'] == 'estimated').sum())}")


def ci_txt(v, lo, hi):
    return "-" if not np.isfinite(v) else f"{v:.2f} ({lo:.2f}-{hi:.2f})"


def q_txt(q):
    return "-" if not np.isfinite(q) else ("<.001" if q < 0.001 else f"{q:.3f}")


def cond_line(lbl, r):
    print(f"    {lbl[:31]:<32}{int(r['n_female']):>5} {r['o_female']:>4.0f}/{r['e_female']:<7.2f}"
          f"{ci_txt(r['sir_female'], r['sir_female_ci_low'], r['sir_female_ci_high']):<19}"
          f"{q_txt(r.get('q_sir_female', NAN)):>6}  {int(r['n_male']):>5} "
          f"{r['o_male']:>4.0f}/{r['e_male']:<7.2f}"
          f"{ci_txt(r['sir_male'], r['sir_male_ci_low'], r['sir_male_ci_high']):<19}"
          f"{q_txt(r.get('q_sir_male', NAN)):>6}  "
          f"{ci_txt(r['ror'], r['ror_ci_low'], r['ror_ci_high']):<19}{q_txt(r.get('q_ror', NAN)):>6}")


for arm in ARMS:
    au.sub(f"Core features per condition ({arm.replace('_', ' ')})")
    print(f"    {'Condition':<32}{'nF':>5} {'O/E F':<12}{'SIR F (95% CI)':<19}{'Q':>6}  {'nM':>5} "
          f"{'O/E M':<12}{'SIR M (95% CI)':<19}{'Q':>6}  {'RoR (95% CI)':<19}{'Q':>6}")
    for _, r in conds[conds["arm"] == arm].sort_values("row_label").iterrows():
        cond_line(r["row_label"], r)
# Sex differential across conditions: DerSimonian-Laird pool of the condition RoRs
d = conds[(conds["arm"] == "all_carriers") & (conds["status_ror"] == "estimated")]
pr = dl_pool(d["ror_coef"], d["ror_se"])
ror_pool = pd.DataFrame([{"k": pr["k"], "pooled_ror": float(np.exp(pr["pooled"])) if pr["k"]
                          else NAN, "ci_low": float(np.exp(pr["lo"])) if pr["k"] else NAN,
                          "ci_high": float(np.exp(pr["hi"])) if pr["k"] else NAN, "p": pr["p"],
                          "tau": float(np.sqrt(pr["tau2"])) if pr["k"] else NAN,
                          "I2": pr["I2"], "Q": pr["Q"], "p_Q": pr["p_Q"]}])
print(f"  Condition RoRs pooled (k {pr['k']}): {ror_pool['pooled_ror'].iloc[0]:.2f}; "
      f"I2 {pr['I2']:.0f}%")


# =============================================================================
# SECTION 3 -- PANEL a: SPREAD ACROSS CONDITIONS (FEMALES, MUTUALLY ADJUSTED)
# =============================================================================
au.section("SECTION 3 -- SPREAD: FEMALES' MUTUALLY ADJUSTED RATE RATIO PER CONDITION")


def spread_model(mask, exclude=()):
    """The spread model (panel a) on the females in mask, without the units in exclude:
    per condition the mutually adjusted rate ratio of core features, their DerSimonian-
    Laird pool and the model terms. Returns (spread, spread_pool, model_terms, gated)."""
    units = [u for u in UNITS if u not in exclude]
    conditions = [u for u in CONDITIONS if u not in exclude]
    ev = {u: float(pop.loc[carriers(u) & mask, CORE].sum()) for u in units}
    own = [u for u in conditions if ev[u] >= MIN_IND_EVENTS]
    gated = [u for u in conditions if u not in own]
    small = [u for u in units if u not in own]
    terms = {u: pop["cond__" + u] for u in own}
    if small:
        terms["smaller_conditions"] = (pop[[f"cond__{u}" for u in small]].sum(axis=1)
                                       > 0).astype(int)
    terms["per_extra_condition"] = (pd.concat(list(terms.values()), axis=1).sum(axis=1) - 1) \
        .clip(lower=0)
    d = pop[mask & (pop["e__core"] > 0)]
    X = pd.DataFrame({k: v[d.index].astype(float) for k, v in terms.items()}, index=d.index)
    X = X.loc[:, X.std() > 0]
    X.insert(0, "const", 1.0)
    fit = sm.GLM(d[CORE].astype(float), X, family=sm.families.Poisson(),
                 offset=np.log(d["e__core"].astype(float))).fit(cov_type="HC0", maxiter=200)
    spread_rows = []
    for u in conditions:
        rec = {"condition": u, "row_label": condition_label(u),
               "n_female": int((carriers(u) & mask).sum()),
               "core_features_female": ev[u], "estimated": int(u in own and u in fit.params),
               "reason": "" if u in own else f"fewer than {MIN_IND_EVENTS} core features among "
                                             f"female carriers"}
        if rec["estimated"]:
            b_, s_ = float(fit.params[u]), float(fit.bse[u])
            rec.update(coef=b_, se=s_, rate_ratio=float(np.exp(b_)),
                       ci_low=float(np.exp(b_ - CRIT * s_)),
                       ci_high=float(np.exp(b_ + CRIT * s_)), p=float(2 * norm.sf(abs(b_ / s_))))
        else:
            rec.update(coef=NAN, se=NAN, rate_ratio=NAN, ci_low=NAN, ci_high=NAN, p=NAN)
        for g in GROUPINGS:
            rec["class_" + g] = class_of(g, u)
        spread_rows.append(rec)
    spread = pd.DataFrame(spread_rows)
    est_ = spread[spread["estimated"] == 1]
    het = dl_pool(est_["coef"], est_["se"])
    spread_pool = pd.DataFrame([{
        "k": het["k"], "pooled": float(np.exp(het["pooled"])),
        "ci_low": float(np.exp(het["lo"])), "ci_high": float(np.exp(het["hi"])), "p": het["p"],
        "tau": float(np.sqrt(het["tau2"])), "I2": het["I2"], "Q": het["Q"], "p_Q": het["p_Q"],
        "prediction_low": float(np.exp(het["pred_lo"])) if np.isfinite(het["pred_lo"]) else NAN,
        "prediction_high": float(np.exp(het["pred_hi"])) if np.isfinite(het["pred_hi"])
        else NAN,
        "n_conditions_not_estimated": len(gated),
        "per_extra_condition_rr": float(np.exp(fit.params["per_extra_condition"]))
        if "per_extra_condition" in fit.params else NAN,
        "smaller_conditions_rr": float(np.exp(fit.params["smaller_conditions"]))
        if "smaller_conditions" in fit.params else NAN}])
    model_terms = pd.DataFrame({"term": fit.params.index, "coef": fit.params.values,
                                "se": fit.bse.values, "p": fit.pvalues.values})
    return spread, spread_pool, model_terms, gated


spread, spread_pool, model_terms, GATED = spread_model(FEMALE)
het = {"k": int(spread_pool["k"].iloc[0]), "I2": spread_pool["I2"].iloc[0]}
print(f"  {het['k']} conditions estimated; pooled {spread_pool['pooled'].iloc[0]:.2f} "
      f"({spread_pool['ci_low'].iloc[0]:.2f}-{spread_pool['ci_high'].iloc[0]:.2f}); tau "
      f"{spread_pool['tau'].iloc[0]:.2f}; I2 {het['I2']:.0f}%; 95% prediction interval "
      f"{spread_pool['prediction_low'].iloc[0]:.2f}-{spread_pool['prediction_high'].iloc[0]:.2f}")
print(f"  Not estimated: {', '.join(condition_label(u) for u in GATED) or 'none'}")


# =============================================================================
# SECTION 4 -- PANEL b: EXCESS CORE FEATURES, SINGLE VS SEVERAL (AID+ FEMALES)
# =============================================================================
au.section("SECTION 4 -- EXCESS CORE FEATURES BY CONDITION, SINGLE VS TWO OR MORE")
x_single = sum(pop["cb__" + c] - pop["e__cb__" + c] for c in COMBOS if COMBO_K[c] == 1)
x_multi = sum(COMBO_K[c] * (pop["cb__" + c] - pop["e__cb__" + c]) for c in COMBOS
              if COMBO_K[c] >= 2)
pop["n_units"] = pop[[f"cond__{u}" for u in UNITS]].sum(axis=1).astype(int)
w_split = (1.0 / pop["n_units"].where(pop["n_units"] > 0)).fillna(0)


def excess(name, wt, m):
    xm = wt[m] * x_multi[m]
    return {"condition": name, "single": float((wt[m] * x_single[m]).sum()),
            "multi": float(xm.sum()), "multi_coded_lbd": float(xm[pop.loc[m, "coded_lbd"] == 1]
                                                               .sum())}


exc_rows = [excess(u, w_split, carriers(u) & FEMALE) for u in CONDITIONS]
if OTHER_UNITS:
    exc_rows.append(excess("other_conditions",
                           w_split * (pop[[f"cond__{u}" for u in OTHER_UNITS]] == 1).sum(axis=1),
                           OTHER_MASK))
excess_by_condition = pd.DataFrame(exc_rows)
excess_by_condition["multi_no_coded_lbd"] = excess_by_condition["multi"] \
    - excess_by_condition["multi_coded_lbd"]
excess_by_condition["total"] = excess_by_condition["single"] + excess_by_condition["multi"]
excess_by_condition["row_label"] = [("Other conditions" if u == "other_conditions"
                                     else condition_label(u))
                                    for u in excess_by_condition["condition"]]
excess_by_condition["n_female"] = [int(OTHER_MASK.sum()) if u == "other_conditions" else NF[u]
                                   for u in excess_by_condition["condition"]]
tot_all = float((pop.loc[WF, "core"] - pop.loc[WF, "e__core"]).sum())
excess_by_condition["share_of_total_pct"] = 100 * excess_by_condition["total"] / tot_all \
    if tot_all else NAN
# totals with bootstrap CIs (AID+ females resampled; expected values fixed)
xs, xm_, lbd = (x_single[WF].to_numpy(dtype=float), x_multi[WF].to_numpy(dtype=float),
                pop.loc[WF, "coded_lbd"].to_numpy())
rng = np.random.default_rng(SPREAD_SEED)
bs = []
for _ in range(N_BOOT_EXCESS):
    i = rng.integers(0, len(xs), len(xs))
    bs.append((xs[i].sum(), xm_[i].sum(), xm_[i][lbd[i] == 1].sum(), xs[i].sum() + xm_[i].sum()))
bs = np.array(bs)
lo_, hi_ = np.percentile(bs, [2.5, 97.5], axis=0)
excess_totals = pd.DataFrame({
    "component": ["single_feature", "two_or_more", "two_or_more_coded_lbd", "total"],
    "excess": [xs.sum(), xm_.sum(), xm_[lbd == 1].sum(), xs.sum() + xm_.sum()],
    "ci_low": lo_, "ci_high": hi_, "n_bootstrap": N_BOOT_EXCESS})
for _, r in excess_totals.iterrows():
    print(f"  {r['component']:<24} {r['excess']:7.1f} ({r['ci_low']:.1f} to {r['ci_high']:.1f})")


# =============================================================================
# SECTION 5 -- CO-OCCURRENCE BEYOND CHANCE (FEMALES; CHECK)
# =============================================================================
au.section("SECTION 5 -- TWO OR MORE CORE FEATURES: OBSERVED VS EXPECTED IF INDEPENDENT (CHECK)")
F_O = pop[[FEAT[f] for f in "FVP"]].to_numpy(dtype=float)
F_E = pop[["e__" + FEAT[f] for f in "FVP"]].to_numpy(dtype=float)
Y2 = pop["core_2plus"].to_numpy(dtype=float)


def cooccurrence(idx):
    """Observed number with >= 2 core features and the number expected if each patient's
    three features were independent (feature probability = AID- stratum rate scaled to the
    group's own observed / expected ratio)."""
    o, e = F_O[idx], F_E[idx]
    k = o.sum(axis=0) / np.where(e.sum(axis=0) > 0, e.sum(axis=0), np.nan)
    p_ = np.clip(e * k, 0, 1)
    pf, pv, pp = p_[:, 0], p_[:, 1], p_[:, 2]
    return float(Y2[idx].sum()), float((pf * pv + pf * pp + pv * pp - 2 * pf * pv * pp).sum())


AIDNEG = pop["aid_status"] == 0
COOC_GROUPS = [("females_aid_neg", "Females", "aid_neg", FEMALE & AIDNEG),
               ("females_aid_pos", "Females", "aid_pos", WF),
               ("females_no_lbd_aid_neg", "Females, coded DLB/PD excluded", "aid_neg",
                FEMALE & AIDNEG & nolbd_ref),
               ("females_no_lbd_aid_pos", "Females, coded DLB/PD excluded", "aid_pos",
                WF & nolbd_ref),
               ("females_coded_dlb", "Females, coded DLB", "reference",
                FEMALE & (pop["coded_dlb"] == 1))]
rng = np.random.default_rng(SPREAD_SEED + 1)
cooc_rows = []
for key, grp, kind, mk in COOC_GROUPS:
    idx = np.flatnonzero((mk & (pop["e__core"] > 0)).to_numpy())
    n = len(idx)
    o2, e2 = cooccurrence(idx) if n else (NAN, NAN)
    pct, lo, hi = au.wilson_ci(int(o2), n) if n else (NAN, NAN, NAN)
    br = [cooccurrence(rng.choice(idx, n, replace=True)) for _ in range(N_BOOT_COOC)] if n else []
    rr = np.array([a / b for a, b in br if b > 0])
    cooc_rows.append({"group": key, "group_label": grp, "aid": kind, "n": n, "observed": o2,
                      "observed_pct": pct, "observed_ci_low": lo, "observed_ci_high": hi,
                      "expected_if_independent": e2,
                      "expected_pct": 100 * e2 / n if n else NAN,
                      "ratio": o2 / e2 if n and e2 > 0 else NAN,
                      "ratio_ci_low": float(np.percentile(rr, 2.5)) if len(rr) else NAN,
                      "ratio_ci_high": float(np.percentile(rr, 97.5)) if len(rr) else NAN,
                      "n_bootstrap": N_BOOT_COOC})
cooccurrence_tab = pd.DataFrame(cooc_rows)
for _, r in cooccurrence_tab.iterrows():
    print(f"  {r['group']:<26} n {r['n']:>6,}  {r['observed_pct']:5.1f}% vs chance "
          f"{r['expected_pct']:5.1f}%  O/E {r['ratio']:.2f} ({r['ratio_ci_low']:.2f}-"
          f"{r['ratio_ci_high']:.2f})")


# =============================================================================
# SECTION 6 -- PANEL d (POSSIBLE MIMICS, IPW) AND THE POSSIBLE-MIMIC PAIR TABLE
# =============================================================================
au.section("SECTION 6 -- POSSIBLE MIMICS: IPW WITH THE CARRIERS REMOVED; PAIR PROFILES")
for f in "FVP":
    if "e__" + FEAT[f] not in pop:
        pop["e__" + FEAT[f]], _ = au.expected_counts(pop, FEAT[f])

# Panel d: female IPW odds ratio of >= 1 and >= 2 core features in all females (stored
# weights) and after removing the AID+ carriers (both sexes) of the conditions in each
# cumulative set; weights reused or refitted by au.ipw_weights_rule.
D_OUTCOMES = [("core_1plus", "1 or more core features", "logit"),
              ("core_2plus", "2 or more core features", "logit")]
FEM_ALL = pop[FEMALE]
d_rows, d_design_rows = [], []
ref_bal = au.ipw_balance(FEM_ALL, FEM_ALL[au.IPW_WEIGHT_COLUMN])
for y, lbl, model in D_OUTCOMES:
    r = au.ipw_female_fit(FEM_ALL, y, FEM_ALL[au.IPW_WEIGHT_COLUMN], model)
    d_rows.append({"set": "reference", "conditions": "", "n_conditions": 0,
                   "aid_pos_female_removed": 0, "aid_pos_female_kept": int(WF.sum()),
                   "outcome": y, "outcome_label": lbl, "odds_ratio": r["estimate"],
                   "ci_low": r["ci_low"], "ci_high": r["ci_high"], "p": r["p"],
                   "status": r["status"], "reason": r["reason"], "weights": "stored",
                   "weights_reason": "all patients (the main analysis)",
                   "max_female_ps_smd": au.ipw_max_ps_smd(ref_bal)})
for nm, tiers in au.AID_MIMIC_SETS:
    cs = au.aid_mimic_set_conditions(tiers, UNITS)
    rm = AIDPOS & (pop[[f"cond__{u}" for u in cs]].sum(axis=1) > 0) if cs else \
        pd.Series(False, index=pop.index)
    res = au.ipw_removal_fits(pop, ~rm, D_OUTCOMES)
    n_rm, n_kept = int((rm & FEMALE).sum()), int((WF & ~rm).sum())
    for y, lbl, model in D_OUTCOMES:
        for design in ("reuse", "refit"):
            r = res["fits"][(design, y)]
            d_design_rows.append({"set": nm, "design": design,
                                  "selected": int(design == res["design"]), "outcome": y,
                                  "outcome_label": lbl, **res["removed"],
                                  "aid_pos_female_kept": n_kept, **r,
                                  "max_female_ps_smd": au.ipw_max_ps_smd(res["balance"][design]),
                                  **{f"smd_weighted_{b['variable']}": b["smd_weighted"]
                                     for b in res["balance"][design]},
                                  **res["weight_change"], "rule_reason": res["reason"]})
        r = res["fits"][(res["design"], y)]
        d_rows.append({"set": nm, "conditions": "; ".join(condition_label(u) for u in cs),
                       "n_conditions": len(cs), "aid_pos_female_removed": n_rm,
                       "aid_pos_female_kept": n_kept, "outcome": y, "outcome_label": lbl,
                       "odds_ratio": r["estimate"], "ci_low": r["ci_low"], "ci_high": r["ci_high"],
                       "p": r["p"], "status": r["status"], "reason": r["reason"],
                       "weights": res["design"], "weights_reason": res["reason"],
                       "max_female_ps_smd": au.ipw_max_ps_smd(res["balance"][res["design"]])})
possible_mimics_ipw = pd.DataFrame(d_rows)
possible_mimics_ipw_designs = pd.DataFrame(d_design_rows)
au.sub("Possible mimics: female IPW odds ratio, AID+ carriers of each cumulative set of possible "
       "mimics removed")
for _, r in possible_mimics_ipw.iterrows():
    print(f"  {r['set']:<13} {r['outcome_label']:<24} removed {r['aid_pos_female_removed']:>4,} "
          f"kept {r['aid_pos_female_kept']:>4,} AID+ females  OR "
          f"{ci_txt(r['odds_ratio'], r['ci_low'], r['ci_high'])}  P {q_txt(r['p'])}  weights "
          f"{r['weights']} (max PS-covariate SMD {r['max_female_ps_smd']:.3f})")
for nm, _ in au.AID_MIMIC_SETS:
    rr = possible_mimics_ipw[possible_mimics_ipw["set"] == nm]
    if len(rr):
        print(f"  {nm}: weights rule: {rr['weights_reason'].iloc[0]}")

prof_rows = []
for c, f, t in au.AID_MIMIC_PAIRS:
    if c not in UNITS:
        prof_rows.append({"condition": c, "row_label": condition_label(c), "feature": f,
                          "feature_label": FEAT_LABEL[f], "tier": t,
                          "tier_label": MIMIC_TIER_LABELS[t], "n_female": 0,
                          "status": "no carriers"})
        continue
    S = carriers(c) & FEMALE
    has = S & (pop[FEAT[f]] == 1)
    others = pop[[FEAT[g] for g in "FVP" if g != f]].sum(axis=1)
    O, E = float(pop.loc[S, FEAT[f]].sum()), float(pop.loc[S, "e__" + FEAT[f]].sum())
    rec = {"condition": c, "row_label": condition_label(c), "feature": f,
           "feature_label": FEAT_LABEL[f], "tier": t, "tier_label": MIMIC_TIER_LABELS[t],
           "n_female": int(S.sum()), "status": "estimated" if S.sum() >= au.MIN_GROUP_N
           else "fewer than 10 female carriers", **sir_cols(O, E),
           "excess": max(0.0, O - E), "n_feature_alone": int((has & (others == 0)).sum()),
           "n_feature_with_other_core": int((has & (others >= 1)).sum()),
           "n_feature_coded_lbd": int((has & (pop["coded_lbd"] == 1)).sum()),
           "n_feature_alone_expected": float(pop.loc[S, "e__cb__" + f].sum())}
    if f == "P":
        for col, name in PARK_COMPONENTS:
            rec[f"n_park_{name}"] = int((has & (pop[col] == 1)).sum())
        rec["n_park_general_term_only"] = int((has & (pop["has_park_core_mckeith"] == 0)).sum())
        rec["n_park_rest_tremor_only"] = int((has & (pop["has_tremor_mckeith"] == 1)
                                              & (pop["has_brady_mckeith"] == 0)
                                              & (pop["has_rigid_mckeith"] == 0)).sum())
    prof_rows.append(rec)
mimic_profile = pd.DataFrame(prof_rows)
for _, r in mimic_profile.iterrows():
    if r["status"] == "no carriers":
        continue
    print(f"  {r['row_label'][:28]:<29}{r['feature']} {r['tier']:<10} n {r['n_female']:>4}  "
          f"O {r['o']:>4.0f} E {r['e']:>6.1f}  alone {r['n_feature_alone']:>3} (E "
          f"{r['n_feature_alone_expected']:.1f})  with other {r['n_feature_with_other_core']:>3}"
          f"  coded DLB/PD {r['n_feature_coded_lbd']:>3}")
    if r["feature"] == "P":
        print(f"      parkinsonism signs: rest tremor {r['n_park_rest_tremor']:.0f}, bradykinesia "
              f"{r['n_park_bradykinesia']:.0f}, rigidity {r['n_park_rigidity']:.0f}, general term only "
              f"{r['n_park_general_term_only']:.0f}, rest tremor as the only sign "
              f"{r['n_park_rest_tremor_only']:.0f}")

# Celiac disease and parkinsonism among AID+ females (two-sided exact)
cel_rows = []
if CELIAC in UNITS:
    d = pop[WF]
    cel, park = d["cond__" + CELIAC] == 1, d["has_park_all_mckeith"] == 1
    a_, b_ = int((cel & park).sum()), int((cel & ~park).sum())
    c_, d_ = int((~cel & park).sum()), int((~cel & ~park).sum())
    rec = {"group": "aid_pos_females", "celiac_park": a_, "celiac_no_park": b_,
           "no_celiac_park": c_, "no_celiac_no_park": d_}
    why = au.two_by_two_reason(a_, a_ + b_, c_, c_ + d_, "parkinsonism (McKeith)")
    if why:
        rec.update(status="not_tested", reason=why)
    else:
        orv, lo, hi, p_ = exact_2x2(a_, b_, c_, d_)
        ok_ = np.isfinite(orv) and orv > 0
        rec.update(status="tested", reason="", odds_ratio=orv if ok_ else NAN,
                   ci_low=lo if ok_ else NAN, ci_high=hi if ok_ else NAN, p=p_)
    cel_rows.append(rec)
celiac_parkinsonism = pd.DataFrame(cel_rows)
for _, r in celiac_parkinsonism.iterrows():
    print(f"  Celiac x parkinsonism, AID+ females: {r['celiac_park']}/{r['celiac_park'] + r['celiac_no_park']}"
          f" celiac vs {r['no_celiac_park']}/{r['no_celiac_park'] + r['no_celiac_no_park']} other; "
          + (f"OR {r['odds_ratio']:.2f} ({r['ci_low']:.2f}-{r['ci_high']:.2f}), Fisher P {r['p']:.3g}"
             if r["status"] == "tested" else f"not tested ({r['reason']})"))

# Possible-mimic conditions as a group: disjoint groups, per-feature pattern, breadth
HAS_MIMIC = AIDPOS & (pop[[f"cond__{u}" for u in MIMIC_CONDITIONS if u in UNITS]].sum(axis=1) > 0)
NON_MIMIC = AIDPOS & ~HAS_MIMIC
mg_rows = []
for lbl, m in (("mimic", HAS_MIMIC), ("non_mimic", NON_MIMIC)):
    for o in ["core"] + CORE_FEATURES:
        mg_rows.append({"group": lbl, "outcome": o, "outcome_label": OUTCOME_LABELS[o],
                        "n_carriers": int(m.sum()), **ror_row(m, o)})
mimic_groups = pd.DataFrame(mg_rows)
mc_rows = []
for o in ["core", "core_2plus"] + CORE_FEATURES:
    d = pop[(HAS_MIMIC | NON_MIMIC) & FEMALE]
    g = HAS_MIMIC[d.index].astype(int).to_numpy()
    mc_rows.append({"outcome": o, "contrast": "female_mimic_vs_non_mimic",
                    **ratio_row(d[o], d["e__" + o], g, ("possible mimic", "other"))})
mimic_contrasts = pd.DataFrame(mc_rows)
for _, r in mimic_contrasts.iterrows():
    print(f"  Females, possible-mimic vs other carriers, {r['outcome']:<22} n "
          f"{r['n_1']}/{r['n_0']}: RR {ci_txt(r['ratio'], r['ci_low'], r['ci_high'])}, "
          f"P {q_txt(r['p'])}")
br_rows = []
for lbl, m in (("mimic", HAS_MIMIC), ("non_mimic", NON_MIMIC), ("aid_neg_reference", AIDNEG)):
    d = pop[m & FEMALE & (pop["core"] >= 1)]
    k2 = int((d["core"] >= 2).sum())
    pc_, lo_, hi_ = au.wilson_ci(k2, len(d))
    br_rows.append({"group": lbl, "n_feature_positive": len(d), "n_two_or_more": k2,
                    "pct": pc_, "ci_low": lo_, "ci_high": hi_})
mimic_breadth = pd.DataFrame(br_rows)
for _, r in mimic_breadth.iterrows():
    print(f"  Two or more of the core features among females with one or more, {r['group']:<18}"
          f" {r['n_two_or_more']}/{r['n_feature_positive']} = {r['pct']:.1f}% "
          f"({r['ci_low']:.1f}-{r['ci_high']:.1f})")


# =============================================================================
# SECTION 7 -- GROUPING TABLE (EXPLORATORY)
# =============================================================================
au.section("SECTION 7 -- PUBLISHED GROUPINGS: CLASS VS CONTRASTING CLASS (FEMALES)")
classes_of = {}
for g in GROUPINGS:
    sets = {}
    for u in UNITS:
        sets.setdefault(class_of(g, u), []).append(u)
    classes_of[g] = {k: (pop[[f"cond__{u}" for u in v]].sum(axis=1) > 0) & AIDPOS
                     for k, v in sets.items() if k != UNASSIGNED}
grp_rows, grp_sir_rows = [], []
for g, spec in GROUPINGS.items():
    cls = classes_of[g]
    for k, mk in cls.items():
        for o, _ in GROUPING_MEASURES:
            grp_sir_rows.append({"grouping": g, "class": k, "class_label": CLASS_LABELS[k],
                                 "outcome": o, "n_female": int((mk & FEMALE).sum()),
                                 **sir_cols(float(pop.loc[mk & FEMALE, o].sum()),
                                            float(pop.loc[mk & FEMALE, "e__" + o].sum()))})
    if g in DESCRIPTIVE_ONLY:          # class SIRs only; no contrast, not in the BH family
        continue
    pairs = [TWO_CLASS[g]] if g in TWO_CLASS else [(k, "rest") for k in cls]
    for k1, k0 in pairs:
        if k1 not in cls:
            continue
        others = [k for k in cls if k != k1] if k0 == "rest" else [k0]
        if not others or not all(k in cls for k in others):
            continue
        m_other = np.any([cls[k] for k in others], axis=0)
        g1 = cls[k1] & ~m_other & FEMALE
        g0 = pd.Series(m_other, index=pop.index) & ~cls[k1] & FEMALE
        d = pop[g1 | g0]
        gg = g1[d.index].astype(int).to_numpy()
        for o, olab in GROUPING_MEASURES:
            grp_rows.append({"grouping": g, "grouping_label": spec["label"], "class": k1,
                             "class_label": CLASS_LABELS[k1],
                             "comparator": "rest" if k0 == "rest" else k0,
                             "comparator_label": "Other classified conditions" if k0 == "rest"
                             else CLASS_LABELS[k0],
                             "n_female_in_both": int((cls[k1] & pd.Series(m_other,
                                                                          index=pop.index)
                                                      & FEMALE).sum()),
                             "outcome": o, "outcome_label": olab,
                             **ratio_row(d[o], d["e__" + o], gg, (CLASS_LABELS[k1], "comparator"))})
groupings = pd.DataFrame(grp_rows)
groupings = bh_rows(groupings, ["outcome"], "p", "q_ratio", groupings["status"] == "estimated")
grouping_sirs = pd.DataFrame(grp_sir_rows)
core_ = groupings["outcome"] == "core"
grouping_table = groupings[core_ & (groupings["status"] == "estimated")].reset_index(drop=True)
grouping_not_shown = groupings.loc[core_ & (groupings["status"] != "estimated"),
                                   ["grouping", "grouping_label", "class", "class_label",
                                    "comparator_label", "n_1", "o_1", "reason"]].reset_index(drop=True)
for _, r in groupings.iterrows():
    print(f"  {r['outcome']:<11}{r['grouping']:<16}{r['class']:<20} vs {r['comparator']:<18} n "
          f"{r['n_1']:>4}/{r['n_0']:<4} O/E {r['o_1']:.0f}/{r['e_1']:.1f} vs {r['o_0']:.0f}/"
          f"{r['e_0']:.1f}  RR {ci_txt(r['ratio'], r['ci_low'], r['ci_high'])}  Q "
          f"{q_txt(r['q_ratio'])}  {r['reason']}")
print(f"  Grouping table: {len(grouping_table)} contrasts shown; not estimated (caption): "
      f"{', '.join(grouping_not_shown['class_label']) or 'none'}")


# =============================================================================
# SECTION 8 -- EXTENDED DATA: COMBINATIONS BY CONDITION (AID+ FEMALES)
# =============================================================================
au.section("SECTION 8 -- CORE-FEATURE COMBINATIONS BY CONDITION")
hm_rows = []
for u in CONDITIONS:
    S = carriers(u) & FEMALE
    for c in COMBOS:
        o, e = float(pop.loc[S, "cb__" + c].sum()), float(pop.loc[S, "e__cb__" + c].sum())
        hm_rows.append({"condition": u, "row_label": condition_label(u), "n_female": NF[u],
                        "combination": c, "combination_label": COMBO_LABELS[c], "observed": o,
                        "expected": e, "sir": o / e if e > 0 else NAN})
combinations = pd.DataFrame(hm_rows)
chk = combinations.groupby("condition")["observed"].sum()
bad = [u for u in CONDITIONS if int(chk[u]) != NF[u]]
if bad:
    raise RuntimeError(f"combination counts do not add up to the female carriers: {bad}")
print(f"  {len(CONDITIONS)} conditions; every condition's combinations add up to its "
      f"female carriers")


# =============================================================================
# SECTION 9 -- CONDITION-LEVEL CHECKS
# =============================================================================
au.section("SECTION 9 -- CONDITION-LEVEL CHECKS")
count_rows = []
for u in UNITS:
    m = carriers(u)
    count_rows.append({"condition": u, "row_label": condition_label(u),
                       "n_female": int((m & FEMALE).sum()), "n_male": int((m & ~FEMALE).sum()),
                       "n_isolated_female": int((m & FEMALE & ISOLATED).sum()),
                       "n_isolated_male": int((m & ~FEMALE & ISOLATED).sum()),
                       "mean_n_conditions": float(pop.loc[m, "n_aids"].mean())})
counts = pd.DataFrame(count_rows)
ibd_members = cond[cond["unit"] == IBD_UNIT].groupby("normalized_label")["person_id"].nunique()
ibd_counts = pd.DataFrame({"member": ibd_members.index, "n_patients": ibd_members.values})

mean_rows = []
for o in EXPECTED_OUTCOMES:
    rec = {"outcome": o, "outcome_label": OUTCOME_LABELS.get(o, o),
           "n_fallback_aid_pos": FALLBACK[o]}
    for a, s, g in ((1, 1, "aid_pos_female"), (1, 0, "aid_pos_male"),
                    (0, 1, "aid_neg_female"), (0, 0, "aid_neg_male")):
        rec["mean_" + g] = float(pop.loc[(pop["aid_status"] == a) & (pop["female"] == s),
                                         o].mean())
    mean_rows.append(rec)
outcome_means = pd.DataFrame(mean_rows)

sec3 = []
for exclude_dlb in (False, True):
    for fam, outs in OUTCOME_FAMILIES:
        if exclude_dlb and fam == "diagnosis":
            continue
        for u in UNITS:
            m = carriers(u)
            nF, nM = int((m & FEMALE).sum()), int((m & ~FEMALE).sum())
            if nF == 0 or nM == 0:
                continue
            for o in outs:
                if exclude_dlb and "e_nodlb__" + o not in pop:
                    pop["e_nodlb__" + o], _ = au.expected_counts(pop, o, reference=nodlb_ref)
                r = ror_row(m, o, exclude_dlb=exclude_dlb)
                rec = {"arm": "coded_dlb_excluded" if exclude_dlb else "all_patients",
                       "family": fam, "in_fdr": int(fam not in NOT_IN_FDR), "condition": u,
                       "outcome": o, "outcome_label": OUTCOME_LABELS.get(o, o), **r,
                       "mde_female": mde_ratio(r["e_female"]), "mde_male": mde_ratio(r["e_male"])}
                for mg in TOST_MARGINS:
                    rec[f"tost_male_inside_{mg:g}"] = tost_inside(r["sir_male"],
                                                                 r["sir_male_ci_low"],
                                                                 r["sir_male_ci_high"], mg)
                sec3.append(rec)
sec3 = pd.DataFrame(sec3)
sec3 = bh_rows(sec3, ["family", "arm"], "p_ror", "q_ror",
               (sec3["status_ror"] == "estimated") & (sec3["in_fdr"] == 1))

esc_rows = []
for k, kn in ((1, 2), (2, 3)):
    gate = pop["core"] >= k
    g = pop[gate].copy()
    out_col = f"core_{kn}plus"
    sizes = {lbl: int((gate & (pop["aid_status"] == a) & (pop["female"] == s)).sum())
             for a, s, lbl in ((1, 1, "aid_pos_female"), (1, 0, "aid_pos_male"),
                               (0, 1, "aid_neg_female"), (0, 0, "aid_neg_male"))}
    if min(sizes["aid_neg_female"], sizes["aid_neg_male"]) < au.MIN_GROUP_N:
        esc_rows.append({"step": f"{k}_to_{kn}", "condition": "", **sizes,
                         "status_ror": "not_estimated",
                         "reason_ror": f"fewer than {au.MIN_GROUP_N} AID- patients of a sex"})
        continue
    g["e_esc"], _ = au.expected_counts(g, out_col)
    for lbl, mk in [(u, (g["aid_status"] == 1) & (g["cond__" + u] == 1)) for u in UNITS]:
        dd = g[mk]
        if not len(dd):
            continue
        esc_rows.append({"step": f"{k}_to_{kn}", "condition": lbl, **sizes,
                         **au.sir_ror_row(dd["female"], dd[out_col], dd["e_esc"])})
escalation = pd.DataFrame(esc_rows)

dec_rows = []
for exclude_dlb in (False, True):
    for u in UNITS:
        base = carriers(u)
        if int((base & FEMALE).sum()) < au.MIN_GROUP_N:
            continue
        r_all = ror_row(base, CORE, exclude_dlb)["ror"]
        r_iso = ror_row(base & ISOLATED, CORE, exclude_dlb)["ror"]
        strip_list = [v for v in COMORBID_STRIP.get(u, []) if v in UNITS]
        strip = base.copy()
        for v in strip_list:
            strip &= pop["cond__" + v] == 0
        r_strip = ror_row(strip, CORE, exclude_dlb)["ror"] if strip_list else NAN
        flag, traj, cur, removed = "", [], base.copy(), []
        if np.isfinite(r_all):
            log0 = np.log(r_all)
            for _ in range(PASSTHROUGH_MAX_ITER):
                co = {v: int(pop.loc[cur, "cond__" + v].sum()) for v in UNITS
                      if v != u and v not in removed}
                co = {kk: vv for kk, vv in co.items() if vv > 0}
                if not co:
                    break
                top = max(co, key=co.get)
                removed.append(top)
                cur = cur & (pop["cond__" + top] == 0)
                rr = ror_row(cur, CORE, exclude_dlb)["ror"]
                traj.append(f"{top} (n={co[top]}, RoR="
                            f"{'not estimated' if not np.isfinite(rr) else repr(round(rr, 6))})")
                if np.isfinite(rr) and abs(log0) > 1e-9 and \
                        abs(np.log(rr) - log0) / abs(log0) > PASSTHROUGH_LOGROR_FRAC:
                    flag = "pass_through: " + top
                    break
        dec_rows.append({"arm": "coded_dlb_excluded" if exclude_dlb else "all_patients",
                         "condition": u, "ror_all_carriers": r_all, "ror_isolated": r_iso,
                         "ror_stripped": r_strip, "stripped_of": ", ".join(strip_list),
                         "passthrough_flag": flag, "removal_trajectory": "; ".join(traj)})
decomposition = pd.DataFrame(dec_rows)

med = au.load_medication_records(MEDICATION_RECORDS_PATH, pop)
med = med.merge(pop[["person_id", "birth_date"]], on="person_id", how="left")
med_age = au.years_between(med["exposure_date"], med["birth_date"])
med_ok = med["exposure_date"].isna() | au.within_age_ceiling(
    med_age, med["person_id"].map(ceil["ceiling_by_person"]))
med_rows = []
for c in ("tnf_i_exposed", "biologic_exposed_broad"):
    ids = set(med.loc[med_ok & (med["drug_class"] == c), "person_id"])
    pop["exp__" + c] = pop["person_id"].isin(ids).astype(int)
    for tname in ("rheumatoid arthritis", IBD_UNIT):
        m = carriers(tname) if tname in UNITS else pd.Series(False, index=pop.index)
        for val, lbl in ((1, "exposed"), (0, "unexposed")):
            mm = m & (pop["exp__" + c] == val)
            med_rows.append({"condition": tname, "drug_class": c, "exposure": lbl,
                             "n_carriers": int(mm.sum()), **ror_row(mm, CORE)})
medication = pd.DataFrame(med_rows)

ma_units = [u for u in UNITS if ror_row(carriers(u), CORE)["status_ror"] == "estimated"]
d = pop[AIDPOS].copy()
d["e_model"] = d["e__core"]
zero_e = d["e_model"] <= 0
d.loc[zero_e, "e_model"] = d.loc[zero_e, "female"].map(
    {s: pop.loc[~AIDPOS & (pop["female"] == s), "core"].mean() for s in (1, 0)})
n_terms = 2 + 2 * len(ma_units)
why = ""
if len(ma_units) < 2:
    why = "fewer than 2 conditions with an estimated RoR"
elif d["core"].sum() < au.MIN_EVENTS_PER_TERM * n_terms:
    why = f"{d['core'].sum():.0f} events < {au.MIN_EVENTS_PER_TERM} x {n_terms} terms"
fit_ma = None
if not why:
    Xm = pd.DataFrame({"const": 1.0, "female": d["female"].astype(float)}, index=d.index)
    for u in ma_units:
        Xm["c_" + u] = d["cond__" + u].astype(float)
        Xm["fx_" + u] = Xm["female"] * Xm["c_" + u]
    try:
        fit_ma = sm.GLM(d["core"].astype(float), Xm, family=sm.families.Poisson(),
                        offset=np.log(d["e_model"].astype(float))).fit(cov_type="HC0",
                                                                         maxiter=200)
    except Exception as ex_:
        why = f"fit error: {type(ex_).__name__}"
ma_rows = []
for u in ma_units:
    marg = conds[(conds["arm"] == "all_carriers") & (conds["condition"] == u)].iloc[0]
    rec = {"condition": u, "row_label": condition_label(u), "ror_marginal": marg["ror"],
           "status": "estimated" if fit_ma is not None else "not_estimated", "reason": why,
           "ror_adjusted": NAN, "ci_low": NAN, "ci_high": NAN, "p_adjusted": NAN}
    if fit_ma is not None:
        b_, s_ = float(fit_ma.params["fx_" + u]), float(fit_ma.bse["fx_" + u])
        rec.update(ror_adjusted=float(np.exp(b_)), ci_low=float(np.exp(b_ - CRIT * s_)),
                   ci_high=float(np.exp(b_ + CRIT * s_)), p_adjusted=float(fit_ma.pvalues["fx_" + u]))
    ma_rows.append(rec)
mutually_adjusted_ror = pd.DataFrame(ma_rows)

nodlb_rows = []
for u in UNITS:
    r = ror_row(carriers(u), CORE, exclude_dlb=True)
    nodlb_rows.append({"condition": u, "row_label": condition_label(u),
                       **{k: r[k] for k in ("n_female", "o_female", "e_female",
                                            "status_sir_female", "reason_sir_female",
                                            "sir_female", "sir_female_ci_low",
                                            "sir_female_ci_high", "p_sir_female")}})
coded_dlb_excluded = pd.DataFrame(nodlb_rows)


# =============================================================================
# SECTION 10 -- SJOGREN'S SYNDROME: TIMING OF THE CORE FEATURES
# =============================================================================
au.section("SECTION 10 -- SJOGREN'S SYNDROME: FIRST CORE FEATURE VS THE SJOGREN'S DIAGNOSIS")
sj_date = cond[cond["normalized_label"] == SJOGRENS].groupby("person_id")["diagnosis_date"].min()
pop["sjogrens_date"] = pd.to_datetime(pop["person_id"].map(sj_date.to_dict()))
pop["first_core_date"] = au.dlb_core_burden_first_date(pop)
pop["sj_gap_yrs"] = au.years_between(pop["first_core_date"], pop["sjogrens_date"])
pop["sj_concurrent"] = (au.documented_by_years_after_diagnosis(pop["first_core_date"],
                                                               pop["sjogrens_date"],
                                                               CONCURRENT_YRS)
                        & au.documented_from_years_before_diagnosis(pop["first_core_date"],
                                                                    pop["sjogrens_date"],
                                                                    CONCURRENT_YRS)).astype(int)
SJ = carriers(SJOGRENS) if SJOGRENS in UNITS else pd.Series(False, index=pop.index)
sj_counts = []
for s, sex in SEXES:
    c = pop[SJ & (pop["female"] == s)]
    w_ = c[c["sjogrens_date"].notna() & c["first_core_date"].notna()]
    sj_counts.append({"sex": sex, "n_carriers": len(c), "n_dated_diagnosis":
                      int(c["sjogrens_date"].notna().sum()),
                      "n_with_counted_core_feature": int((c["core"] >= 1).sum()),
                      "n_with_gap": len(w_), "n_concurrent": int(w_["sj_concurrent"].sum()),
                      "gap_median_yrs": float(w_["sj_gap_yrs"].median()) if len(w_) else NAN})
sjogrens_counts = pd.DataFrame(sj_counts)
fem = pop[SJ & FEMALE]
b = fem[fem["sjogrens_date"].notna() & fem["first_core_date"].notna()].copy()
gap = b["sj_gap_yrs"].to_numpy(dtype=float)
rng = np.random.default_rng(SJOGRENS_SEED)


def gap_summary(gap, conc, rng):
    """Gap (first counted core feature minus the condition's diagnosis date, years) of the
    carriers with both dates: median, IQR, bootstrap CI of the median (rng), n after /
    before / same day, concurrent share (conc: 0/1 per carrier, Wilson CI), Wilcoxon and
    sign P (tested from au.MIN_GROUP_N carriers). Returns (summary dict, gate reason)."""
    n_g = len(gap)
    why = au.group_minimum_reason(n_g) if n_g else "no carriers with a gap"
    summ = {"n": n_g, "status": "not_tested" if why else "tested", "reason": why}
    if n_g:
        if n_g >= 3:
            boots = [np.median(rng.choice(gap, size=n_g, replace=True)) for _ in range(N_BOOT)]
            mlo, mhi = np.percentile(boots, [2.5, 97.5])
        else:
            mlo = mhi = NAN
        n_pos, n_neg = int((gap > 0).sum()), int((gap < 0).sum())
        n_conc = int(conc.sum())
        pc_, plo, phi = au.wilson_ci(n_conc, n_g)
        summ.update(median_yrs=float(np.median(gap)), q1_yrs=float(np.percentile(gap, 25)),
                    q3_yrs=float(np.percentile(gap, 75)), median_ci_low=float(mlo),
                    median_ci_high=float(mhi), n_after=n_pos, n_before=n_neg,
                    n_same_day=int((gap == 0).sum()), pct_after=100 * n_pos / n_g,
                    n_concurrent=n_conc, pct_concurrent=pc_, pct_concurrent_ci_low=plo,
                    pct_concurrent_ci_high=phi, p_wilcoxon=NAN, p_sign=NAN)
        if not why:
            summ["p_wilcoxon"] = float(stats.wilcoxon(gap).pvalue) if (gap != 0).any() else NAN
            summ["p_sign"] = float(stats.binomtest(n_pos, n_pos + n_neg, 0.5).pvalue) \
                if n_pos + n_neg else NAN
    return summ, why


n_g = len(gap)
summ, why = gap_summary(gap, b["sj_concurrent"].to_numpy(), rng)
sj_summary = pd.DataFrame([summ])
if n_g:
    lo_b = np.floor(gap.min() / GAP_BIN_YRS) * GAP_BIN_YRS
    hi_b = (np.floor(gap.max() / GAP_BIN_YRS) + 1) * GAP_BIN_YRS
    edges = np.round(np.arange(lo_b, hi_b + GAP_BIN_YRS / 2, GAP_BIN_YRS), 10)
    idx = np.floor(gap / GAP_BIN_YRS).astype(int) - int(round(lo_b / GAP_BIN_YRS))
    conc_arr = b["sj_concurrent"].to_numpy()
    sj_bins = pd.DataFrame({"bin_start_yrs": edges[:-1], "bin_end_yrs": edges[1:],
                            "n_concurrent": [int(((idx == i) & (conc_arr == 1)).sum())
                                             for i in range(len(edges) - 1)],
                            "n_not_concurrent": [int(((idx == i) & (conc_arr == 0)).sum())
                                                 for i in range(len(edges) - 1)]})
else:
    sj_bins = pd.DataFrame(columns=["bin_start_yrs", "bin_end_yrs", "n_concurrent",
                                    "n_not_concurrent"])
bsum = {"n_carriers": n_g, "status": "not_tested" if why else "tested", "reason": why}
if n_g:
    nf = b["core"].to_numpy(dtype=float)
    cc = b["sj_concurrent"].to_numpy()
    tot_f, conc_f = float(nf.sum()), float(nf[cc == 1].sum())
    pat_share, pat_lo, pat_hi = au.wilson_ci(int(cc.sum()), n_g)
    burd = 100 * conc_f / tot_f
    bsb = []
    for _ in range(N_BOOT):
        s_ = rng.choice(np.arange(n_g), size=n_g, replace=True)
        t_ = nf[s_].sum()
        if t_ > 0:
            bsb.append(100 * nf[s_][cc[s_] == 1].sum() / t_)
    bsum.update(n_concurrent=int(cc.sum()), share_carriers_pct=pat_share,
                share_carriers_ci_low=pat_lo, share_carriers_ci_high=pat_hi,
                n_features_total=tot_f, n_features_concurrent=conc_f, share_burden_pct=burd,
                share_burden_ci_low=float(np.percentile(bsb, 2.5)) if len(bsb) > 10 else NAN,
                share_burden_ci_high=float(np.percentile(bsb, 97.5)) if len(bsb) > 10 else NAN,
                p_permutation=NAN, n_permutations=N_PERM_SJOGRENS, p_mann_whitney=NAN)
    if not why:
        obs_dev = abs(burd - pat_share)
        perm = [abs(100 * nf[rng.permutation(cc) == 1].sum() / tot_f - pat_share)
                for _ in range(N_PERM_SJOGRENS)]
        bsum["p_permutation"] = float((1 + np.sum(np.array(perm) >= obs_dev))
                                      / (N_PERM_SJOGRENS + 1))
        if not au.group_minimum_reason(int(cc.sum()), int((cc == 0).sum())):
            bsum["p_mann_whitney"] = float(stats.mannwhitneyu(nf[cc == 1], nf[cc == 0],
                                                              alternative="two-sided")[1])
sj_burden = pd.DataFrame([bsum])
print(f"  Female Sjogren's carriers with a gap: {n_g:,}; median gap "
      f"{summ.get('median_yrs', NAN):.2f} yr; Wilcoxon P {summ.get('p_wilcoxon', NAN):.3g}")


# =============================================================================
# SECTION 11 -- PANEL c (AID DIAGNOSIS TIMING) AND CHECKS: PEMPHIGOID REMOVED (SPREAD
#               MODEL), PEMPHIGOID TIMING, SEX PATTERN
# =============================================================================
au.section("SECTION 11 -- AID DIAGNOSIS TIMING (PANEL c); CHECKS: PEMPHIGOID REMOVED, "
           "PEMPHIGOID TIMING, SEX PATTERN")


def check_row(analysis, measure, **kw):
    out = {"analysis": analysis, "measure": measure, "n": NAN, "observed": NAN, "expected": NAN,
           "estimate": NAN, "ci_low": NAN, "ci_high": NAN, "p": NAN, "k": NAN, "i2": NAN,
           "tau": NAN, "prediction_low": NAN, "prediction_high": NAN}
    out.update(kw)
    return out


def pool_row(measure, n, sp):
    p_ = sp.iloc[0]
    return check_row("spread_pooled_rate_ratio", measure, n=n, estimate=p_["pooled"],
                     ci_low=p_["ci_low"], ci_high=p_["ci_high"], p=p_["p"], k=p_["k"],
                     i2=p_["I2"], tau=p_["tau"], prediction_low=p_["prediction_low"],
                     prediction_high=p_["prediction_high"])


def check_line(r):
    extra = ""
    if np.isfinite(r["k"]):
        extra = (f"; k {int(r['k'])}, I2 {r['i2']:.0f}%, 95% prediction interval "
                 f"{r['prediction_low']:.2f}-{r['prediction_high']:.2f}")
    oe = (f"O/E {r['observed']:.0f}/{r['expected']:.1f}  " if np.isfinite(r["expected"])
          else "")
    print(f"    {r['analysis']:<32}{r['measure']:<46}{oe}"
          f"{ci_txt(r['estimate'], r['ci_low'], r['ci_high'])}  P {q_txt(r['p'])}{extra}")


def quartiles(v):
    v = pd.Series(v, dtype=float).dropna()
    return ((float(v.median()), float(v.quantile(.25)), float(v.quantile(.75))) if len(v)
            else (NAN, NAN, NAN))


# 11a  Pemphigoid removed: the spread model (panel a) refit on the females without
#      pemphigoid carriers
au.sub("11a  Pemphigoid carriers removed: the spread model refit (reference: all females)")
pem_rows = []
if PEMPHIGOID in UNITS:
    NO_PEM = pop["cond__" + PEMPHIGOID] == 0
    sp_pool = spread_model(FEMALE & NO_PEM, exclude=(PEMPHIGOID,))[1]
    pem_rows += [pool_row("all AID+ (reference)",
                          int((FEMALE & (pop["e__core"] > 0)).sum()), spread_pool),
                 pool_row("pemphigoid removed",
                          int((FEMALE & NO_PEM & (pop["e__core"] > 0)).sum()), sp_pool)]
    print(f"    AID+ females {N_AID_POS[1]:,}; pemphigoid carriers removed "
          f"{int((WF & ~NO_PEM).sum()):,}")
else:
    print("    no pemphigoid carriers")
pemphigoid_removed = pd.DataFrame(pem_rows, columns=list(check_row("", "")))
for _, r in pemphigoid_removed.iterrows():
    check_line(r)

# 11b  Direction: AID diagnosis date (earliest per patient per unit) vs AD diagnosis and
#      first counted core feature
au.sub("11b  AID diagnosis vs AD diagnosis and first counted core feature, per condition "
       "(>= 10 female carriers), each sex")
unit_date = cond.groupby(["unit", "person_id"])["diagnosis_date"].min()
timing_rows = []
for u in CONDITIONS:
    dates = unit_date.loc[u]
    for s, sex in SEXES:
        c = pop[carriers(u) & (pop["female"] == s)]
        aid_dt = pd.to_datetime(c["person_id"].map(dates))
        dated = aid_dt.notna()
        to_ad = au.years_between(c["ad_diagnosis_date"], aid_dt)[dated]
        n_after_ad = int((aid_dt > c["ad_diagnosis_date"])[dated].sum())
        wc = dated & (c["core"] >= 1) & c["first_core_date"].notna()
        to_core = au.years_between(c["first_core_date"], aid_dt)[wc]
        n_after_core = int((aid_dt > c["first_core_date"])[wc].sum())
        md_ad, q1_ad, q3_ad = quartiles(to_ad)
        md_c, q1_c, q3_c = quartiles(to_core)
        timing_rows.append({
            "condition": u, "row_label": condition_label(u), "sex": sex, "n_carriers": len(c),
            "n_dated": int(dated.sum()), "aid_to_ad_median_yrs": md_ad, "aid_to_ad_q1_yrs": q1_ad,
            "aid_to_ad_q3_yrs": q3_ad, "n_aid_after_ad": n_after_ad,
            "pct_aid_after_ad": 100 * n_after_ad / int(dated.sum()) if dated.any() else NAN,
            "n_dated_with_core_feature": int(wc.sum()), "aid_to_first_core_median_yrs": md_c,
            "aid_to_first_core_q1_yrs": q1_c, "aid_to_first_core_q3_yrs": q3_c,
            "n_aid_after_first_core": n_after_core,
            "pct_aid_after_first_core": 100 * n_after_core / int(wc.sum()) if wc.any() else NAN})
aid_dx_timing = pd.DataFrame(timing_rows)
# Panel c: the female rows (conditions shown, >= 10 female carriers)
c_aid_dx_timing = aid_dx_timing.loc[aid_dx_timing["sex"] == "female",
                                    ["condition", "row_label", "n_carriers", "n_dated",
                                     "aid_to_ad_median_yrs", "aid_to_ad_q1_yrs", "aid_to_ad_q3_yrs",
                                     "n_aid_after_ad", "pct_aid_after_ad"]].reset_index(drop=True)
print("    Years from the AID diagnosis to the AD diagnosis / to the first counted core feature "
      "(median, IQR); positive = AID diagnosis first")
for _, r in aid_dx_timing.iterrows():
    print(f"    {r['row_label'][:28]:<29}{r['sex']:<7} n {r['n_carriers']:>4} dated "
          f"{r['n_dated']:>4}  to AD {r['aid_to_ad_median_yrs']:+6.1f} "
          f"({r['aid_to_ad_q1_yrs']:+.1f} to {r['aid_to_ad_q3_yrs']:+.1f}); AID after AD "
          f"{r['n_aid_after_ad']:>3} ({r['pct_aid_after_ad']:.0f}%)  |  with core feature "
          f"{r['n_dated_with_core_feature']:>3}: to first core {r['aid_to_first_core_median_yrs']:+6.1f}"
          f" ({r['aid_to_first_core_q1_yrs']:+.1f} to {r['aid_to_first_core_q3_yrs']:+.1f}); "
          f"AID after first core {r['n_aid_after_first_core']:>3} "
          f"({r['pct_aid_after_first_core']:.0f}%)")

au.sub("11b  Pemphigoid: first counted core feature minus the pemphigoid diagnosis date "
       "(Sjogren's method), each sex")
pem_timing_rows = []
if PEMPHIGOID in UNITS:
    pem_dates = unit_date.loc[PEMPHIGOID]
    for s, sex in SEXES:
        c = pop[carriers(PEMPHIGOID) & (pop["female"] == s)]
        p_dt = pd.to_datetime(c["person_id"].map(pem_dates))
        w = p_dt.notna() & c["first_core_date"].notna()
        gap_p = au.years_between(c["first_core_date"], p_dt)[w].to_numpy(dtype=float)
        conc_p = (au.documented_by_years_after_diagnosis(c["first_core_date"], p_dt,
                                                         CONCURRENT_YRS)
                  & au.documented_from_years_before_diagnosis(c["first_core_date"], p_dt,
                                                              CONCURRENT_YRS))[w] \
            .astype(int).to_numpy()
        summ_p, _ = gap_summary(gap_p, conc_p, np.random.default_rng(PEMPHIGOID_SEED + s))
        pem_timing_rows.append({"sex": sex, "n_carriers": len(c),
                                "n_dated_diagnosis": int(p_dt.notna().sum()),
                                "n_with_counted_core_feature": int((c["core"] >= 1).sum()),
                                **summ_p})
pemphigoid_timing = pd.DataFrame(pem_timing_rows)
for _, r in pemphigoid_timing.iterrows():
    head = (f"    {r['sex']:<7} carriers {r['n_carriers']:>4}, dated {r['n_dated_diagnosis']:>4}, "
            f"with a gap {r['n']:>4}")
    if r["n"] > 0:
        print(head + f": median {r['median_yrs']:+.2f} yr (IQR {r['q1_yrs']:+.2f} to "
              f"{r['q3_yrs']:+.2f}; 95% CI {r['median_ci_low']:+.2f} to "
              f"{r['median_ci_high']:+.2f}); after {r['n_after']}, before {r['n_before']}, same "
              f"day {r['n_same_day']}; Wilcoxon P {q_txt(r['p_wilcoxon'])}, sign P "
              f"{q_txt(r['p_sign'])}" + (f" [{r['reason']}]" if r["reason"] else ""))
    else:
        print(head + f": {r['reason']}")

# 11c  Sex pattern (no new test): conditions with >= 10 carriers of each sex
au.sub("11c  Sex pattern: female SIR, male SIR, RoR per condition (>= 10 carriers of each sex; "
       "sorted by female SIR; flag = male SIR > 1)")
sx = conds[(conds["arm"] == "all_carriers") & (conds["n_female"] >= au.MIN_GROUP_N)
           & (conds["n_male"] >= au.MIN_GROUP_N)].sort_values("sir_female", ascending=False)
for _, r in sx.iterrows():
    print(f"    {r['row_label'][:30]:<31}F {r['n_female']:>4} "
          f"{ci_txt(r['sir_female'], r['sir_female_ci_low'], r['sir_female_ci_high']):<19}"
          f"M {r['n_male']:>4} {ci_txt(r['sir_male'], r['sir_male_ci_low'], r['sir_male_ci_high']):<19}"
          f"RoR {ci_txt(r['ror'], r['ror_ci_low'], r['ror_ci_high']):<19}"
          + ("male SIR > 1" if np.isfinite(r["sir_male"]) and r["sir_male"] > 1 else ""))


# =============================================================================
# SECTION 12 -- SOURCE DATA
# =============================================================================
au.section("SECTION 12 -- SOURCE DATA")
params = pd.DataFrame([
    ("n_population", len(pop)), ("n_aid_pos", int(AIDPOS.sum())),
    ("n_aid_pos_female", N_AID_POS[1]), ("n_aid_pos_male", N_AID_POS[0]),
    ("n_units", len(UNITS)), ("n_conditions_shown", len(CONDITIONS)),
    ("n_other_conditions", len(OTHER_UNITS)), ("n_females_other_conditions", int(OTHER_MASK.sum())),
    ("age_ceiling_female", ceil["ceiling_female"]), ("age_ceiling_male", ceil["ceiling_male"]),
    ("strata_quantiles", au.STRATA_QUANTILES),
    ("strata_age_bands", ", ".join(str(x) for x in au.STRATA_AGE_BANDS)),
    ("n_aid_pos_priced_overall_core", FALLBACK["core"]), ("min_group_n", au.MIN_GROUP_N),
    ("min_events_per_term", au.MIN_EVENTS_PER_TERM), ("min_condition_events", MIN_IND_EVENTS),
    ("crit", CRIT), ("mckeith_window_yrs", au.DLB_MCKEITH_WINDOW_YRS),
    ("n_bootstrap_excess", N_BOOT_EXCESS), ("n_bootstrap_cooccurrence", N_BOOT_COOC),
    ("seed", SPREAD_SEED),
    ("ipw_outcome_model", "y ~ aid_status + age_at_dx, female patients, binomial, "
                          "var_weights = ps_weight_trimmed, HC0"),
    ("ps_formula", au.IPW_PS_FORMULA),
    ("weights_rule_max_p95_abs_log_ratio", au.IPW_RULE_MAX_P95_ABS_LOG_RATIO),
    ("weights_rule_max_ps_smd", au.IPW_RULE_MAX_SMD)],
    columns=["parameter", "value"])
TABLE_COLS = ["condition", "row_label", "pleiotropic", "mimic", "n_carriers",
              "n_female", "o_female", "e_female", "status_sir_female", "reason_sir_female",
              "sir_female", "sir_female_ci_low", "sir_female_ci_high", "p_sir_female",
              "q_sir_female", "fdr_family_size_sir_female", "n_male", "o_male", "e_male",
              "status_sir_male", "reason_sir_male", "sir_male", "sir_male_ci_low",
              "sir_male_ci_high", "p_sir_male", "q_sir_male", "fdr_family_size_sir_male",
              "status_ror", "reason_ror", "ror_coef", "ror_se", "crit", "ror", "ror_ci_low",
              "ror_ci_high", "p_ror", "q_ror", "fdr_family_size_ror", "mde_female", "mde_male"]
au.write_workbook(os.path.join(SOURCE_DATA_DIR, FIGURE_FILE), {
    "a_conditions": spread, "a_pooled": spread_pool, "b_excess": excess_by_condition,
    "b_totals": excess_totals, "c_aid_dx_timing": c_aid_dx_timing,
    "d_possible_mimics_ipw": possible_mimics_ipw, "parameters": params})
au.write_workbook(os.path.join(SOURCE_DATA_DIR, EXTENDED_FILE), {
    "combinations": combinations, "parameters": params})
au.write_workbook(os.path.join(SOURCE_DATA_DIR, TABLE_MIMIC_FILE), {
    "pairs": mimic_profile, "parameters": params})
au.write_workbook(os.path.join(SOURCE_DATA_DIR, TABLE_GROUPING_FILE), {
    "contrasts": grouping_table, "not_shown": grouping_not_shown, "all_measures": groupings,
    "class_sirs": grouping_sirs, "parameters": params})
au.write_workbook(os.path.join(SOURCE_DATA_DIR, FIGURE_SJOGRENS_FILE), {
    "a_bins": sj_bins, "a_summary": sj_summary, "b_burden": sj_burden,
    "parameters": pd.concat([params, pd.DataFrame(
        [("concurrent_yrs", CONCURRENT_YRS), ("gap_bin_yrs", GAP_BIN_YRS), ("n_boot", N_BOOT),
         ("n_permutations", N_PERM_SJOGRENS), ("sjogrens_seed", SJOGRENS_SEED)],
        columns=["parameter", "value"])], ignore_index=True)})
au.write_workbook(os.path.join(SOURCE_DATA_DIR, CHECKS_FILE), {
    "condition_sirs": conds[["arm"] + TABLE_COLS],
    "condition_counts": counts, "ibd_members": ibd_counts, "strata": strata_diag,
    "outcome_means": outcome_means, "spread_model_terms": model_terms,
    "sex_differential_pool": ror_pool, "condition_by_outcome": sec3, "escalation": escalation,
    "decomposition": decomposition, "medication": medication,
    "mutually_adjusted_ror": mutually_adjusted_ror, "coded_dlb_excluded": coded_dlb_excluded,
    "mimic_groups": mimic_groups, "mimic_contrasts": mimic_contrasts,
    "mimic_breadth": mimic_breadth, "cooccurrence_vs_independence": cooccurrence_tab,
    "possible_mimics_ipw_designs": possible_mimics_ipw_designs,
    "celiac_parkinsonism": celiac_parkinsonism, "sjogrens_counts_by_sex": sjogrens_counts,
    "pemphigoid_removed": pemphigoid_removed, "aid_dx_timing": aid_dx_timing,
    "pemphigoid_timing": pemphigoid_timing,
    "parameters": pd.concat([params, pd.DataFrame(
        [("pemphigoid_seed_female", PEMPHIGOID_SEED + 1),
         ("pemphigoid_seed_male", PEMPHIGOID_SEED)], columns=["parameter", "value"])],
        ignore_index=True)})
print("\n" + au.SEP + "\n  COMPLETE\n" + au.SEP)
