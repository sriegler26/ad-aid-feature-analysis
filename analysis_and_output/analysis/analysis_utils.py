"""
analysis_utils.py
=================
Shared code for the analysis scripts in this folder. Every analysis script
imports what it needs from here; nothing is re-implemented locally.

What lives here, and why in one place
-------------------------------------
- Loaders for the patient table, the note table and the PS weights, with
  fixed field names. A missing file or column, a duplicate key, a flag
  value other than 0/1, a missing birth date or AD diagnosis date, a note
  of a patient outside the patient table, a record outside the analysis
  population or a missing weight all stop the script.
- The diagnosis-day split. Pre-diagnosis = strictly before the AD
  diagnosis date (date < ad_diagnosis_date); post-diagnosis = on or after it
  (date >= ad_diagnosis_date). Undated records are neither. No other script
  compares a date with the diagnosis date directly; each calls
  is_pre_diagnosis() / is_post_diagnosis().
- The derived per-patient values (ages, first and last dated note, spans,
  note counts, death flag), computed from the patient and note tables.
- The pre-diagnosis documentation span: AD diagnosis date minus first
  dated note, in years, never clipped. It is negative for a patient whose
  first note follows the diagnosis. It is used only as an adjustment
  covariate, and a negative value carries real information: the patient
  entered the record after diagnosis, and by how much. Clipping at 0 would
  give every post-diagnosis entrant the value of a patient first seen on the
  diagnosis day. Unclipped, age at first note + span = age at AD diagnosis
  holds exactly for every patient, so the timing models, adjusted for age at
  first note and the span, also hold age at AD diagnosis fixed; clipping
  would break this for the post-diagnosis entrants. A note count counts
  events in an interval, so it is >= 0 by construction: a post-diagnosis
  entrant has 0 pre-diagnosis notes.
- Standardized mean differences (SMD), unweighted and weighted, continuous
  and proportion-based.
- Benjamini-Hochberg FDR.
- The floors and linearity thresholds as named constants, and the
  floor checks themselves (model floor, 2x2 group minimum, zero-event
  rule), so every analysis applies the same gate. The model floor
  covers 0/1 outcomes (patients in the rarer outcome class count) and
  continuous outcomes (every patient counts), and a model with a sex term
  needs both sexes. The per-group floor of the DLB features (FDR membership:
  the same events in the rarer outcome class within each compared group;
  group_floor_reason). Descriptive tests (Mann-Whitney U, chi-square over
  several categories) use the group minimum (group_minimum_reason).
- The whole-cohort loader (load_cohort: every patient, in the analysis
  population or not), for the cohort tables, beside the analysis-population
  loader.
- Symptom timing and sequence: first documentation per patient and domain
  or category, AD-core onset, the expected order of first documentation
  (the sequence pairs), the 5-year span floor, and the effect-modification
  test of the sequence analyses (AID+ vs AID- within exposed and unexposed
  patients, and the exact conditional test of the AID x exposure
  interaction).
- The symptom schema: the 56 categories, the primary and secondary domain
  definitions (Supplementary Tables 6/7), the readable labels and the
  positive-control sets. Domains are mapped from the category here.
- The symptom-record loader (dates from the note table), and the loaders of
  the other record tables: inflammation tier, change-language and
  explicit-time records, cognitive scores and medication exposure records.
  The tier names and categories are defined here.
- The cognitive scores the analyses use (absolute_cognitive_scores) and the
  annual change per patient (annual_change), called by every script that
  analyzes the annual change, so the outcome is defined once.
- The DLB feature mentions and their first dates per patient and feature,
  with the parkinsonism composites, and the diagnosis-day windows the DLB
  analyses use (a first documentation by a number of years after the AD
  diagnosis, or from a number of years before it), built on is_pre_diagnosis
  / is_post_diagnosis so that the diagnosis-day split has one
  implementation.
- The Wilson score interval of a proportion, used by every script that
  shows one.
- The age ceiling: the sex-specific 90th percentile of age at last note
  among AID- patients of the analysis population. A record counts in a
  presence analysis only if the patient's age at the record is at or below
  the patient's own-sex ceiling. Used in presence analyses only, never in
  timing or sequence analyses.
- The writers (write_workbook, write_parquet). Every output is a new file
  under the output root, written through write_guard.py: an existing
  file is never changed, replaced, moved or deleted, and a script stops if
  an output already exists.

All ages and spans are whole-day differences divided by 365.25. Only
intervals are used, never calendar dates as such, so the results do not
change if every date of a patient is shifted by the same offset.
"""

import numpy as np
import pandas as pd

# =============================================================================
# CONSTANTS
# =============================================================================
DAYS_PER_YEAR = 365.25

# Floors: events per term, patients per group
MIN_EVENTS_PER_TERM = 10      # events per model term, all terms counted
MIN_GROUP_N = 10              # patients per compared group / interaction cell

# Linearity-check flag thresholds
LINEARITY_OR_SHIFT = 0.20     # logistic: relative OR shift ...
LINEARITY_OR_MIN_DIFF = 0.05  # ... and |OR - 1| at least this
LINEARITY_SHIFT_YEARS = 0.5   # linear, outcome in years
LINEARITY_SHIFT_POINTS = 0.5  # linear, outcome in score points

SEX_LABELS = {1: "Female", 0: "Male"}
AID_LABELS = {1: "AID+", 0: "AID-"}

# Fields of the patient table that every analysis script reads.
PATIENT_CORE_COLUMNS = [
    "person_id", "aid_status", "female", "birth_date", "ad_diagnosis_date",
    "death_date", "in_population",
]
PATIENT_OPTIONAL_COLUMNS = {
    "cohort_end_date": "date", "first_race": "text", "ethnicity": "text",
    "smoking_status": "text",
    "coded_dementia_with_lewy_bodies": "flag", "coded_vascular_dementia": "flag",
    "coded_parkinsons_disease": "flag", "coded_mixed_dementia": "flag",
    "coded_hypertension": "flag",
}
NOTE_COLUMNS = ["person_id", "note_id", "note_date", "cognitive_specialty_note"]
PS_WEIGHT_COLUMNS = ["person_id", "ps_weight_trimmed"]
SYMPTOM_RECORD_COLUMNS = ["person_id", "note_id", "category"]
AID_SUBTYPE_COLUMNS = ["person_id", "has_cns_primary_aid"]

# Age ceiling
AGE_CEILING_QUANTILE = 0.90

# =============================================================================
# SYMPTOM SCHEMA (Supplementary Tables 6/7)
# =============================================================================
# 20 domains; every category except "other" belongs to exactly one domain.
# The secondary definition differs from the primary only in sleep disturbance:
# somatic (primary) or autonomic (secondary).
DOMAINS_PRIMARY = {
    "episodic_memory":                  ["episodic_memory", "verbal_memory"],
    "global_cognition":                 ["global_cognition"],
    "subjective_cognitive_burden":      ["cognitive_overload", "brain_fog"],
    "semantic_language":                ["word_finding", "anomia", "aphasia",
                                         "semantic_fluency", "semantic_comprehension",
                                         "semantic_memory"],
    "working_memory_attention":         ["working_memory", "attention",
                                         "distractibility", "processing_speed"],
    "executive_function":               ["executive_function", "planning", "judgment"],
    "visuospatial":                     ["visuospatial", "constructional_apraxia"],
    "orientation_reality_monitoring":   ["confusion", "orientation",
                                         "thought_disorganization"],
    "self_awareness":                   ["anosognosia", "confabulation"],
    "psychosis":                        ["hallucination", "delusion"],
    "mood_affect":                      ["anxiety", "depression", "anhedonia",
                                         "irritability", "mood_lability"],
    "apathy":                           ["apathy"],
    "behavioral_agitation_personality": ["agitation", "aggression", "disinhibition",
                                         "personality_change", "wandering"],
    "fluctuating_cognition":            ["fluctuating_cognition"],
    "functional_ability":               ["IADL", "ADL"],
    "motor":                            ["gait_instability", "motor_disturbance",
                                         "apraxia", "motor_speech_fluency"],
    "autonomic":                        ["autonomic_dysfunction", "incontinence"],
    "somatic":                          ["appetite_change", "fatigue", "weight_loss",
                                         "sleep_disturbance"],
    "systemic_inflammation_autoimmune": ["systemic_inflammation_autoimmune",
                                         "arthralgia", "myalgia"],
    "seizures":                         ["seizures"],
}
DOMAINS_SECONDARY = {d: list(c) for d, c in DOMAINS_PRIMARY.items()}
DOMAINS_SECONDARY["somatic"] = ["appetite_change", "fatigue", "weight_loss"]
DOMAINS_SECONDARY["autonomic"] = ["autonomic_dysfunction", "incontinence",
                                  "sleep_disturbance"]
DOMAIN_DEFINITIONS = {"primary": DOMAINS_PRIMARY, "secondary": DOMAINS_SECONDARY}

UNASSIGNED_CATEGORIES = ["other"]
SCHEMA_CATEGORIES = sorted({c for cats in DOMAINS_PRIMARY.values() for c in cats}
                           | set(UNASSIGNED_CATEGORIES))

# Positive control: describes autoimmune disease itself; shown with its P
# value, never in an FDR family.
POSITIVE_CONTROL_DOMAINS = {"systemic_inflammation_autoimmune"}
POSITIVE_CONTROL_CATEGORIES = {"systemic_inflammation_autoimmune", "arthralgia",
                               "myalgia"}
# Fitted for information only: no P or Q.
NOT_TESTED_CATEGORIES = {"other"}

DOMAIN_LABELS = {
    "episodic_memory": "Episodic memory", "global_cognition": "Global cognition",
    "subjective_cognitive_burden": "Subjective cognitive burden",
    "semantic_language": "Semantic language",
    "working_memory_attention": "Working memory attention",
    "executive_function": "Executive function", "visuospatial": "Visuospatial",
    "orientation_reality_monitoring": "Orientation reality monitoring",
    "self_awareness": "Self-awareness", "psychosis": "Psychosis",
    "mood_affect": "Mood affect", "apathy": "Apathy",
    "behavioral_agitation_personality": "Behavioral agitation personality",
    "fluctuating_cognition": "Fluctuating cognition",
    "functional_ability": "Functional ability", "motor": "Motor",
    "autonomic": "Autonomic", "somatic": "Somatic",
    "systemic_inflammation_autoimmune": "Systemic inflammation autoimmune",
    "seizures": "Seizures",
}
CATEGORY_LABELS = {
    "episodic_memory": "Episodic memory", "verbal_memory": "Verbal memory",
    "global_cognition": "Global cognition", "cognitive_overload": "Cognitive overload",
    "brain_fog": "Brain fog", "word_finding": "Word finding", "anomia": "Anomia",
    "aphasia": "Aphasia", "semantic_fluency": "Semantic fluency",
    "semantic_comprehension": "Semantic comprehension",
    "semantic_memory": "Semantic memory", "working_memory": "Working memory",
    "attention": "Attention", "distractibility": "Distractibility",
    "processing_speed": "Processing speed", "executive_function": "Executive function",
    "planning": "Planning", "judgment": "Judgment", "visuospatial": "Visuospatial",
    "constructional_apraxia": "Constructional apraxia", "confusion": "Confusion",
    "orientation": "Orientation", "thought_disorganization": "Thought disorganization",
    "anosognosia": "Anosognosia", "confabulation": "Confabulation",
    "hallucination": "Hallucination", "delusion": "Delusion", "anxiety": "Anxiety",
    "depression": "Depression", "anhedonia": "Anhedonia",
    "irritability": "Irritability", "mood_lability": "Mood lability",
    "apathy": "Apathy", "agitation": "Agitation", "aggression": "Aggression",
    "disinhibition": "Disinhibition", "personality_change": "Personality change",
    "wandering": "Wandering", "fluctuating_cognition": "Fluctuating cognition",
    "IADL": "IADL", "ADL": "ADL", "gait_instability": "Gait instability",
    "motor_disturbance": "Motor disturbance", "apraxia": "Apraxia",
    "motor_speech_fluency": "Motor speech fluency",
    "autonomic_dysfunction": "Autonomic dysfunction", "incontinence": "Incontinence",
    "appetite_change": "Appetite change", "fatigue": "Fatigue",
    "weight_loss": "Weight loss", "sleep_disturbance": "Sleep disturbance",
    "systemic_inflammation_autoimmune": "Systemic inflammation autoimmune",
    "arthralgia": "Arthralgia", "myalgia": "Myalgia", "seizures": "Seizures",
    "other": "Other",
}


def category_domain(definition="primary"):
    """{category: domain} under a definition; "other" maps to "unassigned"."""
    out = {c: d for d, cs in DOMAIN_DEFINITIONS[definition].items() for c in cs}
    for c in UNASSIGNED_CATEGORIES:
        out[c] = "unassigned"
    return out


def category_row_label(category, definition="primary"):
    """Readable table label: "Category (Domain)", also when the two names are
    the same ("Apathy (Apathy)"); "Other (not assigned to a domain)"."""
    dom = category_domain(definition)[category]
    if dom == "unassigned":
        return f"{CATEGORY_LABELS[category]} (not assigned to a domain)"
    return f"{CATEGORY_LABELS[category]} ({DOMAIN_LABELS[dom]})"


def _check_schema():
    for name, defn in DOMAIN_DEFINITIONS.items():
        cats = [c for cs in defn.values() for c in cs]
        if len(cats) != len(set(cats)) \
                or set(cats) != set(SCHEMA_CATEGORIES) - set(UNASSIGNED_CATEGORIES):
            raise RuntimeError(f"domain definition {name} does not map each category once")
    if len(DOMAINS_PRIMARY) != 20 or len(SCHEMA_CATEGORIES) != 56:
        raise RuntimeError("expected 20 domains and 56 categories")
    if set(DOMAIN_LABELS) != set(DOMAINS_PRIMARY) or set(CATEGORY_LABELS) != set(SCHEMA_CATEGORIES):
        raise RuntimeError("a domain or category has no readable label")


_check_schema()


# =============================================================================
# SYMPTOM TIMING AND SEQUENCE: FIRST DOCUMENTATION, AD CORE, EXPECTED ORDER
# =============================================================================
# Pre-diagnosis documentation span floor: the sequence analyses include only
# patients with at least 5 years of notes before the AD diagnosis, and the
# AD-core floor check uses the same 5 years. There is no second floor.
PREDX_SPAN_FLOOR_5_YRS = 5

# AD-core onset: the first documentation of any of these domains (same
# categories under both domain definitions). Every script that computes it
# uses ad_core_onset() and prints the combined target and each domain.
AD_CORE_DOMAINS = ["episodic_memory", "executive_function", "global_cognition"]

# Expected order of first documentation in typical (amnestic) AD, used as the
# hypothesis-driven reference sequence: episodic memory first, then the
# categories that depend on later, neocortical involvement in Braak staging
# (temporal neocortex: semantic memory; parietal: visuospatial; frontal:
# executive function, which can also be affected early). Each pair is
# (expected first, expected later). The pairs form the FDR family.
# Disorientation is not paired: it appears early in amnestic AD and has no
# expected order relative to episodic memory.
SEQUENCE_PAIRS = [("episodic_memory", "executive_function"),
                  ("episodic_memory", "semantic_memory"),
                  ("episodic_memory", "visuospatial")]


def sequence_pair_label(first, second):
    """"Episodic memory before semantic memory"."""
    return f"{CATEGORY_LABELS[first]} before {CATEGORY_LABELS[second][0].lower()}" \
           f"{CATEGORY_LABELS[second][1:]}"


def first_documentation(records, population, level, definition="primary"):
    """First documentation per patient: one row per patient x domain (level
    "domain", under `definition`) or patient x category (level "category"),
    from the dated records only; "other" is not used. Columns: person_id,
    code, first_date (earliest note date), first_age (age at it, whole days /
    365.25). No age ceiling (the ceiling applies to presence analyses only)."""
    d = records[records["note_date"].notna()].copy()
    c2d = category_domain(definition)
    d = d[d["category"].map(c2d) != "unassigned"]
    d["code"] = d["category"] if level == "category" else d["category"].map(c2d)
    first = (d.groupby(["person_id", "code"])["note_date"].min().reset_index()
             .rename(columns={"note_date": "first_date"}))
    first = first.merge(population[["person_id", "birth_date"]], on="person_id",
                        how="left", validate="many_to_one")
    first["first_age"] = years_between(first["first_date"], first["birth_date"])
    return first.drop(columns="birth_date")


def exact_2x2(a, b, c, d):
    """Conditional MLE odds ratio, exact 95% CI and two-sided Fisher's exact P
    of [[a, b], [c, d]] (rows compared: first vs second; columns: yes / no)."""
    from scipy.stats import fisher_exact
    from scipy.stats.contingency import odds_ratio
    table = [[int(a), int(b)], [int(c), int(d)]]
    res = odds_ratio(table, kind="conditional")
    ci = res.confidence_interval(0.95)
    return float(res.statistic), float(ci.low), float(ci.high), float(fisher_exact(table)[1])


def _log_choose(n, k):
    from scipy.special import gammaln
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def exact_interaction_p(exposed, unexposed):
    """Two-sided exact conditional P of the AID x exposure interaction on an
    outcome (Zelen's exact test of a common odds ratio over two 2x2 strata).
    Each stratum is (events AID+, n AID+, events AID-, n AID-). Conditional on
    every stratum margin and on the total AID+ events, the AID+ events of the
    exposed stratum follow a distribution that does not depend on the common
    odds ratio; P sums the probabilities no larger than the observed one."""
    (a1, p1, c1, q1), (a2, p2, c2, q2) = exposed, unexposed
    m1, m2, t = a1 + c1, a2 + c2, a1 + a2

    def log_h(a, p, q, m):
        if a < max(0, m - q) or a > min(p, m):
            return -np.inf
        return _log_choose(p, a) + _log_choose(q, m - a)

    lw = np.array([log_h(k, p1, q1, m1) + log_h(t - k, p2, q2, m2) for k in range(t + 1)])
    w = np.exp(lw - lw[np.isfinite(lw)].max())
    w /= w.sum()
    return float(min(1.0, w[w <= w[a1] * (1 + 1e-7)].sum()))


EFFECT_MODIFICATION_STRATA = ("exposed", "unexposed")


def effect_modification_row(exposed, unexposed, event):
    """Is the AID+ vs AID- difference in a 0/1 outcome modified by exposure?
    exposed / unexposed: (events AID+, n AID+, events AID-, n AID-). Returns,
    per stratum (suffix _exposed / _unexposed), the gate (two_by_two_reason),
    the conditional MLE odds ratio with its exact CI and Fisher's P (or_estimable
    0 when the OR is 0 or infinite); and for the row the ratio of the two odds
    ratios (exposed / unexposed; ratio_estimable 0 unless both are estimable)
    with the P of exact_interaction_p. The row is tested only when both strata
    pass the gate; status and reason say which stratum did not."""
    out = {"status": "tested", "reason": "", "ratio_estimable": 0, "or_ratio": float("nan"),
           "p": float("nan")}
    whys = []
    for s, (e1, n1, e2, n2) in zip(EFFECT_MODIFICATION_STRATA, (exposed, unexposed)):
        r = {"status": "tested", "reason": "", "or_estimable": 0, "odds_ratio": float("nan"),
             "ci_low": float("nan"), "ci_high": float("nan"), "p": float("nan")}
        why = two_by_two_reason(e1, n1, e2, n2, event)
        if why:
            r.update(status="not_tested", reason=why)
            whys.append(f"{s} patients: {why}")
        else:
            orv, lo, hi, p = exact_2x2(e1, n1 - e1, e2, n2 - e2)
            r["p"] = p
            if np.isfinite(orv) and orv > 0 and np.isfinite(hi) and lo > 0:
                r.update(or_estimable=1, odds_ratio=orv, ci_low=lo, ci_high=hi)
        out.update({f"{k}_{s}": v for k, v in r.items()})
    if whys:
        out.update(status="not_tested", reason="; ".join(whys))
        return out
    out["p"] = exact_interaction_p(exposed, unexposed)
    if out["or_estimable_exposed"] and out["or_estimable_unexposed"]:
        out.update(ratio_estimable=1,
                   or_ratio=out["odds_ratio_exposed"] / out["odds_ratio_unexposed"])
    return out


def ad_core_onset(first_doc_domains):
    """AD-core onset per patient from first_documentation(level="domain"):
    the earliest first documentation over AD_CORE_DOMAINS. Columns: person_id,
    first_date, first_age, first_ad_core_domain (the domain that set it; on a
    same-day tie, the first in AD_CORE_DOMAINS order)."""
    d = first_doc_domains[first_doc_domains["code"].isin(AD_CORE_DOMAINS)].copy()
    d["_order"] = d["code"].map({c: i for i, c in enumerate(AD_CORE_DOMAINS)})
    d = d.sort_values(["person_id", "first_date", "_order"], kind="mergesort")
    out = d.drop_duplicates("person_id")[["person_id", "first_date", "first_age", "code"]]
    return out.rename(columns={"code": "first_ad_core_domain"}).reset_index(drop=True)


# =============================================================================
# CONSOLE
# =============================================================================
SEP = "=" * 80
SEP2 = "-" * 60


def section(title):
    print("\n" + SEP + "\n  " + title + "\n" + SEP)


def sub(title):
    print("\n" + SEP2 + "\n  " + title + "\n" + SEP2)


def count_line(label, n, total=None):
    """One count line; printed every run, including when the count is 0."""
    if total is None:
        print(f"  [count] {label}: {int(n):,}")
    else:
        print(f"  [count] {label}: {int(n):,} of {int(total):,}")


# =============================================================================
# CHECKS
# =============================================================================
def read_parquet_columns(path, columns, name):
    """Read a parquet and keep exactly `columns`; raise if any is missing."""
    if not path:
        raise ValueError(f"{name}: no path set")
    df = pd.read_parquet(path)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"{name}: missing column(s) {missing}")
    return df[list(columns)].copy()


def check_unique(df, key, name):
    dup = df.duplicated(subset=key, keep=False)
    if dup.any():
        raise ValueError(f"{name}: {int(dup.sum()):,} rows share a duplicate key {key}")


def check_flag(df, col, name):
    bad = ~df[col].isin([0, 1]) | df[col].isna()
    if bad.any():
        raise ValueError(f"{name}: {int(bad.sum()):,} values of {col} are not 0/1")
    df[col] = df[col].astype(int)


def to_date(series):
    return pd.to_datetime(series, errors="raise")


# =============================================================================
# DIAGNOSIS-DAY SPLIT AND INTERVALS
# =============================================================================
def is_pre_diagnosis(dates, ad_diagnosis_date):
    """True where the date is strictly before the AD diagnosis date.
    Undated (NaT) records are False."""
    return (dates < ad_diagnosis_date).fillna(False).astype(bool)


def is_post_diagnosis(dates, ad_diagnosis_date):
    """True where the date is on or after the AD diagnosis date.
    Undated (NaT) records are False."""
    return (dates >= ad_diagnosis_date).fillna(False).astype(bool)


def years_between(later, earlier):
    """(later - earlier) in whole days / 365.25. NaT anywhere gives NaN."""
    return (later - earlier).dt.days / DAYS_PER_YEAR


# =============================================================================
# LOADERS
# =============================================================================
def load_cohort_patients(path, extra_columns=()):
    """The patient table: every patient, in the analysis population or not.
    extra_columns: further optional fields from PATIENT_OPTIONAL_COLUMNS."""
    for c in extra_columns:
        if c not in PATIENT_OPTIONAL_COLUMNS:
            raise KeyError(f"patient table: {c} is not a known column")
    cols = PATIENT_CORE_COLUMNS + list(extra_columns)
    pts = read_parquet_columns(path, cols, "patient table")
    pts["person_id"] = pts["person_id"].astype("Int64")
    if pts["person_id"].isna().any():
        raise ValueError("patient table: null patient ID")
    check_unique(pts, ["person_id"], "patient table")
    for c in ["aid_status", "female", "in_population"]:
        check_flag(pts, c, "patient table")
    for c in ["birth_date", "ad_diagnosis_date", "death_date"]:
        pts[c] = to_date(pts[c])
    for c, kind in PATIENT_OPTIONAL_COLUMNS.items():
        if c in pts.columns:
            if kind == "flag":
                check_flag(pts, c, "patient table")
            elif kind == "date":
                pts[c] = to_date(pts[c])
    if pts["birth_date"].isna().any():
        raise ValueError(f"patient table: {int(pts['birth_date'].isna().sum()):,} "
                         f"patients without a birth date")
    if pts["ad_diagnosis_date"].isna().any():
        raise ValueError(f"patient table: {int(pts['ad_diagnosis_date'].isna().sum()):,} "
                         f"patients without an AD diagnosis date")
    return pts


def load_cohort_notes(path, patients):
    """The note table; every note must belong to a patient in `patients`."""
    notes = read_parquet_columns(path, NOTE_COLUMNS, "note table")
    notes["person_id"] = notes["person_id"].astype("Int64")
    check_unique(notes, ["note_id"], "note table")
    check_flag(notes, "cognitive_specialty_note", "note table")
    notes["note_date"] = to_date(notes["note_date"])
    outside = ~notes["person_id"].isin(patients["person_id"])
    if outside.any():
        raise ValueError(f"note table: {int(outside.sum()):,} notes of patients "
                         f"outside the patient table")
    return notes


def add_derived_values(patients, notes):
    """The derived per-patient values. Patients without notes get counts of 0
    and null dates.

    age_at_dx                    AD diagnosis date - birth date, years
    first_entry_date / last_entry_date   earliest / latest dated note
    age_at_entry, age_at_last_note       age at first / last dated note
    documentation_span_yrs       last - first dated note, years
    predx_documentation_span_yrs AD diagnosis date - first dated note, years,
                                 never clipped
    note_count                   every note, undated included
    note_count_dated             dated notes
    note_count_predx             dated notes strictly before the diagnosis
    note_count_postdx            dated notes on or after the diagnosis
    cog_spec_note_count          cognitive-specialty notes, undated included
    death_flag                   a death date is recorded
    """
    pts = patients.copy()
    n = notes.merge(pts[["person_id", "ad_diagnosis_date"]], on="person_id",
                    how="left", validate="many_to_one")
    n["dated"] = n["note_date"].notna().astype(int)
    n["pre"] = is_pre_diagnosis(n["note_date"], n["ad_diagnosis_date"]).astype(int)
    n["post"] = is_post_diagnosis(n["note_date"], n["ad_diagnosis_date"]).astype(int)
    agg = n.groupby("person_id").agg(
        note_count=("note_id", "size"),
        note_count_dated=("dated", "sum"),
        note_count_predx=("pre", "sum"),
        note_count_postdx=("post", "sum"),
        cog_spec_note_count=("cognitive_specialty_note", "sum"),
        first_entry_date=("note_date", "min"),
        last_entry_date=("note_date", "max"),
    ).reset_index()
    pts = pts.merge(agg, on="person_id", how="left", validate="one_to_one")
    for c in ["note_count", "note_count_dated", "note_count_predx",
              "note_count_postdx", "cog_spec_note_count"]:
        pts[c] = pts[c].fillna(0).astype(int)
    pts["age_at_dx"] = years_between(pts["ad_diagnosis_date"], pts["birth_date"])
    pts["age_at_entry"] = years_between(pts["first_entry_date"], pts["birth_date"])
    pts["age_at_last_note"] = years_between(pts["last_entry_date"], pts["birth_date"])
    pts["documentation_span_yrs"] = years_between(pts["last_entry_date"],
                                                  pts["first_entry_date"])
    pts["predx_documentation_span_yrs"] = years_between(pts["ad_diagnosis_date"],
                                                        pts["first_entry_date"])
    pts["death_flag"] = pts["death_date"].notna().astype(int)
    return pts


def check_in_population(patients, notes):
    """Raise unless the analysis-population flag is set exactly for the
    patients with a note."""
    has_note = patients["person_id"].isin(notes["person_id"]).astype(int)
    if (has_note != patients["in_population"]).any():
        raise ValueError("the analysis-population flag of the patient table disagrees with "
                         "the note table (the population = patients with at least one note)")


def load_cohort(patients_path, notes_path, extra_columns=()):
    """Every patient of the patient table (in the analysis population or not)
    with derived values, and every note. Patients without notes get counts of
    0 and null note dates and ages. For the cohort tables, which compare the
    analysis population with the patients without notes. Prints the counts
    every run."""
    patients = load_cohort_patients(patients_path, extra_columns)
    notes = load_cohort_notes(notes_path, patients)
    check_in_population(patients, notes)
    cohort = add_derived_values(patients, notes)
    pop = cohort["in_population"] == 1
    print(f"  Patient table: {len(cohort):,} patients; analysis population "
          f"(with notes): {int(pop.sum()):,}; without notes: {int((~pop).sum()):,}")
    count_line("notes", len(notes))
    count_line("undated notes", notes["note_date"].isna().sum(), len(notes))
    count_line("population patients with no dated note",
               (cohort.loc[pop, "note_count_dated"] == 0).sum(), int(pop.sum()))
    return cohort, notes


def load_population(patients_path, notes_path, extra_columns=()):
    """The analysis population (patients with notes) with derived values, and
    the notes of those patients. Prints the data-handling counts every run."""
    patients = load_cohort_patients(patients_path, extra_columns)
    notes = load_cohort_notes(notes_path, patients)
    check_in_population(patients, notes)
    pop_ids = patients.loc[patients["in_population"] == 1, "person_id"]
    pop = add_derived_values(patients[patients["in_population"] == 1], notes)
    notes = notes[notes["person_id"].isin(pop_ids)].copy()
    print(f"  Patient table: {len(patients):,} patients; analysis population "
          f"(with notes): {len(pop):,} "
          f"(AID+ {int(pop['aid_status'].sum()):,}, "
          f"AID- {int((pop['aid_status'] == 0).sum()):,})")
    count_line("notes of the population", len(notes))
    count_line("undated notes", notes["note_date"].isna().sum(), len(notes))
    count_line("population patients with no dated note",
               (pop["note_count_dated"] == 0).sum(), len(pop))
    return pop, notes


def load_ps_weights(path, population):
    """The propensity-score weights joined to the population. Raises on a
    missing, non-positive or duplicate weight, or a weight outside the
    population."""
    w = read_parquet_columns(path, PS_WEIGHT_COLUMNS, "PS weights")
    w["person_id"] = w["person_id"].astype("Int64")
    check_unique(w, ["person_id"], "PS weights")
    outside = ~w["person_id"].isin(population["person_id"])
    if outside.any():
        raise ValueError(f"PS weights: {int(outside.sum()):,} weights for patients "
                         f"outside the analysis population")
    out = population.merge(w, on="person_id", how="left", validate="one_to_one")
    bad = out["ps_weight_trimmed"].isna() | (out["ps_weight_trimmed"] <= 0)
    if bad.any():
        raise ValueError(f"PS weights: {int(bad.sum()):,} population patients without "
                         f"a positive PS weight")
    return out


def load_symptom_records(path, population, notes):
    """The symptom records with each record's note date (from the note table).
    Raises on a category outside the 56-category schema, a record of a patient
    outside the analysis population, a note not in the note table, or a note of
    another patient. Undated records are kept (no date) and counted."""
    rec = read_parquet_columns(path, SYMPTOM_RECORD_COLUMNS, "symptom records")
    rec["person_id"] = rec["person_id"].astype("Int64")
    bad = rec["category"].isna() | ~rec["category"].isin(SCHEMA_CATEGORIES)
    if bad.any():
        vals = rec.loc[bad, "category"].astype(str).value_counts().head(10).to_dict()
        raise ValueError(f"symptom records: {int(bad.sum()):,} records with a category "
                         f"outside the 56-category schema: {vals}")
    outside = ~rec["person_id"].isin(population["person_id"])
    if outside.any():
        raise ValueError(f"symptom records: {int(outside.sum()):,} records of patients "
                         f"outside the analysis population")
    nn = notes[["note_id", "person_id", "note_date"]].rename(
        columns={"person_id": "note_person_id"})
    rec = rec.merge(nn, on="note_id", how="left", validate="many_to_one", indicator=True)
    missing = rec["_merge"] == "left_only"
    if missing.any():
        raise ValueError(f"symptom records: {int(missing.sum()):,} records on notes "
                         f"not in the note table")
    wrong = rec["note_person_id"] != rec["person_id"]
    if wrong.any():
        raise ValueError(f"symptom records: {int(wrong.sum()):,} records on a note of "
                         f"another patient")
    rec = rec.drop(columns=["_merge", "note_person_id"])
    undated = rec["note_date"].isna()
    count_line("symptom records", len(rec))
    count_line("symptom records on undated notes", undated.sum(), len(rec))
    count_line("patients with a symptom record on an undated note",
               rec.loc[undated, "person_id"].nunique())
    return rec


def load_aid_subtypes(path, population):
    """The AID subtype table (CNS-primary AID, 0/1) joined to the population.
    Raises on a population patient without a row, a flag other than 0/1, or
    the flag set for an AID- patient."""
    sub_ = read_parquet_columns(path, AID_SUBTYPE_COLUMNS, "AID subtypes")
    sub_["person_id"] = sub_["person_id"].astype("Int64")
    check_unique(sub_, ["person_id"], "AID subtypes")
    check_flag(sub_, "has_cns_primary_aid", "AID subtypes")
    out = population.merge(sub_, on="person_id", how="left", validate="one_to_one")
    if out["has_cns_primary_aid"].isna().any():
        raise ValueError(f"AID subtypes: {int(out['has_cns_primary_aid'].isna().sum()):,} "
                         f"population patients without a row")
    out["has_cns_primary_aid"] = out["has_cns_primary_aid"].astype(int)
    if ((out["has_cns_primary_aid"] == 1) & (out["aid_status"] == 0)).any():
        raise ValueError("AID subtypes: the CNS-primary flag is set for an AID- patient")
    return out


# =============================================================================
# FEATURE RECORDS: TIERS, CHANGE LANGUAGE, EXPLICIT TIME, SCORES,
# MEDICATION EXPOSURE
# =============================================================================
# Inflammation tiers: the records of these categories carry the tiers. The
# inflammation-record tier is mapped from the category here.
TIER_CATEGORIES = ["systemic_inflammation_autoimmune", "arthralgia", "myalgia"]
TIER_RECORD_COLUMNS = ["person_id", "note_id", "category", "tier_lab_result", "tier_flare"]
# (record-level column, patient-level name, readable label)
TIERS = [("tier_inflammation_record", "inflammation_record", "Inflammation record"),
         ("tier_lab_result", "lab_result", "Lab result"),
         ("tier_flare", "flare", "Flare")]
CHANGE_RECORD_COLUMNS = ["person_id", "note_id", "category", "worsening", "improvement",
                         "severity_level"]
SEVERITY_LEVELS = ["severe", "moderate", "mild"]
EXPLICIT_TIME_COLUMNS = ["person_id", "note_id", "category", "kind", "implied_offset_days"]
EXPLICIT_TIME_KINDS = ["years_ago", "months_ago", "since_year"]
COGNITIVE_SCORE_COLUMNS = ["person_id", "note_id", "score_type", "score_value", "score_date",
                           "extraction_confidence", "is_change_score", "sub_score_type"]
MEDICATION_RECORD_COLUMNS = ["person_id", "drug_class", "exposure_date"]
DRUG_CLASSES = ["on_cs", "on_dmard", "on_biologic", "on_antimalarial", "on_ms_dmt",
                "on_chei", "on_nmda_antagonist", "on_anti_amyloid", "on_nsaid",
                "tnf_i_exposed", "biologic_exposed_broad"]


def _note_records(rec, population, notes, name):
    """Shared checks of a note-keyed feature table and the note date join:
    raises on a record of a patient outside the analysis population, a note
    not in the note table, or a note of another patient. Records on
    undated notes are kept (no date) and counted."""
    rec["person_id"] = rec["person_id"].astype("Int64")
    outside = ~rec["person_id"].isin(population["person_id"])
    if outside.any():
        raise ValueError(f"{name}: {int(outside.sum()):,} records of patients outside the "
                         f"analysis population")
    nn = notes[["note_id", "person_id", "note_date"]].rename(
        columns={"person_id": "note_person_id"})
    rec = rec.merge(nn, on="note_id", how="left", validate="many_to_one", indicator=True)
    missing = rec["_merge"] == "left_only"
    if missing.any():
        raise ValueError(f"{name}: {int(missing.sum()):,} records on notes not in the note table")
    wrong = rec["note_person_id"] != rec["person_id"]
    if wrong.any():
        raise ValueError(f"{name}: {int(wrong.sum()):,} records on a note of another patient")
    rec = rec.drop(columns=["_merge", "note_person_id"])
    count_line(f"{name} records", len(rec))
    count_line(f"{name} records on undated notes", rec["note_date"].isna().sum(), len(rec))
    return rec


def load_tier_records(path, population, notes):
    """The inflammation tier records with note dates; adds the
    inflammation-record tier (category = systemic inflammation autoimmune).
    Raises on a category outside the three tier categories or a tier flag
    other than 0/1 (and the shared record checks)."""
    rec = read_parquet_columns(path, TIER_RECORD_COLUMNS, "inflammation tier")
    bad = ~rec["category"].isin(TIER_CATEGORIES)
    if bad.any():
        raise ValueError(f"inflammation tier: {int(bad.sum()):,} records outside the "
                         f"tier categories {TIER_CATEGORIES}")
    for c in ("tier_lab_result", "tier_flare"):
        check_flag(rec, c, "inflammation tier")
    rec["tier_inflammation_record"] = (rec["category"] == TIER_CATEGORIES[0]).astype(int)
    return _note_records(rec, population, notes, "inflammation tier")


def load_change_language_records(path, population, notes):
    """The change-language records with note dates. Raises on a category
    that is not domain-mapped, a flag other than 0/1 or a severity level
    outside severe / moderate / mild (null allowed)."""
    rec = read_parquet_columns(path, CHANGE_RECORD_COLUMNS, "change language")
    mapped = set(SCHEMA_CATEGORIES) - set(UNASSIGNED_CATEGORIES)
    bad = ~rec["category"].isin(mapped)
    if bad.any():
        raise ValueError(f"change language: {int(bad.sum()):,} records with a "
                         f"category that is not domain-mapped")
    for c in ("worsening", "improvement"):
        check_flag(rec, c, "change language")
    lvl = rec["severity_level"].notna() & ~rec["severity_level"].isin(SEVERITY_LEVELS)
    if lvl.any():
        raise ValueError(f"change language: {int(lvl.sum()):,} severity levels "
                         f"outside {SEVERITY_LEVELS}")
    return _note_records(rec, population, notes, "change language")


def load_explicit_time_records(path, population, notes):
    """The explicit-time records with note dates. Raises on a category
    outside the schema, a kind outside EXPLICIT_TIME_KINDS, or a dated record
    without an implied offset from the note date."""
    rec = read_parquet_columns(path, EXPLICIT_TIME_COLUMNS, "explicit time")
    bad = rec["category"].isna() | ~rec["category"].isin(SCHEMA_CATEGORIES)
    if bad.any():
        raise ValueError(f"explicit time: {int(bad.sum()):,} records with a category "
                         f"outside the 56-category schema")
    bad = ~rec["kind"].isin(EXPLICIT_TIME_KINDS)
    if bad.any():
        raise ValueError(f"explicit time: {int(bad.sum()):,} records with a kind "
                         f"outside {EXPLICIT_TIME_KINDS}")
    rec["implied_offset_days"] = rec["implied_offset_days"].astype("Int64")
    rec = _note_records(rec, population, notes, "explicit time")
    bad = rec["note_date"].notna() & rec["implied_offset_days"].isna()
    if bad.any():
        raise ValueError(f"explicit time: {int(bad.sum()):,} dated records without "
                         f"an implied offset")
    return rec


def load_cognitive_scores(path, population, notes):
    """The cognitive score records. Raises on a change-score flag other than
    0/1 and the shared record checks. The score date is the measurement date
    (may be null); the note date is joined for the checks only."""
    rec = read_parquet_columns(path, COGNITIVE_SCORE_COLUMNS, "cognitive score")
    check_flag(rec, "is_change_score", "cognitive score")
    rec["score_value"] = pd.to_numeric(rec["score_value"], errors="raise").astype(float)
    rec["score_date"] = to_date(rec["score_date"])
    return _note_records(rec, population, notes, "cognitive score")


# The scores the analyses use and the annual change: one definition for every
# script that analyzes the annual change of MMSE and MoCA scores.
COGNITIVE_SCORE_TYPES = ["MMSE", "MoCA"]
COGNITIVE_CONFIDENCE_KEPT = ["high", "medium"]
MIN_SLOPE_GAP_YRS = 0.25   # first-to-last score gap for an annual change


def absolute_cognitive_scores(scores, population):
    """The scores the analyses use, from load_cognitive_scores: MMSE and MoCA
    absolute scores with a value, extracted with high or medium confidence,
    not a change or sub-score row, with a measurement date (score_date).
    Prints the score rows and the rows removed by each filter (every run);
    joins aid_status, female and birth_date from `population` and adds
    age_at_score = (score_date - birth_date) in whole days / 365.25.
    Returns (scores, filter rows: filter, rows_removed, rows_before)."""
    print(f"  Score rows: {len(scores):,}")
    filter_rows = []
    for label, keep in (
            ("not MMSE or MoCA", scores["score_type"].isin(COGNITIVE_SCORE_TYPES)),
            ("no score value", scores["score_value"].notna()),
            ("confidence not high or medium",
             scores["extraction_confidence"].isin(COGNITIVE_CONFIDENCE_KEPT)),
            ("change score", scores["is_change_score"] == 0),
            ("sub-score", scores["sub_score_type"].isna()),
            ("no score date", scores["score_date"].notna())):
        count_line(f"rows removed, {label}", (~keep).sum(), len(scores))
        filter_rows.append({"filter": label, "rows_removed": int((~keep).sum()),
                            "rows_before": len(scores)})
        scores = scores[keep]
    scores = scores.merge(population[["person_id", "aid_status", "female", "birth_date"]],
                          on="person_id", how="left", validate="many_to_one")
    scores["age_at_score"] = years_between(scores["score_date"], scores["birth_date"])
    print(f"  Absolute scores kept: {len(scores):,} rows, "
          f"{scores['person_id'].nunique():,} patients")
    return scores, filter_rows


def annual_change(scores, score_type):
    """The annual change of one score type (scores from
    absolute_cognitive_scores): per patient with two or more scores whose
    first and last are at least MIN_SLOPE_GAP_YRS apart (ordered by age at
    the score), slope = (last - first score) / years between them. One row
    per patient: person_id, slope, age_first (age at the first score),
    gap_yrs, last_score_date, aid_status, female, age_c (age_first centered
    within the score type) and age_c_sq."""
    d = scores[scores["score_type"] == score_type].copy()
    srows = []
    for pid, g in d.groupby("person_id"):
        g = g.sort_values("age_at_score")
        if len(g) < 2:
            continue
        gap = g["age_at_score"].iloc[-1] - g["age_at_score"].iloc[0]
        if gap < MIN_SLOPE_GAP_YRS:
            continue
        srows.append({"person_id": pid,
                      "slope": (g["score_value"].iloc[-1] - g["score_value"].iloc[0]) / gap,
                      "age_first": g["age_at_score"].iloc[0], "gap_yrs": gap,
                      "last_score_date": g["score_date"].iloc[-1],
                      "aid_status": g["aid_status"].iloc[0], "female": g["female"].iloc[0]})
    s = pd.DataFrame(srows, columns=["person_id", "slope", "age_first", "gap_yrs",
                                     "last_score_date", "aid_status", "female"])
    s["age_c"] = s["age_first"] - s["age_first"].mean()
    s["age_c_sq"] = s["age_c"] ** 2
    return s


def load_medication_records(path, population):
    """The medication exposure records of the analysis population (records of
    other patients are counted and not used). Raises on a drug class outside
    DRUG_CLASSES. The exposure date may be null."""
    rec = read_parquet_columns(path, MEDICATION_RECORD_COLUMNS, "medication exposure")
    rec["person_id"] = rec["person_id"].astype("Int64")
    bad = ~rec["drug_class"].isin(DRUG_CLASSES)
    if bad.any():
        raise ValueError(f"medication exposure: {int(bad.sum()):,} records with a "
                         f"drug class outside {DRUG_CLASSES}")
    rec["exposure_date"] = to_date(rec["exposure_date"])
    keep = rec["person_id"].isin(population["person_id"])
    count_line("medication records of patients outside the analysis population (not used)",
               (~keep).sum(), len(rec))
    rec = rec[keep].copy()
    count_line("medication records with no exposure date", rec["exposure_date"].isna().sum(),
               len(rec))
    return rec


# =============================================================================
# DLB FEATURES: FIRST DATES AND DIAGNOSIS-DAY WINDOWS
# =============================================================================
DLB_MENTION_COLUMNS = ["person_id", "note_id", "diagnosis_date", "feature", "source",
                       "kept_definition", "kept_final"]
DLB_FEATURES = ["fluct", "vis_hall", "aud_hall", "rbd_conf", "rbd_prob", "ortho", "syncope",
                "hyper", "incont", "constipation", "hyposmia", "visuospatial", "executive",
                "brady", "rigid", "tremor", "park_standalone", "neuro_sens", "neuro_sens_3plus",
                "postural_instability", "repeated_falls", "transient_unresponsiveness",
                "systematized_delusions"]
# Parkinsonism composites: the earliest date over their components.
DLB_COMPOSITES = {"park_core": ["brady", "rigid", "tremor"],
                  "park_all": ["brady", "rigid", "tremor", "park_standalone"]}
DLB_MENTION_SOURCES = ["symptom_row", "diagnosis"]


def load_dlb_feature_mentions(path, population, notes):
    """The DLB feature mentions with each mention's date: the note date (from
    the note table) for a mention in a note, the diagnosis date for a
    coded-diagnosis mention. Raises on a feature or source outside the lists, a
    flag other than 0/1, a mention of a patient outside the analysis
    population, a note mention without a note in the note table or on a note
    of another patient, or a diagnosis mention without a date. Mentions on
    undated notes are kept and counted."""
    rec = read_parquet_columns(path, DLB_MENTION_COLUMNS, "DLB feature mentions")
    rec["person_id"] = rec["person_id"].astype("Int64")
    bad = ~rec["feature"].isin(DLB_FEATURES)
    if bad.any():
        raise ValueError(f"DLB feature mentions: {int(bad.sum()):,} mentions of a feature "
                         f"outside {DLB_FEATURES}")
    bad = ~rec["source"].isin(DLB_MENTION_SOURCES)
    if bad.any():
        raise ValueError(f"DLB feature mentions: {int(bad.sum()):,} mentions with a source "
                         f"outside {DLB_MENTION_SOURCES}")
    for c in ("kept_definition", "kept_final"):
        check_flag(rec, c, "DLB feature mentions")
    outside = ~rec["person_id"].isin(population["person_id"])
    if outside.any():
        raise ValueError(f"DLB feature mentions: {int(outside.sum()):,} mentions of patients "
                         f"outside the analysis population")
    rec["diagnosis_date"] = to_date(rec["diagnosis_date"])
    dx = rec["source"] == "diagnosis"
    if rec.loc[dx, "diagnosis_date"].isna().any():
        raise ValueError(f"DLB feature mentions: "
                         f"{int(rec.loc[dx, 'diagnosis_date'].isna().sum()):,} diagnosis "
                         f"mentions without a date")
    sym = rec[~dx].drop(columns="diagnosis_date")
    nn = notes[["note_id", "person_id", "note_date"]].rename(
        columns={"person_id": "note_person_id"})
    sym = sym.merge(nn, on="note_id", how="left", validate="many_to_one", indicator=True)
    if (sym["_merge"] == "left_only").any():
        raise ValueError(f"DLB feature mentions: {int((sym['_merge'] == 'left_only').sum()):,} "
                         f"symptom mentions on notes not in the note table")
    if (sym["note_person_id"] != sym["person_id"]).any():
        raise ValueError(f"DLB feature mentions: "
                         f"{int((sym['note_person_id'] != sym['person_id']).sum()):,} "
                         f"symptom mentions on a note of another patient")
    sym = sym.rename(columns={"note_date": "date"}).drop(columns=["_merge", "note_person_id"])
    dxr = rec[dx].rename(columns={"diagnosis_date": "date"})
    out = pd.concat([sym, dxr[sym.columns]], ignore_index=True)
    count_line("DLB feature mentions", len(out))
    count_line("DLB feature mentions from coded diagnoses", int(dx.sum()), len(out))
    count_line("DLB feature mentions on undated notes", out["date"].isna().sum(), len(out))
    return out


def dlb_feature_first_dates(mentions):
    """First dates per patient x feature, from the mention flags as given:
    first_date is the earliest analyzed mention, the date every analysis
    uses; the two other dates are carried for completeness. The composites
    park_core and park_all are the earliest over their components, for each
    date. Undated mentions are not used."""
    d = mentions[mentions["date"].notna()]

    def firsts(df):
        return df.groupby(["person_id", "feature"])["date"].min()

    fd = pd.concat({"first_date_matched": firsts(d),
                    "first_date_definition": firsts(d[d["kept_definition"] == 1]),
                    "first_date": firsts(d[d["kept_final"] == 1])}, axis=1).reset_index()
    comp = []
    cols = ["first_date_matched", "first_date_definition", "first_date"]
    for name, parts in DLB_COMPOSITES.items():
        c = fd[fd["feature"].isin(parts)].groupby("person_id")[cols].min().reset_index()
        c["feature"] = name
        comp.append(c)
    return pd.concat([fd] + comp, ignore_index=True)[["person_id", "feature"] + cols]


def _window_days(years):
    """Whole days within `years` years: (date - anchor).days / 365.25 <= years
    holds exactly when (date - anchor).days <= floor(years x 365.25)."""
    return int(np.floor(years * DAYS_PER_YEAR))


def documented_by_years_after_diagnosis(dates, ad_diagnosis_date, years):
    """True where a date falls no later than `years` years after the AD
    diagnosis date ((date - diagnosis).days / 365.25 <= years), e.g. the
    pre/peri-diagnosis window of the DLB analyses. Implemented with
    is_pre_diagnosis against the anchor diagnosis + (floor(years x 365.25) + 1)
    days. Undated records are False."""
    anchor = ad_diagnosis_date + pd.Timedelta(days=_window_days(years) + 1)
    return is_pre_diagnosis(dates, anchor)


def documented_from_years_before_diagnosis(dates, ad_diagnosis_date, years):
    """True where a date falls no earlier than `years` years before the AD
    diagnosis date ((date - diagnosis).days / 365.25 >= -years), e.g. the
    McKeith window of parkinsonism. Implemented with is_post_diagnosis against
    the anchor diagnosis - floor(years x 365.25) days. Undated records are
    False."""
    anchor = ad_diagnosis_date - pd.Timedelta(days=_window_days(years))
    return is_post_diagnosis(dates, anchor)


# =============================================================================
# AGE CEILING
# =============================================================================
def p90_age_ceiling(population):
    """Sex-specific ceiling: the 90th percentile of age_at_last_note among
    AID- patients of the analysis population (population = the rows passed).
    Returns ceiling_female, ceiling_male, n_ref_female, n_ref_male and
    ceiling_by_person (Series, person_id -> the patient's own-sex ceiling)."""
    for c in ("person_id", "aid_status", "female", "age_at_last_note"):
        if c not in population.columns:
            raise KeyError(f"p90_age_ceiling: missing column {c}")
    ref = population[population["aid_status"] == 0]
    ref_f = ref.loc[ref["female"] == 1, "age_at_last_note"].dropna()
    ref_m = ref.loc[ref["female"] == 0, "age_at_last_note"].dropna()
    ceiling = {1: ref_f.quantile(AGE_CEILING_QUANTILE),
               0: ref_m.quantile(AGE_CEILING_QUANTILE)}
    by_person = population.set_index("person_id")["female"].map(ceiling)
    return {"ceiling_female": float(ceiling[1]), "ceiling_male": float(ceiling[0]),
            "n_ref_female": len(ref_f), "n_ref_male": len(ref_m),
            "ceiling_by_person": by_person}


def within_age_ceiling(age_at_record, ceiling):
    """True where the record counts under the ceiling: age at the record
    at or below the patient's ceiling. Undated records (NaN age) are False."""
    return (age_at_record <= ceiling).fillna(False).astype(bool)


# =============================================================================
# FLOOR CHECKS
# =============================================================================
def model_floor_reason(y, aid, n_terms, female=None, sex=None, continuous=False,
                       modifier=None):
    """Why a model of the outcome y on aid (+ other terms) is not fitted, or ""
    when it may be fitted. One gate for every model in every analysis
    (aid=None for a model without an AID term, e.g. the DLB discriminant:
    the events-per-term floor on the rarer outcome class only):
      - >= MIN_EVENTS_PER_TERM x n_terms patients in the rarer outcome
        class of a 0/1 outcome (every right-hand-side term counted,
        interactions included); for a continuous outcome every patient
        counts (continuous=True);
      - both AID groups present; for a 0/1 outcome, the outcome varies
        within each;
      - a model with a sex term (sex = the 0/1 sex column): both sexes
        present;
      - interaction models (female = the 0/1 sex column for AID x sex, or
        modifier = another 0/1 factor for AID x that factor, e.g. an
        inflammation tier or corticosteroid exposure): each of the four
        groups has >= MIN_GROUP_N patients in its rarer outcome class
        (every patient of the group for a continuous outcome)."""
    if female is not None and modifier is not None:
        raise ValueError("model_floor_reason: give female or modifier, not both")
    y = pd.Series(y).reset_index(drop=True)
    if aid is None:
        if female is not None or modifier is not None or sex is not None:
            raise ValueError("model_floor_reason: aid=None takes no group arguments")
        if not continuous:
            y = y.astype(int)
        if len(y) == 0:
            return "no data"
        if continuous:
            m, unit = len(y), "patients"
        else:
            ev = int(y.sum())
            m, unit = min(ev, len(y) - ev), "events"
        if m < MIN_EVENTS_PER_TERM * n_terms:
            return f"{m} {unit} < {MIN_EVENTS_PER_TERM} x {n_terms} terms"
        return ""
    aid = pd.Series(aid).astype(int).reset_index(drop=True)
    if not continuous:
        y = y.astype(int)
    if len(y) == 0:
        return "no data"
    if continuous:
        m, unit = len(y), "patients"
    else:
        ev = int(y.sum())
        m, unit = min(ev, len(y) - ev), "events"
    if m < MIN_EVENTS_PER_TERM * n_terms:
        return f"{m} {unit} < {MIN_EVENTS_PER_TERM} x {n_terms} terms"
    if aid.nunique() < 2:
        return "only one AID group present"
    if not continuous:
        for _, g in y.groupby(aid):
            if g.sum() == 0 or g.sum() == len(g):
                return "outcome constant within an AID group"
    if sex is not None and pd.Series(sex).nunique() < 2:
        return "only one sex present"
    if female is not None or modifier is not None:
        factor = pd.Series(female if female is not None else modifier).astype(int) \
            .reset_index(drop=True)
        groups = (((1, 1, "AID+ female"), (1, 0, "AID+ male"),
                   (0, 1, "AID- female"), (0, 0, "AID- male")) if female is not None else
                  ((1, 1, "AID+ exposed"), (1, 0, "AID+ unexposed"),
                   (0, 1, "AID- exposed"), (0, 0, "AID- unexposed")))
        bad = []
        for a, s, lbl in groups:
            g = y[(aid == a) & (factor == s)]
            if continuous:
                k = len(g)
            else:
                e = int(g.sum())
                k = min(e, len(g) - e)
            if k < MIN_GROUP_N:
                bad.append(lbl)
        if bad:
            what = "patients" if continuous else "patients in the rarer outcome class"
            return f"fewer than {MIN_GROUP_N} {what} in: " + ", ".join(bad)
    return ""


def group_floor_reason(y, group, n_terms, labels=("AID+", "AID-")):
    """Why a 0/1 feature y does not enter an FDR family under the per-group
    floor, or "": at least
    MIN_EVENTS_PER_TERM x n_terms patients in the rarer outcome class of y
    within each of the two groups of the 0/1 column group (1 = labels[0],
    0 = labels[1]). It decides FDR membership only: the model of a feature
    below it is still fitted under model_floor_reason and its estimate kept."""
    y = pd.Series(y).astype(int).reset_index(drop=True)
    g = pd.Series(group).astype(int).reset_index(drop=True)
    need = MIN_EVENTS_PER_TERM * n_terms
    bad = []
    for val, lbl in zip((1, 0), labels):
        yy = y[g == val]
        e = int(yy.sum())
        k = min(e, len(yy) - e)
        if k < need:
            bad.append(f"{lbl} ({k})")
    if bad:
        return f"fewer than {need} patients in the rarer outcome class in: " + ", ".join(bad)
    return ""


def two_by_two_reason(e1, n1, e2, n2, event):
    """Why a 2x2 comparison of e1/n1 with e2/n2 is not tested, or "".
    The group minimum first (>= MIN_GROUP_N patients in each compared group),
    then the zero-event rule (no patient, or every patient, in both groups has
    the event). One group at 0 or at n is tested."""
    if n1 < MIN_GROUP_N or n2 < MIN_GROUP_N:
        return f"fewer than {MIN_GROUP_N} patients in a compared group"
    if e1 + e2 == 0:
        return f"no patient with {event} in either group"
    if e1 + e2 == n1 + n2:
        return f"every patient with {event} in both groups"
    return ""


def group_minimum_reason(*group_sizes):
    """Why a descriptive test comparing groups is not run, or "":
    >= MIN_GROUP_N patients (non-missing values) in each compared group. Used
    for Mann-Whitney U and the chi-square test over several categories; 2x2
    comparisons use two_by_two_reason (the group minimum, then the
    zero-event rule)."""
    if min(int(n) for n in group_sizes) < MIN_GROUP_N:
        return f"fewer than {MIN_GROUP_N} patients in a compared group"
    return ""


# =============================================================================
# STANDARDIZED MEAN DIFFERENCES
# =============================================================================
# Unweighted, continuous: pooled SD = sqrt(mean of the two groups' sample
# variances, ddof=1). Weighted, continuous: weighted means and each group's own
# weighted variance (no ddof correction), averaged. Binary 0/1 variables use
# the proportion-based SMD, (p1 - p2) / sqrt((p1(1-p1) + p2(1-p2)) / 2), with
# weighted proportions in the weighted case.

def smd(var, group, pos=1, neg=0, min_n=5):
    df = pd.DataFrame({"v": var, "g": group}).dropna()
    a = df[df["g"] == pos]["v"]
    b = df[df["g"] == neg]["v"]
    if len(a) < min_n or len(b) < min_n:
        return float("nan")
    pooled_sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return (a.mean() - b.mean()) / pooled_sd if pooled_sd > 0 else float("nan")


def smd_from_groups(a, b, min_n=2):
    a = pd.to_numeric(pd.Series(a), errors="coerce").dropna()
    b = pd.to_numeric(pd.Series(b), errors="coerce").dropna()
    if len(a) < min_n or len(b) < min_n:
        return float("nan")
    pooled_sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return (a.mean() - b.mean()) / pooled_sd if pooled_sd > 0 else float("nan")


def smd_weighted(df, col, group_col="aid_status", weight_col="ps_weight_trimmed",
                 pos=1, neg=0, min_n=10):
    d = df[[group_col, col, weight_col]].dropna()
    if len(d) < min_n:
        return float("nan")
    out = []
    for lvl in (pos, neg):
        g = d[d[group_col] == lvl]
        if len(g) < 2 or g[weight_col].sum() <= 0:
            return float("nan")
        mu = np.average(g[col], weights=g[weight_col])
        vr = np.average((g[col] - mu) ** 2, weights=g[weight_col])
        out.append((mu, vr))
    pooled_sd = np.sqrt((out[0][1] + out[1][1]) / 2)
    return (out[0][0] - out[1][0]) / pooled_sd if pooled_sd > 0 else float("nan")


def smd_weighted_from_groups(df_pos, df_neg, col, wcol="ps_weight_trimmed"):
    out = []
    for g in (df_pos, df_neg):
        d = g[[col, wcol]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(d) < 2 or d[wcol].sum() <= 0:
            return float("nan")
        mu = np.average(d[col], weights=d[wcol])
        vr = np.average((d[col] - mu) ** 2, weights=d[wcol])
        out.append((mu, vr))
    pooled_sd = np.sqrt((out[0][1] + out[1][1]) / 2)
    return (out[0][0] - out[1][0]) / pooled_sd if pooled_sd > 0 else float("nan")


def _proportion_smd(p1, p2):
    denom = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2)
    return (p1 - p2) / denom if denom > 0 else float("nan")


def smd_proportion(var, group, pos=1, neg=0, min_n=5):
    df = pd.DataFrame({"v": var, "g": group}).dropna()
    a = df[df["g"] == pos]["v"]
    b = df[df["g"] == neg]["v"]
    if len(a) < min_n or len(b) < min_n:
        return float("nan")
    return _proportion_smd(a.mean(), b.mean())


def smd_proportion_weighted(df, col, group_col="aid_status",
                            weight_col="ps_weight_trimmed", pos=1, neg=0, min_n=10):
    d = df[[group_col, col, weight_col]].dropna()
    if len(d) < min_n:
        return float("nan")
    props = []
    for lvl in (pos, neg):
        g = d[d[group_col] == lvl]
        if len(g) < 2 or g[weight_col].sum() <= 0:
            return float("nan")
        props.append(np.average(g[col], weights=g[weight_col]))
    return _proportion_smd(props[0], props[1])


def balance_label(s):
    """Residual imbalance from |SMD|: balanced < 0.10 <= moderate < 0.20 <= severe."""
    if pd.isna(s):
        return "n/a"
    if abs(s) < 0.10:
        return "balanced"
    if abs(s) < 0.20:
        return "moderate"
    return "severe"


def effective_sample_size(weights):
    w = pd.Series(weights, dtype=float).dropna()
    return (w.sum() ** 2) / (w ** 2).sum() if len(w) > 0 else 0.0


# =============================================================================
# WILSON SCORE INTERVAL
# =============================================================================
Z_975 = 1.959963984540054   # scipy.stats.norm.ppf(0.975)


def wilson_ci(k, n):
    """Proportion k/n with its Wilson score 95% interval, as percentages
    (pct, low, high); NaN for n = 0. The one implementation used wherever a
    proportion is shown with an interval."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    den = 1 + Z_975 ** 2 / n
    c = (p + Z_975 ** 2 / (2 * n)) / den
    h = Z_975 * np.sqrt(p * (1 - p) / n + Z_975 ** 2 / (4 * n ** 2)) / den
    return 100 * p, 100 * max(0.0, c - h), 100 * min(1.0, c + h)


# =============================================================================
# FDR
# =============================================================================
def bh_fdr(pvals):
    """Benjamini-Hochberg step-up q values, NaN-safe, order-preserving.
    NaN entries (rows not tested) stay NaN and are not in the family."""
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    idx = np.where(~np.isnan(p))[0]
    if len(idx) == 0:
        return out
    sub_p = p[idx]
    order = np.argsort(sub_p, kind="mergesort")
    ranked = sub_p[order]
    n = len(ranked)
    q = ranked / (np.arange(1, n + 1) / float(n))   # statsmodels' arithmetic, bit for bit
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.minimum(q, 1.0)
    res = np.empty(n)
    res[order] = q
    out[idx] = res
    return out


# =============================================================================
# OUTPUT
# =============================================================================
def write_workbook(path, sheets):
    """Write {sheet_name: DataFrame} to one new .xlsx under the output root
    (sheet names <= 31 chars), through write_guard.py, which stops if the file
    exists."""
    import write_guard as wg
    for name in sheets:
        if len(name) > 31:
            raise ValueError(f"sheet name over 31 characters: {name}")

    def _write(tmp):
        with pd.ExcelWriter(tmp, engine="openpyxl") as writer:
            for name, df in sheets.items():
                df.to_excel(writer, sheet_name=name, index=False)
    wg.write_new(path, _write)
    print(f"  Saved: {path}  (sheets: {', '.join(sheets)})")


def write_parquet(df, path):
    """Write df to one new file under the output root, through write_guard.py,
    which stops if the file exists."""
    import write_guard as wg
    wg.write_new(path, lambda tmp: df.to_parquet(tmp, index=False))


# =============================================================================
# DLB PATIENT FLAGS (the DLB concepts: core, AID conditions, Sjogren's timing)
# =============================================================================
DLB_MCKEITH_WINDOW_YRS = 1     # parkinsonism first documented from 1 yr before AD dx
DLB_PRE_PERI_WINDOW_YRS = 1    # first documented by 1 yr after AD dx
DLB_FLAG_FEATURES = DLB_FEATURES + list(DLB_COMPOSITES)
# The core-burden features (McKeith 2017 core features without RBD) and the
# first-date column (t_, within the ceiling) that dates each one.
DLB_CORE_BURDEN = [("has_park_all_mckeith", "t_park_all", "Parkinsonism (McKeith)"),
                   ("has_fluct", "t_fluct", "Fluctuating cognition"),
                   ("has_vis_hall", "t_vis_hall", "Visual hallucinations")]


def dlb_patient_flags(pop, mentions):
    """Every per-patient DLB column of the DLB analyses, added to pop (sorted
    by person_id by the caller): d_<feature> the first analyzed date
    (dlb_feature_first_dates), t_<feature> that date when it is at or below
    the patient's age ceiling, and the presence flags has_<...>.
    Presence = a first date within the ceiling; RBD (has_rbd_any) = rbd_conf,
    or rbd_prob without rbd_conf, decided after the ceiling; parkinsonism
    (McKeith) and postural instability (McKeith; has_pi_mckeith) = first
    documented no earlier than 1 year before the AD diagnosis date;
    visuospatial / executive (and the _preperi flags) = first
    documented by 1 year after it (through the window functions). Prints
    the counts and the two ceilings; returns (pop, ceiling dict)."""
    first = dlb_feature_first_dates(mentions)
    wide = first.pivot(index="person_id", columns="feature", values="first_date")
    for f in DLB_FLAG_FEATURES:
        pop["d_" + f] = pd.to_datetime(pop["person_id"].map(wide[f])) if f in wide.columns \
            else pd.Series(pd.NaT, index=pop.index, dtype="datetime64[ns]")
    count_line("patients with at least one analyzed DLB feature mention",
               first.loc[first["first_date"].notna(), "person_id"].nunique(), len(pop))
    ceil = p90_age_ceiling(pop)
    ceiling = pop["person_id"].map(ceil["ceiling_by_person"])
    print(f"  Age ceiling, female (yr): {ceil['ceiling_female']:.2f}  "
          f"(AID- reference n = {ceil['n_ref_female']:,})")
    print(f"  Age ceiling, male (yr)  : {ceil['ceiling_male']:.2f}  "
          f"(AID- reference n = {ceil['n_ref_male']:,})")
    for f in DLB_FLAG_FEATURES:
        age = years_between(pop["d_" + f], pop["birth_date"])
        pop["t_" + f] = pop["d_" + f].where(within_age_ceiling(age, ceiling))
    pop["t_brady_or_rigid"] = pop[["t_brady", "t_rigid"]].min(axis=1)
    dx = pop["ad_diagnosis_date"]
    for f in ("fluct", "vis_hall", "aud_hall", "ortho", "syncope", "hyper", "incont",
              "constipation", "hyposmia", "rbd_conf", "neuro_sens", "neuro_sens_3plus",
              "postural_instability", "repeated_falls", "transient_unresponsiveness",
              "systematized_delusions"):
        pop["has_" + f] = pop["t_" + f].notna().astype(int)
    pop["has_rbd_prob"] = (pop["t_rbd_prob"].notna() & pop["t_rbd_conf"].isna()).astype(int)
    pop["has_rbd_any"] = ((pop["has_rbd_conf"] == 1) | (pop["has_rbd_prob"] == 1)).astype(int)
    pop["has_park_all_ever"] = pop["t_park_all"].notna().astype(int)
    pop["has_vis_seq"] = documented_by_years_after_diagnosis(
        pop["t_visuospatial"], dx, DLB_PRE_PERI_WINDOW_YRS).astype(int)
    pop["has_exec_seq"] = documented_by_years_after_diagnosis(
        pop["t_executive"], dx, DLB_PRE_PERI_WINDOW_YRS).astype(int)
    for f in ("park_all", "park_core", "brady", "rigid", "tremor", "brady_or_rigid"):
        pop[f"has_{f}_mckeith"] = documented_from_years_before_diagnosis(
            pop["t_" + f], dx, DLB_MCKEITH_WINDOW_YRS).astype(int)
    pop["has_pi_mckeith"] = documented_from_years_before_diagnosis(
        pop["t_postural_instability"], dx, DLB_MCKEITH_WINDOW_YRS).astype(int)
    pop["has_park_all_pdd_type"] = (pop["t_park_all"].notna()
                                    & (pop["has_park_all_mckeith"] == 0)).astype(int)
    for f in ("vis_hall", "fluct", "park_all", "brady_or_rigid"):
        pop[f"has_{f}_preperi"] = documented_by_years_after_diagnosis(
            pop["t_" + f], dx, DLB_PRE_PERI_WINDOW_YRS).astype(int)
    return pop, ceil


def dlb_core_burden_first_date(pop):
    """Earliest first date among the core-burden features the patient counts
    as having (presence flag 1): the analyzed date within the ceiling, and
    for parkinsonism (McKeith) only when inside its window."""
    dates = [pop[t].where(pop[h] == 1) for h, t, _ in DLB_CORE_BURDEN]
    return pd.concat(dates, axis=1).min(axis=1)


# =============================================================================
# AID CONDITIONS, DLB DIAGNOSIS FLAGS, DISCRIMINANT COEFFICIENTS (loaders)
# =============================================================================
AID_CONDITION_COLUMNS = ["person_id", "normalized_label", "diagnosis_date"]
DLB_DIAGNOSIS_FLAG_COLUMNS = ["person_id", "ae_any", "ae_first_date", "mci_any",
                              "mci_first_date", "pdd_any", "pdd_first_date"]
DISCRIMINANT_COEFFICIENT_COLUMNS = ["feature", "coefficient"]


def load_aid_conditions(path, population):
    """The coded AID conditions (one row per patient x condition), rows of the
    analysis population; the label stripped and lowercased. Raises on a
    missing label, a repeated patient x condition, a condition row of an
    AID- patient, or an AID+ population patient without one. The diagnosis
    date may be null."""
    rec = read_parquet_columns(path, AID_CONDITION_COLUMNS, "AID conditions")
    rec["person_id"] = rec["person_id"].astype("Int64")
    if rec["normalized_label"].isna().any():
        raise ValueError(f"AID conditions: {int(rec['normalized_label'].isna().sum()):,} "
                         f"rows without a label")
    rec["normalized_label"] = rec["normalized_label"].astype(str).str.strip().str.lower()
    rec["diagnosis_date"] = to_date(rec["diagnosis_date"])
    check_unique(rec, ["person_id", "normalized_label"], "AID conditions")
    keep = rec["person_id"].isin(population["person_id"])
    count_line("AID condition rows of patients outside the analysis population (not used)",
               (~keep).sum(), len(rec))
    rec = rec[keep].copy()
    status = population.set_index("person_id")["aid_status"]
    if (rec["person_id"].map(status) == 0).any():
        raise ValueError("AID conditions: a condition row of an AID- patient")
    pos = population.loc[population["aid_status"] == 1, "person_id"]
    if (~pos.isin(rec["person_id"])).any():
        raise ValueError(f"AID conditions: {int((~pos.isin(rec['person_id'])).sum()):,} "
                         f"AID+ population patients without a condition row")
    count_line("AID condition rows with no diagnosis date", rec["diagnosis_date"].isna().sum(),
               len(rec))
    return rec


def load_dlb_diagnosis_flags(path, population):
    """The coded-diagnosis flags and their first dates, joined to the population.
    Raises on a population patient without a row, a flag other than 0/1, or a
    first date where the flag is 0."""
    rec = read_parquet_columns(path, DLB_DIAGNOSIS_FLAG_COLUMNS, "coded-diagnosis flags")
    rec["person_id"] = rec["person_id"].astype("Int64")
    check_unique(rec, ["person_id"], "coded-diagnosis flags")
    for x in ("ae", "mci", "pdd"):
        check_flag(rec, f"{x}_any", "coded-diagnosis flags")
        rec[f"{x}_first_date"] = to_date(rec[f"{x}_first_date"])
        if (rec[f"{x}_first_date"].notna() & (rec[f"{x}_any"] == 0)).any():
            raise ValueError(f"coded-diagnosis flags: a {x} first date where the {x} flag is 0")
    out = population.merge(rec, on="person_id", how="left", validate="one_to_one")
    if out["ae_any"].isna().any():
        raise ValueError(f"coded-diagnosis flags: {int(out['ae_any'].isna().sum()):,} population "
                         f"patients without a row")
    for x in ("ae", "mci", "pdd"):
        out[f"{x}_any"] = out[f"{x}_any"].astype(int)
    return out


def load_dlb_discriminant_coefficients(path):
    """The discriminant coefficients written by dlb_core_analysis.py (run
    first): the multivariable coefficient of each tested feature (flag
    column name); NaN where the discriminant was not fitted. Raises on a
    repeated feature."""
    rec = read_parquet_columns(path, DISCRIMINANT_COEFFICIENT_COLUMNS,
                               "discriminant coefficients")
    check_unique(rec, ["feature"], "discriminant coefficients")
    return dict(zip(rec["feature"], rec["coefficient"].astype(float)))


# =============================================================================
# INDIRECT STANDARDIZATION (standardized incidence ratios and their ratio)
# =============================================================================
# Strata per sex, from the AID- patients of that sex: quintiles of the
# pre-diagnosis documentation span x quintiles of the pre-diagnosis note
# count x AD-age bands. The outer quintile edges are open; the bands are
# right-closed: <= 65, 65-75, 75-85, > 85 years.
STRATA_QUANTILES = 5
STRATA_AGE_BANDS = [0, 65, 75, 85, 200]


def standardization_strata(pop):
    """Adds `stratum` (sex-prefixed cell) and `cell` (the same cell without
    the sex) to pop; returns (pop, diagnostics: per sex the number of cells,
    AID- reference n, AID+ n, AID+ patients in a cell without AID- support)."""
    pop = pop.copy()
    pop["stratum"] = pd.Series(["unassigned"] * len(pop), index=pop.index, dtype=object)
    pop["cell"] = pd.Series(["unassigned"] * len(pop), index=pop.index, dtype=object)
    diag = []
    for s, prefix in ((1, "F"), (0, "M")):
        m = pop["female"] == s
        ref = pop[m & (pop["aid_status"] == 0)]
        if not m.any():
            continue
        if len(ref) == 0:
            raise ValueError(f"standardization_strata: no AID- {SEX_LABELS[s].lower()} "
                             f"patients")
        span = pd.qcut(ref["predx_documentation_span_yrs"], STRATA_QUANTILES, retbins=True,
                       duplicates="drop")[1]
        notes_ = pd.qcut(ref["note_count_predx"], STRATA_QUANTILES, retbins=True,
                         duplicates="drop")[1]
        span[0], span[-1] = -np.inf, np.inf
        notes_[0], notes_[-1] = -np.inf, np.inf
        idx = pop.index[m]
        o = pd.cut(pop.loc[idx, "predx_documentation_span_yrs"], bins=span, labels=False,
                   include_lowest=True).fillna(-1).astype(int)
        n = pd.cut(pop.loc[idx, "note_count_predx"], bins=notes_, labels=False,
                   include_lowest=True).fillna(-1).astype(int)
        a = pd.cut(pop.loc[idx, "age_at_dx"], bins=STRATA_AGE_BANDS, labels=False,
                   include_lowest=True).fillna(-1).astype(int)
        cell = "o" + o.astype(str) + "_n" + n.astype(str) + "_a" + a.astype(str)
        pop.loc[idx, "cell"] = np.asarray(cell.values, dtype=object)
        pop.loc[idx, "stratum"] = np.asarray((prefix + "_" + cell).values, dtype=object)
        ref_cells = set(pop.loc[m & (pop["aid_status"] == 0), "stratum"])
        pos = pop[m & (pop["aid_status"] == 1)]
        diag.append({"sex": SEX_LABELS[s].lower(), "n_cells": int(pop.loc[idx, "stratum"].nunique()),
                     "n_reference": len(ref), "n_aid_pos": len(pos),
                     "n_aid_pos_without_reference": int((~pos["stratum"].isin(ref_cells)).sum())})
    return pop, pd.DataFrame(diag)


def expected_counts(pop, outcome, reference=None):
    """Per patient: E = the AID- same-sex rate of the outcome in the patient's
    stratum (the mean of the outcome; for a count outcome the mean count); a
    stratum without AID- support takes the overall AID- same-sex rate.
    reference: a 0/1 Series restricting the AID- reference (e.g. coded DLB
    excluded); default every AID- patient. Returns (E, fallback) Series."""
    e = pd.Series(np.nan, index=pop.index)
    fb = pd.Series(False, index=pop.index)
    ref_ok = pd.Series(True, index=pop.index) if reference is None else reference.astype(bool)
    for s in (1, 0):
        m = pop["female"] == s
        ref = pop[m & (pop["aid_status"] == 0) & ref_ok]
        if len(ref) == 0:
            continue
        g = ref.groupby("stratum")[outcome].agg(["sum", "count"])
        rates = g["sum"] / g["count"]
        overall = ref[outcome].mean()
        r = pop.loc[m, "stratum"].map(rates)
        fb.loc[m] = r.isna()
        e.loc[m] = r.fillna(overall).astype(float)
    return e, fb


def garwood_ci(o):
    """Exact (Garwood) 95% CI of a Poisson count o."""
    from scipy.stats import chi2
    o = float(o)
    lo = chi2.ppf(0.025, 2 * o) / 2 if o > 0 else 0.0
    hi = chi2.ppf(0.975, 2 * (o + 1)) / 2
    return lo, hi


def poisson_exact_p(o, e):
    """Two-sided exact Poisson P of O against Poisson(E): twice the smaller
    tail, capped at 1. o is rounded down to an integer for the tails."""
    from scipy.stats import poisson
    if not (np.isfinite(o) and np.isfinite(e) and e > 0 and o >= 0):
        return float("nan")
    k = int(o)
    return float(min(1.0, 2 * min(poisson.sf(k - 1, e), poisson.cdf(k, e))))


def sir_values(o, e):
    """SIR = O / E with its Garwood CI and two-sided exact Poisson P (NaN
    when E <= 0)."""
    if not (np.isfinite(e) and e > 0):
        return {"sir": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "p": float("nan")}
    lo, hi = garwood_ci(o)
    return {"sir": o / e, "ci_low": lo / e, "ci_high": hi / e, "p": poisson_exact_p(o, e)}


def log_rate_ratio_hc0(y, e, group):
    """Offset Poisson regression of y on a 0/1 group term (log E offset) with
    the HC0 sandwich variance, in closed form: the coefficient is
    log[(O1/E1) / (O0/E0)] exactly, and its variance is V1 + V0 with
    Vg = sum over group g of (y - E x Og/Eg)^2 / Og^2. Returns (coef, se), NaN
    when a group has no event or no expected count."""
    y, e, g = (np.asarray(v, dtype=float) for v in (y, e, group))
    out = []
    for k in (1.0, 0.0):
        yy, ee = y[g == k], e[g == k]
        o, ex = yy.sum(), ee.sum()
        if not (o > 0 and ex > 0):
            return float("nan"), float("nan")
        mu = ee * (o / ex)
        out.append((np.log(o / ex), ((yy - mu) ** 2).sum() / o ** 2))
    return float(out[0][0] - out[1][0]), float(np.sqrt(out[0][1] + out[1][1]))


def log_ratio_of_rate_ratios_hc0(y, e, female, group):
    """The ratio of two disjoint groups' female / male rate ratios (offset
    Poisson with group x female, HC0): the saturated four-cell model in
    closed form. Returns (coef, se)."""
    y, e, f, g = (np.asarray(v, dtype=float) for v in (y, e, female, group))
    b1, s1 = log_rate_ratio_hc0(y[g == 1], e[g == 1], f[g == 1])
    b0, s0 = log_rate_ratio_hc0(y[g == 0], e[g == 0], f[g == 0])
    return b1 - b0, float(np.sqrt(s1 ** 2 + s0 ** 2))


def rate_ratio_floor_reason(n_1, n_0, o_1, o_0, labels=("female", "male")):
    """Why a ratio of two standardized ratios (female / male, or two disjoint
    groups) is not estimated, or "": the floor of standardized ratios,
    >= MIN_GROUP_N patients and >= MIN_EVENTS_PER_TERM observed events in each
    of the two groups (the one-term offset model: 10 events per term in the rarer
    group). A single SIR uses group_minimum_reason (>= MIN_GROUP_N patients)."""
    bad = [lbl for n, lbl in zip((n_1, n_0), labels) if n < MIN_GROUP_N]
    if bad:
        return f"fewer than {MIN_GROUP_N} patients: " + ", ".join(bad)
    bad = [lbl for o, lbl in zip((o_1, o_0), labels) if o < MIN_EVENTS_PER_TERM]
    if bad:
        return f"fewer than {MIN_EVENTS_PER_TERM} observed events: " + ", ".join(bad)
    return ""


def sir_ror_row(female, y, e):
    """Female and male SIRs and their ratio (RoR, female / male) for one set
    of carriers. female, y (observed outcome) and e (expected, from
    expected_counts) are aligned arrays of the carriers. A SIR is estimated
    with >= MIN_GROUP_N carriers of that sex and E > 0 (group_minimum_reason);
    the RoR under rate_ratio_floor_reason, by the offset Poisson model with
    HC0 variance (log_rate_ratio_hc0); crit = norm.ppf(0.975)."""
    from scipy.stats import norm
    f = np.asarray(female, dtype=int)
    y = np.asarray(y, dtype=float)
    e = np.asarray(e, dtype=float)
    out = {}
    for k, s in ((1, "female"), (0, "male")):
        m = f == k
        n, o, ex = int(m.sum()), float(y[m].sum()), float(e[m].sum())
        why = group_minimum_reason(n) if n else "no carriers"
        if not why and not ex > 0:
            why = "no expected events"
        v = sir_values(o, ex) if not why else {"sir": float("nan"), "ci_low": float("nan"),
                                               "ci_high": float("nan"), "p": float("nan")}
        out.update({f"n_{s}": n, f"o_{s}": o, f"e_{s}": ex,
                    f"n_zero_expected_{s}": int((e[m] <= 0).sum()),
                    f"status_sir_{s}": "not_estimated" if why else "estimated",
                    f"reason_sir_{s}": why, f"sir_{s}": v["sir"],
                    f"sir_{s}_ci_low": v["ci_low"], f"sir_{s}_ci_high": v["ci_high"],
                    f"p_sir_{s}": v["p"]})
    why = rate_ratio_floor_reason(out["n_female"], out["n_male"], out["o_female"], out["o_male"])
    coef = se = float("nan")
    if not why:
        coef, se = log_rate_ratio_hc0(y, e, f)
        if not (np.isfinite(coef) and np.isfinite(se) and se > 0):
            why = "no expected events in a sex"
    crit = float(norm.ppf(0.975))
    if why:
        out.update(status_ror="not_estimated", reason_ror=why, ror_coef=float("nan"),
                   ror_se=float("nan"), crit=float("nan"), ror=float("nan"),
                   ror_ci_low=float("nan"), ror_ci_high=float("nan"), p_ror=float("nan"))
    else:
        out.update(status_ror="estimated", reason_ror="", ror_coef=coef, ror_se=se, crit=crit,
                   ror=float(np.exp(coef)), ror_ci_low=float(np.exp(coef - crit * se)),
                   ror_ci_high=float(np.exp(coef + crit * se)),
                   p_ror=float(2 * norm.sf(abs(coef / se))))
    return out


# =============================================================================
# AID CONDITION CONCEPTS: THE SHARED PATIENT FRAME
# =============================================================================
AID_IBD_UNIT = "ibd_any"
AID_IBD_MEMBERS = ["ulcerative colitis", "crohns disease", "inflammatory bowel disease"]


def aid_condition_population(patients_path, notes_path, mentions_path, conditions_path):
    """The patient frame of the AID condition concepts: the analysis population
    (sorted by person_id) with coded_dlb / coded_pd, the DLB flags
    (dlb_patient_flags), one cond__<unit> column per condition unit (each
    coded condition; ulcerative colitis, Crohn's disease and inflammatory
    bowel disease as the one unit "ibd_any"), n_aids (the number of distinct
    coded conditions; isolated carriers have 1), the standardization strata,
    and the core outcomes: core (parkinsonism (McKeith) + fluctuating
    cognition + visual hallucinations, 0-3), core_rbd (+ RBD), core_kplus.
    Returns (pop, conditions, units, ceiling dict, strata diagnostics)."""
    pop, notes = load_population(patients_path, notes_path,
                                 extra_columns=("coded_dementia_with_lewy_bodies",
                                                "coded_parkinsons_disease"))
    pop = pop.sort_values("person_id", kind="mergesort").reset_index(drop=True)
    pop["coded_dlb"] = pop["coded_dementia_with_lewy_bodies"]
    pop["coded_pd"] = pop["coded_parkinsons_disease"]
    mentions = load_dlb_feature_mentions(mentions_path, pop, notes)
    pop, ceil = dlb_patient_flags(pop, mentions)
    cond = load_aid_conditions(conditions_path, pop)
    cond["unit"] = cond["normalized_label"].where(~cond["normalized_label"].isin(AID_IBD_MEMBERS),
                                                  AID_IBD_UNIT)
    pop["n_aids"] = pop["person_id"].map(cond.groupby("person_id")["normalized_label"]
                                         .nunique()).fillna(0).astype(int)
    units = sorted(cond["unit"].unique())
    for u in units:
        pop["cond__" + u] = pop["person_id"].isin(cond.loc[cond["unit"] == u, "person_id"]) \
            .astype(int)
    print(f"  Coded conditions: {cond['normalized_label'].nunique():,} labels; analysis units "
          f"(IBD as one): {len(units):,}")
    count_line("AID+ patients with more than one coded condition",
               int((pop["n_aids"] > 1).sum()), int(pop["aid_status"].sum()))
    pop, strata_diag = standardization_strata(pop)
    for _, r in strata_diag.iterrows():
        print(f"  Strata, {r['sex']}: {r['n_cells']} cells; AID- reference n = "
              f"{r['n_reference']:,}; AID+ in a cell without AID- support: "
              f"{r['n_aid_pos_without_reference']:,} of {r['n_aid_pos']:,}")
    pop["core"] = pop[[h for h, _, _ in DLB_CORE_BURDEN]].sum(axis=1)
    pop["core_rbd"] = pop["core"] + pop["has_rbd_any"]
    for k in (1, 2, 3):
        pop[f"core_{k}plus"] = (pop["core"] >= k).astype(int)
    for k in (1, 2, 3, 4):
        pop[f"core_rbd_{k}plus"] = (pop["core_rbd"] >= k).astype(int)
    return pop, cond, units, ceil, strata_diag
