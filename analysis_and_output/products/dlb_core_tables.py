"""
dlb_core_tables.py
==================
Renders the DLB tables (.docx) from the source-data workbooks written by
dlb_core_analysis.py. Fits no model and computes no statistic: every number
comes from the workbooks, formatted once by render_utils.

Tables
------
- DLB features, AID+ vs AID-: per sex and exclusion (female, all patients;
  female, coded DLB excluded; male, all patients), the odds ratio (95% CI),
  Q, then the patients with the feature in each AID group. Every tested
  feature has a row; a feature below the per-group floor shows its odds ratio
  without a Q.
- Sex differences in the DLB feature associations: the AID odds ratio in
  female and in male patients and their ratio (95% CI), Q (features below the
  per-group floor in either sex: ratio, no Q), then the four group counts.
- McKeith features not used: n (%) by AID status and sex, all patients, and
  the fixed reason each is left out (REASON_LABELS; the caption gives the
  reasons that appear, with their sources).
- Discriminant coefficients, coded DLB vs AD only: multivariable and
  bivariable beta (95% CI), odds ratio (95% CI) and Q; a feature below the
  per-group floor is not in the multivariable model and shows its bivariable
  estimate without a Q.
Results come first, then counts; every CI is inline; rows alphabetical.
Captions: when a caption file is given, the wording comes from it; this
script fills in the numbers and picks the sentences that apply.

Run from the data folder, after dlb_core_analysis.py.
"""

import os

import pandas as pd

import render_utils as ru

# =============================================================================
# PATHS (relative to the data folder the script is run from)
# =============================================================================
SOURCE_DATA_DIR = ""  # folder of the source-data workbooks (.xlsx) written by dlb_core_analysis.py
TABLES_DIR = ""  # folder for the .docx tables written here
CAPTIONS_PATH = ""  # optional JSON with titles and captions; blank = bare tables

FEATURES_FILE = "table_dlb_features"
SEX_FILE = "table_dlb_sex_heterogeneity"
NOT_USED_FILE = "table_dlb_features_not_used"
DISCRIMINANT_FILE = "table_dlb_discriminant_coefficients"

M = ru.MINUS
# (sex, arm, block label)
BLOCKS = [("Female", "all", "Women, all patients"),
          ("Female", "dlb_excluded", "Women, coded DLB excluded"),
          ("Male", "all", "Men, all patients")]
GROUPS = ["aid_pos_female", "aid_neg_female", "aid_pos_male", "aid_neg_male"]
GROUP_HEADERS = ["AID+ women", f"AID{M} women", "AID+ men", f"AID{M} men"]
# Reason codes written by dlb_core_analysis.py (NOT_USED), in caption order
REASON_LABELS = {"common_in_ad": "Common in AD",
                 "common_in_dementia": "Common in dementia"}


def workbook(name):
    path = os.path.join(SOURCE_DATA_DIR, name + ".xlsx")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path}: run dlb_core_analysis.py first")
    return pd.read_excel(path, sheet_name=None)


def parameter(wb, key):
    p = wb["parameters"].set_index("parameter")["value"]
    if key not in p.index:
        raise KeyError(f"parameters: {key} missing")
    return p[key]


def ordered(df):
    return sorted(df.iterrows(), key=lambda x: ru.sort_key(x[1]["row_label"]))


def fitted_reasons(df, status_col="status", fitted=("fitted", "tested")):
    return df[~df[status_col].isin(fitted)]


# a narrower label column and padding keep the inline CIs, the Q values and the
# count headers ("documented," set in bold) on one line per word
WIDE = {"first_width": 2.3, "pad": 2}
NINE = {"first_width": 2.0, "pad": 3}   # the nine-column sex table


def write(name, headers, rows, cap, key, captions, extra=None):
    title = cap[key]["title"] if cap else None
    extra = extra or {}
    ru.write_table(os.path.join(TABLES_DIR, name + ".docx"), headers, rows, title,
                   captions() if cap else (), **extra)


# =============================================================================
# DLB FEATURES
# =============================================================================
def features_table(cap):
    wb = workbook(FEATURES_FILE)
    df = wb["features"]
    rows = []
    for sex, arm, lbl in BLOCKS:
        rows.append({"kind": "header", "label": lbl})
        part = df[(df["sex"] == sex) & (df["arm"] == arm)]
        for _, r in ordered(part):
            est = ru.fmt_ratio_ci(r["odds_ratio"], r["ci_low"], r["ci_high"]) \
                if r["status"] == "fitted" else ru.DASH
            q = ru.fmt_p(r["q"]) if r["fdr_role"] == "tested" and r["status"] == "fitted" \
                else ru.DASH
            rows.append({"kind": "data", "values": [
                r["row_label"], est, q, ru.fmt_count(r["events_aid_pos"]),
                ru.fmt_count(r["events_aid_neg"])]})

    def captions():
        t = cap["features"]
        n = {f"{aid}_{sex.lower()}_{arm}": ru.fmt_count(parameter(wb, f"n_{aid}_{sex.lower()}_{arm}"))
             for sex, arm, _ in BLOCKS for aid in ("aid_pos", "aid_neg")}
        out = [ru.fill(t["body"], n), t["model"], cap["common"]["features"], cap["common"]["ceiling"],
               t["documented"], cap["common"]["ci"]]
        fams = [f"{lbl[0].lower() + lbl[1:]}, {int(parameter(wb, f'fdr_family_size_{sex.lower()}_{arm}'))}"
                for sex, arm, lbl in BLOCKS]
        shown = df[[(s, a) in {(b[0], b[1]) for b in BLOCKS} for s, a in zip(df["sex"], df["arm"])]]
        tested = shown[(shown["fdr_role"] == "tested") & (shown["status"] == "fitted")]
        out.append(ru.fill(t["fdr"], {"families": "; ".join(fams)}) if len(tested)
                   else t["fdr_none"])
        if ((shown["fdr_role"] != "tested") & (shown["status"] == "fitted")).any():
            out.append(ru.fill(t["group_floor"], {
                "group_floor": int(parameter(wb, "group_floor"))}))
        not_fit = shown[shown["status"] != "fitted"]
        if len(not_fit):
            out.append(ru.fill(cap["common"]["dash"], {
                "min_events": int(parameter(wb, "min_events_per_term"))}))
        return out
    headers = ["Feature", "Odds ratio (95% CI)", "*Q*", "AID+\ndocumented, *n*",
               f"AID{M}\ndocumented, *n*"]
    write(FEATURES_FILE, headers, rows, cap, "features", captions, WIDE)


# =============================================================================
# SEX
# =============================================================================
def sex_table(cap):
    wb = workbook(SEX_FILE)
    df = wb["sex_heterogeneity"]
    rows = []
    for _, r in ordered(df):
        female = ru.fmt_ratio_ci(r["odds_ratio_female"], r["ci_low_female"], r["ci_high_female"]) \
            if r["status_female"] == "fitted" else ru.DASH
        male = ru.fmt_ratio_ci(r["odds_ratio_male"], r["ci_low_male"], r["ci_high_male"]) \
            if r["status_male"] == "fitted" else ru.DASH
        tested = r["status"] == "tested"
        in_fdr = tested and r["fdr_role"] == "tested"
        rows.append({"kind": "data", "values": [
            r["row_label"], female, male,
            ru.fmt_ratio_ci(r["ratio"], r["ci_low"], r["ci_high"]) if tested else ru.DASH,
            ru.fmt_p(r["q"]) if in_fdr else ru.DASH]
            + [ru.fmt_count(r[f"events_{g}"]) for g in GROUPS]})

    def captions():
        t = cap["sex"]
        n = {g: ru.fmt_count(parameter(wb, f"n_{g}")) for g in GROUPS}
        out = [ru.fill(t["body"], n), t["ratio"], t["model"], cap["common"]["features_short"],
               t["documented"], cap["common"]["ci"]]
        k = int(parameter(wb, "fdr_family_size"))
        out.append(ru.fill(t["fdr"], {"k": k}) if k else t["fdr_none"])
        if ((df["status"] == "tested") & (df["fdr_role"] != "tested")).any():
            out.append(ru.fill(t["group_floor"], {
                "group_floor": int(parameter(wb, "group_floor"))}))
        if (df["status"] != "tested").any():
            out.append(ru.fill(t["dash"], {"min_events": int(parameter(wb, "min_events_per_term")),
                                           "min_group": int(parameter(wb, "min_group_n"))}))
        return out
    headers = (["Feature", "Women, odds ratio\n(95% CI)", "Men, odds ratio\n(95% CI)",
                "Ratio, women / men\n(95% CI)", "*Q*"]
               + [f"{h}\ndocumented, *n*" for h in GROUP_HEADERS])
    write(SEX_FILE, headers, rows, cap, "sex", captions, NINE)


# =============================================================================
# McKEITH FEATURES NOT USED
# =============================================================================
def not_used_table(cap):
    wb = workbook(NOT_USED_FILE)
    df = wb["features"]
    rows = []
    for _, r in ordered(df):
        rows.append({"kind": "data", "values": [r["row_label"]]
                     + [ru.fmt_n_pct(r[f"events_{g}"], r[f"n_{g}"]) for g in GROUPS]
                     + [ru.fmt_n_pct(r["events"], r["n"]), REASON_LABELS[r["reason_not_used"]]]})
    unknown = set(df["reason_not_used"]) - set(REASON_LABELS)
    if unknown:
        raise KeyError(f"{NOT_USED_FILE}: reason codes without a label: {sorted(unknown)}")

    def captions():
        t = cap["not_used"]
        n = {g: ru.fmt_count(parameter(wb, f"n_{g}")) for g in GROUPS}
        n["n_all"] = ru.fmt_count(parameter(wb, "n_aid_pos") + parameter(wb, "n_aid_neg"))
        present = set(df["reason_not_used"])
        reasons = [t["reasons"][k] for k in REASON_LABELS if k in present]
        return [ru.fill(t["body"], n), t["definitions"],
                t["reasons_intro"] + "; ".join(reasons) + ".",
                cap["common"]["ceiling"], t["descriptive"]]
    headers = (["Feature"] + [f"{h},\n*n* (%)" for h in GROUP_HEADERS]
               + ["All patients,\n*n* (%)", "Reason not used"])
    write(NOT_USED_FILE, headers, rows, cap, "not_used", captions, WIDE)


# =============================================================================
# DISCRIMINANT
# =============================================================================
def discriminant_table(cap):
    wb = workbook(DISCRIMINANT_FILE)
    mv = wb["multivariable"].set_index("feature")
    bv = wb["bivariable"].set_index("feature")
    rows = []
    for f, r in sorted(mv.iterrows(), key=lambda x: ru.sort_key(x[1]["row_label"])):
        vals = [r["row_label"]]
        for m in (r, bv.loc[f]):
            ok = m["status"] == "fitted"
            vals += [ru.fmt_coef_ci(m["coef"], m["coef_ci_low"], m["coef_ci_high"]) if ok
                     else ru.DASH,
                     ru.fmt_ratio_ci(m["odds_ratio"], m["ci_low"], m["ci_high"]) if ok else ru.DASH,
                     ru.fmt_p(m["q"]) if ok and m["fdr_role"] == "tested" else ru.DASH]
        rows.append({"kind": "data", "values": vals})

    def captions():
        t = cap["discriminant"]
        fit = wb["fit"].set_index("metric")["value"]
        n = {"n_dlb": ru.fmt_count(fit["n_dlb"]), "n_ad_only": ru.fmt_count(fit["n_ad_only"])}
        gf = int(parameter(wb, "group_floor"))
        out = [ru.fill(t["body"], n),
               ru.fill(t["model"], {"group_floor": gf,
                                    "k": int(parameter(wb, "n_features_in_model"))}),
               cap["common"]["features_short"], t["beta"], cap["common"]["ci"]]
        if "auc_in_sample" in fit.index:
            out.append(ru.fill(t["auc"], {
                "auc": ru.fmt_num(fit["auc_in_sample"], 2), "folds": int(fit["cv_folds"]),
                "cv_auc": ru.fmt_num(fit["cv_auc_mean"], 2),
                "cv_min": ru.fmt_num(fit["cv_auc_min"], 2),
                "cv_max": ru.fmt_num(fit["cv_auc_max"], 2)}))
        km = int(parameter(wb, "fdr_family_size_multivariable"))
        kb = int(parameter(wb, "fdr_family_size_bivariable"))
        out.append(ru.fill(t["fdr"], {"km": km, "kb": kb}) if km or kb else t["fdr_none"])
        if (mv["status"] == "not_in_model").any():
            out.append(t["group_floor"])
        if (mv["status"].isin(["not_fitted", "failed"])).any() \
                or (bv["status"] != "fitted").any():
            out.append(ru.fill(t["dash"], {"min_events": int(parameter(wb, "min_events_per_term"))}))
        return out
    headers = ["Feature", "Multivariable\nβ (95% CI)", "Multivariable\nodds ratio (95% CI)",
               "*Q*", "Bivariable\nβ (95% CI)", "Bivariable\nodds ratio (95% CI)", "*Q*"]
    write(DISCRIMINANT_FILE, headers, rows, cap, "discriminant", captions, WIDE)


def main():
    cap = ru.load_captions(CAPTIONS_PATH, __file__)
    print(f"  Titles and captions: {'written' if cap else 'none (no caption file set)'}")
    features_table(cap)
    sex_table(cap)
    not_used_table(cap)
    discriminant_table(cap)


if __name__ == "__main__":
    main()