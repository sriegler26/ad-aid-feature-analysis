"""
dlb_feature_timing_tables.py
============================
Renders the DLB feature timing table (.docx) from the source-data workbook
written by dlb_feature_timing.py. Fits no model and computes no statistic:
every number comes from the workbook, formatted once by render_utils.

One table, two panels, female patients:
- Panel A, timing among the patients with each feature: per feature (a group
  row), the AID+ and AID- rows (patients in the group, patients with the
  feature, median (IQR) age at first documentation and years from the AD
  diagnosis), and a row with the AID+ minus AID- differences in the two
  medians (95% CI) and their P values.
- Panel B, the first-documentation rate ratio AID+ / AID- (95% CI) and P
  before and from 5 years after the AD diagnosis, and the ratio of the two
  rate ratios (95% CI) and P, one row per feature.
Features alphabetical. A dash marks an estimate not made, with the
reason in the source data. Years from the diagnosis can be negative, so their
IQR and every CI of a difference read "low to high", as for coefficients.

Captions: when a caption file is given, the wording comes from it; this
script fills in the numbers.

Write safety: every output is a new file under the output root (write_guard.py,
via render_utils); the script stops rather than replace an existing file.

Run from the data folder, after dlb_feature_timing.py.
"""

import os

import pandas as pd

import render_utils as ru

# =============================================================================
# PATHS (relative to the data folder the script is run from)
# =============================================================================
SOURCE_DATA_DIR = ""  # folder of the source-data workbook (.xlsx) written by dlb_feature_timing.py
TABLES_DIR = ""  # folder for the .docx table written here (new files only)
CAPTIONS_PATH = ""  # optional JSON with the title and caption; blank = bare table

TABLE_FILE = "table_dlb_feature_timing"
M = ru.MINUS
GROUP_ROWS = [("aid_pos", "AID+"), ("aid_neg", f"AID{M}")]
WINDOWS = ["<5", ">=5"]
PANEL_A = "A. Age and timing of first documentation among the patients with each feature"
PANEL_B = (f"B. Rate of first documentation per person-year observed, AID+ versus AID{M}, "
           f"before 5 years (<5) and from 5 years (≥5) after the AD diagnosis")
PANEL_B_HEADERS = ["*Feature*", "*<5 years: rate ratio (95% CI)*", "*P*",
                   "*≥5 years: rate ratio (95% CI)*", "*P*", "*Ratio, ≥5 / <5 (95% CI)*", "*P*"]


def read(name):
    path = os.path.join(SOURCE_DATA_DIR, name + ".xlsx")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path}: run dlb_feature_timing.py first")
    return pd.read_excel(path, sheet_name=None)


def params(wb):
    return dict(zip(wb["parameters"]["parameter"], wb["parameters"]["value"]))


def num_ci(est, lo, hi, nd=1):
    """An estimate in years with its CI, "low to high" (bounds can be negative)."""
    if ru._missing(est):
        return ru.DASH
    return f"{ru.fmt_num(est, nd)} ({ru.fmt_num(lo, nd)} to {ru.fmt_num(hi, nd)})"


def years_iqr(med, q1, q3):
    """Median (Q1 to Q3) of years from the diagnosis, 1 decimal."""
    if ru._missing(med) or ru._missing(q1) or ru._missing(q3):
        return ru.DASH
    return f"{ru.fmt_num(med, 1)} ({ru.fmt_num(q1, 1)} to {ru.fmt_num(q3, 1)})"


def one(df, what, **keys):
    m = pd.Series(True, index=df.index)
    for k, v in keys.items():
        m &= df[k] == v
    if m.sum() != 1:
        raise RuntimeError(f"{what} {keys}: {int(m.sum())} source rows")
    return df[m].iloc[0]


def build(cap):
    wb = read(TABLE_FILE)
    tm, df, win, rc, p = (wb["timing"], wb["differences"], wb["window_rates"],
                          wb["rate_ratio_change"], params(wb))
    lab = dict(zip(wb["features"]["feature"], wb["features"]["row_label"]))
    order = sorted(lab, key=lambda f: ru.sort_key(lab[f]))
    rows = [{"kind": "header", "label": PANEL_A}]
    for f in order:
        rows.append({"kind": "header", "label": lab[f]})
        for g, glabel in GROUP_ROWS:
            r = one(tm, "timing", feature=f, group=g)
            est = r["status"] == "estimated"
            rows.append({"kind": "data", "values": [
                glabel, ru.fmt_count(r["n_group"]), ru.fmt_count(r["n"]),
                ru.fmt_median_iqr(r["med_age"], r["q1_age"], r["q3_age"]) if est else ru.DASH, "",
                years_iqr(r["med_yrs_from_dx"], r["q1_yrs_from_dx"], r["q3_yrs_from_dx"])
                if est else ru.DASH, ""]})
        d = one(df, "differences", feature=f)
        est = d["status"] == "estimated"
        rows.append({"kind": "data", "values": [
            f"Difference, AID+ {M} AID{M} (95% CI)", "", "",
            num_ci(d["diff_med_age"], d["diff_med_age_ci_low"], d["diff_med_age_ci_high"])
            if est else ru.DASH,
            ru.fmt_p(d["p_diff_med_age"]) if est else ru.DASH,
            num_ci(d["diff_med_yrs_from_dx"], d["diff_med_yrs_from_dx_ci_low"],
                   d["diff_med_yrs_from_dx_ci_high"]) if est else ru.DASH,
            ru.fmt_p(d["p_diff_med_yrs_from_dx"]) if est else ru.DASH]})
    rows.append({"kind": "header", "label": PANEL_B})
    rows.append({"kind": "data", "values": PANEL_B_HEADERS})
    for f in order:
        vals = [lab[f]]
        for w in WINDOWS:
            r = one(win, "window_rates", feature=f, window_yrs_from_dx=w)
            est = r["status"] == "estimated"
            vals += [ru.fmt_ratio_ci(r["rate_ratio"], r["ci_low"], r["ci_high"]) if est
                     else ru.DASH, ru.fmt_p(r["p"]) if est else ru.DASH]
        r = one(rc, "rate_ratio_change", feature=f)
        est = r["status"] == "estimated"
        vals += [ru.fmt_ratio_ci(r["ratio_of_rate_ratios"], r["ci_low"], r["ci_high"]) if est
                 else ru.DASH, ru.fmt_p(r["p"]) if est else ru.DASH]
        rows.append({"kind": "data", "values": vals})
    headers = ["Feature and group", "Patients in the group, *n*", "With the feature, *n*",
               "Age at first documentation, median (IQR), years", "*P*",
               "Years from the AD diagnosis to first documentation, median (IQR)", "*P*"]
    caps = ()
    if cap:
        t = cap["table"]
        caps = [ru.fill(t["body"], {"n_female": ru.fmt_count(int(p["n_female"]))}),
                ru.fill(t["features"], {"ceiling_female": ru.fmt_num(float(p["ceiling_female"]), 2)}),
                ru.fill(t["parkinsonism"], {"n_park_before_window":
                                            ru.fmt_count(int(p["n_park_before_window"]))}),
                t["weights"], t["panel_a"], t["difference"], t["panel_b"],
                ru.fill(t["bootstrap"], {"n_boot": ru.fmt_count(int(p["n_boot"]))}),
                t["multiplicity"], t["dash"], t["abbreviations"]]
    ru.write_table(os.path.join(TABLES_DIR, TABLE_FILE + ".docx"), headers, rows,
                   cap["title"] if cap else None, caps, first_width=2.3, pad=4)


def main():
    cap = ru.load_captions(CAPTIONS_PATH, __file__)
    print(f"  Title and caption: {'written' if cap else 'none (no caption file set)'}")
    build(cap)


if __name__ == "__main__":
    main()
