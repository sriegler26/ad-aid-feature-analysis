"""
cohort_characteristics_tables.py
================================
Renders the cohort tables (.docx) from the source-data workbooks written by
cohort_characteristics.py. Computes no statistic: every number comes from
the workbooks, formatted once by render_utils.

Tables
------
- Cohort characteristics (file table_1): the analysis population,
  AID+ vs AID-: AID+, AID-, P, SMD.
- Characteristics by sex: the same within female and within male patients, side by side.
- Characteristics of the patients without notes, and the same by sex.
- Lewy body diagnoses: coded DLB, PD and Lewy body disease, AID+ vs AID-
  within female and within male patients, in patients with notes, without notes and
  all patients: odds ratio (95% CI), P, then diagnosed / all n per arm; and
  the ratio of the AID odds ratios with vs without notes.
Characteristic tables keep the conventional baseline-characteristics layout: the two
groups, then P and SMD. Continuous characteristics are median (Q1-Q3),
categorical ones n (%), with the group sizes also in the caption. "-" marks
a test not run; counts are shown for every row. Demographics opens
with the group sizes (n), then sex, race and ethnicity; the other rows are
alphabetical within each section by their readable label; categories
are alphabetical with Other and unknown last; female before male,
AID+ before AID-.

Captions: the titles and caption wording are read from an optional caption
file (CAPTIONS_PATH); this script fills in the numbers and picks the
sentences that apply. With no file set, the tables are written without them.

Run from the data folder, after cohort_characteristics.py.
"""

import os

import pandas as pd

import render_utils as ru

# =============================================================================
# PATHS (relative to the data folder the script is run from)
# =============================================================================
SOURCE_DATA_DIR = ""  # folder with the cohort_characteristics.py workbooks (table_1.xlsx, table_cohort_characteristics_*.xlsx, table_lewy_body_diagnoses.xlsx)
TABLES_DIR = ""  # folder for the .docx tables written here
CAPTIONS_PATH = ""  # optional JSON with titles and captions; blank = bare tables

TABLE_1 = "table_1"
BY_SEX = "table_cohort_characteristics_by_sex"
NO_NOTES = "table_cohort_characteristics_no_notes"
NO_NOTES_BY_SEX = "table_cohort_characteristics_no_notes_by_sex"
LEWY_BODY = "table_lewy_body_diagnoses"

SECTIONS = [("demographics", "Demographics"), ("documentation", "Documentation")]
DEMOGRAPHICS_FIRST = ["female", "first_race", "ethnicity"]   # after the n row; the rest alphabetical
N_LABEL = "*n*"
LAST = ("Other", "Unknown")
INDENT = "    "
# Sex categories as shown in the tables (the workbooks keep Female / Male)
SEX_LABELS = {"Female": "Women", "Male": "Men"}
CAPTIONS = ru.load_captions(CAPTIONS_PATH, __file__)


def read(name):
    return pd.read_excel(os.path.join(SOURCE_DATA_DIR, name + ".xlsx"), sheet_name=None)


def params(sheets):
    return dict(zip(sheets["parameters"]["parameter"], sheets["parameters"]["value"]))


def cap(key):
    return CAPTIONS[key] if CAPTIONS else None


# =============================================================================
# CHARACTERISTIC TABLES
# =============================================================================
def p_cell(r):
    if r["status"] == "tested":
        return ru.fmt_p(r["p"])
    return ru.DASH if r["status"] == "not_tested" else ""


def smd_cell(r):
    if r["row_type"] == "category_header" and r["variable"] != "female":
        return ""
    if r["row_type"] == "category" and r["variable"] == "female":
        return ""                      # the sex SMD is on the header row
    return ru.fmt_smd(r["smd"])


def arm_cell(r, tag):
    if r["row_type"] == "continuous":
        return ru.fmt_median_iqr(r[f"median_{tag}"], r[f"q1_{tag}"], r[f"q3_{tag}"])
    if r["row_type"] in ("binary", "category"):
        return ru.fmt_n_pct(r[f"count_{tag}"], r[f"n_{tag}"])
    return ""


def items(df):
    """Display rows of one comparison: (label, [AID+, AID-, P, SMD]), with
    section header rows (label, None). Demographics: the n row, then sex,
    race and ethnicity, then the other rows alphabetically; other sections
    alphabetical."""
    out = []
    for key, title in SECTIONS:
        sec = df[df["section"] == key]
        if sec.empty:
            continue
        out.append((title, None))
        if key == "demographics":
            out.append((N_LABEL, [ru.fmt_count(sec["n_aid_pos"].iloc[0]),
                                  ru.fmt_count(sec["n_aid_neg"].iloc[0]), "", ""]))
        blocks = []
        for variable, g in sec.groupby("variable", sort=False):
            head = g[g["row_type"] != "category"]
            if head.empty:
                raise RuntimeError(f"{variable}: categories without a header row")
            h = head.iloc[0]
            block = [(h["row_label"], [arm_cell(h, "aid_pos"), arm_cell(h, "aid_neg"),
                                       p_cell(h), smd_cell(h)])]
            cats = g[g["row_type"] == "category"]
            if variable == "female":                     # female before male
                order = {"Female": 0, "Male": 1}
                cat_key = lambda kv: order[kv[1]["row_label"]]
            else:
                cat_key = lambda kv: ru.sort_key(kv[1]["row_label"], LAST)
            for _, c in sorted(cats.iterrows(), key=cat_key):
                shown = SEX_LABELS[c["row_label"]] if variable == "female" else c["row_label"]
                block.append((INDENT + shown,
                              [arm_cell(c, "aid_pos"), arm_cell(c, "aid_neg"), "", smd_cell(c)]))
            blocks.append((variable, block))
        first = DEMOGRAPHICS_FIRST if key == "demographics" else []
        for _, block in sorted(blocks, key=lambda vb: (
                first.index(vb[0]) if vb[0] in first else len(first),
                ru.sort_key(vb[1][0][0]))):
            out += block
    return out


def to_rows(entries):
    return [{"kind": "header", "label": lab} if vals is None
            else {"kind": "data", "values": [lab] + vals} for lab, vals in entries]


def side_by_side(df):
    f = items(df[df["sex"] == "Female"])
    m = items(df[df["sex"] == "Male"])
    if [x[0] for x in f] != [x[0] for x in m]:
        raise RuntimeError("the two sex strata have different rows")
    return [(lf, None) if vf is None else (lf, vf + vm) for (lf, vf), (_, vm) in zip(f, m)]


def legend_reasons(df, p):
    """The reasons a test was not run, as caption phrases (from the caption file)."""
    words = cap("common")["reasons"]
    found = []
    for reason in df.loc[df["status"] == "not_tested", "reason"].dropna().unique():
        if reason.startswith("fewer than") and "compared group" in reason:
            k = "group"
        elif reason.startswith("no patient"):
            k = "none"
        elif reason.startswith("every patient"):
            k = "all"
        elif "categories" in reason:
            k = "categories"
        else:
            raise RuntimeError(f"no caption wording for the reason {reason!r}")
        if k not in found:
            found.append(k)
    return [ru.fill(words[k], {"min_group": p["min_group_n"]}) for k in found]


def small_cells(df, label_of):
    """Tested 2x2 rows with a cell below 5, named with their counts."""
    out = []
    for _, r in df[(df["row_type"] == "binary") & (df["status"] == "tested")].iterrows():
        cells = [r["count_aid_pos"], r["n_aid_pos"] - r["count_aid_pos"],
                 r["count_aid_neg"], r["n_aid_neg"] - r["count_aid_neg"]]
        if min(cells) < 5:
            out.append(f"{label_of(r)} (AID+ {ru.fmt_count_of(r['count_aid_pos'], r['n_aid_pos'])}; "
                       f"AID− {ru.fmt_count_of(r['count_aid_neg'], r['n_aid_neg'])})")
    for _, r in df[(df["row_type"] == "category_header") & (df["variable"] == "female")
                   & (df["status"] == "tested")].iterrows():
        g = df[(df["variable"] == "female") & (df["row_type"] == "category")]
        if "sex" in df.columns:
            g = g[g["sex"] == r["sex"]]
        if min(g["count_aid_pos"].min(), g["count_aid_neg"].min()) < 5:
            out.append(f"{label_of(r)} (AID+ female {ru.fmt_count_of(g.iloc[0]['count_aid_pos'], r['n_aid_pos'])}; "
                       f"AID− female {ru.fmt_count_of(g.iloc[0]['count_aid_neg'], r['n_aid_neg'])})")
    return out


def characteristic_caption(key, df, p, extra_values, share=None):
    if not CAPTIONS:
        return None, ()
    c, common = cap(key), cap("common")
    values = {**{k: ru.fmt_count(v) for k, v in p.items() if k.startswith("n_")},
              "min_group": p["min_group_n"], "min_summary": p["min_summary_n"],
              **extra_values}
    s = [ru.fill(c["body"], values)]
    if share:
        s.append(share)
    s += [common["cells"], ru.fill(common["tests"], {"binary": c["binary"]}),
          ru.fill(common["smd"], {"binary": c["binary"]})]
    if "documentation" in c:
        s.append(c["documentation"])
    k = int(p["first_race_values_not_in_map"]) + int(p["ethnicity_values_not_in_map"])
    s.append(ru.fill(common["race"], {"other": ru.fill(common["race_other"],
                                                        {"k": ru.fmt_count(k)}) if k else ""}))
    if "sex" in df.columns:
        s.append(c["rows"])
    cont = df[df["row_type"] == "continuous"]
    if cont[["median_aid_pos", "median_aid_neg"]].isna().any().any():
        s.append(ru.fill(common["summary_floor"], values))
    reasons = legend_reasons(df, p)
    if reasons:
        s.append(ru.fill(common["legend"], {"reasons": ru.join_and(reasons)}))
    label_of = (lambda r: f"{r['row_label'].lower()}, {'female' if r['sex'] == 'Female' else 'male'}") \
        if "sex" in df.columns else (lambda r: r["row_label"].lower())
    small = small_cells(df, label_of)
    if small:
        s.append(ru.fill(common["small_cells"], {"items": "; ".join(small)}))
    return c["title"], s


def share_sentence(share, stratum_word=""):
    """The share of patients without notes, AID+ vs AID-, for the caption."""
    c = cap("common")
    r = share
    v = {"stratum": stratum_word,
         "e1": ru.fmt_count(r["without_notes_aid_pos"]), "n1": ru.fmt_count(r["n_aid_pos"]),
         "pct1": ru.fmt_pct(100 * r["without_notes_aid_pos"] / r["n_aid_pos"]) if r["n_aid_pos"] else ru.DASH,
         "e2": ru.fmt_count(r["without_notes_aid_neg"]), "n2": ru.fmt_count(r["n_aid_neg"]),
         "pct2": ru.fmt_pct(100 * r["without_notes_aid_neg"] / r["n_aid_neg"]) if r["n_aid_neg"] else ru.DASH,
         "smd": ru.fmt_smd(r["smd"])}
    if r["status"] != "tested":
        return ru.fill(c["share_not_tested"], {**v, "reason": r["reason"]})
    or_ci = "not estimable" if r["or_estimable"] == 0 else ru.fmt_ratio_ci(
        r["odds_ratio"], r["ci_low"], r["ci_high"])
    return ru.fill(c["share"], {**v, "or_ci": or_ci, "p": ru.p_phrase("P", r["p"])})


HEAD = ["Characteristic", "AID+", "AID−", "*P*", "SMD"]
HEAD_SEX = ["Characteristic"] + [f"{s}, {h}" for s in ("Women", "Men")
                                  for h in ("AID+", "AID−", "*P*", "SMD")]


def characteristic_table(name, key, by_sex_, share_rows=None):
    sheets = read(name)
    df, p = sheets["characteristics"], params(sheets)
    entries = side_by_side(df) if by_sex_ else items(df)
    share = None
    if CAPTIONS and share_rows is not None:
        if by_sex_:
            share = cap("common")["share_intro"] + " " + " ".join(
                share_sentence(r, w) for w, r in zip(("Among female patients,", "Among male patients,"),
                                                     share_rows))
        else:
            share = cap("common")["share_intro"] + " " + share_sentence(share_rows[0], "Of all patients,")
    title, caption = characteristic_caption(key, df, p, {}, share)
    ru.write_table(os.path.join(TABLES_DIR, name + ".docx"), HEAD_SEX if by_sex_ else HEAD,
                   to_rows(entries), title, caption,
                   first_width=2.4 if by_sex_ else None, pad=4 if by_sex_ else 0)


# =============================================================================
# LEWY BODY DIAGNOSES
# =============================================================================
COHORT_ORDER = ["with_notes", "all"]


def lewy_body_table():
    sheets = read(LEWY_BODY)
    d, it, p = sheets["diagnoses"], sheets["restriction_ratio"], params(sheets)
    rows, ne, small = [], False, []
    for sx, word in (("Female", "Women"), ("Male", "Men")):
        rows.append({"kind": "header", "label": word})
        dd = d[d["sex"] == sx]
        labels = sorted(dd["row_label"].unique(), key=ru.sort_key)
        for lab in labels:
            g = dd[dd["row_label"] == lab].set_index("cohort").loc[COHORT_ORDER]
            for i, (ck, r) in enumerate(g.iterrows()):
                if r["status"] == "tested":
                    est = "NE" if r["or_estimable"] == 0 else ru.fmt_ratio_ci(
                        r["odds_ratio"], r["ci_low"], r["ci_high"])
                    ne |= r["or_estimable"] == 0
                    pc = ru.fmt_p(r["p"])
                    cells = [r["events_aid_pos"], r["n_aid_pos"] - r["events_aid_pos"],
                             r["events_aid_neg"], r["n_aid_neg"] - r["events_aid_neg"]]
                    if min(cells) < 5:
                        small.append(f"{word.lower()}, {lab}, {r['cohort_label'].lower()} "
                                     f"(AID+ {ru.fmt_count_of(r['events_aid_pos'], r['n_aid_pos'])}; "
                                     f"AID− {ru.fmt_count_of(r['events_aid_neg'], r['n_aid_neg'])})")
                else:
                    est, pc = ru.DASH, ru.DASH
                rows.append({"kind": "data", "values": [
                    lab if i == 0 else "", r["cohort_label"], est, pc,
                    ru.fmt_count_of(r["events_aid_pos"], r["n_aid_pos"]),
                    ru.fmt_count_of(r["events_aid_neg"], r["n_aid_neg"])]})
            ir = it[(it["sex"] == sx) & (it["row_label"] == lab)].iloc[0]
            tested = ir["status"] == "tested"
            rows.append({"kind": "data", "values": [
                "", "Ratio, analysis population / whole cohort",
                ru.fmt_ratio_ci(ir["ratio"], ir["ci_low"], ir["ci_high"]) if tested else ru.DASH,
                ru.fmt_p(ir["p"]) if tested else ru.DASH, "n/a", "n/a"]})
    headers = ["Diagnosis", "Patients", "Odds ratio (95% CI)", "*P*",
               "AID+, diagnosed/all, *n*", "AID−, diagnosed/all, *n*"]
    title, caption = None, ()
    if CAPTIONS:
        c = cap("lewy_body")
        values = {k: ru.fmt_count(v) for k, v in p.items() if k.startswith("n_")}
        values.update(min_group=p["min_group_n"],
                      resamples=ru.fmt_count(p["bootstrap_resamples"]),
                      max_pct=ru.fmt_num(100 * p["bootstrap_max_not_estimable"], 0))
        caption = [ru.fill(c["body"], values), ru.fill(c["model"], values), c["counts"]]
        not_tested = d[d["status"] == "not_tested"]["reason"].tolist()
        reasons = []
        for reason in not_tested:
            k = "group" if "compared group" in reason else ("none" if reason.startswith("no patient")
                                                            else "all")
            if k not in reasons:
                reasons.append(k)
        model_reasons = []
        for reason in it.loc[it["status"] != "tested", "reason"]:
            k = ("cohort" if "one of the two cohorts" in reason
                 else "resamples" if "bootstrap resamples" in reason else None)
            if k is None:
                raise RuntimeError(f"no caption wording for the reason {reason!r}")
            if k not in model_reasons:
                model_reasons.append(k)
        if reasons or model_reasons:
            phrases = [ru.fill(c["reasons"][k], values) for k in reasons] + \
                      [ru.fill(c["model_reasons"][k], values) for k in model_reasons]
            caption.append(ru.fill(c["legend"], {"reasons": ru.join_and(phrases)}))
        caption.append(c["na"])
        if ne:
            caption.append(c["ne"])
        if small:
            caption.append(ru.fill(cap("common")["small_cells"], {"items": "; ".join(small)}))
        title = c["title"]
    ru.write_table(os.path.join(TABLES_DIR, LEWY_BODY + ".docx"), headers, rows, title, caption)


# =============================================================================
# MAIN
# =============================================================================
def main():
    characteristic_table(TABLE_1, "table_1", False)
    characteristic_table(BY_SEX, "by_sex", True)
    share = read(NO_NOTES)["share_without_notes"].set_index("stratum")
    characteristic_table(NO_NOTES, "no_notes", False, [share.loc["all"]])
    characteristic_table(NO_NOTES_BY_SEX, "no_notes_by_sex", True,
                         [share.loc["Female"], share.loc["Male"]])
    lewy_body_table()


if __name__ == "__main__":
    main()