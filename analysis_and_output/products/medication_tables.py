"""
medication_tables.py
====================
Renders the medication tables (.docx) from the source-data workbooks written
by medication_analysis.py. Fits no model and computes no statistic: every
number comes from the workbooks, formatted once by render_utils.

Tables
------
- Medication exposure and symptom prevalence: per medication class, domains
  and categories; the interaction odds ratio (95% CI) of AID status and
  exposure, Q, then the patients with the outcome documented in the four
  AID x exposure groups.
- Medication exposure and age at first documentation: the interaction in
  years, per domain and category (laid out as the prevalence table).
- Medication exposure and symptom sequence: per class and expected-order
  pair, within exposed and within unexposed patients the share inverted in
  AID+ and in AID- and the AID+ vs AID- odds ratio (exact 95% CI), then the
  ratio of the two odds ratios and Q (exact test of the interaction).
Results come first, then counts; every CI is inline. One rule for every row
of the three interaction tables: in a section (prevalence and timing: class
x level; sequence: class) with at least one tested row, every row is shown; a
tested row shows its values, an untested row (Other included: it is never
tested) shows "–" for the estimate and Q, with its counts. A section with no
tested row is one header and no rows ("Antimalarials: not tested in any
domain or category"; "... in any category" / "... in any domain" when only
one level of a class is untested; sequence "... for any
pair"). Tested: a fitted model of a tested or positive-control row
(sequence: a tested interaction). Classes with tested rows first, by their
number of tested rows (most first, ties alphabetical), then the rest
alphabetical; categories before domains; rows alphabetical, Other last. The positive control keeps its place, marked; it is not in the FDR
family, so its Q cell reads "NA (P ...)" with its unadjusted P value.

Captions: the titles and caption wording are read from an optional caption
file (CAPTIONS_PATH); this script fills in the numbers and picks the
sentences that apply. With no file set, the tables are written without them.

Run from the data folder, after medication_analysis.py.
"""

import os

import pandas as pd

import render_utils as ru

# =============================================================================
# PATHS (relative to the data folder the script is run from)
# =============================================================================
SOURCE_DATA_DIR = ""  # folder of the source-data workbooks (.xlsx) written by medication_analysis.py
TABLES_DIR = ""  # folder for the .docx tables written here
CAPTIONS_PATH = ""  # optional JSON with titles and captions; blank = bare tables

PREVALENCE_FILE = "table_medication_prevalence"
TIMING_FILE = "table_medication_timing"
SEQUENCE_FILE = "table_medication_sequence"

M = ru.MINUS
MARK = "†"
CLASSES = ["on_cs", "on_dmard", "on_antimalarial", "on_biologic"]
CLASS_LABELS = {"on_cs": "Corticosteroids", "on_dmard": "Synthetic DMARDs and immunosuppressants",
                "on_antimalarial": "Antimalarials", "on_biologic": "Biologic DMARDs"}
CELLS = ["aid_pos_exposed", "aid_pos_unexposed", "aid_neg_exposed", "aid_neg_unexposed"]
CELL_HEADERS = ["AID+ exposed", "AID+ unexposed", f"AID{M} exposed", f"AID{M} unexposed"]
STRATA = [("exposed", "Exposed"), ("unexposed", "Unexposed")]
NOT_ESTIMABLE = "not estimable"


def workbook(name):
    path = os.path.join(SOURCE_DATA_DIR, name + ".xlsx")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path}: run medication_analysis.py first")
    return pd.read_excel(path, sheet_name=None)


def parameter(wb, key):
    p = wb["parameters"].set_index("parameter")["value"]
    if key not in p.index:
        raise KeyError(f"parameters: {key} missing")
    return p[key]


def class_order():
    return sorted(CLASSES, key=lambda c: ru.sort_key(CLASS_LABELS[c]))


def tested_class_order(df, tested):
    """Classes by number of tested rows (tested: a boolean Series on df), most
    first; ties, and the classes with none, alphabetical."""
    n = df[tested].groupby("drug_class").size()
    return sorted(CLASSES, key=lambda c: (-int(n.get(c, 0)), ru.sort_key(CLASS_LABELS[c])))


def tested_rows(df):
    """Prevalence and timing: a fitted model of a tested or positive-control
    row (Other, fdr_role not_tested, is never tested)."""
    return (df["status"] == "fitted") & (df["fdr_role"] != "not_tested")


def class_text(cap, c):
    return dict(cap["common"]["class_names"])[c]


# =============================================================================
# CELLS
# =============================================================================
def label(r):
    return r["row_label"] + (MARK if r["fdr_role"] == "positive_control" else "")


def q_cell(r):
    """Q of a tested row; "NA (P ...)" with the unadjusted P for the positive
    control."""
    if r["fdr_role"] == "positive_control":
        return f"NA ({ru.p_phrase('P', r['p'])})"
    return ru.fmt_p(r["q"])


def interaction_rows(df, tested, ratio, counts):
    """Every row of a section, alphabetical (Other last): a tested row with its
    estimate and Q, an untested row with "–" for both; counts for every row."""
    rows = []
    for i, r in sorted(df.iterrows(), key=lambda x: ru.sort_key(x[1]["row_label"])):
        if tested[i]:
            est = (ru.fmt_ratio_ci(r["odds_ratio"], r["ci_low"], r["ci_high"]) if ratio
                   else ru.fmt_coef_ci(r["coef"], r["ci_low"], r["ci_high"]))
            vals = [est, q_cell(r)]
        else:
            vals = [ru.DASH, ru.DASH]
        rows.append({"kind": "data", "values": [label(r)] + vals + counts(r)})
    return rows


LEVELS = [("category", "categories", "category"), ("domain", "domains", "domain")]


def level_sections(df, tested, ratio, counts):
    """Prevalence and timing: per class (tested classes first), a categories
    section, then a domains section; a class or level with no tested row is
    one header and no rows."""
    rows = []
    for c in tested_class_order(df, tested):
        part = df["drug_class"] == c
        if not tested[part].any():
            rows.append({"kind": "header",
                         "label": f"{CLASS_LABELS[c]}: not tested in any domain or category"})
            continue
        for level, lbl, one in LEVELS:
            sec = part & (df["level"] == level)
            if not tested[sec].any():
                rows.append({"kind": "header",
                             "label": f"{CLASS_LABELS[c]}: not tested in any {one}"})
                continue
            rows.append({"kind": "header", "label": f"{CLASS_LABELS[c]}: {lbl}"})
            rows += interaction_rows(df[sec], tested, ratio, counts)
    return rows


def share_cell(k, n):
    """"x.x% (k/n)"; "–" for an empty group."""
    if pd.isna(n) or int(n) == 0:
        return ru.DASH
    return f"{ru.fmt_pct(100 * float(k) / float(n))} ({ru.fmt_count_of(k, n)})"


def stratum_cells(r, s):
    """AID+ and AID- inverted shares and the AID+ vs AID- OR of one stratum."""
    if r[f"status_{s}"] != "tested":
        orv = ru.DASH
    elif int(r[f"or_estimable_{s}"]):
        orv = ru.fmt_ratio_ci(r[f"odds_ratio_{s}"], r[f"ci_low_{s}"], r[f"ci_high_{s}"])
    else:
        orv = NOT_ESTIMABLE
    return [share_cell(r[f"inverted_aid_pos_{s}"], r[f"n_aid_pos_{s}"]),
            share_cell(r[f"inverted_aid_neg_{s}"], r[f"n_aid_neg_{s}"]), orv]


def modification_rows(df):
    rows = []
    for _, r in sorted(df.iterrows(), key=lambda x: ru.sort_key(x[1]["row_label"])):
        if r["status"] != "tested":
            res = [ru.DASH, ru.DASH]
        else:
            res = [ru.fmt_ratio(r["or_ratio"]) if int(r["ratio_estimable"]) else NOT_ESTIMABLE,
                   ru.fmt_p(r["q"])]
        rows.append({"kind": "data", "values": [r["row_label"]] + stratum_cells(r, "exposed")
                     + stratum_cells(r, "unexposed") + res})
    return rows


# =============================================================================
# CAPTION PIECES
# =============================================================================
def group_sizes_text(cap, wb, order=None):
    parts = []
    for c in order or class_order():
        sizes = {lbl: ru.fmt_count(parameter(wb, f"n_{lbl}_{c}")) for lbl in CELLS}
        parts.append(ru.fill(cap["common"]["group_sizes_item"],
                             {"class": class_text(cap, c), **sizes}))
    return ru.fill(cap["common"]["group_sizes"], {"items": "; ".join(parts)})


def family_text(cap, wb, key, levels, order=None):
    """levels: [(parameter suffix, unit text)]."""
    parts, total = [], 0
    for c in order or class_order():
        sizes = [int(parameter(wb, f"fdr_family_size_{c}{suffix}")) for suffix, _ in levels]
        total += sum(sizes)
        parts.append(class_text(cap, c) + ", "
                     + " and ".join(f"{n} {unit}" for n, (_, unit) in zip(sizes, levels)))
    if not total:
        return cap[key]["fdr_none"]
    return ru.fill(cap[key]["fdr"], {"families": "; ".join(parts)})


def secondary_sentence(cap, wb, key, fmt):
    sd = wb["secondary_definition"]
    if sd.empty:
        return []
    hits = sd[(sd["became_significant"] == 1) | (sd["lost_significance"] == 1)
              | (sd["change_ge_small"] == 1)]
    if hits.empty:
        return []
    items = []
    for _, r in sorted(hits.iterrows(), key=lambda x: (ru.sort_key(x[1]["row_label"]),
                                                       ru.sort_key(CLASS_LABELS[x[1]["drug_class"]]))):
        crossed = (ru.fill(cap["common"]["secondary_crossed"],
                           {"q_threshold": ru.fmt_p(parameter(wb, "q_threshold"))})
                   if r["became_significant"] or r["lost_significance"] else "")
        items.append(ru.fill(cap[key]["secondary_item"], {
            "domain": r["row_label"], "class": class_text(cap, r["drug_class"]),
            "est_primary": fmt(r["estimate_primary"]), "est_secondary": fmt(r["estimate_secondary"]),
            "q_primary": ru.p_phrase("Q", r["q_primary"]),
            "q_secondary": ru.p_phrase("Q", r["q_secondary"]), "crossed": crossed}))
    return [ru.fill(cap["common"]["secondary_definition"], {"changes": ru.join_and(items)})]


def sizes(wb):
    return {"n_aid_pos": ru.fmt_count(parameter(wb, "n_aid_pos")),
            "n_aid_neg": ru.fmt_count(parameter(wb, "n_aid_neg"))}


# =============================================================================
# TABLES
# =============================================================================
def prevalence_table(cap):
    wb = workbook(PREVALENCE_FILE)
    df = wb["interactions"]
    tested = tested_rows(df)
    counts = lambda r: [ru.fmt_count(r[f"events_{c}"]) for c in CELLS]
    order = tested_class_order(df, tested)
    rows = level_sections(df, tested, True, counts)

    def captions():
        cm, t = cap["common"], cap["prevalence"]
        out = [ru.fill(t["body"], sizes(wb)), group_sizes_text(cap, wb, order),
               t["interaction"], t["model"], t["exposure"], cm["classes"], cm["ceiling"],
               t["documented"], cm["ci"],
               family_text(cap, wb, "prevalence", [("_domain", "domains"),
                                                   ("_category", "categories")], order)]
        if (df.loc[tested, "fdr_role"] == "positive_control").any():
            out.append(t["positive_control"])
        out += secondary_sentence(cap, wb, "prevalence", ru.fmt_ratio)
        if (~tested).any():
            out.append(ru.fill(t["legend"], {
                "min_events": int(parameter(wb, "min_events_per_term")),
                "min_group": int(parameter(wb, "min_group_n"))}))
        return out
    headers = (["Domain or category (domain)", "Interaction, odds ratio\n(95% CI)", "*Q*"]
               + [f"{h}\ndocumented, *n*" for h in CELL_HEADERS])
    write(PREVALENCE_FILE, headers, rows, cap, "prevalence", captions)


def timing_table(cap):
    wb = workbook(TIMING_FILE)
    df = wb["interactions"]
    tested = tested_rows(df)
    counts = lambda r: [ru.fmt_count(r[f"n_{c}"]) for c in CELLS]
    order = tested_class_order(df, tested)
    rows = level_sections(df, tested, False, counts)

    def captions():
        cm, t = cap["common"], cap["timing"]
        out = [ru.fill(t["body"], sizes(wb)), t["interaction"], t["model"], t["exposure"],
               cm["classes"], t["documented"], cm["ci"],
               family_text(cap, wb, "timing", [("", "domains"), ("_category", "categories")],
                           order)]
        controls = df.loc[tested & (df["fdr_role"] == "positive_control"), "level"]
        if (controls == "category").any():
            out.append(t["positive_control_categories"])
        elif len(controls):
            out.append(t["positive_control"])
        out += secondary_sentence(cap, wb, "timing", ru.fmt_coef)
        if (~tested).any():
            out.append(ru.fill(t["legend"], {
                "min_events": int(parameter(wb, "min_events_per_term")),
                "min_group": int(parameter(wb, "min_group_n"))}))
        return out
    headers = (["Domain or category (domain)", "Interaction, years\n(95% CI)", "*Q*"]
               + [f"{h}\ndocumented, *n*" for h in CELL_HEADERS])
    write(TIMING_FILE, headers, rows, cap, "timing", captions)


def sequence_table(cap):
    wb = workbook(SEQUENCE_FILE)
    df = wb["pairs"]
    tested = df["status"] == "tested"
    rows = []
    order = tested_class_order(df, tested)
    for c in order:
        part = df["drug_class"] == c
        if not tested[part].any():
            rows.append({"kind": "header", "label": f"{CLASS_LABELS[c]}: not tested for any pair"})
            continue
        rows.append({"kind": "header", "label": CLASS_LABELS[c]})
        rows += modification_rows(df[part])

    def captions():
        cm, t = cap["common"], cap["sequence"]
        out = [ru.fill(t["body"], sizes(wb)), t["question"], t["ratio"], t["exposure"],
               cm["classes"], t["pairs"],
               ru.fill(t["sample"], {"yrs": int(parameter(wb, "predx_span_floor_yrs"))}),
               t["counts"], cm["ci"], family_text(cap, wb, "sequence", [("", "pairs")], order)]
        if (df["status"] != "tested").any() or any(
                (df[f"status_{s}"] != "tested").any() for s, _ in STRATA):
            out.append(ru.fill(t["legend"], {"min_group": int(parameter(wb, "min_group_n"))}))
        if any(((df[f"status_{s}"] == "tested") & (df[f"or_estimable_{s}"] == 0)).any()
               for s, _ in STRATA):
            out.append(t["not_estimable"])
        return out
    headers = (["Expected order"]
               + [h for _, w in STRATA for h in (f"{w}: AID+ inverted,\n% (*n*/*N*)",
                                                 f"{w}: AID{M} inverted,\n% (*n*/*N*)",
                                                 f"{w}: odds ratio\n(95% CI)")]
               + ["Ratio of odds ratios", "*Q*"])
    write(SEQUENCE_FILE, headers, rows, cap, "sequence", captions)


# seven-column tables: a narrower label column and padding keep the inline CIs,
# the Q values and the count headers on one line per word
WIDE = {"first_width": 2.3, "pad": 2}


def write(name, headers, rows, cap, key, captions):
    title = cap[key]["title"] if cap else None
    ru.write_table(os.path.join(TABLES_DIR, name + ".docx"), headers, rows, title,
                   captions() if cap else (), **(WIDE if len(headers) > 5 else {}))


def main():
    cap = ru.load_captions(CAPTIONS_PATH, __file__)
    print(f"  Titles and captions: {'written' if cap else 'none (no caption file set)'}")
    prevalence_table(cap)
    timing_table(cap)
    sequence_table(cap)


if __name__ == "__main__":
    main()