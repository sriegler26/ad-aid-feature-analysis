"""
render_utils.py
===============
Shared code for the table and figure scripts in this folder: the one number
formatter, the row ordering, the Word table writer, and the
caption-file loader. Every table script imports these; none defines its own.

Formatter. Values are rounded once, half-up on the decimal value of the
full-precision workbook number (the shortest decimal that reproduces the
float, so 2.675 -> 2.68). A value that rounds to zero prints without a minus
sign; a negative value gets the Unicode minus (U+2212), inserted only here.
  P and Q   > 0.01: 2 decimals; 0.001 to 0.01: 3 decimals; < 0.001: "<0.001";
            the band is chosen from the unrounded value
  ratios    2 decimals; CI "low–high" (en dash)
  coefs     2 decimals; CI "low to high" (the bounds can be negative)
  CIs in tables are inline after the estimate, in parentheses:
            "r (lo–hi)", "b (lo to hi)" (no CI column)
  SMD       2 decimals
  median    median (Q1–Q3), 1 decimal
  percent   1 decimal; "<0.1%" for nonzero values under 0.05%; 0 -> "0.0%"
  counts    integer with a thousands separator
  missing   "–" (not tested)

Text markup: *word* in a header, cell or caption is set in italics (used for
the statistical symbols P, Q and n).

Captions: titles and captions are written only when a caption file is
given; with a blank path the scripts write the bare tables and figures. A
figure's title and caption go to a .docx beside the figure (write_caption).

Writing: every table, caption and figure file is a new file under the
output root, written through write_guard.py (write_table, write_caption,
save_figure). An existing file is never changed, replaced, moved or deleted;
a script stops if an output already exists.
"""

import json
import os
import re
from decimal import Decimal, ROUND_HALF_UP

DASH = "–"          # en dash: "not tested", and between CI bounds
MINUS = "−"


# =============================================================================
# NUMBER FORMATTER
# =============================================================================
def _missing(x):
    try:
        return x is None or x != x
    except TypeError:
        return False


def _round(x, nd):
    d = Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP)
    return d


def fmt_num(x, nd=2, plus=False):
    if _missing(x):
        return DASH
    x = float(x)
    if x in (float("inf"), float("-inf")):
        raise ValueError("an infinite value reached the formatter; the table script "
                         "must show it as not estimable")
    d = _round(x, nd)
    if d == 0:
        return f"{abs(d):.{nd}f}"
    s = f"{abs(d):.{nd}f}"
    if d < 0:
        return MINUS + s
    return ("+" + s) if plus else s


def fmt_ratio(x):
    return fmt_num(x, 2)


def fmt_coef(x):
    return fmt_num(x, 2)


def fmt_ci(low, high):
    if _missing(low) or _missing(high):
        return DASH
    return f"{fmt_ratio(low)}{DASH}{fmt_ratio(high)}"


def fmt_ci_coef(low, high):
    """CI of a coefficient (years, points): "low to high", since either bound
    can be negative."""
    if _missing(low) or _missing(high):
        return DASH
    return f"{fmt_coef(low)} to {fmt_coef(high)}"


def fmt_ratio_ci(est, low, high):
    """A ratio with its 95% CI inline: "r (lo–hi)" (no separate CI column)."""
    if _missing(est):
        return DASH
    return f"{fmt_ratio(est)} ({fmt_ci(low, high)})"


def fmt_coef_ci(est, low, high):
    """A coefficient with its 95% CI inline: "b (lo to hi)"."""
    if _missing(est):
        return DASH
    return f"{fmt_coef(est)} ({fmt_ci_coef(low, high)})"


def fmt_smd(x):
    """Standardized mean difference, 2 decimals."""
    return fmt_num(x, 2)


def fmt_median_iqr(median, q1, q3):
    """Median (Q1–Q3), 1 decimal: "m (q1–q3)"."""
    if _missing(median) or _missing(q1) or _missing(q3):
        return DASH
    return f"{fmt_num(median, 1)} ({fmt_num(q1, 1)}{DASH}{fmt_num(q3, 1)})"


def fmt_pct(x):
    """x is already a percentage (x -> "x.x%")."""
    if _missing(x):
        return DASH
    if float(x) == 0:
        return "0.0%"
    if abs(float(x)) < 0.05:
        return "<0.1%"
    return fmt_num(x, 1) + "%"


def fmt_pp(x):
    """Signed difference in percentage points, 1 decimal."""
    return fmt_num(x, 1, plus=True)


def fmt_count(n):
    if _missing(n):
        return DASH
    return f"{int(n):,}"


def fmt_count_of(k, n):
    """"k/n" counts of a group whose size differs by row (no percentage)."""
    if _missing(k) or _missing(n):
        return DASH
    return f"{fmt_count(k)}/{fmt_count(n)}"


def fmt_p(p):
    if _missing(p):
        return DASH
    p = float(p)
    if p < 0.001:
        return "<0.001"
    if p <= 0.01:
        return f"{_round(p, 3):.3f}"
    return f"{_round(p, 2):.2f}"


def p_phrase(symbol, p):
    """"*P* = p" / "*Q* < 0.001" for captions and cells."""
    s = fmt_p(p)
    if s.startswith("<"):
        return f"*{symbol}* < {s[1:]}"
    return f"*{symbol}* = {s}"


def join_and(items):
    """"a", "a and b", "a, b and c" for caption lists."""
    items = list(items)
    if len(items) <= 2:
        return " and ".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def fmt_n_pct(events, n):
    """"n (x.x%)" of a group whose N is stated in the caption."""
    if _missing(n) or int(n) == 0:
        return DASH
    return f"{fmt_count(events)} ({fmt_pct(100 * float(events) / float(n))})"


# =============================================================================
# ROW ORDER
# =============================================================================
def sort_key(label, last=("Other",)):
    """Alphabetical, case-insensitive, footnote markers ignored; labels that
    start with an entry of `last` go last."""
    clean = re.sub(r"[†‡*]", "", str(label)).strip()
    return (any(clean.startswith(x) for x in last), clean.lower())


# =============================================================================
# CAPTIONS
# =============================================================================
def load_captions(path, script_file):
    """The caption file as a dict, or None when no path is set.
    A relative path is looked up in the working folder, then beside the script."""
    if not path:
        return None
    for p in (path, os.path.join(os.path.dirname(os.path.abspath(script_file)), path)):
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                return json.load(fh)
    raise FileNotFoundError(f"caption file not found: {path}")


def fill(template, values):
    """Fill {name} fields; a missing field raises (no silent blanks)."""
    return template.format_map(values)


# =============================================================================
# WORD OUTPUT
# =============================================================================
def _runs(paragraph, text, size, bold=False, italic=False):
    from docx.shared import Pt
    parts = re.split(r"(\*[^*]+\*)", text)
    for part in parts:
        if not part:
            continue
        it = part.startswith("*") and part.endswith("*") and len(part) > 2
        run = paragraph.add_run(part[1:-1] if it else part)
        run.font.size = Pt(size)
        run.bold = bold
        run.italic = italic or it


def _shade(cell, hex_color):
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), hex_color)
    cell._tc.get_or_add_tcPr().append(shd)


def _repeat_header(row):
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    trPr = row._tr.get_or_add_trPr()
    th = OxmlElement("w:tblHeader")
    th.set(qn("w:val"), "true")
    trPr.append(th)


def preview(title, headers, rows):
    """Plain-text console preview of a table."""
    plain = lambda s: str(s).replace("*", "").replace("\n", " ")
    print("\n" + "=" * 72 + f"\n  {plain(title)}\n" + "=" * 72)
    hdr = [plain(h) for h in headers]
    widths = [len(h) for h in hdr]
    for r in rows:
        if r["kind"] == "data":
            widths = [max(w, len(plain(v))) for w, v in zip(widths, r["values"])]
    widths = [min(w, 60 if i == 0 else 26) for i, w in enumerate(widths)]

    def line(vals):
        return "  " + "  ".join(plain(v)[:w].ljust(w) if i == 0 else plain(v)[:w].rjust(w)
                                for i, (v, w) in enumerate(zip(vals, widths)))
    print(line(hdr))
    print("  " + "-" * (sum(widths) + 2 * (len(widths) - 1)))
    for r in rows:
        print(f"\n  [{plain(r['label'])}]" if r["kind"] == "header" else line(r["values"]))


def write_table(path, headers, rows, title=None, captions=(), first_width=None, pad=0,
                page=(11, 8.5)):
    """One landscape .docx: optional title above the table, the caption (a
    list of sentences, joined into one paragraph) after it. rows: {"kind":
    "header", "label": str} (a merged group row) or {"kind": "data", "values":
    [str, ...]}. first_width (inches) and pad (characters added to every other
    column's need) adjust wide tables; page (width, height in inches, landscape)
    gives a very wide table a larger sheet (e.g. (17, 11), tabloid); the
    default is a letter landscape page."""
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Inches

    preview(title or os.path.basename(path), headers, rows)
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Inches(page[0]), Inches(page[1])
    sec.left_margin = sec.right_margin = Inches(0.6)
    if title:
        _runs(doc.add_paragraph(), title, 12, bold=True)
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.autofit = False
    usable = page[0] - 2 * 0.6
    first_w = first_width if first_width else min(3.2, usable * 0.36)
    # the other columns share the rest in proportion to their widest content
    # (the longest cell, or the longest word of the header), so an estimate
    # with its inline CI stays on one line
    plain = lambda s: str(s).replace("*", "")
    need = []
    for i in range(1, len(headers)):
        cells = [len(plain(r["values"][i])) for r in rows if r["kind"] == "data"]
        words = [len(w) for w in plain(headers[i]).split()]
        need.append(max([6] + cells + words) + pad)
    widths = [first_w] + [(usable - first_w) * k / sum(need) for k in need]
    for i, h in enumerate(headers):
        c = table.rows[0].cells[i]
        c.text = ""
        _runs(c.paragraphs[0], h, 9, bold=True)
        _shade(c, "D9D9D9")
    _repeat_header(table.rows[0])
    for r in rows:
        cells = table.add_row().cells
        if r["kind"] == "header":
            merged = cells[0]
            for c in cells[1:]:
                merged = merged.merge(c)
            merged.text = ""
            _runs(merged.paragraphs[0], r["label"], 9, bold=True)
            _shade(merged, "F2F2F2")
            continue
        for i, v in enumerate(r["values"]):
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            _runs(p, str(v), 8.5)
            if i > 0:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    from docx.oxml.ns import qn
    for col, w in zip(table.columns, widths):
        col.width = Inches(w)
    for gc, w in zip(table._tbl.tblGrid.findall(qn("w:gridCol")), widths):
        gc.set(qn("w:w"), str(int(w * 1440)))
    for row in table.rows:
        for c, w in zip(row.cells, widths):
            c.width = Inches(w)
    # caption: one paragraph after the table
    if captions:
        _runs(doc.add_paragraph(), " ".join(captions), 8)
    import write_guard as wg
    wg.write_new(path, lambda tmp: doc.save(tmp))
    print(f"  Saved: {path}")


def write_caption(path, title, captions):
    """A figure's title and caption: one .docx with the title
    in bold and the caption as one paragraph, the same markup as the tables."""
    from docx import Document
    doc = Document()
    _runs(doc.add_paragraph(), title, 12, bold=True)
    _runs(doc.add_paragraph(), " ".join(captions), 9)
    import write_guard as wg
    wg.write_new(path, lambda tmp: doc.save(tmp))
    print(f"  Saved: {path}")


def save_figure(fig, path, **kwargs):
    """fig.savefig to one new file under the output root (write_guard.py:
    stops if the file exists); the format follows the extension."""
    import write_guard as wg
    wg.write_new(path, lambda tmp: fig.savefig(tmp, **kwargs))
    print(f"  Saved: {path}")
