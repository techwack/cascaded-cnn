"""Build the final report in the SRM project-report format, using the example .docx as the formatting base.

    python scripts/make_srm_report.py --base formattedv4.docx --out Report.docx [--pages pages.json]

Pass 1 (no --pages) writes placeholder page numbers; scripts/page_numbers.ps1 asks Word for the real page of every
heading/caption and writes pages.json; pass 2 (--pages pages.json) fills the contents and the lists of figures and tables.
Items the author must supply (course code, guide) are written as yellow-highlighted placeholders.
"""
import argparse
import copy
import json
import re
import sys
import tempfile
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_COLOR_INDEX
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt
from docx.text.paragraph import Paragraph
from docx.text.run import Run

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import importlib.util  # noqa: E402

import srm_content as sc  # noqa: E402

_spec = importlib.util.spec_from_file_location("mf", HERE / "make_final_report.py")
mf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mf)

NAMES = [("Divyanshi Jain", "RA2311026010639"), ("Syed Mohammed Ghouse S", "RA2311026010643"), ("Purvee Singh", "RA2311026010645")]
GUIDE = "Dr. Illakiya T"
DESIG = "Assistant Professor"
FONT = "Times New Roman"


# ------------------------------------------------------------------ low-level helpers
def first_text_run(par_el):
    for r in par_el.findall(qn("w:r")):
        if r.find(qn("w:t")) is not None:
            return r
    return None


def make_run(tmpl, text, bold=None, highlight=False, parent=None):
    r = copy.deepcopy(tmpl) if tmpl is not None else OxmlElement("w:r")
    for ch in list(r):
        if ch.tag != qn("w:rPr"):
            r.remove(ch)
    t = OxmlElement("w:t")
    t.text = text
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    r.append(t)
    run = Run(r, parent)
    if bold is not None:
        run.bold = bold
    if highlight:
        run.font.highlight_color = WD_COLOR_INDEX.YELLOW
    return r


def fill(par, text, bold=None):
    """Replace the paragraph text, keeping paragraph and first-run formatting. {{X}} becomes a highlighted placeholder [X]."""
    tmpl = first_text_run(par._p)
    tmpl = copy.deepcopy(tmpl) if tmpl is not None else None
    for r in list(par._p.findall(qn("w:r"))):
        par._p.remove(r)
    for part in re.split(r"(\{\{[^}]+\}\})", text):
        if not part:
            continue
        hl = part.startswith("{{")
        par._p.append(make_run(tmpl, f"[{part[2:-2]}]" if hl else part, bold, hl, par))


def fill_segments(par, segs):
    tmpl = first_text_run(par._p)
    tmpl = copy.deepcopy(tmpl) if tmpl is not None else None
    for r in list(par._p.findall(qn("w:r"))):
        par._p.remove(r)
    for text, bold in segs:
        par._p.append(make_run(tmpl, text, bold, False, par))


def tc_fill(tc, text, parent, bold=None):
    p = tc.findall(qn("w:p"))[0]
    for extra in tc.findall(qn("w:p"))[1:]:
        tc.remove(extra)
    par = Paragraph(p, parent)
    if first_text_run(p) is None:  # empty cell: borrow formatting from a sibling cell run
        sib = tc.getparent().find(".//" + qn("w:r"))
        if sib is not None:
            p.append(copy.deepcopy(sib))
    fill(par, text, bold)


# ------------------------------------------------------------------ main build
def build(base: Path, out: Path, pages: dict | None):
    doc = Document(str(base))
    P = list(doc.paragraphs)
    parent = P[132]._parent
    body = doc.element.body
    sect = body.find(qn("w:sectPr"))

    protos = {k: copy.deepcopy(P[i]._p) for k, i in dict(chnum=130, chtitle=131, sec=135, sub=170, body=132, cap=195, img=351, figcap=352, refh=712, ref=713).items()}

    # ---------------- front matter ----------------
    fill(P[1], "21CSC402P-PROJECT")
    for i, (n, r) in zip((5, 6, 7), NAMES):
        fill(P[i], f"{n} [{r}]")
    fill(P[11], GUIDE)
    fill(P[12], f"({DESIG}, Department of Computational Intelligence)")

    t0 = doc.tables[0]._tbl
    tc_fill(t0.tr_lst[0].tc_lst[0], sc.TITLE, parent)
    t1 = doc.tables[1]
    tc_fill(t1._tbl.tr_lst[1].tc_lst[2], ", ".join(n for n, _ in NAMES), parent)
    tc_fill(t1._tbl.tr_lst[2].tc_lst[2], ", ".join(r for _, r in NAMES), parent)
    tc_fill(t1._tbl.tr_lst[3].tc_lst[2], sc.TITLE, parent)

    decl = doc.tables[2]._tbl.tr_lst[1].tc_lst[0].findall(qn("w:p"))
    for pe, (n, r) in zip(decl[3:6], NAMES):
        fill(Paragraph(pe, parent), f"{n}, {r}")

    fill(P[59], f"Certified that 21CSC402P - Project report titled \u201c{sc.TITLE}\u201d is the bonafide work of \u201c"
         + ", ".join(f"{n} [{r}]" for n, r in NAMES) + "\u201d who carried out the project work under my supervision. Certified further, that to the best of my knowledge the work reported herein does not form any other "
         "project report or dissertation on the basis of which a degree or award was conferred on an earlier occasion on this or any other candidate.")
    sig = doc.tables[3]._tbl.tr_lst[0].tc_lst[0]
    ps = sig.findall(qn("w:p"))
    fill(Paragraph(ps[2], parent), GUIDE.upper())
    fill(Paragraph(ps[4], parent), DESIG.upper())

    fill(P[74], "We register our immeasurable thanks to our Faculty Advisor, " + GUIDE + ", Department of Computational Intelligence, SRM Institute of Science and Technology, for leading and helping us to complete our course.")
    fill(P[75], "Our inexpressible respect and thanks to our guide, " + GUIDE + ", Department of Computational Intelligence, SRM Institute of Science and Technology, for providing us with an opportunity to pursue our project "
         "under their mentorship, and for the freedom and support to explore the topics of our interest.")
    for i, (n, _) in zip((81, 82, 83), NAMES):
        fill(P[i], n)
    fill(P[87], sc.ABSTRACT)
    fill_segments(P[89], [("Index Terms:", True), (sc.INDEX_TERMS[len("Index Terms:"):], False)])

    # ---------------- remove the example chapters ----------------
    el = P[129]._p.getnext()
    while el is not None and el is not sect:
        nxt = el.getnext()
        body.remove(el)
        el = nxt
    doc.tables[9]._tbl.getparent().remove(doc.tables[9]._tbl)

    tmp = Path(tempfile.mkdtemp())
    charts = sc.mr.charts(tmp)
    mf.per_class_chart(tmp / "perclass.png")
    demo_png = sc.ROOT / "docs" / "figures" / "demo_predict.png"
    blocks = sc.build_blocks(charts, tmp / "perclass.png", demo_png)

    toc, figs, tabs = [], [], []

    def new_par(proto_key, segs, **fmt):
        e = copy.deepcopy(protos[proto_key])
        sect.addprevious(e)
        par = Paragraph(e, parent)
        fill_segments(par, segs)
        pf = par.paragraph_format
        if fmt.get("keep"):
            pf.keep_with_next = True
        if fmt.get("break"):
            pf.page_break_before = True
        if fmt.get("align") is not None:
            par.alignment = fmt["align"]
        if fmt.get("left") is not None:
            pf.left_indent = Cm(fmt["left"])
            pf.first_line_indent = Cm(fmt.get("first", 0))
        return par

    def add_table(rows, widths, bold_rows, size):
        t = doc.add_table(rows=len(rows), cols=len(rows[0]))
        t.style = doc.styles["Table Grid"]
        t.alignment = 1
        t.autofit = False
        for ri, row in enumerate(rows):
            for ci, val in enumerate(row):
                cell = t.cell(ri, ci)
                cell.width = Cm(widths[ci])
                par = cell.paragraphs[0]
                par.alignment = WD_ALIGN_PARAGRAPH.CENTER if ri == 0 else WD_ALIGN_PARAGRAPH.LEFT
                run = par.add_run(str(val))
                run.font.name = FONT
                run.font.size = Pt(size)
                rpr = run._r.get_or_add_rPr()
                rf = rpr.find(qn("w:rFonts"))
                for a in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
                    rf.set(qn(a), FONT)
                if ri == 0 or ri in bold_rows:
                    run.bold = True
                if ri == 0:
                    tcPr = cell._tc.get_or_add_tcPr()
                    shd = OxmlElement("w:shd")
                    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), "E7E6E6")
                    tcPr.append(shd)
        trPr = t.rows[0]._tr.get_or_add_trPr()
        trPr.append(OxmlElement("w:tblHeader"))
        return t

    pending_chapter = None
    for blk in blocks:
        kind = blk[0]
        if kind == "chapter":
            _, n, title = blk
            new_par("chnum", [(f"CHAPTER {n}", True)], **{"break": True, "keep": True})
            new_par("chtitle", [(title, True)], keep=True)
            toc.append(("chapter", str(n), title, f"CHAPTER {n}"))
        elif kind in ("sec", "sub"):
            text = blk[1]
            new_par(kind, [(text, True)], keep=True)
            num, title = text.split(" ", 1)
            toc.append((kind, num, title, text))
        elif kind == "p":
            new_par("body", [(blk[1], False)])
        elif kind == "b":
            segs = [("\u2022  ", False), (blk[1] + " ", True), (blk[2], False)]
            new_par("body", segs, left=1.0, first=-0.5)
        elif kind == "table":
            _, cap, rows, widths, bold_rows, size = blk
            new_par("cap", [(cap, True)], keep=True)
            tabs.append((cap.split(":")[0].replace("Table ", ""), cap.split(": ", 1)[1], cap))
            add_table(rows, widths, bold_rows, size)
            new_par("body", [("", False)], left=0)
        elif kind == "fig":
            _, path, width, cap = blk
            e = copy.deepcopy(protos["img"])
            sect.addprevious(e)
            ip = Paragraph(e, parent)
            for r in list(e.findall(qn("w:r"))):
                e.remove(r)
            ip.add_run().add_picture(str(path), width=Cm(width))
            ip.alignment = WD_ALIGN_PARAGRAPH.CENTER
            ip.paragraph_format.keep_with_next = True
            new_par("figcap", [(cap, True)], align=WD_ALIGN_PARAGRAPH.CENTER)
            num = re.match(r"Fig\. (\d+)\.", cap).group(1)
            figs.append((num, cap.split(". ", 1)[1].rstrip("."), cap))
            new_par("body", [("", False)], left=0)

    new_par("refh", [("REFERENCES", True)], **{"break": True, "keep": True, "align": WD_ALIGN_PARAGRAPH.CENTER}).paragraph_format.first_line_indent = Cm(0)
    toc.append(("front", "", "REFERENCES", "REFERENCES"))
    for ref in sc.REFERENCES:
        new_par("ref", [(ref, False)], left=0.9, first=-0.9)

    # ---------------- contents, lists, abbreviations ----------------
    def roman(n):
        vals = [(10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i")]
        s = ""
        for v, sym in vals:
            while n >= v:
                s += sym
                n -= v
        return s

    def page(key, front=False):
        if not pages or key not in pages:
            return "0"
        return roman(int(pages[key])) if front else str(pages[key])

    def rebuild(table, protos_idx, rows_data):
        tbl = table._tbl
        trs = tbl.tr_lst
        tpl = [copy.deepcopy(trs[i]) for i in protos_idx]
        for tr in trs[1:]:
            tbl.remove(tr)
        for kind, cells in rows_data:
            tr = copy.deepcopy(tpl[kind])
            for tc, text in zip(tr.tc_lst, cells):
                tc_fill(tc, text, parent)
            tbl.append(tr)

    front = [("ABSTRACT", "ABSTRACT"), ("TABLE OF CONTENTS", "TABLE OF CONTENTS"), ("LIST OF FIGURES", "LIST OF FIGURES"), ("LIST OF TABLES", "LIST OF TABLES"), ("ABBREVIATIONS", "ABBREVIATIONS")]
    rows = [(0, ["", t, page(k, True)]) for t, k in front]
    for kind, num, title, key in toc:
        if kind == "chapter":
            rows.append((1, [num, title, page(key)]))
        elif kind == "front":
            rows.append((0, ["", title, page(key)]))
        else:
            rows.append((2, [num, title, page(key)]))
    rebuild(doc.tables[5], (1, 6, 7), rows)
    rebuild(doc.tables[6], (1,), [(0, [n, t, page(c)]) for n, t, c in figs])
    rebuild(doc.tables[7], (1,), [(0, [n, t, page(c)]) for n, t, c in tabs])
    rebuild(doc.tables[8], (1,), [(0, [a, b]) for a, b in sc.ABBREVIATIONS])

    # footers of the template: the list-of-tables page has a typed-in "ix"; make numbering continuous roman instead
    def page_field_paragraph(p_el):
        for r in list(p_el.findall(qn("w:r"))):
            p_el.remove(r)
        for kind, extra in (("begin", None), (None, " PAGE \* MERGEFORMAT "), ("separate", None), (None, "text"), ("end", None)):
            r = OxmlElement("w:r")
            if kind:
                fc = OxmlElement("w:fldChar"); fc.set(qn("w:fldCharType"), kind); r.append(fc)
            elif extra == "text":
                t = OxmlElement("w:t"); t.text = "1"; r.append(t)
            else:
                it = OxmlElement("w:instrText"); it.set("{http://www.w3.org/XML/1998/namespace}space", "preserve"); it.text = extra; r.append(it)
            p_el.append(r)

    for para in doc.sections[3].footer._element.findall(qn("w:p")):
        if "".join(t.text or "" for t in para.iter(qn("w:t"))).strip() == "ix":
            page_field_paragraph(para)
    for idx, keep_start in ((3, False), (4, False)):
        sp = doc.sections[idx]._sectPr
        pg = sp.find(qn("w:pgNumType"))
        if pg is None:
            pg = OxmlElement("w:pgNumType")
            sp.append(pg)
        pg.set(qn("w:fmt"), "lowerRoman")
        if qn("w:start") in pg.attrib:
            del pg.attrib[qn("w:start")]

    doc.core_properties.title = sc.TITLE
    doc.core_properties.author = ", ".join(n for n, _ in NAMES)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out))

    keys = [k for _, k in front] + [k for *_, k in toc] + [c for *_, c in figs] + [c for *_, c in tabs]
    (out.parent / (out.stem + "_keys.json")).write_text(json.dumps(keys), encoding="utf-8")
    print("wrote", out, "| toc entries", len(toc), "| figures", len(figs), "| tables", len(tabs))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=r"C:\Users\Divyanshi\Downloads\formattedv4.docx")
    ap.add_argument("--out", default=r"C:\Users\Divyanshi\Downloads\Cascaded_CNN_Project_Report.docx")
    ap.add_argument("--pages", default=None)
    a = ap.parse_args()
    build(Path(a.base), Path(a.out), json.load(open(a.pages, encoding="utf-8-sig")) if a.pages else None)
