"""Build the Computer Vision project report (format of the NNDL case-study template, content of cv_content.py).

    python scripts/make_cv_report.py --out CV_Project_Report.docx [--pages pages.json]
"""
import argparse
import copy
import json
import re
import sys
import tempfile
from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt
from docx.text.paragraph import Paragraph

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cv_content as cc  # noqa: E402
import make_nndl_report as mn  # noqa: E402

sc, mf = mn.sc, mn.mf
fill, fill_segments, cell_fill, set_fonts = mn.fill, mn.fill_segments, mn.cell_fill, mn.set_fonts
J, C, L, NAMES = mn.J, mn.C, mn.L, mn.NAMES


def highlight(par):
    for r in par.runs:
        h = OxmlElement("w:highlight"); h.set(qn("w:val"), "yellow")
        r._r.get_or_add_rPr().append(h)


def build(base, out, pages):
    doc = Document(str(base))
    P = list(doc.paragraphs)
    parent = P[36]._parent
    body = doc.element.body
    sect = body.find(qn("w:sectPr"))
    proto = copy.deepcopy(P[36]._p)
    T = list(doc.tables)

    fill(P[0], cc.TITLE.upper(), size=15)
    fill(P[4], "PROJECT REPORT")
    fill(P[6], cc.COURSE)
    for i in (2, 3, 8, 9):
        if not P[i].text.strip() and not P[i]._p.xpath(".//w:drawing") and not P[i]._p.xpath(".//w:sectPr"):
            body.remove(P[i]._p)
    for i, (n, r) in zip((11, 12, 13), NAMES):
        fill(P[i], f"{n} [{r}]")
    fill(P[14], "Under the guidance of", size=16)
    fill(P[15], "Dr. Sridhar S", bold=True, size=16)
    fill_segments(P[36], [("This is to certify that the Project Report titled \u201c", False, None), (cc.TITLE, True, None),
                          ("\u201d is a bonafide record of work carried out by the following student(s) as part of the course ", False, None),
                          (cc.COURSE.upper(), True, None), (", during the academic year 2026\u20132027 Odd Semester.", False, None)])
    for tr, (n, r) in zip(T[0]._tbl.tr_lst[1:], NAMES):
        cell_fill(tr.tc_lst[0], r, parent)
        cell_fill(tr.tc_lst[1], n, parent)
    sig = T[1]._tbl.tr_lst[0].tc_lst[0]
    for pe in sig.findall(qn("w:p")):
        if "FACULTY NAME" in "".join(t.text or "" for t in pe.iter(qn("w:t"))):
            pp = Paragraph(pe, parent)
            fill(pp, "Dr. Sridhar S")
            pass
    fill_segments(P[41], [("  Date of Submission: ", True, None), ("17-11-2026", False, None)])

    P[43].paragraph_format.page_break_before = True
    fill(P[44], cc.ABSTRACT)
    P[44].alignment = J
    P[44].paragraph_format.line_spacing = 1.5
    P[44].paragraph_format.first_line_indent = Cm(0)
    for i in range(45, 65):
        if not P[i].text.strip():
            body.remove(P[i]._p)
    for i in range(67, 77):
        if not P[i].text.strip() and not P[i]._p.xpath(".//w:sectPr"):
            body.remove(P[i]._p)

    # abbreviations page before the contents
    ah = copy.deepcopy(P[65]._p)
    P[65]._p.addprevious(ah)
    ahp = Paragraph(ah, parent)
    fill(ahp, "ABBREVIATIONS")
    ahp.paragraph_format.page_break_before = True
    P[65].paragraph_format.page_break_before = True
    ab = doc.add_table(rows=len(cc.ABBREVIATIONS) + 1, cols=2)
    ab.style = T[2].style
    P[65]._p.addprevious(ab._tbl)
    ab.alignment = 1
    for ri, row in enumerate([("Abbreviation", "Expansion")] + cc.ABBREVIATIONS):
        for ci, v in enumerate(row):
            cell = ab.cell(ri, ci)
            cell.width = Cm((3.5, 11.5)[ci])
            run = cell.paragraphs[0].add_run(v)
            set_fonts(run, 11, ri == 0)
            cell.paragraphs[0].paragraph_format.line_spacing = 1.15
    # tables need borders if the style has none
    borders = OxmlElement("w:tblBorders")
    for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
        b = OxmlElement(f"w:{side}")
        b.set(qn("w:val"), "single"); b.set(qn("w:sz"), "4"); b.set(qn("w:space"), "0"); b.set(qn("w:color"), "000000")
        borders.append(b)
    ab._tbl.tblPr.append(borders)

    # drop list of figures / list of tables (the CV format has none)
    P[77]._p.getparent().remove(P[77]._p)
    P[80]._p.getparent().remove(P[80]._p)
    T[3]._tbl.getparent().remove(T[3]._tbl)
    T[4]._tbl.getparent().remove(T[4]._tbl)
    for i in range(78, 94):
        if i in (80,):
            continue
        if not P[i].text.strip() and not P[i]._p.xpath(".//w:sectPr") and P[i]._p.getparent() is not None:
            body.remove(P[i]._p)

    el = P[94]._p.getnext()
    while el is not None and el is not sect:
        nxt = el.getnext()
        body.remove(el)
        el = nxt

    tmp = Path(tempfile.mkdtemp())
    charts = sc.mr.charts(tmp)
    mf.per_class_chart(tmp / "perclass.png")
    blocks = cc.blocks(charts, tmp / "perclass.png")

    has_grid = "Table Grid" in [s.name for s in doc.styles]
    toc = []

    def new_par(segs, align=J, ls=1.5, sa=6, sb=0, left=None, first=None, keep=False, brk=False, size=None):
        e = copy.deepcopy(proto)
        sect.addprevious(e)
        par = Paragraph(e, parent)
        fill_segments(par, segs, size)
        pf = par.paragraph_format
        par.alignment = align
        pf.line_spacing = ls
        pf.space_after = Pt(sa)
        pf.space_before = Pt(sb)
        pf.left_indent = Cm(left) if left is not None else None
        pf.first_line_indent = Cm(first) if first is not None else Cm(0)
        pf.keep_with_next = keep
        pf.page_break_before = brk
        return par

    def add_table(rows, widths, bold_rows, size):
        t = doc.add_table(rows=len(rows), cols=len(rows[0]))
        if has_grid:
            t.style = doc.styles["Table Grid"]
        else:
            bd = OxmlElement("w:tblBorders")
            for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
                b = OxmlElement(f"w:{side}")
                b.set(qn("w:val"), "single"); b.set(qn("w:sz"), "4"); b.set(qn("w:space"), "0"); b.set(qn("w:color"), "000000")
                bd.append(b)
            t._tbl.tblPr.append(bd)
        t.alignment = 1
        t.autofit = False
        for ri, row in enumerate(rows):
            for ci, val in enumerate(row):
                cell = t.cell(ri, ci)
                cell.width = Cm(widths[ci])
                par = cell.paragraphs[0]
                par.alignment = C if ri == 0 else L
                par.paragraph_format.line_spacing = 1.0
                run = par.add_run(str(val))
                set_fonts(run, size, ri == 0 or ri in bold_rows)
                if ri == 0:
                    shd = OxmlElement("w:shd")
                    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), "E7E6E6")
                    cell._tc.get_or_add_tcPr().append(shd)
        t.rows[0]._tr.get_or_add_trPr().append(OxmlElement("w:tblHeader"))

    for blk in blocks:
        kind = blk[0]
        if kind == "chapter":
            _, n, title = blk
            new_par([(f"CHAPTER {n}", True, None)], align=C, ls=1.15, sa=2, keep=True, brk=True, size=14)
            new_par([(title, True, None)], align=C, ls=1.15, sa=10, keep=True, size=14)
            toc.append((str(n), title, f"CHAPTER {n}"))
        elif kind == "sec":
            new_par([(blk[1], True, None)], align=L, ls=1.15, sa=4, sb=8, keep=True, size=12)
            num, _, title = blk[1].partition(" ")
            toc.append((num, title, blk[1]))
        elif kind == "sub":
            new_par([(blk[1], True, True)], align=L, ls=1.15, sa=3, sb=4, keep=True, size=12)
        elif kind == "p":
            new_par([(blk[1], False, None)])
        elif kind == "b":
            new_par([("\u2022  ", False, None), (blk[1] + " ", True, None), (blk[2], False, None)], ls=1.5, sa=2, left=1.0, first=-0.5)
        elif kind == "eq":
            new_par([(blk[1], False, True)], align=C, ls=1.15, sa=6, sb=2)
        elif kind == "table":
            _, cap, rows, widths, bold_rows, size = blk
            new_par([(cap, True, None)], align=C, ls=1.15, sa=3, sb=4, keep=True, size=11)
            add_table(rows, widths, bold_rows, size)
            new_par([("", False, None)], ls=1.0, sa=4)
        elif kind == "fig":
            _, path, width, cap = blk
            e = copy.deepcopy(proto)
            sect.addprevious(e)
            ip = Paragraph(e, parent)
            for r in list(e.findall(qn("w:r"))):
                e.remove(r)
            ip.add_run().add_picture(str(path), width=Cm(width))
            ip.alignment = C
            ip.paragraph_format.line_spacing = 1.0
            ip.paragraph_format.space_after = Pt(2)
            ip.paragraph_format.keep_with_next = True
            new_par([(cap, True, None)], align=C, ls=1.15, sa=8, size=11)

    new_par([("REFERENCES", True, None)], align=C, ls=1.15, sa=10, keep=True, brk=True, size=14)
    toc.append(("", "REFERENCES", "REFERENCES"))
    for i, ref in enumerate(cc.REFERENCES, 1):
        new_par([(f"[{i}]\t{ref}", False, None)], align=J, ls=1.15, sa=6, left=1.0, first=-1.0, size=11)

    def page(key):
        return str(pages[key]) if pages and key in pages else "0"

    tbl = T[2]._tbl
    trs = tbl.tr_lst
    tpl = copy.deepcopy(trs[1])
    for tr in trs[1:]:
        tbl.remove(tr)
    for n, t, k in toc:
        tr = copy.deepcopy(tpl)
        for tc, text in zip(tr.tc_lst, (n, t, page(k))):
            cell_fill(tc, text, parent)
        tbl.append(tr)

    def footer_page(section):
        f = section.footer
        f.is_linked_to_previous = False
        par = f.paragraphs[0]
        par.alignment = C
        for r in list(par._p.findall(qn("w:r"))):
            par._p.remove(r)
        for kind, extra in (("begin", None), (None, "PAGE"), ("separate", None), (None, "text"), ("end", None)):
            r = OxmlElement("w:r")
            if kind:
                fc = OxmlElement("w:fldChar"); fc.set(qn("w:fldCharType"), kind); r.append(fc)
            elif extra == "text":
                tt = OxmlElement("w:t"); tt.text = "1"; r.append(tt)
            else:
                it = OxmlElement("w:instrText"); it.set("{http://www.w3.org/XML/1998/namespace}space", "preserve"); it.text = " PAGE "; r.append(it)
            par._p.append(r)

    footer_page(doc.sections[0])
    footer_page(doc.sections[1])
    doc.core_properties.title = cc.TITLE
    doc.core_properties.author = ", ".join(n for n, _ in NAMES)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out))
    (out.parent / (out.stem + "_keys.json")).write_text(json.dumps([k for *_, k in toc]), encoding="utf-8")
    print("wrote", out, "| toc rows", len(toc))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=r"C:\Users\Divyanshi\Downloads\NNDL_Case_Study_Report_Format.docx")
    ap.add_argument("--out", default=r"C:\Users\Divyanshi\Downloads\CV_Project_Report.docx")
    ap.add_argument("--pages", default=None)
    a = ap.parse_args()
    build(Path(a.base), Path(a.out), json.load(open(a.pages, encoding="utf-8-sig")) if a.pages else None)
