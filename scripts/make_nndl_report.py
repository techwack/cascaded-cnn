"""Build the NNDL case-study report on NNDL_Case_Study_Report_Format.docx.

    python scripts/make_nndl_report.py --out NNDL_Case_Study_Report.docx [--pages pages.json]
Pass 1 writes placeholder page numbers; scripts/page_numbers.ps1 reads the real ones from Word; pass 2 fills them.
"""
import argparse
import copy
import json
import re
import sys
import tempfile
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt
from docx.text.paragraph import Paragraph
from docx.text.run import Run

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import importlib.util  # noqa: E402

import nndl_content as nc  # noqa: E402
import srm_content as sc  # noqa: E402
import srm_content2 as c2  # noqa: E402

_spec = importlib.util.spec_from_file_location("mf", HERE / "make_final_report.py")
mf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mf)

FONT = "Times New Roman"
NAMES = [("Divyanshi Jain", "RA2311026010639"), ("Syed Mohammed Ghouse S", "RA2311026010643"), ("Purvee Singh", "RA2311026010645")]
GUIDE = "Dr. Illakiya T"
J, C, L = WD_ALIGN_PARAGRAPH.JUSTIFY, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT


def first_text_run(par_el):
    for r in par_el.findall(qn("w:r")):
        if r.find(qn("w:t")) is not None:
            return r
    return None


def set_fonts(run, size=None, bold=None, italic=None, name=FONT):
    run.font.name = name
    rf = run._r.get_or_add_rPr().find(qn("w:rFonts"))
    for a in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rf.set(qn(a), name)
    if size:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic


def fill_segments(par, segs, size=None):
    """segs: list of (text, bold, italic)."""
    tmpl = first_text_run(par._p)
    tmpl = copy.deepcopy(tmpl) if tmpl is not None else None
    for r in list(par._p.findall(qn("w:r"))):
        par._p.remove(r)
    for text, bold, italic in segs:
        r = copy.deepcopy(tmpl) if tmpl is not None else OxmlElement("w:r")
        for ch in list(r):
            if ch.tag != qn("w:rPr"):
                r.remove(ch)
        t = OxmlElement("w:t"); t.text = text
        t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        r.append(t)
        par._p.append(r)
        set_fonts(Run(r, par), size, bold, italic)


def fill(par, text, bold=None, size=None):
    fill_segments(par, [(text, bold, None)], size)


def cell_fill(tc, text, parent, bold=None):
    p = tc.findall(qn("w:p"))[0]
    for extra in tc.findall(qn("w:p"))[1:]:
        tc.remove(extra)
    par = Paragraph(p, parent)
    if first_text_run(p) is None:
        sib = tc.getparent().find(".//" + qn("w:r"))
        if sib is not None:
            p.append(copy.deepcopy(sib))
    fill(par, text, bold)


def build(base, out, pages):
    doc = Document(str(base))
    P = list(doc.paragraphs)
    parent = P[36]._parent
    body = doc.element.body
    sect = body.find(qn("w:sectPr"))
    proto = copy.deepcopy(P[36]._p)

    # ---------------- cover, certificate ----------------
    fill(P[0], nc.TITLE.upper(), size=15)
    for i in (2, 3, 8, 9):  # blank spacer lines on the cover: the title is longer than the placeholder
        if not P[i].text.strip() and not P[i]._p.xpath(".//w:drawing") and not P[i]._p.xpath(".//w:sectPr"):
            body.remove(P[i]._p)
    for i, (n, r) in zip((11, 12, 13), NAMES):
        fill(P[i], f"{n} [{r}]")
    t36 = P[36].text.replace("CASE STUDY TITLE", nc.TITLE).replace("\u201c" + "CASE STUDY TITLE", "\u201c" + nc.TITLE)
    fill_segments(P[36], [("This is to certify that the Case Study Report titled \u201c", False, None), (nc.TITLE, True, None),
                          ("\u201d is a bonafide record of work carried out by the following student(s) as part of the course ", False, None),
                          ("21CSC406J - NEURAL NETWORKS AND DEEP LEARNING", True, None), (", during the academic year 2026\u20132027 Odd Semester.", False, None)])
    t0 = doc.tables[0]._tbl
    for tr, (n, r) in zip(t0.tr_lst[1:], NAMES):
        cell_fill(tr.tc_lst[0], r, parent)
        cell_fill(tr.tc_lst[1], n, parent)
    sig = doc.tables[1]._tbl.tr_lst[0].tc_lst[0]
    for pe in sig.findall(qn("w:p")):
        txt = "".join(t.text or "" for t in pe.iter(qn("w:t")))
        if "FACULTY NAME" in txt:
            fill(Paragraph(pe, parent), GUIDE)
    fill_segments(P[41], [("  Date of Submission: ", True, None), ("November 2026", False, None)])

    # abstract: replace the blank lines by the text
    P[43].paragraph_format.page_break_before = True  # abstract starts on its own page
    fill(P[44], nc.ABSTRACT)
    P[44].alignment = J
    P[44].paragraph_format.line_spacing = 1.5
    P[44].paragraph_format.first_line_indent = Cm(0)
    for i in range(45, 65):
        if not P[i].text.strip():
            body.remove(P[i]._p)
    P[65].paragraph_format.page_break_before = True
    for i in range(67, 77):
        if not P[i].text.strip() and not P[i]._p.xpath(".//w:sectPr"):
            body.remove(P[i]._p)
    P[77].paragraph_format.page_break_before = True
    for i in range(82, 94):
        if not P[i].text.strip():
            body.remove(P[i]._p)

    # ---------------- remove the skeleton chapters ----------------
    el = P[94]._p.getnext()
    while el is not None and el is not sect:
        nxt = el.getnext()
        body.remove(el)
        el = nxt

    tmp = Path(tempfile.mkdtemp())
    charts = sc.mr.charts(tmp)
    mf.per_class_chart(tmp / "perclass.png")
    demo_png = sc.ROOT / "docs" / "figures" / "demo_predict.png"
    blocks = nc.blocks(charts, tmp / "perclass.png", demo_png)

    has_grid = "Table Grid" in [s.name for s in doc.styles]
    figs, tabs, toc = [], [], []

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
            borders = OxmlElement("w:tblBorders")
            for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
                b = OxmlElement(f"w:{side}")
                b.set(qn("w:val"), "single"); b.set(qn("w:sz"), "4"); b.set(qn("w:space"), "0"); b.set(qn("w:color"), "000000")
                borders.append(b)
            t._tbl.tblPr.append(borders)
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

    def emit(blocks):
        for blk in blocks:
            kind = blk[0]
            if kind == "chapter":
                _, n, title = blk
                new_par([(f"CHAPTER {n}", True, None)], align=C, ls=1.15, sa=2, keep=True, brk=True, size=14)
                new_par([(title, True, None)], align=C, ls=1.15, sa=10, keep=True, size=14)
                toc.append((str(n), title, f"CHAPTER {n}"))
            elif kind == "sec":
                new_par([(blk[1], True, None)], align=L, ls=1.15, sa=4, sb=8, keep=True, size=12)
            elif kind == "sub":
                new_par([(blk[1], True, True)], align=L, ls=1.15, sa=3, sb=4, keep=True, size=12)
            elif kind == "p":
                new_par([(blk[1], False, None)])
            elif kind == "b":
                new_par([("\u2022  ", False, None), (blk[1] + " ", True, None), (blk[2], False, None)], ls=1.5, sa=2, left=1.0, first=-0.5)
            elif kind == "eq":
                new_par([(blk[1], False, True)], align=C, ls=1.15, sa=6, sb=2)
            elif kind == "code":
                _, cap, text = blk
                new_par([(cap, True, None)], align=C, ls=1.15, sa=3, sb=4, keep=True, size=11)
                for line in text.split(chr(10)):
                    par = new_par([(line if line else " ", False, None)], align=L, ls=1.0, sa=0, left=0.3, size=8.5)
                    for r in par.runs:
                        set_fonts(r, 8.5, False, False, "Courier New")
                    pPr = par._p.get_or_add_pPr()
                    shd = OxmlElement("w:shd")
                    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), "F2F2F2")
                    pPr.append(shd)
                new_par([("", False, None)], ls=1.0, sa=4)
            elif kind == "table":
                _, cap, rows, widths, bold_rows, size = blk
                new_par([(cap, True, None)], align=C, ls=1.15, sa=3, sb=4, keep=True, size=11)
                num, title = cap.split(": ", 1)
                tabs.append((num.replace("Table ", ""), title, cap))
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
                m = re.match(r"Fig\. ([\w.]+?)\. (.*)", cap)
                figs.append((m.group(1), m.group(2).rstrip("."), cap))

    emit(blocks)

    # references (APA, alphabetical, hanging indent)
    new_par([("REFERENCES", True, None)], align=C, ls=1.15, sa=10, keep=True, brk=True, size=14)
    toc.append(("", "REFERENCES", "REFERENCES"))
    for ref in nc.REFERENCES:
        new_par([(ref, False, None)], align=L, ls=1.15, sa=6, left=1.27, first=-1.27, size=11)

    # appendices
    new_par([("APPENDICES", True, None)], align=C, ls=1.15, sa=10, keep=True, brk=True, size=14)
    toc.append(("", "APPENDICES", "APPENDICES"))
    stats = c2.STATS["improved3"]["triage"]
    rows = [["Layer", "Output shape", "Parameters"], ["Input", "1 x 80 x 48", "0"]]
    for name, shape, p_ in stats["layers"]:
        rows.append([name, " x ".join(str(s) for s in shape), f"{p_:,}"])
    rows.append(["Total", "", f"{stats['params']:,}"])
    app = [("sec", "A. Source code and repository"),
           ("p", "The complete source code, configuration files, trained Stage 3 checkpoints, raw results, tests and this report are available in the project repository: https://github.com/techwack/cascaded-cnn. "
                 "The main entry points are python -m solarnet.train --node all --config configs/pretrained.yaml for training, python -m solarnet.evaluate --flat --config configs/pretrained.yaml for evaluation and "
                 "python scripts/predict.py --sample 5 for the live demonstration."),
           ("sec", "B. Model summary output"),
           ("table", "Table A.1: Layer-by-layer summary of CNN-1 (from-scratch variant, generated from the implemented model)", rows, [4.0, 4.2, 3.0], (len(rows) - 1,), 10),
           ("sec", "C. Key source code"),
           ("code", "Listing A.1: Cascade inference with early exit (src/solarnet/models.py)", c2.src("src/solarnet/models.py", r"def predict\(self", r"def predict_soft")),
           ("code", "Listing A.2: Class weights for the imbalanced loss (src/solarnet/data.py)", c2.src("src/solarnet/data.py", r"^def class_weights")),
           ("code", "Listing A.3: Attribution of errors to stages (src/solarnet/evaluate.py)", c2.src("src/solarnet/evaluate.py", r"^def stage_errors", r"^@torch.inference_mode")),
           ("sec", "D. Hyper-parameter configuration"),
           ("code", "Listing A.4: configs/pretrained.yaml", (sc.ROOT / "configs" / "pretrained.yaml").read_text(encoding="utf-8").replace("\r\n", "\n").rstrip()),
           ("sec", "E. Additional training results and demonstration output"),
           ("fig", c2.FIG / "node_comparison.png", 13.0, "Fig. A.1. Best validation macro-F1 of each stage network, from scratch and pretrained."),
           ("fig", sc.ROOT / "docs" / "figures" / "demo_mix.png", 15.0, "Fig. A.2. Output of the live demonstration for random test images (green: correct, red: wrong).")]
    emit(app)

    # ---------------- contents and lists ----------------
    def page(key):
        return str(pages[key]) if pages and key in pages else "0"

    def rebuild(table, rows_data, proto_idx=1):
        tbl = table._tbl
        trs = tbl.tr_lst
        tpl = copy.deepcopy(trs[proto_idx])
        for tr in trs[1:]:
            tbl.remove(tr)
        for cells in rows_data:
            tr = copy.deepcopy(tpl)
            for tc, text in zip(tr.tc_lst, cells):
                cell_fill(tc, text, parent)
            tbl.append(tr)

    rebuild(doc.tables[2], [[n, t, page(k)] for n, t, k in toc])
    rebuild(doc.tables[3], [[n, t.upper() if False else t, page(c)] for n, t, c in figs])
    rebuild(doc.tables[4], [[n, t, page(c)] for n, t, c in tabs])

    # page numbers in the footers (the format file has none)
    def footer_page(section, fmt_roman):
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
                t = OxmlElement("w:t"); t.text = "1"; r.append(t)
            else:
                it = OxmlElement("w:instrText"); it.set("{http://www.w3.org/XML/1998/namespace}space", "preserve"); it.text = " PAGE "; r.append(it)
            par._p.append(r)

    footer_page(doc.sections[0], True)
    footer_page(doc.sections[1], False)

    doc.core_properties.title = nc.TITLE
    doc.core_properties.author = ", ".join(n for n, _ in NAMES)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out))
    keys = [k for *_, k in toc] + [c for *_, c in figs] + [c for *_, c in tabs]
    (out.parent / (out.stem + "_keys.json")).write_text(json.dumps(keys), encoding="utf-8")
    print("wrote", out, "| chapters", len(toc), "| figures", len(figs), "| tables", len(tabs))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=r"C:\Users\Divyanshi\Downloads\NNDL_Case_Study_Report_Format.docx")
    ap.add_argument("--out", default=r"C:\Users\Divyanshi\Downloads\NNDL_Case_Study_Report.docx")
    ap.add_argument("--pages", default=None)
    a = ap.parse_args()
    build(Path(a.base), Path(a.out), json.load(open(a.pages, encoding="utf-8-sig")) if a.pages else None)
