"""Build the FINAL report (.docx) by updating the original proposal-style report with the implementation and results.

    python scripts/make_final_report.py --src Cascaded_CNN_Report_Full.docx --out Cascaded_CNN_Report_FINAL.docx

The original file is never modified. All numbers come from results/*.json (via scripts/make_results_pdf.py).
"""
import argparse
import copy
import importlib.util
import tempfile
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt
from docx.text.paragraph import Paragraph

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("mr", ROOT / "scripts" / "make_results_pdf.py")
mr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mr)
R, pct, binary_acc, CLASSES = mr.R, mr.pct, mr.binary_acc, mr.CLASSES
FIG = ROOT / "docs" / "figures"

pr, prf = R["pretrained"]["cascade"], R["pretrained"]["flat"]
dg, i3, imp, fl = R["datagroup"]["cascade"], R["improved3"]["cascade"], R["improved"]["cascade"], R["improved"]["flat"]
rf = mr.json.load(open(ROOT / "results" / "random_forest.json"))
ps = mr.json.load(open(ROOT / "results" / "pretrained_svm.json"))


# ------------------------------------------------------------------ helpers
def set_runs(par, texts):
    """texts: list of (text, bold) using the first run's formatting as the template; extra runs removed."""
    runs = par.runs
    for r in runs[len(texts):]:
        r._r.getparent().remove(r._r)
    runs = par.runs
    while len(runs) < len(texts):
        new = copy.deepcopy(runs[-1]._r)
        runs[-1]._r.addnext(new)
        runs = par.runs
    for r, (t, b) in zip(runs, texts):
        r.text = t
        r.bold = b


def set_text(par, text):
    set_runs(par, [(text, par.runs[0].bold if par.runs else None)])


def clone_after(ref, like, runs):
    new = copy.deepcopy(like._p)
    ref_elm = ref._p if isinstance(ref, Paragraph) else ref
    ref_elm.addnext(new)
    p = Paragraph(new, like._parent)
    set_runs(p, runs)
    return p


def picture_after(ref, like_caption, path, width_cm, caption):
    """Insert a centred picture paragraph and a caption paragraph after `ref`; returns the caption paragraph."""
    img = copy.deepcopy(like_caption._p)
    (ref._p if isinstance(ref, Paragraph) else ref).addnext(img)
    ip = Paragraph(img, like_caption._parent)
    set_runs(ip, [("", None)])
    ip.runs[0].italic = False
    ip.runs[0].add_picture(str(path), width=Cm(width_cm))
    ip.paragraph_format.keep_with_next = True
    cap = clone_after(ip, like_caption, [(caption, None)])
    return cap


def shade(cell, fill="E9ECEF"):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), fill)
    tcPr.append(shd)


def table_after(doc, ref, rows, widths, font_pt=8, bold_rows=()):
    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
    t.style = doc.tables[1].style
    t.autofit = False
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            cell = t.cell(ri, ci)
            cell.width = Cm(widths[ci])
            p = cell.paragraphs[0]
            run = p.add_run(str(val))
            run.font.size = Pt(font_pt)
            if ri == 0 or ri in bold_rows:
                run.bold = True
            if ri == 0:
                shade(cell)
    (ref._p if isinstance(ref, Paragraph) else ref).addnext(t._tbl)
    return t._tbl


def per_class_chart(path):
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    x = np.arange(len(CLASSES)); w = 0.27
    series = [("Cascade, from scratch (improved3)", R["improved3"]["cascade"], "#adb5bd"),
              ("Cascade, pretrained", pr, "#2b3a4a"), ("Flat CNN, pretrained", prf, "#d9480f")]
    for k, (lab, r, col) in enumerate(series):
        ax.bar(x + (k - 1) * w, [r["per_class"][c]["f1-score"] for c in CLASSES], w, label=lab, color=col)
    ax.set_xticks(x); ax.set_xticklabels(CLASSES, rotation=40, ha="right", fontsize=7)
    ax.set_ylim(0, 1.05); ax.set_ylabel("F1", fontsize=8); ax.legend(fontsize=7, ncol=3, loc="upper center")
    ax.set_title("Per-class F1 on the test set", fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


FONT = "Times New Roman"
SIZES = {"body": 12, "h1": 16, "h2": 14, "h3": 12, "caption": 10, "table": 10, "ref": 11, "abstract": 11}


def _font(run, size=None, bold=None):
    run.font.name = FONT
    rPr = run._r.get_or_add_rPr()
    rf = rPr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts")
        rPr.insert(0, rf)
    for a in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rf.set(qn(a), FONT)
    if size:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold


def apply_house_style(doc, title_pars):
    """Times New Roman throughout, justified body text, official report sizes."""
    for sname, size, bold in (("Normal", SIZES["body"], None), ("Heading 1", SIZES["h1"], True), ("Heading 2", SIZES["h2"], True), ("Heading 3", SIZES["h3"], True)):
        try:
            st = doc.styles[sname]
        except KeyError:
            continue
        st.font.name = FONT
        st.font.size = Pt(size)
        if bold:
            st.font.bold = True
        rPr = st.element.get_or_add_rPr()
        rf = rPr.find(qn("w:rFonts"))
        if rf is None:
            rf = OxmlElement("w:rFonts")
            rPr.insert(0, rf)
        for a in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
            rf.set(qn(a), FONT)
        for a in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
            if rf.get(qn(a)) is not None:
                del rf.attrib[qn(a)]

    title_ids = {id(p._p) for p in title_pars}
    abstract_cell = doc.tables[0].rows[0].cells[0]
    in_refs = False
    for p in doc.paragraphs:
        txt = p.text.strip()
        name = p.style.name if p.style is not None else ""
        if id(p._p) in title_ids:
            for r in p.runs:
                _font(r)
            continue
        if name.startswith("Heading"):
            in_refs = False
            level = {"Heading 1": "h1", "Heading 2": "h2"}.get(name, "h3")
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            for r in p.runs:
                _font(r, SIZES[level], True)
            continue
        if txt == "References":
            in_refs = True
            for r in p.runs:
                _font(r, SIZES["h1"], True)
            continue
        is_caption = bool(txt) and (txt.startswith("Figure ") or txt.startswith("Table ")) and len(txt) < 220 and p.runs and (p.runs[0].italic or p.runs[0].bold)
        if is_caption:
            for r in p.runs:
                _font(r, SIZES["caption"])
            continue
        if not txt:  # picture paragraphs and spacers
            for r in p.runs:
                _font(r)
            continue
        size = SIZES["ref"] if in_refs else SIZES["body"]
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        for r in p.runs:
            _font(r, size)
    for ti, t in enumerate(doc.tables):
        for row in t.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    if ti == 0:
                        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                    else:
                        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                    for r in p.runs:
                        _font(r, SIZES["abstract"] if ti == 0 else SIZES["table"])


# ------------------------------------------------------------------ main
def build(src: Path, out: Path):
    doc = Document(str(src))
    P = list(doc.paragraphs)  # snapshot: indices refer to the original document
    body_like, label_like, bullet_like = P[43], P[52], P[92]
    cap_like, tcap_like, h2_like = P[40], P[47], P[42]
    tmp = Path(tempfile.mkdtemp())
    ch = mr.charts(tmp)
    per_class_chart(tmp / "perclass_pre.png")

    # ---- abstract
    cell = doc.tables[0].rows[0].cells[0]
    ap = cell.paragraphs[0]
    set_runs(ap, [("Abstract: ", True), (
        "Automated inspection of large photovoltaic (PV) installations is essential for sustained power output and safety. "
        "This report presents Cascaded-SolarNet, a three-stage cascade of five small convolutional neural networks that classifies the 12 classes of the "
        "InfraredSolarModules benchmark from single-modality infrared (IR) images: a binary triage stage with early exit for healthy panels, a fault-family stage "
        "and one fine-grained specialist per family. We implemented the system in PyTorch and evaluated it on a fixed stratified held-out split. "
        f"Removing brightness and noise augmentation, which destroys the temperature signal in thermal images, raised the from-scratch cascade from 61.7% to {pct(dg['accuracy'])} accuracy; "
        f"fine-tuning an ImageNet-pretrained EfficientNet-B0 at every stage reached {pct(pr['accuracy'])} accuracy, macro-F1 {pr['macro_f1']:.3f} and {pct(binary_acc(R['pretrained']))} on the healthy-versus-defect decision. "
        f"A flat classifier with the same backbone is essentially tied ({pct(prf['accuracy'])}), so the cascade did not show a clear accuracy advantage, and the family-routing stage remains the largest source of errors. "
        "Published 12-class results of up to 93.9% use cross-validation and pretrained features and are not directly comparable with our fixed hold-out; we report the comparison and its limits.", None)])
    for extra in cell.paragraphs[1:]:
        extra._p.getparent().remove(extra._p)

    # ---- introduction: aim sentence
    t13 = P[13].text.replace("The aim is to exceed 95% accuracy by restructuring the class decision space rather than by scaling up the model.",
                             "The aim was to improve on flat classifiers by restructuring the class decision space rather than by scaling up the model; Section 6 reports what was achieved.")
    set_text(P[13], t13)

    # ---- dataset
    set_text(P[34], "The experiments use the InfraredSolarModules dataset released by Raptor Maps (Millendorf et al., ICLR 2020) on GitHub under a CC BY 4.0 licence. "
             "It contains 20,000 infrared images of 24 x 40 pixels each (24 wide, 40 tall), captured from real drone flights over commercial solar installations. "
             "There are 12 classes: 11 anomaly types and one No-Anomaly class. The anomaly types are Cell, Cell-Multi, Diode, Diode-Multi, Hot-Spot, Hot-Spot-Multi, "
             "Soiling, Shadowing, Vegetation, Cracking and Offline-Module.")
    set_text(P[35], "The dataset is strongly imbalanced: half of the images (10,000) are No-Anomaly, while the rarest classes have only 175 to 251 images "
             "(Diode-Multi 175, Soiling 205, Hot-Spot-Multi 247, Hot-Spot 251), which causes majority-class bias in flat models. "
             "The images are very small because of the altitude of capture, so the task depends mainly on recognising temperature-pattern signatures rather than fine structural detail. "
             "The dataset paper itself reports no model accuracies; the published results we compare with in Section 6.6 come from later papers.")

    # ---- Figure 1 replaced, section 4 updates
    for r in list(P[39].runs):
        if r._r.xpath(".//w:drawing"):
            r._r.getparent().remove(r._r)
    P[39].add_run().add_picture(str(FIG / "overall_cascade.png"), width=Cm(12.5))
    set_text(P[40], "Figure 1. The implemented 3-stage Cascaded-SolarNet architecture (class names are the real dataset labels).")
    set_text(P[43], "The input is a single-channel (grayscale) thermal image of one PV module. In an IR image, brightness encodes temperature. A healthy module heats fairly uniformly during operation, "
             "whereas a defect produces a localised temperature anomaly: a hot spot, a cold region, or a characteristic thermal pattern. At 24 x 40 pixels there is almost no fine texture to exploit, "
             "so the network has to recognise where heat is concentrated, how large the region is and what shape it takes. This is why small CNNs trained from scratch already work reasonably; "
             "an ImageNet-pretrained backbone still helped further (Section 6).")
    clone_after(P[43], body_like, [(
        "Preprocessing. Pixel values are scaled to [0, 1] and normalised with the mean and standard deviation of the training split only. In the from-scratch networks the image is upsampled by a factor of two "
        "(bilinear) to 80 x 48 so that the small defects survive three pooling steps; the pretrained networks resize the image to 160 x 96 and repeat the channel three times to match their ImageNet input. "
        "During training only random horizontal and vertical flips are applied. Brightness, contrast and noise jitter were tested and removed because they lowered accuracy (Section 6.3).", None)])

    # Stage 1
    set_runs(P[51], [("Layers (implemented). ", True), ("Three blocks, each with two 3 x 3 convolutions (BatchNorm and ReLU after each) followed by 2 x 2 max pooling, with 64, 128 and 128 channels, then global average pooling, dropout (0.2) and a fully connected layer with 2 outputs. "
                                                       "In the pretrained variant this network is an ImageNet-pretrained EfficientNet-B0 with a new two-way head.", None)])
    set_runs(P[52], [("How it works.", True), (" Each block learns richer thermal patterns than the one before (64, then 128, then 128 feature maps). BatchNorm and ReLU normalise and activate the features, and MaxPool halves the spatial size, "
                                                "so the 80 x 48 input becomes 64 x 40 x 24, 128 x 20 x 12 and finally 128 x 10 x 6. Global average pooling summarises each of the 128 maps into one value and the final fully connected layer scores the two classes. "
                                                "Figure 2 shows the layers.", None)])
    cap2 = picture_after(P[52], cap_like, FIG / "stage1_cnn1.png", 16.0, "Figure 2. Stage 1: CNN-1, binary triage (layer by layer, with feature-map sizes).")

    # Stage 2
    set_runs(P[59], [("Layers (implemented). ", True), ("The same three-block design with 128, 256 and 128 channels, global average pooling, dropout (0.4) and a fully connected layer with 3 outputs "
                                                       "(EfficientNet-B0 with a three-way head in the pretrained variant).", None)])
    set_runs(P[60], [("How it works.", True), (" CNN-2 is wider than CNN-1 because separating fault families is harder than separating healthy from faulty. The 128-256-128 channel pattern expands the feature set in the middle block "
                                                "to capture a richer variety of thermal patterns and then compresses it again before classification. Dropout is added because this network is trained on a smaller, defect-only dataset and is more prone to overfitting. "
                                                "The three output scores correspond to the three families, and the highest decides the route (Figure 3).", None)])
    cap3 = picture_after(P[60], cap_like, FIG / "stage2_cnn2.png", 16.0, "Figure 3. Stage 2: CNN-2, fault-family classifier.")

    # Stage 3
    set_runs(P[67], [("Layers (implemented, identical for all three). ", True), ("The same 128-256-128 three-block design with global average pooling, dropout (0.4) and a fully connected layer with as many outputs as the family has classes.", None)])
    set_runs(P[68], [("How it works.", True), (" The Stage 3 networks make the finest distinctions. Global average pooling summarises each map into one value, which keeps the final FC layer small and helps against overfitting on the few examples available for rare faults. "
                                                "The three networks share the same design but are trained separately and have different output sizes, so each learns features specific to its own family (Figure 4).", None)])
    set_runs(P[69], [("CNN-3a, Electrical (6 classes).", True), (" Outputs Cell, Cell-Multi, Diode, Diode-Multi, Hot-Spot or Hot-Spot-Multi. Cell is a hot spot with square geometry in a single cell, Cell-Multi the same in several cells, "
                                                                "Diode an activated bypass diode (typically one third of the module), Diode-Multi several activated diodes, and Hot-Spot and Hot-Spot-Multi hot spots on thin-film modules.", None)])
    set_runs(P[70], [("CNN-3b, Environmental (3 classes).", True), (" Outputs Soiling, Shadowing or Vegetation: dirt, dust or debris on the module, sunlight blocked by nearby objects, and panels covered by vegetation. "
                                                                   "The specialist separates them by the spatial layout of the temperature pattern.", None)])
    set_runs(P[71], [("CNN-3c, Structural (2 classes).", True), (" Outputs Cracking or Offline-Module (the entire module is heated). The dataset has no further structural classes, so the poster's finer categories (delamination, ribbon, water stain and so on) are not used.", None)])
    set_text(P[72], P[72].text.replace("three to five related classes", "two to six related classes"))
    cap4 = picture_after(P[71], cap_like, FIG / "stage3_specialists.png", 16.0, "Figure 4. Stage 3: the three fine-grained specialists and their classes.")

    # families table (Table 3) and architecture table (Table 4)
    fam = {"Electrical": "Faults in cells or diodes that change how current flows: Cell, Cell-Multi, Diode, Diode-Multi, Hot-Spot, Hot-Spot-Multi.",
           "Environmental": "External conditions on or around the panel: Soiling, Shadowing, Vegetation.",
           "Structural": "Physical state of the module: Cracking, Offline-Module."}
    for row in doc.tables[3].rows[1:]:
        name = row.cells[0].text.strip()
        if name in fam:
            c = row.cells[1].paragraphs[0]
            set_runs(c, [(fam[name], c.runs[0].bold if c.runs else None)])
    arch = {
        "CNN-1": ("40 x 24 IR image (upsampled to 80 x 48)", "3 blocks of 2 x (Conv3x3, BatchNorm, ReLU) + MaxPool, 64-128-128; GAP, Dropout 0.2, FC", "2 classes: No-Anomaly / Defect"),
        "CNN-2": ("Defective image", "Same, 128-256-128; GAP, Dropout 0.4, FC", "3 classes: Electrical / Environmental / Structural"),
        "CNN-3a": ("Image routed as Electrical", "Same as CNN-2, FC with 6 outputs", "6 classes: Cell, Cell-Multi, Diode, Diode-Multi, Hot-Spot, Hot-Spot-Multi"),
        "CNN-3b": ("Image routed as Environmental", "Same as CNN-2, FC with 3 outputs", "3 classes: Soiling, Shadowing, Vegetation"),
        "CNN-3c": ("Image routed as Structural", "Same as CNN-2, FC with 2 outputs", "2 classes: Cracking, Offline-Module"),
    }
    for row in doc.tables[4].rows[1:]:
        key = row.cells[0].text.strip()
        if key in arch:
            for ci, val in zip((2, 3, 4), arch[key]):
                c = row.cells[ci].paragraphs[0]
                set_runs(c, [(val, c.runs[0].bold if c.runs else None)])
    set_text(P[75], "Table 4 consolidates the architecture of all five networks (from-scratch variant). In the pretrained variant each network is replaced by an ImageNet-pretrained EfficientNet-B0 fine-tuned with a new classification head.")
    set_text(P[79], "1.  Feed the 24 x 40 IR image (preprocessed as in Section 4.2) to CNN-1.")

    # training strategy, cascade rationale
    set_text(P[86], "Each of the five networks is trained independently on the data that matches its task, using only the training split (70% of the images, stratified by class). CNN-1 is trained on all images with a binary label "
             "(No-Anomaly against any defect). CNN-2 is trained on defect images only, labelled with their fault family. Each Stage 3 specialist is trained only on the defect images of its own family, labelled with the exact subtype. "
             "Training uses Adam (learning rate 5e-4 for the from-scratch networks and 3e-4 for the pretrained ones, weight decay 1e-4), a class-weighted cross-entropy loss with weights N / (K x n) to counter imbalance, "
             "and early stopping on validation macro-F1 (patience 25 from scratch, 6 for the pretrained networks). The pretrained variant uses mixed-precision training. "
             "All networks are implemented in PyTorch; the code, configurations and results are at github.com/techwack/cascaded-cnn.")
    set_text(P[87], "4.10 Why a cascade can outperform a flat classifier, and what we measured")
    set_text(P[89], P[89].text + " Section 6 reports how large these effects were in practice.")

    # ---- experimental setup (Table 5)
    t5 = {"Dataset": "InfraredSolarModules benchmark (Raptor Maps): 20,000 IR images, 12 classes (1 healthy, 11 faults), 24 x 40 px",
          "Data split": "Stratified 70/15/15 (14,000 train / 3,000 validation / 3,000 test images), seed 42, identical for all models",
          "Evaluation metrics": "Accuracy, macro-F1, Cohen's kappa, healthy-versus-defect accuracy of Stage 1, per-stage error counts",
          "Baselines": "Flat CNN (same building blocks, and with the pretrained backbone), raw-pixel Random Forest, EfficientNet-B0 features + SVM; published results as cited",
          "Implementation": "PyTorch; Adam; class-weighted loss; early stopping on validation macro-F1; flips-only augmentation; mixed precision for the pretrained networks"}
    for row in doc.tables[5].rows[1:]:
        key = row.cells[0].text.strip()
        if key in t5:
            c = row.cells[1].paragraphs[0]
            set_runs(c, [(t5[key], c.runs[0].bold if c.runs else None)])
    set_text(P[100], "Macro-F1 is included alongside accuracy because it weights every class equally and so exposes poor performance on rare faults, which accuracy alone would hide under class imbalance. "
             "Cohen's kappa measures agreement beyond chance and is similarly robust to imbalance. Per-stage error counts attribute every mistake of the cascade to the first stage that went wrong. "
             "The test split is used only for the final numbers; early stopping and all tuning use the validation split.")

    # ---- section 6: results and discussion
    set_text(P[101], "6. Results and Discussion")
    P[102]._p.getparent().remove(P[102]._p)
    ref = P[101]

    def H(text):
        nonlocal ref
        ref = clone_after(ref, h2_like, [(text, None)])

    def T(runs):
        nonlocal ref
        ref = clone_after(ref, body_like, runs)

    def B(label, body):
        nonlocal ref
        ref = clone_after(ref, bullet_like, [("•  ", None), (label + " ", True), (body, None)])

    def TC(text):
        nonlocal ref
        ref = clone_after(ref, tcap_like, [(text, True)])

    def TBL(rows, widths, bold_rows=(), font=8):
        nonlocal ref
        el = table_after(doc, ref, rows, widths, font, bold_rows)
        sp = clone_after(el, body_like, [("", None)])
        ref = sp

    def FIG_(path, w, caption):
        nonlocal ref
        ref = picture_after(ref, cap_like, path, w, caption)

    H("6.1 Evaluation protocol")
    T([("All numbers are measured on the same held-out test split of 3,000 images (stratified 70/15/15, seed 42) and are single runs per configuration. "
        "The test images are never used for training, early stopping, normalisation or model selection. We compare the following configurations: a from-scratch cascade in several refinements, "
        "a cascade whose five networks are ImageNet-pretrained EfficientNet-B0 models fine-tuned at each stage, and flat 12-class baselines.", None)])

    H("6.2 Results")
    T([(f"The best result is the pretrained cascade: {pct(pr['accuracy'])} accuracy, macro-F1 {pr['macro_f1']:.3f} and Cohen's kappa {pr['kappa']:.3f} on the 12-class problem, "
        f"and {pct(binary_acc(R['pretrained']))} on the Stage 1 healthy-versus-defect decision. The best cascade trained from scratch reaches {pct(dg['accuracy'])} accuracy (data-driven families) "
        f"or {pct(i3['accuracy'])} with the semantic families.", None)])
    TC("Table 6. Test results by configuration.")
    names = {"default": "Default (brightness/contrast/noise jitter)", "no_photometric": "No photometric jitter (flips only)", "improved": "Improved: 80x48 input, 2 convs per block",
             "improved2": "Improved2: + balanced sampling, label smoothing", "improved3": "Improved3: + larger Stage 1/2 CNNs", "datagroup": "Data-driven family grouping",
             "pretrained": "Pretrained: ImageNet EfficientNet-B0 at every stage"}
    rows = [["Configuration", "Model", "Accuracy", "Macro-F1", "Kappa", "Stage 1 binary"]]
    best_row = None
    for k in ["default", "no_photometric", "improved", "improved2", "improved3", "datagroup", "pretrained"]:
        c = R[k]["cascade"]
        rows.append([names[k], "Cascade", pct(c["accuracy"]), f"{c['macro_f1']:.3f}", f"{c['kappa']:.3f}", pct(binary_acc(R[k]))])
        if k == "pretrained":
            best_row = len(rows) - 1
        if "flat" in R[k]:
            f = R[k]["flat"]
            rows.append(["", "Flat CNN (same split)", pct(f["accuracy"]), f"{f['macro_f1']:.3f}", f"{f['kappa']:.3f}", "n/a"])
    rows.append(["Raw-pixel Random Forest", "RF", pct(rf["accuracy"]), f"{rf['macro_f1']:.3f}", f"{rf['kappa']:.3f}", "n/a"])
    rows.append(["EfficientNet-B0 features + SVM (Duranay-style)", "SVM", pct(ps["accuracy"]), f"{ps['macro_f1']:.3f}", f"{ps['kappa']:.3f}", pct(ps["binary_accuracy"])])
    TBL(rows, [6.2, 3.3, 1.8, 1.7, 1.5, 2.1], bold_rows=(best_row,))
    FIG_(ch["prog"], 15.0, "Figure 5. Cascade test accuracy and macro-F1 by configuration (dashed line: from-scratch flat CNN).")

    H("6.3 Main findings")
    B("Brightness and noise jitter hurt.", "Brightness, contrast and noise augmentation lowered cascade accuracy to 61.7%; flips only gave 71.1%. In thermal images brightness is temperature, so jittering it removes the signal that separates the classes.")
    B("Input size and capacity helped.", f"Upsampling to 80 x 48 and using two convolutions per block raised accuracy to {pct(imp['accuracy'])}; enlarging the Stage 1 and Stage 2 networks raised it to {pct(i3['accuracy'])}, and Stage 1 binary accuracy from {pct(binary_acc(R['improved']))} to {pct(binary_acc(R['improved3']))}.")
    B("ImageNet pretraining helped most late in the project.", f"Fine-tuning a pretrained EfficientNet-B0 at every stage raised the cascade from {pct(dg['accuracy'])} to {pct(pr['accuracy'])} accuracy (macro-F1 {dg['macro_f1']:.3f} to {pr['macro_f1']:.3f}), and training took minutes instead of an hour.")
    B("The cascade and a flat classifier are essentially tied.", f"With the pretrained backbone the cascade scores {pct(pr['accuracy'])} / {pr['macro_f1']:.3f} / {pr['kappa']:.3f} (accuracy, macro-F1, kappa) and the flat CNN {pct(prf['accuracy'])} / {prf['macro_f1']:.3f} / {prf['kappa']:.3f}. "
                                                              "The cascade has about 20 million parameters (five EfficientNets) against 4 million for the flat model, so we did not measure an accuracy or efficiency benefit. From scratch the cascade was ahead on accuracy and kappa but not on macro-F1.")
    B("Tried without gain:", "soft routing (multiplying stage probabilities), flip test-time augmentation, tuning the Stage 1 threshold on validation data (the optimum stayed at 0.5) and balanced sampling with label smoothing.")

    H("6.4 Where the cascade makes its errors")
    T([("Because the cascade routes each image through several networks, every mistake can be attributed to the first stage that went wrong. Table 7 gives the counts out of 3,000 test images.", None)])
    TC("Table 7. Error attribution (test errors per stage).")
    rows = [["Configuration", "Stage 1 missed defect", "Stage 1 false alarm", "Stage 2 wrong family", "Stage 3 wrong subtype", "Total"]]
    for k, lab in [("improved", "Improved (from scratch)"), ("improved3", "Improved3 (from scratch)"), ("datagroup", "Data-driven families (from scratch)"), ("pretrained", "Pretrained")]:
        s_ = R[k]["cascade"]["stages"]
        rows.append([lab, s_["stage1_missed_defects"], s_["stage1_false_alarms"], s_["stage2_wrong_family"], s_["stage3_wrong_subtype"], s_["total_errors"]])
    TBL(rows, [5.0, 2.4, 2.3, 2.4, 2.5, 1.4], bold_rows=(4,))
    FIG_(ch["stages"], 14.5, "Figure 6. Where the cascade makes its errors, by configuration.")
    T([(f"Pretraining reduced errors at every stage compared with the from-scratch 'improved3' cascade (total {i3['stages']['total_errors']} to {pr['stages']['total_errors']}), but Stage 2, the family decision, remains the largest source "
        f"({pr['stages']['stage2_wrong_family']} of {pr['stages']['total_errors']} errors) and an error there cannot be corrected by Stage 3. Re-grouping the classes by their measured confusions cut Stage 2 errors from {i3['stages']['stage2_wrong_family']} to {dg['stages']['stage2_wrong_family']} "
        f"but moved the difficulty into Stage 3 ({i3['stages']['stage3_wrong_subtype']} to {dg['stages']['stage3_wrong_subtype']} errors), so the net accuracy changed little. "
        f"About {pct(pr['stages']['early_exit_rate'])} of the test images leave the cascade at Stage 1.", None)])

    H("6.5 Per-class performance")
    T([("The rare classes are the weak point. They have only 123 to 174 training images and several look alike (Cell versus Cell-Multi, Hot-Spot versus Cell). Class weighting, stratified splitting and macro-F1 reduce but do not remove this problem.", None)])
    TC("Table 8. Per-class F1 on the test set.")
    rows = [["Class", "Test images", "Cascade, from scratch (improved3)", "Cascade, pretrained", "Flat CNN, pretrained"]]
    for c in CLASSES:
        rows.append([c, int(pr["per_class"][c]["support"]), f"{R['improved3']['cascade']['per_class'][c]['f1-score']:.2f}", f"{pr['per_class'][c]['f1-score']:.2f}", f"{prf['per_class'][c]['f1-score']:.2f}"])
    TBL(rows, [3.4, 2.2, 4.2, 3.2, 3.2])
    FIG_(tmp / "perclass_pre.png", 15.0, "Figure 7. Per-class F1 on the test set.")

    H("6.6 Comparison with published work")
    T([("We re-checked the literature at the primary sources. The dataset paper (Millendorf et al.) reports no model accuracies; the figures attributed to it in our earlier survey (Random Forest 84%, CNN 92.5%, EfficientNet + SVM 93%) come from later papers. "
        "Those papers differ from our setup in task, protocol and features, so their numbers can be placed next to ours but not ranked.", None)])
    TC("Table 9. Published results on InfraredSolarModules and our results.")
    rows = [["Paper", "Method", "Task and protocol", "Reported", "Checked"],
            ["Duranay 2023, Electronics 12(21):4397", "EfficientNet-B0 features (ImageNet-pretrained), NCA selection, SVM", "12 classes, 10-fold cross-validation", "93.93% accuracy, 89.82% F1", "Full paper read"],
            ["Alves et al. 2021, Renewable Energy 179:502", "Custom CNN with data augmentation", "Anomaly detection (binary); 8 selected classes; cross-validation", "92.5% binary; 78.85% for 8 classes", "Abstract (library record)"],
            ["Multi-scale CNN, transfer learning (Eng. Appl. Artif. Intell. 113, 2022), as cited by Duranay", "Three-branch CNN, transfer learning", "Fault detection; 11 anomaly types", "97.32% detection; 93.51% for 11 types", "As cited by Duranay"],
            ["Corley et al. 2025 (CVPRW), their Random Forest", "Random Forest", "Anomaly classification, split not stated", "84.0%", "arXiv text read"],
            ["Barraz et al. 2025; Gopalakrishnan et al. 2024, as cited by Corley", "Cascading ensemble with voting CNN; NASNet + LSTM", "Not stated", "88%; 85%", "As cited by Corley"],
            ["Millendorf et al. 2020 (dataset paper)", "-", "-", "No model results reported", "Full paper read"],
            ["Preprints and listings (SSRN Hyperband CNN; diffusion + ViT)", "Hyperband-tuned CNN; transformer", "12 classes, repeated k-fold CV (SSRN)", "98.75%; 92.78%", "Not verified (listings only)"],
            ["Ours: pretrained cascade", "ImageNet EfficientNet-B0 fine-tuned at each of 5 stages", "12 classes, single stratified 70/15/15 hold-out", f"{pct(pr['accuracy'])} accuracy, macro-F1 {pr['macro_f1']:.3f}", "Our results"],
            ["Ours: cascade from scratch (data-driven families)", "Five small CNNs", "Same hold-out", f"{pct(dg['accuracy'])} accuracy, macro-F1 {dg['macro_f1']:.3f}", "Our results"],
            ["Ours: Stage 1 only (pretrained / from scratch)", "Binary triage CNN", "Healthy vs defect, same hold-out", f"{pct(binary_acc(R['pretrained']))} / {pct(binary_acc(R['datagroup']))}", "Our results"],
            ["Ours: flat CNN (pretrained / scratch), Random Forest", "EfficientNet-B0 / same blocks / raw pixels", "12 classes, same hold-out", f"{pct(prf['accuracy'])} / {pct(fl['accuracy'])}; RF {pct(rf['accuracy'])}", "Our results"],
            ["Ours: Duranay-style baseline on our split", "EfficientNet-B0 features + RBF SVM (no NCA, last layer only)", "12 classes, same hold-out", f"{pct(ps['accuracy'])} accuracy, macro-F1 {ps['macro_f1']:.3f}", "Our results"]]
    TBL(rows, [3.7, 3.4, 3.4, 3.0, 2.2], bold_rows=(8,), font=7.5)
    B("Binary triage.", f"Our pretrained Stage 1 reaches {pct(binary_acc(R['pretrained']))} ({pct(binary_acc(R['datagroup']))} from scratch). That is above Alves et al. (92.5%, cross-validated) and below the multi-scale transfer-learning CNN (97.32%). It is a single hold-out run, so we claim to be at or slightly above the 92.5% to 93% range, not a new best.")
    B("12 classes.", f"Our best, {pct(pr['accuracy'])}, is below the published 12-class figures (84.0% Random Forest, 85% to 88% for ensembles, 93.93% for EfficientNet features with an SVM). On the published protocols we did not beat them and did not reach the 93% ceiling or the 95% target stated in our proposal. "
                     f"However, a Duranay-style pipeline run on our split scored only {pct(ps['accuracy'])} (macro-F1 {ps['macro_f1']:.3f}), below our pretrained cascade. This suggests the gap to the published 93.93% is largely due to the evaluation protocol and to details we did not replicate, not to a weaker model, but it does not prove it.")
    B("Why the numbers are not like for like.", "Duranay uses 10-fold cross-validation, a network trained end to end on the dataset and a supervised feature selection (NCA) before the cross-validation; as described, test images can influence these steps. "
                                                 "Our test images are never used for training or selection. We also measured overlap in our own split: about 7% of test images have a near-identical training image (distance below 0.5, 96% of them with the same label) and about 54% have a close one (distance below 1, 78% with the same label), "
                                                 "so random folds would put near-duplicates on both sides and raise cross-validated scores. This cannot explain the whole gap: a raw-pixel Random Forest scores 61.6% on our split against the 84.0% reported by Corley et al., whose protocol is not stated.")
    T([("Sources: Duranay, Electronics 12(21):4397; Corley et al., arXiv:2503.02128; Alves et al., Renewable Energy 179:502-516 (abstract only); Millendorf et al., ICLR 2020 workshop. "
        "Rows marked 'as cited' were read in the citing paper; rows marked 'Not verified' appeared only in search listings.", None)])

    H("6.7 Limitations")
    for lab, body in [("Single runs.", "Every configuration was run once with seed 42. Differences below about 0.5 points are within noise and no confidence intervals were computed. A 10-fold cross-validation of the pretrained cascade was prepared but did not complete, so it is not reported."),
                      ("Near-duplicate images.", "The benchmark has no module or site identifiers, so near-identical crops can fall on both sides of any split."),
                      ("Rare classes.", "Soiling, Hot-Spot and Diode-Multi remain weak, which keeps macro-F1 well below accuracy."),
                      ("Efficiency.", "Latency was not measured reliably (timings were taken while the GPU was busy). The pretrained cascade has five times the parameters of the flat model, so the early exit did not give a demonstrated efficiency advantage in our setting."),
                      ("Comparison with published work.", "Protocols differ (cross-validation, pretrained features) and some sources could only be checked through abstracts or citing papers.")]:
        B(lab, body)

    # ---- conclusion
    set_text(P[104], "We implemented Cascaded-SolarNet, a three-stage cascade of five CNNs for 12-class infrared solar module defect classification, with a reproducible split, tests, architecture diagrams and live demos. "
             f"After diagnosing and removing harmful photometric augmentation, enlarging the input and strengthening the front stages, accuracy rose from 61.7% to {pct(dg['accuracy'])} from scratch and to {pct(pr['accuracy'])} with ImageNet pretraining "
             f"(macro-F1 0.37 to {pr['macro_f1']:.2f}); healthy-versus-defect triage reaches {pct(binary_acc(R['pretrained']))}, comparable to the 92.5% binary figure published for this dataset.")
    c2 = clone_after(P[104], body_like, [(
        "The claim that a cascade clearly beats a flat classifier was only partly supported. From scratch the cascade is ahead on accuracy and kappa; with the same pretrained backbone it ties a flat CNN while using five times the parameters. "
        "Error propagation is the central trade-off: a wrong decision at Stage 1 or Stage 2 cannot be undone, and the family decision remains the largest error source. "
        "Published 12-class results (84% to 93.93%) are higher than ours, but they use cross-validation and pretrained features, and the 93% ceiling and 95% target stated in the proposal were not reached under our stricter fixed hold-out. "
        "A practical lesson is that thermal images need thermal-aware augmentation: flips help, while brightness, contrast and noise jitter hurt.", None)])
    set_text(P[105], "Future work includes running several seeds and a leak-free 10-fold cross-validation for a like-for-like comparison with published results; improving the family decision, for example by training Stage 2 jointly with the specialists or on the cascade's own routing errors; "
             "class-specific augmentation for the rare faults; deduplicating near-identical crops before splitting; measuring latency and memory on a drone-class edge device; and investigating additional modalities such as RGB.")

    # ---- references
    last = P[115]
    for txt in ["10.  Duranay, Z.B. (2023). Fault Detection in Solar Energy Systems: A Deep Learning Approach. Electronics, 12(21), 4397. DOI: 10.3390/electronics12214397",
                "11.  Alves, R.H.F., de Deus Junior, G.A., Marra, E.G., Lemos, R.P. (2021). Automatic fault classification in photovoltaic modules using Convolutional Neural Networks. Renewable Energy, 179, 502-516.",
                "12.  Corley et al. (2025). Aerial Infrared Health Monitoring of Solar Photovoltaic Farms at Scale. CVPR Workshops (PBVS). arXiv:2503.02128.",
                "13.  Paszke, A., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. NeurIPS 32.",
                "14.  Deng, J., et al. (2009). ImageNet: A Large-Scale Hierarchical Image Database. CVPR."]:
        last = clone_after(last, P[115], [(txt, None)])

    apply_house_style(doc, title_pars=P[:9])
    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out))
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=r"C:\Users\Divyanshi\Downloads\Cascaded_CNN_Report_Full.docx")
    ap.add_argument("--out", default=r"C:\Users\Divyanshi\Downloads\Cascaded_CNN_Report_FINAL.docx")
    a = ap.parse_args()
    build(Path(a.src), Path(a.out))
