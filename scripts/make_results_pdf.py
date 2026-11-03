"""Build docs/Results_Comparison_Conclusion.pdf from results/*.json.

    python scripts/make_results_pdf.py
"""
import json
import tempfile
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "docs" / "Results_Comparison_Conclusion.pdf"
N_TEST = 3000


def load(name):
    return json.load(open(RES / f"{name}.json"))


R = {k: load(k) for k in ["default", "no_photometric", "improved", "improved2", "improved3", "datagroup"]}
CLASSES = R["improved3"]["classes"]


def binary_acc(r):
    s = r["cascade"]["stages"]
    return 1 - (s["stage1_missed_defects"] + s["stage1_false_alarms"]) / N_TEST


CFG = [  # (label, key, model)
    ("Default\n(jitter on)", "default", "cascade"),
    ("No photometric\njitter", "no_photometric", "cascade"),
    ("Improved\n(80x48, deeper)", "improved", "cascade"),
    ("Improved3\n(bigger S1/S2)", "improved3", "cascade"),
    ("Data-driven\nfamilies", "datagroup", "cascade"),
]


def charts(tmp: Path) -> dict:
    out = {}
    # 1) progression
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    x = np.arange(len(CFG)); w = 0.38
    acc = [R[k][m]["accuracy"] * 100 for _, k, m in CFG]
    f1 = [R[k][m]["macro_f1"] * 100 for _, k, m in CFG]
    b1 = ax.bar(x - w / 2, acc, w, label="Accuracy (%)", color="#2b3a4a")
    b2 = ax.bar(x + w / 2, f1, w, label="Macro-F1 (x100)", color="#d9480f")
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{b.get_height():.1f}", ha="center", fontsize=7)
    fl = R["improved"]["flat"]
    ax.axhline(fl["accuracy"] * 100, color="#2b8a3e", ls="--", lw=1, label=f"Flat CNN accuracy ({fl['accuracy']*100:.1f})")
    ax.set_xticks(x); ax.set_xticklabels([c[0] for c in CFG], fontsize=7)
    ax.set_ylim(0, 100); ax.legend(fontsize=7, loc="upper left"); ax.set_title("Cascade test performance by configuration", fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); p = tmp / "prog.png"; fig.savefig(p, dpi=200); plt.close(fig); out["prog"] = p
    # 2) per-class F1
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    x = np.arange(len(CLASSES)); w = 0.4
    c3 = [R["improved3"]["cascade"]["per_class"][c]["f1-score"] for c in CLASSES]
    fc = [R["improved"]["flat"]["per_class"][c]["f1-score"] for c in CLASSES]
    ax.bar(x - w / 2, c3, w, label="Cascade (improved3)", color="#2b3a4a")
    ax.bar(x + w / 2, fc, w, label="Flat CNN (improved)", color="#adb5bd")
    ax.set_xticks(x); ax.set_xticklabels(CLASSES, rotation=40, ha="right", fontsize=7)
    ax.set_ylim(0, 1.05); ax.set_ylabel("F1", fontsize=8); ax.legend(fontsize=7)
    ax.set_title("Per-class F1 on the test set", fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); p = tmp / "perclass.png"; fig.savefig(p, dpi=200); plt.close(fig); out["perclass"] = p
    # 3) stage errors
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    keys = [("stage1_missed_defects", "Stage 1: missed defect", "#c92a2a"), ("stage1_false_alarms", "Stage 1: false alarm", "#f08c00"),
            ("stage2_wrong_family", "Stage 2: wrong family", "#1c7ed6"), ("stage3_wrong_subtype", "Stage 3: wrong subtype", "#2b8a3e")]
    names = [("Default", "default"), ("No jitter", "no_photometric"), ("Improved", "improved"), ("Improved3", "improved3"), ("Data-driven", "datagroup")]
    left = np.zeros(len(names))
    for k, lab, col in keys:
        v = np.array([R[n]["cascade"]["stages"][k] for _, n in names])
        ax.barh(range(len(names)), v, left=left, label=lab, color=col)
        left += v
    ax.set_yticks(range(len(names))); ax.set_yticklabels([n for n, _ in names], fontsize=8); ax.invert_yaxis()
    ax.set_xlabel("Errors out of 3,000 test images", fontsize=8); ax.legend(fontsize=7, loc="lower right")
    ax.set_title("Where the cascade makes its errors", fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); p = tmp / "stages.png"; fig.savefig(p, dpi=200); plt.close(fig); out["stages"] = p
    return out


ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=16, spaceAfter=8, textColor=colors.HexColor("#1b1f24"))
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=12, spaceBefore=10, spaceAfter=4)
B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=9.5, leading=13.5, spaceAfter=5)
SM = ParagraphStyle("SM", parent=B, fontSize=8, leading=10.5, textColor=colors.HexColor("#555555"))
CELL = ParagraphStyle("CELL", parent=B, fontSize=8, leading=10, spaceAfter=0)
CELLB = ParagraphStyle("CELLB", parent=CELL, fontName="Helvetica-Bold")


def P(t, s=B):
    return Paragraph(t, s)


def table(rows, widths, header=True, zebra=True, bold_rows=()):
    data = [[Paragraph(str(c), CELLB if (header and i == 0) or i in bold_rows else CELL) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c9ced4")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9ecef")))
    if zebra:
        for i in range(1, len(rows)):
            if i % 2 == 0:
                st.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f8f9fa")))
    t.setStyle(TableStyle(st))
    return t


def pct(x):
    return f"{x * 100:.1f}%"


def build():
    tmp = Path(tempfile.mkdtemp())
    ch = charts(tmp)
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=1.9 * cm, rightMargin=1.9 * cm, topMargin=1.7 * cm,
                            bottomMargin=1.7 * cm, title="Cascaded-SolarNet: Results, Comparison and Conclusion",
                            author="Divyamshi Jain, Syed Mohammad Ghouse, Purvee Singh")
    s = []
    i3, dg, imp, fl = R["improved3"]["cascade"], R["datagroup"]["cascade"], R["improved"]["cascade"], R["improved"]["flat"]

    s += [P("Cascaded-SolarNet: Results, Comparison with Base Papers and Conclusion", H1),
          P("Cascaded Convolutional Neural Network for Solar Panel Defect Classification (InfraredSolarModules, 12 classes). "
            "All numbers are measured on the same held-out test split of 3,000 images (stratified 70/15/15, seed 42), single run per configuration. "
            "Code, configs and raw metrics: github.com/techwack/cascaded-cnn.", SM),
          P("1. Results", H2)]
    s.append(P(f"The best cascade reaches <b>{pct(dg['accuracy'])}</b> accuracy, macro-F1 <b>{dg['macro_f1']:.3f}</b> and Cohen's kappa "
               f"<b>{dg['kappa']:.3f}</b> on the 12-class problem (data-driven families), and <b>{pct(binary_acc(R['datagroup']))}</b> accuracy "
               f"on the Stage 1 healthy-versus-defect decision. With the semantic families (Electrical, Environmental, Structural) the "
               f"'improved3' configuration reaches {pct(i3['accuracy'])} accuracy and macro-F1 {i3['macro_f1']:.3f}."))
    rows = [["Configuration", "Model", "Accuracy", "Macro-F1", "Kappa", "Stage 1 binary acc."]]
    names = {"default": "Default (brightness/contrast/noise jitter)", "no_photometric": "No photometric jitter (flips only)",
             "improved": "Improved: 80x48 input, 2 convs/block", "improved2": "Improved2: + balanced sampling, label smoothing",
             "improved3": "Improved3: + larger Stage 1/2 CNNs", "datagroup": "Data-driven family grouping"}
    for k in ["default", "no_photometric", "improved", "improved2", "improved3", "datagroup"]:
        c = R[k]["cascade"]
        rows.append([names[k], "Cascade", pct(c["accuracy"]), f"{c['macro_f1']:.3f}", f"{c['kappa']:.3f}", pct(binary_acc(R[k]))])
        if "flat" in R[k]:
            f = R[k]["flat"]
            rows.append(["", "Flat CNN (same split)", pct(f["accuracy"]), f"{f['macro_f1']:.3f}", f"{f['kappa']:.3f}", "n/a"])
    s += [table(rows, [6.2, 3.2, 1.8, 1.7, 1.5, 2.6]), Spacer(1, 6), Image(str(ch["prog"]), 15.5 * cm, 6.9 * cm)]
    s.append(P("Main findings", H2))
    for t in [
        "<b>Removing photometric jitter was the biggest single gain.</b> Brightness, contrast and noise augmentation lowered cascade accuracy to 61.7%; flips only gave 71.1%. "
        "In thermal images brightness is temperature, so jittering it destroys the signal that separates the classes.",
        f"<b>Larger input and capacity helped next.</b> Upsampling to 80x48 and using two convolutions per block raised accuracy to {pct(imp['accuracy'])}; "
        f"enlarging the Stage 1 and Stage 2 networks raised it to {pct(i3['accuracy'])}, and Stage 1 binary accuracy from {pct(binary_acc(R['improved']))} to {pct(binary_acc(R['improved3']))}.",
        f"<b>Cascade versus flat CNN.</b> The cascade is ahead on accuracy ({pct(i3['accuracy'])} vs {pct(fl['accuracy'])}) and kappa ({i3['kappa']:.3f} vs {fl['kappa']:.3f}) but slightly behind "
        f"on macro-F1 ({i3['macro_f1']:.3f} vs {fl['macro_f1']:.3f}), so the advantage is modest. {pct(i3['stages']['early_exit_rate'])} of test images leave the cascade at Stage 1 and never reach the larger networks.",
        "<b>Error attribution.</b> With the semantic families, wrong-family decisions at Stage 2 are the largest error source. The data-driven grouping cut Stage 2 errors from "
        f"{i3['stages']['stage2_wrong_family']} to {dg['stages']['stage2_wrong_family']}, but the hard confusions moved into Stage 3 ({i3['stages']['stage3_wrong_subtype']} to {dg['stages']['stage3_wrong_subtype']} errors), "
        f"so the net gain is small ({pct(i3['accuracy'])} to {pct(dg['accuracy'])}, within single-run noise).",
        "<b>Tried without gain:</b> soft routing (multiplying stage probabilities), flip test-time augmentation, tuning the Stage 1 threshold on validation data (optimum stayed at 0.5) and balanced sampling with label smoothing.",
    ]:
        s.append(P("- " + t))
    s += [Image(str(ch["stages"]), 15.5 * cm, 6.5 * cm), PageBreak()]

    s.append(P("Per-class performance", H2))
    s.append(P("Rare classes are the weak point: they have only 123 to 250 training images, and several look alike (Cell vs Cell-Multi, Hot-Spot vs Cell). "
               "Class weighting in the loss, stratified splitting and macro-F1 were used to keep them from being ignored, but they remain the hardest classes."))
    rows = [["Class", "Test images", "Cascade F1 (improved3)", "Cascade F1 (data-driven)", "Flat CNN F1"]]
    for c in CLASSES:
        rows.append([c, int(R["improved3"]["cascade"]["per_class"][c]["support"]), f"{R['improved3']['cascade']['per_class'][c]['f1-score']:.2f}",
                     f"{R['datagroup']['cascade']['per_class'][c]['f1-score']:.2f}", f"{R['improved']['flat']['per_class'][c]['f1-score']:.2f}"])
    s += [table(rows, [3.6, 2.4, 3.8, 3.8, 2.6]), Spacer(1, 6), Image(str(ch["perclass"]), 15.5 * cm, 6.9 * cm)]

    s.append(P("2. Comparison with the Base Papers", H2))
    s.append(P("The base work in our literature survey falls into two groups: results on the same InfraredSolarModules benchmark, and results on other data, "
               "modalities or tasks. Only the first group is a fair comparison, and even there the task (binary or 12-class) is not stated in the survey. "
               "We therefore compare our Stage 1 binary accuracy and our 12-class accuracy separately."))
    rows = [["Source (as cited in our survey)", "Data / task", "Reported", "Our result", "Comparable?"],
            ["Millendorf et al. 2020 (dataset): Random Forest", "InfraredSolarModules, task not stated", "84%", f"Binary {pct(binary_acc(R['datagroup']))}; 12-class {pct(dg['accuracy'])}", "Same data. Task unclear."],
            ["Millendorf et al. 2020: custom CNN", "InfraredSolarModules, task not stated", "92.5%", f"Binary {pct(binary_acc(R['datagroup']))}; 12-class {pct(dg['accuracy'])}", "Same data. Task unclear."],
            ["Millendorf et al. 2020: EfficientNet + SVM", "InfraredSolarModules, task not stated", "93%", f"Binary {pct(binary_acc(R['datagroup']))}; 12-class {pct(dg['accuracy'])}", "Same data. Task unclear."],
            ["PV-faultNet (arXiv:2411.02997)", "Lightweight CNN, dataset not specified in survey", "93% at 8 ms/image", "Not measured on same hardware", "No (different setup)"],
            ["Tella et al. 2025 (8 CNNs/ViTs)", "ELPV benchmark (different dataset)", "about 68% multi-class, 73% binary", f"12-class {pct(dg['accuracy'])}", "Directional only"],
            ["B-CNN (Pathak and Patil 2025)", "Thermal + visual fusion (two sensors)", "98.5% accuracy", "Single IR sensor", "No (more information)"],
            ["Field-adapted ResNet-50", "Field data, transfer learning", "98.59%", "-", "No (different data)"],
            ["CNN + BFO (VGG-16)", "Pretrained CNNs on other data", "98.75%", "-", "No (different data)"]]
    s += [table(rows, [4.4, 3.7, 2.6, 3.4, 2.7]), Spacer(1, 6)]
    s.append(P("<b>How to read this.</b>"))
    for t in [f"<b>Binary (healthy vs defect):</b> our Stage 1 reaches {pct(binary_acc(R['datagroup']))}, which is at or above the 92.5% to 93% figures if those refer to the binary task, as their magnitude suggests. "
              "The margin is 1 to 2 points on a single run and a different protocol, so we claim parity rather than a clear win.",
              f"<b>12-class:</b> our {pct(dg['accuracy'])} is not comparable to 93%. If the 92.5% to 93% figures were 12-class results, our model would be about 13 points below them. "
              "It is, however, above the roughly 68% multi-class level that Tella et al. report across eight architectures, although on a different dataset.",
              "<b>Models above 98%</b> use two sensors, pretrained backbones on other data or field-specific data. They show what additional information can buy, not a like-for-like baseline for a single-camera, 24x40 pixel IR model.",
              "<b>Our own baselines:</b> the flat CNN with identical building blocks reaches " + f"{pct(fl['accuracy'])} accuracy and {fl['macro_f1']:.3f} macro-F1. A random forest baseline was planned but not run on our final split."]:
        s.append(P("- " + t))
    s.append(P("<b>Note on sources.</b> The original dataset paper (Millendorf et al., ICLR 2020 workshop, 5 pages) describes the data and lists benchmark classification "
               "results as a goal, but it reports no model accuracies. The 84%, 92.5% and 93% figures in our survey are quoted from elsewhere and could not be traced to that paper. "
               "They should be verified against their original source before being presented as published baselines. The report's stated target of above 95% accuracy was not reached.", SM))
    s.append(PageBreak())

    s.append(P("3. Conclusion", H2))
    for t in [
        f"We implemented Cascaded-SolarNet, a three-stage cascade of five small CNNs for 12-class infrared solar module defect classification, in PyTorch, with a reproducible split, tests, diagrams and live demos.",
        f"<b>The cascade works.</b> After diagnosing and removing harmful photometric augmentation, enlarging the input and strengthening the front stages, accuracy rose from 61.7% to {pct(dg['accuracy'])} "
        f"(macro-F1 0.37 to {dg['macro_f1']:.2f}). Healthy-versus-defect triage reaches about {pct(binary_acc(R['datagroup']))}, in line with published binary results on this dataset.",
        f"<b>The claim that a cascade clearly beats a flat classifier was only partly supported.</b> It is ahead on accuracy and kappa and processes about half of all images with only the small Stage 1 network, "
        f"but a flat CNN with the same building blocks is within about 3 accuracy points and slightly better on macro-F1. The target of exceeding 93% on all 12 classes was not met.",
        "<b>Error propagation is the central trade-off.</b> A wrong decision at Stage 1 or Stage 2 cannot be undone. Re-grouping the classes by their actual confusions moved errors between stages "
        "rather than removing them, and soft routing did not recover them.",
        "<b>Thermal images need thermal-aware augmentation.</b> Geometric flips help; brightness, contrast and noise jitter hurt, because brightness encodes temperature.",
    ]:
        s.append(P("- " + t))
    s.append(P("Limitations", H2))
    for t in ["Single run per configuration (seed 42); differences below about 0.5 points are within noise and no confidence intervals were computed.",
              "The benchmark has no module or site identifiers, so near-duplicate crops can appear in both training and test data, which may inflate all reported numbers, ours and published ones.",
              "Rare classes (Soiling, Hot-Spot, Diode-Multi) remain weak, which keeps macro-F1 well below accuracy.",
              "Latency on edge hardware was not measured reliably; the early-exit rate (about half of the images) is the supporting evidence for efficiency.",
              "The comparison with published results is limited by missing information about which task the quoted baselines address."]:
        s.append(P("- " + t))
    s.append(P("Future work", H2))
    for t in ["Run several seeds and report mean and standard deviation; run the random forest and an EfficientNet baseline on the same split.",
              "Improve Stage 2, for example by training the family classifier jointly with the specialists or on the cascade's own routing errors, or with class-specific augmentation for rare faults.",
              "Add the planned ablations (no early exit, unweighted loss, upsampling factor in isolation) and deduplicate near-identical crops before splitting.",
              "Measure real latency and memory on a drone-class edge device, and test additional modalities such as RGB."]:
        s.append(P("- " + t))
    doc.build(s)
    print("wrote", OUT)


if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    build()
