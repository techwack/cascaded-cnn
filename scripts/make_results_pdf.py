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


R = {k: load(k) for k in ["default", "no_photometric", "improved", "improved2", "improved3", "datagroup", "pretrained"]}
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
    ("Pretrained\nEfficientNet-B0", "pretrained", "cascade"),
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
    names = [("Default", "default"), ("No jitter", "no_photometric"), ("Improved", "improved"), ("Improved3", "improved3"), ("Data-driven", "datagroup"), ("Pretrained", "pretrained")]
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
    pr, prf = R["pretrained"]["cascade"], R["pretrained"]["flat"]
    i3, dg, imp, fl = R["improved3"]["cascade"], R["datagroup"]["cascade"], R["improved"]["cascade"], R["improved"]["flat"]

    s += [P("Cascaded-SolarNet: Results, Comparison with Base Papers and Conclusion", H1),
          P("Cascaded Convolutional Neural Network for Solar Panel Defect Classification (InfraredSolarModules, 12 classes). "
            "All numbers are measured on the same held-out test split of 3,000 images (stratified 70/15/15, seed 42), single run per configuration. "
            "Code, configs and raw metrics: github.com/techwack/cascaded-cnn.", SM),
          P("1. Results", H2)]
    s.append(P(f"The best result is the <b>pretrained cascade</b> (an ImageNet EfficientNet-B0 under each of the five stages): <b>{pct(pr['accuracy'])}</b> accuracy, macro-F1 <b>{pr['macro_f1']:.3f}</b>, "
               f"Cohen's kappa <b>{pr['kappa']:.3f}</b> on the 12-class problem, and <b>{pct(binary_acc(R['pretrained']))}</b> on the Stage 1 healthy-versus-defect decision. "
               f"The best cascade trained from scratch (data-driven families) reaches {pct(dg['accuracy'])} accuracy and macro-F1 {dg['macro_f1']:.3f}; "
               f"with the semantic families (Electrical, Environmental, Structural) the 'improved3' configuration reaches {pct(i3['accuracy'])}."))
    rows = [["Configuration", "Model", "Accuracy", "Macro-F1", "Kappa", "Stage 1 binary acc."]]
    names = {"default": "Default (brightness/contrast/noise jitter)", "no_photometric": "No photometric jitter (flips only)",
             "improved": "Improved: 80x48 input, 2 convs/block", "improved2": "Improved2: + balanced sampling, label smoothing",
             "improved3": "Improved3: + larger Stage 1/2 CNNs", "datagroup": "Data-driven family grouping", "pretrained": "Pretrained: ImageNet EfficientNet-B0 under every stage"}
    for k in ["default", "no_photometric", "improved", "improved2", "improved3", "datagroup", "pretrained"]:
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
        f"<b>ImageNet pretraining gave the largest late gain.</b> Fine-tuning a pretrained EfficientNet-B0 at every stage raised the cascade from {pct(dg['accuracy'])} to {pct(pr['accuracy'])} accuracy and macro-F1 from {dg['macro_f1']:.3f} to {pr['macro_f1']:.3f}, and the flat CNN from {pct(fl['accuracy'])} to {pct(prf['accuracy'])}. "
        f"Training also converged in minutes instead of an hour.",
        f"<b>Cascade versus flat CNN.</b> From scratch the cascade is ahead on accuracy and kappa but slightly behind on macro-F1. With pretraining the two are essentially tied: cascade {pct(pr['accuracy'])} / {pr['macro_f1']:.3f} / {pr['kappa']:.3f} against flat {pct(prf['accuracy'])} / {prf['macro_f1']:.3f} / {prf['kappa']:.3f} (accuracy, macro-F1, kappa). The pretrained cascade has about 20M parameters (five EfficientNets) against 4M for the flat model, so the cascade did not give a measurable benefit here. "
        f"Earlier from-scratch comparison: The cascade is ahead on accuracy ({pct(i3['accuracy'])} vs {pct(fl['accuracy'])}) and kappa ({i3['kappa']:.3f} vs {fl['kappa']:.3f}) but slightly behind "
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

    rf = json.load(open(RES / "random_forest.json"))
    ps = json.load(open(RES / "pretrained_svm.json"))
    s.append(P("2. Comparison with the Base Papers", H2))
    s.append(P("We re-checked the literature at the primary sources. Two points matter for any comparison. First, the dataset paper itself "
               "(Millendorf et al., ICLR 2020 workshop) reports <b>no model accuracies</b>; the numbers our earlier survey attributed to it "
               "(Random Forest 84%, CNN 92.5%, EfficientNet+SVM 93%) come from later papers. Second, those papers differ from our setup in task, "
               "split and features, so the numbers can be placed side by side but not ranked."))
    rows = [["Paper", "Method", "Task and protocol", "Reported", "Checked"],
            ["Duranay 2023, Electronics 12(21):4397", "EfficientNet-B0 features (ImageNet-pretrained), NCA selection, SVM", "12 classes, 10-fold cross-validation", "93.93% accuracy, 89.82% F1", "Full paper read"],
            ["Alves et al. 2021, Renewable Energy 179:502", "Custom CNN with data augmentation", "Anomaly detection (binary); 8 selected classes; cross-validation", "92.5% binary; 78.85% for 8 classes", "Abstract (library record)"],
            ["Multi-scale CNN, transfer learning (Eng. Appl. Artif. Intell. 113, 2022), as cited by Duranay", "Three-branch CNN, transfer learning", "Fault detection; 11 anomaly types", "97.32% detection; 93.51% for 11 types", "As cited by Duranay"],
            ["Corley et al. 2025 (CVPRW), their Random Forest", "Random Forest", "'Anomaly classification', split not stated", "84.0%", "arXiv text read"],
            ["Barraz et al. 2025; Gopalakrishnan et al. 2024, as cited by Corley", "Cascading ensemble with voting CNN; NASNet + LSTM", "Not stated", "88%; 85%", "As cited by Corley"],
            ["Millendorf et al. 2020 (dataset paper)", "-", "-", "No model results reported", "Full paper read"],
            ["Preprints and listings (SSRN Hyperband CNN; diffusion + ViT)", "Hyperband-tuned CNN; transformer", "12 classes, repeated k-fold CV (SSRN)", "98.75%; 92.78%", "<b>Not verified</b> (listings only)"],
            ["<b>Ours: pretrained cascade</b>", "ImageNet EfficientNet-B0 fine-tuned at each of 5 stages", "12 classes, single stratified 70/15/15 hold-out", f"{pct(pr['accuracy'])} accuracy, macro-F1 {pr['macro_f1']:.3f}", "Our results"],
            ["Ours: cascade (data-driven families)", "Five small CNNs, trained from scratch", "12 classes, single stratified 70/15/15 hold-out", f"{pct(dg['accuracy'])} accuracy, macro-F1 {dg['macro_f1']:.3f}", "Our results"],
            ["Ours: Stage 1 only (pretrained / from scratch)", "Binary triage CNN", "Healthy vs defect, same hold-out", f"{pct(binary_acc(R['pretrained']))} / {pct(binary_acc(R['datagroup']))}", "Our results"],
            ["Ours: flat CNN (pretrained / scratch) / Random Forest", "EfficientNet-B0 / same blocks / raw pixels", "12 classes, same hold-out", f"{pct(prf['accuracy'])} / {pct(fl['accuracy'])} / {pct(rf['accuracy'])}", "Our results"],
            ["Ours: Duranay-style baseline on our split", "EfficientNet-B0 (ImageNet) 1280-d features + RBF SVM (no NCA, last layer only)", "12 classes, same hold-out",
             f"{pct(ps['accuracy'])} accuracy, macro-F1 {ps['macro_f1']:.3f}; binary {pct(ps['binary_accuracy'])}", "Our results"]]
    s += [table(rows, [3.9, 3.6, 3.5, 3.0, 2.2], bold_rows=(8,)), Spacer(1, 6)]
    s.append(P("How to read this", H2))
    for t in [
        f"<b>Binary triage.</b> Our pretrained Stage 1 reaches {pct(binary_acc(R['pretrained']))} ({pct(binary_acc(R['datagroup']))} from scratch). That is above Alves et al. (92.5%, cross-validated) and below the multi-scale transfer-learning CNN (97.32%). "
        "It is a single hold-out run, so we claim to be at or slightly above the 92.5% to 93% range and not a new best.",
        f"<b>12 classes.</b> Our best, {pct(pr['accuracy'])}, is <b>below</b> the published 12-class figures (84.0% Random Forest, 85% to 88% for ensembles, 93.93% for EfficientNet features with an SVM). On the published protocols we did not beat them, and we did not reach the 93% ceiling or the 95% target in our proposal. "
        f"But a Duranay-style pipeline (pretrained EfficientNet-B0 features and an SVM) run on <b>our</b> split scored only {pct(ps['accuracy'])} (macro-F1 {ps['macro_f1']:.3f}, binary {pct(ps['binary_accuracy'])}), "
        f"below our pretrained cascade ({pct(pr['accuracy'])}, macro-F1 {pr['macro_f1']:.3f}, binary {pct(binary_acc(R['pretrained']))}) and about level with our from-scratch cascade ({pct(dg['accuracy'])}). This suggests the gap to the published 93.93% is largely due to the evaluation protocol (cross-validation with near-duplicate images), and possibly to details we did not replicate such as selecting 17,000 multi-layer features with NCA, rather than to our model being weaker. It does not prove it.",
        f"<b>Why the numbers are not like-for-like.</b> Duranay uses 10-fold cross-validation and ImageNet-pretrained features; ours is one fixed hold-out split and models trained from scratch. We also measured overlap between our splits: about 7% of test images have a near-identical training image (distance below 0.5; 96% of those share the label) and about 54% have a close one (distance below 1; 78% share the label). "
        "Random fold splits would put such near-duplicates on both sides and raise cross-validated scores. This overlap cannot explain the whole gap, though: on our split a raw-pixel Random Forest scores "
        f"{pct(rf['accuracy'])} and a raw-pixel 1-nearest-neighbour 62.4%, against the 84.0% that Corley et al. report for a Random Forest. How that figure was obtained is not stated, so the cause of that gap is unknown.",
        f"<b>What we can say.</b> Under our fixed hold-out, the pretrained cascade is about {(pr['accuracy'] - ps['accuracy']) * 100:.0f} points ahead of a pretrained-feature SVM and about {(pr['accuracy'] - rf['accuracy']) * 100:.0f} points ahead of a raw-pixel Random Forest; it is level with a pretrained flat CNN. This is a Duranay-style baseline, not an exact replication (they select 17,000 features with NCA from several layers). Our contribution is the cascade design and the ablation and error analysis (which stage causes which errors), not a new accuracy record.",
        "<b>To make the comparison fully fair</b>, a leak-free 10-fold cross-validation of the pretrained cascade is running (progress in results/LIVE_STATUS.md on GitHub); its numbers are not part of this document yet.",
    ]:
        s.append(P("- " + t))
    s.append(P("Sources: Duranay, Electronics 12(21):4397 (mdpi.com/2079-9292/12/21/4397); Corley et al., arXiv:2503.02128; Alves et al., Renewable Energy 179:502-516 (abstract via library record); "
               "Millendorf et al., ICLR 2020 AI for Earth Sciences workshop. Rows marked 'as cited' were read in the citing paper, not in the original. Rows marked 'Not verified' appeared only in search listings.", SM))
    s.append(P("3. Conclusion", H2))
    for t in [
        f"We implemented Cascaded-SolarNet, a three-stage cascade of five small CNNs for 12-class infrared solar module defect classification, in PyTorch, with a reproducible split, tests, diagrams and live demos.",
        f"<b>The cascade works.</b> After diagnosing and removing harmful photometric augmentation, enlarging the input and strengthening the front stages, accuracy rose from 61.7% to {pct(dg['accuracy'])} from scratch and to {pct(pr['accuracy'])} with ImageNet pretraining "
        f"(macro-F1 0.37 to {pr['macro_f1']:.2f}). Healthy-versus-defect triage reaches about {pct(binary_acc(R['pretrained']))}, comparable to the 92.5% binary figure published for this dataset.",
        f"<b>The claim that a cascade clearly beats a flat classifier was only partly supported.</b> From scratch it is ahead on accuracy and kappa, and with pretraining it ties a flat CNN (accuracy {pct(pr['accuracy'])} vs {pct(prf['accuracy'])}, macro-F1 {pr['macro_f1']:.3f} vs {prf['macro_f1']:.3f}) while using five times the parameters. "
        f" Published 12-class results (84% to 93.93%) are higher than ours, although they use cross-validation and pretrained features; the target of exceeding 93% on all 12 classes was not met.",
        "<b>Error propagation is the central trade-off.</b> A wrong decision at Stage 1 or Stage 2 cannot be undone. Re-grouping the classes by their actual confusions moved errors between stages "
        "rather than removing them, and soft routing did not recover them.",
        "<b>Thermal images need thermal-aware augmentation.</b> Geometric flips help; brightness, contrast and noise jitter hurt, because brightness encodes temperature.",
    ]:
        s.append(P("- " + t))
    s.append(P("Limitations", H2))
    for t in ["Single run per configuration (seed 42); differences below about 0.5 points are within noise and no confidence intervals were computed.",
              "The benchmark has no module or site identifiers, we measured that about 7% of our test images have a near-identical training image, and cross-validated results in the literature are likely to be more affected by this than our fixed hold-out.",
              "Rare classes (Soiling, Hot-Spot, Diode-Multi) remain weak, which keeps macro-F1 well below accuracy.",
              "Latency on edge hardware was not measured reliably; the early-exit rate (about half of the images) is the supporting evidence for efficiency.",
              "Published numbers use different protocols (cross-validation, pretrained features), and some sources could only be checked through abstracts or citing papers."]:
        s.append(P("- " + t))
    s.append(P("Future work", H2))
    for t in ["Run several seeds and report mean and standard deviation; evaluate under 10-fold cross-validation for a like-for-like comparison with published results, and run the EfficientNet-B0 + SVM pipeline on our split.",
              "Improve Stage 2, for example by training the family classifier jointly with the specialists or on the cascade's own routing errors, or with class-specific augmentation for rare faults.",
              "Add the planned ablations (no early exit, unweighted loss, upsampling factor in isolation) and deduplicate near-identical crops before splitting.",
              "Measure real latency and memory on a drone-class edge device, and test additional modalities such as RGB."]:
        s.append(P("- " + t))
    doc.build(s)
    print("wrote", OUT)


if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    build()
