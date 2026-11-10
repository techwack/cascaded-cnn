"""Short speaker document: results/technical comparison and conclusion/limitations/future work."""
import importlib.util
import tempfile
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("mr", ROOT / "scripts" / "make_results_pdf.py")
mr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mr)
R, pct, bacc = mr.R, mr.pct, mr.binary_acc
pr, prf, dg = R["pretrained"]["cascade"], R["pretrained"]["flat"], R["datagroup"]["cascade"]
ps = mr.json.load(open(ROOT / "results" / "pretrained_svm.json"))
rf = mr.json.load(open(ROOT / "results" / "random_forest.json"))
S = pr["stages"]
FONT = "Times New Roman"

doc = Document()
sec = doc.sections[0]
sec.left_margin = sec.right_margin = Cm(2.2)
sec.top_margin = sec.bottom_margin = Cm(2.0)
st = doc.styles["Normal"]
st.font.name = FONT
st.font.size = Pt(12)
st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)


def run_font(r, size=None, bold=None, italic=None):
    r.font.name = FONT
    rf_ = r._r.get_or_add_rPr().find(qn("w:rFonts"))
    for a in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rf_.set(qn(a), FONT)
    if size:
        r.font.size = Pt(size)
    if bold is not None:
        r.bold = bold
    if italic is not None:
        r.italic = italic


def H(text, size=16):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    run_font(p.add_run(text), size, True)


def P(text, bold_lead=None, italic=False, indent=0):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.left_indent = Cm(indent)
    if bold_lead:
        run_font(p.add_run(bold_lead + " "), 12, True)
    run_font(p.add_run(text), 12, False, italic)
    return p


def BL(lead, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.left_indent = Cm(0.9)
    p.paragraph_format.first_line_indent = Cm(-0.5)
    p.paragraph_format.space_after = Pt(2)
    run_font(p.add_run("\u2022  "), 12)
    if lead:
        run_font(p.add_run(lead + " "), 12, True)
    run_font(p.add_run(text), 12)


def TBL(rows, widths, bold_rows=(), size=10.5):
    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
    t.style = "Table Grid"
    t.alignment = 1
    for ri, row in enumerate(rows):
        for ci, v in enumerate(row):
            c = t.cell(ri, ci)
            c.width = Cm(widths[ci])
            par = c.paragraphs[0]
            par.alignment = WD_ALIGN_PARAGRAPH.CENTER if ri == 0 else WD_ALIGN_PARAGRAPH.LEFT
            run_font(par.add_run(str(v)), size, ri == 0 or ri in bold_rows)
            if ri == 0:
                shd = OxmlElement("w:shd")
                shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), "E7E6E6")
                c._tc.get_or_add_tcPr().append(shd)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


# ------------------------------------------------------------------ title
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run_font(p.add_run("Cascaded Convolutional Neural Network for Solar Panel Defect Classification"), 15, True)
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run_font(p.add_run("Presentation notes: Results and technical comparison (5 marks); Conclusion, limitations and future work (3 marks)"), 11, False, True)
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run_font(p.add_run("Divyanshi Jain [RA2311026010639]  |  Syed Mohammed Ghouse S [RA2311026010643]  |  Purvee Singh [RA2311026010645]"), 10.5)

# ------------------------------------------------------------------ part 1
H("1. Results and technical comparison")
P("Everything below is measured on one fixed held-out set of 3,000 images that was never used for training, early stopping or tuning. These are single runs (seed 42). Report chapter: 6.", "How we measured.")

H("1.1 Our results (12 classes, test set)", 13)
rows = [["Model", "Accuracy", "Macro-F1", "Stage 1: healthy vs defect"],
        ["First version (with brightness, contrast, noise jitter)", pct(R["default"]["cascade"]["accuracy"]), f"{R['default']['cascade']['macro_f1']:.2f}", pct(bacc(R["default"]))],
        ["After removing the jitter (flips only)", pct(R["no_photometric"]["cascade"]["accuracy"]), f"{R['no_photometric']['cascade']['macro_f1']:.2f}", pct(bacc(R["no_photometric"]))],
        ["Bigger input and networks, from scratch", pct(dg["accuracy"]), f"{dg['macro_f1']:.2f}", pct(bacc(R["datagroup"]))],
        ["Pretrained EfficientNet-B0 cascade (best)", pct(pr["accuracy"]), f"{pr['macro_f1']:.2f}", pct(bacc(R["pretrained"]))],
        ["Pretrained flat CNN (single network, same backbone)", pct(prf["accuracy"]), f"{prf['macro_f1']:.2f}", "n/a"]]
TBL(rows, [8.2, 2.4, 2.4, 3.6], bold_rows=(4,))
P("Accuracy is the share of correct predictions. Macro-F1 averages the score of all 12 classes equally, so it shows how the rare faults are doing. Stage 1 is the healthy-versus-defect decision on its own.", italic=True)

H("1.2 What caused the improvement", 13)
BL("Removing brightness and noise jitter (+10 points).", "In an infrared image brightness is temperature. Jittering it destroys the evidence that separates a hot spot from a healthy module.")
BL("Larger input and networks (about +8 points).", "Upsampling the image and strengthening the first two stages cut missed defects from 233 to 108.")
BL("ImageNet pretraining (+3.4 points).", "The pretrained filters transfer even to tiny thermal images, and training took minutes instead of an hour.")
BL("Did not help:", "soft routing, test-time augmentation, threshold tuning and balanced sampling.")

H("1.3 Cascade versus flat model (the main technical comparison)", 13)
P(f"With the same pretrained backbone the cascade scores {pct(pr['accuracy'])} accuracy and the flat model {pct(prf['accuracy'])}, so they are essentially tied. The cascade uses about 20 million parameters (five networks) against 4 million for the flat model, so it did not show an accuracy or efficiency advantage. "
  "What it does give is a stage-by-stage view of the errors.")
rows = [["Where the errors come from", "Errors (of 3,000)", "Share of the cascade's errors"],
        ["Stage 1: missed a defect", S["stage1_missed_defects"], f"{100 * S['stage1_missed_defects'] / S['total_errors']:.0f}%"],
        ["Stage 1: false alarm", S["stage1_false_alarms"], f"{100 * S['stage1_false_alarms'] / S['total_errors']:.0f}%"],
        ["Stage 2: wrong family", S["stage2_wrong_family"], f"{100 * S['stage2_wrong_family'] / S['total_errors']:.0f}%"],
        ["Stage 3: wrong subtype", S["stage3_wrong_subtype"], f"{100 * S['stage3_wrong_subtype'] / S['total_errors']:.0f}%"],
        ["Total", S["total_errors"], "100%"]]
TBL(rows, [7.4, 3.8, 5.4], bold_rows=(5,))
P(f"Stage 2, the family decision, is the weak link, and an error there cannot be corrected by Stage 3. Stage 1 is accurate: {pct(bacc(R['pretrained']))} on healthy versus defect, and about {pct(S['early_exit_rate'])} of images exit there.")

H("1.4 Comparison with published work", 13)
rows = [["Source", "Task and protocol", "Reported", "Ours"],
        ["Alves et al. 2021", "Binary detection, cross-validation", "92.5%", f"{pct(bacc(R['pretrained']))} (higher)"],
        ["Corley et al. 2025 (Random Forest)", "12 classes, split not stated", "84.0%", f"{pct(pr['accuracy'])} (slightly lower)"],
        ["Duranay 2023 (EfficientNet-B0 + SVM)", "12 classes, 10-fold cross-validation", "93.93%", f"{pct(pr['accuracy'])} (lower)"],
        ["Duranay-style pipeline on our split", "12 classes, our held-out test set", "-", f"{pct(ps['accuracy'])} (below our cascade)"]]
TBL(rows, [5.0, 5.0, 2.2, 4.4], bold_rows=(4,))
P("The 93.93% uses 10-fold cross-validation, an ImageNet-pretrained network and a supervised feature selection applied before the cross-validation. As the paper describes it, the test images can influence those steps. Our test images never do. "
  f"Also, about 7% of our test images have a near-identical training image, which inflates cross-validated scores. When the same kind of pipeline is run on our held-out split it scores only {pct(ps['accuracy'])}. "
  "So the gap is most likely due to the protocol; we say this suggests it, not that it proves it.", "How to explain the gap.")
P("The dataset paper itself reports no model accuracies. The 84%, 92.5% and 93% figures often attributed to it come from later papers.", "A point that earns credit.")

# ------------------------------------------------------------------ part 2
H("2. Conclusion, limitations and future work")
H("2.1 Conclusion (say these three points)", 13)
BL("1.", f"We built and tested a three-stage cascade of five small CNNs that classifies infrared solar panel defects from a single sensor. Accuracy rose from 61.7% to {pct(pr['accuracy'])}, and the healthy-versus-defect stage reaches {pct(bacc(R['pretrained']))}.")
BL("2.", "Our hypothesis that a cascade clearly beats a flat classifier was only partly supported. With the same backbone they tie, and the family decision is the weak link.")
BL("3.", "A practical lesson: thermal images need thermal-aware augmentation. Flips help; brightness, contrast and noise jitter hurt.")

H("2.2 Limitations (be upfront, it earns marks)", 13)
BL("Single runs.", "One seed, so differences below about 0.5 points are noise. The 10-fold cross-validation was prepared but did not complete, so it is not reported.")
BL("Rare classes.", "Soiling, Hot-Spot and Diode-Multi stay weak, which is why macro-F1 (0.70) is well below accuracy.")
BL("Near-duplicate images.", "The dataset has no module IDs, so near-identical crops can appear on both sides of a split.")
BL("Efficiency.", "The early exit gave no measured advantage, because the pretrained cascade has five times the parameters of the flat model, and latency was not measured reliably.")
BL("Targets.", "We did not reach the 93% ceiling or the 95% target stated in the proposal, and published 12-class results use different protocols.")

H("2.3 Future work", 13)
BL("Fair comparison.", "Several seeds and a leak-free 10-fold cross-validation, plus a full replication of the published feature-selection pipeline.")
BL("Better family decision.", "Train Stage 2 jointly with the specialists, or on the cascade's own routing errors.")
BL("Rare faults.", "Class-specific augmentation or oversampling, and removal of near-duplicate crops before splitting.")
BL("Deployment.", "Measure latency, memory and energy on a drone-class device, and try distilling the cascade into smaller networks.")
BL("More information.", "Add RGB or neighbouring-panel context.")

# ------------------------------------------------------------------ Q&A
H("3. Likely questions and short answers")
qa = [("Why is it not above 93%?", "That figure uses cross-validation and a pipeline that, as described, sees all the labels before testing. Under our fixed split the same kind of method gets 78.7%, and our cascade gets 83.2%."),
      ("Is the cascade worth it?", "It matches a flat model and shows exactly which stage causes each error, and Stage 1 is a reliable gatekeeper (95.8%). We did not demonstrate an efficiency gain."),
      ("What was your biggest finding?", "Brightness and noise augmentation hurt thermal images by about ten points, because brightness is temperature."),
      ("Which stage is the problem?", "Stage 2, the family decision: 272 of the 505 errors."),
      ("Why use pretrained models?", "They gave +3.4 points and trained in minutes. Our proposal expected little benefit, and the measurement showed the opposite."),
      ("How do you know the result is clean?", "The 3,000 test images were never used for training, early stopping, normalisation or model selection.")]
for q, a in qa:
    BL(q, a)

# ------------------------------------------------------------------ figures
tmp = Path(tempfile.mkdtemp())
ch = mr.charts(tmp)
H("4. Figures to show")
doc.add_picture(str(ch["prog"]), width=Cm(14.5))
doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
c = doc.add_paragraph()
c.alignment = WD_ALIGN_PARAGRAPH.CENTER
run_font(c.add_run("Test accuracy and macro-F1 by configuration (report Fig. 8)."), 10, False, True)
doc.add_picture(str(ch["stages"]), width=Cm(14.0))
doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
c = doc.add_paragraph()
c.alignment = WD_ALIGN_PARAGRAPH.CENTER
run_font(c.add_run("Where the cascade makes its errors (report Fig. 11)."), 10, False, True)

out = Path(r"C:\Users\Divyanshi\Downloads\Presentation_Notes_Results_Conclusion.docx")
doc.save(str(out))
print("wrote", out)
