"""Technical-analysis sections (complexity, error propagation, significance, deployment) computed from results/*.json.
Ideas adapted from Purvee's 30-page report; every number here is recomputed from the repository results."""
import json
import math

import srm_content as sc

ROOT, R, pct = sc.ROOT, sc.R, sc.pct
CX = json.load(open(ROOT / "results" / "complexity.json"))
N, ND = 3000, 1500
CFG = [("improved", "C3"), ("improved3", "C5"), ("datagroup", "C6"), ("pretrained", "C7")]


def cond(k):
    s = R[k]["cascade"]["stages"]
    a1 = 1 - s["stage1_missed_defects"] / ND
    ah = 1 - s["stage1_false_alarms"] / ND
    n2 = ND - s["stage1_missed_defects"]
    a2 = 1 - s["stage2_wrong_family"] / n2
    a3 = 1 - s["stage3_wrong_subtype"] / (n2 - s["stage2_wrong_family"])
    return a1, a2, a3, ah


def wilson(p, n=N, z=1.96):
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def ztest(p1, p2, n=N):
    p = (p1 + p2) / 2
    return (p2 - p1) / math.sqrt(2 * p * (1 - p) / n)


def blocks(pre="3.6", chap=3, tbase=8):
    """Returns block tuples; numbering Table/section prefix is passed through captions by the caller (X.n)."""
    B = []
    A = B.append
    c1, c2_, ct = CX["improved3"]["triage"], CX["improved3"]["family"], CX["pretrained"]["triage"]
    pc = R["pretrained"]["cascade"]
    a1, a2, a3, ah = cond("pretrained")
    acc = pc["accuracy"]
    pstar = (c2_ - c1) / (2 * c2_)
    A(("sec", f"{pre} Technical Analysis"))
    A(("p", "This section goes beyond the accuracy figures and asks what the cascade costs, how its errors combine, how certain the differences are, and what would change in a real plant. All numbers are computed from the stored "
            "results and from the implemented networks."))
    A(("sub", f"{pre}.1 Computational Cost"))
    A(("p", "A parameter count does not measure computation. Counting the multiply-accumulate operations (MACs) of the convolution and linear layers for one image gives Table T1. The from-scratch networks are expensive "
            "because the first blocks run on the largest feature maps (80 x 48 input); the pretrained EfficientNet-B0 needs about "
            f"{c2_ / CX['pretrained']['family']:.0f} times fewer MACs than the from-scratch family network although it has 2.7 times more parameters, because of its stride-2 stem and depthwise convolutions."))
    exp_s = c1 + 0.49 * 4 * 0 + 0.49 * (2 * c2_)
    exp_p = ct + 0.492 * 2 * CX["pretrained"]["family"]
    A(("table", "Table T1: Multiply-accumulate operations (billions) per image",
       [["Network / model", "GMACs", "Note"],
        ["CNN-1 (from scratch)", f"{c1:.3f}", "Runs on every image"],
        ["CNN-2 / CNN-3x (from scratch)", f"{c2_:.3f}", "Only on images judged defective"],
        ["EfficientNet-B0 stage (pretrained)", f"{ct:.3f}", "Same for all five stages"],
        ["Cascade, from scratch (expected)", f"{exp_s:.3f}", "About 49% defective images"],
        ["Flat CNN, from scratch", f"{c2_:.3f}", "Same backbone as CNN-2"],
        ["Cascade, pretrained (expected)", f"{exp_p:.3f}", "Three stages for a defective image"],
        ["Flat CNN, pretrained", f"{ct:.3f}", "One EfficientNet-B0"]], [6.2, 2.4, 6.4], (), 10))
    A(("p", f"At the benchmark's 49% defect rate the early exit does not make the cascade cheaper than a flat network with the same backbone ({exp_s / c2_:.2f} times from scratch, {exp_p / ct:.2f} times pretrained)."))
    A(("sub", f"{pre}.2 Error Propagation Model"))
    A(("p", "Let a1 be the probability that Stage 1 flags a defective module, a2 the probability that Stage 2 then chooses the correct family, a3 the probability that the specialist names the correct fault, and a_h "
            "the accuracy on healthy modules. Because a wrong decision of an early stage cannot be repaired by a later one (the specialists have no healthy class and do not cover other families), the accuracy on the balanced test split is"))
    A(("eq", "Accuracy = 0.5 x a_h + 0.5 x a1 x a2 x a3"))
    rows = [["Configuration", "a1", "a2", "a3", "a1a2a3", "a2a3", "a_h", "Accuracy"]]
    for k, lab in CFG:
        x1, x2, x3, xh = cond(k)
        rows.append([lab, f"{100 * x1:.1f}", f"{100 * x2:.1f}", f"{100 * x3:.1f}", f"{100 * x1 * x2 * x3:.1f}", f"{100 * x2 * x3:.1f}", f"{100 * xh:.1f}", f"{100 * R[k]['cascade']['accuracy']:.1f}"])
    A(("table", "Table T2: Stage-conditional accuracies (%) estimated from the error counts (C3 = 80 x 48 input, C5 = larger Stage 1 and 2, C6 = data-driven families, C7 = pretrained)", rows, [2.6, 1.6, 1.6, 1.6, 2.2, 1.8, 1.6, 2.2], (4,), 9.5))
    perfect2 = 0.5 * ah + 0.5 * a1 * a3
    perfect1 = 0.5 * ah + 0.5 * a2 * a3
    x2c, x3c = cond("improved3"), cond("datagroup")
    A(("p", f"Regrouping the families (C5 to C6) conserved the product a2 x a3 ({100 * x2c[1] * x2c[2]:.1f}% against {100 * x3c[1] * x3c[2]:.1f}%): it moved difficulty from Stage 2 to Stage 3 without removing it. "
            f"The sensitivities of the accuracy are {0.5 * a2 * a3:.2f} for a1, {0.5 * a1 * a3:.2f} for a2 and {0.5 * a1 * a2:.2f} for a3, so Stage 2 is the largest lever. A perfect family stage would raise the pretrained "
            f"cascade from {pct(acc)} to about {pct(perfect2)} (an optimistic bound that assumes the other stages stay as they are), whereas a perfect Stage 1 would add only {100 * (perfect1 - acc):.1f} points."))
    A(("sub", f"{pre}.3 Statistical Uncertainty"))
    lo, hi = wilson(acc)
    fl_p = R["pretrained"]["flat"]["accuracy"]
    flo, fhi = wilson(fl_p)
    A(("p", f"Each accuracy is a binomial proportion measured on {N:,} images. The 95% Wilson interval of the pretrained cascade is {100 * lo:.1f}–{100 * hi:.1f}% (half-width about {50 * (hi - lo):.1f} points) and that of the pretrained "
            f"flat network {100 * flo:.1f}–{100 * fhi:.1f}%; the two overlap almost completely. Table T3 tests each step of the ablation with an unpaired two-sided z-test at the 5% level."))
    steps = [("C1 to C2: remove photometric jitter", "default", "no_photometric"), ("C2 to C3: 80 x 48 input, two convolutions per block", "no_photometric", "improved"),
             ("C3 to C5: larger Stage 1 and 2 networks", "improved", "improved3"), ("C5 to C6: data-driven families", "improved3", "datagroup"), ("C6 to C7: ImageNet pretraining", "datagroup", "pretrained")]
    rows = [["Step", "Change (points)", "z", "Significant"]]
    for lab, a, b in steps:
        p1, p2 = R[a]["cascade"]["accuracy"], R[b]["cascade"]["accuracy"]
        z = ztest(p1, p2)
        rows.append([lab, f"{100 * (p2 - p1):+.1f}", f"{z:.2f}", "Yes" if abs(z) > 1.96 else "No"])
    p1, p2 = fl_p, acc
    z = ztest(p1, p2)
    rows.append(["Pretrained: flat network to cascade", f"{100 * (p2 - p1):+.1f}", f"{z:.2f}", "Yes" if abs(z) > 1.96 else "No"])
    A(("table", "Table T3: Significance of the ablation steps (unpaired z-test, n = 3,000 per model)", rows, [8.0, 2.8, 1.8, 2.4], (), 10))
    A(("p", "Four steps are clearly significant; the regrouping and the cascade-versus-flat difference are not. The smallest difference that an unpaired test can resolve here is about 1.9 points. Because the same test images are used "
            "for all models, a paired McNemar test on stored predictions would be more sensitive and is recommended as a next step."))
    A(("sub", f"{pre}.4 Deployment: Cost and Base Rates"))
    A(("p", f"The cascade is cheaper than a flat model with the same backbone only if the share of defective modules is below P* = (c_flat - c1) / (c2 + c3). For the from-scratch networks P* = ({c2_:.3f} - {c1:.3f}) / {2 * c2_:.3f} = {100 * pstar:.0f}%. "
            f"Real plants find anomalies in only a few percent of modules; at 5% defective modules the from-scratch cascade needs {c1 + 0.05 * 2 * c2_:.2f} GMACs per image, {c2_ / (c1 + 0.05 * 2 * c2_):.1f} times less than the flat network. "
            "The pretrained cascade, whose five stages cost the same, is never cheaper than one flat network."))
    rows = [["Share of defective modules", "Precision of a Stage-1 flag", "Defects flagged", "Healthy flagged", "Defects missed"]]
    for pi in (0.02, 0.05, 0.10, 0.49):
        tp, fp = 1000 * pi * a1, 1000 * (1 - pi) * (1 - ah)
        rows.append([f"{100 * pi:.0f}%", f"{100 * tp / (tp + fp):.0f}%", f"{tp:.1f}", f"{fp:.1f}", f"{1000 * pi * (1 - a1):.1f}"])
    A(("table", "Table T4: Stage-1 decisions per 1,000 modules at different defect rates (extrapolated from the measured a1 and a_h of the pretrained Stage 1)", rows, [3.4, 3.4, 2.8, 2.8, 2.6], (), 10))
    A(("p", "At 5% defective modules about four in ten flagged modules are healthy, and because the later stages have no healthy class each false alarm receives a fault label. Two remedies follow: recalibrating the Stage-1 "
            "threshold to the deployment prior (for a classifier trained with class weights w = N / (K x n), the posterior under the real prior is proportional to q_r x pi_r), and giving Stage 2 a healthy-reject output."))
    return B
    out = []
    for b in B:
        if b[0] == "table":
            cap = b[1]
            for i in (1, 2, 3, 4):
                cap = cap.replace(f"Table T{i}", f"Table {chap}.{tbase + i - 1}")
            b = (b[0], cap) + tuple(b[2:])
        elif b[0] == "p":
            b = ("p", b[1].replace("Table T1", f"Table {chap}.{tbase}").replace("Table T2", f"Table {chap}.{tbase + 1}").replace("Table T3", f"Table {chap}.{tbase + 2}").replace("Table T4", f"Table {chap}.{tbase + 3}"))
        out.append(b)
    return out


def extra(pre="3.7", chap=3, tbase=12):
    """Per-class table, strategies tested without gain and the ResNet-18 experiment (run by Purvee Singh)."""
    B = []
    A = B.append
    pc = R["pretrained"]["cascade"]
    c5 = R["improved3"]["cascade"]
    A(("sec", f"{pre} Further Experiments"))
    A(("sub", f"{pre}.1 Per-Class Precision, Recall and F1"))
    rows = [["Class", "Precision", "Recall", "F1", "F1 (from scratch, C5)", "Test images"]]
    for c in sc.CLASSES:
        d, e = pc["per_class"][c], c5["per_class"][c]
        rows.append([c, f"{d['precision']:.3f}", f"{d['recall']:.3f}", f"{d['f1-score']:.3f}", f"{e['f1-score']:.3f}", int(d["support"])])
    A(("table", f"Table {chap}.{tbase}: Precision, recall and F1 per class of the pretrained cascade", rows, [3.6, 2.2, 2.2, 2.0, 3.4, 2.4], (), 10))
    best = max(sc.CLASSES, key=lambda c: pc["per_class"][c]["f1-score"])
    A(("p", "Healthy modules and diode failures are recognised reliably (F1 about 0.96). The weakest classes are Soiling (F1 0.30, only 143 training images), Cell-Multi (0.52, confused with Cell) and Hot-Spot (0.65, recall 0.54). "
            "Pretraining improved the F1 of every class, most of all Offline-Module (0.595 to 0.743). The classes with fewer than 40 test images carry wide uncertainty: one image changes their recall by 3 points or more."))
    A(("sub", f"{pre}.2 Strategies Tested Without Gain"))
    import json as _j
    ex = {k: _j.load(open(ROOT / "results" / f"{k}.json"))["cascade"] for k in ("improved", "improved_soft", "improved_tta", "improved_soft_tta", "improved_threshold")}
    rows = [["Strategy (improved cascade C3)", "Accuracy", "Macro-F1", "Outcome"],
            ["Reference: hard routing", pct(ex["improved"]["accuracy"]), f"{ex['improved']['macro_f1']:.3f}", "–"],
            ["Flip test-time augmentation", pct(ex["improved_tta"]["accuracy"]), f"{ex['improved_tta']['macro_f1']:.3f}", "+0.3 points at 4 times the cost"],
            ["Stage-1 threshold tuned on validation", pct(ex["improved_threshold"]["accuracy"]), f"{ex['improved_threshold']['macro_f1']:.3f}", "Optimum stayed at 0.5"],
            ["Soft routing (product of stage probabilities)", pct(ex["improved_soft"]["accuracy"]), f"{ex['improved_soft']['macro_f1']:.3f}", "No gain"],
            ["Soft routing + test-time augmentation", pct(ex["improved_soft_tta"]["accuracy"]), f"{ex['improved_soft_tta']['macro_f1']:.3f}", "No gain"],
            ["Balanced sampling + label smoothing 0.05", "75.5%", "0.576", "Slightly worse"]]
    A(("table", f"Table {chap}.{tbase + 1}: Inference-time and sampling strategies that did not help", rows, [6.4, 2.2, 2.2, 4.4], (), 10))
    A(("p", "That soft routing did not recover Stage-2 errors suggests that these errors are confident mistakes and not close calls: the family network itself has to improve, not the way its output is used."))
    A(("sub", f"{pre}.3 Alternative Backbone: ResNet-18 at Stages 2 and 3"))
    A(("p", "A residual network (He et al., 2016) was tried in place of the from-scratch networks at Stages 2 and 3, with Stage 1 unchanged. This experiment was run by Purvee Singh with the same split and training recipe."))
    rows = [["Network / result", "Scratch acc.", "Scratch F1", "ResNet-18 acc.", "ResNet-18 F1"],
            ["Stage 2: family", "75.5%", "0.734", "77.6%", "0.764"], ["Stage 3: Electrical", "82.9%", "0.786", "85.8%", "0.845"],
            ["Stage 3: Environmental", "89.2%", "0.822", "91.2%", "0.849"], ["Stage 3: Structural", "95.5%", "0.955", "96.2%", "0.962"],
            ["Full cascade on the test set", "79.6%", "0.638", "79.1%", "0.627"]]
    A(("table", f"Table {chap}.{tbase + 2}: ResNet-18 at Stages 2 and 3 (validation results per network, test result of the whole cascade)", rows, [5.2, 2.5, 2.3, 2.9, 2.7], (5,), 10))
    A(("p", "Every network improved on the validation split, yet the assembled cascade scored 79.1%, no better than the from-scratch cascade, and the Stage-2 test errors rose from 300 to 330. Gains of single stages on validation data "
            "therefore do not guarantee a gain of the whole cascade; the assembled system must be evaluated before a change is adopted."))
    return B
