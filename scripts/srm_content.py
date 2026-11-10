"""Chapter content for the SRM-format final report. Every number comes from results/*.json."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"
_spec = importlib.util.spec_from_file_location("mr", ROOT / "scripts" / "make_results_pdf.py")
mr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mr)
R, pct, binary_acc, CLASSES = mr.R, mr.pct, mr.binary_acc, mr.CLASSES
pr, prf = R["pretrained"]["cascade"], R["pretrained"]["flat"]
dg, i3, imp, fl = R["datagroup"]["cascade"], R["improved3"]["cascade"], R["improved"]["cascade"], R["improved"]["flat"]
rf = mr.json.load(open(ROOT / "results" / "random_forest.json"))
ps = mr.json.load(open(ROOT / "results" / "pretrained_svm.json"))
TITLE = "Cascaded Convolutional Neural Network for Solar Panel Defect Classification"

ABSTRACT = (
    "Automated inspection of large photovoltaic (PV) installations is essential for sustained power output and safety, but manual thermal surveys do not scale and flat single-stage classifiers "
    "struggle with the severe class imbalance and look-alike faults of the InfraredSolarModules benchmark. This project presents Cascaded-SolarNet, a three-stage cascade of five small convolutional "
    "neural networks that classifies the 12 classes of the benchmark from single-modality infrared images of only 24 x 40 pixels. Stage 1 separates healthy from defective panels and lets healthy panels "
    "exit early; Stage 2 assigns a defect to a fault family; and Stage 3 applies one specialist network per family to name the exact fault. The system was implemented in PyTorch and evaluated on a fixed "
    "stratified held-out split of 3,000 images that is never used for training, early stopping or model selection. Removing brightness, contrast and noise augmentation, which destroys the temperature signal "
    f"of thermal images, raised the from-scratch cascade from 61.7% to {pct(dg['accuracy'])} accuracy, and fine-tuning an ImageNet-pretrained EfficientNet-B0 at every stage reached {pct(pr['accuracy'])} accuracy, "
    f"macro-F1 {pr['macro_f1']:.3f} and {pct(binary_acc(R['pretrained']))} on the healthy-versus-defect decision. A flat classifier with the same backbone is essentially tied ({pct(prf['accuracy'])}), so the cascade did not "
    "show a clear accuracy advantage, and the family-routing stage remains the largest source of errors. Published 12-class results of up to 93.9% rely on cross-validation and pretrained features and are not directly "
    "comparable with a fixed hold-out; the report presents this comparison and its limits. Code, trained models, architecture diagrams and interactive demonstrations are provided."
)
INDEX_TERMS = "Index Terms: Solar panel defect classification, infrared thermography, cascaded convolutional neural network, early exit, hierarchical classification, transfer learning, class imbalance."

ABBREVIATIONS = [
    ("AI", "Artificial Intelligence"), ("AMP", "Automatic Mixed Precision"), ("BN", "Batch Normalisation"), ("CNN", "Convolutional Neural Network"),
    ("CV", "Cross-Validation"), ("EL", "Electroluminescence"), ("F1", "Harmonic mean of precision and recall"), ("FC", "Fully Connected (layer)"),
    ("GAP", "Global Average Pooling"), ("GPU", "Graphics Processing Unit"), ("IR", "Infrared"), ("NCA", "Neighbourhood Component Analysis"),
    ("PV", "Photovoltaic"), ("ReLU", "Rectified Linear Unit"), ("RF", "Random Forest"), ("SDG", "Sustainable Development Goal"),
    ("SVM", "Support Vector Machine"), ("TTA", "Test-Time Augmentation"), ("UAV", "Unmanned Aerial Vehicle"), ("ViT", "Vision Transformer"),
]


def _t6_rows():
    names = {"default": "C1 Default (jitter on)", "no_photometric": "C2 No photometric jitter", "improved": "C3 Improved (80x48, deeper)", "improved2": "C4 Improved2 (balanced sampling)",
             "improved3": "C5 Improved3 (bigger Stage 1/2)", "datagroup": "C6 Data-driven families", "pretrained": "C7 Pretrained EfficientNet-B0"}
    rows = [["Configuration", "Model", "Accuracy", "Macro-F1", "Kappa", "Stage 1 binary"]]
    best = None
    for k in ["default", "no_photometric", "improved", "improved2", "improved3", "datagroup", "pretrained"]:
        c = R[k]["cascade"]
        rows.append([names[k], "Cascade", pct(c["accuracy"]), f"{c['macro_f1']:.3f}", f"{c['kappa']:.3f}", pct(binary_acc(R[k]))])
        if k == "pretrained":
            best = len(rows) - 1
        if "flat" in R[k]:
            f = R[k]["flat"]
            rows.append(["", "Flat CNN", pct(f["accuracy"]), f"{f['macro_f1']:.3f}", f"{f['kappa']:.3f}", "n/a"])
    rows.append(["Raw-pixel Random Forest", "RF", pct(rf["accuracy"]), f"{rf['macro_f1']:.3f}", f"{rf['kappa']:.3f}", "n/a"])
    rows.append(["EfficientNet-B0 features + SVM", "SVM", pct(ps["accuracy"]), f"{ps['macro_f1']:.3f}", f"{ps['kappa']:.3f}", pct(ps["binary_accuracy"])])
    return rows, best


def build_blocks(charts, perclass_png, demo_png):
    B = []
    A = B.append
    chap = lambda n, t: A(("chapter", n, t))
    sec = lambda t: A(("sec", t))
    sub = lambda t: A(("sub", t))
    p = lambda t: A(("p", t))
    bl = lambda lead, t: A(("b", lead, t))

    # ------------------------------------------------------------------ CHAPTER 1
    chap(1, "INTRODUCTION")
    p("Solar farms contain thousands of photovoltaic modules that develop defects over time, including hot spots, cracks, soiling, shadowing and diode failures. Each of these reduces power output, "
      "and some, such as undetected hot spots, create a fire risk. Manual thermal inspection is slow, expensive, labour-intensive and error-prone, and it cannot scale to utility-grade installations. "
      "Deep learning applied to drone-captured infrared (IR) imagery offers an automated alternative: an unmanned aerial vehicle with a thermal camera can image thousands of modules in one flight, and a trained "
      "classifier can label each module in milliseconds.")
    p("This project builds and evaluates such a classifier on the InfraredSolarModules benchmark, a public dataset of 20,000 IR images of 24 x 40 pixels with 12 classes (one healthy class and eleven fault types). "
      "Instead of a single network that chooses directly among all twelve classes, the project proposes a cascade of small networks that answers three easier questions in turn.")
    sec("1.1 Introduction to Solar Panel Defect Classification")
    p("In an infrared image brightness encodes temperature. A healthy module heats fairly uniformly during operation, whereas a defect produces a localised temperature anomaly: a hot spot, a cold region or a "
      "characteristic thermal pattern. Classifying these patterns is therefore a pattern-recognition problem on very small images, where fine texture is absent and the position, size and shape of heat "
      "concentrations carry the information. Typical defect families are electrical faults (cells and bypass diodes), environmental conditions (soiling, shadows, vegetation) and structural damage (cracks, "
      "offline modules).")
    p("The task is difficult for three reasons. The classes are strongly imbalanced, because half of the images show healthy modules. Several fault types look alike at low resolution. And the system has to be "
      "cheap enough to run on drone hardware. These properties motivate a design that removes the dominant class first and then specialises.")
    sec("1.2 Problem Statement and Description")
    p("Most existing CNN approaches use a flat, single-stage classifier that must choose directly among all classes. This design has two weaknesses. First, visually similar fault types produce overlapping "
      "features, so the network confuses them. Second, healthy panels vastly outnumber faulty ones, which biases the network towards the majority class and hurts the rare faults. The literature reviewed in this "
      "project reports that accuracy falls from about 73% for binary tasks to about 68% for multi-class tasks across eight architectures, and that single-stage results on this benchmark stop at about 93%.")
    p("The problem addressed here is to classify a single-modality 24 x 40 pixel IR image of a PV module into one of 12 classes accurately, with a model small enough for edge use, and to understand where "
      "the errors of the proposed design come from. The literature survey identified six gaps:")
    bl("G1.", "Dual-sensor approaches are impractical. Models reaching 98% to 99% accuracy need paired thermal and visual data that a single-camera drone cannot supply.")
    bl("G2.", "Scaling the model gives negligible gains. Heavier backbones add little accuracy over lightweight ones.")
    bl("G3.", "Bounding-box detectors split thin defects. Detection models mislocalise fine crack boundaries.")
    bl("G4.", "Naive cascades add latency. Detection-plus-classification pipelines are too slow for real-time use.")
    bl("G5.", "Accuracy drops as the number of classes grows, from roughly 73% for binary to 68% for multi-class classification.")
    bl("G6.", "Similar defect types are systematically confused, and the confusion persists in flat classifiers.")
    sec("1.3 Motivation")
    p("The practical cost of misclassification is concrete. A missed hot spot can become a fire; a false alarm sends a maintenance crew to a healthy module. Because most modules are healthy, a classifier that "
      "wastes effort on them is both slow and biased. A hierarchical design addresses this in a natural way: a small first network handles the easy and most frequent decision and passes only suspicious modules "
      "on, and later networks can then spend their capacity on the genuinely hard distinctions.")
    p("A second motivation is scientific. Cascades are often assumed to beat flat classifiers, but the literature offers few controlled comparisons on a small thermal benchmark. This project measures that "
      "claim with identical building blocks, identical data splits and a stage-by-stage attribution of errors, and reports the result honestly, including where the cascade does not help.")
    sec("1.4 Sustainable Development Goal of the Project")
    p("The project contributes most directly to SDG 7, Affordable and Clean Energy. Solar installations that are inspected faster and more cheaply lose less energy to undetected defects, which improves the "
      "return on clean-energy investment. It also supports SDG 13, Climate Action, because every percentage point of recovered solar output displaces fossil generation, and SDG 9, Industry, Innovation and "
      "Infrastructure, through low-cost automated inspection that can run on small drones.")

    # ------------------------------------------------------------------ CHAPTER 2
    chap(2, "LITERATURE SURVEY")
    p("This chapter reviews deep-learning approaches to PV defect detection and classification, with emphasis on infrared imagery and on the InfraredSolarModules benchmark, and derives the research gaps and "
      "objectives of the project. The studies were re-checked against their primary sources while preparing the final report; where a result had been attributed to the wrong paper in the first survey, "
      "it has been corrected below.")
    sec("2.1 Overview of PV Defect Detection and Classification Methods")
    p("Automated analysis of PV imagery has moved through several generations. Early work extracted handcrafted thermal features and classified them with classical learners such as random forests. "
      "The second generation applied convolutional networks end to end, from small custom CNNs to ImageNet-pretrained VGG, ResNet and EfficientNet backbones. A third thread moved to object detection and "
      "segmentation (Faster R-CNN, Cascade R-CNN, VarifocalNet, YOLO) to localise defects. The most recent work explores cascades, ensembles, transfer learning with heavy augmentation and feature-selection "
      "pipelines, usually at the cost of extra sensors or computation.")
    sec("2.2 Deep Learning Approaches for PV Defect Analysis")
    sub("2.2.1 The InfraredSolarModules Benchmark and Its Baselines")
    p("Millendorf, Obropta and Vadhavkar of Raptor Maps released the InfraredSolarModules dataset at the ICLR 2020 Workshop on AI for Earth Sciences [1]. The dataset paper describes the data and lists "
      "benchmark classification results as a goal, but it reports no model accuracies. Baselines on the dataset come from later papers: a random forest at 84.0% [4], a custom CNN with augmentation at 92.5% "
      "for binary anomaly detection and 78.85% for eight selected classes [3], and an EfficientNet-B0 feature pipeline with an SVM at 93.93% for all 12 classes under 10-fold cross-validation [2].")
    sub("2.2.2 Cascaded and Dual-Sensor Approaches")
    p("Pathak and Patil proposed a Binary Cascaded CNN that first separates healthy from faulty modules and then identifies the fault, fusing thermal images with co-registered visible-spectrum images [5]. "
      "It reports 99.8% precision and 98.5% accuracy, the highest in the survey, but it needs paired dual-sensor data and substantial computation, which a single-camera drone does not provide. "
      "Barraz et al. describe a cascading ensemble with a voting CNN reaching 88% on InfraredSolarModules, as cited in [4].")
    sub("2.2.3 Lightweight Networks")
    p("PV-faultNet benchmarked architectures from compact CNNs to Siamese and multi-spectral models and found that a carefully designed compact CNN reaches 93% accuracy at 8 ms per image, with larger "
      "backbones adding little [6]. The authors conclude that the bottleneck is the low information content of 24 x 40 pixel images and not model capacity, which supports small per-stage networks.")
    sub("2.2.4 Detection and Segmentation Pipelines")
    p("Detection-based work extends VarifocalNet and compares it with Cascade R-CNN and YOLOv7 for localising defects in drone IR images [7]. Both detectors struggle with thin linear defects such as cracks, "
      "producing fragmented or offset boxes. Otamendi et al. combine Faster R-CNN with an EfficientNet classifier on electroluminescence images [8]; it is precise but needs powered panels imaged in darkness and "
      "adds seconds of latency. These results motivate whole-image classification for the low-resolution IR domain.")
    sub("2.2.5 Transfer Learning, Ensembles and Feature Selection")
    p("A field-adapted ResNet-50 with 8x augmentation reaches 98.59% on field data but is heavy for edge use. CNN ensembles with Bacterial Foraging Optimisation for feature selection reach 98.75% with VGG-16 "
      "but must run five backbones. Tella et al. benchmarked eight CNN and ViT architectures on the ELPV benchmark and found an average of about 73% for binary and 68% for multi-class classification [9]. "
      "A four-class RGB study (normal, cracked, dusty, shadowed) reaches 96% to 98% but its coarse labels hide the real difficulty. Table 2.1 summarises the studies.")
    A(("table", "Table 2.1: Summary of the reviewed studies",
       [["#", "Study", "Approach", "Key result", "Limitation"],
        ["1", "Millendorf et al., 2020", "Dataset paper (InfraredSolarModules)", "No model results reported", "Baselines come from later work"],
        ["2", "Pathak and Patil, 2025", "Binary cascaded CNN with thermal-visual fusion", "99.8% precision, 98.5% accuracy", "Needs paired thermal and visual data"],
        ["3", "PV-faultNet, 2024", "Compact CNN versus Siamese and multi-spectral", "93% at 8 ms per image", "Larger backbones gain little"],
        ["4", "Improved VarifocalNet, 2024", "Cascade R-CNN, YOLOv7 comparison", "High coarse detection confidence", "Mislocalises thin cracks"],
        ["5", "Otamendi et al., 2021", "Faster R-CNN + EfficientNet on EL images", "High precision", "Too slow for real-time drones"],
        ["6", "Field-adapted ResNet-50", "Transfer learning with 8x augmentation", "98.59% on field data", "High resource demand"],
        ["7", "Tella et al., 2025", "8 CNN/ViT models on ELPV", "About 68% multi-class vs 73% binary", "Large binary to multi-class drop"],
        ["8", "CNN on RGB images", "4-class normal/cracked/dusty/shadowed", "96% to 98%", "Coarse classes only"],
        ["9", "CNN + Bacterial Foraging Optimisation", "Five pretrained CNNs with feature selection", "VGG-16 + BFO: 98.75%", "Extra tuning, five backbones"],
        ["10", "Duranay, 2023", "EfficientNet-B0 features, NCA, SVM", "93.93% on 12 classes (10-fold CV)", "Cross-validation, pretrained features"]],
       [0.8, 3.4, 4.2, 3.6, 3.8], (), 9))
    sec("2.3 Limitations of Existing Methods")
    p("Four limitations recur across the reviewed work. The highest accuracies depend on information a drone does not have (paired visible images, powered-panel electroluminescence) or on heavy computation "
      "(ensembles, large backbones). Detection pipelines add failure modes on thin defects. Results are reported under different protocols, mostly cross-validation, which makes them hard to compare, "
      "and near-duplicate crops in the dataset can inflate cross-validated scores. Finally, almost no study reports where a hierarchical model loses accuracy, so the benefit of cascading remains unmeasured.")
    sec("2.4 Research Gaps and Contributions")
    p("Table 2.2 maps each gap to the contribution of this project.")
    A(("table", "Table 2.2: Research gaps and the contributions that address them",
       [["Gap", "Research gap", "Contribution"],
        ["G1", "Dual-sensor approaches are impractical", "C4: single-modality design using only single-camera IR images"],
        ["G2", "Scaling the model gives negligible gains", "C1: five small per-stage networks instead of one large model"],
        ["G3", "Bounding boxes fragment thin defects", "C5: whole-image classification with no localisation step"],
        ["G4", "Naive cascades add latency", "C2: early-exit binary front end for healthy panels"],
        ["G5", "Accuracy drops as classes grow", "C1: hierarchical decomposition into 2-, 3- and 2-to-6-way decisions"],
        ["G6", "Similar defects are confused", "C3: family grouping before fine classification, semantic and data-driven"]],
       [1.4, 6.4, 8.0], (), 10))
    sec("2.5 Research Objectives")
    bl("O1.", "To design a three-stage cascade of small CNNs for 12-class IR solar module classification from a single sensor.")
    bl("O2.", "To implement the system reproducibly in PyTorch with a fixed stratified split, tests and published code.")
    bl("O3.", "To measure, with controlled ablations, which design choices (augmentation, input size, capacity, grouping, pretraining) matter.")
    bl("O4.", "To attribute the cascade's errors to the stage that caused them and to compare it with a flat classifier and with published results.")
    sec("2.6 Plan of Action")
    p("The work followed six steps: (1) literature survey and gap analysis; (2) dataset preparation with a stratified 70/15/15 split; (3) implementation of the five-network cascade and a flat baseline; "
      "(4) controlled ablations of augmentation, input size, network capacity, family grouping and inference variants; (5) fine-tuning of an ImageNet-pretrained backbone at every stage; and "
      "(6) evaluation, error attribution, comparison with published work, architecture diagrams, demonstrations and this report.")

    # ------------------------------------------------------------------ CHAPTER 3
    chap(3, "PROBLEM FORMULATION")
    p("This chapter defines the classification problem precisely, decomposes it into sub-problems, states the constraints under which the system is built and explains why a naive solution is not enough.")
    sec("3.1 Formal Problem Definition")
    p("Let x be a single-channel infrared image of one PV module with 40 rows and 24 columns, x in R^(40 x 24) with values scaled to [0, 1]. Let the label set be Y = {y0, y1, ..., y11}, where y0 is "
      "No-Anomaly and y1 to y11 are the fault classes. The goal is to learn a function f that maps x to a label so that the expected accuracy and the macro-averaged F1 score over Y are maximised on unseen "
      "images, subject to the constraints of Section 3.3.")
    p("The label set carries a hierarchy: Y = {y0} U F_E U F_V U F_S, where the families F_E (Electrical), F_V (Environmental) and F_S (Structural) partition the eleven faults. This structure is exploited "
      "by the cascade: f(x) = g_fam(x)(x) if the triage function h(x) predicts a defect, and y0 otherwise.")
    sec("3.2 Decomposition into Sub-Problems")
    sub("3.2.1 Healthy Versus Defective (Triage)")
    p("The first sub-problem is the binary function h(x) in {healthy, defect}. It is the easiest decision in the system and the one that removes the dominant class. An error here is irreversible: a missed defect "
      "is never seen by later stages, and a false alarm only costs computation and a wrong label.")
    sub("3.2.2 Fault Family")
    p("For images judged defective, the second sub-problem is the three-way function that selects a family. Faults that share a family should look similar to each other and different from other families, so "
      "this decision should be much easier than an eleven-way choice. Section 6 tests how well this holds.")
    sub("3.2.3 Fine-Grained Subtype")
    p("Within a family, a specialist function chooses the exact fault: six classes for Electrical, three for Environmental and two for Structural. Each specialist is trained only on its own family and can "
      "spend all its capacity on the distinctions inside it.")
    sec("3.3 Constraints and Assumptions")
    sub("3.3.1 Dataset")
    p("The InfraredSolarModules dataset contains 20,000 images from real drone flights, 24 pixels wide and 40 pixels tall. The classes and their distribution in the stratified split used throughout this report are given in Table 3.1.")
    A(("table", "Table 3.1: Class distribution and stratified split (seed 42)",
       [["Class", "Family", "Total", "Train", "Validation", "Test"],
        ["No-Anomaly", "Healthy", 10000, 7000, 1500, 1500], ["Cell", "Electrical", 1877, 1314, 282, 281], ["Cell-Multi", "Electrical", 1288, 902, 193, 193],
        ["Diode", "Electrical", 1499, 1049, 225, 225], ["Diode-Multi", "Electrical", 175, 123, 26, 26], ["Hot-Spot", "Electrical", 249, 174, 38, 37],
        ["Hot-Spot-Multi", "Electrical", 246, 172, 37, 37], ["Soiling", "Environmental", 204, 143, 30, 31], ["Shadowing", "Environmental", 1056, 739, 158, 159],
        ["Vegetation", "Environmental", 1639, 1147, 246, 246], ["Cracking", "Structural", 940, 658, 141, 141], ["Offline-Module", "Structural", 827, 579, 124, 124],
        ["Total", "", 20000, 14000, 3000, 3000]],
       [3.4, 3.0, 2.2, 2.2, 2.6, 2.2], (13,), 10))
    sub("3.3.2 Input Modality and Image Size")
    p("Only single-channel IR images are used, because a single-camera drone cannot supply paired visible images. The images are tiny, so networks must recognise where heat is concentrated and what shape it takes rather than rely on texture.")
    sub("3.3.3 Whole-Image Classification")
    p("The system classifies the whole module image and does not localise defects. Detection pipelines were shown in the literature to fragment thin defects, and the dataset provides one label per crop and no boxes.")
    sub("3.3.4 Evaluation Protocol")
    p("All results are measured on one fixed, stratified held-out test split of 3,000 images. The test images are never used for training, early stopping, normalisation statistics or model selection; "
      "all tuning uses a separate 3,000-image validation split.")
    sec("3.4 Why the Problem Is Non-Trivial")
    sub("3.4.1 Class Imbalance")
    p("Half of the images are healthy, and the rarest classes have 175 to 249 images in total (123 to 174 for training). A classifier that always answers No-Anomaly scores 50% accuracy and learns nothing about faults.")
    sub("3.4.2 Look-Alike Faults")
    p("Several classes differ only in how many cells or diodes are affected (Cell and Cell-Multi, Diode and Diode-Multi, Hot-Spot and Hot-Spot-Multi), and at 24 x 40 pixels these differences are subtle.")
    sub("3.4.3 Error Propagation")
    p("In a cascade, a wrong decision at an early stage cannot be corrected later. The overall accuracy of a defective image is roughly the product of the accuracies of the stages it passes through, so the early "
      "stages must be very accurate. Whether a cascade nevertheless beats a flat classifier is an empirical question, which this project answers by measurement.")
    sec("3.5 Problem Statement Summary")
    p("Given single-modality 40 x 24 IR images of PV modules, build a small-network system that classifies them into 12 classes, evaluate it on a clean held-out split, attribute its errors to individual "
      "stages and compare it with a flat classifier and with published results.")

    # ------------------------------------------------------------------ CHAPTER 4
    chap(4, "PROPOSED SOLUTION")
    p("This chapter describes the design decisions of Cascaded-SolarNet and relates each one to a gap identified in Chapter 2. The detailed architecture and implementation follow in Chapter 5.")
    sec("4.1 Hierarchical Three-Stage Cascade")
    p("Cascaded-SolarNet replaces one large 12-way classifier with five small specialist CNNs arranged in three stages. Each network answers one small, well-defined question, and the answer of one stage decides "
      "which network runs next. A healthy panel passes through only one network; a defective panel passes through exactly three: the triage network, the family network and one specialist.")
    sec("4.2 Early Exit for Healthy Panels")
    p("Stage 1 is a binary classifier. When it predicts No-Anomaly the pipeline outputs that label and stops. Because healthy panels are the large majority, this removes the dominant class from all later "
      "networks, handles class imbalance by structure instead of relying only on loss re-weighting, and in principle saves computation on most images.")
    sec("4.3 Fault-Family Grouping")
    p("Stage 2 routes each defect to one of three families. The semantic grouping follows the physical origin of the faults: Electrical (Cell, Cell-Multi, Diode, Diode-Multi, Hot-Spot, Hot-Spot-Multi), "
      "Environmental (Soiling, Shadowing, Vegetation) and Structural (Cracking, Offline-Module). As an alternative, a data-driven grouping was derived by testing every partition of the eleven faults into three families "
      "of at least three classes and choosing the one that minimises the confusions of a trained flat network on the validation split: group A (Cell, Cell-Multi, Soiling, Vegetation, Cracking), "
      "group B (Diode, Diode-Multi, Offline-Module) and group C (Hot-Spot, Hot-Spot-Multi, Shadowing). This reduced cross-family confusions on the validation split from 264 to 87.")
    sec("4.4 Two Backbones: Lightweight From Scratch and ImageNet-Pretrained")
    p("Every stage network is available in two variants that share the cascade logic. The from-scratch variant uses small convolutional blocks and has no external data. The pretrained variant uses an "
      "ImageNet-pretrained EfficientNet-B0 with a new classification head and is fine-tuned on the stage's data. The comparison of the two variants quantifies what pretraining adds on this tiny thermal domain.")
    sec("4.5 Single-Modality, Whole-Image Pipeline")
    p("The system takes only a single-channel IR image and outputs a label for the whole module. It needs no second sensor, no alignment step and no localisation, which removes the failure modes G1 and G3.")
    sec("4.6 Summary of Contributions")
    A(("table", "Table 4.1: Contributions and the gaps they address",
       [["Contribution", "Description", "Gaps addressed"],
        ["C1", "Hierarchical three-stage cascade in place of a flat 12-class classifier", "G2, G5"],
        ["C2", "Early-exit binary front end that removes the dominant No-Anomaly class", "G4, imbalance"],
        ["C3", "Fault-family grouping (semantic and data-driven) before fine classification", "G6"],
        ["C4", "Single-modality design using only single-camera IR images", "G1"],
        ["C5", "Pure classification pipeline without localisation or bounding boxes", "G3"],
        ["C6", "Controlled ablations and per-stage error attribution, with a leak-free evaluation protocol", "Comparability"]],
       [2.6, 10.0, 3.2], (), 10))

    # ------------------------------------------------------------------ CHAPTER 5
    chap(5, "ARCHITECTURE AND IMPLEMENTATION")
    sec("5.1 Overview of the Implemented System")
    p("Figure 1 shows the implemented cascade. The system is implemented in Python 3.12 with PyTorch 2.11 and torchvision, scikit-learn for metrics, and Matplotlib for plots. Training and evaluation ran on an "
      "NVIDIA GeForce RTX 5060 Laptop GPU with 8 GB of memory. The code is organised as a Python package with separate modules for the class hierarchy, data handling, models, training and evaluation, plus "
      "configuration files, scripts and a test suite of 15 tests that run on synthetic data.")
    A(("fig", FIG / "overall_cascade.png", 12.0, "Fig. 1. The implemented three-stage Cascaded-SolarNet architecture."))
    sec("5.2 Data Pipeline")
    sub("5.2.1 Split and Normalisation")
    p("The 20,000 images are split once into 70% training (14,000), 15% validation (3,000) and 15% test (3,000) images, stratified by class with a fixed seed so every model sees the same split. Pixel values are scaled "
      "to [0, 1] and normalised with the mean and standard deviation of the training split only.")
    sub("5.2.2 Augmentation")
    p("During training only random horizontal and vertical flips (probability 0.5 each) are applied. Brightness, contrast and noise jitter were implemented and tested but removed, because in thermal images brightness "
      "is temperature and jittering it lowered accuracy (Section 6.3). Quarter-turn rotations are not used because they would change the 40 x 24 shape.")
    sub("5.2.3 Handling Class Imbalance")
    p("Each network is trained with a class-weighted cross-entropy loss whose weights are N / (K x n_r), where N is the number of training samples of that network, K the number of its classes and n_r the "
      "count of class r. Stage 1 additionally benefits from the early-exit structure itself, and the metrics reported include macro-F1, which weights every class equally.")
    sec("5.3 Stage 1: CNN-1, Binary Triage")
    p("CNN-1 answers whether a module is healthy or defective. In the from-scratch variant it has three blocks of two 3 x 3 convolutions, each followed by BatchNorm and ReLU, then 2 x 2 max pooling, with 64, 128 and 128 "
      "channels, followed by global average pooling, dropout (0.2) and a fully connected layer with two outputs. The 80 x 48 upsampled input is reduced to 64 x 40 x 24, 128 x 20 x 12 and 128 x 10 x 6 feature maps. "
      "A prediction of No-Anomaly ends the pipeline (the early exit); a defect continues to Stage 2.")
    A(("fig", FIG / "stage1_cnn1.png", 15.5, "Fig. 2. Stage 1: CNN-1, binary triage (layers and feature-map sizes)."))
    sec("5.4 Stage 2: CNN-2, Fault-Family Classifier")
    p("CNN-2 has the same three-block design with 128, 256 and 128 channels, dropout 0.4 and three outputs. It is trained only on defective images labelled by family. Its highest-scoring output selects which "
      "specialist runs next. Faults of one family should look alike, so the decision is easier than choosing among eleven faults directly.")
    A(("fig", FIG / "stage2_cnn2.png", 15.5, "Fig. 3. Stage 2: CNN-2, fault-family classifier."))
    sec("5.5 Stage 3: Fine-Grained Specialists")
    p("Three specialists share the 128-256-128 design and differ only in their output layer: CNN-3a (Electrical, six classes), CNN-3b (Environmental, three classes) and CNN-3c (Structural, two classes). "
      "Each is trained only on the defective images of its own family, so it can spend all its capacity on the distinctions inside that family. The final label is the family's class offset plus the "
      "specialist's prediction.")
    A(("fig", FIG / "stage3_specialists.png", 15.5, "Fig. 4. Stage 3: the three specialists and their classes."))
    A(("table", "Table 5.1: Architecture summary of the five networks (from-scratch variant)",
       [["Network", "Stage", "Input", "Layers", "Output"],
        ["CNN-1", "1: triage", "IR image, 80 x 48", "3 blocks 2 x (Conv3x3, BN, ReLU) + MaxPool, 64-128-128; GAP, Dropout 0.2, FC", "2 classes"],
        ["CNN-2", "2: family", "Defective image", "Same, 128-256-128; Dropout 0.4, FC", "3 families"],
        ["CNN-3a", "3: Electrical", "Routed image", "Same as CNN-2, FC with 6 outputs", "6 classes"],
        ["CNN-3b", "3: Environmental", "Routed image", "Same as CNN-2, FC with 3 outputs", "3 classes"],
        ["CNN-3c", "3: Structural", "Routed image", "Same as CNN-2, FC with 2 outputs", "2 classes"]],
       [1.8, 2.4, 2.8, 6.6, 2.2], (), 9.5))
    sec("5.6 Pretrained Variant")
    p("In the pretrained variant each of the five networks is an ImageNet-pretrained EfficientNet-B0 whose classifier is replaced by dropout and a fully connected layer with the stage's number of outputs. "
      "The grayscale image is resized to 160 x 96 and repeated to three channels inside the model, so the rest of the pipeline is unchanged. The five networks hold about 20 million parameters in total, against about "
      "6.5 million for the from-scratch cascade.")
    sec("5.7 Training Procedure")
    p("Each network is trained independently on the data matching its task (ground-truth routing). Training uses the Adam optimiser with weight decay 1e-4, batch size 64, and the class-weighted loss of Section 5.2.3. "
      "The learning rate is 5e-4 for the from-scratch networks and 3e-4 for the pretrained ones, which also use mixed-precision training. The checkpoint with the best validation macro-F1 is kept, and training stops when "
      "that metric has not improved for 25 epochs (6 for the pretrained networks). At test time the stages run in sequence on the held-out images.")
    sec("5.8 Evaluation Tools and Demonstrations")
    p("The evaluation script runs the whole cascade end to end on the test split and reports accuracy, macro-F1, Cohen's kappa, per-class scores, the confusion matrix and the error attribution per stage "
      "(missed defects and false alarms at Stage 1, wrong family at Stage 2, wrong subtype at Stage 3). Further scripts reproduce the baselines, run 10-fold cross-validation, search for the data-driven family "
      "grouping and generate the figures and reports. Two demonstrations show the cascade at work: a command-line tool that draws each stage's confidence bars for chosen test images (Figure 5), and an "
      "interactive browser page in which a class can be selected and the three stages reveal their decisions one after another.")
    A(("fig", demo_png, 15.5, "Fig. 5. Output of the live demonstration: each stage's decision and confidence for Hot-Spot test images."))
    sec("5.9 Chapter Summary")
    p("The system consists of five small stage networks in two variants, a leak-free data pipeline, a class-weighted training procedure and an evaluation module that attributes every error to a stage. "
      "The next chapter evaluates it.")

    # ------------------------------------------------------------------ CHAPTER 6
    chap(6, "RESULTS AND ANALYSIS")
    sec("6.1 Experimental Setup")
    sub("6.1.1 Dataset and Protocol")
    p("All numbers are measured on the 3,000-image held-out test split of Table 3.1 and are single runs per configuration (seed 42). The test images are not used for training, early stopping, normalisation or "
      "model selection.")
    sub("6.1.2 Implementation and Training Configuration")
    A(("table", "Table 6.1: Training configuration of the evaluated models",
       [["Group", "Parameter", "Value"],
        ["Data", "Split", "Stratified 70/15/15 (14,000 / 3,000 / 3,000), seed 42"], ["Data", "Normalisation", "Mean and standard deviation of the training split"],
        ["Data", "Augmentation", "Random horizontal and vertical flips (p = 0.5)"],
        ["From scratch", "Input size", "80 x 48 (2x bilinear upsampling)"], ["From scratch", "Blocks and channels", "3 blocks of 2 x (Conv3x3, BN, ReLU) + MaxPool; 64-128-128 (Stage 1), 128-256-128 (others)"],
        ["From scratch", "Dropout", "0.2 (Stage 1), 0.4 (Stage 2 and 3)"], ["From scratch", "Learning rate, patience", "5e-4; 25 epochs (maximum 150)"],
        ["Pretrained", "Backbone", "EfficientNet-B0, ImageNet weights, new head"], ["Pretrained", "Input size", "160 x 96, channel repeated to 3"],
        ["Pretrained", "Dropout", "0.2 (Stage 1), 0.3 (others)"], ["Pretrained", "Learning rate, patience", "3e-4; 6 epochs (maximum 25), mixed precision"],
        ["Common", "Optimiser", "Adam, weight decay 1e-4, batch size 64"], ["Common", "Loss", "Class-weighted cross-entropy, w = N / (K x n)"],
        ["Common", "Model selection", "Early stopping on validation macro-F1"], ["Common", "Hardware, software", "NVIDIA RTX 5060 Laptop GPU; PyTorch 2.11"]],
       [2.6, 4.0, 9.2], (), 10))
    sub("6.1.3 Evaluation Metrics")
    p("Accuracy is reported together with macro-F1 and Cohen's kappa, because accuracy alone hides poor performance on rare classes under class imbalance. The Stage 1 binary accuracy measures the healthy-versus-defect "
      "decision, and per-stage error counts attribute each mistake to the first stage that went wrong.")
    sub("6.1.4 Configurations")
    A(("table", "Table 6.2: Definition of the evaluated configurations",
       [["ID", "Variant", "Modification"],
        ["C1", "Default", "Flips plus brightness (20%), contrast (15%) and noise jitter; 40 x 24 input; 60 epochs"],
        ["C2", "No photometric jitter", "C1 with flips only; longer schedule (100 epochs)"], ["C3", "Improved", "C2 with 80 x 48 input and two convolutions per block; 150 epochs"],
        ["C4", "Improved2", "C3 with balanced sampling and label smoothing instead of loss weights"], ["C5", "Improved3", "C3 with larger Stage 1 and Stage 2 networks"],
        ["C6", "Data-driven families", "C5 with the data-driven family grouping of Section 4.3"], ["C7", "Pretrained", "EfficientNet-B0 (ImageNet) fine-tuned at every stage"]],
       [1.2, 4.2, 10.4], (), 10))
    sec("6.2 Overall Performance")
    p(f"The best result is the pretrained cascade (C7) with {pct(pr['accuracy'])} accuracy, macro-F1 {pr['macro_f1']:.3f}, Cohen's kappa {pr['kappa']:.3f} and {pct(binary_acc(R['pretrained']))} on the Stage 1 healthy-versus-defect "
      f"decision. The best from-scratch cascade reaches {pct(dg['accuracy'])} accuracy. Table 6.3 lists all configurations and baselines, and Figure 6 plots them.")
    rows, best = _t6_rows()
    A(("table", "Table 6.3: Test results by configuration", rows, [5.6, 2.2, 2.2, 2.2, 1.8, 2.4], (best,), 9.5))
    A(("fig", charts["prog"], 14.5, "Fig. 6. Cascade test accuracy and macro-F1 by configuration (dashed line: from-scratch flat CNN)."))
    sec("6.3 Ablation Findings")
    bl("Brightness and noise jitter hurt.", "Removing brightness, contrast and noise augmentation raised cascade accuracy from 61.7% to 71.1% (C1 to C2). In thermal images brightness is temperature, so jittering it removes the signal that separates the classes.")
    bl("Input size and capacity helped.", f"Upsampling to 80 x 48 with two convolutions per block raised accuracy to {pct(imp['accuracy'])} (C3), and enlarging the Stage 1 and Stage 2 networks raised it to {pct(i3['accuracy'])} (C5); Stage 1 binary accuracy rose from {pct(binary_acc(R['improved']))} to {pct(binary_acc(R['improved3']))}.")
    bl("Pretraining helped most late in the project.", f"Fine-tuning ImageNet-pretrained EfficientNet-B0 at every stage raised the cascade from {pct(dg['accuracy'])} to {pct(pr['accuracy'])} accuracy and macro-F1 from {dg['macro_f1']:.3f} to {pr['macro_f1']:.3f}, and training took minutes instead of an hour.")
    bl("The cascade and a flat classifier are essentially tied.", f"With the pretrained backbone the cascade scores {pct(pr['accuracy'])} / {pr['macro_f1']:.3f} / {pr['kappa']:.3f} (accuracy, macro-F1, kappa) and the flat CNN {pct(prf['accuracy'])} / {prf['macro_f1']:.3f} / {prf['kappa']:.3f}, while the cascade has about five times the parameters. From scratch the cascade was ahead on accuracy and kappa but not on macro-F1.")
    bl("Tried without gain.", "Soft routing (multiplying stage probabilities), flip test-time augmentation, tuning the Stage 1 threshold on validation data (the optimum stayed at 0.5) and balanced sampling with label smoothing (C4).")
    sec("6.4 Error Analysis by Stage")
    p("Table 6.4 attributes each test error to the first stage that went wrong.")
    rows = [["Configuration", "Stage 1 missed defect", "Stage 1 false alarm", "Stage 2 wrong family", "Stage 3 wrong subtype", "Total"]]
    for k, lab in [("improved", "C3 Improved"), ("improved3", "C5 Improved3"), ("datagroup", "C6 Data-driven"), ("pretrained", "C7 Pretrained")]:
        s_ = R[k]["cascade"]["stages"]
        rows.append([lab, s_["stage1_missed_defects"], s_["stage1_false_alarms"], s_["stage2_wrong_family"], s_["stage3_wrong_subtype"], s_["total_errors"]])
    A(("table", "Table 6.4: Test errors per stage (out of 3,000 images)", rows, [3.6, 2.5, 2.4, 2.6, 2.7, 1.5], (4,), 10))
    A(("fig", charts["stages"], 14.0, "Fig. 7. Where the cascade makes its errors, by configuration."))
    p(f"Pretraining reduced errors at every stage compared with C5 (total {i3['stages']['total_errors']} to {pr['stages']['total_errors']}), but Stage 2, the family decision, remains the largest source "
      f"({pr['stages']['stage2_wrong_family']} of {pr['stages']['total_errors']} errors), and an error there cannot be corrected by Stage 3. The data-driven grouping cut Stage 2 errors from "
      f"{i3['stages']['stage2_wrong_family']} to {dg['stages']['stage2_wrong_family']} but moved the difficulty into Stage 3 ({i3['stages']['stage3_wrong_subtype']} to {dg['stages']['stage3_wrong_subtype']} errors), "
      f"so the net accuracy changed little. About {pct(pr['stages']['early_exit_rate'])} of the test images leave the cascade at Stage 1.")
    sec("6.5 Per-Class Performance")
    p("The rare classes are the weak point. They have only 123 to 174 training images, and several look alike. Class weighting, stratified splitting and macro-F1 reduce but do not remove the problem.")
    rows = [["Class", "Test images", "Cascade, from scratch (C5)", "Cascade, pretrained (C7)", "Flat CNN, pretrained"]]
    for c in CLASSES:
        rows.append([c, int(pr["per_class"][c]["support"]), f"{R['improved3']['cascade']['per_class'][c]['f1-score']:.2f}", f"{pr['per_class'][c]['f1-score']:.2f}", f"{prf['per_class'][c]['f1-score']:.2f}"])
    A(("table", "Table 6.5: Per-class F1 on the test set", rows, [3.4, 2.2, 4.0, 3.4, 3.0], (), 10))
    A(("fig", perclass_png, 14.5, "Fig. 8. Per-class F1 on the test set."))
    sec("6.6 Comparison with Published Work")
    p("The dataset paper reports no model accuracies, and the figures often attributed to it (Random Forest 84%, CNN 92.5%, EfficientNet + SVM 93%) come from later papers that differ in task, protocol and features. "
      "Table 6.6 therefore places published results next to ours but does not rank them.")
    rows = [["Paper", "Method", "Task and protocol", "Reported", "Checked"],
            ["Duranay 2023 [2]", "EfficientNet-B0 features (ImageNet), NCA selection, SVM", "12 classes, 10-fold cross-validation", "93.93% accuracy, 89.82% F1", "Full paper read"],
            ["Alves et al. 2021 [3]", "Custom CNN with augmentation", "Binary detection; 8 selected classes; cross-validation", "92.5% binary; 78.85% for 8 classes", "Abstract only"],
            ["Corley et al. 2025 [4]", "Random Forest", "Anomaly classification, split not stated", "84.0%", "Paper text read"],
            ["Barraz 2025; Gopalakrishnan 2024, as cited in [4]", "Cascading ensemble; NASNet + LSTM", "Not stated", "88%; 85%", "As cited"],
            ["Millendorf et al. 2020 [1]", "Dataset paper", "-", "No model results", "Full paper read"],
            ["Ours: pretrained cascade", "EfficientNet-B0 at each of 5 stages", "12 classes, fixed 70/15/15 hold-out", f"{pct(pr['accuracy'])}, macro-F1 {pr['macro_f1']:.3f}", "Our results"],
            ["Ours: from-scratch cascade", "Five small CNNs", "Same hold-out", f"{pct(dg['accuracy'])}, macro-F1 {dg['macro_f1']:.3f}", "Our results"],
            ["Ours: Stage 1 only (pretrained)", "Binary triage", "Healthy vs defect, same hold-out", pct(binary_acc(R['pretrained'])), "Our results"],
            ["Ours: flat CNN / Random Forest", "EfficientNet-B0 / raw pixels", "12 classes, same hold-out", f"{pct(prf['accuracy'])} / {pct(rf['accuracy'])}", "Our results"],
            ["Ours: Duranay-style baseline", "EfficientNet-B0 features + SVM", "12 classes, same hold-out", f"{pct(ps['accuracy'])}, macro-F1 {ps['macro_f1']:.3f}", "Our results"]]
    A(("table", "Table 6.6: Published results on InfraredSolarModules and results of this project", rows, [3.3, 3.5, 3.5, 3.1, 2.4], (6,), 9))
    bl("Binary triage.", f"Our pretrained Stage 1 reaches {pct(binary_acc(R['pretrained']))} ({pct(binary_acc(R['datagroup']))} from scratch), above the 92.5% of Alves et al. and below the 97.32% of a multi-scale transfer-learning CNN cited by Duranay. It is a single hold-out run, so we claim to be at or slightly above the 92.5% to 93% range and not a new best.")
    bl("12 classes.", f"Our best, {pct(pr['accuracy'])}, is below the published 12-class figures (84.0%, 85% to 88% for ensembles, 93.93%). We did not reach the 93% ceiling or the 95% target stated in the proposal. A Duranay-style pipeline run on our split scored only {pct(ps['accuracy'])}, below our pretrained cascade, which suggests that the gap to 93.93% is largely due to the evaluation protocol rather than to a weaker model; this is a suggestion and not a proof.")
    bl("Why the results are not like for like.", "Duranay uses 10-fold cross-validation, a network trained end to end on the dataset and a supervised feature selection before the cross-validation, so as described, test images can influence these steps. Our test images never do. In our own split about 7% of test images have a near-identical training image (96% with the same label) and about 54% have a close one, so random folds would raise cross-validated scores. A raw-pixel Random Forest scores 61.6% on our split against 84.0% in [4], whose protocol is not stated, so part of the gap remains unexplained.")
    sec("6.7 Discussion and Limitations")
    sub("6.7.1 Interpretation of Findings")
    p("Three results stand out. Thermal-aware augmentation mattered more than any architectural change. Pretraining gave a large and cheap gain even on 24 x 40 pixel thermal images, which refutes the expectation in the "
      "proposal that pretrained backbones give little benefit. And the central hypothesis, that a cascade clearly beats a flat classifier, was only partly supported: the family decision is the weak link, "
      "and with equal backbones the two designs tie.")
    sub("6.7.2 Limitations")
    bl("Single runs.", "Every configuration was run once with seed 42; differences below about 0.5 points are within noise and no confidence intervals were computed. A 10-fold cross-validation of the pretrained cascade was prepared but did not complete, so it is not reported.")
    bl("Near-duplicate images.", "The benchmark has no module or site identifiers, so near-identical crops can fall on both sides of any split.")
    bl("Rare classes.", "Soiling, Hot-Spot and Diode-Multi remain weak, which keeps macro-F1 well below accuracy.")
    bl("Efficiency.", "Latency was not measured reliably, because timings were taken while the GPU was busy, and the pretrained cascade has five times the parameters of the flat model, so the early exit did not give a demonstrated efficiency advantage.")
    bl("Comparison with published work.", "Protocols differ, and some sources could only be checked through abstracts or citing papers.")
    sec("6.8 Chapter Summary")
    p(f"The best cascade reaches {pct(pr['accuracy'])} accuracy with ImageNet pretraining and {pct(dg['accuracy'])} from scratch, with Stage 1 at {pct(binary_acc(R['pretrained']))}. The cascade matches but does not clearly beat a flat classifier, "
      "the family stage is its largest error source, and published 12-class results are higher under protocols that are not comparable with a fixed hold-out.")

    # ------------------------------------------------------------------ CHAPTER 7
    chap(7, "CONCLUSION AND FUTURE SCOPE")
    sec("7.1 Conclusion")
    p("This project implemented Cascaded-SolarNet, a three-stage cascade of five CNNs for 12-class infrared solar module defect classification, with a reproducible split, tests, architecture diagrams and live demonstrations. "
      f"After removing harmful photometric augmentation, enlarging the input and strengthening the front stages, accuracy rose from 61.7% to {pct(dg['accuracy'])} from scratch and to {pct(pr['accuracy'])} with ImageNet pretraining "
      f"(macro-F1 0.37 to {pr['macro_f1']:.2f}). Healthy-versus-defect triage reaches {pct(binary_acc(R['pretrained']))}, comparable to the 92.5% binary figure published for this dataset.")
    p("The claim that a cascade clearly beats a flat classifier was only partly supported. From scratch the cascade is ahead on accuracy and kappa; with the same pretrained backbone it ties a flat CNN while using five times "
      "the parameters. Error propagation is the central trade-off: a wrong decision at Stage 1 or Stage 2 cannot be undone, and the family decision remains the largest error source. Published 12-class results (84% to 93.93%) are "
      "higher than ours, but they use cross-validation and pretrained features, and the 93% ceiling and 95% target of the proposal were not reached under a stricter fixed hold-out. A practical lesson is that thermal "
      "images need thermal-aware augmentation: flips help, while brightness, contrast and noise jitter hurt.")
    sec("7.2 Future Scope")
    bl("Like-for-like evaluation.", "Run several seeds and a leak-free 10-fold cross-validation, and replicate the full NCA pipeline, for a fair comparison with published results.")
    bl("Better family routing.", "Train Stage 2 jointly with the specialists or on the cascade's own routing errors, or use soft hand-offs trained end to end.")
    bl("Rare classes.", "Apply class-specific augmentation or synthetic oversampling for Soiling, Hot-Spot and Diode-Multi, and deduplicate near-identical crops before splitting.")
    bl("Edge deployment.", "Measure real latency, memory and energy on a drone-class device, and test distillation of the pretrained cascade into smaller networks.")
    bl("Additional modalities.", "Investigate RGB or contextual information from neighbouring panels, which the crop-level dataset cannot provide.")
    return B


REFERENCES = [
    "[1] M. Millendorf, E. Obropta, and N. Vadhavkar, \"Infrared Solar Module Dataset for Anomaly Detection,\" ICLR 2020 Workshop on AI for Earth Sciences, 2020.",
    "[2] Z. B. Duranay, \"Fault Detection in Solar Energy Systems: A Deep Learning Approach,\" Electronics, vol. 12, no. 21, p. 4397, 2023, doi: 10.3390/electronics12214397.",
    "[3] R. H. F. Alves, G. A. de Deus Junior, E. G. Marra, and R. P. Lemos, \"Automatic fault classification in photovoltaic modules using Convolutional Neural Networks,\" Renewable Energy, vol. 179, pp. 502-516, 2021.",
    "[4] Corley et al., \"Aerial Infrared Health Monitoring of Solar Photovoltaic Farms at Scale,\" CVPR Workshops (PBVS), 2025, arXiv:2503.02128.",
    "[5] S. P. Pathak and S. A. Patil, \"Enhanced Fault Identification in Solar Panels through Binary Cascaded Convolutional Classifiers with Thermal-Visual Augmentation,\" Int. J. Image, Graphics and Signal Processing, vol. 17, no. 4, 2025.",
    "[6] \"PV-faultNet: Optimized Convolutional Neural Network to Detect Defects in PV Production,\" arXiv:2411.02997, 2024.",
    "[7] \"Defect Detection of Photovoltaic Modules Based on Improved VarifocalNet,\" Scientific Reports, vol. 14, 2024, doi: 10.1038/s41598-024-66234-3.",
    "[8] U. Otamendi, I. Martinez, M. Quartulli, et al., \"Segmentation of Cell-Level Anomalies in Electroluminescence Images of Photovoltaic Modules,\" Solar Energy, vol. 220, pp. 914-926, 2021.",
    "[9] H. Tella, A. Hussein, S. Rehman, et al., \"PV Panel Defect Classification Using Deep Learning Ensemble Methods,\" Case Studies in Thermal Engineering, vol. 66, 2025.",
    "[10] Z. Cai and N. Vasconcelos, \"Cascade R-CNN: Delving into High Quality Object Detection,\" in Proc. CVPR, 2018, pp. 6154-6162.",
    "[11] M. Tan and Q. V. Le, \"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks,\" in Proc. ICML, 2019, arXiv:1905.11946.",
    "[12] K. He, X. Zhang, S. Ren, and J. Sun, \"Deep Residual Learning for Image Recognition,\" in Proc. CVPR, 2016, pp. 770-778.",
    "[13] A. Paszke et al., \"PyTorch: An Imperative Style, High-Performance Deep Learning Library,\" in Advances in Neural Information Processing Systems 32, 2019.",
    "[14] J. Deng et al., \"ImageNet: A Large-Scale Hierarchical Image Database,\" in Proc. CVPR, 2009.",
]
