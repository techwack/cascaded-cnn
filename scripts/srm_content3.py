"""Expanded Chapter 2 (Literature Survey) for the SRM-format report."""
import srm_content as sc


def chapter_2():
    B = []
    A = B.append
    sec = lambda t: A(("sec", t))
    sub = lambda t: A(("sub", t))
    p = lambda t: A(("p", t))
    bl = lambda lead, t: A(("b", lead, t))

    A(("chapter", 2, "LITERATURE SURVEY"))
    p("This chapter reviews deep-learning approaches to PV defect detection and classification, with emphasis on infrared imagery and on the InfraredSolarModules benchmark, and derives the research gaps and "
      "objectives of the project. Ten studies from 2020 to 2025 are discussed in turn, followed by a comparative analysis. The studies were re-checked against their primary sources while preparing the final report; "
      "where a result had been attributed to the wrong paper in the first version of the survey, it has been corrected below.")

    sec("2.1 Overview of PV Defect Detection and Classification Methods")
    sub("2.1.1 Imaging Modalities for PV Inspection")
    p("Three imaging modalities are used for PV inspection. Infrared (IR) thermography is the primary modality for operating plants: in a healthy panel heat is distributed uniformly, whereas defects create localised "
      "temperature anomalies such as hot spots, cold regions or characteristic thermal signatures, and a drone can capture thousands of thermal images per flight. Electroluminescence (EL) imaging reveals cell-level "
      "cracks and inactive regions with very high detail, but requires the panel to be electrically driven and imaged in darkness, so it suits factory and laboratory inspection and not routine drone surveys. "
      "Visible-spectrum (RGB) imaging is cheap and detailed but shows only what is visible on the surface, such as dust, shadows and gross cracks, and says little about electrical faults.")
    sub("2.1.2 The InfraredSolarModules Benchmark")
    p("The InfraredSolarModules dataset, released by Raptor Maps [1], standardises 12 classes from real commercial drone flights: 11 anomaly types and one No-Anomaly class. The images are only 24 x 40 pixels because of "
      "the altitude of capture, so classification depends mainly on recognising temperature-pattern signatures and not on structural detail. The classes fall naturally into electrical faults (Cell, Cell-Multi, Diode, "
      "Diode-Multi, Hot-Spot, Hot-Spot-Multi), environmental conditions (Soiling, Shadowing, Vegetation) and structural conditions (Cracking, Offline-Module), which motivates hierarchical strategies over flat ones.")
    sub("2.1.3 Evolution of the Methods")
    p("Automated analysis of PV imagery has progressed through several generations. Early work extracted handcrafted thermal features and classified them with classical learners such as random forests. The second "
      "generation applied convolutional networks end to end, from small custom CNNs to ImageNet-pretrained VGG, ResNet and EfficientNet backbones, usually with transfer learning because labelled PV datasets are small. "
      "A third thread moved to object detection and segmentation (Faster R-CNN, Cascade R-CNN, VarifocalNet, YOLO) to localise defects within a module or panel image. The most recent work explores cascades, ensembles, "
      "feature-selection pipelines and multi-sensor fusion, usually at the cost of extra sensors or computation.")

    sec("2.2 Deep Learning Approaches for PV Defect Analysis")
    sub("2.2.1 Infrared Solar Module Dataset (Millendorf et al., 2020)")
    p("Millendorf, Obropta and Vadhavkar of Raptor Maps introduced the benchmark at the ICLR 2020 Workshop on AI for Earth Sciences [1]. The paper motivates automation with industry figures: aerial inspections "
      "find anomalies in a few percent of modules, and the dataset was released to let researchers quickly train models and tune hyper-parameters. It describes the 12 classes and the data format and lists "
      "benchmarking of classification results as a goal, but it reports no model accuracies. The widely quoted baselines on this dataset (Random Forest 84%, custom CNN 92.5%, EfficientNet with an SVM 93%) were "
      "published in later papers, which are discussed in Section 2.2.11.")
    sub("2.2.2 Binary Cascaded CNN with Thermal-Visual Augmentation (Pathak and Patil, 2025)")
    p("Pathak and Patil published a cascaded CNN in the International Journal of Image, Graphics and Signal Processing (vol. 17, no. 4, 2025) that fuses thermal infrared images with co-registered visible-spectrum "
      "images [5]. Their Binary Cascaded CNN (B-CNN) has two stages: the first classifies a module as healthy or faulty and the second identifies the specific fault. The thermal-visual augmentation greatly enriches "
      "the feature space and compensates for the low resolution of IR-only inputs, reaching 99.8% precision and 98.5% accuracy, the highest results in this survey. The critical limitation is the requirement for paired "
      "dual-sensor data, which most UAVs, carrying only a thermal camera, cannot provide, and the fusion adds hardware cost, synchronisation effort and computation. This is the closest relative of our cascade, and it shows "
      "that a binary-first cascade is effective; our design keeps the cascade idea but uses a single sensor.")
    sub("2.2.3 PV-faultNet: Lightweight Optimised CNN (2024)")
    p("PV-faultNet investigates the design space of CNN architectures for efficient PV defect classification [6]. The authors benchmark architectures from compact custom CNNs to larger Siamese networks and "
      "multi-spectral models, and find that a carefully designed compact CNN reaches 93% accuracy at 8 ms per image, competitive with much larger models. The key finding is that larger backbones give negligible gains "
      "because of the low information content of small IR images: the bottleneck is the input resolution, not the model capacity. The work stresses efficient architecture design for real-time edge inference, but "
      "it does not address the accuracy ceiling or class imbalance and remains at about 93% single-stage accuracy. It supports our choice of small per-stage networks.")
    sub("2.2.4 Defect Detection Based on Improved VarifocalNet (2024)")
    p("This study, published in Scientific Reports (vol. 14, 2024), extends the VarifocalNet detector to localise defects in drone-captured IR images and compares it with Cascade R-CNN and YOLOv7 on a custom dataset "
      "of annotated images [7]. VarifocalNet achieves the highest confidence for coarse defect detection, and Cascade R-CNN performs well on common categories, but both struggle with fine crack localisation and "
      "produce fragmented or offset bounding boxes for thin linear defects. Detection-based approaches thus add a failure mode beyond classification errors: a defect may be correctly classified but poorly "
      "localised, which defeats the inspection purpose. This motivates whole-image classification for the low-resolution IR domain.")
    sub("2.2.5 Segmentation of Cell-Level Anomalies in EL Images (Otamendi et al., 2021)")
    p("Otamendi, Martinez and Quartulli published a pipeline in Solar Energy (vol. 220, pp. 914-926, 2021) for cell-level anomaly detection in electroluminescence images [8]. Faster R-CNN proposes regions and an "
      "EfficientNet-based classifier identifies the anomaly type inside each region, reaching high precision on their EL dataset. EL imaging, however, needs electrically active panels imaged in darkness and is much "
      "less practical than IR thermography for routine drone inspection, and the detection-then-classification pipeline adds seconds of latency per image, which is incompatible with real-time use. The work "
      "nevertheless shows that cell-level granularity is feasible.")
    sub("2.2.6 Field-Adapted ResNet-50 with Transfer Learning")
    p("This study addresses the gap between laboratory PV defect datasets and field conditions. A ResNet-50 pretrained on ImageNet is fine-tuned on field-collected images with extensive augmentation (8x, including "
      "rotations, flips, brightness perturbation and noise injection), reaching 98.59% accuracy on the field data. The augmentation is critical because field IR images vary in thermal contrast, panel orientation, "
      "atmospheric conditions and camera angle. Limitations are the high resource demand of ResNet-50, which makes edge deployment hard, and that class imbalance and fine-grained subtypes are not addressed. "
      "Notably, its brightness and noise augmentation is the kind that our own ablation found harmful on the InfraredSolarModules benchmark, which suggests that the right augmentation depends on how the "
      "dataset was captured and normalised.")
    sub("2.2.7 Deep Learning Ensembles for PV Defect Classification (Tella et al., 2025)")
    p("Tella, Hussein and Rehman published a multi-model benchmark in Case Studies in Thermal Engineering (vol. 66, 2025) [9]. It evaluates eight architectures, including VGG-16, ResNet-50, DenseNet, MobileNet, "
      "EfficientNet and Vision Transformer variants, on the ELPV benchmark. The key finding is a systematic gap between binary classification (about 73% on average) and multi-class fault classification (about 68% "
      "on average) that no single model closes, and the gap reflects the structural difficulty of multi-class decision boundaries between look-alike faults at low resolution and not model capacity. An ensemble "
      "gives only marginal gains. This is the main published evidence for the weakness of the flat single-stage paradigm that motivates our decomposition.")
    sub("2.2.8 CNN-Based Classification on RGB Images")
    p("This work applies VGG and ResNet variants to RGB images of solar panels with a four-class taxonomy (normal, cracked, dusty, shadowed). Accuracy is high, about 96% to 98%, because there are few classes and "
      "the images are rich. The coarse formulation, however, hides the real operational challenge: distinguishing electrical fault subtypes such as hot spots, diode failures and bypass faults requires thermal imaging "
      "and finer granularity. The results cannot be compared with IR studies, and the study is useful mainly as a baseline for RGB-based inspection.")
    sub("2.2.9 CNN with Bacterial Foraging Optimisation")
    p("This study combines five pretrained CNNs (VGG-16, AlexNet, SqueezeNet, GoogleNet and ResNet) with Bacterial Foraging Optimisation (BFO), a bio-inspired metaheuristic, to select the most discriminative "
      "subset of features from the concatenated feature vectors. VGG-16 with BFO reaches the best accuracy of 98.75% on their data. The price is a large overhead: the feature selection adds computation in training "
      "and inference, and all five backbones must run together, which is impractical for real-time drone hardware. It shows that feature selection over multiple backbones can raise accuracy at an unacceptable latency cost.")
    sub("2.2.10 Cascade R-CNN for Defect Detection")
    p("Cascade R-CNN, proposed by Cai and Vasconcelos [10], is a multi-stage detector in which cascaded heads refine region proposals at progressively higher intersection-over-union thresholds. Applied to PV defects "
      "it improves localisation for defects of variable aspect ratio, but it shares the crack-fragmentation problem of the other detectors on thin linear defects. Conceptually, cascade detection operates on bounding-box "
      "proposals while a cascaded classifier operates on the class decision space. The work motivates the use of progressive, refined decisions, but argues for applying them at the classification level in the "
      "low-resolution IR setting.")
    sub("2.2.11 Later Work on the InfraredSolarModules Benchmark")
    p("Several later papers report results on the benchmark, and they must be read with their protocols in mind. Alves et al. trained a custom CNN with data augmentation and report 92.5% test accuracy with cross-validation "
      "for the detection of anomalies (a binary task) and 78.85% for eight selected classes [3]. Duranay extracts features with an ImageNet-pretrained EfficientNet-B0, selects 17,000 of them with Neighbourhood Component "
      "Analysis and classifies them with an SVM, reporting 93.93% accuracy and 89.82% F1 on all 12 classes with 10-fold cross-validation [2]. Corley et al. report a random forest baseline of 84.0% and cite further "
      "results (a cascading ensemble with a voting CNN at 88% [15] and a NASNet-LSTM model at 85% [16]), and they argue that the dataset may be too simple for deployment at scale because the crop level lacks the context "
      "of neighbouring panels [4]. These are the numbers against which Chapter 6 places the results of this project.")

    sec("2.3 Comparative Analysis")
    p("Table 2.1 summarises the studies. Table 2.2 groups the approaches by strategy and Table 2.3 collects the results reported on InfraredSolarModules.")
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
    A(("table", "Table 2.2: Comparison of approach families",
       [["Approach family", "Examples", "Strengths", "Weaknesses"],
        ["Flat single-stage CNN", "Custom CNNs, PV-faultNet, Alves et al.", "Simple, fast, small", "Plateau near 93%; confuses look-alike faults; biased by imbalance"],
        ["Pretrained backbone plus classifier", "Duranay, ResNet-50, CNN + BFO", "High accuracy from transfer learning", "Heavy or multi-backbone; protocols often cross-validated"],
        ["Detection and segmentation", "VarifocalNet, Cascade R-CNN, Otamendi et al.", "Locate the defect", "Fragment thin defects; slow; need box labels"],
        ["Multi-sensor cascade", "Pathak and Patil", "Highest accuracy", "Needs thermal and visual sensors and fusion"],
        ["Ensembles", "Tella et al., voting CNNs", "Marginal accuracy gains", "Cost grows with every model"],
        ["Single-sensor classification cascade (this project)", "Cascaded-SolarNet", "One IR camera, small networks, early exit, error attribution", "Errors propagate between stages"]],
       [3.6, 3.8, 3.6, 4.8], (), 9))
    A(("table", "Table 2.3: Results reported on InfraredSolarModules",
       [["Source", "Method", "Task", "Reported accuracy"],
        ["Duranay 2023 [2]", "EfficientNet-B0 features, NCA, SVM", "12 classes, 10-fold CV", "93.93%"],
        ["Alves et al. 2021 [3]", "Custom CNN with augmentation", "Binary detection, cross-validation", "92.5%"],
        ["Alves et al. 2021 [3]", "Custom CNN with augmentation", "8 selected classes, cross-validation", "78.85%"],
        ["Barraz et al. 2025, as cited in [4]", "Cascading ensemble, voting CNN", "Not stated", "88%"],
        ["Gopalakrishnan et al. 2024, as cited in [4]", "NASNet + LSTM", "Not stated", "85%"],
        ["Corley et al. 2025 [4]", "Random Forest", "Anomaly classification", "84.0%"]],
       [4.6, 4.6, 4.0, 2.6], (), 9.5))
    sub("2.3.1 Accuracy Versus Available Information")
    p("The highest accuracies in the survey (98% to 99%) come from methods that use more information than a single IR crop gives: a second sensor, a different modality with higher detail, field-specific training data or "
      "an ensemble of backbones. Methods restricted to single IR crops cluster around 93%. The comparison shows that accuracy is bought with sensors and computation, and that the single-sensor setting is a harder and "
      "more practical benchmark than many headline figures suggest.")
    sub("2.3.2 Binary Versus Multi-Class Classification")
    p("Binary healthy-versus-faulty detection reaches 92.5% to 97% in several studies, while multi-class results are lower: about 68% across eight architectures in the ELPV benchmark [9] and 78.85% for eight "
      "classes in [3]. This supports decomposing the task so that the easy binary decision is made first and by a dedicated network, which is the design of Cascaded-SolarNet.")
    sub("2.3.3 Evaluation Protocols")
    p("Most reported results use k-fold cross-validation, some preprocessing or feature selection before the split is applied, and several datasets or class definitions. Results from different papers are therefore "
      "not directly comparable. In this project all results use one fixed held-out split that is never touched during training or selection, and Chapter 6 discusses what this means for the comparison.")

    sec("2.4 Limitations of Existing Methods")
    p("Four limitations recur across the reviewed work. First, the highest accuracies depend on information that a drone does not have (paired visible images, powered-panel electroluminescence) or on heavy computation "
      "(ensembles, large backbones, feature selection over several networks). Second, detection pipelines add failure modes on thin defects and need bounding-box labels that the crop-level benchmark lacks. Third, results "
      "are reported under different protocols, mostly cross-validation, which makes them hard to compare, and near-duplicate crops in the dataset can inflate cross-validated scores. Fourth, almost no study reports "
      "where a hierarchical model loses accuracy, so the benefit of cascading remains unmeasured.")
    sec("2.5 Research Gaps and Contributions")
    p("The survey identifies six research gaps. Table 2.4 maps each gap to the contribution of this project.")
    A(("table", "Table 2.4: Research gaps and the contributions that address them",
       [["Gap", "Research gap", "Contribution"],
        ["G1", "Dual-sensor approaches are impractical", "C4: single-modality design using only single-camera IR images"],
        ["G2", "Scaling the model gives negligible gains", "C1: five small per-stage networks instead of one large model"],
        ["G3", "Bounding boxes fragment thin defects", "C5: whole-image classification with no localisation step"],
        ["G4", "Naive cascades add latency", "C2: early-exit binary front end for healthy panels"],
        ["G5", "Accuracy drops as classes grow", "C1: hierarchical decomposition into 2-, 3- and 2-to-6-way decisions"],
        ["G6", "Similar defects are confused", "C3: family grouping before fine classification, semantic and data-driven"]],
       [1.4, 6.4, 8.0], (), 10))
    sec("2.6 Research Objectives")
    bl("O1.", "To design a three-stage cascade of small CNNs for 12-class IR solar module classification from a single sensor.")
    bl("O2.", "To implement the system reproducibly in PyTorch with a fixed stratified split, tests and published code.")
    bl("O3.", "To measure, with controlled ablations, which design choices (augmentation, input size, capacity, grouping, pretraining) matter.")
    bl("O4.", "To attribute the cascade's errors to the stage that caused them and to compare it with a flat classifier and with published results.")
    sec("2.7 Plan of Action")
    p("The work followed six steps: (1) literature survey and gap analysis; (2) dataset preparation with a stratified 70/15/15 split; (3) implementation of the five-network cascade and a flat baseline; "
      "(4) controlled ablations of augmentation, input size, network capacity, family grouping and inference variants; (5) fine-tuning of an ImageNet-pretrained backbone at every stage; and "
      "(6) evaluation, error attribution, comparison with published work, architecture diagrams, demonstrations and this report.")
    return B
