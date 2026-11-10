"""Expanded Chapters 5 and 6 plus appendices for the SRM-format report (replaces the compact versions in srm_content.py)."""
import json
import re
from pathlib import Path

import srm_content as sc

ROOT, FIG, R, pct, binary_acc, CLASSES = sc.ROOT, sc.FIG, sc.R, sc.pct, sc.binary_acc, sc.CLASSES
pr, prf, dg, i3, imp, fl, rf, ps = sc.pr, sc.prf, sc.dg, sc.i3, sc.imp, sc.fl, sc.rf, sc.ps
STATS = json.load(open(ROOT / "results" / "model_stats.json"))
VAL = json.load(open(ROOT / "results" / "node_val_f1.json"))
OVERLAP = json.load(open(ROOT / "results" / "data_overlap_check.json"))


def src(path, start, end=None):
    """Lines of a source file from the line matching `start` up to (not including) the line matching `end`."""
    lines = (ROOT / path).read_text(encoding="utf-8").replace("\r\n", "\n").split("\n")
    i = next(k for k, l in enumerate(lines) if re.search(start, l))
    j = next((k for k in range(i + 1, len(lines)) if end and re.search(end, lines[k])), len(lines))
    return "\n".join(lines[i:j]).rstrip()


def layer_rows(cfg, node):
    rows = [["Layer", "Output shape", "Parameters"], ["Input", "1 x 80 x 48", "0"]]
    for name, shape, p in STATS[cfg][node]["layers"]:
        rows.append([name, " x ".join(str(s) for s in shape), f"{p:,}"])
    rows.append(["Total", "", f"{STATS[cfg][node]['params']:,}"])
    return rows


def top_confusions(res, k=6):
    cm = res["confusion"]
    out = []
    for i in range(12):
        for j in range(12):
            if i != j and cm[i][j] > 0:
                out.append((cm[i][j], CLASSES[i], CLASSES[j], sum(cm[i])))
    return sorted(out, reverse=True)[:k]


def chapters_5_6(charts, perclass_png, demo_png, nndl=False):
    B = []
    A = B.append
    sec = lambda t: A(("sec", t))
    sub = lambda t: A(("sub", t))
    p = lambda t: A(("p", t))
    eq = lambda t: A(("eq", t))
    bl = lambda lead, t: A(("b", lead, t))
    s_all = {k: STATS["improved3"][k]["params"] for k in STATS["improved3"]}
    p_all = {k: STATS["pretrained"][k]["params"] for k in STATS["pretrained"]}

    # =================================================================== CHAPTER 5
    A(("chapter", 5, "ARCHITECTURE AND IMPLEMENTATION"))
    sec("5.1 Overview of the Implemented System")
    p("Figure 1 shows the implemented cascade. The system is implemented in Python 3.12 with PyTorch 2.11 and torchvision, scikit-learn for the metrics and Matplotlib for plots. Training and evaluation ran on an "
      "NVIDIA GeForce RTX 5060 Laptop GPU with 8 GB of memory. The code is organised as a Python package (solarnet) with separate modules for the class hierarchy (taxonomy), data handling (data), the models, shared "
      "helpers (common), training (train) and evaluation (evaluate), plus configuration files, scripts for the baselines, cross-validation, figure and report generation, and a test suite. Appendix D lists the repository layout.")
    A(("fig", FIG / "overall_cascade.png", 12.0, "Fig. 1. The implemented three-stage Cascaded-SolarNet architecture."))
    p("An image is processed as follows. It is normalised and passed to CNN-1. If CNN-1 predicts No-Anomaly the pipeline stops. Otherwise CNN-2 predicts a family and the image is passed to the specialist of that family, "
      "whose prediction is converted to a global class index by adding the family's class offset. Every network is trained independently on ground-truth routing, and the cascade is assembled only at evaluation time, so "
      "the error of a stage is measured exactly as it would occur in deployment.")
    sec("5.2 Data Pipeline")
    sub("5.2.1 Dataset and Class Distribution")
    p("The InfraredSolarModules dataset has 20,000 grayscale images of 24 pixels wide and 40 pixels tall. Figure 2 shows the class distribution on a logarithmic axis: half of the images are healthy, and the four rarest "
      "classes (Diode-Multi, Soiling, Hot-Spot-Multi and Hot-Spot) have between 175 and 249 images each. Figure 3 shows one test image per class with a thermal colour map. Even at this size the healthy panel is "
      "nearly uniform, while faults show localised bright regions whose position and extent differ by class.")
    A(("fig", FIG / "class_distribution.png", 14.5, "Fig. 2. Class distribution of the InfraredSolarModules dataset (logarithmic axis)."))
    A(("fig", FIG / "sample_images.png", 15.0, "Fig. 3. One test image per class, shown with the inferno colour map."))
    sub("5.2.2 Split and Normalisation")
    p("The images are split once into 70% training (14,000), 15% validation (3,000) and 15% test (3,000) images, stratified by class with a fixed random seed, so that every model sees exactly the same split "
      "(Table 3.1). Pixel values are scaled to [0, 1] and normalised with statistics computed on the training split only, which prevents information from the validation and test images from entering the pipeline:")
    eq("x' = (x - mu_train) / sigma_train,   mu_train = 0.618,  sigma_train = 0.156")
    sub("5.2.3 Augmentation")
    p("During training only random horizontal and vertical flips (probability 0.5 each) are applied. A flip does not change the class of a module, because the faults are defined by which cells or diodes are hot and not by "
      "orientation. Brightness, contrast and noise jitter were implemented and tested but removed, because in thermal images brightness is temperature, and perturbing it lowered accuracy by about ten points (Section 6.3.1). "
      "Quarter-turn rotations are not used, because they would change the 40 x 24 shape and the convolutional pipeline assumes a fixed size.")
    sub("5.2.4 Handling Class Imbalance")
    p("Class imbalance is handled at three levels: structurally by the early exit of Stage 1, in the loss by class weights, and in the evaluation by macro-averaged metrics. For a network with K classes trained on N samples, "
      "class r with n_r samples receives the weight")
    eq("w_r = N / (K x n_r)")
    p("so that every class contributes equally to the loss in expectation. For example, in the flat 12-class setting the healthy class receives a weight of 0.17 and Diode-Multi a weight of 9.5.")
    sec("5.3 Building Blocks and Their Mathematics")
    p("All networks are assembled from a small set of layers. This section states the operation of each layer, so that the architecture tables and parameter counts in the following sections can be checked by hand.")
    sub("5.3.1 Convolution")
    p("A convolutional layer with C_in input channels, C_out output channels and a k x k kernel computes, for every output channel c and position (i, j),")
    eq("y[c, i, j] = b[c] + sum over (m, u, v) of  W[c, m, u, v] * x[m, i + u, j + v]")
    p("with padding that preserves the spatial size (k = 3, padding 1). The layer has k^2 x C_in x C_out weights and C_out biases (the bias is omitted when a BatchNorm layer follows).")
    sub("5.3.2 Batch Normalisation")
    p("BatchNorm normalises each channel over the mini-batch and then rescales it with two learned parameters per channel:")
    eq("x_hat = (x - mean_B) / sqrt(var_B + eps),    y = gamma * x_hat + beta")
    p("It stabilises and speeds up training and reduces sensitivity to weight initialisation, which matters for small networks trained on imbalanced data. At test time running averages of the mean and variance are used.")
    sub("5.3.3 ReLU and Max Pooling")
    p("The rectified linear unit f(x) = max(0, x) introduces the non-linearity that lets the network learn complex decision boundaries. Max pooling with a 2 x 2 window and stride 2 keeps the maximum of each window, "
      "halving height and width, reducing computation and giving tolerance to small shifts in where the heat pattern sits in the image.")
    sub("5.3.4 Global Average Pooling and Classification Head")
    p("After the last block each of the C feature maps of size H x W is summarised by its mean, z_c = (1 / HW) sum of x_c(i, j). Dropout with probability p randomly zeroes activations during training as a regulariser, and a fully "
      "connected layer maps the C values to K class scores s_k. The class probabilities are given by the softmax function:")
    eq("p_k = exp(s_k) / sum over j of exp(s_j)")
    sub("5.3.5 Loss Function")
    p("Each network minimises the class-weighted cross-entropy over a mini-batch of B samples with labels y_i:")
    eq("L = - (1 / B) * sum over i of  w[y_i] * log p[i, y_i]")
    sub("5.3.6 Counting Parameters")
    p("A block of two convolutions followed by BatchNorm has k^2 x C_in x C_mid + k^2 x C_mid x C_out weights and 2 x (C_mid + C_out) BatchNorm parameters. The tables in the following sections were generated by running the "
      "implemented models, and the totals match this formula.")
    sec("5.4 Stage 1: CNN-1, Binary Triage")
    p("CNN-1 answers whether a module is healthy or defective. In the from-scratch variant it has three blocks of two 3 x 3 convolutions, each followed by BatchNorm and ReLU, then 2 x 2 max pooling, with 64, 128 and 128 "
      f"channels, followed by global average pooling, dropout (0.2) and a fully connected layer with two outputs, {s_all['triage']:,} parameters in total. The 80 x 48 upsampled input is reduced to 64 x 40 x 24, 128 x 20 x 12 and "
      "128 x 10 x 6 feature maps. A prediction of No-Anomaly ends the pipeline (the early exit); a defect continues to Stage 2. Table 5.1 lists every layer.")
    A(("fig", FIG / "stage1_cnn1.png", 15.5, "Fig. 4. Stage 1: CNN-1, binary triage (layers and feature-map sizes)."))
    A(("table", "Table 5.1: Layer-by-layer structure of CNN-1 (from-scratch variant)", layer_rows("improved3", "triage"), [4.0, 4.2, 3.0], (len(layer_rows("improved3", "triage")) - 1,), 10))
    sec("5.5 Stage 2: CNN-2, Fault-Family Classifier")
    p("CNN-2 has the same three-block design with 128, 256 and 128 channels, dropout 0.4 and three outputs, "
      f"{s_all['family']:,} parameters in total. It is trained only on defective images labelled by family. Its highest-scoring output selects which specialist runs next. Faults of one family should look alike, "
      "so the decision should be easier than choosing among eleven faults directly. Section 6.4 shows how well this holds.")
    A(("fig", FIG / "stage2_cnn2.png", 15.5, "Fig. 5. Stage 2: CNN-2, fault-family classifier."))
    A(("table", "Table 5.2: Layer-by-layer structure of CNN-2 (from-scratch variant)", layer_rows("improved3", "family"), [4.0, 4.2, 3.0], (len(layer_rows("improved3", "family")) - 1,), 10))
    sec("5.6 Stage 3: Fine-Grained Specialists")
    p("Three specialists share the 128-256-128 design and differ only in their output layer: CNN-3a (Electrical, six classes), CNN-3b (Environmental, three classes) and CNN-3c (Structural, two classes). "
      "Each is trained only on the defective images of its own family, so it can spend all its capacity on the distinctions inside that family. The final label is the family's class offset plus the specialist's prediction. "
      "Table 5.3 gives the parameter counts of all five networks in both variants.")
    A(("fig", FIG / "stage3_specialists.png", 15.5, "Fig. 6. Stage 3: the three specialists and their classes."))
    A(("table", "Table 5.3: Parameter counts of the five stage networks",
       [["Network", "Output classes", "From scratch", "Pretrained EfficientNet-B0"],
        ["CNN-1 (triage)", 2, f"{s_all['triage']:,}", f"{p_all['triage']:,}"], ["CNN-2 (family)", 3, f"{s_all['family']:,}", f"{p_all['family']:,}"],
        ["CNN-3a (Electrical)", 6, f"{s_all['electrical']:,}", f"{p_all['electrical']:,}"], ["CNN-3b (Environmental)", 3, f"{s_all['environmental']:,}", f"{p_all['environmental']:,}"],
        ["CNN-3c (Structural)", 2, f"{s_all['structural']:,}", f"{p_all['structural']:,}"],
        ["Cascade total", "", f"{sum(s_all.values()):,}", f"{sum(p_all.values()):,}"]],
       [4.6, 3.0, 3.4, 4.6], (6,), 10))
    A(("table", "Table 5.4: Architecture summary of the five networks (from-scratch variant)",
       [["Network", "Stage", "Input", "Layers", "Output"],
        ["CNN-1", "1: triage", "IR image, 80 x 48", "3 blocks 2 x (Conv3x3, BN, ReLU) + MaxPool, 64-128-128; GAP, Dropout 0.2, FC", "2 classes"],
        ["CNN-2", "2: family", "Defective image", "Same, 128-256-128; Dropout 0.4, FC", "3 families"],
        ["CNN-3a", "3: Electrical", "Routed image", "Same as CNN-2, FC with 6 outputs", "6 classes"],
        ["CNN-3b", "3: Environmental", "Routed image", "Same as CNN-2, FC with 3 outputs", "3 classes"],
        ["CNN-3c", "3: Structural", "Routed image", "Same as CNN-2, FC with 2 outputs", "2 classes"]],
       [1.8, 2.4, 2.8, 6.6, 2.2], (), 9.5))
    sec("5.7 Pretrained Variant")
    p("In the pretrained variant each of the five networks is an ImageNet-pretrained EfficientNet-B0 [11] whose classifier is replaced by dropout and a fully connected layer with the stage's number of outputs. "
      "EfficientNet-B0 is built from mobile inverted bottleneck (MBConv) blocks: each block expands the channels with a 1 x 1 convolution, applies a depthwise convolution, re-weights channels with a squeeze-and-excitation "
      "module and projects back with another 1 x 1 convolution, adding a residual connection when the shape allows. The family of networks was designed by compound scaling of depth, width and resolution, which gives "
      "B0 about 4 million parameters in the form used here.")
    p("The grayscale image is resized to 160 x 96 with bilinear interpolation and repeated to three channels inside the model, so the rest of the pipeline is unchanged and the pretrained first layer can be used as is. "
      f"The five networks hold {sum(p_all.values()) / 1e6:.1f} million parameters in total, against {sum(s_all.values()) / 1e6:.1f} million for the from-scratch cascade. All weights are fine-tuned, with a learning rate of 3e-4 "
      "and mixed-precision arithmetic.")
    sec("5.8 Training Procedure")
    p("Each network is trained independently on the data matching its task (ground-truth routing). Table 5.5 shows the training and validation sizes of the five tasks. Training uses the Adam optimiser with weight decay 1e-4, "
      "batch size 64 and the class-weighted loss of Section 5.3.5. The learning rate is 5e-4 for the from-scratch networks and 3e-4 for the pretrained ones. The checkpoint with the best validation macro-F1 is kept, the learning "
      "rate is halved when that metric stalls, and training stops after 25 epochs (6 for the pretrained networks) without improvement.")
    A(("table", "Table 5.5: Training data of each stage network",
       [["Network", "Task", "Training images", "Validation images", "Classes"],
        ["CNN-1", "Healthy vs defect", "14,000", "3,000", 2], ["CNN-2", "Family of a defect", "7,000", "1,500", 3], ["CNN-3a", "Electrical subtype", "3,734", "801", 6],
        ["CNN-3b", "Environmental subtype", "2,029", "434", 3], ["CNN-3c", "Structural subtype", "1,237", "265", 2]], [2.4, 4.6, 3.2, 3.4, 2.0], (), 10))
    A(("code", "Algorithm 5.1: Training of one stage network", "\n".join([
        "input : training set D_train, validation set D_val, network f, class weights w",
        "output: best checkpoint f*",
        "best <- -1;  wait <- 0",
        "for epoch = 1 .. E_max:",
        "    for each mini-batch (x, y) of D_train:",
        "        x <- normalise(random_flip(x))",
        "        loss <- weighted_cross_entropy(f(x), y, w)",
        "        update the parameters of f with Adam on loss",
        "    F1_val <- macro_F1(f on D_val)",
        "    if F1_val > best:  best <- F1_val;  save checkpoint f*;  wait <- 0",
        "    else:              wait <- wait + 1;  if wait >= patience: stop",
        "return f*"])))
    sec("5.9 Cascade Inference and Cost Analysis")
    A(("code", "Algorithm 5.2: Inference of Cascaded-SolarNet on one image", "\n".join([
        "input : image x",
        "output: class label y",
        "x <- normalise(x)",
        "if CNN-1(x) = No-Anomaly:  return No-Anomaly          # early exit",
        "family <- argmax CNN-2(x)",
        "k      <- argmax CNN-3[family](x)",
        "return class_offset[family] + k"])))
    p(f"Let c1, c2 and c3 be the costs of the three kinds of network. A healthy image costs c1 and a defective one c1 + c2 + c3, so the expected cost per image is c1 + P(defect)(c2 + c3), where P(defect) is the "
      f"fraction of images that pass Stage 1. On the test split {pct(pr['stages']['early_exit_rate'])} of the images leave at Stage 1. With similar-sized networks in all stages the cascade therefore costs about "
      "1.5 network evaluations per image on average, instead of one for a flat model, and it only saves time when the first stage is much cheaper than the others. In the from-scratch variant CNN-1 has about 40% of the "
      "parameters of the others, but in the pretrained variant all five networks are the same size, which is why the early exit gave no efficiency advantage in our measurements.")
    sec("5.10 Evaluation Tools and Demonstrations")
    p("The evaluation script runs the whole cascade end to end on the test split and reports accuracy, macro-F1, Cohen's kappa, per-class scores, the confusion matrix and the error attribution per stage "
      "(missed defects and false alarms at Stage 1, wrong family at Stage 2, wrong subtype at Stage 3). Further scripts reproduce the baselines, run 10-fold cross-validation, search for the data-driven family "
      "grouping and generate the figures and reports. Two demonstrations show the cascade at work: a command-line tool that draws each stage's confidence bars for chosen test images (Figure 7), and an "
      "interactive browser page in which a class can be selected and the three stages reveal their decisions one after another. Appendix C explains how to run them.")
    A(("fig", demo_png, 15.5, "Fig. 7. Output of the live demonstration: each stage's decision and confidence for Hot-Spot test images."))
    sec("5.11 Testing and Reproducibility")
    p("A suite of 15 automated tests runs on synthetic data and needs no dataset download. It verifies the class hierarchy, the data split, the augmentation, the shapes of all networks and the routing of the cascade. "
      "Table 5.6 lists the tests. Reproducibility is supported by a fixed seed, a saved split file, configuration files for every experiment and result files for every reported number.")
    A(("table", "Table 5.6: Automated tests",
       [["Test", "What it verifies"],
        ["Twelve classes, healthy first", "The taxonomy has 12 classes and the healthy class has global index 0"],
        ["Node labels cover every class once", "Each class belongs to exactly one family and receives a consistent local label"],
        ["Global index matches family offsets", "Family offset plus local index reproduces the global class index"],
        ["Duplicate class rejected", "A taxonomy listing a class twice raises an error"],
        ["Class weights formula", "Weights equal N / (K x n) and sum correctly"],
        ["Stratified split", "Train, validation and test ids are disjoint, complete and reproducible for a fixed seed"],
        ["Node output shape (5 nodes)", "Each stage network outputs one score per class for a 40 x 24 input"],
        ["Augmentation shape and range", "Flips and jitter keep the tensor shape and values within [0, 1]"],
        ["Cascade routing and early exit", "Healthy images exit at Stage 1 and defects reach the correct specialist and offset"],
        ["Pretrained stage output shape", "The EfficientNet-B0 stage outputs the right number of classes"]],
       [5.6, 10.2], (), 10))
    if nndl:
        import srm_content_nndl as sn
        B.extend(sn.tech_sections())
        sec("5.17 Chapter Summary")
    else:
        sec("5.12 Chapter Summary")
    p("The system consists of five small stage networks in two variants, a leak-free data pipeline, a class-weighted training procedure, an evaluation module that attributes every error to a stage and an automated test "
      "suite. The next chapter evaluates it.")

    # =================================================================== CHAPTER 6
    A(("chapter", 6, "RESULTS AND ANALYSIS"))
    sec("6.1 Experimental Setup")
    sub("6.1.1 Dataset and Protocol")
    p("All numbers are measured on the 3,000-image held-out test split of Table 3.1 and are single runs per configuration (seed 42). The test images are not used for training, early stopping, normalisation or "
      "model selection. Choices such as the learning rate were made on the validation split.")
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
    p("With TP, FP and FN the true positives, false positives and false negatives of a class, precision is TP / (TP + FP), recall is TP / (TP + FN) and the F1 score is their harmonic mean:")
    eq("F1 = 2 x precision x recall / (precision + recall),     macro-F1 = (1 / K) x sum over classes of F1_r")
    p("Macro-F1 weights every class equally and therefore exposes poor performance on rare faults, which accuracy hides under class imbalance. Cohen's kappa measures agreement beyond chance:")
    eq("kappa = (p_o - p_e) / (1 - p_e)")
    p("where p_o is the observed accuracy and p_e the accuracy expected by chance from the class frequencies. The Stage 1 binary accuracy measures the healthy-versus-defect decision, and per-stage error counts "
      "attribute each mistake to the first stage that went wrong.")
    sub("6.1.4 Configurations")
    A(("table", "Table 6.2: Definition of the evaluated configurations",
       [["ID", "Variant", "Modification"],
        ["C1", "Default", "Flips plus brightness (20%), contrast (15%) and noise jitter; 40 x 24 input; 60 epochs"],
        ["C2", "No photometric jitter", "C1 with flips only; longer schedule (100 epochs)"], ["C3", "Improved", "C2 with 80 x 48 input and two convolutions per block; 150 epochs"],
        ["C4", "Improved2", "C3 with balanced sampling and label smoothing instead of loss weights"], ["C5", "Improved3", "C3 with larger Stage 1 and Stage 2 networks"],
        ["C6", "Data-driven families", "C5 with the data-driven family grouping of Section 4.3"], ["C7", "Pretrained", "EfficientNet-B0 (ImageNet) fine-tuned at every stage"]],
       [1.2, 4.2, 10.4], (), 10))
    sec("6.2 Overall Performance")
    sub("6.2.1 Test Results")
    p(f"The best result is the pretrained cascade (C7) with {pct(pr['accuracy'])} accuracy, macro-F1 {pr['macro_f1']:.3f}, Cohen's kappa {pr['kappa']:.3f} and {pct(binary_acc(R['pretrained']))} on the Stage 1 healthy-versus-defect "
      f"decision. The best from-scratch cascade reaches {pct(dg['accuracy'])} accuracy. Table 6.3 lists all configurations and baselines, and Figure 8 plots them.")
    rows, best = sc._t6_rows()
    A(("table", "Table 6.3: Test results by configuration", rows, [5.6, 2.2, 2.2, 2.2, 1.8, 2.4], (best,), 9.5))
    A(("fig", charts["prog"], 14.5, "Fig. 8. Cascade test accuracy and macro-F1 by configuration (dashed line: from-scratch flat CNN)."))
    sub("6.2.2 Per-Network Validation Results and Training Behaviour")
    p("Figure 9 compares the best validation macro-F1 of each of the five networks in the two variants, and Figure 10 shows the training curves. Pretraining improved every network, most clearly Stage 2 "
      f"(validation macro-F1 {VAL['From scratch (improved3)|family']:.3f} to {VAL['Pretrained EfficientNet-B0|family']:.3f}) and the Electrical specialist ({VAL['From scratch (improved3)|electrical']:.3f} to {VAL['Pretrained EfficientNet-B0|electrical']:.3f}). "
      "The pretrained networks reach their best validation score within 10 to 20 epochs, while the from-scratch networks need 50 to 140 epochs, and the validation curves of the from-scratch family and electrical networks are noticeably noisier.")
    A(("fig", FIG / "node_comparison.png", 14.0, "Fig. 9. Best validation macro-F1 of each stage network, from scratch and pretrained."))
    A(("fig", FIG / "training_curves.png", 15.5, "Fig. 10. Validation macro-F1 and training loss of the five stage networks."))
    sec("6.3 Ablation Study")
    p("The ablations change one design choice at a time, in the order in which the project progressed (Table 6.2). Each subsection reports what changed and what was measured.")
    sub("6.3.1 Photometric Augmentation")
    def row(k, lab):
        c = R[k]["cascade"]
        return [lab, pct(c["accuracy"]), f"{c['macro_f1']:.3f}", f"{c['kappa']:.3f}", pct(binary_acc(R[k]))]
    A(("table", "Table 6.4: Effect of photometric augmentation (cascade, 40 x 24 input)",
       [["Configuration", "Accuracy", "Macro-F1", "Kappa", "Stage 1 binary"], row("default", "C1 with brightness, contrast and noise jitter"), row("no_photometric", "C2 flips only")], [7.0, 2.2, 2.2, 2.0, 2.6], (), 10))
    p("Removing the jitter raised accuracy by about ten points and cut Stage 1 false alarms from 250 to 118. In an infrared image the absolute brightness of a region is its temperature, which is exactly what separates a hot "
      "spot from a healthy module, so scaling or shifting brightness changes the evidence and not only the appearance. This was the largest single gain of the project and the main lesson about thermal data.")
    sub("6.3.2 Input Size and Network Capacity")
    A(("table", "Table 6.5: Effect of input size and capacity (cascade, flips only)",
       [["Configuration", "Accuracy", "Macro-F1", "Kappa", "Stage 1 binary"], row("no_photometric", "C2 40 x 24 input"), row("improved", "C3 80 x 48 input, two convolutions per block"),
        row("improved3", "C5 plus larger Stage 1 and Stage 2 networks")], [7.0, 2.2, 2.2, 2.0, 2.6], (), 10))
    p("Upsampling the image to 80 x 48 and adding a second convolution per block gave 76.0% accuracy, and enlarging the two front stages gave 79.6%. Stage 1 missed defects fell from 233 to 108. These changes target the "
      "early stages because their errors cannot be corrected later.")
    sub("6.3.3 Balanced Sampling and Label Smoothing")
    p(f"Replacing the class-weighted loss by sampling with weight proportional to 1/sqrt(class count) and adding label smoothing of 0.05 (C4) gave {pct(R['improved2']['cascade']['accuracy'])} accuracy and macro-F1 "
      f"{R['improved2']['cascade']['macro_f1']:.3f}, slightly below the weighted loss ({pct(imp['accuracy'])}, {imp['macro_f1']:.3f}). The class-weighted loss was therefore kept.")
    sub("6.3.4 Inference-Time Variants")
    def vrow(name, lab):
        c = mr_json(name)["cascade"]
        return [lab, pct(c["accuracy"]), f"{c['macro_f1']:.3f}", f"{c['kappa']:.3f}"]
    A(("table", "Table 6.6: Inference-time variants on the improved cascade (no retraining)",
       [["Variant", "Accuracy", "Macro-F1", "Kappa"], vrow("improved", "Hard routing (baseline)"), vrow("improved_tta", "Flip test-time augmentation"),
        vrow("improved_threshold", "Stage 1 threshold tuned on validation data"), vrow("improved_soft", "Soft routing (multiplied probabilities)"), vrow("improved_soft_tta", "Soft routing with test-time augmentation")],
       [8.0, 2.4, 2.4, 2.4], (), 10))
    p("None of the variants helped meaningfully. The validation-optimal Stage 1 threshold stayed at 0.5, so tuning it changed nothing. Soft routing, which multiplies the probabilities of the stages so that a doubtful "
      "family decision can be outvoted, did not recover the Stage 2 errors, which suggests that those errors are confident mistakes and not close calls. Test-time augmentation added 0.3 points at four times the cost.")
    sub("6.3.5 Fault-Family Grouping")
    p("Table 6.7 compares the semantic grouping with the data-driven grouping of Section 4.3. The data-driven grouping reduced wrong-family errors at Stage 2 from 300 to 116, but moved the difficulty into the fine-grained stage, "
      "where its largest family now contains the visually similar classes Cell, Cell-Multi, Soiling, Vegetation and Cracking. The net effect on accuracy was small.")
    A(("table", "Table 6.7: Semantic versus data-driven family grouping (from-scratch cascade)",
       [["Grouping", "Stage 2 errors", "Stage 3 errors", "Total errors", "Accuracy", "Macro-F1"],
        ["Semantic (Electrical, Environmental, Structural)", i3["stages"]["stage2_wrong_family"], i3["stages"]["stage3_wrong_subtype"], i3["stages"]["total_errors"], pct(i3["accuracy"]), f"{i3['macro_f1']:.3f}"],
        ["Data-driven (groups A, B, C)", dg["stages"]["stage2_wrong_family"], dg["stages"]["stage3_wrong_subtype"], dg["stages"]["total_errors"], pct(dg["accuracy"]), f"{dg['macro_f1']:.3f}"]],
       [5.8, 2.2, 2.2, 2.0, 1.8, 1.8], (), 9.5))
    sub("6.3.6 ImageNet Pretraining")
    p(f"Fine-tuning ImageNet-pretrained EfficientNet-B0 at every stage raised the cascade from {pct(dg['accuracy'])} to {pct(pr['accuracy'])} accuracy and macro-F1 from {dg['macro_f1']:.3f} to {pr['macro_f1']:.3f}, and training took minutes instead of "
      "an hour. The pretrained flat CNN improved even more, from 76.8% to 82.6%, so that the two designs end up tied. Pretraining helps on these small thermal images because the early filters (edges, blobs, gradients) transfer, "
      "which refutes the expectation in the project proposal that pretrained backbones would give little benefit.")
    sub("6.3.7 Summary of the Ablations")
    p("In order of effect: removing photometric jitter (+9.4 points), pretraining (+3.4 and +5.8 for the flat model), larger front stages and input (+3.6 and +5.0 on the way), while sampling changes, inference variants and test-time "
      "augmentation gave nothing. The changes that helped were those that preserved or exposed the thermal signal or brought in better features. The changes that did not help tried to repair Stage 2 after the fact.")
    sec("6.4 Error Analysis by Stage")
    p("Table 6.8 attributes each test error to the first stage that went wrong.")
    rows = [["Configuration", "Stage 1 missed defect", "Stage 1 false alarm", "Stage 2 wrong family", "Stage 3 wrong subtype", "Total"]]
    for k, lab in [("improved", "C3 Improved"), ("improved3", "C5 Improved3"), ("datagroup", "C6 Data-driven"), ("pretrained", "C7 Pretrained")]:
        s_ = R[k]["cascade"]["stages"]
        rows.append([lab, s_["stage1_missed_defects"], s_["stage1_false_alarms"], s_["stage2_wrong_family"], s_["stage3_wrong_subtype"], s_["total_errors"]])
    A(("table", "Table 6.8: Test errors per stage (out of 3,000 images)", rows, [3.6, 2.5, 2.4, 2.6, 2.7, 1.5], (4,), 10))
    A(("fig", charts["stages"], 14.0, "Fig. 11. Where the cascade makes its errors, by configuration."))
    p(f"Pretraining reduced errors at every stage compared with C5 (total {i3['stages']['total_errors']} to {pr['stages']['total_errors']}), but Stage 2, the family decision, remains the largest source "
      f"({pr['stages']['stage2_wrong_family']} of {pr['stages']['total_errors']} errors), and an error there cannot be corrected by Stage 3. About {pct(pr['stages']['early_exit_rate'])} of the test images leave the cascade at Stage 1. "
      f"Stage 1 itself is accurate: only {pr['stages']['stage1_missed_defects']} defects were missed and {pr['stages']['stage1_false_alarms']} healthy modules flagged, a binary accuracy of {pct(binary_acc(R['pretrained']))}. "
      "The missed defects matter most in practice, because a missed hot spot is a safety risk, whereas a false alarm costs only an inspection.")
    sec("6.5 Per-Class Performance and Confusion")
    sub("6.5.1 Per-Class F1")
    p("The rare classes are the weak point. They have only 123 to 174 training images, and several look alike. Class weighting, stratified splitting and macro-F1 reduce but do not remove the problem.")
    rows = [["Class", "Test images", "Cascade, from scratch (C5)", "Cascade, pretrained (C7)", "Flat CNN, pretrained"]]
    for c in CLASSES:
        rows.append([c, int(pr["per_class"][c]["support"]), f"{R['improved3']['cascade']['per_class'][c]['f1-score']:.2f}", f"{pr['per_class'][c]['f1-score']:.2f}", f"{prf['per_class'][c]['f1-score']:.2f}"])
    A(("table", "Table 6.9: Per-class F1 on the test set", rows, [3.4, 2.2, 4.0, 3.4, 3.0], (), 10))
    A(("fig", perclass_png, 14.5, "Fig. 12. Per-class F1 on the test set."))
    sub("6.5.2 Confusion Matrices")
    p("Figures 13 and 14 show the row-normalised confusion matrices of the pretrained cascade and the pretrained flat CNN. Both are diagonal for the healthy class and for Diode, Diode-Multi and Cracking, and both "
      "spread the probability mass of Cell, Cell-Multi, Hot-Spot and Soiling over several neighbouring classes. Table 6.10 lists the most frequent confusions of the cascade.")
    A(("fig", FIG / "confusion_cascade.png", 13.5, "Fig. 13. Confusion matrix of the pretrained cascade on the test set (row-normalised, counts inside cells)."))
    A(("fig", FIG / "confusion_flat.png", 13.5, "Fig. 14. Confusion matrix of the pretrained flat CNN on the test set."))
    tc = top_confusions(pr)
    A(("table", "Table 6.10: Most frequent confusions of the pretrained cascade",
       [["True class", "Predicted as", "Images", "Share of the true class"]] + [[a, b, n, f"{100 * n / tot:.0f}%"] for n, a, b, tot in tc], [4.6, 4.6, 2.2, 4.0], (), 10))
    p("The most frequent confusions are between classes of the same physical phenomenon (a single versus several affected cells or diodes) and between faults and the healthy class, in particular for faint "
      "hot spots. These are also the cases where a human inspector would hesitate in a 24 x 40 pixel image.")
    sec("6.6 Comparison with Published Work")
    p("The dataset paper reports no model accuracies, and the figures often attributed to it (Random Forest 84%, CNN 92.5%, EfficientNet + SVM 93%) come from later papers that differ in task, protocol and features. "
      "Table 6.11 therefore sets each published result next to our result on the same or the closest task and gives the difference, but a difference is only a ranking where the protocols agree, as the last column states.")
    d = lambda x, y: f"{(x - y) * 100:+.1f} points"
    rows = [["Published result", "Task and protocol", "Reported", "Our comparable result", "Difference and verdict"],
            ["Alves et al. 2021 [3] (abstract only)", "Binary healthy vs defect, cross-validation", "92.5%", f"{pct(binary_acc(R['pretrained']))} (Stage 1, pretrained)", f"{d(binary_acc(R['pretrained']), 0.925)}: ours higher; one hold-out run"],
            ["Alves et al. 2021 [3]", "8 selected classes, cross-validation", "78.85%", f"{pct(pr['accuracy'])} on all 12 classes", f"{d(pr['accuracy'], 0.7885)}: ours higher on a harder task; classes differ"],
            ["Corley et al. 2025 [4]", "Random Forest, split not stated", "84.0%", f"{pct(pr['accuracy'])} (cascade); {pct(rf['accuracy'])} (our Random Forest)", f"{d(pr['accuracy'], 0.84)}: about equal; our RF is far lower"],
            ["Barraz 2025; Gopalakrishnan 2024, as cited in [4]", "Not stated", "88%; 85%", f"{pct(pr['accuracy'])}", "Lower than 88%, close to 85%; protocols unknown"],
            ["Duranay 2023 [2] (full paper read)", "12 classes, 10-fold cross-validation", "93.93% acc., 89.82% F1", f"{pct(pr['accuracy'])} acc., {pr['macro_f1']:.3f} macro-F1", f"{d(pr['accuracy'], 0.9393)}: ours lower; protocols differ"],
            ["Duranay's method re-run on our split", "12 classes, our fixed hold-out", f"{pct(ps['accuracy'])} acc., {ps['macro_f1']:.3f} F1 (our run)", f"{pct(pr['accuracy'])} acc., {pr['macro_f1']:.3f} macro-F1", f"{d(pr['accuracy'], ps['accuracy'])}: ours higher on identical data"],
            ["Flat CNN, same pretrained backbone", "12 classes, our fixed hold-out", f"{pct(prf['accuracy'])} acc., {prf['macro_f1']:.3f} F1 (our run)", f"{pct(pr['accuracy'])} acc., {pr['macro_f1']:.3f} macro-F1", f"{d(pr['accuracy'], prf['accuracy'])}: essentially tied"],
            ["Millendorf et al. 2020 [1]", "Dataset paper", "No model results", "-", "Nothing to compare"]]
    A(("table", "Table 6.11: Direct comparison of published results with the results of this project", rows, [3.4, 3.2, 2.7, 3.4, 3.3], (5, 6), 8.5))
    bl("Binary triage.", f"Our pretrained Stage 1 reaches {pct(binary_acc(R['pretrained']))} ({pct(binary_acc(R['datagroup']))} from scratch), above the 92.5% of Alves et al. and below the 97.32% of a multi-scale transfer-learning CNN cited by Duranay. It is a single hold-out run, so we claim to be at or slightly above the 92.5% to 93% range and not a new best.")
    bl("12 classes.", f"Our best, {pct(pr['accuracy'])}, is below the published 12-class figures (84.0%, 85% to 88% for ensembles, 93.93%). We did not reach the 93% ceiling or the 95% target stated in the proposal. A Duranay-style pipeline run on our split scored only {pct(ps['accuracy'])}, below our pretrained cascade, which suggests that the gap to 93.93% is largely due to the evaluation protocol rather than to a weaker model; this is a suggestion and not a proof.")
    bl("Why the results are not like for like.", "Duranay uses 10-fold cross-validation, a network trained end to end on the dataset and a supervised feature selection before the cross-validation, so as described, test images can influence these steps. Our test images never do. A raw-pixel Random Forest scores 61.6% on our split against 84.0% in [4], whose protocol is not stated, so part of the gap remains unexplained.")
    sec("6.7 Data Overlap Analysis")
    p("The benchmark has no module or site identifiers, so near-identical crops of the same module can exist. To measure how much this matters, every test image was compared with every training image by the Euclidean "
      "distance between their 24 x 40 pixel vectors. Table 6.12 shows the result. A raw-pixel nearest-neighbour classifier with no training already reaches "
      f"{100 * OVERLAP['one_nn_test_accuracy']:.1f}% accuracy, the same as the raw-pixel Random Forest ({pct(rf['accuracy'])}).")
    A(("table", "Table 6.12: Similarity between test and training images (raw pixels)",
       [["Measure", "Value"],
        ["Test images with a training image within distance 0.5", f"{100 * OVERLAP['test_images_with_train_image_within_0.5']:.1f}%"], ["Label agreement for those images", f"{100 * OVERLAP['label_agreement_within_0.5']:.1f}%"],
        ["Test images with a training image within distance 1.0", f"{100 * OVERLAP['test_images_with_train_image_within_1.0']:.1f}%"], ["Label agreement for those images", f"{100 * OVERLAP['label_agreement_within_1.0']:.1f}%"],
        ["Median distance to the nearest training image", f"{OVERLAP['median_nn_distance']:.2f}"], ["Median distance between random image pairs", f"{OVERLAP['median_random_pair_distance']:.2f}"]],
       [10.0, 4.0], (), 10))
    p("About 7% of the test images have a near-identical training image and about half have a close one, so random fold splits, as used in cross-validated studies, would place such near-duplicates on both sides and raise "
      "their scores. This explains part of the difference to published numbers but not all of it, since a Random Forest scores far lower on our split than the 84.0% reported elsewhere.")
    sec("6.8 Discussion and Limitations")
    sub("6.8.1 Interpretation of Findings")
    p("Three results stand out. Thermal-aware augmentation mattered more than any architectural change. Pretraining gave a large and cheap gain even on 24 x 40 pixel thermal images. And the central hypothesis, that a "
      "cascade clearly beats a flat classifier, was only partly supported: the family decision is the weak link, and with equal backbones the two designs tie. The cascade does give something a flat model cannot: it tells where "
      "an error came from, and it shows that Stage 1 is a reliable gatekeeper.")
    sub("6.8.2 Limitations")
    bl("Single runs.", "Every configuration was run once with seed 42; differences below about 0.5 points are within noise and no confidence intervals were computed. A 10-fold cross-validation of the pretrained cascade was prepared but did not complete, so it is not reported.")
    bl("Near-duplicate images.", "The benchmark has no module or site identifiers, so near-identical crops can fall on both sides of any split.")
    bl("Rare classes.", "Soiling, Hot-Spot and Diode-Multi remain weak, which keeps macro-F1 well below accuracy.")
    bl("Efficiency.", "Latency was not measured reliably, because timings were taken while the GPU was busy, and the pretrained cascade has five times the parameters of the flat model, so the early exit did not give a demonstrated efficiency advantage.")
    bl("Comparison with published work.", "Protocols differ, and some sources could only be checked through abstracts or citing papers.")
    sub("6.8.3 Threats to Validity")
    p("Internal validity is limited by the single seed and by the fact that hyper-parameters were tuned on one validation split. External validity is limited to one dataset of small crops from one provider; the results "
      "may not transfer to higher-resolution images or other sites. Construct validity is limited by label noise, which cannot be assessed because no second annotation exists.")
    if nndl:
        import srm_content_nndl as sn
        B.extend(sn.discussion_nndl())
    sec("6.9 Chapter Summary")
    p(f"The best cascade reaches {pct(pr['accuracy'])} accuracy with ImageNet pretraining and {pct(dg['accuracy'])} from scratch, with Stage 1 at {pct(binary_acc(R['pretrained']))}. The cascade matches but does not clearly beat a flat classifier, "
      "the family stage is its largest error source, and published 12-class results are higher under protocols that are not comparable with a fixed hold-out.")
    return B


def mr_json(name):
    return json.load(open(ROOT / "results" / f"{name}.json"))


def appendix_blocks(demo_mix_png):
    B = []
    A = B.append
    A(("appendix", "A", "CONFIGURATION FILES"))
    p = lambda t: A(("p", t))
    p("This appendix reproduces the two configuration files that define the best from-scratch experiment and the pretrained experiment. Every reported result can be regenerated from the configuration, the saved split file and the seed.")
    A(("code", "Listing A.1: configs/improved3.yaml (from-scratch cascade)", (ROOT / "configs" / "improved3.yaml").read_text(encoding="utf-8").replace("\r\n", "\n").rstrip()))
    A(("code", "Listing A.2: configs/pretrained.yaml (pretrained cascade)", (ROOT / "configs" / "pretrained.yaml").read_text(encoding="utf-8").replace("\r\n", "\n").rstrip()))
    A(("code", "Listing A.3: configs/taxonomy.yaml (semantic class hierarchy)", (ROOT / "configs" / "taxonomy.yaml").read_text(encoding="utf-8").replace("\r\n", "\n").rstrip()))
    A(("appendix", "B", "KEY SOURCE CODE"))
    p("The listings show the three pieces of code that define the behaviour of the system: the stage network and the cascade routing, the augmentation, and the attribution of errors to stages. The complete code is available in the project repository.")
    A(("code", "Listing B.1: Stage network and cascade routing (src/solarnet/models.py)", src("src/solarnet/models.py", r"^class StageCNN", r"^class PretrainedStage")))
    A(("code", "Listing B.2: Cascade inference with early exit (src/solarnet/models.py)", src("src/solarnet/models.py", r"def predict\(self", r"def predict_soft")))
    A(("code", "Listing B.3: Training augmentation (src/solarnet/common.py)", src("src/solarnet/common.py", r"^def augment", r"^def preprocess")))
    A(("code", "Listing B.4: Attribution of errors to stages (src/solarnet/evaluate.py)", src("src/solarnet/evaluate.py", r"^def stage_errors", r"^@torch.inference_mode")))
    A(("code", "Listing B.5: Class weights for the imbalanced loss (src/solarnet/data.py)", src("src/solarnet/data.py", r"^def class_weights")))
    A(("appendix", "C", "RUNNING THE DEMONSTRATIONS"))
    p("Two demonstrations are provided. Both need the trained checkpoints and the dataset, which is downloaded by a script from the public repository of the dataset.")
    A(("b", "Command-line demonstration.", "python scripts/predict.py --sample 5 draws, for random test images, the decision and confidence of each stage and saves the figure. The option --class Hot-Spot restricts the sample to one class and --image path loads any image."))
    A(("b", "Browser demonstration.", "python scripts/make_demo_html.py creates a single HTML page with three test images per class. Opening it in a browser shows the selected image, and pressing Run cascade reveals Stage 1, Stage 2 and Stage 3 one after another, with the final prediction in green if it is correct and red if not."))
    A(("b", "Reproducing the results.", "python scripts/make_splits.py creates the split, python -m solarnet.train --node all --config configs/pretrained.yaml trains the five networks, and python -m solarnet.evaluate --flat --config configs/pretrained.yaml evaluates the cascade and the flat baseline."))
    A(("fig", demo_mix_png, 15.5, "Fig. 15. Example output of the command-line demonstration for random test images (green: correct, red: wrong)."))
    A(("appendix", "D", "REPOSITORY STRUCTURE"))
    A(("table", "Table D.1: Layout of the project repository",
       [["Path", "Content"],
        ["src/solarnet/", "Package: taxonomy, data, models, common helpers, training, evaluation"], ["configs/", "Experiment configurations and the class hierarchies (semantic and data-driven)"],
        ["scripts/", "Download and split scripts, baselines, cross-validation, figure and report generators, demonstrations"], ["tests/", "Automated tests that run on synthetic data"],
        ["results/", "Raw metrics of every reported experiment (JSON)"], ["checkpoints/", "Trained Stage 3 specialists shared with the team"],
        ["docs/", "Architecture diagrams and the reports"], ["notebooks/", "Notebook that runs the whole pipeline on Colab"]],
       [4.0, 11.8], (), 10))
    return B


def build_all(charts, perclass_png, demo_png, demo_mix_png, nndl=False):
    import srm_content3 as sc3
    old = sc.build_blocks(charts, perclass_png, demo_png)
    i2 = next(i for i, b in enumerate(old) if b[0] == "chapter" and b[1] == 2)
    i3_ = next(i for i, b in enumerate(old) if b[0] == "chapter" and b[1] == 3)
    i5 = next(i for i, b in enumerate(old) if b[0] == "chapter" and b[1] == 5)
    i7 = next(i for i, b in enumerate(old) if b[0] == "chapter" and b[1] == 7)
    blocks = old[:i2] + sc3.chapter_2() + old[i3_:i5] + chapters_5_6(charts, perclass_png, demo_png, nndl) + old[i7:]
    return blocks, appendix_blocks(demo_mix_png)
