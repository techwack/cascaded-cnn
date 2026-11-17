"""Content of the Computer Vision project report (format: sample GOI CV report; IEEE numbered references)."""
import json

import srm_content as sc
import srm_content2 as c2

ROOT, FIG, R, pct, binary_acc, CLASSES = sc.ROOT, sc.FIG, sc.R, sc.pct, sc.binary_acc, sc.CLASSES
pr, prf, dg, i3, imp, fl, rf, ps = sc.pr, sc.prf, sc.dg, sc.i3, sc.imp, sc.fl, sc.rf, sc.ps
HC = json.load(open(ROOT / "results" / "handcrafted_svm.json"))
TITLE = sc.TITLE
S = pr["stages"]
COURSE = "21CSE454T – COMPUTER VISION"

ABSTRACT = (
    "Automated inspection of photovoltaic (PV) installations from drone images is a computer-vision problem: the images are thermal, tiny (24 x 40 pixels), strongly imbalanced across 12 classes and must be "
    "classified fast on edge hardware. This project studies it end to end on the InfraredSolarModules benchmark (20,000 images). It compares a classical pipeline (41 hand-crafted intensity, gradient and hot-region "
    "features with an SVM) with convolutional networks, and proposes Cascaded-SolarNet, a cascade of five small CNNs that first separates healthy from defective modules, then assigns a fault family and finally the exact "
    "fault. The report analyses the image data, derives why brightness is a measurement and not an appearance (so photometric augmentation destroys information), visualises learned filters, feature maps and receptive fields, "
    "and uses Grad-CAM to show where each stage looks. On a fixed held-out split the classical pipeline reaches "
    f"{pct(HC['accuracy'])} accuracy, the from-scratch cascade {pct(dg['accuracy'])} and the cascade with an ImageNet-pretrained EfficientNet-B0 {pct(pr['accuracy'])} (macro-F1 {pr['macro_f1']:.2f}), with "
    f"{pct(binary_acc(R['pretrained']))} on the healthy-versus-defect decision. A flat network with the same backbone is essentially tied with the cascade, and the family decision causes more than half of the errors. "
    "Published 12-class results use cross-validation and are compared with these results with their limits stated."
)

ABBREVIATIONS = [
    ("CAM", "Class Activation Map"), ("CNN", "Convolutional Neural Network"), ("CV", "Computer Vision / Cross-Validation"), ("FC", "Fully Connected (layer)"),
    ("GAP", "Global Average Pooling"), ("Grad-CAM", "Gradient-weighted Class Activation Mapping"), ("HOG", "Histogram of Oriented Gradients"), ("IR", "Infrared"),
    ("MBConv", "Mobile Inverted Bottleneck Convolution"), ("PV", "Photovoltaic"), ("ReLU", "Rectified Linear Unit"), ("RBF", "Radial Basis Function"),
    ("SE", "Squeeze-and-Excitation"), ("SVM", "Support Vector Machine"), ("UAV", "Unmanned Aerial Vehicle"), ("F1", "Harmonic mean of precision and recall"),
    ("TTA", "Test-Time Augmentation"), ("NCA", "Neighbourhood Component Analysis"),
]

REFERENCES = [
    "M. Millendorf, E. Obropta, and N. Vadhavkar, “Infrared solar module dataset for anomaly detection,” ICLR 2020 Workshop on AI for Earth Sciences, 2020.",
    "Z. B. Duranay, “Fault detection in solar energy systems: A deep learning approach,” Electronics, vol. 12, no. 21, p. 4397, 2023, doi: 10.3390/electronics12214397.",
    "R. H. F. Alves, G. A. de Deus Junior, E. G. Marra, and R. P. Lemos, “Automatic fault classification in photovoltaic modules using convolutional neural networks,” Renewable Energy, vol. 179, pp. 502–516, 2021.",
    "Corley et al., “Aerial infrared health monitoring of solar photovoltaic farms at scale,” in CVPR Workshops (PBVS), 2025, arXiv:2503.02128.",
    "S. P. Pathak and S. A. Patil, “Enhanced fault identification in solar panels through binary cascaded convolutional classifiers with thermal-visual augmentation,” Int. J. Image, Graphics and Signal Processing, vol. 17, no. 4, 2025.",
    "“PV-faultNet: Optimized convolutional neural network to detect defects in PV production,” arXiv:2411.02997, 2024.",
    "U. Otamendi, I. Martinez, M. Quartulli, et al., “Segmentation of cell-level anomalies in electroluminescence images of photovoltaic modules,” Solar Energy, vol. 220, pp. 914–926, 2021.",
    "H. Tella, A. Hussein, S. Rehman, et al., “PV panel defect classification using deep learning ensemble methods,” Case Studies in Thermal Engineering, vol. 66, 2025.",
    "Z. Cai and N. Vasconcelos, “Cascade R-CNN: Delving into high quality object detection,” in Proc. CVPR, 2018, pp. 6154–6162.",
    "P. Viola and M. Jones, “Rapid object detection using a boosted cascade of simple features,” in Proc. CVPR, 2001.",
    "Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, “Gradient-based learning applied to document recognition,” Proc. IEEE, vol. 86, no. 11, pp. 2278–2324, 1998.",
    "A. Krizhevsky, I. Sutskever, and G. E. Hinton, “ImageNet classification with deep convolutional neural networks,” in Advances in Neural Information Processing Systems 25, 2012.",
    "K. Simonyan and A. Zisserman, “Very deep convolutional networks for large-scale image recognition,” in Proc. ICLR, 2015.",
    "K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proc. CVPR, 2016, pp. 770–778.",
    "M. Tan and Q. V. Le, “EfficientNet: Rethinking model scaling for convolutional neural networks,” in Proc. ICML, 2019, arXiv:1905.11946.",
    "N. Dalal and B. Triggs, “Histograms of oriented gradients for human detection,” in Proc. CVPR, 2005.",
    "R. R. Selvaraju et al., “Grad-CAM: Visual explanations from deep networks via gradient-based localization,” in Proc. ICCV, 2017.",
    "S. Ioffe and C. Szegedy, “Batch normalization: Accelerating deep network training by reducing internal covariate shift,” in Proc. ICML, 2015.",
    "D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” in Proc. ICLR, 2015.",
    "N. Srivastava et al., “Dropout: A simple way to prevent neural networks from overfitting,” J. Mach. Learn. Res., vol. 15, pp. 1929–1958, 2014.",
    "J. Yosinski, J. Clune, Y. Bengio, and H. Lipson, “How transferable are features in deep neural networks?” in Advances in Neural Information Processing Systems 27, 2014.",
    "J. Deng et al., “ImageNet: A large-scale hierarchical image database,” in Proc. CVPR, 2009.",
    "J. Hu, L. Shen, and G. Sun, “Squeeze-and-excitation networks,” in Proc. CVPR, 2018.",
    "A. Paszke et al., “PyTorch: An imperative style, high-performance deep learning library,” in Advances in Neural Information Processing Systems 32, 2019.",
    "R. C. Gonzalez and R. E. Woods, Digital Image Processing, 4th ed. Pearson, 2018.",
    "Z. Barraz, I. Sebari, N. Lamrini, K. Ait El Kadi, and I. Ait Abdelmoula, “A cascading decision system for enhanced anomaly classification of large-scale photovoltaic systems using drone’s thermal data with class-imbalance problem,” Results in Engineering, vol. 25, p. 103876, 2025.",
]


def blocks(charts, perclass_png):
    B = []
    A = B.append
    sec = lambda t: A(("sec", t))
    sub = lambda t: A(("sub", t))
    p = lambda t: A(("p", t))
    eq = lambda t: A(("eq", t))
    bl = lambda lead, t: A(("b", lead, t))
    F = lambda path, w, cap: A(("fig", path, w, cap))

    # =============================================================== CHAPTER 1
    A(("chapter", 1, "INTRODUCTION"))
    sec("1.1 Introduction to the Project")
    p("Photovoltaic (PV) power plants contain thousands of modules, and defects such as hot spots, cracked cells, soiling, shadowing and failed bypass diodes lower their output and can cause fires. Manual inspection with "
      "handheld thermal cameras is slow and does not scale, so unmanned aerial vehicles (UAVs) with infrared (IR) cameras are used, and a computer-vision system has to label every module in the images automatically [1], [4].")
    p("This project treats the task as image classification on a very small, thermal image domain. Each input is one 24 x 40 pixel grayscale image of a module, and the output is one of 12 classes: a healthy class and eleven "
      "fault types. The project compares classical feature engineering with convolutional neural networks (CNNs) [11], [12], [14], studies what the networks learn (filters, feature maps, receptive fields, Grad-CAM heat maps) "
      "[16], [17], and builds a cascade of small CNNs inspired by the classical idea of rejecting easy negatives first [10]. Figure 1.1 shows one example per class.")
    F(FIG / "sample_images.png", 14.5, "Figure 1.1: One test image of each class of the InfraredSolarModules dataset (inferno colour map).")
    sec("1.2 Problem Statement and Description")
    p("Given a single-channel IR image of 24 x 40 pixels showing one PV module, assign it to one of 12 classes. Four properties make the problem hard from a computer-vision point of view:")
    bl("Tiny images.", "There is almost no texture; the information is the position, size and shape of warm regions.")
    bl("Class imbalance.", "Half of the 20,000 images are healthy, while the rarest classes have 175 to 249 images.")
    bl("Look-alike classes.", "Several classes differ only in how many cells or diodes are affected (Cell and Cell-Multi, Diode and Diode-Multi, Hot-Spot and Hot-Spot-Multi).")
    bl("Edge deployment.", "The classifier must be small and fast enough for drone hardware.")
    p("A single network that chooses among all 12 classes (a flat classifier) tends to plateau and to favour the healthy class. The project asks whether a cascade of specialised networks does better, and where each approach fails.")
    sec("1.3 Motivation")
    p("The cost of a wrong decision is concrete: a missed hot spot is a fire risk, and a false alarm wastes an inspection. A good system must therefore be accurate on the rare, dangerous classes and not only on average. "
      "From a computer-vision perspective the dataset is also a good laboratory: the images are small enough to train many models quickly, the imbalance is severe, and the physical meaning of the pixel values (temperature) "
      "creates a clear test of which standard image-processing practices are valid. The project shows that one common practice, brightness augmentation, is invalid here.")
    sec("1.4 Relevance and Application Areas")
    p("The methods apply to any inspection task with small images, rare defects and a dominant normal class: solar-farm inspection by drone, thermal inspection of power lines and buildings, and industrial quality control. "
      "Reliable automatic inspection supports clean-energy output (UN Sustainable Development Goal 7), reduces the cost of maintenance, and the cascade idea of letting cheap models handle the easy majority supports "
      "low-power edge computing.")

    # =============================================================== CHAPTER 2
    A(("chapter", 2, "LITERATURE SURVEY"))
    p("This chapter reviews image-classification methods that are relevant to the problem, from classical features to CNNs, and the work on PV defect analysis, and derives the research objectives.")
    sec("2.1 Image Classification: From Hand-Crafted Features to CNNs")
    p("Classical image classification extracts hand-designed features and trains a classifier on them. Histograms of oriented gradients (HOG) describe local edge directions [16], and boosted cascades of simple features "
      "made real-time detection possible by rejecting most windows early [10]. Convolutional neural networks learn the features instead. LeNet showed the principle on digits [11]; AlexNet won ImageNet in 2012 by a large margin "
      "[12]; VGG showed the benefit of depth with 3 x 3 filters [13]; residual connections made very deep networks trainable [14]; and EfficientNet scaled depth, width and resolution together [15]. Batch normalisation [18], "
      "dropout [20] and the Adam optimiser [19] are the standard ingredients of training.")
    sec("2.2 Thermal Imaging and PV Defect Analysis")
    p("In infrared thermography brightness encodes temperature: healthy modules heat nearly uniformly, defects create localised or characteristic patterns. The InfraredSolarModules dataset [1] standardises 12 classes from "
      "real drone flights at 24 x 40 pixels. The dataset paper reports no model accuracies. Later work reports 92.5% for binary detection and 78.85% for eight classes with a CNN [3], 93.93% for 12 classes with "
      "ImageNet-pretrained EfficientNet-B0 features and an SVM under 10-fold cross-validation [2], and 84.0% for a random forest [4]. Pathak and Patil use a binary cascade with thermal and visible images and report "
      "98.5% accuracy [5], but need two sensors. PV-faultNet finds that a compact CNN reaches 93% at 8 ms per image and larger backbones add little [6].")
    sec("2.3 Detection and Segmentation Approaches")
    p("Detection and segmentation pipelines (Faster R-CNN, Cascade R-CNN, VarifocalNet, YOLO) localise defects [9] and have been applied to electroluminescence images [7]. They need box or mask labels, add latency, and "
      "fragment thin defects such as cracks. The InfraredSolarModules benchmark has one label per crop and no boxes, so whole-image classification is the appropriate formulation.")
    sec("2.4 Transfer Learning and Augmentation")
    p("Features learned on ImageNet [22] transfer to other tasks, especially in their early layers [21], so fine-tuning a pretrained network is the default for small datasets. Augmentation enlarges the training set with "
      "transformations that preserve the label. For natural images these include flips, crops, brightness and contrast changes. Whether a transformation preserves the label depends on what the pixels mean, "
      "which is the key issue for thermal images (Chapter 3).")
    sec("2.5 Cascaded Classifiers")
    p("A cascade applies a cheap test first and passes on only the hard cases. In detection it removes most background windows early [10]; Cascade R-CNN refines boxes in stages [9]; and Pathak and Patil use a "
      "binary-first cascade for PV faults [5]. A cascade has a cost: an error of an early stage cannot be corrected later, so its accuracy is roughly the product of the stages' accuracies. Whether the benefit outweighs this "
      "cost is an empirical question.")
    sec("2.6 Comparison of Existing Approaches")
    A(("table", "Table 2.1: Comparison of existing approaches",
       [["Approach", "Example", "Strengths", "Weaknesses"],
        ["Hand-crafted features + classifier", "HOG, intensity statistics, SVM or random forest [4], [16]", "Small, interpretable, fast", "Features limited by design; lower accuracy"],
        ["Flat CNN", "Custom CNN [3], PV-faultNet [6]", "Simple, end to end", "Plateaus; biased to the dominant class"],
        ["Pretrained backbone + classifier", "EfficientNet-B0 + SVM [2]", "High accuracy", "Heavier; usually cross-validated"],
        ["Detection / segmentation", "Cascade R-CNN, VarifocalNet [7], [9]", "Locates defects", "Needs boxes; slow; fragments thin defects"],
        ["Two-sensor cascade", "Binary cascade with visible images [5]", "Highest accuracy", "Needs two cameras and fusion"],
        ["Single-sensor classification cascade", "This project", "One IR camera, small networks, error attribution", "Errors propagate between stages"]],
       [3.6, 4.2, 3.6, 4.4], (), 9))
    sec("2.7 Research Objectives and Contributions")
    bl("O1.", "Compare a classical hand-crafted-feature pipeline with CNNs on the same fixed split.")
    bl("O2.", "Determine which image-processing practices are valid for thermal data, in particular augmentation.")
    bl("O3.", "Design and evaluate a cascade of small CNNs and a pretrained variant, and compare them with a flat network.")
    bl("O4.", "Explain the models visually (filters, feature maps, receptive fields, Grad-CAM) and attribute every error to a stage.")
    p("Contributions: a reproducible implementation with a leak-free evaluation protocol, a controlled ablation study, a measured finding that brightness augmentation costs about ten points on thermal images, "
      "an error analysis by stage, and a corrected comparison with published results.")

    # =============================================================== CHAPTER 3
    A(("chapter", 3, "METHODOLOGY"))
    sec("3.1 Image Formation and the Meaning of Pixel Values")
    p("An infrared camera measures the radiation emitted by a surface. The radiant exitance of a body follows the Stefan-Boltzmann law, M = epsilon x sigma x T^4, where T is the absolute temperature, sigma the "
      "Stefan-Boltzmann constant and epsilon the emissivity of the surface [25]. The sensor converts the received radiation into a signal that is mapped to a grey level, which is a monotonic function of the apparent temperature. "
      "In the dataset the images are stored as 8-bit grayscale; the exact temperature-to-grey mapping is not documented, so brightness is treated as a monotonic proxy of temperature.")
    p("This has a decisive consequence. In a photograph, a brightness change is mostly a change of illumination and leaves the class unchanged. In a thermal image, a brightness change of a region is a change of "
      "its temperature, which is exactly the evidence that separates a hot spot from a healthy module. The pipeline therefore avoids photometric transformations.")
    sec("3.2 Pipeline Overview")
    p("Figure 3.1 shows the complete system. The preprocessing normalises the image; the first network decides healthy or defective; defective images go to a family network and then to one specialist network.")
    F(FIG / "overall_cascade.png", 11.5, "Figure 3.1: The Cascaded-SolarNet pipeline: triage, family classification and one specialist per family.")
    sec("3.3 Preprocessing and Data Augmentation")
    sub("3.3.1 Normalisation and Resizing")
    p("Pixel values are scaled to [0, 1] and standardised with the mean and standard deviation of the training split only, x' = (x - mean_train) / std_train (mean 0.618, std 0.156). The from-scratch networks upsample the "
      "image by a factor of two with bilinear interpolation, I'(u, v) = sum of the four neighbouring pixels weighted by their distance, to 80 x 48 so that small warm regions survive three pooling steps. The pretrained networks "
      "resize the image to 160 x 96 and repeat the grey channel three times to match the RGB input of ImageNet.")
    sub("3.3.2 Augmentation as a Question of Invariance")
    p("Augmentation encodes which transformations should not change the label. Horizontal and vertical flips are valid: a fault is defined by which cells or diodes are hot and not by the orientation of the module "
      "in the image. Rotations by 90 degrees would change the 40 x 24 shape and are not used. Brightness, contrast and noise jitter were tested, and removed because they changed the evidence (Section 4.3).")
    sec("3.4 Classical Baseline: Hand-Crafted Features and SVM")
    p("As a classical computer-vision baseline, 41 features were designed from the physical properties of the images (Table 3.1) and classified with an RBF support-vector machine. The features describe the intensity "
      "distribution, where the heat is located, how sharp the edges are, and the size and number of hot regions, which are found with a threshold and a 4-connected component analysis.")
    A(("table", "Table 3.1: The 41 hand-crafted features",
       [["Group", "Features", "Count"],
        ["Intensity statistics", "Mean, standard deviation, minimum, maximum, percentiles (5, 25, 50, 75, 95), skewness, kurtosis", 11],
        ["Spatial layout", "Mean of the top, middle and bottom thirds; left and right halves; variation of the column and row means", 7],
        ["Gradient", "Mean, standard deviation and maximum of the gradient magnitude; 8-bin magnitude-weighted orientation histogram (HOG-like)", 11],
        ["Hot regions", "For thresholds mean + 1 std and mean + 2 std: area fraction, number of components, largest component, centroid (row, column)", 10],
        ["Extremes", "Fraction of very bright (> 0.8) and very dark (< 0.3) pixels", 2]],
       [3.4, 10.4, 1.6], (), 9.5))
    sec("3.5 CNN Feature Extraction")
    sub("3.5.1 Convolution as Learned Filtering")
    p("A convolutional layer correlates the image with learned small kernels. For output channel c, y[c, i, j] = b[c] + sum over (m, u, v) of W[c, m, u, v] x x[m, i + u, j + v]. In classical image processing "
      "the kernels (edge detectors, blob detectors) are designed by hand; here they are learned. Figure 3.2 shows the 64 learned 3 x 3 filters of the first layer of the triage network: many resemble oriented edge "
      "and gradient detectors, others respond to the local brightness level.")
    F(FIG / "cv_filters.png", 8.5, "Figure 3.2: The 64 learned 3 x 3 filters of the first convolutional layer of CNN-1.")
    sub("3.5.2 Network Structure and Feature Maps")
    p("Each stage network has three blocks of two 3 x 3 convolutions (BatchNorm and ReLU after each) and a 2 x 2 max pooling, with 64, 128 and 128 channels in the first stage and 128, 256 and 128 in the others, followed by "
      "global average pooling, dropout and a fully connected layer (Figure 3.3, Table 3.2). Figure 3.4 shows feature maps for a Hot-Spot image: early maps stay close to the input and highlight edges and intensity levels, later "
      "maps are smaller and respond to larger structures.")
    F(FIG / "stage1_cnn1.png", 15.5, "Figure 3.3: Structure of the triage network CNN-1 with the feature-map size after each block.")
    F(FIG / "cv_featuremaps.png", 15.5, "Figure 3.4: Feature maps of CNN-1 for a Hot-Spot image (first 8 channels after the first ReLU of each block).")
    A(("table", "Table 3.2: The five stage networks (from-scratch variant)",
       [["Network", "Task", "Channels", "Output", "Parameters"],
        ["CNN-1", "Healthy vs defect", "64-128-128", "2", c2.STATS["improved3"]["triage"]["params"]], ["CNN-2", "Fault family", "128-256-128", "3", c2.STATS["improved3"]["family"]["params"]],
        ["CNN-3a", "Electrical subtype", "128-256-128", "6", c2.STATS["improved3"]["electrical"]["params"]], ["CNN-3b", "Environmental subtype", "128-256-128", "3", c2.STATS["improved3"]["environmental"]["params"]],
        ["CNN-3c", "Structural subtype", "128-256-128", "2", c2.STATS["improved3"]["structural"]["params"]]], [2.2, 4.4, 3.2, 2.2, 3.4], (), 10))
    sub("3.5.3 Receptive Field")
    p("The receptive field of a unit is the region of the input that can influence it. It grows by (k - 1) times the current jump per layer, and a stride s multiplies the jump by s. For the three blocks of CNN-1 it grows "
      "from 3 to 36 pixels of the 80 x 48 input (Figure 3.5), which is 18 x 18 pixels of the original image, about three quarters of its width. The network can therefore relate the position of a warm region to the layout of the whole module.")
    F(FIG / "nndl_receptive_field.png", 14.0, "Figure 3.5: Growth of the receptive field through the layers of CNN-1.")
    sec("3.6 The Cascade")
    p("CNN-1 decides healthy or defective; a healthy prediction ends the pipeline (early exit). CNN-2 assigns a defect to one of three families (Electrical, Environmental, Structural). One specialist per family (6, 3 and 2 "
      "classes) names the fault, and the final label is the family's class offset plus the specialist's prediction. Each network is trained on ground-truth routing and the cascade is assembled at test time, so "
      "the error of every stage is measured as it would occur in use. The family grouping can be semantic or data-driven (derived from the confusions of a flat network on the validation split).")
    sec("3.7 Transfer Learning")
    p("In the pretrained variant each network is an ImageNet-pretrained EfficientNet-B0 [15], [22] with a new classification head, fine-tuned at a small learning rate (Figure 3.6). EfficientNet-B0 uses mobile inverted-bottleneck "
      "blocks with depthwise convolutions, residual connections and squeeze-and-excitation channel attention [23]. The source domain (natural colour photographs) differs from the target (tiny thermal images), and "
      "the experiments measure whether the early filters transfer anyway.")
    F(FIG / "nndl_transfer_learning.png", 15.0, "Figure 3.6: Transfer learning from ImageNet to the thermal defect task.")
    sec("3.8 Training and Evaluation Protocol")
    p("The 20,000 images are split once, stratified by class with a fixed seed, into 70% training, 15% validation and 15% test images. The test images are never used for training, early stopping, normalisation or tuning. "
      "Training uses Adam [19] (weight decay 1e-4, batch size 64), a class-weighted cross-entropy loss with weights N / (K x n), random flips as the only augmentation, and early stopping on the validation macro-F1. "
      "The learning rate is 5e-4 for the from-scratch networks and 3e-4 for the pretrained ones. Results are reported as accuracy, macro-F1 (which weights every class equally), Cohen's kappa, the accuracy of the "
      "healthy-versus-defect decision, and the number of errors caused by each stage. All experiments are single runs (seed 42), implemented in PyTorch [24].")
    sec("3.9 Visual Explanation with Grad-CAM")
    p("Grad-CAM [17] shows which image regions influence a decision. For a class score y and the activation maps A of the last convolutional layer, the channel weights are the average gradients "
      "alpha_c = (1 / Z) sum over (i, j) of dy / dA_c(i, j), and the heat map is")
    eq("L = ReLU( sum over c of alpha_c x A_c )")
    p("up-sampled to the image size. It is computed here for the defect score of CNN-1 and for the predicted class of the selected specialist, which shows where each stage looks.")

    # =============================================================== CHAPTER 4
    A(("chapter", 4, "RESULTS AND DISCUSSION"))
    sec("4.1 Experimental Methodology")
    p("All numbers are measured on the 3,000 held-out test images (Table 4.1). Eleven configurations were compared: the classical baseline, a raw-pixel random forest, flat and cascaded CNNs from scratch in several "
      "refinements, and the pretrained variants.")
    A(("table", "Table 4.1: Class distribution and stratified split",
       [["Class", "Total", "Train", "Validation", "Test"],
        ["No-Anomaly", 10000, 7000, 1500, 1500], ["Cell", 1877, 1314, 282, 281], ["Cell-Multi", 1288, 902, 193, 193], ["Diode", 1499, 1049, 225, 225],
        ["Diode-Multi", 175, 123, 26, 26], ["Hot-Spot", 249, 174, 38, 37], ["Hot-Spot-Multi", 246, 172, 37, 37], ["Soiling", 204, 143, 30, 31],
        ["Shadowing", 1056, 739, 158, 159], ["Vegetation", 1639, 1147, 246, 246], ["Cracking", 940, 658, 141, 141], ["Offline-Module", 827, 579, 124, 124],
        ["Total", 20000, 14000, 3000, 3000]], [4.4, 2.6, 2.6, 3.0, 2.6], (13,), 10))
    sec("4.2 Experimental Results")
    sub("4.2.1 Overall Performance")
    p(f"Table 4.2 lists the results. The classical pipeline with {HC['n_features']} hand-crafted features reaches {pct(HC['accuracy'])} accuracy (macro-F1 {HC['macro_f1']:.3f}), clearly better than the raw-pixel random forest "
      f"({pct(rf['accuracy'])}), which shows that features designed from the physics of the images help. CNNs learn better features: the from-scratch cascade reaches {pct(dg['accuracy'])}, and the pretrained cascade "
      f"{pct(pr['accuracy'])} with macro-F1 {pr['macro_f1']:.3f}, Cohen's kappa {pr['kappa']:.3f} and {pct(binary_acc(R['pretrained']))} on the healthy-versus-defect decision (Figure 4.1).")
    rows, best = sc._t6_rows()
    rows.insert(len(rows) - 2, ["Hand-crafted features + SVM", "SVM", pct(HC["accuracy"]), f"{HC['macro_f1']:.3f}", f"{HC['kappa']:.3f}", pct(HC["binary_accuracy"])])
    A(("table", "Table 4.2: Test results of all configurations", rows, [5.6, 2.2, 2.2, 2.2, 1.8, 2.4], (best,), 9.5))
    F(charts["prog"], 14.0, "Figure 4.1: Test accuracy and macro-F1 of the cascade by configuration.")
    sub("4.2.2 Training Behaviour")
    p("Figure 4.2 shows the validation macro-F1 and training loss of the five networks. The pretrained networks reach their best validation score within 10 to 20 epochs and their curves are smooth. "
      "The from-scratch family and electrical networks need 50 to 140 epochs and their validation curves are noisy, a sign of high variance on small training sets.")
    F(FIG / "training_curves.png", 15.5, "Figure 4.2: Validation macro-F1 and training loss of the five stage networks, from scratch and pretrained.")
    sec("4.3 Ablation: What Matters in Image Processing for Thermal Data")
    def row(k, lab):
        c = R[k]["cascade"]
        return [lab, pct(c["accuracy"]), f"{c['macro_f1']:.3f}", f"{c['kappa']:.3f}", pct(binary_acc(R[k]))]
    A(("table", "Table 4.3: Effect of the main design choices (cascade)",
       [["Configuration", "Accuracy", "Macro-F1", "Kappa", "Stage 1 binary"],
        row("default", "C1 with brightness, contrast and noise jitter"), row("no_photometric", "C2 flips only"), row("improved", "C3 + 80 x 48 input, two convolutions per block"),
        row("improved3", "C5 + larger Stage 1 and 2 networks"), row("datagroup", "C6 data-driven families"), row("pretrained", "C7 pretrained EfficientNet-B0")], [7.0, 2.2, 2.2, 2.0, 2.6], (6,), 10))
    bl("Photometric augmentation.", "Removing brightness, contrast and noise jitter raised accuracy by about ten points (61.7% to 71.1%) and cut Stage 1 false alarms from 250 to 118. Since brightness is temperature, the jitter changes the evidence and not only the appearance, as predicted in Section 3.1.")
    bl("Resolution and capacity.", f"Upsampling to 80 x 48 with two convolutions per block gave {pct(imp['accuracy'])}, and larger front stages {pct(i3['accuracy'])}.")
    bl("Transfer learning.", f"Pretraining raised the cascade to {pct(pr['accuracy'])} and the flat network from {pct(fl['accuracy'])} to {pct(prf['accuracy'])}, so the two designs tie.")
    bl("Did not help.", "Soft routing, flip test-time augmentation, threshold tuning and balanced sampling with label smoothing.")
    sec("4.4 Error Analysis")
    sub("4.4.1 Errors by Stage")
    rows = [["Configuration", "Stage 1 missed", "Stage 1 false alarm", "Stage 2 wrong family", "Stage 3 wrong subtype", "Total"]]
    for k, lab in [("improved3", "From scratch (C5)"), ("pretrained", "Pretrained (C7)")]:
        s_ = R[k]["cascade"]["stages"]
        rows.append([lab, s_["stage1_missed_defects"], s_["stage1_false_alarms"], s_["stage2_wrong_family"], s_["stage3_wrong_subtype"], s_["total_errors"]])
    A(("table", "Table 4.4: Test errors per stage (out of 3,000 images)", rows, [3.4, 2.4, 2.6, 2.7, 2.8, 1.5], (2,), 10))
    p(f"Stage 2, the family decision, causes {S['stage2_wrong_family']} of {S['total_errors']} errors of the pretrained cascade, and an error there cannot be corrected by Stage 3. Stage 1 is accurate ({pct(binary_acc(R['pretrained']))}); "
      f"about {pct(S['early_exit_rate'])} of the images exit there.")
    F(charts["stages"], 13.5, "Figure 4.3: Where the cascade makes its errors, by configuration.")
    sub("4.4.2 Per-Class Behaviour and Confusion")
    p("The rare classes are the weakest (Figure 4.4). The confusion matrix of the pretrained cascade (Figure 4.5) shows that errors concentrate between classes of one physical phenomenon, a single versus several "
      "affected cells or diodes, and between faint hot spots and the healthy class.")
    F(perclass_png, 14.0, "Figure 4.4: Per-class F1 on the test set.")
    F(FIG / "confusion_cascade.png", 12.0, "Figure 4.5: Confusion matrix of the pretrained cascade (row-normalised, counts in cells).")
    tc = c2.top_confusions(pr)
    A(("table", "Table 4.5: Most frequent confusions of the pretrained cascade",
       [["True class", "Predicted as", "Images", "Share of the true class"]] + [[a, b, n, f"{100 * n / tot:.0f}%"] for n, a, b, tot in tc], [4.6, 4.6, 2.2, 4.0], (), 10))
    sec("4.5 Visual Explanation with Grad-CAM")
    p("Figure 4.6 shows Grad-CAM maps for the first test image of six classes (not selected for being easy). The first stage and the specialist look at different places: the triage network responds to compact, "
      "bright or contrasting structures, for example the top and bottom corners of the Offline-Module image and the central warm blob of the Cell image, while the specialist spreads over larger regions. For Cell, Diode, "
      "Cracking and Offline-Module the final prediction is correct and the two maps overlap on the same region (right column), which suggests that both stages use the same evidence. The two failures are instructive: the "
      "Hot-Spot image is routed to the Structural family and called Offline-Module, and the Soiling image is routed to the Electrical family and called Hot-Spot-Multi. In both cases the specialist's map covers a large "
      "warm area, which is typical of the class it wrongly chose. This agrees with the error analysis: the family decision is the weak link.")
    F(FIG / "cv_gradcam.png", 13.0, "Figure 4.6: Grad-CAM of Stage 1 and of the selected specialist for six test images (red = high influence).")
    sec("4.6 Comparison with Published Work")
    p("Published results differ in task, protocol and features, so they are placed next to ours without ranking. The dataset paper itself reports no accuracies [1].")
    d = lambda x, y: f"{(x - y) * 100:+.1f} points"
    A(("table", "Table 4.6: Direct comparison of published results with the results of this project",
       [["Published result", "Task and protocol", "Reported", "Our comparable result", "Difference and verdict"],
        ["Alves et al. [3]", "Binary healthy vs defect, cross-validation", "92.5%", f"{pct(binary_acc(R['pretrained']))} (Stage 1)", f"{d(binary_acc(R['pretrained']), 0.925)}: ours higher; one hold-out run"],
        ["Corley et al. [4], Random Forest", "12 classes, split not stated", "84.0%", f"{pct(pr['accuracy'])} (cascade); {pct(rf['accuracy'])} (our RF)", f"{d(pr['accuracy'], 0.84)}: about equal"],
        ["Duranay [2]", "12 classes, 10-fold cross-validation", "93.93%", f"{pct(pr['accuracy'])}", f"{d(pr['accuracy'], 0.9393)}: ours lower; protocols differ"],
        ["Duranay's method on our split", "12 classes, fixed hold-out", f"{pct(ps['accuracy'])} (our run)", f"{pct(pr['accuracy'])}", f"{d(pr['accuracy'], ps['accuracy'])}: ours higher on identical data"],
        ["Flat CNN, same backbone", "12 classes, fixed hold-out", f"{pct(prf['accuracy'])} (our run)", f"{pct(pr['accuracy'])}", f"{d(pr['accuracy'], prf['accuracy'])}: tied"]],
       [3.2, 3.4, 2.4, 3.2, 3.8], (), 8.5))
    p("Duranay [2] uses 10-fold cross-validation, an ImageNet-pretrained network and a supervised feature selection applied before the cross-validation, so, as described, test images can influence these steps. "
      "About 7% of our test images have a near-identical training image, which would inflate cross-validated scores. When the same kind of pipeline is run on our held-out split it scores only "
      f"{pct(ps['accuracy'])}, below our cascade, which suggests that the gap to 93.93% is largely due to the protocol; this is a suggestion, not a proof.")
    sec("4.7 Discussion")
    p("Four observations follow. First, domain knowledge matters: features designed from the physics of the images already beat raw pixels by about nine points, and knowing that brightness is temperature "
      "prevented a harmful augmentation. Second, learned features are better still, and pretrained features are better than learned-from-scratch ones even on 24 x 40 pixel thermal images. Third, the cascade did "
      "not show a clear advantage over a flat network with the same backbone, and it has five times the parameters. Fourth, Grad-CAM and the confusion analysis agree that the weak point is the family decision. "
      "The results are single runs with one seed, so differences below about 0.5 points are within noise.")

    # =============================================================== CHAPTER 5
    A(("chapter", 5, "CONCLUSION AND FUTURE ENHANCEMENTS"))
    sec("5.1 Conclusion")
    p(f"This project compared classical and deep approaches to classifying 12 classes of PV modules from tiny infrared images, and built and analysed a cascade of five CNNs. Hand-crafted features with an SVM reach {pct(HC['accuracy'])}, "
      f"the from-scratch cascade {pct(dg['accuracy'])}, and the cascade with ImageNet-pretrained EfficientNet-B0 {pct(pr['accuracy'])}, with {pct(binary_acc(R['pretrained']))} on the healthy-versus-defect decision.")
    bl("1.", "In thermal images brightness is temperature; photometric augmentation destroyed the evidence and cost about ten points.")
    bl("2.", "Transfer learning works on tiny thermal images: +3.4 points for the cascade and +5.8 for the flat network, with faster, smoother training.")
    bl("3.", "A cascade is not automatically better than a flat network. With the same backbone the two tie, and the family decision causes more than half of the errors.")
    bl("4.", "Visual analysis (filters, feature maps, Grad-CAM) and error attribution explain where and why the models fail.")
    sec("5.2 Limitations")
    bl("1.", "Single runs with one seed; differences below about 0.5 points are noise. A 10-fold cross-validation was prepared but did not complete.")
    bl("2.", "The rare classes (Soiling, Hot-Spot, Diode-Multi) remain weak, which keeps macro-F1 below accuracy.")
    bl("3.", "The benchmark has no module identifiers, so near-identical crops can appear on both sides of a split.")
    bl("4.", "The temperature-to-grey mapping of the dataset is undocumented, so brightness can only be treated as a monotonic proxy.")
    bl("5.", "Efficiency was not measured reliably; the early exit gave no demonstrated advantage. The 93% ceiling and 95% target of the proposal were not reached under the strict protocol.")
    sec("5.3 Future Enhancements")
    p("The following directions may be investigated next.")
    bl("1.", "Train the family network jointly with the specialists, or on the cascade's own routing errors, and test self-attention for the family decision.")
    bl("2.", "Generate or augment images of the rare classes with class-specific geometric transforms or a generative model.")
    bl("3.", "Run several seeds and a leak-free 10-fold cross-validation, and deduplicate near-identical crops.")
    bl("4.", "Distil the pretrained cascade into the small networks and measure latency on drone-class hardware.")
    bl("5.", "Use temperature-calibrated radiometric images, when available, instead of 8-bit grayscale.")
    bl("6.", "Add visible images or neighbouring-panel context, which the crop-level dataset cannot provide.")
    return B
