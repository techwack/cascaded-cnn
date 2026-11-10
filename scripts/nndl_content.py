"""Content of the NNDL case-study report (format: NNDL_Case_Study_Report_Format.docx). APA in-text citations."""
import srm_content as sc
import srm_content2 as c2

ROOT, FIG, R, pct, binary_acc, CLASSES = sc.ROOT, sc.FIG, sc.R, sc.pct, sc.binary_acc, sc.CLASSES
pr, prf, dg, i3, imp, fl, rf, ps = sc.pr, sc.prf, sc.dg, sc.i3, sc.imp, sc.fl, sc.rf, sc.ps
TITLE = sc.TITLE
S = pr["stages"]


def s_all_param(n):
    return f"{c2.STATS['improved3'][n]['params']:,}"


def p_all_param(n):
    return f"{c2.STATS['pretrained'][n]['params']:,}"

ABSTRACT = (
    "This case study applies the main ideas of the Neural Networks and Deep Learning course, convolutional neural networks, transfer learning, attention and generative models, to a practical problem: "
    "classifying defects of solar photovoltaic modules from 24 x 40 pixel infrared images (12 classes, 20,000 images). A flat classifier struggles with the class imbalance and look-alike faults, so a cascade of five small "
    "CNNs is built: a triage network that separates healthy from defective modules, a family network and one specialist per family. The report summarises the theory of the layers, the optimisation and regularisation used, "
    "transfer learning with an ImageNet-pretrained EfficientNet-B0 (whose blocks contain squeeze-and-excitation channel attention) and the role GANs could play for rare classes, and maps each concept to the implemented system. "
    f"On a fixed held-out split the from-scratch cascade reaches {pct(dg['accuracy'])} accuracy, and fine-tuning the pretrained backbone reaches {pct(pr['accuracy'])} accuracy, macro-F1 {pr['macro_f1']:.2f} and "
    f"{pct(binary_acc(R['pretrained']))} on the healthy-versus-defect decision. Removing brightness augmentation, which destroys the temperature signal, was the largest single gain. A flat network with the same backbone is "
    "essentially tied with the cascade, and the family decision is the main source of errors. Improvements and optimisation strategies are proposed for the remaining errors."
)

REFERENCES = [
    "Alves, R. H. F., de Deus Junior, G. A., Marra, E. G., & Lemos, R. P. (2021). Automatic fault classification in photovoltaic modules using convolutional neural networks. Renewable Energy, 179, 502-516.",
    "Corley, et al. (2025). Aerial infrared health monitoring of solar photovoltaic farms at scale. CVPR Workshops (PBVS). arXiv:2503.02128.",
    "Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., & Fei-Fei, L. (2009). ImageNet: A large-scale hierarchical image database. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).",
    "Dosovitskiy, A., et al. (2021). An image is worth 16x16 words: Transformers for image recognition at scale. International Conference on Learning Representations (ICLR).",
    "Duranay, Z. B. (2023). Fault detection in solar energy systems: A deep learning approach. Electronics, 12(21), 4397. https://doi.org/10.3390/electronics12214397",
    "Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep learning. MIT Press.",
    "Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., & Bengio, Y. (2014). Generative adversarial nets. Advances in Neural Information Processing Systems, 27.",
    "He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification. Proceedings of the IEEE International Conference on Computer Vision (ICCV).",
    "He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 770-778.",
    "Hu, J., Shen, L., & Sun, G. (2018). Squeeze-and-excitation networks. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).",
    "Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. Proceedings of the International Conference on Machine Learning (ICML).",
    "Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. International Conference on Learning Representations (ICLR).",
    "Millendorf, M., Obropta, E., & Vadhavkar, N. (2020). Infrared solar module dataset for anomaly detection. ICLR 2020 Workshop on AI for Earth Sciences.",
    "Paszke, A., et al. (2019). PyTorch: An imperative style, high-performance deep learning library. Advances in Neural Information Processing Systems, 32.",
    "Pathak, S. P., & Patil, S. A. (2025). Enhanced fault identification in solar panels through binary cascaded convolutional classifiers with thermal-visual augmentation. International Journal of Image, Graphics and Signal Processing, 17(4).",
    "Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A simple way to prevent neural networks from overfitting. Journal of Machine Learning Research, 15, 1929-1958.",
    "Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. Proceedings of the International Conference on Machine Learning (ICML). arXiv:1905.11946.",
    "Tella, H., Hussein, A., Rehman, S., et al. (2025). PV panel defect classification using deep learning ensemble methods. Case Studies in Thermal Engineering, 66.",
    "Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention is all you need. Advances in Neural Information Processing Systems, 30.",
    "Yosinski, J., Clune, J., Bengio, Y., & Lipson, H. (2014). How transferable are features in deep neural networks? Advances in Neural Information Processing Systems, 27.",
]


def blocks(charts, perclass_png, demo_png):
    B = []
    A = B.append
    sec = lambda t: A(("sec", t))
    sub = lambda t: A(("sub", t))
    p = lambda t: A(("p", t))
    eq = lambda t: A(("eq", t))
    bl = lambda lead, t: A(("b", lead, t))
    sp = lambda: A(("fig_gap",))

    # =================================================================== 1
    A(("chapter", 1, "INTRODUCTION"))
    sec("1.1 Background")
    p("Photovoltaic (PV) power plants contain thousands of modules, and defects such as hot spots, cracks, soiling, shadowing and bypass-diode failures lower their output and can cause fires. Manual thermal inspection "
      "does not scale, so drones with infrared (IR) cameras are used to image the modules, and a classifier has to label each module automatically. In an IR image brightness encodes temperature: a healthy module is "
      "heated almost uniformly, while a defect creates a localised or characteristic thermal pattern (Millendorf et al., 2020).")
    p("Deep learning is well suited to this recognition task, and the case touches the main topics of the course. Convolutional neural networks (CNNs) extract spatial features, transfer learning reuses features learned "
      "on ImageNet, attention mechanisms re-weight features inside modern backbones, and generative adversarial networks (GANs) offer a way to create synthetic examples of rare classes. This report studies how these "
      "ideas behave on one concrete dataset, the InfraredSolarModules benchmark.")
    sec("1.2 Problem Statement")
    p("Given a single-channel IR image of 24 x 40 pixels showing one PV module, classify it into one of 12 classes: No-Anomaly and eleven fault types. Three properties make the problem hard: half of the images are "
      "healthy while the rarest faults have fewer than 250 images; several faults differ only in how many cells or diodes are affected; and a drone needs a model that is small and fast. A single network that chooses "
      "among all 12 classes (a flat classifier) therefore tends to plateau and to be biased towards the healthy class.")
    sec("1.3 Objectives")
    bl("O1.", "To explain the neural-network concepts relevant to the case: CNN layers, optimisation and regularisation, transfer learning, attention and GANs.")
    bl("O2.", "To design and train a cascade of five small CNNs (healthy versus defect, fault family, fault subtype) and a pretrained variant, and to compare it with a flat classifier.")
    bl("O3.", "To map every concept to the implemented system, and to measure the effect of the main design choices with controlled experiments.")
    bl("O4.", "To analyse where the model fails and to propose improvement and optimisation strategies.")
    p("The report follows this order: Chapter 2 gives the theory, Chapter 3 describes and analyses the case, Chapter 4 proposes improvements, and Chapter 5 concludes.")

    # =================================================================== 2
    A(("chapter", 2, "THEORETICAL FOUNDATION: CNN, TRANSFER LEARNING, ATTENTION & GANs"))
    sec("2.1 Convolutional Neural Networks")
    p("A CNN processes an image with layers that share weights across positions, which exploits two properties of images: nearby pixels are correlated and the same pattern can occur anywhere (Goodfellow et al., 2016). "
      "The stage networks of the case have the structure shown in Figure 2.1.")
    A(("fig", FIG / "stage1_cnn1.png", 15.0, "Fig. 2.1. Architecture of a convolutional neural network: the Stage 1 triage CNN of the case study."))
    sub("2.1.1 Convolution, Normalisation, Activation and Pooling")
    p("A convolutional layer with C_in input and C_out output channels and a k x k kernel computes, for every output channel c and position (i, j),")
    eq("y[c, i, j] = b[c] + sum over (m, u, v) of  W[c, m, u, v] x x[m, i + u, j + v]")
    p("with k^2 x C_in x C_out weights. Batch normalisation (Ioffe & Szegedy, 2015) standardises each channel over the mini-batch and rescales it with learned parameters, y = gamma x (x - mean) / sqrt(var + eps) + beta, which "
      "stabilises training. The ReLU activation f(x) = max(0, x) adds non-linearity without saturating, and 2 x 2 max pooling halves the spatial size and gives tolerance to small shifts. Global average pooling "
      "replaces flattening by the mean of each feature map, z_c = mean of x_c(i, j), which keeps the classifier small, and a fully connected layer with softmax gives class probabilities p_k = exp(s_k) / sum_j exp(s_j).")
    sub("2.1.2 Receptive Field and Feature Hierarchy")
    p("Stacked small convolutions and pooling enlarge the region of the input that a unit can see. Each layer adds (k - 1) x jump to the receptive field, and a stride s multiplies the jump by s. For the three blocks of "
      "the triage network the receptive field grows from 3 to 36 pixels (Figure 2.2), so that early layers respond to edges and intensity gradients of a heated region, middle layers to blobs and their shape, and the last "
      "layer to the arrangement of regions across the module.")
    A(("fig", FIG / "nndl_receptive_field.png", 14.0, "Fig. 2.2. Growth of the receptive field through the layers of the triage CNN."))
    sub("2.1.3 Training: Loss, Backpropagation and Adam")
    p("The networks minimise the class-weighted cross-entropy L = -(1/B) sum_i w[y_i] log p[i, y_i], with weights w_r = N / (K x n_r) that give rare classes a larger gradient. Backpropagation applies the chain rule "
      "layer by layer; for softmax with cross-entropy the gradient with respect to the scores is dL/ds_k = w_y (p_k - 1[k = y]). Weights are updated with Adam (Kingma & Ba, 2015), which keeps running averages of the "
      "gradient and of its square:")
    eq("m_t = 0.9 m_(t-1) + 0.1 g_t,   v_t = 0.999 v_(t-1) + 0.001 g_t^2,   theta <- theta - alpha x m_hat / (sqrt(v_hat) + eps)")
    p("so that each weight gets its own step size. The learning rate is halved when the validation score stalls.")
    sub("2.1.4 Regularisation and Initialisation")
    p("Dropout zeroes each unit with probability p during training and rescales the others (Srivastava et al., 2014); weight decay adds an L2 penalty; early stopping keeps the checkpoint with the best validation "
      "score; flips enlarge the training set. Weights of the from-scratch convolutions use the Kaiming scheme, with variance 2 / fan_out, which keeps the activation variance stable through ReLU layers (He et al., 2015). "
      "Together these control the bias-variance trade-off: a small network is biased, a large one fits noise.")
    sec("2.2 Transfer Learning")
    p("Transfer learning initialises a network with weights learned on a large source task. Early convolutional layers learn generic features (edges, gradients, textures) that transfer across tasks, while later layers are "
      "more specific to the source classes (Yosinski et al., 2014). Fine-tuning keeps the feature extractor, replaces the classifier head and trains all layers with a small learning rate (Figure 2.3). "
      "This reduces the number of target samples needed and gives the optimiser a good starting region.")
    A(("fig", FIG / "nndl_transfer_learning.png", 15.0, "Fig. 2.3. Transfer learning from ImageNet to the thermal defect task."))
    p("The pretrained backbone used here is EfficientNet-B0 (Tan & Le, 2019), built from mobile inverted-bottleneck (MBConv) blocks with depthwise convolutions and residual connections (cf. He et al., 2016), and scaled "
      "by a compound coefficient of depth, width and resolution. It has about 4 million parameters. The source images (ImageNet; Deng et al., 2009) are natural RGB photographs, so a domain shift exists: the grayscale "
      "thermal channel is repeated three times and the image resized to 160 x 96 to match the scale the filters were trained for.")
    sec("2.3 Attention Mechanisms")
    p("Attention lets a network weight parts of its representation by their relevance. Channel attention, as in squeeze-and-excitation (SE) blocks (Hu et al., 2018), pools each channel to a number, passes the vector "
      "through two fully connected layers and a sigmoid to obtain a weight per channel, and multiplies the feature map by these weights (Figure 2.4):")
    eq("s = sigmoid( W2 x ReLU( W1 x z ) ),    x' = s (channel-wise) x")
    A(("fig", FIG / "nndl_se_attention.png", 15.0, "Fig. 2.4. Squeeze-and-excitation channel attention, as used inside EfficientNet-B0."))
    p("Self-attention, the core of Transformers (Vaswani et al., 2017) and Vision Transformers (Dosovitskiy et al., 2021), computes softmax(Q K^T / sqrt(d)) V, so that every position can attend to every other. "
      "In this case study attention is used implicitly: every MBConv block of the pretrained backbone contains an SE module. Self-attention is a candidate for improving the family decision (Chapter 4).")
    sec("2.4 Generative Adversarial Networks")
    p("A GAN (Goodfellow et al., 2014) trains a generator G, which maps noise z to an image, against a discriminator D, which decides whether an image is real or generated, in a minimax game:")
    eq("min over G, max over D of  E[log D(x)] + E[log(1 - D(G(z)))]")
    A(("fig", FIG / "nndl_gan.png", 15.0, "Fig. 2.5. Generative adversarial network: generator and discriminator."))
    p("When training converges the generator produces images that follow the data distribution, which makes GANs a tool for synthetic augmentation of rare classes. GANs are not used in the implemented system, because "
      "the rare classes have only 123 to 174 training images, too few to train a stable GAN; they are proposed as an improvement in Chapter 4.")
    sec("2.5 Summary of the Concepts")
    A(("table", "Table 2.1: The four concept families and their role in the case study",
       [["Concept", "Key idea", "Role in this case study"],
        ["CNN", "Weight-sharing convolutions, pooling, hierarchical features", "All five stage networks; from-scratch variant"],
        ["Transfer learning", "Reuse features from a source task and fine-tune", "Pretrained EfficientNet-B0 at every stage (+3.4 points)"],
        ["Attention", "Re-weight features by relevance (channel or self-attention)", "SE channel attention inside EfficientNet-B0; self-attention proposed for Stage 2"],
        ["GAN", "Generator and discriminator in a minimax game", "Not used; proposed for synthetic rare-class images"]],
       [3.0, 6.4, 6.4], (), 10))

    # =================================================================== 3
    A(("chapter", 3, "CASE ANALYSIS & IMPLEMENTATION"))
    sec("3.1 Case Description and Dataset")
    p("The InfraredSolarModules dataset (Millendorf et al., 2020) contains 20,000 grayscale IR images of 24 pixels wide and 40 tall, captured on commercial drone flights, with 12 classes (Table 3.1). Half of the images "
      "are healthy, and the four rarest classes have 175 to 249 images (Figure 3.1). Figure 3.2 shows one test image per class. The images are split once, stratified by class with a fixed seed, into 70% training, 15% "
      "validation and 15% test images, and the test images are never used for training, early stopping, normalisation or tuning.")
    A(("table", "Table 3.1: Dataset description and stratified split",
       [["Class", "Family", "Total", "Train", "Validation", "Test"],
        ["No-Anomaly", "Healthy", 10000, 7000, 1500, 1500], ["Cell", "Electrical", 1877, 1314, 282, 281], ["Cell-Multi", "Electrical", 1288, 902, 193, 193],
        ["Diode", "Electrical", 1499, 1049, 225, 225], ["Diode-Multi", "Electrical", 175, 123, 26, 26], ["Hot-Spot", "Electrical", 249, 174, 38, 37],
        ["Hot-Spot-Multi", "Electrical", 246, 172, 37, 37], ["Soiling", "Environmental", 204, 143, 30, 31], ["Shadowing", "Environmental", 1056, 739, 158, 159],
        ["Vegetation", "Environmental", 1639, 1147, 246, 246], ["Cracking", "Structural", 940, 658, 141, 141], ["Offline-Module", "Structural", 827, 579, 124, 124],
        ["Total", "", 20000, 14000, 3000, 3000]], [3.4, 3.0, 2.2, 2.2, 2.6, 2.2], (13,), 10))
    A(("fig", FIG / "class_distribution.png", 13.5, "Fig. 3.1. Class distribution of the dataset (logarithmic axis)."))
    A(("fig", FIG / "sample_images.png", 14.5, "Fig. 3.2. One test image per class (inferno colour map)."))
    sec("3.2 Model Architecture and Training")
    sub("3.2.1 The Cascade")
    p("Instead of one 12-way classifier the system uses five small networks in three stages (Figure 3.3). CNN-1 decides healthy or defective and lets healthy modules exit early. CNN-2 assigns a defect to one of three "
      "families (Electrical, Environmental, Structural). One specialist per family (CNN-3a, 3b, 3c with 6, 3 and 2 classes) names the exact fault. Each network is trained independently on the data matching its task, and the "
      "cascade is assembled at test time, so a stage's error is measured as it would occur in use.")
    A(("fig", FIG / "overall_cascade.png", 11.5, "Fig. 3.3. The implemented three-stage cascade."))
    sub("3.2.2 Network Structure")
    p("In the from-scratch variant every network has three blocks of two 3 x 3 convolutions (BatchNorm and ReLU after each) followed by 2 x 2 max pooling, then global average pooling, dropout and a fully connected layer "
      "(Table 3.2). The input is upsampled to 80 x 48 so that small defects survive three pooling steps. In the pretrained variant every network is an ImageNet-pretrained EfficientNet-B0 with a new head.")
    A(("table", "Table 3.2: Architecture summary of the stage networks",
       [["Network", "Task", "Channels, dropout", "Output", "Parameters (scratch / pretrained)"],
        ["CNN-1", "Healthy vs defect", "64-128-128, 0.2", "2", f"{s_all_param('triage')} / {p_all_param('triage')}"],
        ["CNN-2", "Family", "128-256-128, 0.4", "3", f"{s_all_param('family')} / {p_all_param('family')}"],
        ["CNN-3a", "Electrical subtype", "128-256-128, 0.4", "6", f"{s_all_param('electrical')} / {p_all_param('electrical')}"],
        ["CNN-3b", "Environmental subtype", "128-256-128, 0.4", "3", f"{s_all_param('environmental')} / {p_all_param('environmental')}"],
        ["CNN-3c", "Structural subtype", "128-256-128, 0.4", "2", f"{s_all_param('structural')} / {p_all_param('structural')}"]],
       [2.0, 3.6, 3.8, 1.6, 4.8], (), 9.5))
    sub("3.2.3 Training")
    p("Training uses Adam (weight decay 1e-4, batch size 64), a class-weighted cross-entropy loss, random flips as the only augmentation, and early stopping on validation macro-F1. Table 3.3 lists the settings.")
    A(("table", "Table 3.3: Training configuration",
       [["Setting", "From scratch", "Pretrained"],
        ["Input size", "80 x 48 (2x upsampling)", "160 x 96, channel repeated 3 times"], ["Initialisation", "Kaiming normal (fan-out)", "ImageNet weights"],
        ["Learning rate", "5e-4", "3e-4"], ["Early stopping patience (maximum epochs)", "25 (150)", "6 (25)"],
        ["Precision", "32-bit", "Mixed precision"], ["Loss", "Class-weighted cross-entropy", "Class-weighted cross-entropy"],
        ["Hardware", "NVIDIA RTX 5060 Laptop GPU", "NVIDIA RTX 5060 Laptop GPU"]], [5.4, 5.2, 5.2], (), 10))
    sec("3.3 Mapping Concepts to the Case")
    p("Table 3.4 maps each concept of Chapter 2 to the place where it appears in the implementation and to what was observed.")
    A(("table", "Table 3.4: Mapping of course concepts to the case study",
       [["Concept", "Where it appears", "Observation"],
        ["Convolution, BatchNorm, ReLU, pooling", "Every block of the stage networks", "Three blocks give a receptive field of 36 pixels (Fig. 2.2)"],
        ["Class-weighted cross-entropy", "Loss of all networks", "Rare classes are not ignored, but stay the weakest"],
        ["Adam and learning-rate schedule", "Optimiser of all networks", "Pretrained networks converge in 10 to 20 epochs"],
        ["Dropout, weight decay, early stopping", "All networks", "From-scratch family network has noisy validation curves (high variance)"],
        ["Data augmentation", "Random flips", "Brightness and noise jitter lowered accuracy by about 10 points"],
        ["Transfer learning", "EfficientNet-B0 at every stage", f"{pct(dg['accuracy'])} to {pct(pr['accuracy'])} accuracy"],
        ["Attention", "SE modules inside EfficientNet-B0", "Used implicitly; not isolated in an ablation"],
        ["GAN", "Not implemented", "Proposed for rare classes (Chapter 4)"]],
       [4.4, 4.6, 6.8], (), 9.5))
    sec("3.4 Results and Discussion")
    p(f"All numbers are measured on the 3,000 test images. The best result is the pretrained cascade with {pct(pr['accuracy'])} accuracy, macro-F1 {pr['macro_f1']:.3f}, Cohen's kappa {pr['kappa']:.3f} and "
      f"{pct(binary_acc(R['pretrained']))} on the healthy-versus-defect decision (Table 3.5, Figure 3.4). Figure 3.5 shows the training behaviour of the five networks.")
    rows, best = sc._t6_rows()
    A(("table", "Table 3.5: Test results by configuration", rows, [5.6, 2.2, 2.2, 2.2, 1.8, 2.4], (best,), 9.5))
    A(("fig", charts["prog"], 14.0, "Fig. 3.4. Test accuracy and macro-F1 by configuration."))
    A(("fig", c2.FIG / "training_curves.png", 15.0, "Fig. 3.5. Validation macro-F1 and training loss of the five stage networks."))
    bl("Augmentation.", "Removing brightness, contrast and noise jitter raised accuracy from 61.7% to 71.1%. In thermal images brightness is temperature, so the jitter violates the assumption that augmentations preserve the label.")
    bl("Capacity and input size.", f"Upsampling and two convolutions per block gave {pct(imp['accuracy'])}, and larger front stages {pct(i3['accuracy'])}: a reduction of bias.")
    bl("Transfer learning.", f"Pretraining raised the cascade to {pct(pr['accuracy'])} and the flat network from {pct(fl['accuracy'])} to {pct(prf['accuracy'])}, so that the two designs end up tied. The pretrained networks converge faster and more smoothly.")
    bl("Cascade versus flat.", f"{pct(pr['accuracy'])} / {pr['macro_f1']:.3f} (cascade) against {pct(prf['accuracy'])} / {prf['macro_f1']:.3f} (flat) in accuracy and macro-F1, with five times the parameters for the cascade. The cascade did not show a clear advantage.")
    bl("Did not help.", "Soft routing, test-time augmentation, threshold tuning and balanced sampling.")
    p("Comparison with the literature is limited by different protocols. Duranay (2023) reports 93.93% accuracy for 12 classes with 10-fold cross-validation and an ImageNet-pretrained EfficientNet-B0 with a feature selection that "
      f"uses the labels; the same kind of pipeline run on our held-out split scored {pct(ps['accuracy'])}, below our cascade. For the binary decision, Alves et al. (2021) report 92.5% with cross-validation, against "
      f"{pct(binary_acc(R['pretrained']))} for our Stage 1. A random forest on our split scores {pct(rf['accuracy'])}, against 84.0% reported by Corley et al. (2025), whose protocol is not stated. We therefore claim parity "
      "for the binary task and do not claim to beat the published 12-class figures.")
    sec("3.5 Error Analysis")
    sub("3.5.1 Errors by Stage")
    p("Every error of the cascade can be attributed to the first stage that went wrong (Table 3.6, Figure 3.6). Stage 2, the family decision, causes more than half of all errors, and an error there cannot be corrected by Stage 3.")
    rows = [["Configuration", "Stage 1 missed defect", "Stage 1 false alarm", "Stage 2 wrong family", "Stage 3 wrong subtype", "Total"]]
    for k, lab in [("improved3", "From scratch"), ("pretrained", "Pretrained")]:
        s_ = R[k]["cascade"]["stages"]
        rows.append([lab, s_["stage1_missed_defects"], s_["stage1_false_alarms"], s_["stage2_wrong_family"], s_["stage3_wrong_subtype"], s_["total_errors"]])
    A(("table", "Table 3.6: Test errors per stage (out of 3,000 images)", rows, [3.2, 2.6, 2.4, 2.7, 2.8, 1.5], (2,), 10))
    A(("fig", charts["stages"], 13.5, "Fig. 3.6. Where the cascade makes its errors."))
    sub("3.5.2 Per-Class Behaviour and Confusion")
    p("Rare classes are the weakest (Figure 3.7). The confusion matrix of the pretrained cascade (Figure 3.8) shows that errors concentrate between classes of one physical phenomenon, a single versus several affected cells or "
      "diodes, and between faint hot spots and the healthy class.")
    A(("fig", perclass_png, 13.5, "Fig. 3.7. Per-class F1 on the test set."))
    A(("fig", FIG / "confusion_cascade.png", 12.0, "Fig. 3.8. Confusion matrix of the pretrained cascade (row-normalised, counts inside cells)."))
    tc = c2.top_confusions(pr)
    A(("table", "Table 3.7: Most frequent confusions of the pretrained cascade",
       [["True class", "Predicted as", "Images", "Share of the true class"]] + [[a, b, n, f"{100 * n / tot:.0f}%"] for n, a, b, tot in tc], [4.6, 4.6, 2.2, 4.0], (), 10))
    sub("3.5.3 Interpretation")
    p("The errors are explained by the concepts of Chapter 2. High variance of the small family and electrical networks on few training images is reduced by pretraining but not removed. Error propagation between "
      "independently trained stages means that a misrouted image meets a specialist that has never seen its class. And with only 123 to 174 training images, rare classes do not constrain the weights enough, whatever the loss weights.")

    # =================================================================== 4
    A(("chapter", 4, "PROPOSED IMPROVEMENTS / OPTIMIZATION STRATEGIES"))
    p("The analysis points to three weaknesses: the family decision, the rare classes and the cost of the cascade. The following strategies are proposed. They are not implemented in the case study, and the expected "
      "effects are hypotheses to be tested.")
    sec("4.1 Better Family Decision")
    p("Train CNN-2 jointly with the specialists, so that the stages are optimised for the final label, or train the specialists on the routing errors of CNN-2. Self-attention over the pooled feature map, in the style of a small "
      "Vision Transformer (Dosovitskiy et al., 2021), could help to relate the positions of heated regions across the module, which matters for the family.")
    sec("4.2 Rare Classes: GAN and Class-Specific Augmentation")
    p("Train a conditional GAN or a lightweight variational model on the rare classes (Soiling, Hot-Spot, Diode-Multi) to create synthetic images, or use class-specific geometric augmentation and mix-up. The generator would be "
      "evaluated on a held-out validation set, and its images used only for training. With 123 to 174 real images, a pretrained or small generator and strong regularisation of the discriminator would be needed.")
    sec("4.3 Loss and Optimisation")
    p("Replace the class weights by focal loss, which down-weights easy examples, or by label-distribution-aware margins. Tune the learning rate, weight decay and dropout with a search on the validation split, and use "
      "a cosine learning-rate schedule with warm-up for the fine-tuned networks.")
    sec("4.4 Model Compression and Efficiency")
    p("The pretrained cascade has five EfficientNet-B0 networks (20 million parameters). Knowledge distillation into the small from-scratch networks, quantisation to 8 bits and a shared backbone with five heads would reduce "
      "memory and latency so that the early exit pays off on a drone.")
    sec("4.5 Evaluation Strategy")
    p("Run several seeds and a leak-free 10-fold cross-validation, deduplicate near-identical crops before splitting, and report confidence intervals. Ensembles of the cascade and the flat network could combine their "
      "complementary errors.")
    A(("table", "Table 4.1: Proposed improvements and optimisation strategies",
       [["Strategy", "Problem addressed", "Expected effect", "Cost"],
        ["Joint or error-aware training of the stages", "Stage 2 errors (272 of 505)", "Fewer wrong-family errors", "Longer training, more tuning"],
        ["Self-attention for the family decision", "Look-alike families", "Better use of the global layout", "More parameters"],
        ["GAN or class-specific augmentation", "Rare classes", "Higher macro-F1", "Unstable GAN training on 123 to 174 images"],
        ["Focal loss, tuned schedule", "Easy-example dominance, tuning noise", "Small accuracy gain", "Low"],
        ["Distillation, quantisation, shared backbone", "Cost of five networks", "Lower latency and memory", "Possible small accuracy loss"],
        ["Cross-validation, deduplication, seeds", "Comparability with the literature", "Reliable estimates", "10x training time"]],
       [4.6, 3.8, 3.6, 3.8], (), 9.5))

    # =================================================================== 5
    A(("chapter", 5, "CONCLUSION AND FUTURE WORK"))
    sec("5.1 Conclusion")
    p(f"The case study applied CNNs, transfer learning and attention-based backbones to classify solar module defects from tiny infrared images. A cascade of five small CNNs reached {pct(dg['accuracy'])} accuracy from scratch "
      f"and {pct(pr['accuracy'])} with an ImageNet-pretrained EfficientNet-B0, with {pct(binary_acc(R['pretrained']))} on the healthy-versus-defect decision. Three lessons follow from the experiments.")
    bl("1.", "Augmentation must preserve the label-relevant signal: brightness is temperature in thermal images, and jittering it cost about ten points.")
    bl("2.", "Transfer learning helps even on 24 x 40 pixel thermal images: it raised accuracy by 3.4 points and made training fast and smooth.")
    bl("3.", "A cascade is not automatically better than a flat network. With the same backbone the two tie, the family decision causes more than half of the errors, and the cascade has five times the parameters.")
    p("Limitations: single runs with one seed, a cross-validation that was prepared but not completed, weak rare classes, near-duplicate images in the dataset, and no measured efficiency gain. The 93% ceiling and the "
      "95% target set at the start were not reached under the strict held-out protocol.")
    sec("5.2 Future Work")
    p("Implement and test the strategies of Chapter 4, in particular joint training of the stages, class-specific augmentation or a GAN for the rare classes, and distillation for edge deployment. Evaluate with several seeds and a "
      "leak-free cross-validation, measure latency on drone hardware, and investigate additional information such as RGB images or neighbouring panels.")
    return B


APPENDIX = "Appendices"
