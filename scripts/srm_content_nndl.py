"""Extra technical sections for the NNDL (21CSC406J) version of the report."""
import srm_content2 as c2

R, pr, prf, dg, i3, pct = c2.R, c2.pr, c2.prf, c2.dg, c2.i3, c2.pct


def tech_sections():
    B = []
    A = B.append
    sec = lambda t: A(("sec", t))
    sub = lambda t: A(("sub", t))
    p = lambda t: A(("p", t))
    eq = lambda t: A(("eq", t))
    bl = lambda lead, t: A(("b", lead, t))

    sec("5.12 Backpropagation and Optimisation")
    p("Training adjusts every weight of a stage network by gradient descent on the loss of Section 5.3.5. The gradient is computed layer by layer with the chain rule, which is called backpropagation. This section "
      "derives the gradients of the layers used in the stage networks.")
    sub("5.12.1 Gradient of Softmax and Cross-Entropy")
    p("For one sample with class scores s_k, probabilities p_k = exp(s_k) / sum_j exp(s_j), true label y and class weight w_y, the loss is L = -w_y log p_y. Its gradient with respect to the score of class k simplifies to")
    eq("dL / ds_k = w_y x ( p_k - 1[k = y] )")
    p("where 1[k = y] is one for the true class and zero otherwise. The gradient is therefore the difference between the predicted probability and the target, scaled by the class weight, which is why rare classes "
      "(large w_y) produce larger updates when they are misclassified.")
    sub("5.12.2 Backpropagation Through the Layers")
    p("Let delta denote the gradient of the loss with respect to the output of a layer. The gradients of the layers of a stage network are as follows.")
    bl("Fully connected layer.", "With input z and weights W, the gradients are dL/dW = delta z^T and dL/dz = W^T delta.")
    bl("Global average pooling.", "Each of the H x W positions of a feature map receives the same share of the gradient: dL/dx(i, j) = delta / (H W).")
    bl("Dropout.", "The gradient is passed only through the units that were kept and is scaled by 1 / (1 - p), the same factor as in the forward pass.")
    bl("Max pooling.", "The gradient of a window is routed entirely to the position that held the maximum; the other three positions receive zero.")
    bl("ReLU.", "The gradient passes where the input was positive and is zero elsewhere: dL/dx = delta x 1[x > 0].")
    bl("Convolution.", "The gradient of a kernel is the correlation of the input with the output gradient, and the gradient passed to the input is the full convolution of the output gradient with the flipped kernel:")
    eq("dL / dW[c, m, u, v] = sum over (i, j) of  delta[c, i, j] x x[m, i + u, j + v]")
    bl("Batch normalisation.", "The gradient with respect to gamma and beta is the sum over the batch of delta x x_hat and of delta. The gradient with respect to the input also accounts for the dependence of the batch mean and variance on every sample, "
                               "which couples the samples of a mini-batch and is part of why BatchNorm stabilises training.")
    sub("5.12.3 The Adam Optimiser")
    p("All networks are trained with Adam, which keeps running estimates of the first and second moments of the gradient g_t. With the default decay rates beta1 = 0.9 and beta2 = 0.999, the update at step t is")
    eq("m_t = beta1 m_(t-1) + (1 - beta1) g_t,    v_t = beta2 v_(t-1) + (1 - beta2) g_t^2")
    eq("m_hat = m_t / (1 - beta1^t),   v_hat = v_t / (1 - beta2^t),   theta <- theta - alpha x m_hat / (sqrt(v_hat) + eps)")
    p("Dividing by the root of the second moment gives every weight its own effective step size, which suits networks whose layers have very different gradient scales, and the momentum term smooths the noise of "
      "mini-batch gradients. Weight decay of 1e-4 is added to the gradient (g <- g + lambda x theta), which is L2 regularisation. The base learning rate alpha is 5e-4 for the from-scratch networks and 3e-4 for the fine-tuned "
      "pretrained ones, where a smaller step avoids destroying the pretrained filters.")
    sub("5.12.4 Learning-Rate Schedule and Early Stopping")
    p("The learning rate is halved when the validation macro-F1 has not improved for a number of epochs (8 for the from-scratch and 2 for the pretrained networks, one third of the early-stopping patience), "
      "and training stops after 25 (6) epochs without improvement. Reducing the step when progress stalls lets the optimiser settle into a narrower minimum, and stopping on a validation metric limits overfitting.")

    sec("5.13 Regularisation and Generalisation")
    p("A network with a few million parameters trained on 14,000 images can memorise the training set, so several regularisers are combined.")
    bl("Dropout.", "During training each unit is zeroed with probability p and the survivors are scaled by 1 / (1 - p). It prevents units from co-adapting and acts like training an ensemble of thinned networks. The rate is larger "
                   "(0.4) in the later stages, which have smaller training sets and more capacity.")
    bl("Weight decay.", "The L2 penalty lambda ||theta||^2 / 2 pulls weights towards zero, favouring smoother functions.")
    bl("Batch normalisation.", "The noise of the batch statistics has a mild regularising effect, and the normalisation allows a higher learning rate.")
    bl("Data augmentation.", "Flips enlarge the effective training set without changing labels. Augmentation that changes the label-relevant signal, such as brightness jitter on thermal images, is harmful (Section 6.3.1).")
    bl("Early stopping.", "The validation macro-F1 is monitored and the best checkpoint is kept, which bounds the effective training time.")
    p("The trade-off behind these choices is the bias-variance trade-off: a small network is biased because it cannot represent the decision boundary, a large one has high variance because it fits noise. The results "
      "of Chapter 6 are consistent with this view. Doubling the input size and capacity lowered the bias and raised accuracy (Section 6.3.2), while the noisy validation curves of the from-scratch family and electrical "
      "networks (Figure 10) are a sign of high variance on small training sets, which pretraining reduced.")

    sec("5.14 Initialisation, Activations and Gradient Flow")
    p("Deep networks are hard to train if the signal shrinks or grows from layer to layer. The weights of the from-scratch convolutions are initialised with the Kaiming (He) normal scheme in fan-out mode, which draws "
      "each weight from a zero-mean normal distribution with variance")
    eq("Var(W) = 2 / fan_out,    fan_out = k^2 x C_out")
    p("so that the variance of the activations is preserved through ReLU layers. The ReLU activation f(x) = max(0, x) has a gradient of one for positive inputs, and therefore does not saturate as sigmoid and tanh "
      "do, whose gradients are below 0.25 and 1 and shrink with every layer, causing vanishing gradients. BatchNorm further keeps the pre-activations in a well-scaled range. In the EfficientNet-B0 networks of the pretrained "
      "variant the residual connections of the MBConv blocks give the gradient a direct path to early layers, y = x + F(x), so that dy/dx = 1 + dF/dx.")

    sec("5.15 Receptive Field and Feature Hierarchy")
    p("The receptive field of a unit is the region of the input that can influence it. For a layer with kernel size k, its receptive field grows by (k - 1) times the current jump (the distance in input pixels between "
      "adjacent units), and a stride s multiplies the jump by s. Applying this to CNN-1 on the 80 x 48 input gives the values of Table 5.7.")
    A(("table", "Table 5.7: Receptive field of CNN-1 (80 x 48 input)",
       [["Layer", "Kernel / stride", "Jump (pixels)", "Receptive field (pixels)"],
        ["Block 1: Conv 3 x 3", "3 / 1", "1", "3"], ["Block 1: Conv 3 x 3", "3 / 1", "1", "5"], ["Block 1: MaxPool", "2 / 2", "2", "6"],
        ["Block 2: Conv 3 x 3", "3 / 1", "2", "10"], ["Block 2: Conv 3 x 3", "3 / 1", "2", "14"], ["Block 2: MaxPool", "2 / 2", "4", "16"],
        ["Block 3: Conv 3 x 3", "3 / 1", "4", "24"], ["Block 3: Conv 3 x 3", "3 / 1", "4", "32"], ["Block 3: MaxPool", "2 / 2", "8", "36"]],
       [4.6, 3.4, 3.2, 4.4], (), 10))
    p("Each unit of the last feature map sees a 36 x 36 patch of the upsampled image, which corresponds to 18 x 18 pixels of the original 40 x 24 image, about three quarters of its width. The hierarchy is therefore "
      "local-to-global: the first block responds to small intensity gradients and edges of a heated region, the second to blobs and their shape, and the third to the arrangement of regions across the module. Global "
      "average pooling then summarises the whole map, so that the final decision depends on where and how large the heat patterns are, which is exactly the information that separates the fault classes.")

    sec("5.16 Transfer Learning Theory")
    p("Transfer learning initialises a network with weights learned on a large source task and adapts them to the target task. For convolutional networks the early layers learn generic features (edges, gradients, "
      "blobs) that are useful for most images, and the later layers learn features specific to the source classes. Fine-tuning keeps the generic filters as a good starting point and adjusts all layers with a small learning rate.")
    p("The source task here is ImageNet classification of natural RGB photographs, and the target is the classification of tiny grayscale thermal images, so a domain shift exists. It is bridged by repeating the grayscale "
      "channel three times and by resizing the image to 160 x 96 so that the filters operate at a scale they were trained for. Pretraining reduces the number of samples needed to fit the target task, because the "
      "optimiser starts in a region of the weight space where useful features already exist. This explains the measured effect: the pretrained networks reached their best validation score within 10 to 20 epochs, "
      f"and raised the cascade from {pct(dg['accuracy'])} to {pct(pr['accuracy'])} accuracy and the flat network from {pct(R['improved']['flat']['accuracy'])} to {pct(prf['accuracy'])}. The risk of negative transfer, in which source features "
      "mislead the target task, did not materialise on this dataset.")
    p("EfficientNet-B0 [11] is built from MBConv blocks. A block expands the channels, applies a depthwise convolution (one k x k filter per channel, which needs k^2 C parameters instead of k^2 C^2), re-weights the channels "
      "with a squeeze-and-excitation module and projects back. The squeeze-and-excitation module computes channel weights from the globally pooled feature map:")
    eq("s = sigmoid( W2 x ReLU( W1 x z ) ),    x' = s (channel-wise) x")
    p("and the family of networks is scaled with a compound coefficient phi, depth d = alpha^phi, width w = beta^phi and resolution r = gamma^phi with alpha x beta^2 x gamma^2 approximately 2, so that the cost "
      "roughly doubles for each increment of phi. B0 is the baseline of this family, with about 4 million parameters.")
    return B


def discussion_nndl():
    B = []
    A = B.append
    A(("sub", "6.8.4 The Results Read Through Neural-Network Concepts"))
    A(("p", "Several findings of this chapter correspond directly to concepts of neural-network theory. (1) The ten-point loss caused by brightness jitter is a case of label-relevant signal being destroyed by augmentation, "
            "a violation of the assumption that augmentations are label-preserving. (2) The gain from a larger input and more convolutions is a reduction of bias: more layers and a larger receptive field let the network represent finer "
            "thermal patterns. (3) The noisy validation curves of the from-scratch family and electrical networks indicate high variance on small training sets, and the smooth, fast convergence of the pretrained networks shows that a "
            "better initialisation acts as a strong prior. (4) The persistent Stage 2 errors are an error-propagation effect: the family network and the specialists are trained independently, so a misrouted image meets a specialist "
            "that has never seen examples of its class, and no downstream stage can undo the mistake. (5) The tie between the cascade and the flat network with equal backbones shows that decomposing the task reduces the "
            "difficulty of each decision only as much as the routing errors cost."))
    return B
