# Training a Handwritten Kanji Recognition Model with ETL9B (PyTorch Outline)

## 1. Dataset Acquisition & Preprocessing (ETL9B)

### Download & Extraction
Obtain the ETL9B dataset from the AIST ETL Character Database. ETL9B contains ~607,200 handwritten character images (binary 128×127 pixel bitmaps) representing 3,036 classes (JIS level-1 Kanji plus hiragana). Since there is no built-in PyTorch loader, use provided data files and documentation to extract images and labels (e.g., via a custom script or an ETL data reader library).

### Parsing & Formatting
Parse the ETL9B binary files to load each character sample into a NumPy array or PIL image. Each sample includes an image and a label (character code). Map the character codes to class indices (0–3035). Convert images to grayscale tensors (if not already) and normalize pixel values (e.g. scale 0–255 to 0–1 float). Ensure the handwriting is in the correct orientation and contrast (black strokes on white background).

### Dataset Preparation
Split the data into training and validation sets. A common approach is to hold out a subset of writers or a percentage of samples per class for validation. Organize the data using a PyTorch Dataset and DataLoader for batching. Apply any necessary padding or cropping so that images have a consistent size (e.g., pad to 128×128) if required by your model. This step yields a ready-to-use PyTorch dataset of Kanji images with their class labels.

## 2. Potential Complementary Datasets (for Kanji and Stroke Data)

### Kuzushiji-Kanji
A dataset of historical cursive kanji characters (3,832 classes, ~140k images) can supplement training data or pretraining. Although the style (cursive Edo-period script) differs from modern handwriting, using Kuzushiji-Kanji can help the model learn general kanji patterns beyond the ETL9B data. It is especially useful for testing generalization or for multi-stage training (pretrain on Kuzushiji, then fine-tune on ETL9B).

### Online Handwriting Datasets
To incorporate stroke-sequence learning, consider an online Kanji handwriting dataset. For example, the *Nakayosi* or *Kuchibue* databases contain stroke by-stroke pen trajectory data for thousands of Japanese characters. These can provide ground-truth stroke order and stroke count information. By training a model on online data (pen coordinates over time) or using it to annotate stroke order for ETL9B images, you can infuse stroke sequence awareness.

### Chinese Character Databases
Large Chinese handwriting datasets like *CASIA-HWDB* (contains 3,755 Chinese characters with thousands of samples each) are valuable for transfer learning due to the overlap in character structure. Many Kanji are derived from Chinese characters, so their strokes/radicals are similar. A model pretrained on Chinese characters can recognize common sub-components of Kanji, as discussed in Step 5.

### Synthetic Data (Fonts)
You can generate synthetic training data by rendering Kanji using various fonts and applying distortions. Create images of Kanji from standard fonts, then apply transformations (noise, deformation, stroke thinning/thickening) to mimic handwriting. This font-generated dataset can be used to pretrain a model on basic Kanji shapes before it sees real handwritten variability. It can also provide stroke-order information if combined with vector font data (e.g., using KanjiVG stroke definitions) to draw one stroke at a time.

## 3. Model Architecture Suggestions (Stroke-Aware Kanji Recognition)

### CNN Baseline
Start with a convolutional neural network that can learn spatial features of characters. A deep CNN (e.g. a ResNet-18 or a custom 8-12 layer CNN) is suitable for classifying Kanji images. The network should be designed to handle the input image size (~128×128) and output 3,036 classes. For instance, you can use a ResNet architecture (modified for single-channel input and 3036 outputs) as a backbone. This provides a strong baseline for overall character recognition, learning hierarchies of strokes and radicals from the pixel data.

### Stroke-Sequence-Aware Design
To emphasize stroke order or structure, consider a hybrid architecture that combines CNN features with sequence modeling:

#### Multi-Task CNN
One approach is to have the CNN simultaneously predict the character class **and** intermediate stroke information. For example, add auxiliary output nodes that predict the presence or absence of each stroke (or stroke type) in the character. Gyohten et al. (2020) demonstrated that a CNN can be trained on images with missing strokes to develop internal units indicating each stroke's presence. This kind of model learns stroke-level features: the CNN's final layers could branch into a classifier head (for overall character) and a stroke detection head. During training, the stroke head would require stroke-labels (which could come from a stroke decomposition of the character using a tool like KanjiVG or an online dataset).

#### CNN + RNN (Encoder-Decoder)
Another architecture is an encoder-decoder model. Use a CNN encoder to extract a high-level feature map of the image, then a recurrent decoder (LSTM/GRU) to generate a sequence of strokes or stroke labels. The CNN encoder learns a representation that the decoder uses to predict stroke order step-by-step (as if "tracing" the character). The model can be trained with sequence supervision (if actual stroke sequences are available) or in a multi-task fashion: sequence prediction plus a final classification output. This stroke-sequence modeling forces the encoder to preserve stroke-order information in its latent features.

#### Alternative Models
Consider experimenting with novel architectures like Vision Transformers (ViT) or Capsule Networks. A ViT can encode the image as patches, which might capture different parts of the character (though it doesn't inherently know strokes). Capsule Networks attempt to preserve entity parts (strokes) and their relationships, which could align well with Kanji's compositional nature. These alternatives may be more complex to train, but they offer different ways of modeling strokes and spatial relationships. In practice, a well-designed CNN (with perhaps an added recurrent or multi-head component for strokes) is the most straightforward approach in PyTorch.

## 4. Training Loop Design, Loss Functions, and Evaluation Metrics

### Training Loop
Implement a standard PyTorch training loop. For each epoch, iterate over the training DataLoader:
- Forward each batch of images through the model to get predictions.
- Compute the loss (see below for loss choices).
- Backpropagate the loss and update weights using an optimizer (e.g. SGD or Adam).
- Use learning rate scheduling if needed (for example, reduce LR on plateau or a cosine schedule) to help convergence.

Continuously monitor training loss and accuracy. It's often useful to print or log progress every few batches or epochs given the dataset size.

### Loss Functions
For a pure classification approach, use **cross-entropy loss** on the 3036-class output. This will train the network to predict the correct Kanji label. If you introduced auxiliary outputs (e.g. stroke presence predictions), include additional loss terms for those – for instance, a binary cross-entropy or multi-label loss for stroke presence (with a target vector indicating which strokes are in the character).

In a CNN-LSTM sequence model, you might use a sequence loss (e.g. cross-entropy at each step of the stroke sequence, or a CTC loss if using an unaligned sequence approach). When combining losses, weight them so that the main task (character classification) remains the primary objective (e.g., total loss = class_loss + λ * stroke_loss).

### Evaluation Metrics
During training, evaluate on the validation set after each epoch. Compute **accuracy** as the primary metric – the percentage of Kanji images correctly classified. Given the large number of classes, you might also track **Top-5 accuracy** (the rate at which the correct class is in the model's top-5 predictions) to get a sense of partial correctness.

For stroke-sequence outputs, you can measure sequence accuracy or edit distance between predicted and true stroke sequences (if ground truth sequences are available). Additionally, observe the model's performance per subset (e.g., how it fares on very complex characters vs simpler ones). Use these metrics to decide when to stop training (early stopping) or which epoch's model to deploy (choose the model with highest validation accuracy).

### Testing
Once satisfied with training, evaluate the final model on a test set (or reserved validation set) that was not used in training at all. Compute overall accuracy and possibly a confusion matrix to see which characters are most frequently confused. This can highlight if certain similar Kanji are problematic.

## 5. Transfer Learning Strategies (leveraging existing knowledge)

### Pretrained CNN Models
Utilize transfer learning to accelerate training. One option is to take a CNN pretrained on ImageNet (for example, ResNet-18) and adapt it to Kanji. Modify the first convolution to accept 1-channel input (or repeat the grayscale image into 3 channels), and replace the final fully-connected layer with one that has 3,036 outputs. You can then fine-tune this network on the ETL9B data.

Pretrained models bring learned low-level features (edges, curves) that are beneficial for character recognition. Often, you would freeze the early layers initially and only train the later layers/classifier, then gradually unfreeze more layers as needed. This approach can significantly speed up convergence.

### Cross-Language Pretraining
A very effective strategy (if data permits) is to pretrain on a large Chinese character dataset and then fine-tune on Japanese Kanji. The rationale is that Chinese characters and Japanese Kanji share many fundamental components – strokes and radicals – due to their common origins. A model first trained on a broad set of Chinese characters will learn robust representations of these recurring substructures (common radicals, stroke patterns), which serve as a strong initialization for Kanji recognition.

In practice, researchers have found that training on the CASIA-HWDB Chinese handwriting database and then fine-tuning on a Japanese Kanji dataset yields improved accuracy. For example, one can train a ResNet-based model on CASIA-HWDB (which is extensive and balanced) to learn stroke features, then fine-tune on ETL9B or Kuzushiji-Kanji to adapt to the style of Japanese handwriting. This cross-language transfer can increase recognition rates and is a relevant approach for Kanji, as noted in recent studies.

### Domain Adaptation & Fine-Tuning
Regardless of pretraining source, always fine-tune on the target ETL9B data. Monitor performance closely during fine-tuning – a small learning rate is often used to avoid distorting the pretrained features too quickly. If using an auxiliary stroke loss, you might pretrain the network's stroke-recognition capability on some synthetic stroke data or an online dataset, then fine-tune the whole model on the offline data.

Transfer learning isn't limited to weights: you could also transfer knowledge via techniques like knowledge distillation (training a smaller model to mimic a larger one trained on a related task), though this is an optional advanced idea. The main point is to initialize the training with some prior knowledge, either from generic vision models or from character-specific models, to improve convergence and final accuracy.

## 6. Data Augmentation & Style Generalization

### Geometric Augmentations
Introduce random perturbations to the training images to make the model invariant to minor differences. Apply small rotations (e.g. ±5 degrees) and translations (shifting a few pixels) to simulate the natural variation in how a character might be written or scanned. Scaling (zoom in/out by a small factor) can help the model cope with slightly different stroke sizes or writing area.

Avoid extreme rotations or flips – for instance, flipping a character horizontally or vertically would generally create a different character or an invalid one, so such transformations are not appropriate for Kanji.

### Photometric Augmentations
Since ETL9B images are binary (black and white), you can simulate grayscale variations to mimic real pen ink intensity differences. For example, add a bit of noise or blur to strokes to simulate ink spread or a fuzzy scan. Randomly thicken or thin strokes by applying morphological operations (dilate or erode the binary image slightly) – this helps the model handle different pen weights or writing pressure. You can also vary brightness/contrast if the dataset images have varying darkness.

### Stroke-Level Augmentations
To emphasize stroke structure, consider generating augmented samples that alter strokes. A powerful technique is **stroke dropout** augmentation: remove or erase a random stroke from the character image occasionally during training. This idea, inspired by Gyohten et al.'s method, forces the network to learn to identify each individual stroke. By training on images missing a stroke (with the class label unchanged), the model learns to rely on the presence of specific strokes for recognition and becomes sensitive to stroke omissions.

Another augmentation is using stroke order data (e.g., from KanjiVG or an online source) to draw characters in different stroke orders or segment the image by stroke – for instance, overlay strokes in random colored layers – to teach the model the concept of stroke separation. These advanced augmentations inject stroke-sequence knowledge even when training on static images.

### Style Generalization
ETL9B has many writers, but to ensure the model generalizes to new handwriting styles, apply augmentations that simulate style differences. For example, use elastic distortions to slightly warp strokes (mimicking a more cursive or shaky handwriting). You might also mix in samples from other datasets (as mentioned in Step 2) during training as a form of augmentation – e.g., train on a combination of ETL9B and a subset of Kuzushiji-Kanji or other handwriting to expose the model to a wider range of stroke shapes.

Always ensure augmented images remain recognizable as the same character. The goal is to prevent the model from overfitting to the specific idiosyncrasies of the training writers and instead focus on the core shape of the Kanji. Through aggressive and clever augmentation, the model will better handle variations in stroke order, pressure, noise, and personal style.

## 7. Expected Challenges and Mitigation Strategies

### Large Number of Classes
With 3,036 Kanji classes, the model must learn to distinguish many similar characters. Some Kanji differ by only a single stroke or a slight variation in stroke placement, making misclassifications common. This is challenging because a network might confuse characters that share most strokes in common.

Mitigation: ensure the model has sufficient capacity (depth and filters) to learn fine details, and use the stroke-aware strategies mentioned (auxiliary stroke prediction or training on stroke-missing images) to sharpen its ability to detect the presence/absence of critical strokes. You can also organize classes hierarchically (for analysis purposes) by radicals or stroke counts to see where confusions occur, although the model itself will handle all classes jointly. Monitoring per-class accuracy can help identify if certain characters need more training data or augmentation.

### Variability in Handwriting Styles
Different people have varying handwriting – some write strokes with different shapes, proportions, or slight stylizations. The model might struggle with styles not represented in the training set. To combat this, use extensive data augmentation (as in Step 6) and possibly include diverse datasets or synthetic styles in training.

Another approach is to do *writer-independent evaluation*: reserve all samples from a few writers as a test set to simulate encountering a new person's handwriting. If performance is lower on new writers, it indicates a need for more robust training (more augmentation or data). Techniques like adversarial training (making the model invariant to style by adding noise in feature space) could be considered if style variance remains an issue.

### Stroke Order Ambiguity
In offline images, the stroke drawing order is not explicitly given, yet different stroke orders or directions can produce visually similar end results. If the model attempts to infer stroke order internally, this ambiguity can be difficult. One writer might write a character with a non-standard stroke order, yielding a slightly different look.

Mitigation: relying on final appearance (which the CNN naturally does) is usually sufficient for recognition, but if stroke sequence learning is a goal, you need external data or assumptions. Providing the model with standard stroke order information (from a lookup table like KanjiVG or an online sample) for each character during training can guide it, but be aware that writers occasionally violate the standard order. The model should learn to handle slight deviations – focusing on stroke presence and shape rather than strict sequential order unless an exact sequence output is required.

### Data Size and Training Time
ETL9B's ~600k images make training computationally intensive. Training a deep CNN on this dataset will require a good GPU and potentially many epochs to reach peak accuracy. Watch out for overfitting: even with 600k samples, overfitting can happen, especially if the model is very large relative to the diversity of the data.

Regularization techniques like dropout and weight decay are recommended. Also consider early stopping if validation performance starts dropping. To speed up training, you can filter or compress the dataset (for example, start training on a smaller subset or at lower resolution, then gradually increase the data size or image resolution). Using multiple GPUs or distributed training can help handle the volume of data.

### Class Imbalance and Rare Characters
If some Kanji classes have fewer samples or are inherently harder (though ETL9B is fairly balanced by design), the model might underperform on them. Monitor the confusion matrix: if you see certain characters often predicted as a similar-looking one, that indicates a need for targeted data augmentation for that pair of classes.

In an extreme case, you could implement a two-stage classifier: first a coarse classifier (perhaps by radical or first stroke) then a fine classifier among similar candidates – but with modern deep networks, a single-stage model is usually sufficient given enough training data.

### Evaluation and Error Analysis
A challenge after training is understanding errors. The model might achieve high overall accuracy, but it's important to verify it's truly learning stroke patterns and not just overfitting to writer-specific quirks. Use visualization techniques like Grad-CAM to see which parts of the character the network focuses on. This can confirm if it's attending to expected stroke regions.

If the focus seems odd or the model makes systematic mistakes (e.g., confusing all characters that contain a particular component), you may need to adjust the training process or architecture (such as explicitly teaching that component via auxiliary tasks). Recognizing *why* a misclassification happens (as Gyohten et al. aimed to do with stroke-missing analysis) can guide further improvements.

For example, if the model confuses characters that have an extra stroke, implementing the stroke-presence auxiliary loss (so the model has neurons dedicated to that extra stroke) can help it differentiate those cases. In summary, expect that distinguishing thousands of Kanji – especially those that are visually similar – is non-trivial; a combination of strong convolutional features, stroke-aware learning, robust augmentation, and transfer learning will be needed to meet this challenge.

## References

1. [Datasets for Computer Vision](https://dev.to/hyperkai/datasets-for-computer-vision-1-1p0f)
2. [Gyohten et al. (2020)](https://www.scitepress.org/Papers/2020/89490/89490.pdf)
3. [Cross-Language Transfer-Learning Approach via a Pretrained Preact ResNet-18 Architecture for Improving Kanji Recognition Accuracy and Enhancing a Number of Recognizable Kanji](https://www.mdpi.com/2076-3417/15/9/4894)
