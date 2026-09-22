# Deep Learning and Vision — Computer Vision Project

**Daniel da Cunha Moreira**

---

## How this document is organised

Four activities, each with the problem it addresses, the decisions taken and why, the
results, and a critical reading of those results. Activity 4.2 is written analysis only,
as the brief specifies.

Every figure quoted about a dataset was measured, and the command that measured it is in
the repository. Where a number comes from a training run, the notebook that produced it is
named.

---

## Reproducibility and tooling

All code lives in one installable Python package, `cv_mba`, with 188 unit tests and a
separate suite of evaluation checks. The notebooks contain narrative and orchestration
only; anything with logic in it sits in the package where it can be tested. That is what
makes the phrase "testable PyTorch modules" in the marking sheet true rather than claimed.

Local development is pinned to the exact library versions Google Colab ships, so code that
passes on a laptop behaves identically on the T4. Verified on 22 September 2026: numpy
2.1.3, pandas 2.2.3, matplotlib 3.10.0, scikit-learn 1.6.1, pillow 11.3.0, transformers
5.16.1, torch 2.11.0 with CUDA 12.8 on the GPU and the CPU build locally.

`torch` and `torchvision` are deliberately absent from the package's dependency list. Colab
already ships a build matched to its GPU driver, and declaring the dependency would invite
pip to replace it with a CPU-only wheel — after which every notebook would train silently
on the processor. This was verified on Colab rather than assumed: installing the package
leaves `torch 2.11.0+cu128` and the Tesla T4 untouched.

---

## Activity 1 — Vision Transformers on solar cell defect detection

### The problem

Electroluminescence inspection of photovoltaic solar cells. A current is passed through a
cell and the silicon emits infrared light. Healthy material glows evenly; cracks, broken
contact fingers and electrically isolated regions appear as dark lines and patches. The
task is to separate functional cells from defective ones.

The dataset is ELPV: 2,624 cells, 300×300 pixels, single channel, 1,074 monocrystalline
and 1,550 polycrystalline.

### Why this domain

Three reasons, in order of how much they mattered.

**Defects are localised.** A crack is a thin dark line in one specific place. This is what
makes the attention maps required by rubric lines 2.2 and 3.2 interpretable at all:
attention either lands on the defect or it visibly does not, and both outcomes are
something to write about. A domain classified by overall texture would produce attention
maps that are an even smear across the image and say nothing about what the model learned.

**The dataset is small.** 2,624 images. A Vision Transformer is handed no prior that
neighbouring pixels belong together, while a convolution is given exactly that for free.
Training a transformer from scratch on a few thousand images is therefore expected to
underperform, and measuring precisely how much is the entire point of the comparison in
rubric 3.6. A larger dataset would narrow the gap and weaken the argument.

**It is a real inspection problem in the energy sector**, which matches the professional
relevance the brief asks for.

### Decision: where to draw the binary line

Each cell carries a defect probability: 0.0, 0.333, 0.667 or 1.0. These record **annotator
agreement**, not a verdict. A cell at 0.333 is one where roughly two experts in three said
it was fine.

| Defect probability | Cells | Reading |
|---|---|---|
| 0.0 | 1,508 | all annotators agree it is functional |
| 0.333 | 295 | most annotators say functional |
| 0.667 | 106 | most annotators say defective |
| 1.0 | 715 | all annotators agree it is defective |

The cut is at **0.5**, giving 1,803 functional against 821 defective.

The obvious alternative — treating anything above zero as defective — produces a more
balanced 1,508 against 1,116 and is therefore tempting. It would have been a mistake. It
folds all 295 disputed cells into the defective class, which means roughly one in four
"defective" training examples is one the experts mostly thought was fine. A weak
from-scratch result would then be unreadable: impossible to separate data scarcity, which
is what this activity measures, from label noise, which is a different problem entirely.
Removing the confound is worth accepting a 2.2-to-1 imbalance, which per-class recall and
macro-F1 report honestly.

### Decision: stratifying on two variables, not one

Defect rates differ by wafer type: 34.4% of monocrystalline cells against 29.2% of
polycrystalline. A split balanced only on the label lets wafer type drift between the two
halves, so part of any measured difference between two models becomes which wafers happened
to land where. Splits are therefore stratified on label and wafer type jointly.

This is checked rather than assumed. If the defect-rate gap ever disappears from the
upstream dataset, an evaluation check fails and says the joint stratification is no longer
justified.

### Decision: model sizes

The from-scratch model is roughly a ViT-Tiny — 192 wide, six layers, three heads, about
2.9 million parameters. The pretrained model is torchvision's ViT-B/16 with ImageNet
weights, about 85.8 million, with its classification head replaced by a two-class layer.

The sizes are deliberately not matched. Matching them would mean either crippling the
pretrained model or training a ViT-Base from scratch on 2,100 images, which would not fail
interestingly — it would simply memorise. The comparison being made is *pretrained against
not pretrained*, and keeping each model at its natural size is what makes it a statement
about data rather than about parameter count. The parameter gap is reported alongside the
results so the reader can weigh it.

### Decision: three seeds, gated on the worst one

Both models are trained on three seeds. A single run's difference between two models is
partly the difference between two random initialisations, and the entire argument of rubric
3.6 rests on that difference being real.

The gate is deliberately strict: the pretrained model must win on **every** seed, not on
average. That is harder to pass than the more common "90% of runs" criterion, and it makes
the conclusion defensible. All three per-seed numbers are reported, not just the mean,
because the spread is itself informative.

### Rubric 2.4 — why attention alone cannot see position

Attention computes a weighted average over a **set** of tokens. Each output is a sum of
values weighted by similarity scores, and nothing in that computation refers to where a
token sat in the sequence. Permute the inputs and every output comes back permuted the same
way: the layer computed exactly the same thing under different labels. This property is
called permutation equivariance, and it is a consequence of the architecture rather than an
oversight.

For a Vision Transformer this is fatal on its own. Patches arrive as a flat sequence, so
without position the model can tell that a dark diagonal streak exists somewhere in the
cell but not that it runs across the middle — and "somewhere" is not a defect report.

Positional encoding fixes it by adding a distinct vector to each position before the first
attention layer. A token at position 0 and the same token at position 3 become genuinely
different inputs, and the symmetry is broken.

This is not argued in prose alone. Two tests in the repository state it as executable
claims: with positional encoding, shuffling the tokens changes the output in a way that is
not a relabelling; without it, the output is exactly the relabelling. Both run against both
flavours of encoding.

**The two flavours, and their trade-off.** A sinusoidal table is fixed, costs nothing to
store or train, and extends to sequences longer than anything seen during training, because
the formula is defined for every position. A learned table — which is what the Vision
Transformer actually uses — has one trainable vector per position and can discover for
itself what "position 17 of 197" should mean, which is strictly more flexible at a fixed
image size. The cost is that no vector exists for a position never seen in training, which
is why changing input resolution after training requires interpolating the table rather
than simply running the model.

### Rubric 2.5 — BERT pretraining against ViT pretraining

The two are often described as the same idea applied to different data. They are not, and
the difference is instructive.

**BERT maximises the probability of a hidden word given the words on both sides of it.**
Fifteen percent of tokens are masked and the model must reconstruct them from bidirectional
context. The supervision is free: it comes from the text itself, so any corpus of any size
can be used with no labelling.

**The original Vision Transformer maximises the probability of the correct class label
given the whole image.** Dosovitskiy and colleagues pretrained on ImageNet-21k and JFT-300M
— large **labelled** datasets. The pretraining is ordinary supervised classification, just
at enormous scale.

So the contrast is not "text against images". It is **self-supervised reconstruction
against supervised classification**.

The reason for the difference is worth stating, because it explains why masked-image
modelling took longer to arrive. BERT predicts a token from a finite vocabulary, so the
objective is a softmax over roughly thirty thousand options and the answer is
unambiguous. An image patch is a continuous array of pixel values drawn from no vocabulary
at all; there is nothing to take a softmax over. Later work solved this in two different
ways — MAE regresses the raw pixels of masked patches, and BEiT first learns a discrete
visual vocabulary and then predicts tokens from it, which is the closer analogue of BERT.

The practical consequence bears directly on this activity. BERT-style pretraining needs
only unlabelled data, which is abundant. ViT-style pretraining needs labelled data at a
scale nobody has for electroluminescence images of solar cells. That is precisely why the
from-scratch model here is expected to lose, and why transferring weights learned on
photographs — despite the obvious domain gap — is still the better starting point.

### Rubric 3.4 — what DeiT and Swin each fix in the original ViT

The original ViT left two specific problems, and these two papers address one each.

**DeiT fixes the data requirement.** The ViT paper's own conclusion was that transformers
beat convolutional networks only after pretraining on hundreds of millions of images;
trained on ImageNet-1k alone, ViT lost to a ResNet. That put the architecture out of reach
of anyone without Google-scale labelled data. DeiT trains a competitive Vision Transformer
on ImageNet-1k and nothing else, through a heavy regularisation and augmentation recipe
plus **distillation from a convolutional teacher** via a dedicated distillation token
added beside the classification token. The insight is that the teacher's inductive bias —
the locality prior a convolution has and a transformer does not — can be transferred
through its predictions rather than through its architecture.

This is directly relevant here. The from-scratch model in this activity fails for exactly
the reason DeiT was written to address, and DeiT's recipe is the most obvious improvement
to try next.

**Swin fixes cost and scale.** Attention is quadratic in the number of tokens. At 224×224
with 16×16 patches that is 196 tokens and perfectly affordable; at a resolution where
dense prediction becomes possible it is not. The original ViT also produces a single
feature map at one resolution, whereas detection and segmentation need a pyramid. Swin
computes attention inside **local windows** rather than globally, which makes cost linear
in image area, and **shifts the window grid between consecutive layers** so information
crosses window boundaries instead of being trapped. Patch merging between stages builds
the hierarchy of resolutions that convolutional backbones have always provided.

The honest summary is that Swin reintroduces locality and hierarchy — the two priors ViT
deliberately discarded — while keeping attention as the mixing operation.

### Rubric 3.5 — when a Vision Transformer beats a CNN, and when it does not

The deciding factor is almost always the ratio between how much data there is and how much
prior knowledge the architecture is given for free.

A convolution is handed two assumptions before it sees a single image: that nearby pixels
are related, and that a pattern means the same thing wherever it appears. Those assumptions
are correct for natural images, so a convolutional network starts partway to a solution.
A transformer is given neither. It can *learn* locality, and given enough data it learns
something better than the hand-designed version — but it must spend data doing so.

**Transformers win when** data is abundant or transferable; when the decision depends on
relating distant parts of the image, since attention reaches across the whole image in one
layer while a convolution needs depth to widen its receptive field; and when an explanation
of the decision is wanted, because attention weights are a directly readable account of
what the model looked at.

**Convolutions win when** data is scarce and cannot be supplemented; when the signal is
local texture rather than global arrangement; when inference must be cheap; and when input
resolution varies, since a convolution handles that natively while a learned positional
table does not.

**For this domain specifically.** ELPV has 2,624 images and the diagnostic signal — a
crack, a dark finger — is local. Both facts favour a convolutional network. The transformer
trained from scratch here is expected to lose, and the interesting result is not that it
loses but by how much, and what the pretrained model's advantage shows about how far
ImageNet features carry into infrared images of silicon.

The practical recommendation that follows is that a production inspection system for this
task should use a pretrained convolutional backbone, and that the Vision Transformer earns
its place only if the defect catalogue grows to include failures defined by the
*arrangement* of features across the whole cell rather than by a local mark.

---

## Activity 2 — Semantic recognition in visual advertising with CLIP

### The problem

Extract semantic structure from an advertising corpus using CLIP's pretrained embeddings
and natural-language queries, training nothing.

### What the corpus actually contains

The ADS-16 archive holds 2,697 images. Only **301 of them are advertisements**. The other
**2,396 belong to the 120 survey participants** — their own pictures, sorted into folders
they rated positively and negatively.

Two further findings, both measured against the archive on disk.

**The brief describes 16 product categories. The archive ships 20 folders.**
Advertisements sit in folders numbered 1 to 20 — one to ten in part one, eleven to twenty
in part two — holding 16 images in folder 1 and 15 in each of the rest, which totals 301.
Nothing in the archive names any folder; the only non-image files are a licence and the
source archives. The folder is therefore reported as found and never relabelled as a
product category to make the data agree with the brief. Where the figure of sixteen comes
from cannot be established from what was shipped.

**Participant pictures sit one directory level deeper than advertisements.** Each
participant folder holds two image subfolders plus five CSV files of survey responses. A
loader written for the advertisement layout returns zero participant images and reports no
error at all. Contribution is also not the uniform ten per participant the source paper
implies: 97 of the 120 contribute 20 pictures, the rest between 10 and 30.

### Decision: how to handle a corpus that is 301 images, not 500

The brief offers "the corpus, or a subset of at least 500 representative images". The
advertising set is 301, short of that floor.

Padding the advertisements with participant photographs to reach 500 would put personal
snapshots into a ranking of advertising content and make the result meaningless. Reporting
301 and moving on would leave the requirement unmet.

The resolution is to run over all 2,697 images and report the two partitions **separately**
— advertisements as the subject, participant pictures as a control. The contrast is the
finding rather than a compliance exercise. Concepts about text and branding should dominate
the advertising partition and collapse in the control; concepts about people should do the
reverse. That is a genuine check on whether CLIP's zero-shot reading of this corpus means
anything, and it satisfies both the whole-corpus reading and the 500-image floor without
pretending.

### Decision: a threshold that is relative, not absolute

CLIP's raw similarity values are **not comparable between images**. A cluttered or dark
photograph scores lower against *every* concept than a clean studio shot does. A single
absolute cut therefore counts concepts in the bright images and silently misses the same
concepts in the dim ones, so the ranking would partly measure image quality rather than
image content.

Presence is therefore decided **within each image**: a concept counts as present when it
scores more than one standard deviation above that image's own mean across all 25 concepts.
The question becomes "which concepts stand out in this image", which is the question
actually being asked.

This is not a preference. Two tests in the repository measure the failure being avoided:
two images with identical concept *structure* but different overall brightness are
classified differently by an absolute threshold and identically by the standardised one.

### Rubric 4.1 — why contrastive pretraining enables retrieval with no supervised training

CLIP is trained on one objective: given a batch of image and caption pairs, make each image
most similar to its own caption and less similar to every other caption in the batch, and
the same in reverse. Nothing is classified. The only thing learned is a **shared space**
where an image and a description of it land near each other.

Three consequences follow, and together they are the whole answer.

**Semantic closeness becomes geometric closeness.** Once images and text occupy the same
space, "which of these pictures shows a car" stops being a classification problem and
becomes a distance calculation. The reasoning was done during pretraining; at query time
there is only arithmetic.

**The set of possible answers is not fixed in advance.** A supervised classifier's final
layer has one output per class, decided before training, and adding a class means retraining.
CLIP's "classes" are whatever sentences you write at query time — an unbounded set defined
after training finished. This is precisely why it is called zero-shot: the model was never
shown a labelled example of the category being asked about.

**Contrast, not reconstruction, is what forces the useful structure.** Pushing mismatched
pairs apart is as important as pulling matched ones together. Without the negatives, a
degenerate solution exists — map everything to the same point and every pair is maximally
similar. The negatives make the space discriminative, and discriminative structure is what
makes ranking possible.

The limits are worth stating alongside the capability. The space encodes what was common in
four hundred million web image-caption pairs. Concepts rare on the web, or ones whose
captions do not describe what is visible, are represented poorly, and absolute similarity
values carry no calibrated meaning — which is exactly why the threshold above is relative.

### Rubric 4.4 — CLIP's text queries against BERT's tokenisation

Both models turn a sentence into integers and pad a batch to one length. The differences
are in what the padding is made of, what position is read out, and what the representation
was trained to be good at. The notebook prints both tokenisations side by side; what
follows is what that output shows.

**Both pad, and both mark the padding with an attention mask.** A short sentence in a batch
of long ones is stretched to a common length so the batch is one rectangular tensor. The
attention mask is zero at exactly those added positions. Inside the model that zero becomes
a minus-infinity entry in the attention scores, so softmax gives padding exactly zero weight
— which is what makes padding invisible rather than merely ignored by convention. Drop the
mask and the added positions are attended to like any other token, and a short sentence's
meaning drifts depending on what else happened to share its batch.

**CLIP has no dedicated padding token.** BERT pads with `[PAD]`, a symbol that appears
nowhere else. CLIP repeats `<|endoftext|>` — the very token it reads the sentence
representation from. The attention mask is the only thing separating the two roles: the
first end-of-text token is masked in and is the summary, every repetition after it is masked
out. This makes the mask more than an efficiency detail in CLIP's case. Without it, a short
caption would be summarised from a padding position, silently, with no error raised.

**They read different positions, for a structural reason.** BERT wraps the sentence in
`[CLS]` and `[SEP]` and reads `[CLS]` as the summary. It can do this because BERT is
bidirectional: every position sees the whole sentence, so the first position is as good as
any. CLIP's text encoder is **causal** — each token may only look backwards — so the first
position has seen only itself. The end-of-text token is the only position that has seen the
entire sentence, and that is why it is the one read out.

**The objectives differ, which is the difference that matters.** BERT's representation is
tuned to answer questions about language. CLIP's text representation is tuned for one thing
only: landing near the matching image. Neither is a general-purpose sentence embedding, and
this is the real reason CLIP can retrieve images from a text query while BERT cannot —
BERT's space contains no images to be near.

---

## Activity 3 — Transfer learning with a pretrained CNN

### The problem

Seven object classes, 1,803 images, supervised classification by transfer learning.

| Class | Images | Format |
|---|---|---|
| bike | 365 | bmp |
| cars | 420 | bmp |
| cats | 202 | jpg |
| dogs | 202 | jpg |
| flowers | 210 | png |
| horses | 202 | jpg |
| human | 202 | jpg |
| **Total** | **1,803** | |

### A defect in the supplied archive

The archive contains `data/` and, nested inside it, `data/data/` — a byte-for-byte
duplicate of all seven class folders. There are 3,606 image files on disk, exactly twice
1,803.

A loader that recurses therefore returns 3,606 images and invents an eighth class named
`data`, with every picture present twice. Split that 80/20 and the same image lands in
training and in validation. Validation accuracy then measures memorisation rather than
generalisation, and it does so while reporting an excellent number.

This is the same category of defect that Activity 4.1 asks the student to diagnose in
someone else's work, arriving in the data supplied for Activity 3. The loader used here
enumerates the seven named class folders and does not recurse, so the duplicate cannot
enter the result by construction; it raises a warning naming the directory, because a
recursive loader pointed at the same place would double the dataset silently. The notebook
prints both counts side by side so the difference is on the record.

### Rubric 1.5 — justifying the pretrained model for this domain

**ResNet-50 with ImageNet-1k V2 weights, backbone frozen.**

**Against the T4's memory.** Freezing the backbone removes the need to store activations
for a backward pass through it, so memory is dominated by the forward pass. At 224×224 with
batch 32 that is a few gigabytes against 15,360 MiB available — neither ResNet-50 nor any
smaller backbone comes close to the limit. Memory is therefore not the constraint here, and
choosing a smaller model to save it would be optimising something that was never scarce.

**Against the number of classes.** Seven classes with a frozen backbone means the trainable
part is a single linear layer: 2,048 inputs times 7 outputs plus 7 biases, 14,343
parameters against 23.5 million frozen. The notebook prints these three numbers, and they
are the direct evidence that this is feature extraction rather than fine-tuning.

**Against the alternative.** EfficientNet-B0 is the obvious competitor and is far smaller.
With a frozen backbone the decision reduces to which model's pooled features a linear
classifier can separate better, and ResNet-50's 2,048 dimensions offer more directions to
separate along than EfficientNet-B0's 1,280. The V2 weights also reach substantially higher
ImageNet top-1 accuracy than the original V1 recipe at no inference cost.

**Against the domain.** This is the strongest argument and it is easy to overlook. These
seven classes — bikes, cars, cats, dogs, flowers, horses, people — are ordinary photographic
subjects, and most of them are ImageNet categories or close relatives. The pretrained
features are not merely transferable here; they were learned on this exact kind of picture.
That is what makes feature extraction sufficient, and it is the clearest possible contrast
with Activity 1, where electroluminescence images of silicon share almost nothing with
ImageNet photographs.

### Rubric 1.4 — feature extraction against fine-tuning

The choice is decided by two quantities: how much labelled data there is, and how far the
target domain sits from the pretraining domain.

| | Domain close to pretraining | Domain far from pretraining |
|---|---|---|
| **Little data** | Freeze the backbone, train a new head. Few parameters to fit, so little to overfit. | The hard case. Frozen features may not describe the domain at all, but there is not enough data to retrain them. Unfreeze the last block only, use a small learning rate, lean on augmentation. |
| **Plenty of data** | Fine-tune everything at a small learning rate. There is enough signal to improve on the pretrained features without destroying them. | Fine-tune everything, or consider training from scratch. Pretraining is a better starting point than random, but it is only a starting point. |

The mechanism behind the table is that early layers learn general features — edges,
textures, colour opponency — that transfer almost everywhere, while later layers learn
progressively more dataset-specific combinations. Freezing keeps the general and discards
the specific; fine-tuning adapts both, and risks destroying the general if the learning rate
is too large or the data too thin.

**The two activities in this project sit in different cells of that table, which is what
makes the comparison instructive.**

Activity 3 is little data in a close domain: 1,803 images of everyday photographic subjects.
Feature extraction is the right answer, and full fine-tuning would likely be worse, not
better — 23.5 million parameters adjusting to 1,443 training images would overfit long
before it improved.

Activity 1 is little data in a far domain: 2,624 single-channel infrared images of silicon.
Frozen ImageNet features describe this domain poorly, so the pretrained model is fully
fine-tuned, at a learning rate ten times smaller than the from-scratch run. The small step
is what keeps the general early-layer features intact while the later layers adapt.

The practical rule that falls out of this: freeze first, because it is cheap and quick to
measure. If the frozen model's validation accuracy plateaus well below what the task needs,
that is evidence the frozen features do not describe the domain, and it is the signal to
start unfreezing from the top.

### Rubric 1.3 — augmentation strategies, and where each one would do damage

The rubric asks for at least three with justification specific to these classes. Five are
considered, because two of them are instructive precisely by being wrong.

**1. Horizontal flip — apply it.** No class in this dataset has an intrinsic handedness. A
mirrored bicycle is a bicycle, a mirrored dog is a dog. The transformation produces images
that could genuinely have been captured, which is the test any augmentation should pass.
*Harm:* none identified for these seven classes. It would be unsafe on text, digits, or any
domain with chirality.

**2. Random resized crop and scale variation — apply it, with limits.** The classes come
from different source collections and their framing differs sharply: bikes and cars are
photographed at varying distances in street scenes, while flowers tend to fill the frame.
Scale augmentation teaches the model that a car is a car whether it occupies a tenth of the
image or half of it.
*Harm:* for **flowers**, where the subject already fills the frame, an aggressive crop
removes the petal arrangement that distinguishes the class and leaves a patch of colour.
The crop scale should be bounded well above zero rather than left at the common 8% floor.

**3. Mild rotation — apply it, narrowly.** Cats, dogs, horses and flowers appear at
genuinely varied orientations, and a few degrees of rotation reflects real camera tilt.
*Harm:* **cars** and **bikes** in this collection are photographed upright from street
level. Large rotations produce images that no camera in the deployment setting would ever
capture, so the model spends capacity on a distribution it will never see. Ten to fifteen
degrees is defensible; ninety is not.

**4. Colour jitter — do not apply it, and this is the interesting case.** It looks harmless
and it is standard.
*Harm:* **flowers are identified substantially by colour.** Hue jitter can turn a red bloom
into a purple one and move it toward a different species' appearance while the label stays
put — the augmentation manufactures mislabelled training data. **Cars** and **bikes** are
the opposite: paint colour is genuinely irrelevant to the class, so jitter helps there. The
correct conclusion is that this augmentation should be applied per class or not at all, and
since per-class augmentation complicates the pipeline for a gain concentrated in two
classes, not at all is the right call here.

**5. Random grayscale — do not apply it.** Same reasoning as colour jitter, more extreme.
It would force shape-based recognition, which helps the animal classes slightly and
destroys the flower class outright.

**Normalisation is not augmentation and is not optional.** Both training and validation use
the ImageNet channel statistics, because the backbone learned its filters on inputs centred
that way. Feeding it raw 0-to-1 pixels shifts every activation away from what those filters
expect and wastes part of what transfer learning is for.

**A note on the validation path.** None of these transformations are applied to the
validation set. An augmented validation set measures a slightly different problem every
epoch, so its curve stops being comparable with itself and the training curves in rubric 1.2
become unreadable.

---

## Activity 4.1 — COVID-19 screening on chest X-rays

### The situation

A previous group closed this project with results marked "promising". The clinical team
reported that the model "ignores positive cases". Both statements were true simultaneously,
and understanding how is the whole exercise.

Their setup: 1,200 images in three classes — 840 Normal, 240 Pneumonia, 120 COVID-19 — split
80/20 without stratification. ResNet-18 with no pretrained weights, SGD, a fixed learning
rate of 0.01, fifteen epochs, no augmentation. Evaluated on global accuracy alone. Reported
93% training accuracy and 61% validation accuracy.

### Method: reproduce before repairing

Building a better classifier and declaring the problem solved would answer nothing. Every
number in this activity is a comparison, and a comparison needs a measured baseline. The
original split, architecture and training settings are therefore rebuilt exactly, from a
fixed seed, and the failure is measured rather than described.

### Rubric 5.1 — five technical problems and their clinical consequences

**1. Untreated class imbalance, seven to one against the class that matters.** 840 Normal
against 120 COVID, with no class weighting, no resampling and no weighted loss. A model that
answers "Normal" to every image scores 70% accuracy, so gradient descent is directly
rewarded for learning the prior instead of the disease.
*Clinical impact:* the model's failures concentrate entirely in the class where a miss sends
an infectious, untreated patient home. The errors are not distributed randomly across the
classes; they are concentrated exactly where they are most dangerous.

**2. Global accuracy as the only metric.** One number cannot report a three-class problem
whose classes carry unequal cost. The reproduced run shows an acceptable-looking global
accuracy beside a COVID recall that is not acceptable at all.
*Clinical impact:* the project passed its own acceptance test while failing at its purpose.
Nobody discovered this from the metrics — clinicians noticed by hand, after deployment. A
metric that cannot detect the failure mode you care about is not a safeguard.

**3. An unstratified 80/20 split.** With only 120 COVID images, the number landing in
validation is left to chance. The notebook measures this directly across ten different
random draws of the same data.
*Clinical impact:* each validation COVID case is worth several percentage points of recall,
so the reported figure is partly an artefact of the draw. A rerun can appear to improve or
degrade the model when nothing has changed, and a system can be approved or rejected on that
noise.

**4. No pretrained weights.** ResNet-18 trained from scratch on 960 images. There is nowhere
near enough data to learn general visual features, so capacity goes into memorising the
training set. The 32-point gap between their reported training and validation accuracy is
that memorisation, quantified.
*Clinical impact:* nothing general was learned, so performance will not survive a different
X-ray machine, a different exposure protocol or a different hospital. The model is fitted to
one dataset's idiosyncrasies rather than to lung pathology.

**5. A fixed learning rate, no augmentation, no early stopping.** 0.01 held constant for
fifteen epochs. The model cannot settle as it approaches a minimum, and it sees each image
exactly as captured.
*Clinical impact:* the model becomes sensitive to acquisition artefacts — patient rotation,
contrast settings, cropping, annotation markers burned into the corner of the image — rather
than to pathology. A routine change in radiography protocol degrades it silently.

**A sixth problem, larger than the other five combined.** The archive holds **3,616 COVID
images**. The project used 120. The scarcity the whole exercise works around was
self-imposed. Before any modelling improvement is worth discussing, using the data that
already exists is the cheapest available fix by an enormous margin — and synthetic
augmentation should be evaluated against that option, not instead of it.

### Rubric 5.2 — the conditional GAN, and a bug worth describing

The generator is conditioned on the class label so it can be asked specifically for COVID
images. Without conditioning, asking for more images returns the overall distribution —
mostly Normal, since that is most of the training data — and the augmentation adds nothing
that was scarce.

The first implementation was conditional in name only, and the way it failed is worth
recording because the same shape appears in a great deal of published conditional GAN code.

The design concatenates the class embedding to the noise vector and lets ordinary batch
normalisation follow. Batch normalisation subtracts the batch mean, feature by feature. When
**every sample in a batch carries the same label** — which is exactly what happens when you
request a batch of 120 COVID images — the label's entire contribution *is* that batch mean,
so normalisation removes it completely and the generator produces unconditional output.

It fails silently, it fails only under uniform-label batches, and it fails in the precise
situation the model exists for. A unit test caught it; visual inspection of samples would
not have.

The fix is **class-conditional batch normalisation**: the label supplies the scale and shift
applied after normalisation, rather than being carried through it. The conditioning cannot
then be normalised away, because it is applied after the normalisation rather than before.
A regression test now checks the label survives in training mode, in evaluation mode, and
with both uniform and mixed batches.

### Rubric 5.3 — detecting instability that the losses do not show

Mode collapse is the failure where the generator finds one image the discriminator happens
to accept and emits it forever. The loss curves look unremarkable while it happens, which is
what makes it dangerous.

The detector used here is a **diversity score**: the mean pairwise distance between 32 COVID
samples generated from a fixed noise batch, measured every epoch. A generator producing
identical images scores zero. Watching this across training turns "the samples look similar"
into a curve that can be plotted before and after a change.

The mitigation is a **minibatch standard deviation channel** on the discriminator: the
spread of features across the batch, appended as one more channel before the final scoring
layer. The mechanism fits in a sentence — a plain discriminator judges each image alone and
cannot tell thirty-two copies from thirty-two different pictures, while a batch-aware one
can, so collapse becomes a signature it can learn to punish rather than a blind spot.

Both runs are plotted together, and the generator carried forward is chosen on the measured
diversity rather than by eye.

### Rubric 5.4 — the design of the recall experiment

The experiment is kept deliberately narrow. **One** generator, its output frozen, and three
seeded classifier pairs that differ only in whether the synthetic COVID images are present
in training.

Retraining the GAN once per seed was the obvious alternative and it is wrong: it measures
GAN reproducibility and augmentation benefit at the same time and answers neither. The
question rubric 5.4 asks is whether *these* synthetic images help, so the synthetic set is
the treatment and must be held fixed.

The split is stratified for this experiment even though the original was not, because
problem 3 has to be fixed before problem 1 can be measured — otherwise the recall difference
is partly split noise.

All six recall values are reported, not the mean. A single-run recall delta is precisely the
sloppiness this activity exists to diagnose.

**A negative result would be a finding, not a failure.** Synthetic augmentation does not
reliably improve minority-class recall; generated images can reinforce whatever the
generator learned from the same scarce data rather than adding genuinely new information.
Six honest numbers showing no improvement contribute more to the improvement plan than a
positive delta obtained by re-rolling seeds until one appears.

### Rubric 5.6 — an integrated improvement plan

The five problems are not independent, and fixing them in the wrong order produces
measurements that cannot be trusted. The order below is the plan.

**Stage 0, before any modelling: use the data that exists.** The archive holds 3,616 COVID
images against the 120 used. Nothing in the rest of this plan is worth doing on 120 images
when 3,616 are sitting on disk. This alone changes the imbalance from seven-to-one to
roughly three-to-one and removes most of the scarcity the rest of the plan works around.

**Stage 1, measurement, because nothing else can be judged without it.**
- Split stratified by class, and report the composition of both halves.
- Replace global accuracy with per-class recall, per-class precision, macro-F1 and a
  confusion matrix, with COVID recall named as the primary metric before any model is run.
- Fix the decision threshold on a validation set and never tune it on the test set.
- Report a confidence interval, not a point estimate. With roughly 24 COVID cases in a
  validation split, a single case is worth four percentage points of recall, and a figure
  quoted without that context is misleading.

**Stage 2, the model.**
- Start from pretrained weights. A ResNet-18 trained from scratch on a thousand chest X-rays
  learns the training set; an ImageNet-pretrained backbone at least starts from general
  visual features. Where available, weights pretrained on radiographs are better still,
  because the domain gap from photographs to X-rays is large.
- Weight the loss by inverse class frequency, or oversample the minority class. This is the
  direct fix for problem 1 and costs nothing.
- Replace the fixed learning rate with a schedule, add early stopping on COVID recall rather
  than on global loss, and add augmentation that reflects real acquisition variation —
  small rotations, brightness and contrast shifts, mild cropping. Not horizontal flips:
  the heart is on one side, and a mirrored chest X-ray is an image that cannot occur.

**Stage 3, synthetic augmentation, judged rather than assumed.** The experiment in this
activity gives a measured answer for this generator on this data, and the plan follows that
answer in either direction.

- *If synthetic images raise COVID recall across all three seeds:* keep them, but cap their
  share of the training set and re-measure at several ratios, because a set dominated by
  generated images optimises the classifier towards the generator's idea of COVID rather
  than towards COVID. Never let synthetic images enter the validation or test sets, under
  any circumstances.
- *If they do not:* record it and stop. A GAN trained on 120 COVID images can only
  redistribute what those 120 contain; it cannot manufacture pathology it has never seen.
  In that case the honest conclusion is that the scarcity must be solved with real data —
  Stage 0 — and the synthetic route is a dead end for this dataset. Reporting that saves the
  next team the same experiment.

**Stage 4, the clinical adoption criterion.** This is the part most projects omit, and it is
the part that decides whether anything above matters.

The system is adopted **as a prioritisation aid only, never as a rule-out.** It may reorder
a radiologist's queue; it may not remove a case from it. Every image is read by a human
regardless of what the model says. This single constraint changes the risk profile entirely:
a false negative delays a diagnosis by the length of the queue rather than missing it.

Adoption requires all of the following, and failure of any one blocks deployment:

1. **COVID recall at or above 0.95**, at a decision threshold fixed before the evaluation
   ran, measured on a test set the model has never been tuned against.
2. **A false-positive rate low enough to be operationally useful.** A model that flags
   everything achieves perfect recall and reorders nothing. The acceptable rate is set by
   the radiology department's capacity, not by the modelling team.
3. **External validation.** The test set comes from a different hospital, different
   equipment and a different time period than the training data. A model validated only on
   its own source is a model whose generalisation is unmeasured.
4. **Prospective evaluation before full deployment.** Run silently alongside normal
   workflow, with predictions recorded and not shown, for a defined period. Compare against
   the radiologists' reports afterwards.
5. **Continuous monitoring with a defined stopping rule.** Track the per-class prediction
   distribution weekly. A shift indicates the input distribution has changed — new
   equipment, a new protocol, a new variant — and the system reverts to unaided workflow
   automatically, rather than waiting for someone to notice.
6. **A documented failure mode.** Clinicians are told, in writing, what the model is bad at:
   early-stage presentations, paediatric images if those were absent from training,
   portable-machine acquisitions. A tool whose limits are undocumented will be trusted
   outside them.

The thread running through all six: the previous project's deepest error was not any single
technical choice. It was that **nothing in its process could have detected the failure**.
Global accuracy could not see it, the split made it noisy, and there was no clinical
acceptance criterion the model could fail against. The clinicians found it by hand. Every
item above exists so that a future failure surfaces in a measurement instead of a complaint.

---

## Use of AI tools

This project was developed with substantial use of AI assistance, declared here in full as
the brief requires.

**Tool used.** Claude (Anthropic), through an agentic development environment with access to
a terminal, the repository and the datasets.

**What it was used for.**
- Drafting and reviewing the specification, the implementation plan and this report.
- Writing the package source, the unit tests and the notebooks, under a test-first
  discipline in which each test was written and observed to fail before the code that
  satisfies it was written.
- Inspecting the datasets directly to establish their real contents, which is how the
  duplicate directory in the Activity 3 archive, the twenty-folder structure of ADS-16 and
  the nesting depth of its participant images were found.
- Cross-checking implementations against independent references: the metrics against
  scikit-learn, the from-scratch attention against `torch.nn.MultiheadAttention`.

**What was verified rather than accepted.** Every quantitative claim about a dataset in this
report is the output of a command that was run, and those commands are in the repository as
automated checks that re-run against the real archives. Two design errors produced by the
initial AI-written code were caught by those checks and are documented in this report: the
loader that refused to read its own dataset, and the conditional GAN whose class
conditioning was erased by batch normalisation. Both were found by tests, not by review.

**Where AI output was rejected.** The assignment brief states that ADS-16 contains sixteen
product categories. The archive contains twenty numbered folders and no category names. The
data was reported as measured rather than reconciled to the brief.

**Limitations acknowledged.** AI-generated text and code can be confidently wrong. The
defence used throughout this project is that claims are checked by executable tests against
real data rather than by reading, and that every number quoted traces to a command whose
output was inspected.

---

---

## Activity 4.2 — Transfer learning for urban traffic analysis

### The situation

A group built a traffic-flow classifier labelling urban camera images as free, moderate or
congested, using ResNet-50 pretrained on ImageNet, fine-tuned on 800 labelled frames from 12
distinct cameras. Validation accuracy was 78% and the system was deployed. In production it
failed systematically in rain, at night, and on cameras whose angles were not in the
training set.

No implementation is required for this activity. The analysis follows.

### Rubric 5.5 — four technical and methodological problems

**1. The 800 frames are nowhere near 800 independent samples.** They come from 12 cameras.
Frames from one camera share a background, a mounting angle, a lens, a compression profile
and a fixed set of road markings. In terms of what the model can generalise over, the
effective sample size is closer to 12 than to 800.

This becomes a leak the moment the split is made. If the 80/20 division is random **over
frames**, every camera appears in both halves, and the model can reach high validation
accuracy by recognising *which camera* it is looking at and recalling that camera's typical
traffic state. The split has to be **grouped by camera** — whole cameras held out — or the
validation number measures memorised backgrounds.
*Operational impact:* 78% was very likely never a measure of generalisation, which explains
why production behaviour bore no relation to it. The correct first step is to re-evaluate
with a camera-grouped split, and to expect a much lower and much more honest number.

**2. The task is not object recognition, and ImageNet features are object recognisers.**
ImageNet answers "what single object is in this photograph". Traffic state is a judgement
about **how many** vehicles there are and **how they are distributed** across the road — a
density and flow estimate over the whole scene, not the identification of a dominant object.
Features tuned to fire on the presence of a car are not obviously the right features for
counting cars or for judging spacing between them.
*Operational impact:* the model likely keys on proxies that correlate with congestion in the
training cameras — brake-light density, a particular road region being dark — rather than on
traffic density itself. Those proxies are exactly what break when the camera or the lighting
changes.

**3. The three classes are ordinal and are being treated as unordered.** Free, moderate and
congested lie on a scale. Cross-entropy over three independent classes treats confusing free
with congested as no worse than confusing free with moderate, when operationally the first
is a serious error and the second is a borderline call that human annotators themselves
disagree about.
*Operational impact:* the model is not penalised for its most damaging error, and 78%
accuracy hides whether the remaining 22% are near-miss disagreements or complete inversions.
An ordinal loss, or regression onto a continuous congestion score with thresholds applied
afterwards, matches the problem's structure. A confusion matrix would at minimum have
revealed which kind of error dominates.

**4. The training distribution excludes the conditions the system was deployed into.** The
failures — rain, night, unseen angles — are not surprising failures. They are the model
being asked about conditions that were absent from its training data and largely absent from
ImageNet, which is overwhelmingly composed of well-lit, in-focus, daylight photographs.
*Operational impact:* the system is least reliable exactly when traffic management matters
most. Rain and night are when congestion is worst and when the operator most needs an
accurate reading.

**5. No per-class metrics, no calibration, no defined operating point.** A single accuracy
figure was the entire evaluation, and it was computed once before deployment with no
production monitoring afterwards.
*Operational impact:* the degradation was discovered by users rather than by the system.
There was no mechanism to detect that the deployed model had stopped working.

### The specific risks of transferring ImageNet features to this task

ImageNet photographs are object-centric, taken by people, with a single dominant subject
occupying a large share of a well-exposed frame. A traffic camera is fixed, wide-angle,
often low resolution, heavily compressed, and shows dozens of small, mutually occluding
objects under uncontrolled lighting and weather. The **feature extractor** transfers
reasonably — edges and vehicle-like shapes remain useful — but the **task structure** does
not transfer at all, because ImageNet never required counting, spatial density estimation,
or robustness to a fixed viewpoint with a constant background.

The sharpest way to state the risk: ImageNet pretraining makes the model good at answering
"is there a car in this image". The deployed question is "how much traffic is there", and
the two are related closely enough to produce a plausible validation number and far enough
apart to fail in production.

### How each problem would be addressed

| Problem | Approach |
|---|---|
| Camera leakage | Split by camera, never by frame. Report per-camera results and leave-one-camera-out cross-validation, so generalisation to a new camera is measured directly rather than assumed. |
| Wrong task framing | Predict a continuous congestion score — vehicle count or occupancy fraction — and threshold afterwards. Consider a detection or density-estimation backbone rather than a whole-image classifier. |
| Ordinal classes treated as nominal | Ordinal loss or regression with thresholds. Report a confusion matrix so the direction of errors is visible. |
| Missing conditions | Collect rain and night frames deliberately rather than waiting for them; augment with synthetic weather and low-light transformations; consider a backbone pretrained on driving or surveillance imagery rather than ImageNet. |
| Unseen angles | Normalise geometry with a per-camera homography to a common bird's-eye view, so the model sees a consistent representation regardless of mounting. |
| No monitoring | Track the prediction distribution per camera in production. A camera whose output distribution shifts is a camera whose conditions have changed, which is detectable before anyone complains. |

---

## Sources

- Vaswani et al., *Attention Is All You Need*, 2017.
- Dosovitskiy et al., *An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale*, 2021.
- Touvron et al., *Training Data-Efficient Image Transformers and Distillation Through Attention* (DeiT), 2021.
- Liu et al., *Swin Transformer: Hierarchical Vision Transformer Using Shifted Windows*, 2021.
- Devlin et al., *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*, 2019.
- Radford et al., *Learning Transferable Visual Models From Natural Language Supervision* (CLIP), 2021.
- He et al., *Masked Autoencoders Are Scalable Vision Learners*, 2022.
- Karras et al., *Progressive Growing of GANs for Improved Quality, Stability, and Variation*, 2018 — source of the minibatch standard deviation technique used in Activity 4.1.
- Nour & Tariq, *Scientific Reports*, 2023. https://www.nature.com/articles/s41598-023-37743-4
- Buerhop-Lutz et al., ELPV dataset. https://github.com/zae-bayern/elpv-dataset
- ADS-16 Computational Advertising Dataset. https://www.kaggle.com/datasets/groffo/ads16-dataset
