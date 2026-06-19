---
source_url: https://arxiv.org/pdf/2509.20787
---

## Page 1

Real-Time Object Detection Meets DINOv3
Shihua Huang1⋆, Yongjie Hou1, 2⋆, Longfei Liu1⋆, Xuanlong Yu1, Xi Shen1†
1 Intellindust AI Lab; 2Xiamen University
⋆ Equal Contribution; † Corresponding author.
Project Page: https://intellindust-ai-lab.github.io/projects/DEIMv2
Code & Weights: https://github.com/Intellindust-AI-Lab/DEIMv2
10 20 30 40 50 60 70
#Params (M)
46
48
50
52
54
56
58COCO APval
50.9
53.0
56.0
57.8
DEIMv2 (ours)
DEIM
D-FINE
YOLOv11
YOLOv12
(a)COCO performance v.s. Number of Parameters.
25 50 75 100 125 150 175 200 225
#FLOPs (G)
46
48
50
52
54
56
58COCO APval
50.9
53.0
56.0
57.8
DEIMv2 (ours)
DEIM
D-FINE
YOLOv11
YOLOv12 (b)COCO performance v.s. FLOPs.
Figure 1. Compared with state-of-the-art real-time object detectors on COCO [12], all variants of our proposed DEIMv2 (S, M, L, X)
achieve superior performance in terms of average precision (AP) while maintaining similar parameters and less computational cost.
Abstract
Driven by the simple and effective Dense O2O, DEIM
demonstrates faster convergence and enhanced perfor-
mance. In this work, we extend it with DINOv3 features,
resulting in DEIMv2. DEIMv2 spans eight model sizes from
X to Atto, covering GPU, edge, and mobile deployment. For
the X, L, M, and S variants, we adopt DINOv3-pretrained /
distilled backbones and introduce a Spatial Tuning Adapter
(STA), which efficiently converts DINOv3’s single-scale
output into multi-scale features and complements strong se-
mantics with fine-grained details to enhance detection. For
ultra-lightweight models (Nano, Pico, Femto, and Atto),
we employ HGNetv2 with depth and width pruning to meet
strict resource budgets. Together with a simplified decoder
and an upgraded Dense O2O, this unified design enables
DEIMv2 to achieve a superior performance–cost trade-off
across diverse scenarios, establishing new state-of-the-art
results. Notably, our largest model, DEIMv2-X, achieves
57.8 AP with only 50.3M parameters, surpassing prior X-
scale models that require over 60M parameters for just 56.5
AP . On the compact side, DEIMv2-S is the first sub-10M
model (9.71M) to exceed the 50 AP milestone on COCO,
reaching 50.9 AP . Even the ultra-lightweight DEIMv2-Pico,
with just 1.5M parameters, delivers 38.5 AP—matching
YOLOv10-Nano (2.3M) with∼50% fewer parameters.
1. Introduction
Real-time object detection [6, 18, 22, 29] is a critical
component in many practical applications, including au-
tonomous driving [11], robotics [16], and industrial defect
detection [8]. Achieving a good balance between detec-
tion performance and computational efficiency remains a
key challenge, especially for lightweight models suitable
for edge and mobile devices.
Among prevailing real-time detectors, DETR-based
methods are increasingly favored for their end-to-end na-
ture. Despite the robust feature representations offered by
DINOv3 [21] in diverse scenarios, the feasibility and effec-
arXiv:2509.20787v4  [cs.CV]  26 Jan 2026

## Page 2

Table 1.Architectural details and main results of DEIMv2 variants.We report the configurations for different variants and the final
Average Precision (AP) on the COCO benchmark.
Variant Backbone Hidden Dimension Layers #Scales #Query #Params GFLOPs Latency APModel Adapterd Back. dEnc. dDec. #Back. #Dec.
X ViT-S+ STA 384 256 256 12 6 3 300 50.26 151.6 13.75 57.8
L ViT-S STA 384 256 256 12 4 3 300 32.18 96.32 10.47 56.0
M ViT-T+ STA 256 256 256 12 4 3 300 18.11 52.20 8.80 53.0
S ViT-T STA 192 192 192 12 4 3 300 9.71 25.62 5.78 50.9
Nano HGv2-B0 - 1024 128 128 5 3 2 300 3.57 6.86 2.32 43.0
Pico HGv2-P - 512 112 112 4 3 2 200 1.51 5.15 2.14 38.5
Femto HGv2-F - 256 96 96 3 3 2 150 0.96 1.67 1.91 31.0
Atto HGv2-A - 256 64 64 3 3 2 100 0.49 0.76 1.61 23.8
tiveness of integrating it into DETR-based models have yet
to be fully investigated.
In this work, we introduce DEIMv2, a real-time object
detector built upon our previous DEIM [7] pipeline and en-
hanced with DINOv3 [21] features. DEIMv2 employs of-
ficial DINOv3-pretrained backbones (ViT-Small and ViT-
Small+) for its largest variants (L and X sizes) to maxi-
mize feature richness, while its S and M variants leverage
ViT-Tiny and ViT-Tiny+ backbones distilled from DINOv3,
carefully balancing performance and efficiency. To address
ultra-lightweight scenarios, we further introduce four spe-
cialized variants: Nano, Pico, Femto, and Atto, extending
DEIMv2’s scalability across a wide spectrum of computa-
tional budgets.
To better leverage the strong feature representations of
DINOv3, pretrained on large-scale data, under real-time
constraints, we design the Spatial Tuning Adapter (STA).
Operating in parallel with DINOv3, STA efficiently con-
verts its single-scale outputs into the multi-scale features re-
quired for object detection in a parameter-free manner. Si-
multaneously, it performs fast downsampling of the input
image to provide fine-grained, multi-scale detail features
with very small receptive fields, complementing DINOv3’s
strong semantics.
We further simplify the decoder by drawing on advances
from the Transformer community. Specifically, we replace
the conventional FFN and LayerNorm with SwishFFN [20]
and RMSNorm [27], both of which have been shown to be
efficient without significantly affecting performance. We
additionally note that object query locations change mini-
mally during iterative refinement, motivating sharing query
position embeddings across all decoder layers. Beyond this,
we enhance Dense O2O by introducing object-level Copy-
Blend augmentation, which increases effective supervision
and further improves model performance.
Extensive experiments on COCO [12] demonstrate that
DEIMv2 achieves state-of-the-art performance across mul-
tiple model scales, which can be seen in Figure 1. De-
spite its simplicity, the family of DEIMv2 exhibits strong
performance. For instance, our largest variant, DEIMv2-
X, achieves 57.6 AP on COCO with only 50.3M parame-
ters, surpassing prior best X-scale detectors DEIM-X that
requires over 60M parameters yet attains only 56.5 AP. At
the smaller end, DEIMv2-S establishes a notable milestone
as the first model with fewer than 10M parameters to exceed
50 AP, highlighting the effectiveness of our design at com-
pact scales. Furthermore, our ultra-lightweight DEIMv2-
Pico, with merely 1.5M parameters, attains 38.5 AP, match-
ing the performance of YOLOv10-Nano (2.3M parame-
ters) while reducing parameter count by∼50%, thereby
redefining the efficiency–accuracy frontier at the extreme
lightweight regime.
Our work highlights how DINOv3 [21] features can be
effectively adapted for real-time object detection and pro-
vides a versatile framework spanning ultra-lightweight to
high-performance models. To our knowledge, this is the
first work in real-time object detection to simultaneously
address such a wide range of deployment scenarios.
The main contributions of this work are summarized as
follows:
• We present DEIMv2, which offers eight model sizes cov-
ering GPU, edge, and mobile deployment.
• For larger models, we leverage DINOv3 for strong se-
mantic features and introduce the STA to efficiently inte-
grate them into real-time object detection.
• For ultra-lightweight models, we leverage expert knowl-
edge to effectively prune the depth and width of
HGNetv2-B0, meeting strict computational constraints.
• Beyond backbone, we further simplify the decoder and
upgrade Dense O2O, pushing the performance bound-
aries even further.
• Finally, we demonstrate on COCO that DEIMv2 outper-
forms existing state-of-the-art methods across all resource
settings, establishing new SOTA results.
2. Method
Overall architecture.Our overall architecture follows
the design of RT-DETR [14], comprising a backbone, a hy-

## Page 3

Conv3x3
S=2
MaxPool
S=2
Conv3x3
S=2
Conv3x3
S=2
Conv3x3
S=2
Spatial T uningAdapter
Patch
Embed
DINOv3
… …
Attention
Block1
Attention
Block5
Attention
Block11
Attention
Block8
Bilinear
Concat
 Conv1x1
Bi-Fusion
details
context…
𝔹 𝔹 𝔹
𝔹
1/8 1/16 1/32
Figure 2.Backbone design of our ViT-based variants. We integrate DINOv3 with the proposed Spatial Tuning Adapter (STA).
brid encoder, and a decoder. As shown in Table 1, for the
mainstream variants X, L, M, and S, the backbone is based
on DINOv3 with our proposed Spatial Tuning Adapter
(STA), while the remaining variants use HGNetv2 [1].
Multi-scale features from the backbone are first processed
by the encoder to produce initial detection results and select
the top-Kcandidate bounding boxes. The decoder itera-
tively refines these candidates to generate the final predic-
tions.
ViT-based variants.For the larger DEIMv2 variants (S,
M, L, X), we carefully design the backbones around the Vi-
sion Transformer [3] (ViT) family, balancing model capac-
ity with efficiency. For L and X, we leverage two public
DINOv3 models [21]: ViT-Small and ViT-Small+, which
provide strong semantic representations with 12 layers and
a 384-dimensional hidden size. For the lighter S and M
variants, we distill compact backbones, ViT-Tiny and ViT-
Tiny+, directly from ViT-Small DINOv3, preserving the 12-
layer depth while reducing the hidden dimensions to 192
and 256. This design delivers a smooth scaling path across
S→M→L→X, ensuring that each variant maintains
competitive accuracy while adapting to different efficiency
requirements.
HGNetv2-based variants.HGNetv2 [1], developed by
the Baidu PaddlePaddle team, is widely used in real-
time DETR frameworks for its efficiency—for example,
D-FINE [17] adopts the full HGNetv2 series as its back-
bone. In our ultra-lightweight DEIMv2 models (Nano,
Pico, Femto, and Atto), we also build on HGNetv2-B0,
but progressively prune its depth and width to meet differ-
ent parameter budgets. Specifically, the Pico backbone re-
moves the fourth stage of B0, keeping outputs only up to
1/16 scale. Femto further reduces the number of blocks in
Pico’s last stage from two to one. Atto goes a step further
by shrinking the channels of that last block from 512 to 256.
Spatial Tuning Adapter.To better adapt DINOv3 fea-
tures for real-time object detection, we propose the Spa-
tial Tuning Adapter (STA), as illustrated in Fig. 2. STA
is a fully convolutional network that integrates an ultra-
lightweight feedforward network for extracting fine-grained
multi-scale details, together with a Bi-Fusion operator that
further strengthens feature representations from DINOv3.
DINOv3 is based on a ViT backbone, which naturally
produces single-scale (1/16) dense features. In object de-
tection, however, objects vary widely in size, and multi-
scale features are one of the most effective ways to im-
prove performance. To this end, ViTDet [10] introduced
the Feature2Pyramid module, which generates multi-scale
features from the final ViT output using deconvolution. In
contrast, our STA is even simpler: we directly resize the
1/16-scale features from several ViT blocks (e.g., the 5th,
8th, and 11th) into multiple scales via parameter-free bi-
linear interpolation. These multi-scale features are further
enhanced by the Bi-Fusion operator consisting of1×1con-
volution with an ultra-lightweight CNN designed to extract
fine-grained details and complement DINOv3’s output fea-
tures. This design achieves an excellent trade-off between
efficiency and accuracy, making it well-suited for real-time
detection.
Efficient Decoder.We enhance the standard deformable
attention decoder [31] by incorporating several efficiency-
oriented techniques widely adopted in the Transformer
community, achieving a favorable performance–cost trade-
off. Specifically, we integrate SwiGLUFFN [20]
to strengthen nonlinear representation capacity, RM-
SNorm [27] to stabilize and accelerate training effi-
ciently. Noting that object query locations undergo mini-
mal changes during iterative refinement, we further propose

## Page 4

Table 2.Detailed training hyperparameters in DEIMv2.Back. denotes the Backbone. We useLocal Lossto denote the Fine-Grained
Localization (FGL) Loss and the Decoupled Distillation Focal (DDF) Loss in D-FINE [17].
HyperParams X L M S Nano Pico Femto Atto
Resolution 640 640 640 640 640 640 416 320
Weigt Decay 1.25e-4 1.25e-4 1e-4 1e-4 1e-4 1e-4 1e-4 1e-4
Base LR 5e-4 5e-4 5e-4 5e-4 8e-4 1.6e-3 1.6e-3 2e-3
Min LR 2.5e-4 2.5e-4 2.5e-4 2.5e-4 8e-4 8e-4 8e-4 1e-3
Back. LR 1e-6 1.25e-5 2.5e-5 2.5e-5 4e-4 8e-4 8e-4 1e-3
Back. MinLR 5e-7 6.25e-6 1.25e-5 1.25e-5 4e-4 4e-4 4e-4 5e-4
Local Loss ✓ ✓ ✓ ✓ ✓
Epochs 50 60 90 120 148 468 468 468
Mosaicprob 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.3
Mosaicepochs [4, 29] [4, 34] [4, 49] [4, 64] [4, 78] [4, 250] [4, 250] [4, 250]
MixUpprob 0.5 0.5 0.5 0.5 0.5 0.0 0.0 0.0
MixUpepochs [4, 29] [4, 34] [4, 49] [4, 64] [4, 78] [ ] [ ] [ ]
CopyBprob 0.5 0.5 0.5 0.5 0.4 0.0 0.0 0.0
CopyBepoch [4, 50] [4, 60] [4, 90] [4, 120] [4, 78] [ ] [ ] [ ]
sharing a single position embedding across all decoder lay-
ers, eliminating redundant computation.
Enhanced Dense O2O.In our previous DEIM [7], we
proposed Dense O2O, which increases the number of ob-
jects per training image to provide stronger supervision,
improving convergence and performance. Its effectiveness
was initially demonstrated using image-level augmentations
such as Mosaic and MixUp [28]. In DEIMv2, we further
explore Dense O2O at the object level with Copy-Blend,
which adds new objects without their backgrounds. Unlike
Copy-Paste [4], which fully overwrites the target region,
Copy-Blend blends new objects with the image, better suit-
ing our scenario and consistently improving performance.
Training setting and loss.The overall optimization ob-
jective is a weighted sum of five components: Matchability-
Aware Loss (MAL) [7], Fine-Grained Localization (FGL)
Loss [17], Decoupled Distillation Focal (DDF) Loss [17],
BBox Loss (L1), and GIoU Loss [19]. The total loss is de-
fined as:
Ltotal =λ1Lmal +λ 2Lf gl+λ 3Ldd f
+λ 4Lbbox +λ 5Lgiou
(1)
with weights set toλ 1 = 1.0,λ2 = 0.15,λ3 = 1.5,λ4 = 5,
andλ 5 = 2across all experiments.
We summarize the training hyperparameters in Table 2,
covering input resolution, learning rate, training epochs,
and Dense O2O settings. An interesting observation is that
applying FGL and DDF losses to ultra-lightweight models
degrades performance. We attribute this to their limited ca-
pacity and inherently weaker baseline accuracy, which re-
duces the effectiveness of self-distillation. Consequently,
we exclude these two components (i.e., the local loss) from
training the Pico, Femto, and Atto variants.
3. Experiments
Comparison to state-of-the-art real-time object detec-
tors.Table 3 summarizes the performance of DEIMv2
across the S, M, L, and X variants, demonstrating substan-
tial improvements over prior state-of-the-art detectors. For
instance, the largest variant, DEIMv2-X, attains 57.8 AP
with only∼50M parameters and 151 GFLOPs, surpass-
ing the previous best DEIM-X (56.5 AP with 62M param-
eters and 202 GFLOPs). This demonstrates that DEIMv2
can deliver superior accuracy with both fewer parame-
ters and lower computational cost. At the smaller end,
DEIMv2-S sets a new milestone as the first model with
fewer than 10M parameters to exceed the 50 AP thresh-
old on COCO, achieving 50.9 AP with only 11M parame-
ters and 26 GFLOPs. This marks a clear improvement over
the prior DEIM-S (49.0 AP with 10M parameters), while
requiring nearly the same model size. While CNN-based
backbones are generally more hardware-friendly, our ViT-
based backbone achieves a lightweight design with fewer
parameters and lower FLOPs, offering better scalability and
deployment flexibility. It is worth noting that the latency of
the proposed methods has not been optimized. Techniques
such as Flash Attention [2], as in Yolov12 [22], could fur-
ther accelerate inference. Overall, the reduced FLOPs high-
light the potential of ViT-based backbones to achieve low-

## Page 5

Table 3.Comparison with real-time object detectors on COCO [12]val2017, sorted by parameter size.
Model #Epochs #Params. GFLOPs Latency (ms) APval APval
50 APval
75 APval
S APval
M APval
L
YOLOv10-S [23] 500 7 22 2.52 46.3 63.0 50.4 26.8 51.0 63.8
YOLOv9-S [25] 500 7 26 8.05 46.8 61.8 48.6 25.7 49.9 61.0
YOLOv12-S-turbo [22] 600 9 19 6.28 47.5 64.1 - - - -
YOLO11-S [6] 500 9 22 7.05 46.6 63.4 50.3 28.7 51.3 64.1
D-FINE-S [17] 120 10 25 3.66 48.5 65.6 52.6 29.1 52.2 65.4
DEIM-S [7] 120 10 25 3.66 49.0 65.9 53.1 30.4 52.6 65.7
YOLOv8-S [5] 500 11 29 6.97 44.9 61.8 48.6 25.7 49.9 61.0
DEIMv2-S 120 10 26 5.78 50.9 68.3 55.1 31.4 55.3 70.3
YOLOv10-M [23] 500 15 59 4.70 51.1 68.1 55.8 33.8 56.5 67.0
D-FINE-M [17] 120 19 57 5.91 52.3 69.8 56.4 33.2 56.5 70.2
DEIM-M [7] 90 19 57 5.91 52.7 70.0 57.3 35.3 56.7 69.5
RT-DETRv2-S [14] 120 20 60 4.61 48.1 65.1 57.4 36.1 57.9 70.8
YOLOv12-M-turbo [22] 600 20 60 8.44 52.6 69.5 - - - -
YOLO11-M [6] 500 20 68 9.02 51.2 67.9 55.3 33.0 56.7 67.5
YOLOv9-M [25] 500 20 76 10.18 51.4 67.2 54.6 32.0 55.7 66.4
Gold-YOLO-S [24] 300 22 46 6.96 46.4 63.4 - 25.3 51.3 63.6
DEIMv2-M 90 18 52 8.80 53.0 70.2 57.6 34.2 57.4 71.5
YOLOv10-L [23] 500 24 120 7.38 53.2 70.1 58.1 35.8 58.5 69.4
YOLO11-L [6] 500 25 87 10.41 53.4 70.1 58.2 35.6 59.1 69.2
YOLOv9-C [25] 500 25 102 10.76 53.0 70.2 57.8 36.2 58.5 69.3
YOLOv8-M [5] 500 26 79 9.46 50.2 67.2 54.6 32.0 55.7 66.4
YOLOv12-L-turbo [22] 600 27 82 10.45 54.0 70.6 - - - -
YOLOv10-X [23] 500 30 160 10.47 54.4 71.3 59.3 37.0 59.8 70.9
D-FINE-L [17] 72 31 91 8.15 54.0 71.6 58.4 36.5 58.0 71.9
DEIM-L [7] 50 31 91 8.15 54.7 72.4 59.4 36.9 59.6 71.8
RT-DETRv2-M [14] 120 31 92 6.91 49.9 67.5 58.6 35.8 58.6 72.1
Gold-YOLO-M [24] 300 41 88 9.21 51.1 68.5 - 32.3 56.1 68.6
RT-DETRv2-L [14] 72 42 136 9.29 53.4 71.6 57.4 36.1 57.9 70.8
YOLOv8-L [5] 500 43 165 12.20 52.9 69.8 57.5 35.3 58.3 69.8
DEIMv2-L 60 32 96 10.47 56.0 73.4 60.9 37.5 60.8 75.2
YOLOv9-E [25] 500 57 189 20.52 55.6 72.8 60.6 40.2 61.0 71.4
YOLO11-X [6] 500 57 195 15.53 54.7 71.6 59.5 37.7 59.7 70.2
YOLOv12-X-turbo [22] 600 59 185 15.79 55.7 72.2 - - - -
D-FINE-X [17] 72 62 202 12.90 55.8 73.7 60.2 37.3 60.5 73.4
DEIM-X [7] 50 62 202 12.90 56.5 74.0 61.5 38.8 61.4 74.2
YOLOv8-X [5] 500 68 257 15.89 53.9 71.0 58.7 35.7 59.3 70.7
Gold-YOLO-L [24] 300 75 152 12.31 53.3 70.9 - 33.8 58.9 69.9
RT-DETRv2-X [14] 72 76 259 13.88 54.3 72.8 58.8 35.8 58.8 72.1
DEIMv2-X 50 50 151 13.75 57.8 75.4 63.2 39.2 62.9 75.9
latency performance with proper optimization.
Interestingly, when comparing DINOv3-based DEIMv2
models with their previous DEIM counterparts under com-
parable parameter and FLOP budgets, the accuracy gains
primarily arise from improvements on medium and large
objects, while performance on small objects remains largely
unchanged. For instance, DEIMv2-S achieves 55.3 AP M
and 70.3 AP L, clearly surpassing DEIM-S (52.6 AP M and
65.7 APL), yet the small-object scores are nearly identical
(31.4 vs. 30.4 AP S). A similar trend is observed for larger
models: DEIMv2-X improves AP M from 61.4 to 62.8 and
APL from 74.2 to 75.9, while its small-object AP (39.2) re-
mains close to that of DEIM-M (38.8). These results indi-
cate that DEIMv2’s main advantage lies in enhancing the
representation and detection of medium-to-large objects,
whereas small-object detection remains a challenge across
scales. This observation further confirms that DINOv3 ex-
cels at capturing strong global semantics but has limited

## Page 6

Table 4.Comparison with real-time object detectors on COCO [12]val2017for ultra-light models.
Model Input Size Params (M) FLOPs (G) COCO AP
NanoDet-M[15] 416x416 1.0 0.7 23.5
DEIMv2-Atto 320x320 0.5 0.8 23.8
YOLOX-Nano[30] 416x416 0.9 1.1 25.8
PicoDet-S[26] 416x416 1.0 1.2 30.7
PP-YOLO-Tiny[13] 416x416 1.1 1.0 22.7
PicoDet-ShufflenetV2 1x[26] 416x416 1.2 1.5 30.0
DEIMv2-Femto 416x416 1.0 1.7 31.0
YOLOv10-N[23] 640x640 2.3 6.7 38.5
YOLOv8-N[5] 640x640 3.2 8.7 37.4
YOLOv6-3.0-N[9] 640x640 4.7 11.4 37.0
DEIMv2-Pico 640x640 1.5 5.2 38.5
YOLOv12-N [22] 640x640 2.6 6.5 40.6
DEIM-Nano [7] 640x640 3.6 6.9 43.0
D-FINE-Nano [17] 640x640 3.8 7.2 42.8
DEIMv2-Nano 640x640 3.6 6.9 43.0
ability to represent fine-grained details. Exploring ways
to better integrate DINOv3 features into real-time detectors
thus represents an interesting direction for future work.
Comparison to competitive ultra-light object detectors.
The ultra-light variants of DEIMv2 also exhibit strong per-
formance, as summarized in Table 4. DEIMv2-Atto, with
only 0.49M parameters, achieves performance compara-
ble to NanoDet-M despite its substantially smaller size.
Similarly, DEIMv2-Pico attains performance on par with
YOLOv10-N [23] while requiring less than half the parame-
ters. These results underscore the effectiveness of DEIMv2
in extremely compact regimes and highlight its suitability
for deployment on resource-constrained edge devices.
4. Conclusion
In this report, we introduced DEIMv2, a new generation
of real-time object detectors that combines the strong se-
mantic representations of DINOv3 with our lightweight
STA. Through careful design and scaling, DEIMv2 achieves
state-of-the-art performance across the full spectrum of
model sizes. At the high end, DEIMv2-X delivers 57.8
AP with significantly fewer parameters than previous large-
scale detectors. At the compact end, DEIMv2-S is the
first model of its size to surpass 50 AP, and the ultra-
lightweight DEIMv2-Pico matches YOLOv10-N while us-
ing over 50% fewer parameters. Together, these results
demonstrate that DEIMv2 is not only efficient but also
highly scalable, offering a unified framework that advances
the accuracy–efficiency frontier. This versatility makes
DEIMv2 well-suited for deployment in diverse scenarios,
ranging from resource-constrained edge devices to high-
performance detection systems, paving the way for broader
adoption of real-time detection in practical applications.
References
[1] Cheng Cui, Ruoyu Guo, Yuning Du, Dongliang He, Fu Li,
Zewu Wu, Qiwen Liu, Shilei Wen, Jizhou Huang, Xiaoguang
Hu, et al. Beyond self-supervision: A simple yet effective
network distillation alternative to improve backbones.arXiv,
2021. 3
[2] Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christo-
pher R ´e. Flashattention: Fast and memory-efficient exact
attention with io-awareness. InNeurIPS, 2022. 4
[3] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov,
Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner,
Mostafa Dehghani, Matthias Minderer, Georg Heigold, Syl-
vain Gelly, et al. An image is worth 16x16 words: Trans-
formers for image recognition at scale. InICLR, 2020. 3
[4] Golnaz Ghiasi, Yin Cui, Aravind Srinivas, Rui Qian, Tsung-
Yi Lin, Ekin D Cubuk, Quoc V Le, and Barret Zoph. Simple
copy-paste is a strong data augmentation method for instance
segmentation. InCVPR, 2021. 4
[5] Jocher Glenn. Yolov8.https://docs.ultralytics.
com/models/yolov8/, 2023. 5, 6
[6] Jocher Glenn. Yolo11.https://docs.ultralytics.
com/models/yolo11/, 2024. 1, 5
[7] Shihua Huang, Zhichao Lu, Xiaodong Cun, Yongjun Yu,
Xiao Zhou, and Xi Shen. Deim: Detr with improved match-
ing for fast convergence. InCVPR, 2025. 2, 4, 5, 6
[8] Muhammad Hussain. Yolo-v1 to yolo-v8, the rise of yolo
and its complementary nature toward digital manufacturing
and industrial defect detection. 2023. 1

## Page 7

[9] Chuyi Li, Lulu Li, Yifei Geng, Hongliang Jiang, Meng
Cheng, Bo Zhang, Zaidan Ke, Xiaoming Xu, and Xiangx-
iang Chu. Yolov6 v3.0: A full-scale reloading.arXiv, 2023.
6
[10] Yanghao Li, Hanzi Mao, Ross Girshick, and Kaiming He.
Exploring plain vision transformer backbones for object de-
tection. InECCV, 2022. 3
[11] Siyuan Liang, Hao Wu, Li Zhen, Qiaozhi Hua, Sahil Garg,
Georges Kaddoum, Mohammad Mehedi Hassan, and Keping
Yu. Edge yolo: Real-time intelligent object detection sys-
tem based on edge-cloud cooperation in autonomous vehi-
cles.IEEE Transactions on Intelligent Transportation Sys-
tems, 2022. 1
[12] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays,
Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence
Zitnick. Microsoft coco: Common objects in context. In
ECCV, 2014. 1, 2, 5, 6
[13] Xiang Long, Kaipeng Deng, Guanzhong Wang, Yang Zhang,
Qingqing Dang, Yuan Gao, Hui Shen, Jianguo Ren, Shumin
Han, Errui Ding, and Shilei Wen. Pp-yolo: An effective and
efficient implementation of object detector.arXiv, 2020. 6
[14] Wenyu Lv, Yian Zhao, Qinyao Chang, Kui Huang,
Guanzhong Wang, and Yi Liu. Rt-detrv2: Improved base-
line with bag-of-freebies for real-time detection transformer.
arXiv, 2024. 2, 5
[15] Rangi Lyu. Nanodet-plus.https://github.com/
RangiLyu / nanodet / releases / tag / v1 . 0 . 0 -
alpha-1, 2021. Version 1.0.0-alpha-1. 6
[16] Debapriya Maji, Soyeb Nagori, Manu Mathew, and Deepak
Poddar. Yolo-6d-pose: Enhancing yolo for single-stage
monocular multi-object 6d pose estimation. In3DV, 2024.
1
[17] Yansong Peng, Hebei Li, Peixi Wu, Yueyi Zhang, Xiaoyan
Sun, and Feng Wu. D-fine: Redefine regression task in detrs
as fine-grained distribution refinement. InICLR, 2024. 3, 4,
5, 6
[18] Joseph Redmon, Santosh Divvala, Ross Girshick, and Ali
Farhadi. You only look once: Unified, real-time object de-
tection. InCVPR, 2016. 1
[19] Hamid Rezatofighi, Nathan Tsoi, JunYoung Gwak, Amir
Sadeghian, Ian Reid, and Silvio Savarese. Generalized in-
tersection over union: A metric and a loss for bounding box
regression. InCVPR, 2019. 4
[20] Noam Shazeer. Glu variants improve transformer.arXiv
preprint arXiv:2002.05202, 2020. 2, 3
[21] Oriane Sim ´eoni, Huy V V o, Maximilian Seitzer, Federico
Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov,
Marc Szafraniec, Seungeun Yi, Micha ¨el Ramamonjisoa,
et al. Dinov3.arXiv, 2025. 1, 2, 3
[22] Yunjie Tian, Qixiang Ye, and David Doermann. Yolov12:
Attention-centric real-time object detectors. InNeurIPS,
2025. 1, 4, 5, 6
[23] Ao Wang, Hui Chen, Lihao Liu, Kai Chen, Zijia Lin, Jun-
gong Han, and Guiguang Ding. Yolov10: Real-time end-to-
end object detection. InNeurIPS, 2024. 5, 6
[24] Chengcheng Wang, Wei He, Ying Nie, Jianyuan Guo,
Chuanjian Liu, Yunhe Wang, and Kai Han. Gold-yolo: Ef-
ficient object detector via gather-and-distribute mechanism.
InNeurIPS, 2023. 5
[25] Chien-Yao Wang, I-Hau Yeh, and Hong-Yuan Mark Liao.
Yolov9: Learning what you want to learn using pro-
grammable gradient information.arXiv, 2024. 5
[26] Guanghua Yu, Qinyao Chang, Wenyu Lv, Chang Xu, Cheng
Cui, Wei Ji, Qingqing Dang, Kaipeng Deng, Guanzhong
Wang, Yuning Du, et al. Pp-picodet: A better real-time ob-
ject detector on mobile devices.arXiv, 2021. 6
[27] Biao Zhang and Rico Sennrich. Root mean square layer nor-
malization. InNeurIPS, 2019. 2, 3
[28] Hongyi Zhang. mixup: Beyond empirical risk minimization.
InICLR, 2017. 4
[29] Yian Zhao, Wenyu Lv, Shangliang Xu, Jinman Wei,
Guanzhong Wang, Qingqing Dang, Yi Liu, and Jie Chen.
Detrs beat yolos on real-time object detection. InCVPR,
2024. 1
[30] Ge Zheng, Liu Songtao, Wang Feng, Li Zeming, and Sun
Jian. Yolox: Exceeding yolo series in 2021.arXiv, 2021. 6
[31] Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang Wang,
and Jifeng Dai. Deformable detr: Deformable transformers
for end-to-end object detection. InICLR, 2021. 3
