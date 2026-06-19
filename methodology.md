# Methodology and Technical Reference

This document outlines the systematic methodology, experimental design, and mathematical verification procedures implemented within this dual-backend machine learning pipeline. The pipeline supports both native **Ultralytics** and foundation-based **LightlyTrain** workflows.

---

## 1. Experimental Objective
The objective of this pipeline is to provide a mathematically rigorous framework for training, distilling, optimizing, and evaluating deep learning object detection models on high-resolution drone/aerial datasets (specifically tree-crown detection). The pipeline enforces complete experimental determinism and statistical validity across multi-seed production trials and two-stage hyperparameter sweeps.

---

## 2. Supported and Verified Models

The following model variants are verified to work with the respective backends:

| Backend | Model Type | Configuration / String Identifier | Verification Status |
| :--- | :--- | :--- | :--- |
| **Ultralytics** | YOLOv11 | `yolo11n.pt`, `yolo11s.pt`, `yolo11m.pt`, `yolo11l.pt`, `yolo11x.pt` | Verified |
| **Ultralytics** | YOLO26 | `yolo26n.pt`, `yolo26s.pt`, `yolo26m.pt`, `yolo26l.pt`, `yolo26x.pt` | Verified |
| **Ultralytics** | YOLOv12 | `yolov12n.pt`, `yolov12s.pt`, `yolov12m.pt`, `yolov12l.pt`, `yolov12x.pt` | Verified |
| **Ultralytics** | RT-DETR | `rtdetr-l.pt`, `rtdetr-x.pt` | Verified |
| **LightlyTrain** | YOLOv11 DINOv3 | `ultralytics/yolo11n.yaml` (DINOv3 backbone + YOLOv11 Head) | Verified |
| **LightlyTrain** | YOLO26 DINOv3 | `ultralytics/yolo26n.yaml` (DINOv3 backbone + YOLO26 Head) | Verified |
| **LightlyTrain** | YOLOv12 DINOv3 | `ultralytics/yolov12n.yaml` (DINOv3 backbone + YOLOv12 Head) | Verified |
| **LightlyTrain** | RT-DETR / LTDETR | `dinov3/convnext-large-ltdetr-coco` (DINOv3 ConvNeXt-L + LTDETR Head) | Verified |
| **LightlyTrain** | Pretraining Backbones | `dinov3/convnext-large`, `dinov3/vitl16`, `dinov3/vitl16-sat493m` (Meta SAT-493M satellite ViT) | Verified |
| *Unsupported* | DEIMv2 | `deimv2` | *Skipped with warning* |

*Note: DEIMv2 is not supported by the underlying frameworks and is automatically skipped with a validation warning in the training loops to prevent execution failures.*

---

## 3. Data Subsetting & Isolation Control

### 3.1. Non-Biased Dataset Subsetting
Fractional dataset subsetting (e.g. `--fraction 0.1`) is frequently biased by sequential file indexing or filesystem sorting, which groups images by capture time or geographic region (spatial-temporal capture bias). To address this:
1. Files are sorted alphabetically to establish a stable base.
2. A deterministic pseudo-random number generator (PRNG) seeded with a constant value (`seed = 42`) extracts a uniform sample:
   $$\text{Sample} = \text{sample\_deterministic}(\mathcal{D}, f)$$
   where $\mathcal{D}$ is the dataset and $f$ is the target fraction.
3. This guarantees that subset experiments represent a statistically unbiased sample of the overall dataset.

### 3.2. Hash-Based Ground Truth Partitioning
To save computational overhead, YOLO annotations are converted to COCO ground-truth JSON formats and cached. However, changes in dataset splits or fractional configurations can cause cache pollution if file naming conventions overlap. 
We resolve this by computing a unique 8-character MD5 hash representing the absolute directory path of the partition split:
$$\text{Hash} = \text{MD5}(\text{abspath}(\mathcal{D}_{\text{split}}))[:8]$$
The ground truth file is then compiled and cached as:
$$\text{coco\_gt\_}\{\text{split}\}\_\{\text{hash}\}.\text{json}$$
This prevents cross-validation contamination and guarantees that predictions are evaluated against their exact ground-truth partition.

---

## 4. DINOv3 Knowledge Distillation & Pretraining

### 4.1. Self-Supervised Knowledge Distillation (LVD-1689M & SAT-493M)
To improve downstream detection capabilities, a student network (e.g., `yolo26n`) is pretrained using self-supervised teacher-student representation alignment.
* **Teacher Network**: Meta's vision foundation model DINOv3 Vision Transformer (`dinov3/vitl16` or the satellite-pretrained `dinov3/vitl16-sat493m`).
* **Student Network**: The backbone of the target detector.
* **Pretraining Objective**: The student's representation space is trained to mimic the global and spatial feature representations of the teacher under heavy augmentations:
  $$\mathcal{L}_{\text{distill}} = (1 - \lambda)\mathcal{L}_{\text{global}}(\mathbf{z}_s, \mathbf{z}_t) + \lambda \mathcal{L}_{\text{dense}}(\mathbf{F}_s, \mathbf{F}_t)$$
  where $\mathbf{z}$ and $\mathbf{F}$ are global embeddings and spatial feature maps, respectively.
* **Deterministic Initialization**: The distillation routine ([run_distillation_pretrain](file:///C:/Users/emilb/_trainer_lightly/run_training.py#L183-L209)) enforces a constant seed (`seed = 42`) for random augmentations and weight initialization. This ensures that the distilled starting weights are strictly identical across independent experiments.

### 4.2. Hugging Face Backbone Gating and Stream Downloader
To evaluate custom vision transformers from Hugging Face (e.g. `facebook/dinov3-vitl16-pretrain-lvd1689m` and `facebook/dinov3-vitl16-pretrain-sat493m`), a programmatic download and conversion flow is implemented in [eval_utils.py](file:///C:/Users/emilb/_trainer_lightly/eval_utils.py):
1. **Token Storage**: The private Hugging Face access token is securely loaded from [.env](file:///C:/Users/emilb/_trainer_lightly/.env) at the workspace root as `HF_TOKEN`. This allows accessing gated models without exposing credentials in the code or CLI parameters.
2. **Streaming Requests Download**: Rather than using `huggingface_hub`'s standard helper which can deadlock on Windows when lock files are orphaned or when symlinks are degraded, we download the raw weights directly from Hugging Face's resolve API via `requests` stream iteration in unbuffered chunks.
3. **Real-time Logging**: Progress is logged to stdout/stderr in 10 MB increments to prevent tasks from appearing hung during large downloads (1.2 GB+), which can take up to 40 minutes on restricted network speeds.
4. **On-the-fly Weights Conversion & Space Optimization**:
   - The downloader downloads `model.safetensors` to a temporary path under `hf_cache/`.
   - It parses the safetensors file via `safetensors.torch.load_file` to construct a standard PyTorch state dict.
   - It saves it as a PyTorch checkpoint `.pt` under `hf_cache/` for ingestion by `lightly_train`.
   - It automatically deletes the raw `.safetensors` file from local storage upon completion, reclaiming 1.2 GB of disk space per model.

### 4.3. ViT-L Backbones and LTDETR Coupling in LightlyTrain
For object detection, ViT backbones must be mapped to compatible heads. In `lightly_train`, Vision Transformer backbones (like `facebook/dinov3-vitl16-pretrain-sat493m` or `facebook/dinov3-vitl16-pretrain-lvd1689m`) are automatically mapped to LTDETR heads (`dinov3/vitl16-ltdetr`) due to feature map compatibility. Standard YOLO models are trained via their custom configurations (`ultralytics/yolo11n.yaml`), which employ YOLO's native FPN/PAN convolutional backbones rather than direct transformer backbones. For optimal geospatial performance, coupling the satellite ViT-L/16 backbone with the LTDETR head is the recommended architectural layout.

---

## 5. Two-Stage Hyperparameter Optimization (HPO)

To search the parameter space systematically, the hyperparameter sweep is divided into two distinct, isolated phases managed through Weights & Biases (W&B) and custom YAML sweep schedules.

```
          [Stage 1: Augmentation Tuning]
                        │
                        ▼  (Identifies optimal augmentations)
         Retrieve Best Phase 1 Run Config
                        │
                        ▼  (Loads augmentations & locks them)
             [Stage 2: Learning HPO]
```

### 5.1. Phase 1: Data Augmentation Tuning
* **Objective**: Optimize data augmentations to match dataset characteristics (e.g., aerial perspective, occlusion, tree crown scale variations).
* **Locked Parameters**: Optimizer settings are held constant at established baselines (SGD, $\text{lr}_0 = 0.0054$, $\text{momentum} = 0.947$, etc.).
* **Free Parameters**: Custom Albumentations probability boundaries (spatial flips, color jitter, coarse dropouts, blur/noise levels) and YOLO native parameters (mosaic, copy-paste).

### 5.2. Phase 2: Learning Rate & Optimizer HPO
* **Objective**: Find the optimal learning rate schedule and optimization algorithm for the model.
* **Locked Parameters**: Custom augmentations are set to the optimal configurations retrieved programmatically from Phase 1.
* **Free Parameters**: Learning rate ($\text{lr}_0$), final learning rate fraction ($\text{lrf}$), momentum, weight decay, warmup epochs, and optimizer type (`AdamW`, `Adam`, `MuSGD`).

### 5.3. Seeding for HPO Strictness
In typical sweep configurations, trial execution has a stochastic component due to random initializations. To ensure that hyperparameter improvements are due to hyperparameter settings rather than random initialization variance, every sweep trial agent executes `set_reproducibility(42)`. This locks weight initializations, data sequencing, and random seeds across all trials within the sweep.

---

## 6. Strict COCO Evaluation

Many modern frameworks perform validation utilizing internal indices, leading to evaluation bias. The evaluation module [eval_utils.py](file:///C:/Users/emilb/_trainer_lightly/eval_utils.py) implements strict pycocotools verification:
1. Coordinates are converted from YOLO normalized formats $(x_c, y_c, w_n, h_n)$ into absolute COCO pixel bounding boxes $[x_{\text{min}}, y_{\text{min}}, w_p, h_p]$:
   $$x_{\text{min}} = \left(x_c - \frac{w_n}{2}\right) \times W_{\text{img}}$$
   $$y_{\text{min}} = \left(y_c - \frac{h_n}{2}\right) \times H_{\text{img}}$$
   $$w_p = w_n \times W_{\text{img}}, \quad h_p = h_n \times H_{\text{img}}$$
2. All inference predictions are mapped back to their unique integer ground truth indices.
3. COCO evaluations compute standardized Mean Average Precision (mAP) and Mean Average Recall (mAR) over 12 metrics, including scale-wise subdivisions (AP_small, AP_medium, AP_large).

---

## 7. Statistical Aggregation and Variance Inference

To establish statistical confidence, production models are trained across three independent seeds ($N=3$): `42`, `100`, and `999`.

### 7.1. Sample Standard Deviation
The sample standard deviation $\sigma$ is computed using Bessel's correction to adjust for bias in small-sample estimators:
$$\sigma = \sqrt{\frac{1}{N - 1} \sum_{i=1}^{N} (x_i - \bar{x})^2}$$
where $\bar{x}$ is the sample mean.

### 7.2. Standard Error of the Mean (SEM)
The Standard Error of the Mean represents the precision of the sample mean estimate:
$$SEM = \frac{\sigma}{\sqrt{N}}$$
Reporting metrics as $\text{Mean} \pm \text{SEM}$ is the scientific standard for small sample sizes ($N=3$) because it represents the variance of the sample mean estimate rather than the population spread.

### 7.3. Configurable Error Metrics
The aggregation module [plot_results.py](file:///C:/Users/emilb/_trainer_lightly/plot_results.py) allows the user to configure the plotted error metrics. Using the `--error-metric` CLI argument, users can toggle between:
* `sem` (Default): Evaluates the precision of the sample mean.
* `std`: Evaluates the distribution variance across independent runs.

---

## 8. Unified Baseline Benchmark Suite Orchestration

To support comparative vision transformer and convolutional object detector evaluations, the pipeline includes an automated orchestration wrapper [run_baseline_benchmark.py](file:///C:/Users/emilb/_trainer_lightly/run_baseline_benchmark.py) exposed via the CLI task `pixi run train-baseline`.

### 8.1. Model Ingestion and Backend Mapping
The orchestrator executes standard-settings training runs across the following baseline models:
* **Ultralytics Backend**:
  * YOLO11s (`yolo11s.pt`)
  * YOLO26s (`yolo26s.pt`)
  * YOLOv12s (`yolov12s.pt`)
  * RT-DETR-L (`rtdetr-l.pt`)
* **LightlyTrain Backend (LTDETR Head)**:
  * DINOv3 ViT-L/16 with Satellite weights (`facebook/dinov3-vitl16-pretrain-sat493m`)
  * DINOv3 ViT-L/16 with Web/LVD weights (`facebook/dinov3-vitl16-pretrain-lvd1689m`)

### 8.2. Deterministic Seeding and Replication
Each of the 6 models is trained and evaluated across 3 independent seeds (`42`, `100`, `999`), resulting in a total of 18 training runs. In each run:
1. Weight initialization, data shuffling, and random augmentations are locked via `set_reproducibility(seed)`.
2. Hardware settings (AMP, workers, batch size) are kept identical to ensure mathematical fairness.
3. Training processes and metrics are fully logged to Weights & Biases (W&B).

### 8.3. Strict Evaluation and Folder Localization
To isolate baseline benchmarks from sweeps and sweep production runs:
1. All check-pointed weights and intermediate logs are outputted to the distinctive runs folder `runs/baseline/`.
2. During post-training evaluation, COCO validation metrics are exported to the distinctive folder `evaluation_results/baseline/`.
3. The results aggregator `plot_results.py` is invoked with `--eval-dir evaluation_results/baseline` to compile the final comparative figures and markdown summary table directly inside that folder.

---

## 9. Visual Verification and Interactive Inspection via FiftyOne

Standard quantitative COCO metrics do not reveal localization failure modes (such as false positives on overlapping tree crowns or bounding box coordinates drifts). To enable qualitative verification, the visualizer [visualize_fiftyone.py](file:///C:/Users/emilb/_trainer_lightly/visualize_fiftyone.py) maps quantitative results into an interactive visual inspection space:
1. **FiftyOne Dataset Construction**: The script parses `dataset.yaml` to identify splitting paths and corresponding annotation text files.
2. **Ground Truth Conversion**: Center-based normalized YOLO labels $(x_c, y_c, w_n, h_n)$ are translated to FiftyOne corner-based normalized bounding boxes $[x_{\text{min}}, y_{\text{min}}, w_n, h_n]$:
   $$x_{\text{min}} = x_c - \frac{w_n}{2}, \quad y_{\text{min}} = y_c - \frac{h_n}{2}$$
3. **Predictions Parsing**: The pycocotools output `predictions.json` is loaded. Absolute pixel COCO bboxes $[x_p, y_p, w_p, h_p]$ are converted to normalized corner-based coordinates based on individual image dimensions $(W, H)$:
   $$x_{\text{min}} = \frac{x_p}{W}, \quad y_{\text{min}} = \frac{y_p}{H}, \quad w_n = \frac{w_p}{W}, \quad h_n = \frac{h_p}{H}$$
4. **App Session Launch**: Samples are populated with `ground_truth` and `predictions` detection fields (retaining scores/confidence attributes), launching a local interactive FiftyOne session.

---

## 10. Weights & Biases Pipeline Logging & Project Configuration

To ensure standardized experiment tracking and prevent logs fragmentation, all baseline training runs, sweeps, and validation evaluations are managed dynamically under a centralized project layout:
1. **Centralized Settings**: Project and entity credentials are configured inside [config.py](file:///C:/Users/emilb/_trainer_lightly/config.py#L4-L9) under `PipelineConfig`:
   * `entity`: The W&B username or organization account mapping (defaults to `brezemil`).
   * `project`: The W&B project dashboard destination (defaults to `_trainer`).
2. **Sensible Run Naming**: Standardized string keys are generated for W&B logging runs to group experiments cleanly. In baseline runs, names follow `{model}_seed_{seed}` (e.g., `yolo11s_seed_42` or `facebook--dinov3-vitl16-pretrain-sat493m_seed_100`). In sweeps, trials are tagged with their unique sweep ID and Phase parameters (`Phase 1: Augmentation` vs `Phase 2: HPO`) to enable instant filtering and grouping.


