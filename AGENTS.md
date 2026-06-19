# Agent Directives

This document provides critical operational rules, grounding context, and a documentation lookup index for all AI coding assistants and agents working in this workspace.

---

## 1. Local Documentation Library

All documentation is cached locally under `./agy/docs/`. **Before generating, refactoring, or reviewing code** that uses any of the libraries below, you **must** read and cross-reference the relevant files. Do not rely on parametric knowledge or guess API signatures — they may be outdated or incorrect.

| Library / Source | Local Path | Description |
|:---|:---|:---|
| **LightlyTrain** | `./agy/docs/lightly_train/` | SSL pretraining, distillation, object detection, train/pretrain settings, Python API |
| **Ultralytics** | `./agy/docs/ultralytics/` | YOLO11/YOLO26/YOLOv12/RT-DETR training, validation, export, augmentation, HPO |
| **Weights & Biases** | `./agy/docs/wandb/` | Experiment tracking, sweeps, artifacts, runs, public API, SDK reference |
| **FiftyOne** | `./agy/docs/fiftyone/` | Dataset curation, visual analysis, model evaluation, embeddings, App visualization |
| **SAHI** | `./agy/docs/sahi/` | Slices Aided Hyper Inference (tiled inference), slicing config, post-processing |
| **DINOv3 Paper** | `./agy/docs/dinov3.md` | Meta's DINOv3 foundation model architecture, ViT-L/16, satellite pretraining, LTDETR head |
| **DEIMv2 Paper** | `./agy/docs/deimv2.md` | Real-time object detection with DINOv3 integration (DEIM architecture) |

---

## 2. Pixi Environment Execution & Verification

You are operating inside a locked Pixi environment containing a high-performance Geospatial CV and YOLO stack. 

### Terminal Tooling Rules
* **Strict Pixi Execution:** All terminal operations, script executions, and linting routines MUST be run via Pixi tasks. Never execute naked `python` or global CLI commands.
* **Verification Pipeline:** Whenever you create a new Python script, modify existing code, or generate runtime components, you must run the local quality assurance pipeline and fix any bugs or type violations surfaced by the linter or type-checker before completing the task:
  `pixi run qa`

### Available Environment Shortcuts
* **Linting & Code Style**: `pixi run lint` (Invokes Ruff auto-fix)
* **Formatting**: `pixi run format` (Invokes Ruff code layout formatting)
* **Static Type Inferences**: `pixi run check-types` (Invokes Pyright strict validation)
* **GPU Diagnostics**: `pixi run check-gpu` (Verifies PyTorch, CUDA, and GPU availability)

---

## 3. Link Navigation Rules

* **Flattened Directory Structure:** Documentation files use a flat naming convention with `__` as path separator (e.g., `ar__modes__train` = Ultralytics → Modes → Train). Files are cross-referenced via relative Markdown links within each document.
* **Sequential Document Traversal:** If a single file lacks sufficient context, follow the internal links to neighboring files. Each doc library is internally linked.
* **No Hallucinations:** Always base implementations on the exact function parameter names, classes, and types documented in these files.

---

## 4. Documentation Lookup Index

### 4A. LightlyTrain (`./agy/docs/lightly_train/`)

Use this when writing code that imports `lightly_train` or configures LightlyTrain training/pretraining.

| Topic | File |
|:---|:---|
| **Python API (core functions, `pretrain()`, `train()`, `load_model()`, `export()`)** | `python_api__lightly_train.md` |
| **Train settings (all training parameters, scheduler, optimizer)** | `settings__train_settings.md` |
| **Pretrain settings (all pretraining parameters)** | `settings__pretrain_settings.md` |
| **Object detection task config** | `object_detection.md` |
| **Image classification** | `image_classification.md` |
| **Semantic segmentation** | `semantic_segmentation.md` |
| **Instance segmentation** | `instance_segmentation.md` |
| **Panoptic segmentation** | `panoptic_segmentation.md` |
| **Distillation method (teacher-student)** | `pretrain_distill__methods__distillation.md` |
| **DINO SSL pretraining method** | `pretrain_distill__methods__dino.md` |
| **DINOv2 SSL pretraining method** | `pretrain_distill__methods__dinov2.md` |
| **SimCLR SSL pretraining method** | `pretrain_distill__methods__simclr.md` |
| **DINOv3 model configs** | `pretrain_distill__models__dinov3.md` |
| **DINOv2 model configs** | `pretrain_distill__models__dinov2.md` |
| **Ultralytics models in LightlyTrain** | `pretrain_distill__models__ultralytics.md` |
| **RT-DETR model configs** | `pretrain_distill__models__rtdetr.md` |
| **RF-DETR model configs** | `pretrain_distill__models__rfdetr.md` |
| **YOLOv12 model configs** | `pretrain_distill__models__yolov12.md` |
| **SuperGradients models** | `pretrain_distill__models__supergradients.md` |
| **TIMM models** | `pretrain_distill__models__timm.md` |
| **Torchvision models** | `pretrain_distill__models__torchvision.md` |
| **Custom models** | `pretrain_distill__models__custom_models.md` |
| **Dataset embeddings / extraction** | `embed.md` |
| **Export pretrained/distilled models** | `pretrain_distill__export.md` |
| **Docker execution & config** | `docker.md` |
| **Autolabeling predictions** | `predict_autolabel.md` |
| **Multi-GPU training** | `performance__multi_gpu.md` |
| **Multi-node cluster training** | `performance__multi_node.md` |
| **Hardware recommendations** | `performance__hardware_recommendations.md` |
| **Quick start: object detection** | `quick_start_object_detection.md` |
| **Quick start: distillation** | `quick_start_distillation.md` |
| **FAQ** | `faq.md` |
| **Changelog** | `changelog.md` |

### 4B. Ultralytics (`./agy/docs/ultralytics/`)

Use this when writing code that imports `ultralytics` or configures YOLO/RT-DETR training. Files use `ar__` prefix (docs root).

| Topic | File |
|:---|:---|
| **Training mode (all train args, CLI, Python API)** | `ar__modes__train` |
| **Validation mode** | `ar__modes__val` |
| **Prediction / Inference mode** | `ar__modes__predict` |
| **Export mode** | `ar__modes__export` |
| **Benchmarking mode** | `ar__modes__benchmark` |
| **Tracking mode** | `ar__modes__track` |
| **Full configuration reference (all YOLO settings)** | `ar__usage__cfg` |
| **CLI usage** | `ar__usage__cli` |
| **Python API usage** | `ar__usage__python` |
| **Detection task** | `ar__tasks__detect` |
| **YOLO11 model architecture** | `ar__models__yolo11` |
| **YOLO26 model architecture** | `ar__models__yolo26` |
| **YOLOv12 model architecture** | `ar__models__yolo12` |
| **RT-DETR model** | `compare__rtdetr-vs-yolo11` (comparison context) |
| **Albumentations integration** | `ar__integrations__albumentations` |
| **W&B integration** | `ar__integrations__weights-biases` |
| **Hyperparameter tuning guide** | `ar__guides__hyperparameter-tuning` |
| **Fine-tuning guide** | `ar__guides__finetuning-guide` |
| **Model training tips** | `ar__guides__model-training-tips` |
| **Data augmentation guide** | `ar__guides__yolo-data-augmentation` |
| **Performance metrics guide** | `ar__guides__yolo-performance-metrics` |
| **Common issues / debugging** | `ar__guides__yolo-common-issues` |
| **COCO dataset format** | `ar__datasets__detect__coco` |
| **Custom data training (YOLOv5 guide)** | `ar__yolov5__tutorials__train-custom-data` |
| **SAHI tiled inference** | `ar__guides__sahi-tiled-inference` |
| **YOLO26 training recipe** | `ar__guides__yolo26-training-recipe` |

### 4C. Weights & Biases (`./agy/docs/wandb/`)

Use this when writing code that imports `wandb` or configures sweeps/runs/artifacts.

| Topic | File |
|:---|:---|
| **Sweeps overview** | `models__sweeps` |
| **Sweep config keys (method, metric, parameters, early_terminate)** | `models__sweeps__sweep-config-keys` |
| **Define sweep configuration** | `models__sweeps__define-sweep-configuration` |
| **Initialize sweeps** | `models__sweeps__initialize-sweeps` |
| **Start sweep agents** | `models__sweeps__start-sweep-agents` |
| **Parallelize agents** | `models__sweeps__parallelize-agents` |
| **Pause/resume/cancel sweeps** | `models__sweeps__pause-resume-and-cancel-sweeps` |
| **Visualize sweep results** | `models__sweeps__visualize-sweep-results` |
| **Troubleshoot sweeps** | `models__sweeps__troubleshoot-sweeps` |
| **Sweep walkthrough** | `models__sweeps__walkthrough` |
| **`wandb.init()` reference** | `models__ref__python__functions__init` |
| **Run object API** | `models__ref__python__experiments__run` |
| **Settings reference** | `models__ref__python__experiments__settings` |
| **Artifact object API** | `models__ref__python__experiments__artifact` |
| **Public API (`wandb.Api()`)** | `models__ref__python__public-api__api` |
| **Sweep public API** | `models__ref__python__public-api__sweep` |
| **Run public API** | `models__ref__python__public-api__run` |
| **`wandb sweep` CLI** | `models__ref__cli__wandb-sweep` |
| **`wandb agent` CLI** | `models__ref__cli__wandb-agent` |
| **`wandb login` CLI** | `models__ref__cli__wandb-login` |
| **Logging metrics** | `models__track__log` |
| **Logging media (images, tables)** | `models__track__log__media` |
| **Config tracking** | `models__track__config` |
| **Environment variables** | `models__track__environment-variables` |
| **Run naming/identifiers** | `models__runs__run-identifiers` |
| **Run grouping** | `models__runs__grouping` |
| **Run tagging** | `models__runs__tags` |
| **Custom charts** | `models__app__features__custom-charts` |
| **SDK cheat sheet: logging** | `models__ref__sdk-coding-cheat-sheet__logging` |
| **SDK cheat sheet: runs** | `models__ref__sdk-coding-cheat-sheet__runs` |
| **PyTorch integration** | `models__integrations__pytorch` |

### 4D. Research Papers (`./agy/docs/`)

Use these for understanding model architectures, training methodology, and experimental setups.

| Paper | File | Key Content |
|:---|:---|:---|
| **DINOv3** | `dinov3.md` | ViT architecture, LTDETR head, satellite pretraining (SAT-493M), LVD-1689M dataset, self-supervised learning, feature extraction |
| **DEIMv2** | `deimv2.md` | Real-time detection with DINOv3, DEIM architecture, benchmark comparisons |

### 4E. FiftyOne (`./agy/docs/fiftyone/`)

Use this when writing code that imports `fiftyone` or configures datasets, views, query expressions, or App visualization.

| Topic | File |
|:---|:---|
| **Overview & entry point** | `index.md` |
| **Terminology & core concepts** | `cheat_sheets__fiftyone_terminology.md` |
| **Dataset views creation cheat sheet** | `cheat_sheets__views_cheat_sheet.md` |
| **Data filtering query reference** | `cheat_sheets__filtering_cheat_sheet.md` |
| **Pandas vs FiftyOne API comparison** | `cheat_sheets__pandas_vs_fiftyone.md` |
| **Model evaluation workflows** | `getting_started__model_evaluation__summary.md` |
| **Object detection dataset summary** | `getting_started__object_detection__summary.md` |
| **Ultralytics YOLO integration** | `integrations__ultralytics.md` |
| **COCO dataset loading reference** | `integrations__coco.md` |
| **Annotation tool integrations** | `integrations__annotation.md` |
| **Core FiftyOne Python API index** | `api__fiftyone.md` |

### 4F. SAHI (`./agy/docs/sahi/`)

Use this when writing code that configures Sliced Aided Hyper Inference (tiled inference) on large resolution aerial/satellite images.

| Topic | File |
|:---|:---|
| **Overview & installation** | `index.md` |
| **Quick start for sliced inference** | `quick-start` |
| **Sliced inference options & guides** | `guides__sliced-inference` |
| **Supported model backends guide** | `guides__models` |
| **Slicing configuration parameters** | `slicing` |
| **AutoModel API class reference** | `auto_model` |
| **Post-processing & NMS combine** | `postprocess__combine` |

---

## 5. Framework-Specific Coding Rules

### 5A. LightlyTrain

- **API Lookup:** Always consult `python_api__lightly_train.md` for `lightly_train.train()`, `lightly_train.pretrain()`, `lightly_train.load_model()`, and `lightly_train.export()` signatures before writing calls.
- **Train Settings:** Consult `settings__train_settings.md` for valid parameter names (e.g., `scheduler`, `scheduler_flat_steps`, `scheduler_no_aug_steps`, `optimizer`, `lr`). Do not invent parameter names.
- **DINOv3 Model Strings:** Valid model identifiers are documented in `pretrain_distill__models__dinov3.md` (e.g., `"dinov3/vitl16"`, `"dinov3/vitl16-ltdetr"`, `"dinov3/convnext-large-ltdetr-coco"`).
- **Hardware Acceleration:** Ensure code uses proper PyTorch accelerators (`cuda`, `mps`, or `cpu`) as specified. Refer to `performance__hardware_recommendations.md`.
- **Avoid Resource Leaks:** When calling `lightly_train.load_model(...)`, ensure GPU memory cleanup or use context-managed blocks to prevent OOM.
- **Config Rigor:** Match the expected schema for dataset mapping dictionaries (must contain correct `train`, `val`, and `test` data keys).

### 5B. Ultralytics

- **Train Args:** Consult `ar__usage__cfg` for the complete list of valid training arguments before writing `model.train(...)` or `model.val(...)` calls. Key args include `epochs`, `batch`, `imgsz`, `device`, `workers`, `amp`, `fraction`, `save_json`, `lr0`, `lrf`, `momentum`, `weight_decay`, `warmup_epochs`, `mosaic`, `mixup`, `copy_paste`.
- **Model Loading:** Use `YOLO("yolo11s.pt")` for YOLO models, `RTDETR("rtdetr-l.pt")` for RT-DETR. Consult `ar__models__yolo11`, `ar__models__yolo26`, `ar__models__yolo12` for model variant names.
- **Augmentation:** Consult `ar__integrations__albumentations` and `ar__guides__yolo-data-augmentation` before modifying augmentation pipelines.

### 5C. Weights & Biases

- **Sweep Configs:** Consult `models__sweeps__sweep-config-keys` for valid YAML keys (`method`, `metric`, `parameters`, `early_terminate`, `command`, `program`).
- **`wandb.init()` Parameters:** Consult `models__ref__python__functions__init` for valid `project`, `entity`, `name`, `config`, `group`, `tags`, `notes` arguments.
- **Public API:** For fetching sweep results programmatically, consult `models__ref__python__public-api__api` and `models__ref__python__public-api__sweep`.
- **Run Naming:** Sensible W&B run names should follow the pattern `{model}_seed_{seed}` for baselines and include phase/sweep identifiers for sweeps.