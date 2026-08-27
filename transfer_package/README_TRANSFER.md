# 📦 Self-Contained Project Transfer & Deployment Package

This package contains all exported model weights, distilled backbones, code scripts, configurations, and thesis documentation required to transfer your work and continue training or evaluation on another PC.

---

## 📁 Package Directory Structure

```
transfer_package/
├── pretrained_backbones/          # Stage 1 Distilled Backbone Weights (Ready for Fine-Tuning)
│   ├── distill_lvd1689m_to_dinov3_vitt16_last.pt
│   ├── distill_sat493m_to_dinov3_vitt16_last.pt
│   ├── distill_lvd1689m_to_yolo12s_last.pt
│   └── distill_sat493m_to_yolo12s_last.pt
├── fine_tuned_weights/             # Exported Best Model Checkpoints (Ready for Direct Evaluation)
│   ├── vitt16_lvd1689m_seed42_best.pt
│   ├── vitt16_sat493m_seed42_best.pt
│   ├── vitt16_sat493m_seed100_best.pt
│   ├── vitt16_sat493m_seed999_best.pt
│   ├── yolo12s_lvd1689m_seed42_best.pt
│   ├── yolo12s_sat493m_seed42_best.pt
│   ├── yolo12s_seed100_best.pt
│   └── yolo12s_seed999_best.pt
├── evaluation_results/            # Complete JSON Hold-Out Test Metric Benchmarks
├── code/                          # Python Scripts & Pixi Environment Specs
│   ├── run_training.py
│   ├── run_distillation.py
│   ├── config.py
│   ├── eval_utils.py
│   ├── pixi.toml
│   └── pixi.lock
└── docs/                          # Essential Thesis Documentation & W&B Manifests
    ├── thesis_notes.md
    ├── wandb_runs_manifest.md
    └── AGENTS.md
```

---

## 🚀 Step-by-Step Setup Guide on the New PC

### Step 1: Environment Installation
On your new PC, install **Pixi** (the environment manager) and lock the exact PyTorch + CUDA environment:

```powershell
# 1. Install Pixi (if not already installed)
iwr -useb https://pixi.sh/install.ps1 | iex

# 2. Enter code directory and install environment
cd code
pixi install
```

---

### Step 2: Running Supervised Fine-Tuning Using Transferred Backbones

To run 150 epochs of supervised fine-tuning using one of the transferred distilled backbones (e.g. `distill_lvd1689m_to_yolo12s_last.pt` or `distill_sat493m_to_dinov3_vitt16_last.pt`):

#### A. Supervised Fine-Tuning for YOLO12s:
```powershell
pixi run python run_training.py --model yolo12s.pt --backbone-weights ../pretrained_backbones/distill_lvd1689m_to_yolo12s_last.pt --seed 100 --backend ultralytics --epochs 150 --patience 30 --batch 8 --imgsz 1024
```

#### B. Supervised Fine-Tuning for ViT-T (Unfrozen Backbone):
```powershell
pixi run python run_training.py --model dinov3/vitt16-ltdetr --backbone-weights ../pretrained_backbones/distill_sat493m_to_dinov3_vitt16_last.pt --seed 100 --backend lightly --backbone-freeze false --epochs 100 --patience 30 --batch 8 --imgsz 1024
```

---

### Step 3: Running Stage 1 Distillation on the New PC

To run Stage 1 DINOv3 Feature Distillation into `yolo26n` (or any other student) at 512px resolution:

```powershell
pixi run python run_distillation.py --teacher dinov3/vitl16 --student yolo26n --skip-finetune --epochs 11 --batch_size 32 --pretrain-imgsz 512 --data "/path/to/your/unlabeled_images"
```

---

### Step 4: Evaluating Transferred Checkpoints on Hold-Out Test Set

To run test set evaluation on any of the transferred fine-tuned weights inside `fine_tuned_weights/`:

```powershell
pixi run python -c "
from eval_utils import evaluate_model_coco, safe_load_model

m = safe_load_model('../fine_tuned_weights/vitt16_lvd1689m_seed42_best.pt')
res = evaluate_model_coco(m, dataset_yaml_path='/path/to/dataset.yaml', split='test')
print(res['metrics'])
"
```
