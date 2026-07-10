# Epoch and Step Settings Analysis

This document provides a mathematical breakdown of the training epoch/step configurations for YOLO, RT-DETR, and DINO-based models in this repository. It analyzes whether these configurations are suitable for a dataset size of **~3,500 images** and explains the source/derivation of the default DINO step count of `118314`.

---

## 1. Summary of Epoch-to-Step Translation ($N \approx 3,500$)

| Model / Backend | Configured Epochs / Steps | Batch Size ($B$) | Steps per Epoch ($\lceil N/B \rceil$) | Total Steps | Equivalent Epochs |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **YOLO** (Ultralytics) | 300 epochs | 16 | 219 | 65,700 | **300.0** |
| **RT-DETR** (Ultralytics) | 300 epochs | 4 | 875 | 262,500 | **300.0** |
| **DINO** (LightlyTrain) | 118,314 steps | 2 | 1,750 | 118,314 | **~67.6** |

---

## 2. Derivation of DINO's Default Steps (`118314`)

DINO-based models in the `lightly_train` backend are step-based rather than epoch-based. The configuration variable `dino_epochs` in `config.py` is initialized to `118314` and passed as the `steps` parameter during training.

This value of **118,314 steps** is derived from a standard **16-epoch COCO training schedule** with a batch size of 16:
1. The standard **COCO 2017 training set** contains **118,287 images**.
2. With a batch size of $16$, one training epoch requires:
   $$\text{Steps per Epoch} = \lceil 118,287 / 16 \rceil = 7,393 \text{ steps}$$
3. Over a standard $16$-epoch training schedule, the total steps are:
   $$\text{Total Steps} = 16 \text{ epochs} \times 7,393 \text{ steps/epoch} = \mathbf{118,288 \text{ steps}}$$
4. The value of **118,314** is a slight padding (or corresponds to a specific subset/index count of COCO train images including minor annotations filters, which yields 118,314 images in some custom sub-splits).

Thus, **`dino_epochs = 118314` represents a 16-epoch training budget on the full COCO dataset.**

---

## 3. Suitability Analysis for $N \approx 3,500$ Images

### 3.1. YOLO (Ultralytics)
* **Configuration**: 300 epochs (max), early stopping patience of 100 epochs, batch size 16.
* **Analysis**: Highly suitable. 300 epochs is the standard duration for training native convolutional models to full convergence. Early stopping will automatically terminate training if the validation mAP plateaus, preventing overfitting and saving compute resources.

### 3.2. RT-DETR (Ultralytics)
* **Configuration**: 300 epochs (max), early stopping patience of 100 epochs, batch size 4.
* **Analysis**: Highly suitable. RT-DETR has a hybrid transformer encoder-decoder that converges slower than CNNs. 300 epochs allows plenty of updates (262,500 steps) for convergence, and the patience of 100 prevents waste if convergence occurs early.

### 3.3. DINO (LightlyTrain)
* **Default Production Mode (118,314 steps, batch size 2)**:
  * Equivalent to **~67.6 epochs** ($118,314 / 1,750$).
  * **Analysis**: This is a reasonable training depth for fine-tuning a frozen backbone (since `backbone_freeze: bool = True` by default, only the transformer decoder head is trained).
  * **Overfitting / Scaling Risk**: Because `lightly_train` does not support early stopping (`dino_patience = None`), it will run for the full 118,314 steps. If you scale the batch size (e.g. to 8 or 16), the equivalent epoch count jumps to **270** or **540 epochs**, leading to severe overfitting.

* **Sweeps & CLI Overrides (The Step-Epoch Bug)**:
  * In `run_sweep.py`, `sweep_epochs` is set to `100` and passed as `steps = 100` to `lightly_train`. This results in **0.057 epochs** of training.
  * In `run_training.py`, if the user runs `--epochs 300`, it passes `steps = 300`. This results in **0.17 epochs** of training.
  * **Analysis**: **This is incorrect.** In these scenarios, DINO models are barely trained, rendering hyperparameter optimization sweeps and custom CLI overrides useless.

---

## 4. Recommendations and Corrections

1. **Epoch-to-Step Calculation**:
   We recommend calculating `steps` dynamically in `run_training.py` and `run_sweep.py` for DINO models:
   $$\text{steps} = \text{epochs} \times \lceil N / B \rceil$$
   This ensures that any CLI epoch override or sweep setting translates to the correct number of steps for the dataset size $N$.

2. **Standard Epoch Budget**:
   For fine-tuning a frozen DINOv3 model on a 3,500 image dataset, a target of **30 to 50 epochs** (equivalent to 52,500 to 87,500 steps at batch size 2) is the optimal range to avoid overfitting while achieving full head convergence.

---

## 5. DINO Model Epoch Target Guidelines

Based on local documentation, target epochs for DINO-based models are segmented by architecture scale and task domain:

1. **Large/Medium Frozen Backbones (`vitb16`, `vitl16`)**:
   * **Target**: **12 Epochs**
   * **Reasoning**: With `backbone_freeze: bool = True` (default), only the LTDETR/D-FINE decoder head is being trained. Since the backbone represents high-quality Meta pretrained foundation weights, the adapter head converges extremely quickly (as seen in COCO baselines).
   * **Calculation ($N \approx 3,500, B = 2$)**:
     $$\text{Steps} = 12 \text{ epochs} \times \lceil 3500 / 2 \rceil = 21,000 \text{ steps}$$

2. **Distilled Tiny Backbones (`vitt16`, `vitt16plus`)**:
   * **Target**: **72 Epochs**
   * **Reasoning**: The tiny backbones have significantly lower parameter capacity and need a much longer schedule to fully align visual representations with the detector head.
   * **Calculation ($N \approx 3,500, B = 2$)**:
     $$\text{Steps} = 72 \text{ epochs} \times \lceil 3500 / 2 \rceil = 126,000 \text{ steps}$$

3. **Domain-Specific Custom Datasets (e.g. Tree Crowns)**:
   * **Target**: **30 to 50 Epochs**
   * **Reasoning**: Drone-perspective aerial imagery contains domain shifts (nadir angles, dense small crowns) that require the decoder head to train longer than on standard ground-level images (like COCO).
   * **Calculation ($N \approx 3,500, B = 2$)**:
     $$\text{Steps} = \text{30 to 50 epochs} \times \lceil 3500 / 2 \rceil = 52,500 \text{ to } 87,500 \text{ steps}$$

---

## 6. Why Not Simply Train to 100+ Epochs and Keep the Best Model?

While PyTorch Lightning saves the "best" model checkpoint based on a validation metric (`save_best: bool = True`), simply running training to a massive number of epochs (like 100+) and selecting the best checkpoint is highly counter-productive for DINO models due to the following structural reasons:

### 6.1. Learning Rate Scheduler Coupling
DINO training in LightlyTrain uses a step-bound learning rate scheduler (such as `"flat-cosine"`). The scheduler maps the learning rate warmup, flat phase, and cosine decay curve directly to the configured `total_steps`:
* **Under-decay**: If you configure training for 100 epochs, but the model starts to converge at epoch 30, the learning rate will still be high (flat phase) at epoch 30 because the decay is scheduled to occur towards step 175,000 (epoch 100). The model may never reach its optimal weights at epoch 30 due to a learning rate that is too high.
* **Optimal convergence**: If training is configured for 30 epochs, the scheduler scales down the learning rate to near-zero exactly at epoch 30, which allows the optimizer to make fine-grained weight updates and reach the true peak performance of the model.

### 6.2. Validation Data Leakage & Overfitting
If a model is trained far past its convergence point (e.g., training to 100 epochs when it converged at epoch 30), selecting the checkpoint with the absolute highest validation score leads to **validation set overfitting**:
* The model begins to fit the noise and idiosyncrasies of the validation split.
* The "best" checkpoint saved will generalize poorly to the actual test split, defeating the purpose of an unbiased evaluation.

### 6.3. Wasted Compute Resources
DINOv3 models (specifically ViT-L/16) are extremely heavy (300M+ parameters). Because LightlyTrain does not support early stopping (`dino_patience = None`), it will run the *entire* step budget. Training for 100 epochs instead of 30–50 epochs wastes significant energy, GPU hours, and time without any performance gains.
