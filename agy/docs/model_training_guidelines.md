# Model Training Guidelines: Epochs, Batch Sizes, and VRAM Optimization

This document outlines the recommended epochs, patience, and batch size configurations for training YOLO, RT-DETR, and DINO-based models in this repository, with special optimization guidelines for **16 GB VRAM GPUs**.

---

## 1. Summary Configuration Table (1024px Resolution)

| Model Family | Recommended Epochs / Steps | Recommended Patience | Recommended Batch Size (16 GB GPU) | Recommended Batch Size (24 GB+ GPU) |
| :--- | :--- | :--- | :--- | :--- |
| **YOLO** (Ultralytics) | 300 epochs | 100 epochs | **16** | 32 |
| **RT-DETR** (Ultralytics) | 72 epochs | 100 epochs | **4** | 8 |
| **DINO** (LightlyTrain) | 7,200 to 360,000 steps | *N/A* (Not supported) | **2** | 4 to 8 |

---

## 2. YOLO-based Models (Ultralytics)

YOLO models (such as YOLO11s and YOLO26s) are computationally lightweight, meaning they require relatively small memory footprints.

* **Epochs & Patience**: Standard training requires **300 epochs** with **100 patience** to ensure full convergence while preventing overfitting.
* **Batch Size**: 
  * A batch size of **16** is the safe default for high-resolution images (1024px) on a 16 GB card.
  * If resolution is reduced to 640px, the batch size can be safely increased to **32** or **64**.
* **Sources & References**:
  * [ar__modes__train.md](file:///C:/Users/emilb/_trainer_lightly/agy/docs/ultralytics/ar__modes__train.md#L205): Outlines default training arguments and patience description.
  * [ar__usage__cfg.md](file:///C:/Users/emilb/_trainer_lightly/agy/docs/ultralytics/ar__usage__cfg.md#L84): Full parameters list reference.
  * [guides__yolo26-training-recipe.md](file:///C:/Users/emilb/_trainer_lightly/agy/docs/ultralytics/guides__yolo26-training-recipe.md#L208): Discusses model scaling, convergence rate, and early stopping.

---

## 3. RT-DETR Models (Ultralytics)

RT-DETR introduces a hybrid encoder-decoder transformer design which significantly increases VRAM requirements during backpropagation.

* **Epochs & Patience**: Default baseline training uses **72 epochs** (often called the `6x` schedule) with **100 patience** (or **50 patience** for faster termination).
* **Batch Size**: 
  * A batch size of **4** is highly recommended on 16 GB GPUs to prevent Out-Of-Memory (OOM) errors at 1024px.
  * Batch size **8** can be used with mixed precision (AMP) if image resolution is reduced.
* **Sources & References**:
  * [instance_segmentation.md](file:///C:/Users/emilb/_trainer_lightly/agy/docs/lightly_train/instance_segmentation.md#L32): Establishes the 540,000 steps (~72 epochs) ratio for transformer-based model fine-tuning.

---

## 4. DINO-based Models (LightlyTrain)

DINOv3-based ViT-L models are massive foundation architectures. Since training is done using the `lightly_train` backend, duration is configured in **steps** rather than epochs.

> [!NOTE]
> Epoch-based training is currently not supported in LightlyTrain (see [settings__train_settings.md](file:///C:/Users/emilb/_trainer_lightly/agy/docs/lightly_train/settings__train_settings.md#L209)).

* **Backbone Freezing**: Because the pretrained backbone is frozen (`backbone_freeze: bool = True`), we only fine-tune the detector head. This allows the model to converge very quickly (requiring only **12 epochs** on COCO).
* **Batch Size**:
  * A batch size of **2** is the recommended baseline on a 16 GB GPU to avoid OOM errors (as outlined in [config.py](file:///C:/Users/emilb/_trainer_lightly/config.py#L102-L108) under `eval_fallback_batch_size`).
* **Formula for Step Calibration**:
  To train for a target number of epochs, compute the required training steps using your dataset size ($N$) and batch size ($B$):
  
  $$\text{Steps} = \text{Target Epochs} \times \frac{N}{B}$$

  For example, to fine-tune for 12 epochs on a custom dataset of **5,000 training images** with a batch size of **2**:
  
  $$\text{Steps} = 12 \times \frac{5,000}{2} = 30,000 \text{ steps}$$

* **Sources & References**:
  * [settings__train_settings.md](file:///C:/Users/emilb/_trainer_lightly/agy/docs/lightly_train/settings__train_settings.md#L105): Outlines automatic learning rate scaling per batch size.
  * [dinov3.md](file:///C:/Users/emilb/_trainer_lightly/agy/docs/dinov3.md#L1275-L1276): Establishes the 12-epoch downstream COCO validation baseline schedule.
  * [instance_segmentation.md](file:///C:/Users/emilb/_trainer_lightly/agy/docs/lightly_train/instance_segmentation.md#L32): Calibrates step training limits (90K steps = 12 epochs at batch size 16).
