# DINOv3 Smoketest Troubleshooting & Fixes

This document records the diagnostics, fixes, and verification results from troubleshooting the DINOv3 smoketest pipeline on consumer hardware (specifically an 8GB RTX 3060 Ti GPU).

---

## 1. Diagnostics & Root Cause Analysis

### 1A. GPU Out of Memory (OOM) on ViT-L
When running the initial DINOv3 smoketest configuration:
```bash
pixi run train --model facebook/dinov3-vitl16-pretrain-lvd1689m --backend lightly --epochs 2 --fraction 0.01 --batch 1 --seed 42
```
The run terminated with a CUDA Out Of Memory (OOM) error:
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 26.00 MiB. 
GPU 0 has a total capacity of 8.00 GiB of which 0 bytes is free. 
Of the allocated memory 14.18 GiB is allocated by PyTorch...
```
* **Root Cause**: The DINOv3 ViT-L model (307M parameters fine-tuned with the RT-DETR decoder head) requires **~14.6 GB** of memory during training at batch size 1, making it impossible to fine-tune on consumer GPUs with 8GB VRAM.

### 1B. Dynamic Model Resolution Hardcoding
To run a fast smoketest, a smaller model (such as ViT-S with 21.6M parameters) is required. However:
* **Root Cause**: [run_training.py](file:///C:/Users/emilb/Documents/GitHub/_trainer_lightly/run_training.py) hardcoded all model names starting with `facebook/` to use the `dinov3/vitl16-ltdetr` LightlyTrain configuration:
  ```python
  if model_name.startswith("facebook/"):
      lightly_model_name = "dinov3/vitl16-ltdetr"
  ```
  This prevented the pipeline from loading the appropriate architecture configuration for smaller models.

### 1C. Checkpoint Conversion Layer/Dimension Mismatch
When fallback was attempted with `facebook/dinov3-vits16-pretrain-lvd1689m`, the script crashed with a `RuntimeError` during weight loading:
```
RuntimeError: Error(s) in loading state_dict for DinoVisionTransformer:
    Unexpected key(s) in state_dict: "blocks.12.norm1.weight", "blocks.12.norm1.bias"...
    size mismatch for blocks.0.attn.qkv.bias_mask: copying a param with shape torch.Size([3072]) from checkpoint, the shape in current model is torch.Size([1152]).
```
* **Root Cause**: In [eval_utils.py](file:///C:/Users/emilb/Documents/GitHub/_trainer_lightly/eval_utils.py), the `get_huggingface_backbone` converter copied missing parameters (e.g. `bias_mask`) and constants from a default `dinov3_vitl16` model (24 layers, dimension 1024). When converting a ViT-S model (12 layers, dimension 384), this added extra blocks (12-23) and mismatched parameter shapes to the target checkpoint.

### 1D. Evaluation Path FileNotFoundError
When running evaluation on a model with `facebook/` in its name, the program crashed with:
```
FileNotFoundError: [Errno 2] No such file or directory: '.../evaluation_results/facebook/dinov3-vits16-pretrain-lvd1689m_coco_predictions.json'
```
* **Root Cause**: The output filename utilized `run_name` directly (which contains `facebook/` as a nested subpath), but the parent directory was not recursively created before open/write operations.

---

## 2. Implemented Codebase Solutions

### Fix 2A: Dynamic Model Name Resolution
Modified [run_training.py](file:///C:/Users/emilb/Documents/GitHub/_trainer_lightly/run_training.py#L605-L614) to parse the backbone size from the Hugging Face identifier:
```python
if model_name.startswith("facebook/"):
    if "vits16" in model_name:
        lightly_model_name = "dinov3/vits16-ltdetr"
    elif "vitb16" in model_name:
        lightly_model_name = "dinov3/vitb16-ltdetr"
    elif "vitt16" in model_name:
        lightly_model_name = "dinov3/vitt16-ltdetr"
    else:
        lightly_model_name = "dinov3/vitl16-ltdetr"
    hf_weights = get_huggingface_backbone(model_name)
```

### Fix 2B: Dynamic Conversion Template Builder Selection
Modified [eval_utils.py](file:///C:/Users/emilb/Documents/GitHub/_trainer_lightly/eval_utils.py#L813-L828) to select the correct template backbone size:
```python
        try:
            from lightly_train._models.dinov3.dinov3_src.hub.backbones import (
                dinov3_vitt16,
                dinov3_vits16,
                dinov3_vitb16,
                dinov3_vitl16,
            )

            if "vits16" in model_id:
                temp_model = dinov3_vits16(pretrained=False)
            elif "vitb16" in model_id:
                temp_model = dinov3_vitb16(pretrained=False)
            elif "vitt16" in model_id:
                temp_model = dinov3_vitt16(pretrained=False)
            else:
                temp_model = dinov3_vitl16(pretrained=False)

            temp_sd = temp_model.state_dict()
            for k, v in temp_sd.items():
                if k not in new_state_dict:
                    new_state_dict[k] = v
        except Exception as e:
            print(f"Warning: could not load default template model to copy remaining keys: {e}")
```

### Fix 2C: Recursive Directory Creation
Added `os.makedirs(os.path.dirname(pred_file), exist_ok=True)` in [eval_utils.py](file:///C:/Users/emilb/Documents/GitHub/_trainer_lightly/eval_utils.py#L274-L275) to cleanly support nested model names containing slash characters (e.g. `facebook/...`).

---

## 3. Verification & Resource Usage

A full training and evaluation verification run was executed:
```bash
pixi run train --model facebook/dinov3-vits16-pretrain-lvd1689m --backend lightly --epochs 2 --fraction 0.01 --batch 1 --seed 42
```

### Results:
* **Validation & Test Evaluation**: Succeeds completely without errors.
* **VRAM Consumption**: **~2.3 GB** (down from >14 GB).
* **Execution Time**: **~30 seconds**.
* **Metrics Location**: `evaluation_results/facebook/dinov3-vits16-pretrain-lvd1689m_seed_42_rtdetrv2_coco_metrics.json`
* **Checkpoints Location**: `runs/facebook/dinov3-vits16-pretrain-lvd1689m_seed_42_rtdetrv2/checkpoints/`
