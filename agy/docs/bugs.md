# Isolated Bugs Log & Solutions

This document logs the runtime bugs identified during the baseline benchmark execution, their root cause analysis, and planned resolution steps.

---

## Bug 1: RT-DETR Index Out of Range at Low Resolution (`rtdetr-l.pt`)

### 1A. Traceback Snippet
```
Error occurred during training of rtdetr-l_seed_42: selected index k out of range
Traceback (most recent call last):
  ...
  File ".../ultralytics/nn/modules/head.py", line 1709, in _get_decoder_input
    topk_ind = torch.topk(enc_outputs_scores.max(-1).values, self.num_queries, dim=1).indices.view(-1)
RuntimeError: selected index k out of range
```

### 1B. Root Cause Analysis
- At `--imgsz 80`, Ultralytics auto-adjusted the image size to `96` (the nearest multiple of max stride 32).
- At size 96, the multi-scale encoder features at strides 8, 16, and 32 have resolutions of $12\times 12 = 144$, $6\times 6 = 36$, and $3\times 3 = 9$ respectively.
- This results in a total of $144 + 36 + 9 = 189$ candidate encoder features.
- However, RT-DETR is configured to select the top $k = 300$ queries (`self.num_queries`) from these candidates. Since $k$ (300) is greater than the total number of features (189), `torch.topk` crashes with an out-of-range index error.

### 1C. Solution Plan
- Enforce a minimum image resolution of **`160`** (or at least `128`) for all RT-DETR-based models in `run_training.py` and `run_baseline_benchmark.py`.
- Add validation logic that checks if the spatial feature count is less than `num_queries` and raises a clear warning/exception before training starts.

---

## Bug 2: DINOv3 Concatenation Dimension Mismatch on Odd Patch Grids

### 2A. Traceback Snippet
```
  File ".../lightly_train/_task_models/object_detection_components/hybrid_encoder.py", line 413, in forward
    torch.concat([downsample_feat, feat_height], dim=1)
RuntimeError: Sizes of tensors must match except in dimension 1. Expected size 3 but got size 2 for tensor number 1 in the list.
```

### 2B. Root Cause Analysis
- The DINOv3 backbone uses a ViT-L/16 architecture (patch size 16).
- At `--imgsz 80`, the patch grid size is $80 / 16 = 5$ (an odd number).
- During forward passes through the multi-scale hybrid encoder, stride-2 downsampling on odd-sized feature maps (e.g. $5\times 5$) results in rounding differences (e.g., one path rounds to $3\times 3$, while another matches $2\times 2$). This leads to a shape mismatch during concatenation in `hybrid_encoder.py`.

### 2C. Solution Plan
- Enforce that when running DINOv3 models, the image resolution **must be a multiple of 32** (ensuring even patch grids like $96/16=6$, $128/16=8$, $160/16=10$).
- Prevent odd patch grids from being passed to DINO backbones.

---

## Bug 3: Disk Space Exhaustion (`OSError: [Errno 28] No space left on device`)

### 3A. Traceback Snippet
```
RuntimeError: [enforce fail at inline_container.cc:783] . PytorchStreamWriter failed writing file data/714: file write failed
...
OSError: [Errno 28] No space left on device
```

### 3B. Root Cause Analysis
- The ViT-L backbone models are massive: a single PyTorch Lightning Fabric checkpoint (`.ckpt`) is roughly **2 GB+** (saving optimizer states, scheduler, and model params), and each exported PyTorch model (`.pt`) is **1.2 GB+**.
- Running the baseline suite across multiple models and seeds generates tens of gigabytes of checkpoint files very quickly, fully exhausting the user's C: drive space.

### 3C. Solution Plan
- Inform the user to free up space on their `C:` drive.
- Provide a clean-up command/option to remove completed run checkpoint states that are no longer needed (e.g. intermediate checkpoint files) or run training on another drive.
