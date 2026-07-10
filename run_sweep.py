"""
Multi-Stage YOLO/LTDETR Hyperparameter Optimization (HPO) Sweep Runner.

This script coordinates both HPO Phase 1 (Augmentation tuning) and Phase 2 (Learning HPO)
runs using Weights & Biases (W&B) and Albumentations for both Ultralytics and lightly_train.
"""

# ruff: noqa: E402
import os
import tempfile

# Set up local temp directory to avoid Windows System Temp cleanup issues with W&B staging
workspace_dir = os.path.abspath(os.path.dirname(__file__))
local_tmp_dir = os.path.join(workspace_dir, ".tmp")
os.makedirs(local_tmp_dir, exist_ok=True)
os.environ["TMP"] = local_tmp_dir
os.environ["TEMP"] = local_tmp_dir
tempfile.tempdir = local_tmp_dir

# Set environment variables to prevent Windows PyTorch multiprocessing deadlock
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["WANDB_START_METHOD"] = "thread"

import pyarrow  # noqa: F401
import wandb
import sys

# Import PIL before torchvision or lightly_train to resolve DLL dependency conflict on Windows
from PIL import Image  # noqa: F401
import gc
import torch
import warnings
from config import PipelineConfig
from eval_utils import (
    build_albumentations_pipeline,
    evaluate_model_coco,
    get_huggingface_backbone,
)

# Suppress python warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*torchvision.*")
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["PYTHONWARNINGS"] = "ignore"

# Model baseline mapping for lightly counterparts
LIGHTLY_BASELINE_MAP = {
    "yolo11n.pt": "ultralytics/yolo11n.yaml",
    "yolo26s.pt": "ultralytics/yolo26s.yaml",
    "yolo26n.pt": "ultralytics/yolo26n.yaml",
    "rtdetr-l.pt": "dinov3/convnext-large-ltdetr-coco",
}


def set_reproducibility(seed: int) -> None:
    """
    Sets seeds for Python, NumPy, and PyTorch across CPU and CUDA backends
    to guarantee strict mathematical reproducibility of runs.
    """
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            try:
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            except Exception as e:
                print(
                    f"Warning: Could not seed CUDA generator: {e}. If a prior CUDA OOM occurred, this is expected."
                )
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception as e:
        print(
            f"Warning: Could not set PyTorch seeds: {e}. If a prior CUDA OOM occurred, this is expected."
        )


def main() -> None:
    # Set reproducibility seed at start of sweep trial agent to lock weight initialization
    set_reproducibility(42)

    cfg = PipelineConfig()

    # Enable W&B integration in Ultralytics settings
    from ultralytics import settings

    settings.update({"wandb": True})

    # Initialize the W&B run (agent injects current sweep parameters)
    run = wandb.init(project=cfg.project, entity=cfg.entity)
    wb_config = dict(run.config)
    phase = wb_config.get("phase", "production")

    # Decide which backend to use based on the model_variant
    model_variant = wb_config.get("model_variant", cfg.model_variant)
    # Let's support both backends dynamically
    backend = (
        "lightly"
        if "dinov" in model_variant.lower()
        or "ltdetr" in model_variant.lower()
        or model_variant.startswith("facebook/")
        else "ultralytics"
    )
    if wb_config.get("backend"):
        backend = wb_config.get("backend")

    print("=" * 60)
    print(f"Sweep run initiated for Phase: {phase} | Backend: {backend}")
    print(f"Model: {model_variant} | Dataset: {cfg.dataset_path}")
    print("=" * 60)

    # 1. Setup shared/fixed arguments
    imgsz = cfg.image_size
    epochs = cfg.sweep_epochs

    # Resolve model-specific batch size and patience
    _, model_patience, model_batch_size = cfg.get_model_params(model_variant, backend)
    batch_size = model_batch_size
    device = cfg.device

    # For lightly backend, batch_size=-1 (auto-batching) is not supported.
    # We fall back to "auto" which is supported by lightly_train.
    if backend == "lightly" and batch_size == -1:
        batch_size = "auto"
    workers = cfg.workers
    fraction = cfg.fraction
    amp = cfg.amp

    # 2. Setup Phase 1 vs Phase 2 sweep parameters
    if phase == "augmentation":
        # Phase 1: Tune augmentations. Keep learning hyperparameters locked to baseline values.
        optimizer_name = "MuSGD"
        lr0_val = 0.0054
        lrf_val = 0.0495
        momentum_val = 0.947
        weight_decay_val = 0.00064
        warmup_epochs_val = 0.98

        # Read augmentations config from sweep config
        active_aug_config = wb_config
    else:
        # Phase 2: Tune HPO. Retrieve the best augmentations from a previous sweep.
        prev_sweep_id = wb_config.get("prev_aug_sweep_id")
        if not prev_sweep_id:
            raise ValueError(
                "Phase 2 HPO requires 'prev_aug_sweep_id' to load optimal augmentations."
            )

        best_aug_config = cfg.get_best_sweep_config(
            prev_sweep_id, cfg.project, cfg.entity
        )
        active_aug_config = best_aug_config

        # Read learning parameters from current sweep config
        optimizer_name = wb_config.get("optimizer", "AdamW")
        lr0_val = float(wb_config.get("lr0", 0.001))
        lrf_val = float(wb_config.get("lrf", 0.01))
        momentum_val = float(wb_config.get("momentum", 0.937))
        weight_decay_val = float(wb_config.get("weight_decay", 0.0005))
        warmup_epochs_val = float(wb_config.get("warmup_epochs", 1.0))

    eval_model = None
    model = None

    # 3. Train using selected backend
    try:
        if backend == "ultralytics":
            is_rtdetr = "rtdetr" in model_variant.lower()
            if is_rtdetr:
                from ultralytics import RTDETR

                model = RTDETR(model_variant)
            else:
                from ultralytics import YOLO

                model = YOLO(model_variant)

            train_kwargs = {
                "data": cfg.dataset_path,
                "epochs": epochs,
                "imgsz": imgsz,
                "device": device,
                "batch": batch_size,
                "workers": workers,
                "fraction": fraction,
                "seed": 42,
                "exist_ok": True,
                "amp": amp,
            }
            if model_patience is not None:
                train_kwargs["patience"] = model_patience
            if cfg.fixed_loss:
                train_kwargs.update(cfg.fixed_loss)

            # Disable default YOLO spatial/color augmentations
            train_kwargs.update(
                {
                    "hsv_h": 0.0,
                    "hsv_s": 0.0,
                    "hsv_v": 0.0,
                    "degrees": 0.0,
                    "fliplr": 0.0,
                    "flipud": 0.0,
                    "scale": 0.0,
                    "translate": 0.0,
                    "shear": 0.0,
                    "bgr": 0.0,
                    "close_mosaic": 15,
                }
            )

            if not is_rtdetr:
                train_kwargs.update(
                    {
                        "mosaic": active_aug_config.get("mosaic", 0.0),
                        "mixup": active_aug_config.get("mixup", 0.0),
                        "copy_paste": active_aug_config.get("copy_paste", 0.0),
                    }
                )

            # Build and inject custom Albumentations pipeline
            train_kwargs["augmentations"] = build_albumentations_pipeline(
                active_aug_config, imgsz
            )

            # Inject learning parameters
            train_kwargs.update(
                {
                    "optimizer": optimizer_name,
                    "lr0": lr0_val,
                    "lrf": lrf_val,
                    "momentum": momentum_val,
                    "weight_decay": weight_decay_val,
                    "warmup_epochs": warmup_epochs_val,
                }
            )

            # Run YOLO training
            model.train(**train_kwargs)
            eval_model = model

        else:
            # --- lightly_train Training Backend ---
            import lightly_train

            if model_variant.startswith("facebook/"):
                if "vits16" in model_variant:
                    lightly_model_name = "dinov3/vits16-ltdetr"
                elif "vitb16" in model_variant:
                    lightly_model_name = "dinov3/vitb16-ltdetr"
                elif "vitt16" in model_variant:
                    lightly_model_name = "dinov3/vitt16-ltdetr"
                elif "sat493m" in model_variant:
                    lightly_model_name = "dinov3/vitl16-ltdetr"
                else:
                    lightly_model_name = "dinov3/vitl16-ltdetr"
                hf_weights = get_huggingface_backbone(model_variant)
            else:
                lightly_model_name = LIGHTLY_BASELINE_MAP.get(
                    model_variant,
                    f"ultralytics/{model_variant.replace('.pt', '.yaml')}",
                )
                hf_weights = None

            accel = "cpu"
            device_arg = "auto"
            if isinstance(device, int) or (
                isinstance(device, str) and device.isdigit()
            ):
                accel = "gpu"
                device_arg = [int(device)]
            elif isinstance(device, str) and "cpu" not in device.lower():
                accel = "gpu"
                device_arg = "auto"

            # Calculate steps dynamically from epochs
            import math
            import yaml

            with open(cfg.dataset_path, "r") as f:
                data_yaml = yaml.safe_load(f)

            # Resolve absolute path for training images
            base_path = data_yaml.get("path", "")
            if not os.path.isabs(base_path):
                base_path = os.path.abspath(
                    os.path.join(os.path.dirname(cfg.dataset_path), base_path)
                )
            train_rel = data_yaml.get("train", "")
            if not os.path.isabs(train_rel):
                train_img_dir = os.path.join(base_path, train_rel)
            else:
                train_img_dir = train_rel

            num_train_images = 3500  # Fallback default
            if os.path.exists(train_img_dir):
                valid_extensions = {
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".ppm",
                    ".bmp",
                    ".pgm",
                    ".tif",
                    ".tiff",
                    ".webp",
                }
                num_train_images = sum(
                    1
                    for file in os.listdir(train_img_dir)
                    if os.path.splitext(file)[1].lower() in valid_extensions
                )

            calc_batch_size = 16 if batch_size in ("auto", -1) else int(batch_size)
            # For lightly_train object detection, default target batch size is 32.
            # It uses gradient accumulation max(1, 32 // calc_batch_size) to reach 32.
            effective_batch_size = (
                max(32, calc_batch_size) if calc_batch_size >= 32 else 32
            )
            steps_raw = math.ceil(epochs * (num_train_images / effective_batch_size))
            dino_steps = math.ceil(steps_raw / 1000) * 1000

            print("\n" + "=" * 60)
            print(" DINO DYNAMIC STEP CALCULATION FOR SWEEP")
            print("-" * 60)
            print(f" Target Epochs:         {epochs}")
            print(f" Training Images (N):   {num_train_images}")
            print(f" Global Batch Size (B):  {calc_batch_size}")
            print(f" Effective Batch Size:  {effective_batch_size} (via Accumulation)")
            print(
                f" Steps per Epoch:       {math.ceil(num_train_images / effective_batch_size)}"
            )
            print(f" Total Steps:           {dino_steps}")
            print("=" * 60 + "\n")

            model_args = {
                "lr": lr0_val,
                "weight_decay": weight_decay_val,
                "scheduler_name": "flat-cosine",
            }
            if dino_steps < 2000:
                model_args["lr_warmup_steps"] = 0
                model_args["ema_warmup_steps"] = 0
                model_args["scheduler_flat_steps"] = 0
                model_args["scheduler_no_aug_steps"] = 0
            if hf_weights:
                model_args["backbone_weights"] = hf_weights
            if "sat493m" in model_variant:
                model_args["backbone_args"] = {"is_sat493m_weights": True}

            transform_args = {
                "image_size": (imgsz, imgsz),
                "random_flip": {
                    "horizontal_prob": float(
                        active_aug_config.get("albu_spatial_p", 0.5)
                    ),
                    "vertical_prob": float(
                        active_aug_config.get("albu_spatial_p", 0.5)
                    ),
                },
                "color_jitter": {
                    "prob": float(active_aug_config.get("albu_color_p", 0.5)),
                    "strength": 1.0,
                    "brightness": 0.2,
                    "contrast": 0.2,
                    "saturation": 0.2,
                    "hue": 0.05,
                },
                "random_rotate_90": {
                    "prob": float(active_aug_config.get("albu_rotate90_p", 0.5))
                },
            }

            out_run_dir = os.path.join(cfg.runs_dir, f"sweep_run_{run.id}")

            # Run training
            import yaml

            with open(cfg.dataset_path, "r") as f:
                data_dict = yaml.safe_load(f)
            data_dict["format"] = "yolo"

            if cfg.entity:
                os.environ["WANDB_ENTITY"] = cfg.entity
            lightly_train.train_object_detection(
                out=out_run_dir,
                model=lightly_model_name,
                data=data_dict,
                steps=dino_steps,
                batch_size=batch_size,
                num_workers=workers,
                devices=device_arg,
                accelerator=accel,
                precision="16-mixed" if amp else "32-true",
                seed=42,
                overwrite=True,
                model_args=model_args,
                transform_args=transform_args,
                logger_args={
                    "wandb": {"project": cfg.project, "name": f"sweep_run_{run.id}"}
                },
            )

            # Load best checkpoint for validation evaluation
            best_ckpt = os.path.join(out_run_dir, "exported_models", "exported_best.pt")
            if not os.path.exists(best_ckpt):
                best_ckpt = os.path.join(
                    out_run_dir, "exported_models", "exported_last.pt"
                )

            from eval_utils import safe_load_model

            eval_model = safe_load_model(best_ckpt)

        # 4. Strict COCO Evaluation
        eval_batch_size = (
            cfg.eval_fallback_batch_size
            if (batch_size == "auto" or batch_size == -1)
            else batch_size
        )
        metrics = evaluate_model_coco(
            model_path_or_model=eval_model,
            dataset_yaml_path=cfg.dataset_path,
            split="val",  # sweeps tune based on validation set
            eval_results_dir=cfg.eval_results_dir,
            run_name=f"sweep_run_{run.id}",
            device=device,
            batch_size=eval_batch_size,
            imgsz=imgsz,
            workers=workers,
        )

        # Log COCOeval validation metrics to W&B
        # W&B Sweep metrics controller reads "metrics/AP"
        wandb.log({f"metrics/{k}": v for k, v in metrics["metrics"].items()})

        # Automatically tag top-performing runs on the W&B dashboard for easy filtering
        val_map = metrics["metrics"].get("AP", 0.0)
        if val_map > 0.5:
            run.tags = run.tags + ("top_performer",) if run.tags else ("top_performer",)

    except Exception as e:
        print(f"Error occurred during sweep training run: {e}", file=sys.stderr)
    finally:
        wandb.finish()

        # Free GPU memory
        eval_model = None
        model = None
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


if __name__ == "__main__":
    main()
