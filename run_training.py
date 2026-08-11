"""
Stock Training Run Execution Script.

This script executes training runs for YOLO11, YOLO26, and RT-DETR at stock settings,
supporting both lightly_train and ultralytics backends, as well as distillation pretraining.
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
import argparse
import sys

# Import PIL before torchvision or lightly_train to resolve DLL dependency conflict on Windows
from PIL import Image  # noqa: F401

import gc
import shutil
import torch
import yaml
import wandb
import lightly_train
from ultralytics import settings, YOLO
from config import PipelineConfig
from eval_utils import (
    evaluate_model_coco,
    build_albumentations_pipeline,
    get_huggingface_backbone,
)

# Baseline models mapping for lightly_train counterparts
LIGHTLY_BASELINE_MAP = {
    "yolo11n.pt": "ultralytics/yolo11n.yaml",
    "yolo26n.pt": "ultralytics/yolo26n.yaml",
    "rtdetr-l.pt": "dinov3/convnext-large-ltdetr-coco",  # RT-DETR DINOv3 large equivalent in lightly
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLO11, YOLO26, and RT-DETR with support for lightly_train."
    )
    parser.add_argument(
        "--model",
        type=str,
        nargs="+",
        default=None,
        help="Specify one or more models to train (e.g. yolo11s.pt yolo26s.pt). If not specified, trains all configured models.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Specify a single seed to train with. If not specified, trains across all configured seeds.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the number of epochs to train for.",
    )
    parser.add_argument(
        "--batch", type=int, default=None, help="Override the batch size."
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Override the early stopping patience (for YOLO/RT-DETR).",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Override the device (e.g., 0 or cpu)."
    )
    parser.add_argument(
        "--imgsz", type=int, default=None, help="Override the image size."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Override the number of dataloader workers.",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=None,
        help="Override the fraction of dataset to train on (e.g. 0.01 for 1% of data).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Override the dataset.yaml configuration file path.",
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default=None,
        help="Override the runs directory for training checkpoints.",
    )
    parser.add_argument(
        "--wandb-dir", type=str, default=None, help="Override the wandb log directory."
    )
    parser.add_argument(
        "--eval-results-dir",
        type=str,
        default=None,
        help="Override the evaluation results directory for strict metrics.",
    )
    parser.add_argument(
        "--aug-sweep-id",
        type=str,
        default=None,
        help="W&B sweep ID for Phase 1 (augmentation tuning).",
    )
    parser.add_argument(
        "--hpo-sweep-id", type=str, default=None, help="W&B sweep ID for Phase 2 (HPO)."
    )
    parser.add_argument(
        "--amp",
        type=str,
        default=None,
        help="Enable/disable Automatic Mixed Precision (AMP) (True/False).",
    )
    parser.add_argument(
        "--distill",
        action="store_true",
        help="Enable teacher-student DINOv3 distillation pretraining before object detection fine-tuning.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["ultralytics", "lightly"],
        default="lightly",
        help="Backend to use for training (default: lightly)",
    )
    parser.add_argument(
        "--decoder",
        type=str,
        choices=["rtdetrv2", "dfine"],
        default=None,
        help="Specify the decoder head of LTDETR models (rtdetrv2 or dfine). Defaults to configuration setting.",
    )
    parser.add_argument(
        "--tags",
        type=str,
        nargs="+",
        default=None,
        help="List of tags to assign to the Weights & Biases run.",
    )
    parser.add_argument(
        "--backbone-weights",
        type=str,
        default=None,
        help="Path to custom pretrained or distilled backbone weights file (e.g. exported_last.pt).",
    )
    parser.add_argument(
        "--backbone-freeze",
        type=str,
        default=None,
        help="Override the backbone_freeze setting (True/False).",
    )
    parser.add_argument(
        "--wandb-offline",
        action="store_true",
        help="Run Weights & Biases in offline mode to prevent network-related deadlocks.",
    )
    parser.add_argument(
        "--raw-steps",
        action="store_true",
        help="Disable DINO steps rounding to the next full 1,000 steps, running the exact epoch count.",
    )
    parser.add_argument(
        "--resume-interrupted",
        action="store_true",
        help="Resume training from an interrupted or crashed run.",
    )
    return parser.parse_args()


def prepare_subset_dataset(dataset_path: str, fraction: float, temp_dir: str) -> str:
    """
    Subsets a YOLO format dataset to use only a fraction of its images,
    creating a temporary dataset config file.
    """
    import shutil

    with open(dataset_path, "r") as f:
        data = yaml.safe_load(f)

    orig_base = data["path"]
    if not os.path.isabs(orig_base):
        orig_base = os.path.abspath(
            os.path.join(os.path.dirname(dataset_path), orig_base)
        )

    names = data.get("names", {0: "ail"})

    # Clean and recreate temp folder
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    for split in ["train", "val", "test"]:
        orig_img_dir = os.path.join(orig_base, data[split])
        orig_lbl_dir = orig_img_dir.replace("images", "labels")

        target_img_dir = os.path.join(temp_dir, "images", split)
        target_lbl_dir = os.path.join(temp_dir, "labels", split)
        os.makedirs(target_img_dir, exist_ok=True)
        os.makedirs(target_lbl_dir, exist_ok=True)

        if os.path.exists(orig_img_dir):
            images = sorted(os.listdir(orig_img_dir))
            num_select = max(1, int(len(images) * fraction))

            # Scientifically rigorous deterministic subsetting to avoid file ordering bias
            import random

            rng = random.Random(42)
            selected_images = rng.sample(images, num_select)

            for img in selected_images:
                # Copy image
                shutil.copy2(
                    os.path.join(orig_img_dir, img), os.path.join(target_img_dir, img)
                )
                # Copy labels
                base_name, _ = os.path.splitext(img)
                lbl_file = f"{base_name}.txt"
                orig_lbl_path = os.path.join(orig_lbl_dir, lbl_file)
                if os.path.exists(orig_lbl_path):
                    shutil.copy2(orig_lbl_path, os.path.join(target_lbl_dir, lbl_file))

    # Write new temp dataset.yaml
    temp_yaml_path = os.path.join(temp_dir, "dataset.yaml")
    temp_data = {
        "path": temp_dir,
        "train": r"images/train",
        "val": r"images/val",
        "test": r"images/test",
        "names": names,
    }
    with open(temp_yaml_path, "w") as f:
        yaml.dump(temp_data, f, default_flow_style=False)

    return os.path.abspath(temp_yaml_path)


def run_distillation_pretrain(
    student_name: str,
    teacher_name: str,
    train_img_dir: str,
    out_dir: str,
    steps: int,
    seed: int = 42,
) -> str:
    """Runs DINOv3 knowledge distillation using lightly_train.pretrain."""
    import lightly_train

    print("\n=== Running DINOv3 Distillation Pretraining ===")
    print(f"Student: {student_name} | Teacher: {teacher_name}")
    print(f"Training images: {train_img_dir}")
    print(f"Outputs to: {out_dir}")
    print(f"Seed: {seed}")

    lightly_train.pretrain(
        out=out_dir,
        data=train_img_dir,
        model=student_name,
        method="distillation",
        method_args={"teacher": teacher_name},
        steps=steps,  # type: ignore
        devices=1 if torch.cuda.is_available() else "auto",
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        seed=seed,
        overwrite=True,
    )

    best_weights = os.path.join(out_dir, "exported_models", "exported_best.pt")
    if not os.path.exists(best_weights):
        best_weights = os.path.join(out_dir, "exported_models", "exported_last.pt")
    return best_weights


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
    args = parse_args()
    if args.wandb_offline:
        os.environ["WANDB_MODE"] = "offline"
    cfg = PipelineConfig()

    # Automatically enable Tensor Cores on capable GPUs
    if torch.cuda.is_available():
        try:
            device_id = torch.cuda.current_device()
            major, minor = torch.cuda.get_device_capability(device_id)
            if major >= 7:
                torch.set_float32_matmul_precision("high")
                print(
                    f"\n[INFO] CUDA Compute Capability {major}.{minor} detected. "
                    "Tensor Cores are available. Enabled float32_matmul_precision='high'."
                )
        except Exception as e:
            print(
                f"\n[WARNING] Failed to query CUDA capability / set float32 precision: {e}"
            )

    # Enable W&B integration in Ultralytics settings
    settings.update({"wandb": True})

    # Determine which models to train
    if args.model:
        models_to_train = args.model
    else:
        models_to_train = list(cfg.models)

    # Determine which seeds to use
    if args.seed is not None:
        seeds_to_use = [args.seed]
    else:
        seeds_to_use = list(cfg.seeds)

    epochs_display = str(args.epochs) if args.epochs is not None else "Model-Specific"
    batch_size_display = str(args.batch) if args.batch is not None else "Model-Specific"
    device = args.device if args.device is not None else cfg.device
    imgsz = args.imgsz if args.imgsz is not None else cfg.image_size
    workers = args.workers if args.workers is not None else cfg.workers
    fraction = args.fraction if args.fraction is not None else cfg.fraction
    runs_dir = args.runs_dir if args.runs_dir is not None else cfg.runs_dir
    wandb_dir = args.wandb_dir if args.wandb_dir is not None else cfg.wandb_dir
    eval_results_dir = (
        args.eval_results_dir
        if args.eval_results_dir is not None
        else cfg.eval_results_dir
    )

    aug_sweep_id = (
        args.aug_sweep_id if args.aug_sweep_id is not None else cfg.aug_sweep_id
    )
    hpo_sweep_id = (
        args.hpo_sweep_id if args.hpo_sweep_id is not None else cfg.hpo_sweep_id
    )

    decoder_name = args.decoder if args.decoder is not None else cfg.decoder_name

    amp = cfg.amp
    if args.amp is not None:
        amp = args.amp.lower() in ("true", "1", "yes")

    backbone_freeze = cfg.backbone_freeze
    if args.backbone_freeze is not None:
        backbone_freeze = args.backbone_freeze.lower() in ("true", "1", "yes")

    # Resolve relative paths
    if not os.path.isabs(runs_dir):
        runs_dir = os.path.abspath(runs_dir)
    if not os.path.isabs(wandb_dir):
        wandb_dir = os.path.abspath(wandb_dir)
    if not os.path.isabs(eval_results_dir):
        eval_results_dir = os.path.abspath(eval_results_dir)

    # Clean up large training checkpoints (*.ckpt) in runs_dir to free up space
    if os.path.exists(runs_dir) and not args.resume_interrupted:
        print(
            f"\nScanning '{runs_dir}' to clean up existing training checkpoints and free up disk space..."
        )
        reclaimed_bytes = 0
        for root, dirs, files in os.walk(runs_dir):
            if os.path.basename(root) == "checkpoints":
                for file in files:
                    if file.endswith(".ckpt"):
                        file_path = os.path.join(root, file)
                        try:
                            file_size = os.path.getsize(file_path)
                            os.remove(file_path)
                            reclaimed_bytes += file_size
                            print(
                                f"[CLEANUP] Deleted checkpoint: {file_path} ({file_size / (1024**3):.2f} GB)"
                            )
                        except Exception as e:
                            print(f"[WARNING] Failed to delete {file_path}: {e}")
        if reclaimed_bytes > 0:
            print(
                f"[CLEANUP] Successfully reclaimed {reclaimed_bytes / (1024**3):.2f} GB of disk space!\n"
            )

    # Fetch best sweep configurations
    best_aug_config = None
    if aug_sweep_id:
        print(
            f"\nFetching best configuration from augmentation sweep: {aug_sweep_id}..."
        )
        try:
            best_aug_config = cfg.get_best_sweep_config(
                aug_sweep_id, cfg.project, cfg.entity
            )
            print("Successfully loaded best augmentation configurations.")
        except Exception as e:
            print(
                f"Error fetching augmentation sweep config: {e}. Proceeding without sweep results.",
                file=sys.stderr,
            )

    best_hpo_config = None
    if hpo_sweep_id:
        print(f"\nFetching best configuration from HPO sweep: {hpo_sweep_id}...")
        try:
            best_hpo_config = cfg.get_best_sweep_config(
                hpo_sweep_id, cfg.project, cfg.entity
            )
            print("Successfully loaded best HPO configurations.")
        except Exception as e:
            print(
                f"Error fetching HPO sweep config: {e}. Proceeding without sweep results.",
                file=sys.stderr,
            )

    # 1. Dataset fraction management
    dataset_path = args.dataset if args.dataset is not None else cfg.dataset_path
    temp_subset_dir = ""
    if fraction < 1.0:
        import uuid

        temp_subset_dir = os.path.join(runs_dir, f"temp_subset_{uuid.uuid4().hex[:8]}")
        dataset_path = prepare_subset_dataset(dataset_path, fraction, temp_subset_dir)
        print(
            f"Created temporary subset dataset at {dataset_path} containing {fraction * 100}% of files."
        )
    else:
        dataset_path = args.dataset if args.dataset is not None else cfg.dataset_path

    # Warn if deimv2 is referenced
    for m in models_to_train:
        if "deim" in m.lower():
            print(
                "\nWARNING: Model deimv2 does not exist in lightly_train library! Skipping."
            )
            models_to_train.remove(m)

    print("=" * 60)
    print("Starting stock training runs configuration:")
    print(f"Backend: {args.backend}")
    if args.backend == "lightly":
        print(f"Decoder Head: {decoder_name}")
    print(f"Models: {models_to_train}")
    print(f"Seeds: {seeds_to_use}")
    print(f"Epochs: {epochs_display}")
    print(f"Batch Size: {batch_size_display}")
    print(f"Image Size: {imgsz}")
    print(f"Device: {device}")
    print(f"Workers: {workers}")
    print(f"Dataset Fraction: {fraction}")
    print(f"Runs Dir: {runs_dir}")
    print(f"W&B Dir: {wandb_dir}")
    print(f"Dataset: {dataset_path}")
    print(f"W&B Entity: {cfg.entity}")
    print(f"W&B Project: {cfg.project}")
    print(f"Aug Sweep ID: {aug_sweep_id}")
    print(f"HPO Sweep ID: {hpo_sweep_id}")
    print(f"Distill pretrain enabled: {args.distill}")
    print("=" * 60)

    total_runs = len(models_to_train) * len(seeds_to_use)
    run_idx = 1

    for model_name in models_to_train:
        model_base = model_name.replace(".pt", "")

        # Resolve model-specific defaults if not overridden by command line arguments
        model_epochs, model_patience, model_batch_size = cfg.get_model_params(
            model_name, args.backend
        )
        epochs = args.epochs if args.epochs is not None else model_epochs
        batch_size = args.batch if args.batch is not None else model_batch_size
        patience = args.patience if args.patience is not None else model_patience

        # For lightly backend, batch_size=-1 (auto-batching) is not supported.
        # We fall back to "auto" which is supported by lightly_train.
        if args.backend == "lightly" and batch_size == -1:
            batch_size = "auto"

        # Calculate steps dynamically for lightly backend
        if args.backend == "lightly":
            import math

            with open(dataset_path, "r") as f:
                data_yaml = yaml.safe_load(f)

            # Resolve absolute path for training images
            base_path = data_yaml.get("path", "")
            if not os.path.isabs(base_path):
                base_path = os.path.abspath(
                    os.path.join(os.path.dirname(dataset_path), base_path)
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
            if args.raw_steps:
                dino_steps = steps_raw
            else:
                dino_steps = math.ceil(steps_raw / 1000) * 1000

            print("\n" + "=" * 60)
            print(" DINO DYNAMIC STEP CALCULATION")
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
        else:
            dino_steps = epochs

        # 2. Distillation Pretraining (If enabled and using lightly backend)
        distilled_weights = None
        if args.distill and args.backend == "lightly":
            # Map model name to student pretraining backbone
            # E.g. yolo12n.pt -> ultralytics/yolo12n.yaml
            student_name = LIGHTLY_BASELINE_MAP.get(
                model_name, f"ultralytics/{model_name.replace('.pt', '.yaml')}"
            )

            # If student_name corresponds to rtdetr-l.pt, map student backbone to ConvNeXt-Large
            if "rtdetr" in model_name.lower():
                student_name = "dinov3/convnext-large"

            distill_out_dir = os.path.join(runs_dir, "distill", f"distill_{model_base}")

            # Read image dir path from dataset config
            with open(dataset_path, "r") as f:
                data_yaml = yaml.safe_load(f)
            train_img_dir = os.path.join(data_yaml["path"], data_yaml["train"])

            try:
                distilled_weights = run_distillation_pretrain(
                    student_name=student_name,
                    teacher_name="dinov3/vitl16",
                    train_img_dir=train_img_dir,
                    out_dir=distill_out_dir,
                    steps=dino_steps,
                    seed=42,
                )
            except Exception as e:
                print(
                    f"Error during distillation: {e}. Proceeding with standard training.",
                    file=sys.stderr,
                )

        for seed in seeds_to_use:
            eval_model = None
            model = None
            suffix = ""
            if aug_sweep_id and hpo_sweep_id:
                suffix = "_best_aug_hpo"
            elif aug_sweep_id:
                suffix = "_best_aug"
            elif hpo_sweep_id:
                suffix = "_best_hpo"
            if args.distill and distilled_weights:
                suffix += "_distilled"
            if args.backend == "lightly":
                suffix += f"_{decoder_name}_steps_{dino_steps}"

            run_name = f"{model_base}_seed_{seed}{suffix}"
            print(
                f"\n[{run_idx}/{total_runs}] Initializing training for: {run_name}..."
            )

            # Enforce mathematical seeding for reproducibility
            set_reproducibility(seed)

            # Prepare configuration dictionary for W&B
            wb_run_config = {
                "model": model_name,
                "seed": seed,
                "epochs": epochs,
                "imgsz": imgsz,
                "batch_size": batch_size,
                "device": device,
                "workers": workers,
                "fraction": fraction,
                "runs_dir": runs_dir,
                "wandb_dir": wandb_dir,
                "dataset": dataset_path,
                "backend": args.backend,
            }
            if patience is not None:
                wb_run_config["patience"] = patience
            if args.backend == "lightly":
                wb_run_config["decoder_name"] = decoder_name

            # Initialize W&B run cleanly
            run = wandb.init(
                project=cfg.project,
                entity=cfg.entity,
                name=run_name,
                dir=wandb_dir,
                config=wb_run_config,
                reinit=True,
                tags=args.tags,
            )

            # 3. Train Model
            try:
                if args.backend == "ultralytics":
                    # --- Ultralytics Training Backend ---
                    is_rtdetr = "rtdetr" in model_name.lower()
                    if is_rtdetr:
                        from ultralytics import RTDETR

                        print(f"Loading RT-DETR model: {model_name}")
                        model = RTDETR(model_name)
                    else:
                        print(f"Loading YOLO model: {model_name}")
                        model = YOLO(model_name)

                    train_kwargs = {
                        "data": dataset_path,
                        "epochs": epochs,
                        "imgsz": imgsz,
                        "device": device,
                        "batch": batch_size,
                        "seed": seed,
                        "workers": workers,
                        "project": runs_dir,
                        "name": run_name,
                        "exist_ok": True,
                        "amp": amp,
                    }
                    if patience is not None:
                        train_kwargs["patience"] = patience

                    if (aug_sweep_id or hpo_sweep_id) and cfg.fixed_loss:
                        train_kwargs.update(cfg.fixed_loss)

                    if best_aug_config:
                        # Disable default YOLO spatial/color augmentations to isolate sweep influence
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
                                    "mosaic": best_aug_config.get("mosaic", 0.0),
                                    "mixup": best_aug_config.get("mixup", 0.0),
                                    "copy_paste": best_aug_config.get(
                                        "copy_paste", 0.0
                                    ),
                                }
                            )
                        train_kwargs["augmentations"] = build_albumentations_pipeline(
                            best_aug_config, imgsz
                        )

                    if best_hpo_config:
                        hpo_keys = [
                            "optimizer",
                            "lr0",
                            "lrf",
                            "momentum",
                            "weight_decay",
                            "warmup_epochs",
                        ]
                        hpo_params = {
                            k: best_hpo_config[k]
                            for k in hpo_keys
                            if k in best_hpo_config
                        }
                        train_kwargs.update(hpo_params)

                    try:
                        model.train(**train_kwargs)
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower() and device != "cpu":
                            print(
                                f"\n[WARNING] CUDA OOM occurred training {run_name}. Retrying on CPU...",
                                file=sys.stderr,
                            )
                            if wandb.run is not None:
                                try:
                                    wandb.finish(exit_code=1)
                                except Exception:
                                    pass
                            del model
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                            # Reload model and train on CPU
                            if is_rtdetr:
                                from ultralytics import RTDETR

                                model = RTDETR(model_name)
                            else:
                                model = YOLO(model_name)
                            train_kwargs["device"] = "cpu"
                            model.train(**train_kwargs)
                        else:
                            raise e
                    eval_model = model

                else:
                    # --- lightly_train Training Backend ---
                    # Resolve lightly-train model mapping (support custom Hugging Face backbones)
                    if model_name.startswith("facebook/"):
                        if "vits16" in model_name:
                            lightly_model_name = "dinov3/vits16-ltdetr"
                        elif "vitb16" in model_name:
                            lightly_model_name = "dinov3/vitb16-ltdetr"
                        elif "vitt16" in model_name:
                            lightly_model_name = "dinov3/vitt16-ltdetr"
                        elif "sat493m" in model_name:
                            lightly_model_name = "dinov3/vitl16-ltdetr"
                        else:
                            lightly_model_name = "dinov3/vitl16-ltdetr"
                        hf_weights = get_huggingface_backbone(model_name)
                    elif model_name.startswith("dinov3/"):
                        # If a model name is passed like dinov3/vits16 or dinov3/vits16-ltdetr
                        if "-ltdetr" in model_name:
                            lightly_model_name = model_name
                        else:
                            lightly_model_name = f"{model_name}-ltdetr"
                        hf_weights = None
                    else:
                        lightly_model_name = LIGHTLY_BASELINE_MAP.get(
                            model_name,
                            f"ultralytics/{model_name.replace('.pt', '.yaml')}",
                        )
                        hf_weights = None

                    # Convert device (index to device string)
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

                    # Setup defaults
                    model_args = {
                        "lr": 0.001,
                        "weight_decay": 0.0001,
                        "scheduler_name": "flat-cosine",
                        "decoder_name": decoder_name,
                        "backbone_freeze": backbone_freeze,
                    }
                    if "sat493m" in model_name:
                        model_args["backbone_args"] = {"is_sat493m_weights": True}
                    if dino_steps < 2000:
                        model_args["lr_warmup_steps"] = 0
                        model_args["ema_warmup_steps"] = 0
                        model_args["scheduler_flat_steps"] = 0
                        model_args["scheduler_no_aug_steps"] = 0

                    # Prepare transform arguments dictionary
                    from typing import Any as TypeAny

                    transform_args: dict[str, TypeAny] = {"image_size": (imgsz, imgsz)}

                    if hf_weights:
                        model_args["backbone_weights"] = hf_weights
                    elif distilled_weights:
                        model_args["backbone_weights"] = distilled_weights
                    elif args.backbone_weights:
                        model_args["backbone_weights"] = os.path.abspath(
                            args.backbone_weights
                        )

                    # Apply configurations from W&B Phase 1 & 2 Sweeps
                    if best_aug_config:
                        transform_args.update(  # type: ignore
                            {
                                "random_flip": {
                                    "horizontal_prob": float(
                                        best_aug_config.get("albu_spatial_p", 0.5)
                                    ),
                                    "vertical_prob": float(
                                        best_aug_config.get("albu_spatial_p", 0.5)
                                    ),
                                },
                                "color_jitter": {
                                    "prob": float(
                                        best_aug_config.get("albu_color_p", 0.5)
                                    ),
                                    "strength": 1.0,
                                    "brightness": 0.2,
                                    "contrast": 0.2,
                                    "saturation": 0.2,
                                    "hue": 0.05,
                                },
                                "random_rotate_90": {
                                    "prob": float(
                                        best_aug_config.get("albu_rotate90_p", 0.5)
                                    )
                                },
                            }
                        )

                    if best_hpo_config:
                        model_args.update(
                            {
                                "lr": float(best_hpo_config.get("lr0", 1e-3)),
                                "weight_decay": float(
                                    best_hpo_config.get("weight_decay", 1e-4)
                                ),
                            }
                        )

                    out_path = os.path.join(runs_dir, run_name)

                    # Load and patch dataset config to include format='yolo' for lightly_train validation
                    with open(dataset_path, "r") as f:
                        data_dict = yaml.safe_load(f)
                    data_dict["format"] = "yolo"

                    if cfg.entity:
                        os.environ["WANDB_ENTITY"] = cfg.entity
                    retry_on_cpu = False
                    try:
                        # Use bf16-mixed if GPU supports it to avoid numerical degradation of fp16
                        precision_mode = "32-true"
                        if amp:
                            if (
                                torch.cuda.is_available()
                                and torch.cuda.is_bf16_supported()
                            ):
                                precision_mode = "bf16-mixed"
                            else:
                                precision_mode = "16-mixed"

                        lightly_train.train_object_detection(
                            out=out_path,
                            model=lightly_model_name,
                            data=data_dict,
                            steps=dino_steps,
                            batch_size=batch_size,
                            num_workers=workers,
                            devices=device_arg,
                            accelerator=accel,
                            precision=precision_mode,
                            seed=seed,
                            overwrite=False if args.resume_interrupted else True,
                            resume_interrupted=args.resume_interrupted,
                            model_args=model_args,
                            transform_args=transform_args,
                            logger_args={
                                "wandb": {"project": cfg.project, "name": run_name}
                            },
                        )
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower() and accel != "cpu":
                            print(
                                f"\n[WARNING] CUDA OOM occurred training {run_name}. Retrying on CPU...",
                                file=sys.stderr,
                            )
                            if wandb.run is not None:
                                try:
                                    wandb.finish(exit_code=1)
                                except Exception:
                                    pass
                            retry_on_cpu = True
                        else:
                            raise e

                    if retry_on_cpu:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                        lightly_train.train_object_detection(
                            out=out_path,
                            model=lightly_model_name,
                            data=data_dict,
                            steps=dino_steps,
                            batch_size=batch_size,
                            num_workers=workers,
                            devices="auto",
                            accelerator="cpu",
                            precision="32-true",
                            seed=seed,
                            overwrite=False if args.resume_interrupted else True,
                            resume_interrupted=args.resume_interrupted,
                            model_args=model_args,
                            transform_args=transform_args,
                            logger_args={
                                "wandb": {"project": cfg.project, "name": run_name}
                            },
                        )

                    # Load best checkpoint for validation evaluation
                    best_ckpt = os.path.join(
                        out_path, "exported_models", "exported_best.pt"
                    )
                    if not os.path.exists(best_ckpt):
                        best_ckpt = os.path.join(
                            out_path, "exported_models", "exported_last.pt"
                        )

                    from eval_utils import safe_load_model

                    eval_model = safe_load_model(best_ckpt)

                    # Clean up large training checkpoints (.ckpt files) to conserve space
                    ckpt_dir = os.path.join(out_path, "checkpoints")
                    if os.path.exists(ckpt_dir):
                        try:
                            shutil.rmtree(ckpt_dir)
                            print(
                                f"[CLEANUP] Successfully removed checkpoints directory to conserve disk space: {ckpt_dir}"
                            )
                        except Exception as e:
                            print(
                                f"[WARNING] Failed to clean up checkpoints directory: {e}"
                            )

                # 4. Strict Evaluation
                eval_batch_size = (
                    cfg.eval_fallback_batch_size
                    if (batch_size == "auto" or batch_size == -1)
                    else batch_size
                )
                try:
                    metrics = evaluate_model_coco(
                        model_path_or_model=eval_model,
                        dataset_yaml_path=dataset_path,
                        split="test",
                        eval_results_dir=eval_results_dir,
                        run_name=run_name,
                        device=device,
                        batch_size=eval_batch_size,
                        imgsz=imgsz,
                        workers=workers,
                    )
                except RuntimeError as e:
                    if "out of memory" in str(e).lower() and device != "cpu":
                        print(
                            f"\n[WARNING] CUDA OOM occurred during evaluation of {run_name}. Retrying on CPU...",
                            file=sys.stderr,
                        )
                        metrics = evaluate_model_coco(
                            model_path_or_model=eval_model,
                            dataset_yaml_path=dataset_path,
                            split="test",
                            eval_results_dir=eval_results_dir,
                            run_name=run_name,
                            device="cpu",
                            batch_size=eval_batch_size,
                            imgsz=imgsz,
                            workers=workers,
                        )
                    else:
                        raise e

                # Log metrics to W&B (resume if closed by Ultralytics)
                if wandb.run is None:
                    run = wandb.init(
                        project=cfg.project, entity=cfg.entity, id=run.id, resume="must"
                    )
                wandb.log({f"metrics/{k}": v for k, v in metrics["metrics"].items()})

            except Exception as e:
                import traceback

                print(
                    f"Error occurred during training of {run_name}: {e}",
                    file=sys.stderr,
                )
                traceback.print_exc()
                sys.exit(1)
            finally:
                wandb.finish()

                # Free GPU memory to prevent memory leaks in loop
                eval_model = None
                model = None
                gc.collect()
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

                run_idx += 1

    # Cleanup temporary dataset folder
    if fraction < 1.0 and os.path.exists(temp_subset_dir):
        try:
            shutil.rmtree(temp_subset_dir)
        except Exception:
            pass

    print("\nAll training runs complete!")


if __name__ == "__main__":
    main()
