"""Standalone Knowledge Distillation & Fine-Tuning Execution Script.

Distills features from Meta's DINOv3 Vision Foundation Models into efficient YOLO models
(e.g., YOLO12s, YOLO12n) using LightlyTrain's distillationv3 method, followed by
supervised fine-tuning on labeled data and strict COCO evaluation.

Supports:
1. Teacher Model Selection:
   - `dinov3/vitl16-sat493m` (Meta 493M Satellite/Aerial Pretrained Vision Foundation)
   - `dinov3/vitl16` (Meta 1.68B ImageNet/LVD Pretrained Vision Foundation)
2. Student Model Selection:
   - `yolo12s` / `ultralytics/yolo12s.yaml` (Recommended Master Student)
   - `yolo12n` / `ultralytics/yolo12n.yaml` (Nano Edge Student)
3. Separate W&B Run Tracking:
   - Stage 1: Pretraining W&B run (`pretrain_distill_<teacher>_to_<student>`)
   - Stage 2: Fine-Tuning & COCO Eval W&B run (`finetune_distill_<teacher>_to_<student>`)
"""

# ruff: noqa: E402
import argparse
import gc
import os
import sys

os.environ["PYTHONIOENCODING"] = "utf-8"
_reconfig_out = getattr(sys.stdout, "reconfigure", None)
if callable(_reconfig_out):
    _reconfig_out(encoding="utf-8", errors="replace")
_reconfig_err = getattr(sys.stderr, "reconfigure", None)
if callable(_reconfig_err):
    _reconfig_err(encoding="utf-8", errors="replace")

from PIL import Image  # noqa: F401  # Fix for Windows DLL load order issue with torchvision/_imaging
import lightly_train
import torch
from ultralytics import YOLO
import wandb
from config import PipelineConfig
from eval_utils import evaluate_model_coco, parse_dataset_yaml, safe_load_model


def resolve_distillation_data_paths(
    dataset_yaml_path: str, custom_data_arg: str | None = None
) -> str | list[str]:
    """Resolves valid image directory path(s) for lightly_train.pretrain.

    If dataset_yaml_path points to a YAML file (e.g. dataset.yaml), it extracts the
    training images directory path (path + train).
    """
    train_dir = None
    if os.path.exists(dataset_yaml_path):
        if dataset_yaml_path.endswith(".yaml") or dataset_yaml_path.endswith(".yml"):
            db_info = parse_dataset_yaml(dataset_yaml_path)
            base_p = db_info.get("path", os.path.dirname(dataset_yaml_path))
            train_rel = db_info.get("train", "train/images")
            candidate = (
                os.path.join(base_p, train_rel)
                if not os.path.isabs(train_rel)
                else train_rel
            )
            if os.path.exists(candidate):
                train_dir = os.path.abspath(candidate)
        elif os.path.isdir(dataset_yaml_path):
            train_dir = os.path.abspath(dataset_yaml_path)

    dirs = []
    if train_dir and os.path.isdir(train_dir):
        dirs.append(train_dir)

    if custom_data_arg:
        custom_p = os.path.abspath(custom_data_arg)
        if os.path.isdir(custom_p):
            dirs.append(custom_p)
        elif os.path.exists(custom_p):
            dirs.append(custom_p)

    if not dirs:
        raise ValueError(
            f"Could not resolve any valid image directory from dataset_path '{dataset_yaml_path}' and custom_data '{custom_data_arg}'"
        )

    return dirs[0] if len(dirs) == 1 else dirs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Distill DINOv3 (SAT-493M / LVD-1689M) knowledge into YOLO models with fine-tuning & COCO evaluation."
    )
    parser.add_argument(
        "--teacher",
        type=str,
        default="dinov3/vitl16-sat493m",
        choices=["dinov3/vitl16-sat493m", "dinov3/vitl16", "dinov3/vitb16"],
        help="Teacher model variant (default: dinov3/vitl16-sat493m).",
    )
    parser.add_argument(
        "--student",
        type=str,
        default="yolo12s",
        choices=[
            "yolo12s",
            "yolo12n",
            "yolo26s",
            "yolo26n",
            "yolo11s",
            "yolo11n",
            "dinov3/vitt16",
            "dinov3/vits16",
            "dinov3/vitb16",
        ],
        help="Student architecture variant (e.g. yolo12s, dinov3/vitt16; default: yolo12s).",
    )
    parser.add_argument(
        "--epochs",
        type=str,
        default="100",
        help="Pretraining distillation epochs (default: '100'; LightlyTrain recommends 100 to 300+, or 'auto').",
    )
    parser.add_argument(
        "--finetune-epochs",
        type=str,
        default="300",
        help="Supervised fine-tuning epochs on labeled dataset.yaml (default: '300'; accepts 'auto' or integer).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=30,
        help="Epoch patience for early stopping during YOLO supervised fine-tuning (default: 30; set 0 to disable).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Global batch size across devices (default: 32).",
    )
    parser.add_argument(
        "--pretrain-batch-size",
        type=int,
        default=None,
        help="Specific batch size for Stage 1 pretraining, e.g. 32 or 64 at 512px (default: fallback to --batch_size).",
    )
    parser.add_argument(
        "--finetune-batch-size",
        type=int,
        default=None,
        help="Specific batch size for Stage 2 fine-tuning, e.g. 16 at 1024px (default: fallback to --batch_size).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0 if sys.platform == "win32" else 4,
        help="Dataloader worker processes (default: 0 on Windows to prevent .mmap IPC deadlock, 4 on Linux).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=512,
        help="Global image resolution in pixels for pretraining & fine-tuning (default: 512).",
    )
    parser.add_argument(
        "--pretrain-imgsz",
        type=int,
        default=None,
        help="Specific image resolution for Stage 1 pretraining (default: fallback to --imgsz).",
    )
    parser.add_argument(
        "--finetune-imgsz",
        type=int,
        default=None,
        help="Specific image resolution for Stage 2 fine-tuning, e.g. 1024 (default: fallback to --imgsz).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="CUDA device index or 'auto' (default: '0').",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Optional path to custom folder with pretraining/unlabeled images (default: dataset train/images).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output directory path (default: auto-generated inside runs/distill/).",
    )
    parser.add_argument(
        "--decoder",
        type=str,
        default="dfine",
        choices=["dfine", "rtdetrv2"],
        help="Transformer decoder head selection for DINO foundation models (default: dfine).",
    )
    parser.add_argument(
        "--raw-steps",
        action="store_true",
        help="Disable rounding to the next full 1,000 steps for DINO fine-tuning, using exact raw steps.",
    )
    parser.add_argument(
        "--skip-pretrain",
        action="store_true",
        help="Skip Stage 1 pretraining distillation and proceed directly to Stage 2 fine-tuning using existing exported weights.",
    )
    parser.add_argument(
        "--skip-finetune",
        action="store_true",
        help="Skip supervised fine-tuning and evaluation after distillation pretraining.",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Skip pretraining & fine-tuning to evaluate an existing checkpoint.",
    )
    parser.add_argument(
        "--wandb-offline",
        action="store_true",
        help="Run Weights & Biases in offline mode.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = PipelineConfig()

    if args.wandb_offline:
        os.environ["WANDB_MODE"] = "offline"

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["PYTHONNOUSERSITE"] = "1"

    teacher_tag = "sat493m" if "sat493m" in args.teacher else "lvd1689m"
    student_tag = args.student.replace(".pt", "").replace("/", "_")

    base_run_name = f"distill_{teacher_tag}_to_{student_tag}"
    pretrain_run_name = f"pretrain_{base_run_name}"
    finetune_run_name = f"finetune_{base_run_name}"

    if args.out:
        out_dir = os.path.abspath(args.out)
    else:
        out_dir = os.path.abspath(os.path.join(cfg.runs_dir, "distill", base_run_name))

    pretrain_out = os.path.join(out_dir, "pretrain")
    finetune_out = os.path.join(out_dir, "finetune")

    # Resolve student model string for LightlyTrain
    if "dinov3/" in args.student:
        student_model_yaml = args.student
    else:
        raw_name = args.student.replace(".pt", "")
        student_model_yaml = f"ultralytics/{raw_name}.yaml"

    # Setup hardware accelerator
    accel = "gpu" if torch.cuda.is_available() and args.device != "cpu" else "cpu"
    device_arg = [int(args.device)] if args.device.isdigit() else "auto"

    # Resolve image resolution for pretraining vs fine-tuning
    pretrain_imgsz = (
        args.pretrain_imgsz if args.pretrain_imgsz is not None else args.imgsz
    )
    finetune_imgsz = (
        args.finetune_imgsz if args.finetune_imgsz is not None else args.imgsz
    )

    # Resolve batch size for pretraining vs fine-tuning
    pretrain_batch_size = (
        args.pretrain_batch_size
        if args.pretrain_batch_size is not None
        else args.batch_size
    )
    finetune_batch_size = (
        args.finetune_batch_size
        if args.finetune_batch_size is not None
        else args.batch_size
    )

    epochs_val = int(args.epochs) if args.epochs.isdigit() else "auto"
    finetune_epochs_val: int | str = (
        int(args.finetune_epochs)
        if str(args.finetune_epochs).isdigit()
        else ("auto" if "dinov3/" in args.student else 300)
    )
    data_input = resolve_distillation_data_paths(cfg.dataset_path, args.data)

    best_distill_ckpt = os.path.join(
        pretrain_out, "exported_models", "exported_best.pt"
    )
    if not os.path.exists(best_distill_ckpt):
        best_distill_ckpt = os.path.join(
            pretrain_out, "exported_models", "exported_last.pt"
        )

    # =========================================================================
    # STAGE 1: Distillation Pretraining (LightlyTrain)
    # =========================================================================
    if not args.eval_only and not args.skip_pretrain:
        print("\n" + "=" * 80)
        print(" STAGE 1: KNOWLEDGE DISTILLATION PRETRAINING")
        print("=" * 80)
        print(f" Teacher Model:          {args.teacher}")
        print(f" Student Architecture:   {args.student}")
        print(f" Pretrain Epoch Budget:  {epochs_val}")
        print(f" Pretrain Batch Size:    {pretrain_batch_size}")
        print(f" Pretrain Image Size:    {pretrain_imgsz} px")
        print(f" Target Pretrain Output: {pretrain_out}")
        print(f" W&B Pretrain Run Name:  {pretrain_run_name} (Group: {base_run_name})")
        print("=" * 80 + "\n")

        # Initialize W&B Run 1 (Pretraining)
        wandb.init(
            project=cfg.project,
            entity=cfg.entity,
            name=pretrain_run_name,
            group=base_run_name,
            job_type="pretrain",
            tags=["distillation", "pretrain", teacher_tag, student_tag],
            config={
                "stage": "distillation_pretraining",
                "teacher": args.teacher,
                "student": args.student,
                "epochs": epochs_val,
                "pretrain_batch_size": pretrain_batch_size,
                "finetune_batch_size": finetune_batch_size,
                "pretrain_imgsz": pretrain_imgsz,
                "finetune_imgsz": finetune_imgsz,
                "device": args.device,
                "dataset_path": cfg.dataset_path,
            },
            reinit=True,
        )

        lightly_train.pretrain(
            out=pretrain_out,
            data=data_input,
            model=student_model_yaml,
            method="distillationv3",
            epochs=epochs_val,
            batch_size=pretrain_batch_size,
            num_workers=args.num_workers,
            devices=device_arg,
            accelerator=accel,
            precision="16-mixed" if cfg.amp else "32-true",
            seed=42,
            overwrite=True,
            method_args={
                "teacher": args.teacher,
                "loss_local_weight": 1.0,
            },
            transform_args={"image_size": [pretrain_imgsz, pretrain_imgsz]},
            loggers={"wandb": {"project": cfg.project, "name": pretrain_run_name}},
        )

        wandb.finish()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        best_distill_ckpt = os.path.join(
            pretrain_out, "exported_models", "exported_best.pt"
        )
        if not os.path.exists(best_distill_ckpt):
            best_distill_ckpt = os.path.join(
                pretrain_out, "exported_models", "exported_last.pt"
            )

    # =========================================================================
    # STAGE 2: Supervised Fine-Tuning on Labeled Dataset
    # =========================================================================
    finetuned_ckpt = None
    if not args.eval_only and not args.skip_finetune:
        if not os.path.exists(best_distill_ckpt):
            print(
                f"\nWarning: Pretrained checkpoint not found at {best_distill_ckpt}. Cannot perform fine-tuning."
            )
        else:
            print("\n" + "=" * 80)
            print(" STAGE 2: SUPERVISED FINE-TUNING ON LABELED DATASET")
            print("=" * 80)
            print(f" Distilled Base Weights: {best_distill_ckpt}")
            print(f" Fine-tune Epoch Budget: {finetune_epochs_val}")
            print(f" Fine-tune Batch Size:   {finetune_batch_size}")
            print(f" Target Fine-tune Output:{finetune_out}")
            print(
                f" W&B Fine-tune Run Name: {finetune_run_name} (Group: {base_run_name})"
            )
            print("=" * 80 + "\n")

            # Initialize W&B Run 2 (Fine-tuning)
            wandb.init(
                project=cfg.project,
                entity=cfg.entity,
                name=finetune_run_name,
                group=base_run_name,
                job_type="finetune",
                tags=["distillation", "finetune", teacher_tag, student_tag],
                config={
                    "stage": "supervised_finetuning",
                    "teacher": args.teacher,
                    "student": args.student,
                    "finetune_epochs": finetune_epochs_val,
                    "pretrain_batch_size": pretrain_batch_size,
                    "finetune_batch_size": finetune_batch_size,
                    "pretrain_imgsz": pretrain_imgsz,
                    "finetune_imgsz": finetune_imgsz,
                    "device": args.device,
                    "pretrained_checkpoint": best_distill_ckpt,
                    "dataset_path": cfg.dataset_path,
                },
                reinit=True,
            )

            if "dinov3/" in args.student:
                import yaml

                with open(cfg.dataset_path, "r") as f:
                    data_dict = yaml.safe_load(f)
                data_dict["format"] = "yolo"

                lightly_model_name = (
                    "dinov3/vitt16-ltdetr"
                    if "vitt16" in args.student
                    else f"{args.student}-ltdetr"
                )

                import math

                num_train_images = 5553
                effective_batch_size = (
                    max(32, finetune_batch_size)
                    if isinstance(finetune_batch_size, int)
                    and finetune_batch_size >= 32
                    else 32
                )
                if isinstance(finetune_epochs_val, int):
                    steps_raw = math.ceil(
                        finetune_epochs_val * (num_train_images / effective_batch_size)
                    )
                    if args.raw_steps:
                        finetune_steps = steps_raw
                    else:
                        finetune_steps = math.ceil(steps_raw / 1000) * 1000
                else:
                    finetune_steps = finetune_epochs_val

                checkpoint_arg = None
                model_args_dict = {"decoder_name": args.decoder}
                if os.path.exists(best_distill_ckpt):
                    model_args_dict["backbone_weights"] = best_distill_ckpt

                lightly_train.train_object_detection(
                    out=finetune_out,
                    model=lightly_model_name,
                    data=data_dict,
                    checkpoint=checkpoint_arg,
                    steps=finetune_steps,
                    batch_size=finetune_batch_size,
                    num_workers=args.num_workers,
                    devices=device_arg,
                    accelerator=accel,
                    precision="16-mixed" if cfg.amp else "32-true",
                    seed=42,
                    overwrite=True,
                    model_args=model_args_dict,
                    logger_args={
                        "wandb": {
                            "project": cfg.project,
                            "name": finetune_run_name,
                        }
                    },
                )

                candidate_best = os.path.join(
                    finetune_out, "exported_models", "exported_best.pt"
                )
                if not os.path.exists(candidate_best):
                    candidate_best = os.path.join(
                        finetune_out, "exported_models", "exported_last.pt"
                    )
                finetuned_ckpt = candidate_best
            else:
                student_model = YOLO(best_distill_ckpt)
                student_model.train(
                    data=cfg.dataset_path,
                    epochs=finetune_epochs_val
                    if isinstance(finetune_epochs_val, int)
                    else 300,
                    batch=finetune_batch_size,
                    workers=args.num_workers,
                    imgsz=finetune_imgsz,
                    device=args.device,
                    project=os.path.dirname(finetune_out),
                    name=os.path.basename(finetune_out),
                    patience=args.patience,
                    exist_ok=True,
                    plots=True,
                    amp=cfg.amp,
                    seed=42,
                )

                candidate_best = os.path.join(finetune_out, "weights", "best.pt")
                if os.path.exists(candidate_best):
                    finetuned_ckpt = candidate_best
                else:
                    finetuned_ckpt = os.path.join(finetune_out, "weights", "last.pt")

    if finetuned_ckpt is None:
        possible_ckpts = [
            os.path.join(finetune_out, "weights", "best.pt"),
            os.path.join(finetune_out, "exported_models", "exported_best.pt"),
            os.path.join(finetune_out, "weights", "last.pt"),
            os.path.join(finetune_out, "exported_models", "exported_last.pt"),
        ]
        for p in possible_ckpts:
            if os.path.exists(p):
                finetuned_ckpt = p
                break

    # =========================================================================
    # STAGE 3: Strict COCO Evaluation on Fine-Tuned Model
    # =========================================================================
    eval_ckpt = (
        finetuned_ckpt
        if (finetuned_ckpt and os.path.exists(finetuned_ckpt))
        else best_distill_ckpt
    )

    if eval_ckpt and os.path.exists(eval_ckpt):
        print("\n" + "=" * 80)
        print(" STAGE 3: EXECUTING STRICT COCOEVALUATION ON FINE-TUNED MODEL")
        print("=" * 80)
        print(f" Evaluation Checkpoint:  {eval_ckpt}")
        print("=" * 80 + "\n")

        if wandb.run is None:
            wandb.init(
                project=cfg.project,
                entity=cfg.entity,
                name=finetune_run_name,
                group=base_run_name,
                job_type="eval",
                reinit=True,
            )

        eval_model = safe_load_model(eval_ckpt)

        print("Evaluating on Validation Set (val)...")
        _ = evaluate_model_coco(
            model_path_or_model=eval_model,
            dataset_yaml_path=cfg.dataset_path,
            split="val",
            eval_results_dir=cfg.eval_results_dir,
            run_name=f"{finetune_run_name}_VAL",
            device=args.device,
            batch_size=args.batch_size,
            imgsz=cfg.image_size,
            workers=4,
        )

        print("\nEvaluating on Hold-Out Test Set (test)...")
        metrics = evaluate_model_coco(
            model_path_or_model=eval_model,
            dataset_yaml_path=cfg.dataset_path,
            split="test",
            eval_results_dir=cfg.eval_results_dir,
            run_name=f"{finetune_run_name}_TEST",
            device=args.device,
            batch_size=args.batch_size,
            imgsz=cfg.image_size,
            workers=4,
        )

        # Log evaluation metrics to Fine-tuning W&B run
        log_dict = {f"metrics/{k}": v for k, v in metrics["metrics"].items()}
        if "AP50" in metrics["metrics"]:
            log_dict["metrics/mAP50(B)"] = metrics["metrics"]["AP50"]
        wandb.log(log_dict)

        print("\n" + "=" * 80)
        print(f" FINE-TUNED {student_tag.upper()} COCOEVAL PERFORMANCE RESULTS")
        print("-" * 80)
        print(f" mAP (50-95):   {metrics['metrics'].get('AP', 0.0):.4f}")
        print(f" mAP50:         {metrics['metrics'].get('AP50', 0.0):.4f}")
        print(
            f" mAP30:         {metrics['metrics'].get('AP30', metrics['metrics'].get('mAP30', 0.0)):.4f}"
        )
        print(f" AR_max100:     {metrics['metrics'].get('AR_max100', 0.0):.4f}")
        print("=" * 80 + "\n")
    else:
        print(f"\nWarning: Checkpoint not found at {eval_ckpt}. Skipping evaluation.")

    if wandb.run is not None:
        wandb.finish()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
