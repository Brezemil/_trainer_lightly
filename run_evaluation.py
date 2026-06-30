"""
Standalone Evaluation Script.

This script runs strict COCO evaluation (via pycocotools) on the test split for trained checkpoints.
It checks for checkpoints at runs/{run_name}/weights/best.pt or runs/{run_name}/exported_models/exported_best.pt.
"""

import pyarrow  # noqa: F401
import argparse
import os
import sys
from config import PipelineConfig
from eval_utils import evaluate_model_coco


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate trained YOLO/RT-DETR checkpoints on the test split."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Specify a single model to evaluate (e.g. yolo11s.pt, yolo26s.pt, rtdetr-l.pt).",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Specify a single seed to evaluate."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate on (default: test).",
    )
    parser.add_argument("--batch", type=int, default=None, help="Override batch size.")
    parser.add_argument(
        "--device", type=str, default=None, help="Override device (e.g., 0 or cpu)."
    )
    parser.add_argument("--imgsz", type=int, default=None, help="Override image size.")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Override number of dataloader workers.",
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default=None,
        help="Override the runs directory where training checkpoints are saved.",
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

    # SAHI options
    parser.add_argument(
        "--sahi",
        dest="sahi",
        action="store_true",
        help="Enable Slicing Aided Hyper Inference (SAHI) evaluation.",
    )
    parser.add_argument(
        "--no-sahi",
        dest="sahi",
        action="store_false",
        help="Disable Slicing Aided Hyper Inference (SAHI) evaluation.",
    )
    parser.set_defaults(sahi=None)

    parser.add_argument(
        "--sahi-slice-height",
        type=int,
        default=None,
        help="SAHI slice height in pixels.",
    )
    parser.add_argument(
        "--sahi-slice-width", type=int, default=None, help="SAHI slice width in pixels."
    )
    parser.add_argument(
        "--sahi-overlap",
        type=float,
        default=None,
        help="Fractional overlap ratio for SAHI (default from config).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to a different dataset YAML file to evaluate on (overrides PipelineConfig dataset_path).",
    )
    parser.add_argument(
        "--decoder",
        type=str,
        choices=["rtdetrv2", "dfine"],
        default=None,
        help="Specify the decoder head of LTDETR models to evaluate (rtdetrv2 or dfine). Defaults to configuration setting.",
    )
    parser.add_argument(
        "--upload-wandb",
        action="store_true",
        help="Upload evaluation results back to the original training run in Weights & Biases.",
    )
    parser.add_argument(
        "--tags",
        type=str,
        nargs="+",
        default=None,
        help="List of tags to append to the resumed Weights & Biases run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = PipelineConfig()

    # Resolve parameters
    models_to_eval = [args.model] if args.model else list(cfg.models)
    seeds_to_use = [args.seed] if args.seed is not None else list(cfg.seeds)

    split = args.split
    batch_size = args.batch if args.batch is not None else cfg.batch_size
    device = args.device if args.device is not None else cfg.device
    imgsz = args.imgsz if args.imgsz is not None else cfg.image_size
    workers = args.workers if args.workers is not None else cfg.workers
    runs_dir = args.runs_dir if args.runs_dir is not None else cfg.runs_dir
    dataset_yaml_path = args.dataset if args.dataset is not None else cfg.dataset_path

    aug_sweep_id = (
        args.aug_sweep_id if args.aug_sweep_id is not None else cfg.aug_sweep_id
    )
    hpo_sweep_id = (
        args.hpo_sweep_id if args.hpo_sweep_id is not None else cfg.hpo_sweep_id
    )
    decoder_name = args.decoder if args.decoder is not None else cfg.decoder_name

    # Resolve SAHI parameters
    sahi_enabled = args.sahi if args.sahi is not None else cfg.sahi_enabled
    sahi_slice_height = (
        args.sahi_slice_height
        if args.sahi_slice_height is not None
        else cfg.sahi_slice_height
    )
    sahi_slice_width = (
        args.sahi_slice_width
        if args.sahi_slice_width is not None
        else cfg.sahi_slice_width
    )
    sahi_overlap = (
        args.sahi_overlap
        if args.sahi_overlap is not None
        else cfg.sahi_overlap_height_ratio
    )

    # Construct suffix indicating sweep integration
    suffix = ""
    if aug_sweep_id and hpo_sweep_id:
        suffix = "_best_aug_hpo"
    elif aug_sweep_id:
        suffix = "_best_aug"
    elif hpo_sweep_id:
        suffix = "_best_hpo"

    # Resolve relative path to absolute
    if not os.path.isabs(runs_dir):
        runs_dir = os.path.abspath(runs_dir)

    print("=" * 60)
    print(f"Starting COCO evaluation on {split} split:")
    print(f"Models: {models_to_eval}")
    print(f"Seeds: {seeds_to_use}")
    print(f"Image Size: {imgsz}")
    print(f"Device: {device}")
    print(f"Workers: {workers}")
    print(f"Runs Directory: {runs_dir}")
    print(f"Dataset: {dataset_yaml_path}")
    print(f"Results Directory: {cfg.eval_results_dir}")
    print(f"Aug Sweep ID: {aug_sweep_id}")
    print(f"HPO Sweep ID: {hpo_sweep_id}")
    print(f"Decoder Head: {decoder_name}")
    print(f"SAHI Enabled: {sahi_enabled}")
    if sahi_enabled:
        print(f"SAHI Slice Size: {sahi_slice_height}x{sahi_slice_width}")
        print(f"SAHI Overlap: {sahi_overlap}")
    print("=" * 60)

    evaluated_count = 0
    missing_count = 0

    for model_name in models_to_eval:
        for seed in seeds_to_use:
            model_base = model_name.replace(".pt", "")

            # Try with decoder suffix first
            run_name_with_dec = f"{model_base}_seed_{seed}{suffix}_{decoder_name}"
            checkpoint_path = os.path.join(
                runs_dir, run_name_with_dec, "weights", "best.pt"
            )
            if not os.path.exists(checkpoint_path):
                checkpoint_path = os.path.join(
                    runs_dir, run_name_with_dec, "exported_models", "exported_best.pt"
                )

            if not os.path.exists(checkpoint_path):
                # Fallback to without decoder suffix (in case it is a YOLO model or older run)
                run_name_no_dec = f"{model_base}_seed_{seed}{suffix}"
                checkpoint_path = os.path.join(
                    runs_dir, run_name_no_dec, "weights", "best.pt"
                )
                if not os.path.exists(checkpoint_path):
                    checkpoint_path = os.path.join(
                        runs_dir, run_name_no_dec, "exported_models", "exported_best.pt"
                    )
                run_name = run_name_no_dec
            else:
                run_name = run_name_with_dec

            if not os.path.exists(checkpoint_path):
                print(
                    f"Checkpoint not found for {run_name} at: {checkpoint_path}. Skipping."
                )
                missing_count += 1
                continue

            print(f"\nEvaluating found checkpoint: {checkpoint_path}")
            try:
                metrics = evaluate_model_coco(
                    model_path_or_model=checkpoint_path,
                    dataset_yaml_path=dataset_yaml_path,
                    split=split,
                    eval_results_dir=cfg.eval_results_dir,
                    run_name=run_name,
                    device=device,
                    batch_size=batch_size,
                    imgsz=imgsz,
                    workers=workers,
                    sahi_enabled=sahi_enabled,
                    sahi_slice_height=sahi_slice_height,
                    sahi_slice_width=sahi_slice_width,
                    sahi_overlap_height_ratio=sahi_overlap,
                    sahi_overlap_width_ratio=sahi_overlap,
                )
                evaluated_count += 1

                if args.upload_wandb:
                    import wandb

                    print(
                        f"Searching for original W&B run named '{run_name}' in project {cfg.entity}/{cfg.project}..."
                    )
                    try:
                        api = wandb.Api()
                        runs = api.runs(
                            path=f"{cfg.entity}/{cfg.project}",
                            filters={"display_name": run_name},
                        )
                        if len(runs) > 0:
                            run_id = runs[0].id
                            print(
                                f"Found run ID {run_id}. Resuming run to log evaluation metrics..."
                            )
                            # Initialize run in resume mode
                            if args.tags:
                                run = wandb.init(
                                    project=cfg.project,
                                    entity=cfg.entity,
                                    id=run_id,
                                    resume="must",
                                )
                                run.tags = (
                                    list(run.tags) if run.tags is not None else []
                                ) + args.tags
                            else:
                                wandb.init(
                                    project=cfg.project,
                                    entity=cfg.entity,
                                    id=run_id,
                                    resume="must",
                                )
                            # Log metrics to W&B
                            wandb.log(
                                {
                                    f"metrics/{k}": v
                                    for k, v in metrics["metrics"].items()
                                }
                            )
                            wandb.finish()
                            print(
                                f"Successfully uploaded metrics to W&B run {run_name} ({run_id})"
                            )
                        else:
                            print(
                                f"[WARNING] No active W&B run found with name '{run_name}' in project {cfg.entity}/{cfg.project}. Cannot upload metrics."
                            )
                    except Exception as wandb_err:
                        print(
                            f"[WARNING] Failed to upload metrics to W&B: {wandb_err}",
                            file=sys.stderr,
                        )
            except Exception as e:
                print(f"Error evaluating {run_name}: {e}", file=sys.stderr)
            finally:
                # Free GPU memory
                import gc
                import torch

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    print(
        f"\nEvaluation complete. Evaluated: {evaluated_count}, Missing: {missing_count}"
    )


if __name__ == "__main__":
    main()
