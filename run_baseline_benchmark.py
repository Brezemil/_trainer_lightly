"""
Orchestration Script to Run the Entire Baseline Benchmark Suite.

This script executes:
1. YOLO11s, YOLO26s, YOLOv12s, and RT-DETR-L baseline runs using the Ultralytics backend.
2. DINOv3 ViT-L models (with sat493m and lvd1689m backbones) using the LightlyTrain backend.
3. Strict evaluation for each run across 3 seeds using pycocotools.
4. Beautiful plotting and aggregation of the results into a distinctive baseline folder.
"""

import os
import subprocess


def run_cmd(command_args, env=None):
    print(f"\n[ORCHESTRATOR] Executing: {' '.join(command_args)}")
    # Run command and let output stream directly to the terminal
    res = subprocess.run(command_args, env=env, capture_output=False, text=True)
    if res.returncode != 0:
        print(f"[ORCHESTRATOR] Warning: Command exited with code {res.returncode}")
    return res.returncode


def run_training_with_fallback(command_args):
    # Try running the command on default/GPU device first
    rc = run_cmd(command_args)
    if rc != 0:
        print(
            "[ORCHESTRATOR] WARNING: Training command failed on default device. Retrying on CPU with CUDA hidden to guarantee execution..."
        )
        import copy

        cpu_cmd = copy.deepcopy(command_args)

        # Replace '--device' argument if it exists, or append it
        if "--device" in cpu_cmd:
            idx = cpu_cmd.index("--device")
            cpu_cmd[idx + 1] = "cpu"
        else:
            cpu_cmd.extend(["--device", "cpu"])

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""

        print("[ORCHESTRATOR] Executing CPU fallback run...")
        rc_fallback = run_cmd(cpu_cmd, env=env)
        if rc_fallback != 0:
            print(
                f"[ORCHESTRATOR] ERROR: CPU fallback training run also failed with code {rc_fallback}"
            )
            return rc_fallback
        return 0
    return 0


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the entire baseline benchmark suite."
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override training epochs."
    )
    parser.add_argument("--batch", type=int, default=None, help="Override batch size.")
    parser.add_argument("--device", type=str, default=None, help="Override device.")
    parser.add_argument(
        "--fraction", type=float, default=None, help="Override dataset fraction."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Override dataset.yaml configuration file path.",
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default=None,
        help="Override the directory where training checkpoints are saved.",
    )
    parser.add_argument(
        "--eval-results-dir",
        type=str,
        default=None,
        help="Override the directory where evaluation results are saved.",
    )
    parser.add_argument(
        "--workers", type=int, default=None, help="Override dataloader workers."
    )
    parser.add_argument("--imgsz", type=int, default=None, help="Override image size.")
    parser.add_argument(
        "--tags",
        type=str,
        nargs="+",
        default=None,
        help="List of tags to assign to the Weights & Biases runs.",
    )
    # Model selection flags
    parser.add_argument("--yolo12n", action="store_true", help="Run YOLOv12n baseline.")
    parser.add_argument("--yolo12s", action="store_true", help="Run YOLOv12s baseline.")
    parser.add_argument("--yolo26n", action="store_true", help="Run YOLO26n baseline.")
    parser.add_argument("--yolo26s", action="store_true", help="Run YOLO26s baseline.")
    parser.add_argument("--yolo11n", action="store_true", help="Run YOLO11n baseline.")
    parser.add_argument("--yolo11s", action="store_true", help="Run YOLO11s baseline.")
    parser.add_argument(
        "--rtdetr-l", action="store_true", help="Run RT-DETR-L baseline."
    )
    parser.add_argument(
        "--dinov3-l",
        action="store_true",
        help="Run LightlyTrain DINOv3 ViT-L models (with D-FINE & RT-DETRv2 heads).",
    )
    parser.add_argument(
        "--dinov3-sat",
        action="store_true",
        help="Run LightlyTrain custom DINOv3 ViT-L satellite-pretrained backbone (sat493m) only.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Run with a single specific random seed (e.g. 42 for a quick smoketest).",
    )
    parser.add_argument(
        "--dino-epochs",
        type=int,
        default=None,
        help="Override the number of training epochs specifically for DINO-based models.",
    )
    parser.add_argument(
        "--amp",
        type=str,
        default=None,
        help="Enable/disable Automatic Mixed Precision (AMP) (True/False).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine which models to run
    run_yolo12n = args.yolo12n
    run_yolo12s = args.yolo12s
    run_yolo26n = args.yolo26n
    run_yolo26s = args.yolo26s
    run_yolo11n = args.yolo11n
    run_yolo11s = args.yolo11s
    run_rtdetr_l = getattr(args, "rtdetr_l", False)
    run_dinov3_l = args.dinov3_l
    run_dinov3_sat = args.dinov3_sat

    # If no specific flags are selected, run ALL of them by default
    if not (
        run_yolo12n
        or run_yolo12s
        or run_yolo26n
        or run_yolo26s
        or run_yolo11n
        or run_yolo11s
        or run_rtdetr_l
        or run_dinov3_l
        or run_dinov3_sat
    ):
        run_yolo12s = True
        run_yolo26s = True
        run_yolo11s = True
        run_rtdetr_l = True
        run_dinov3_l = True

    # Arguments to forward to run_training.py
    forward_args = []
    if args.epochs is not None:
        forward_args.extend(["--epochs", str(args.epochs)])
    if args.batch is not None:
        forward_args.extend(["--batch", str(args.batch)])
    if args.device is not None:
        forward_args.extend(["--device", str(args.device)])
    if args.fraction is not None:
        forward_args.extend(["--fraction", str(args.fraction)])
    if args.workers is not None:
        forward_args.extend(["--workers", str(args.workers)])
    if args.imgsz is not None:
        forward_args.extend(["--imgsz", str(args.imgsz)])
    if args.tags is not None:
        forward_args.extend(["--tags"] + args.tags)
    if args.amp is not None:
        forward_args.extend(["--amp", args.amp])
    if args.dataset is not None:
        forward_args.extend(["--dataset", args.dataset])

    # Arguments specific to DINO-based models
    dino_forward_args = []
    dino_epochs_val = args.dino_epochs if args.dino_epochs is not None else args.epochs
    if dino_epochs_val is not None:
        dino_forward_args.extend(["--epochs", str(dino_epochs_val)])
    if args.batch is not None:
        dino_forward_args.extend(["--batch", str(args.batch)])
    if args.device is not None:
        dino_forward_args.extend(["--device", str(args.device)])
    if args.fraction is not None:
        dino_forward_args.extend(["--fraction", str(args.fraction)])
    if args.workers is not None:
        dino_forward_args.extend(["--workers", str(args.workers)])
    if args.imgsz is not None:
        dino_forward_args.extend(["--imgsz", str(args.imgsz)])
    if args.tags is not None:
        dino_forward_args.extend(["--tags"] + args.tags)
    if args.amp is not None:
        dino_forward_args.extend(["--amp", args.amp])
    if args.dataset is not None:
        dino_forward_args.extend(["--dataset", args.dataset])

    project_root = os.path.dirname(os.path.abspath(__file__))

    # Distinctive baseline folders
    baseline_runs_dir = (
        args.runs_dir
        if args.runs_dir is not None
        else os.path.join(project_root, "runs", "baseline")
    )
    baseline_eval_dir = (
        args.eval_results_dir
        if args.eval_results_dir is not None
        else os.path.join(project_root, "evaluation_results", "baseline")
    )

    os.makedirs(baseline_runs_dir, exist_ok=True)
    os.makedirs(baseline_eval_dir, exist_ok=True)

    print("=" * 70)
    print("STARTING COMPREHENSIVE BASELINE BENCHMARK SUITE")
    print(f"Checkpoints Folder: {baseline_runs_dir}")
    print(f"COCO Metrics Folder: {baseline_eval_dir}")
    print("=" * 70)

    seeds = [args.seed] if args.seed is not None else [42, 100, 999]

    # 1. Train and evaluate YOLO12s, YOLO11s, YOLO26s, RT-DETR-L (Ultralytics backend)
    ultralytics_models = []
    if run_yolo11n:
        ultralytics_models.append("yolo11n.pt")
    if run_yolo11s:
        ultralytics_models.append("yolo11s.pt")
    if run_yolo26n:
        ultralytics_models.append("yolo26n.pt")
    if run_yolo26s:
        ultralytics_models.append("yolo26s.pt")
    if run_yolo12n:
        ultralytics_models.append("yolo12n.pt")
    if run_yolo12s:
        ultralytics_models.append("yolo12s.pt")
    if run_rtdetr_l:
        ultralytics_models.append("rtdetr-l.pt")

    if ultralytics_models:
        print(
            f"\n>>> Running standard settings baseline for selected Ultralytics models: {ultralytics_models}..."
        )
        for model in ultralytics_models:
            for seed in seeds:
                run_training_with_fallback(
                    [
                        "pixi",
                        "run",
                        "python",
                        "run_training.py",
                        "--model",
                        model,
                        "--seed",
                        str(seed),
                        "--backend",
                        "ultralytics",
                        "--runs-dir",
                        baseline_runs_dir,
                        "--eval-results-dir",
                        baseline_eval_dir,
                    ]
                    + forward_args
                )

    # 2. Train and evaluate LightlyTrain DINOv3 backbones (sat493m and lvd1689m)
    lightly_models = []
    if run_dinov3_l:
        lightly_models = [
            "facebook/dinov3-vitl16-pretrain-sat493m",
            "facebook/dinov3-vitl16-pretrain-lvd1689m",
        ]
    elif run_dinov3_sat:
        lightly_models = [
            "facebook/dinov3-vitl16-pretrain-sat493m",
        ]

    if lightly_models:
        print(
            f"\n>>> Running baseline for selected LightlyTrain custom DINOv3 backbones: {lightly_models}..."
        )
        for model in lightly_models:
            # Run 2A: RT-DETRv2 decoder head combination
            print(
                f"\n>>> Executing RT-DETRv2 head variant baseline runs for {model}..."
            )
            for seed in seeds:
                run_training_with_fallback(
                    [
                        "pixi",
                        "run",
                        "python",
                        "run_training.py",
                        "--model",
                        model,
                        "--seed",
                        str(seed),
                        "--backend",
                        "lightly",
                        "--decoder",
                        "rtdetrv2",
                        "--runs-dir",
                        baseline_runs_dir,
                        "--eval-results-dir",
                        baseline_eval_dir,
                    ]
                    + dino_forward_args
                )

            # Run 2B: D-FINE decoder head combination
            print(f"\n>>> Executing D-FINE head variant baseline runs for {model}...")
            for seed in seeds:
                run_training_with_fallback(
                    [
                        "pixi",
                        "run",
                        "python",
                        "run_training.py",
                        "--model",
                        model,
                        "--seed",
                        str(seed),
                        "--backend",
                        "lightly",
                        "--decoder",
                        "dfine",
                        "--runs-dir",
                        baseline_runs_dir,
                        "--eval-results-dir",
                        baseline_eval_dir,
                    ]
                    + dino_forward_args
                )

    # 3. Aggregate results and plot comparison charts
    print("\n>>> Generating aggregated charts and markdown summaries...")
    run_cmd(
        [
            "pixi",
            "run",
            "python",
            "plot_results.py",
            "--eval-dir",
            baseline_eval_dir,
            "--error-metric",
            "sem",
        ]
    )

    print("=" * 70)
    print("BASELINE BENCHMARK SUITE COMPLETE!")
    print(f"Checkpoints saved in:  {baseline_runs_dir}")
    print(f"Plots and summaries:   {baseline_eval_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
