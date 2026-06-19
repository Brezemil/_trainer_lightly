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
        "--workers", type=int, default=None, help="Override dataloader workers."
    )
    parser.add_argument("--imgsz", type=int, default=None, help="Override image size.")
    return parser.parse_args()


def main():
    args = parse_args()

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

    project_root = os.path.dirname(os.path.abspath(__file__))

    # Distinctive baseline folders
    baseline_runs_dir = os.path.join(project_root, "runs", "baseline")
    baseline_eval_dir = os.path.join(project_root, "evaluation_results", "baseline")

    os.makedirs(baseline_runs_dir, exist_ok=True)
    os.makedirs(baseline_eval_dir, exist_ok=True)

    print("=" * 70)
    print("STARTING COMPREHENSIVE BASELINE BENCHMARK SUITE")
    print(f"Checkpoints Folder: {baseline_runs_dir}")
    print(f"COCO Metrics Folder: {baseline_eval_dir}")
    print("=" * 70)

    seeds = [42, 100, 999]

    # 1. Train and evaluate YOLO12s, YOLO11s, YOLO26s, RT-DETR-L (Ultralytics backend)
    ultralytics_models = ["yolo11s.pt", "yolo26s.pt", "yolo12s.pt", "rtdetr-l.pt"]
    print(
        "\n>>> Running standard settings baseline for YOLO11s, YOLO26s, YOLOv12s, and RT-DETR-L..."
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
    lightly_models = [
        "facebook/dinov3-vitl16-pretrain-sat493m",
        "facebook/dinov3-vitl16-pretrain-lvd1689m",
    ]
    print("\n>>> Running baseline for LightlyTrain custom DINOv3 backbones...")
    for model in lightly_models:
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
                    "--runs-dir",
                    baseline_runs_dir,
                    "--eval-results-dir",
                    baseline_eval_dir,
                ]
                + forward_args
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
