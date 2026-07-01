"""
Results Aggregation and Plotting Script.

This script reads all metric files matching *_coco_metrics.json in the evaluation results folder,
groups them by model name, calculates mean and standard deviation across different seeds,
creates comparison plots, and saves a markdown summary table.
"""

import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List
from config import PipelineConfig


def load_metrics(eval_dir: str) -> List[Dict[str, Any]]:
    """Loads all COCO metrics JSON files from the evaluation results directory."""
    pattern = os.path.join(eval_dir, "**", "*_coco_metrics.json")
    files = glob.glob(pattern, recursive=True)

    results = []
    for filepath in files:
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
                results.append(data)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")

    return results


def group_and_aggregate(
    results: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Groups results by model variant and aggregates across seeds
    calculating mean and standard deviation.
    """
    grouped_data = {}

    for r in results:
        run_name = r["run_name"]

        # Parse model base name and seed from run_name
        # E.g. "yolo11s_seed_42" -> model_base="yolo11s", seed="42"
        # E.g. "yolo11s_seed_42_best_aug" -> model_base="yolo11s_best_aug", seed="42"
        if "_seed_" in run_name:
            parts = run_name.split("_seed_")
            model_prefix = parts[0]
            seed_and_suffix = parts[1].split("_", 1)
            if len(seed_and_suffix) > 1:
                model_base = f"{model_prefix}_{seed_and_suffix[1]}"
            else:
                model_base = model_prefix
        else:
            model_base = run_name

        if model_base not in grouped_data:
            grouped_data[model_base] = {metric: [] for metric in r["metrics"].keys()}

        for metric_name, val in r["metrics"].items():
            grouped_data[model_base][metric_name].append(val)

    # Calculate stats
    stats = {}
    for model_base, metrics in grouped_data.items():
        stats[model_base] = {}
        for metric_name, values in metrics.items():
            vals = np.array(values)
            count = len(vals)
            mean_val = float(np.mean(vals))
            # Use sample standard deviation (ddof=1) for scientific rigor
            std_val = float(np.std(vals, ddof=1)) if count > 1 else 0.0
            sem_val = std_val / np.sqrt(count) if count > 0 else 0.0

            stats[model_base][metric_name] = {
                "values": values,
                "mean": mean_val,
                "std": std_val,
                "sem": sem_val,
                "count": count,
            }

    return stats


def plot_ap_comparison(
    stats: Dict[str, Dict[str, Dict[str, float]]],
    save_dir: str,
    error_metric: str = "sem",
) -> None:
    """Generates a bar plot of AP, AP50, and AP75 with specified error bars (Standard Deviation or SEM)."""
    models = list(stats.keys())
    if not models:
        print("No models found to plot.")
        return

    err_label = "SEM" if error_metric == "sem" else "Std"

    ap_means = [stats[m]["AP"]["mean"] for m in models]
    ap_errs = [stats[m]["AP"][error_metric] for m in models]

    ap50_means = [stats[m]["AP50"]["mean"] for m in models]
    ap50_errs = [stats[m]["AP50"][error_metric] for m in models]

    ap75_means = [stats[m]["AP75"]["mean"] for m in models]
    ap75_errs = [stats[m]["AP75"][error_metric] for m in models]

    x = np.arange(len(models))
    width = 0.25

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(11, 6))

    rects1 = ax.bar(
        x - width,
        ap_means,
        width,
        yerr=ap_errs,
        label="mAP@0.50:0.95",
        capsize=4,
        color="#4A90E2",
        edgecolor="black",
        alpha=0.9,
    )
    rects2 = ax.bar(
        x,
        ap50_means,
        width,
        yerr=ap50_errs,
        label="mAP@0.50",
        capsize=4,
        color="#50E3C2",
        edgecolor="black",
        alpha=0.9,
    )
    rects3 = ax.bar(
        x + width,
        ap75_means,
        width,
        yerr=ap75_errs,
        label="mAP@0.75",
        capsize=4,
        color="#F5A623",
        edgecolor="black",
        alpha=0.9,
    )

    ax.set_ylabel("Score")
    ax.set_title(f"Strict COCO AP Metrics Comparison (Mean ± {err_label} across Seeds)")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()

    # Add values on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.4f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="semibold",
            )

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    fig.tight_layout()
    plot_path = os.path.join(save_dir, "ap_metrics_comparison.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved AP comparison plot to {plot_path}")


def plot_scale_comparison(
    stats: Dict[str, Dict[str, Dict[str, float]]],
    save_dir: str,
    error_metric: str = "sem",
) -> None:
    """Generates a bar plot comparing AP_small, AP_medium, and AP_large with specified error bars."""
    models = list(stats.keys())
    if not models:
        return

    err_label = "SEM" if error_metric == "sem" else "Std"

    ap_s_means = [stats[m]["AP_small"]["mean"] for m in models]
    ap_s_errs = [stats[m]["AP_small"][error_metric] for m in models]

    ap_m_means = [stats[m]["AP_medium"]["mean"] for m in models]
    ap_m_errs = [stats[m]["AP_medium"][error_metric] for m in models]

    ap_l_means = [stats[m]["AP_large"]["mean"] for m in models]
    ap_l_errs = [stats[m]["AP_large"][error_metric] for m in models]

    x = np.arange(len(models))
    width = 0.25

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(11, 6))

    rects1 = ax.bar(
        x - width,
        ap_s_means,
        width,
        yerr=ap_s_errs,
        label="AP (Small)",
        capsize=4,
        color="#E28490",
        edgecolor="black",
        alpha=0.9,
    )
    rects2 = ax.bar(
        x,
        ap_m_means,
        width,
        yerr=ap_m_errs,
        label="AP (Medium)",
        capsize=4,
        color="#9B51E0",
        edgecolor="black",
        alpha=0.9,
    )
    rects3 = ax.bar(
        x + width,
        ap_l_means,
        width,
        yerr=ap_l_errs,
        label="AP (Large)",
        capsize=4,
        color="#27AE60",
        edgecolor="black",
        alpha=0.9,
    )

    ax.set_ylabel("Score")
    ax.set_title(f"Strict COCO AP by Object Scale (Mean ± {err_label} across Seeds)")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.4f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="semibold",
            )

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    fig.tight_layout()
    plot_path = os.path.join(save_dir, "ap_scale_comparison.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved AP scale comparison plot to {plot_path}")


def plot_ar_comparison(
    stats: Dict[str, Dict[str, Dict[str, float]]],
    save_dir: str,
    error_metric: str = "sem",
) -> None:
    """Generates a bar plot comparing AR_max1, AR_max10, and AR_max100 with specified error bars."""
    models = list(stats.keys())
    if not models:
        return

    err_label = "SEM" if error_metric == "sem" else "Std"

    ar1_means = [stats[m]["AR_max1"]["mean"] for m in models]
    ar1_errs = [stats[m]["AR_max1"][error_metric] for m in models]

    ar10_means = [stats[m]["AR_max10"]["mean"] for m in models]
    ar10_errs = [stats[m]["AR_max10"][error_metric] for m in models]

    ar100_means = [stats[m]["AR_max100"]["mean"] for m in models]
    ar100_errs = [stats[m]["AR_max100"][error_metric] for m in models]

    x = np.arange(len(models))
    width = 0.25

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(11, 6))

    rects1 = ax.bar(
        x - width,
        ar1_means,
        width,
        yerr=ar1_errs,
        label="AR @ 1 max det",
        capsize=4,
        color="#F2994A",
        edgecolor="black",
        alpha=0.9,
    )
    rects2 = ax.bar(
        x,
        ar10_means,
        width,
        yerr=ar10_errs,
        label="AR @ 10 max det",
        capsize=4,
        color="#EB5757",
        edgecolor="black",
        alpha=0.9,
    )
    rects3 = ax.bar(
        x + width,
        ar100_means,
        width,
        yerr=ar100_errs,
        label="AR @ 100 max det",
        capsize=4,
        color="#2F80ED",
        edgecolor="black",
        alpha=0.9,
    )

    ax.set_ylabel("Score")
    ax.set_title(
        f"Strict COCO Average Recall (AR) Comparison (Mean ± {err_label} across Seeds)"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.4f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="semibold",
            )

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    fig.tight_layout()
    plot_path = os.path.join(save_dir, "ar_metrics_comparison.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved AR comparison plot to {plot_path}")


def plot_individual_seeds(
    stats: Dict[str, Dict[str, Dict[str, float]]], save_dir: str
) -> None:
    """Generates a scatter/swarm plot showing individual seed scores for each model."""
    models = list(stats.keys())
    if not models:
        return

    x_data = []
    y_data = []

    for m in models:
        vals = stats[m]["AP"]["values"]
        for v in vals:  # type: ignore
            x_data.append(m)
            y_data.append(v)

    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")

    # Plot mean with a bar
    for i, m in enumerate(models):
        mean_val = stats[m]["AP"]["mean"]
        plt.bar(m, mean_val, color="gray", alpha=0.2, edgecolor="black", width=0.5)

    # Plot individual seeds
    sns.stripplot(
        x=x_data,
        y=y_data,
        hue=x_data,
        jitter=0.15,
        size=10,
        linewidth=1,
        palette="Set2",
        legend=False,
    )

    plt.title("mAP@0.50:0.95 Scores by Seed")
    plt.ylabel("mAP@0.50:0.95")
    plt.xlabel("Model")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plot_path = os.path.join(save_dir, "ap_seed_distribution.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved seed distribution plot to {plot_path}")


def generate_markdown_summary(
    stats: Dict[str, Dict[str, Dict[str, float]]], save_dir: str
) -> None:
    """Generates a markdown table summarizing the evaluation results and saves it to a file."""
    models = sorted(list(stats.keys()))
    if not models:
        return

    lines = [
        "# Strict Test Set Evaluation Summary (COCO Metrics)",
        "",
        "This table aggregates the strict pycocotools AP/AR metrics computed on the test split.",
        "Scores are reported as **Mean ± Standard Error of the Mean (SEM)** across the configured seeds.",
        "",
        "| Model Variant | Runs | mAP@0.50:0.95 | mAP@0.50 | mAP@0.75 | AP (Small) | AP (Medium) | AP (Large) | AR@100 |",
        "| :--- | :---: | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for m in models:
        count = stats[m]["AP"]["count"]
        ap = f"{stats[m]['AP']['mean']:.5f} ± {stats[m]['AP']['sem']:.5f}"
        ap50 = f"{stats[m]['AP50']['mean']:.5f} ± {stats[m]['AP50']['sem']:.5f}"
        ap75 = f"{stats[m]['AP75']['mean']:.5f} ± {stats[m]['AP75']['sem']:.5f}"
        ap_s = f"{stats[m]['AP_small']['mean']:.5f} ± {stats[m]['AP_small']['sem']:.5f}"
        ap_m = (
            f"{stats[m]['AP_medium']['mean']:.5f} ± {stats[m]['AP_medium']['sem']:.5f}"
        )
        ap_l = f"{stats[m]['AP_large']['mean']:.5f} ± {stats[m]['AP_large']['sem']:.5f}"
        ar100 = (
            f"{stats[m]['AR_max100']['mean']:.5f} ± {stats[m]['AR_max100']['sem']:.5f}"
        )

        lines.append(
            f"| {m} | {count} | {ap} | {ap50} | {ap75} | {ap_s} | {ap_m} | {ap_l} | {ar100} |"
        )

    summary_path = os.path.join(save_dir, "evaluation_summary.md")
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))

    print(f"\nSaved markdown evaluation summary to {summary_path}")
    print("\n" + "\n".join(lines) + "\n")

    # Save combined results to a JSON file
    summary_json_path = os.path.join(save_dir, "evaluation_summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(stats, f, indent=4)
    print(f"Saved combined JSON evaluation summary to {summary_json_path}")


def compute_dataset_stats(dataset_yaml_path: str, save_dir: str) -> None:
    """Computes dataset statistics and creates class distribution plots."""
    import yaml
    import glob

    if not os.path.exists(dataset_yaml_path):
        print(f"Dataset YAML {dataset_yaml_path} not found. Skipping dataset stats.")
        return

    try:
        with open(dataset_yaml_path, "r") as f:
            data = yaml.safe_load(f)

        base_path = data.get("path", "")
        if not os.path.isabs(base_path):
            yaml_dir = os.path.dirname(os.path.abspath(dataset_yaml_path))
            base_path = os.path.abspath(os.path.join(yaml_dir, base_path))

        names_dict = data.get("names", {})
        splits = ["train", "val", "test"]
        stats = {}

        for split in splits:
            split_dir = data.get(split, "")
            if not split_dir:
                continue

            img_dir = (
                os.path.join(base_path, split_dir)
                if not os.path.isabs(split_dir)
                else split_dir
            )
            lbl_dir = img_dir.replace("images", "labels")

            if not os.path.exists(img_dir):
                continue

            img_files = sorted(glob.glob(os.path.join(img_dir, "*.*")))
            img_files = [
                p for p in img_files if p.lower().endswith((".png", ".jpg", ".jpeg"))
            ]

            total_images = len(img_files)
            total_annotations = 0
            class_counts = {int(k): 0 for k in names_dict.keys()}

            for img_path in img_files:
                img_name = os.path.splitext(os.path.basename(img_path))[0]
                lbl_path = os.path.join(lbl_dir, img_name + ".txt")
                if os.path.exists(lbl_path):
                    with open(lbl_path, "r") as lf:
                        lines = lf.readlines()
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        class_id = int(parts[0])
                        if class_id in class_counts:
                            class_counts[class_id] += 1
                        total_annotations += 1

            stats[split] = {
                "images": total_images,
                "annotations": total_annotations,
                "classes": class_counts,
            }

        # Write markdown report
        md_lines = [
            "# Dataset Statistics Report",
            "",
            "This report summarizes the image and bounding box distribution across splits.",
            "",
            "## Summary Table",
            "",
            "| Split | Total Images | Total Bounding Boxes | Avg Bounding Boxes per Image |",
            "| :--- | :---: | :---: | :---: |",
        ]
        for split in splits:
            if split in stats:
                img_count = stats[split]["images"]
                box_count = stats[split]["annotations"]
                avg_boxes = box_count / img_count if img_count > 0 else 0.0
                md_lines.append(
                    f"| {split.capitalize()} | {img_count} | {box_count} | {avg_boxes:.2f} |"
                )

        md_lines.append("")
        md_lines.append("## Class Distribution Table")
        md_lines.append("")

        header_row = "| Class ID | Class Name |"
        separator_row = "| :--- | :--- |"
        for split in splits:
            if split in stats:
                header_row += f" {split.capitalize()} Instances |"
                separator_row += " :---: |"
        md_lines.append(header_row)
        md_lines.append(separator_row)

        for cid, cname in names_dict.items():
            row = f"| {cid} | {cname} |"
            for split in splits:
                if split in stats:
                    count = stats[split]["classes"].get(int(cid), 0)
                    row += f" {count} |"
            md_lines.append(row)

        report_path = os.path.join(save_dir, "dataset_statistics.md")
        with open(report_path, "w") as f:
            f.write("\n".join(md_lines))
        print(f"Saved dataset statistics to {report_path}")

        # Plot class distribution
        if len(names_dict) > 0 and stats:
            plt.figure(figsize=(10, 6))
            sns.set_theme(style="whitegrid")

            categories = [names_dict[k] for k in sorted(names_dict.keys())]
            x_indices = np.arange(len(categories))
            width = 0.2

            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ["#4A90E2", "#50E3C2", "#F5A623"]

            for idx, split in enumerate(splits):
                if split in stats:
                    counts = [
                        stats[split]["classes"].get(int(k), 0)
                        for k in sorted(names_dict.keys())
                    ]
                    offset = (idx - 1) * width if len(stats) > 1 else 0
                    ax.bar(
                        x_indices + offset,
                        counts,
                        width,
                        label=split.capitalize(),
                        color=colors[idx % len(colors)],
                        edgecolor="black",
                        alpha=0.9,
                    )

            ax.set_ylabel("Instance Count")
            ax.set_title("Class Instance Distribution across Dataset Splits")
            ax.set_xticks(x_indices)
            ax.set_xticklabels(categories)
            ax.legend()

            fig.tight_layout()
            plot_path = os.path.join(save_dir, "dataset_class_distribution.png")
            plt.savefig(plot_path, dpi=300)
            plt.close()
            print(f"Saved dataset distribution plot to {plot_path}")

    except Exception as e:
        print(f"Error computing dataset statistics: {e}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Results Aggregation and Plotting Script."
    )
    parser.add_argument(
        "--error-metric",
        type=str,
        choices=["std", "sem"],
        default="sem",
        help="Error bar metric: 'std' (Standard Deviation) or 'sem' (Standard Error of the Mean, default).",
    )
    parser.add_argument(
        "--eval-dir",
        type=str,
        default=None,
        help="Override the evaluation results directory containing the metric JSON files.",
    )
    args = parser.parse_args()

    cfg = PipelineConfig()
    eval_dir = args.eval_dir if args.eval_dir is not None else cfg.eval_results_dir

    if not os.path.exists(eval_dir):
        print(f"Evaluation directory {eval_dir} does not exist. No results to plot.")
        return

    results = load_metrics(eval_dir)
    if not results:
        print(f"No metric JSON files (*_coco_metrics.json) found in {eval_dir}.")
        return

    print(f"Loaded {len(results)} metrics files from {eval_dir}.")

    stats = group_and_aggregate(results)

    # Generate outputs
    plot_ap_comparison(stats, eval_dir, args.error_metric)
    plot_scale_comparison(stats, eval_dir, args.error_metric)
    plot_ar_comparison(stats, eval_dir, args.error_metric)
    plot_individual_seeds(stats, eval_dir)
    generate_markdown_summary(stats, eval_dir)

    # Generate dataset stats
    compute_dataset_stats(cfg.dataset_path, eval_dir)


if __name__ == "__main__":
    main()
