"""
Teacher-Student Feature Alignment Analysis Script.

Computes Spatial Cosine Similarity and Spatial MSE feature alignment between
DINOv3 Teacher foundation backbones and YOLO/Lightly Student models.
Generates publication-quality visualizations (300 DPI) and uploads metrics & figures to Weights & Biases.
"""

from PIL import Image  # noqa: F401
import argparse
import glob
import json
import os
import gc
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from torchvision import transforms
import wandb

from config import PipelineConfig
from eval_utils import safe_load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Teacher-Student Feature Alignment (Cosine Similarity & Spatial MSE)."
    )
    parser.add_argument(
        "--teacher",
        type=str,
        default="dinov3/vitl16-sat493m",
        help="Teacher foundation model identifier (default: dinov3/vitl16-sat493m).",
    )
    parser.add_argument(
        "--student",
        type=str,
        default="yolo12s",
        help="Student architecture or path to distilled checkpoint (default: yolo12s).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to dataset YAML file or image directory (default: PipelineConfig dataset_path).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate alignment on (default: val).",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of images to process for alignment calculation (default: 100).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=512,
        help="Image resolution in pixels for feature extraction (default: 512).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for feature extraction (default: 8).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="CUDA device index or 'cpu' (default: '0').",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for reports and figures (default: evaluation_results/alignment).",
    )
    parser.add_argument(
        "--upload_wandb",
        action="store_true",
        help="Upload summary metrics and alignment plots to Weights & Biases.",
    )
    parser.add_argument(
        "--wandb-offline",
        action="store_true",
        help="Run W&B in offline mode.",
    )
    return parser.parse_args()


def load_teacher_model(teacher_id: str, device: torch.device) -> torch.nn.Module:
    """Loads DINOv3 teacher model using lightly_train internal builder."""
    print(f"Loading Teacher model: {teacher_id}...")
    from typing import Any
    import lightly_train._models.dinov3.dinov3_vit as vit_module

    vit_mod_any: Any = vit_module
    func_name = teacher_id.replace("dinov3/", "dinov3_").replace("-", "_")
    if hasattr(vit_mod_any, func_name):
        builder_fn = getattr(vit_mod_any, func_name)
        teacher_model = builder_fn()
    else:
        teacher_model = vit_mod_any.dinov3_vitl16_sat493m()
    return teacher_model.to(device).eval()


def load_student_model(
    student_id_or_path: str, device: torch.device
) -> torch.nn.Module:
    """Loads student model from path checkpoint or Ultralytics model string."""
    print(f"Loading Student model: {student_id_or_path}...")
    if os.path.exists(student_id_or_path):
        student_obj = safe_load_model(student_id_or_path, device=device)
        backbone = getattr(student_obj, "backbone", None)
        if backbone is not None and hasattr(backbone, "to"):
            return backbone.to(device).eval()  # type: ignore
        sub_model = getattr(student_obj, "model", None)
        if sub_model is not None and hasattr(sub_model, "to"):
            return sub_model.to(device).eval()  # type: ignore
        return student_obj.to(device).eval()  # type: ignore
    else:
        from ultralytics import YOLO

        model_name = student_id_or_path
        if not model_name.endswith(".pt") and not model_name.endswith(".yaml"):
            model_name = f"{model_name}.pt"
        yolo_obj = YOLO(model_name)
        return yolo_obj.model.to(device).eval()  # type: ignore


def get_image_paths(dataset_arg: str | None, split: str, num_samples: int) -> list[str]:
    """Resolves dataset image file paths."""
    cfg = PipelineConfig()
    ds_path = dataset_arg if dataset_arg is not None else cfg.dataset_path

    if os.path.isdir(ds_path):
        img_dir = ds_path
    elif os.path.isfile(ds_path) and ds_path.endswith((".yaml", ".yml")):
        import yaml

        with open(ds_path, "r") as f:
            data_dict = yaml.safe_load(f)
        base = data_dict.get("path", "")
        if not os.path.isabs(base):
            yaml_dir = os.path.dirname(os.path.abspath(ds_path))
            base = os.path.abspath(os.path.join(yaml_dir, base))
        split_rel = data_dict.get(split, "images/val")
        img_dir = (
            os.path.join(base, split_rel) if not os.path.isabs(split_rel) else split_rel
        )
    else:
        raise FileNotFoundError(f"Could not resolve dataset path: {ds_path}")

    all_images = sorted(glob.glob(os.path.join(img_dir, "*.*")))
    valid_images = [
        p
        for p in all_images
        if p.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
        )
    ]

    if not valid_images:
        raise FileNotFoundError(f"No valid image files found in {img_dir}")

    # Deterministic sampling
    import random

    rng = random.Random(42)
    selected = rng.sample(valid_images, min(num_samples, len(valid_images)))
    print(
        f"Selected {len(selected)} images from {img_dir} for feature alignment evaluation."
    )
    return selected


def main() -> None:
    args = parse_args()
    cfg = PipelineConfig()

    if args.wandb_offline:
        os.environ["WANDB_MODE"] = "offline"

    out_dir = (
        os.path.abspath(args.out_dir)
        if args.out_dir
        else os.path.abspath(os.path.join(cfg.eval_results_dir, "alignment"))
    )
    os.makedirs(out_dir, exist_ok=True)

    device_str = (
        f"cuda:{args.device}"
        if args.device.isdigit() and torch.cuda.is_available()
        else ("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    )
    device = torch.device(device_str)

    print("\n" + "=" * 80)
    print(" TEACHER-STUDENT FEATURE ALIGNMENT ANALYSIS")
    print("=" * 80)
    print(f" Teacher Model:          {args.teacher}")
    print(f" Student Architecture:   {args.student}")
    print(f" Resolution (imgsz):     {args.imgsz}x{args.imgsz}")
    print(f" Dataset Split:          {args.split}")
    print(f" Evaluation Device:      {device}")
    print(f" Output Directory:       {out_dir}")
    print("=" * 80 + "\n")

    teacher_model = load_teacher_model(args.teacher, device)
    student_model = load_student_model(args.student, device)

    image_paths = get_image_paths(args.dataset, args.split, args.num_samples)

    # Standard ImageNet normalization for feature extraction
    transform_pipeline = transforms.Compose(
        [
            transforms.Resize((args.imgsz, args.imgsz)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    all_cosine_sims: list[float] = []
    all_spatial_mses: list[float] = []
    sample_vis_data: list[dict[str, np.ndarray]] = []

    # Process in mini-batches
    num_batches = (len(image_paths) + args.batch_size - 1) // args.batch_size

    print(
        f"Extracting features across {len(image_paths)} images ({num_batches} batches)..."
    )

    for b_idx in range(num_batches):
        batch_paths = image_paths[
            b_idx * args.batch_size : (b_idx + 1) * args.batch_size
        ]
        tensors = []
        raw_images = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            raw_images.append(np.array(img.resize((args.imgsz, args.imgsz))))
            tensors.append(transform_pipeline(img))

        batch_tensor = torch.stack(tensors).to(device)

        with torch.no_grad():
            # 1. Teacher Feature Extraction
            t_out = teacher_model(batch_tensor)
            if isinstance(t_out, dict) and "x_norm_patchtokens" in t_out:
                t_patches = t_out["x_norm_patchtokens"]  # [B, N, C_t]
            elif isinstance(t_out, torch.Tensor):
                t_patches = t_out
            else:
                t_patches = t_out[0]

            if t_patches.ndim == 3:
                grid_dim = int(np.sqrt(t_patches.shape[1]))
                t_spatial = t_patches.permute(0, 2, 1).reshape(
                    len(batch_paths), -1, grid_dim, grid_dim
                )
            else:
                t_spatial = t_patches
                grid_dim = t_spatial.shape[2]

            # 2. Student Feature Extraction via Hook
            student_features = []

            def hook_fn(module, inp, outp):
                student_features.append(outp)

            # Register hook on backbone feature map
            from typing import Any

            children_list = list(student_model.children())
            if children_list and hasattr(children_list[0], "__getitem__"):
                sub_child: Any = children_list[0]
                hook_target: Any = sub_child[-2]
            else:
                hook_target = student_model
            handle = hook_target.register_forward_hook(hook_fn)
            _ = student_model(batch_tensor)
            handle.remove()

            s_feat = student_features[0] if student_features else batch_tensor
            if isinstance(s_feat, (list, tuple)):
                s_feat = s_feat[0]

            s_spatial = F.interpolate(
                s_feat, size=(grid_dim, grid_dim), mode="bilinear", align_corners=False
            )

            # 3. Compute Spatial Correlation Matrix & Alignment Maps
            b_size = len(batch_paths)
            t_flat = F.normalize(t_spatial.flatten(2), p=2, dim=1)  # [B, C_t, N]
            s_flat = F.normalize(s_spatial.flatten(2), p=2, dim=1)  # [B, C_s, N]

            # Spatial Gram Correlation Matrices: R_t, R_s in [B, N, N]
            R_t = torch.bmm(t_flat.transpose(1, 2), t_flat)
            R_s = torch.bmm(s_flat.transpose(1, 2), s_flat)

            R_t_norm = F.normalize(R_t, p=2, dim=2)
            R_s_norm = F.normalize(R_s, p=2, dim=2)

            cosine_sim_maps = (
                (R_t_norm * R_s_norm).sum(dim=-1).reshape(b_size, grid_dim, grid_dim)
            )
            spatial_mse_maps = (
                ((R_t - R_s) ** 2).mean(dim=-1).reshape(b_size, grid_dim, grid_dim)
            )

            t_intensity_maps = t_spatial.norm(dim=1).cpu().numpy()
            s_intensity_maps = s_spatial.norm(dim=1).cpu().numpy()
            cosine_maps_np = cosine_sim_maps.cpu().numpy()
            mse_maps_np = spatial_mse_maps.cpu().numpy()

            for i in range(b_size):
                c_mean = float(cosine_maps_np[i].mean())
                m_mean = float(mse_maps_np[i].mean())
                all_cosine_sims.append(c_mean)
                all_spatial_mses.append(m_mean)

                if len(sample_vis_data) < 3:
                    sample_vis_data.append(
                        {
                            "raw_img": raw_images[i],
                            "t_map": t_intensity_maps[i],
                            "s_map": s_intensity_maps[i],
                            "sim_map": cosine_maps_np[i],
                        }
                    )

    # 4. Compute Statistical Summary
    cos_arr = np.array(all_cosine_sims)
    mse_arr = np.array(all_spatial_mses)

    cos_mean = float(np.mean(cos_arr))
    cos_std = float(np.std(cos_arr))
    cos_median = float(np.median(cos_arr))
    cos_min = float(np.min(cos_arr))
    cos_max = float(np.max(cos_arr))

    mse_mean = float(np.mean(mse_arr))
    mse_std = float(np.std(mse_arr))
    mse_median = float(np.median(mse_arr))

    print("\n" + "=" * 80)
    print(" ALIGNMENT STATISTICAL RESULTS SUMMARY")
    print("-" * 80)
    print(f" Mean Cosine Similarity:   {cos_mean:.4f} ± {cos_std:.4f}")
    print(f" Median Cosine Similarity: {cos_median:.4f}")
    print(f" Cosine Sim Range:         [{cos_min:.4f}, {cos_max:.4f}]")
    print(f" Mean Spatial MSE:         {mse_mean:.6f} ± {mse_std:.6f}")
    print(f" Median Spatial MSE:       {mse_median:.6f}")
    print("=" * 80 + "\n")

    # Save JSON report
    report_dict = {
        "teacher": args.teacher,
        "student": args.student,
        "num_samples": len(image_paths),
        "imgsz": args.imgsz,
        "cosine_similarity": {
            "mean": cos_mean,
            "std": cos_std,
            "median": cos_median,
            "min": cos_min,
            "max": cos_max,
        },
        "spatial_mse": {
            "mean": mse_mean,
            "std": mse_std,
            "median": mse_median,
        },
    }
    report_json_path = os.path.join(out_dir, "alignment_report.json")
    with open(report_json_path, "w") as f:
        json.dump(report_dict, f, indent=4)
    print(f"Saved alignment JSON report to: {report_json_path}")

    # 5. Generate Multi-Panel Publication Figure (300 DPI)
    plt.style.use(
        "seaborn-v0_8-whitegrid"
        if "seaborn-v0_8-whitegrid" in plt.style.available
        else "default"
    )
    fig = plt.figure(figsize=(18, 12), dpi=300)
    gs = fig.add_gridspec(3, 4, height_ratios=[1.2, 1.0, 1.0], hspace=0.35, wspace=0.25)

    # Row 1: Visual Snapshots (Image, Teacher Map, Student Map, Cosine Sim Overlay)
    for sample_idx, sdata in enumerate(sample_vis_data[:3]):
        # Original RGB
        ax_rgb = fig.add_subplot(gs[sample_idx, 0])
        ax_rgb.imshow(sdata["raw_img"])
        ax_rgb.set_title(
            f"Sample {sample_idx + 1}: Input RGB", fontsize=10, fontweight="bold"
        )
        ax_rgb.axis("off")

        # Teacher Map
        ax_t = fig.add_subplot(gs[sample_idx, 1])
        im_t = ax_t.imshow(sdata["t_map"], cmap="viridis")
        ax_t.set_title(
            f"Sample {sample_idx + 1}: DINOv3 Teacher Feats",
            fontsize=10,
            fontweight="bold",
        )
        ax_t.axis("off")
        plt.colorbar(im_t, ax=ax_t, fraction=0.046, pad=0.04)

        # Student Map
        ax_s = fig.add_subplot(gs[sample_idx, 2])
        im_s = ax_s.imshow(sdata["s_map"], cmap="magma")
        ax_s.set_title(
            f"Sample {sample_idx + 1}: YOLO Student Feats",
            fontsize=10,
            fontweight="bold",
        )
        ax_s.axis("off")
        plt.colorbar(im_s, ax=ax_s, fraction=0.046, pad=0.04)

        # Cosine Sim Heatmap
        ax_sim = fig.add_subplot(gs[sample_idx, 3])
        im_sim = ax_sim.imshow(sdata["sim_map"], cmap="plasma", vmin=0.0, vmax=1.0)
        ax_sim.set_title(
            f"Sample {sample_idx + 1}: Cosine Sim Map (μ={sdata['sim_map'].mean():.3f})",
            fontsize=10,
            fontweight="bold",
        )
        ax_sim.axis("off")
        plt.colorbar(im_sim, ax=ax_sim, fraction=0.046, pad=0.04)

    # Row 3 (Bottom): Distributions & Dashboard Summary Card
    ax_dist1 = fig.add_subplot(gs[2, 0:2])
    sns.histplot(
        cos_arr,
        kde=True,
        ax=ax_dist1,
        color="#1f77b4",
        bins=25,
        edgecolor="white",
        alpha=0.6,
    )
    ax_dist1.axvline(
        cos_mean,
        color="#d62728",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {cos_mean:.4f}",
    )
    ax_dist1.axvline(
        cos_median,
        color="#2ca02c",
        linestyle=":",
        linewidth=2,
        label=f"Median: {cos_median:.4f}",
    )
    ax_dist1.set_title(
        "Teacher-Student Cosine Similarity Distribution", fontsize=12, fontweight="bold"
    )
    ax_dist1.set_xlabel("Cosine Similarity", fontsize=10)
    ax_dist1.set_ylabel("Sample Density", fontsize=10)
    ax_dist1.legend(frameon=True, facecolor="white", loc="upper left")

    ax_dist2 = fig.add_subplot(gs[2, 2])
    sns.histplot(
        mse_arr,
        kde=True,
        ax=ax_dist2,
        color="#ff7f0e",
        bins=25,
        edgecolor="white",
        alpha=0.6,
    )
    ax_dist2.axvline(
        mse_mean,
        color="#d62728",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mse_mean:.5f}",
    )
    ax_dist2.set_title("Spatial MSE Distribution", fontsize=12, fontweight="bold")
    ax_dist2.set_xlabel("Spatial MSE Loss", fontsize=10)
    ax_dist2.set_ylabel("Sample Density", fontsize=10)
    ax_dist2.legend(frameon=True, facecolor="white", loc="upper right")

    # Metric Dashboard Card
    ax_card = fig.add_subplot(gs[2, 3])
    ax_card.axis("off")

    alignment_tier = (
        "High Alignment"
        if cos_mean >= 0.5
        else ("Moderate Alignment" if cos_mean >= 0.2 else "Low / Initial Alignment")
    )
    card_text = (
        f"  FEATURE ALIGNMENT REPORT\n"
        f"  {'=' * 30}\n"
        f"  Teacher:   {args.teacher.split('/')[-1]}\n"
        f"  Student:   {args.student}\n"
        f"  Samples:   {len(image_paths)} images\n"
        f"  Resolution: {args.imgsz}x{args.imgsz}\n"
        f"  ------------------------------\n"
        f"  Cosine Sim:  {cos_mean:.4f} ± {cos_std:.4f}\n"
        f"  Median Sim:  {cos_median:.4f}\n"
        f"  Spatial MSE: {mse_mean:.6f}\n"
        f"  ------------------------------\n"
        f"  Status: {alignment_tier}"
    )

    ax_card.text(
        0.05,
        0.5,
        card_text,
        transform=ax_card.transAxes,
        fontsize=10,
        family="monospace",
        verticalalignment="center",
        bbox=dict(
            boxstyle="round,pad=0.8",
            facecolor="#f8f9fa",
            edgecolor="#ced4da",
            linewidth=2,
        ),
    )

    fig.suptitle(
        f"DINOv3 Teacher-Student Spatial Feature Alignment Analysis\nTeacher: {args.teacher} | Student: {args.student}",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    fig_path = os.path.join(out_dir, "teacher_student_alignment.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Publication figure saved to: {fig_path}")

    # 6. Upload Results & Figure to Weights & Biases (If requested)
    if args.upload_wandb:
        print("\nUploading metrics and alignment plot to Weights & Biases...")
        run_name = f"alignment_{args.teacher.split('/')[-1]}_to_{args.student.replace('.pt', '')}"
        wandb.init(
            project=cfg.project,
            entity=cfg.entity,
            name=run_name,
            config={
                "pipeline": "feature_alignment_analysis",
                "teacher": args.teacher,
                "student": args.student,
                "num_samples": len(image_paths),
                "imgsz": args.imgsz,
            },
            reinit=True,
        )

        wandb.log(
            {
                "alignment/cosine_similarity_mean": cos_mean,
                "alignment/cosine_similarity_std": cos_std,
                "alignment/cosine_similarity_median": cos_median,
                "alignment/cosine_similarity_min": cos_min,
                "alignment/cosine_similarity_max": cos_max,
                "alignment/spatial_mse_mean": mse_mean,
                "alignment/spatial_mse_std": mse_std,
                "alignment/spatial_mse_median": mse_median,
                "alignment/visualization_panel": wandb.Image(fig_path),
            }
        )
        wandb.finish()
        print("Successfully uploaded alignment metrics and panel figure to W&B!")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
