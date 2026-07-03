import pyarrow  # noqa: F401
import os
import json
import glob
import shutil
from typing import Dict, Any, List
from PIL import Image
import yaml
import torch
import gc


def parse_dataset_yaml(dataset_yaml_path: str) -> Dict[str, Any]:
    """Parses dataset.yaml to extract path details and class names."""
    with open(dataset_yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return data


def generate_coco_gt(dataset_yaml_path: str, split: str, save_path: str) -> str:
    """
    Converts a YOLO format dataset split annotations to a COCO ground-truth JSON file.
    If the file already exists, it skips generation to save time.
    """
    if os.path.exists(save_path):
        print(
            f"Ground truth COCO JSON already exists at {save_path}. Skipping generation."
        )
        return save_path

    print(f"Generating COCO ground truth for split: {split}...")
    db_info = parse_dataset_yaml(dataset_yaml_path)

    # Resolve absolute paths
    base_path = db_info.get("path", "")
    split_img_dir = db_info.get(split, "")

    # Handle absolute vs relative paths in yaml
    if not os.path.isabs(base_path):
        yaml_dir = os.path.dirname(os.path.abspath(dataset_yaml_path))
        base_path = os.path.abspath(os.path.join(yaml_dir, base_path))

    images_dir = (
        os.path.join(base_path, split_img_dir)
        if not os.path.isabs(split_img_dir)
        else split_img_dir
    )
    labels_dir = images_dir.replace("images", "labels")

    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    # Gather images
    image_paths = sorted(glob.glob(os.path.join(images_dir, "*.*")))
    image_paths = [
        p for p in image_paths if p.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    name_to_id = {}
    images_list = []

    for idx, img_path in enumerate(image_paths, 1):
        filename = os.path.basename(img_path)
        img_name, _ = os.path.splitext(filename)
        name_to_id[img_name] = idx

        with Image.open(img_path) as img:
            w, h = img.size

        images_list.append({"id": idx, "file_name": filename, "width": w, "height": h})

    # Build categories list
    names_dict = db_info.get("names", {})
    categories = []
    for cls_id, cls_name in names_dict.items():
        categories.append(
            {"id": int(cls_id) + 1, "name": str(cls_name), "supercategory": "none"}
        )

    # Build annotations
    annotations = []
    ann_id_counter = 1

    for img_path in image_paths:
        filename = os.path.basename(img_path)
        img_name, _ = os.path.splitext(filename)
        image_id = name_to_id[img_name]

        # Check label file
        label_path = os.path.join(labels_dir, img_name + ".txt")
        if not os.path.exists(label_path):
            continue

        with Image.open(img_path) as img:
            img_w, img_h = img.size

        with open(label_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            cls_id = int(parts[0])
            x_c, y_c, w, h = map(float, parts[1:])

            # Convert YOLO normalized center format to COCO absolute pixel format [x_min, y_min, w, h]
            w_pixel = w * img_w
            h_pixel = h * img_h
            x_min = (x_c - w / 2.0) * img_w
            y_min = (y_c - h / 2.0) * img_h
            area = w_pixel * h_pixel

            annotations.append(
                {
                    "id": ann_id_counter,
                    "image_id": image_id,
                    "category_id": cls_id + 1,  # 1-indexed for COCO
                    "bbox": [x_min, y_min, w_pixel, h_pixel],
                    "area": area,
                    "iscrowd": 0,
                }
            )
            ann_id_counter += 1

    coco_gt_dict = {
        "images": images_list,
        "annotations": annotations,
        "categories": categories,
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(coco_gt_dict, f, indent=4)

    print(
        f"Successfully generated COCO ground truth at {save_path} with {len(images_list)} images."
    )
    return save_path


def safe_load_model(
    model_path: Any,
    device: Any = None,
) -> Any:
    """Safely loads a lightly_train model from checkpoint, fixing custom backbone config issues (e.g. DINOv3 SAT-493M)."""
    import torch
    from lightly_train._task_models.task_model_helpers import (
        _resolve_device,
        init_model_from_checkpoint,
    )

    if isinstance(device, int) or (isinstance(device, str) and device.isdigit()):
        device = f"cuda:{device}"
    device = _resolve_device(device)
    ckpt = torch.load(model_path, weights_only=False, map_location=device)

    # Bugfix: If it's a DINOv3 SAT-493M checkpoint, ensure backbone_args.is_sat493m_weights is set to True.
    # This prevents RuntimeError: Unexpected key(s) in state_dict: "backbone.dinov3.local_cls_norm.weight"...
    patched = False
    original_init = None
    if "sat493m" in str(model_path).lower():
        if "model_init_args" in ckpt:
            if (
                "backbone_args" not in ckpt["model_init_args"]
                or ckpt["model_init_args"]["backbone_args"] is None
            ):
                ckpt["model_init_args"]["backbone_args"] = {}
            ckpt["model_init_args"]["backbone_args"]["is_sat493m_weights"] = True
            ckpt["model_init_args"]["backbone_args"]["weights"] = "sat493m"

        # Monkeypatch DinoVisionTransformer.__init__ to force untie_global_and_local_cls_norm=True
        # when loading a SAT-493M backbone checkpoint (since load_weights=False overrides weights to None).
        try:
            from lightly_train._models.dinov3.dinov3_src.models.vision_transformer import (
                DinoVisionTransformer,
            )

            original_init = DinoVisionTransformer.__init__

            def patched_init(self, *args, **kwargs):
                import inspect

                is_sat = False
                frame = inspect.currentframe()
                while frame:
                    if frame.f_code.co_name.startswith("dinov3_"):
                        if frame.f_locals.get("is_sat493m_weights"):
                            is_sat = True
                            break
                    frame = frame.f_back

                if (
                    is_sat
                    or kwargs.get("is_sat493m_weights")
                    or "sat493m" in str(kwargs.get("weights", "")).lower()
                ):
                    kwargs["untie_global_and_local_cls_norm"] = True
                original_init(self, *args, **kwargs)

            DinoVisionTransformer.__init__ = patched_init
            patched = True
        except Exception as e:
            print(f"Warning: Failed to patch DinoVisionTransformer for SAT-493M: {e}")

    try:
        model_instance = init_model_from_checkpoint(checkpoint=ckpt, device=device)
    finally:
        if patched and original_init is not None:
            try:
                from lightly_train._models.dinov3.dinov3_src.models.vision_transformer import (
                    DinoVisionTransformer,
                )

                DinoVisionTransformer.__init__ = original_init
            except Exception:
                pass
    return model_instance


def evaluate_model_coco(
    model_path_or_model: Any,
    dataset_yaml_path: str,
    split: str,
    eval_results_dir: str,
    run_name: str,
    device: Any = 0,
    batch_size: int = 16,
    imgsz: int = 640,
    workers: int = 0,
    sahi_enabled: bool | None = None,
    sahi_slice_height: int | None = None,
    sahi_slice_width: int | None = None,
    sahi_overlap_height_ratio: float | None = None,
    sahi_overlap_width_ratio: float | None = None,
    sahi_perform_standard_pred: bool | None = None,
    sahi_postprocess_type: str | None = None,
    sahi_postprocess_match_metric: str | None = None,
    sahi_postprocess_match_threshold: float | None = None,
    sahi_global_local_iou_threshold: float | None = None,
) -> Dict[str, Any]:
    """
    Runs model validation on the specified split using pycocotools COCOeval.
    Supports both Ultralytics models and lightly_train models, with optional SAHI inference.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from config import PipelineConfig

    cfg = PipelineConfig()

    if sahi_enabled is None:
        sahi_enabled = getattr(cfg, "sahi_enabled", False)
    if sahi_slice_height is None:
        sahi_slice_height = getattr(cfg, "sahi_slice_height", 512)
    if sahi_slice_width is None:
        sahi_slice_width = getattr(cfg, "sahi_slice_width", 512)
    if sahi_overlap_height_ratio is None:
        sahi_overlap_height_ratio = getattr(cfg, "sahi_overlap_height_ratio", 0.2)
    if sahi_overlap_width_ratio is None:
        sahi_overlap_width_ratio = getattr(cfg, "sahi_overlap_width_ratio", 0.2)
    if sahi_perform_standard_pred is None:
        sahi_perform_standard_pred = getattr(cfg, "sahi_perform_standard_pred", True)
    if sahi_postprocess_type is None:
        sahi_postprocess_type = getattr(cfg, "sahi_postprocess_type", "GREEDYNMM")
    if sahi_postprocess_match_metric is None:
        sahi_postprocess_match_metric = getattr(
            cfg, "sahi_postprocess_match_metric", "IOS"
        )
    if sahi_postprocess_match_threshold is None:
        sahi_postprocess_match_threshold = getattr(
            cfg, "sahi_postprocess_match_threshold", 0.5
        )
    if sahi_global_local_iou_threshold is None:
        sahi_global_local_iou_threshold = getattr(
            cfg, "sahi_global_local_iou_threshold", 0.1
        )

    eval_results_dir = os.path.abspath(eval_results_dir)
    os.makedirs(eval_results_dir, exist_ok=True)

    # 1. Determine model framework type
    is_yolo_fw = False
    model = model_path_or_model

    if isinstance(model_path_or_model, str):
        model_name = os.path.basename(model_path_or_model)
        # Check if it's an ultralytics model or lightly model path
        if (
            "exported_best.pt" in model_path_or_model
            or "exported_last.pt" in model_path_or_model
        ):
            model = safe_load_model(model_path_or_model, device=device)
            is_yolo_fw = False
        elif "yolo" in model_name.lower() or "rtdetr" in model_name.lower():
            if "rtdetr" in model_name.lower() and "dinov3" not in model_path_or_model:
                from ultralytics import RTDETR

                model = RTDETR(model_path_or_model)
            else:
                from ultralytics import YOLO

                model = YOLO(model_path_or_model)
            is_yolo_fw = True
        else:
            # Try loading with lightly
            try:
                model = safe_load_model(model_path_or_model, device=device)
                is_yolo_fw = False
            except Exception:
                from ultralytics import YOLO

                model = YOLO(model_path_or_model)
                is_yolo_fw = True
    else:
        # Check instance type or attributes
        if hasattr(model, "postprocessor") or hasattr(model, "forward_backend"):
            is_yolo_fw = False
        else:
            is_yolo_fw = True

    print(
        f"Evaluating {run_name} ({'Ultralytics' if is_yolo_fw else 'LightlyTrain'}) on the {split} set (SAHI={sahi_enabled})..."
    )

    # 2. Generate/resolve COCO ground truth uniquely based on the split path
    import hashlib

    db_info = parse_dataset_yaml(dataset_yaml_path)
    base_path = db_info.get("path", "")
    split_img_dir = db_info.get(split, "")
    if not os.path.isabs(base_path):
        yaml_dir = os.path.dirname(os.path.abspath(dataset_yaml_path))
        base_path = os.path.abspath(os.path.join(yaml_dir, base_path))
    full_split_path = os.path.abspath(os.path.join(base_path, split_img_dir))
    path_hash = hashlib.md5(full_split_path.encode("utf-8")).hexdigest()[:8]

    gt_file = os.path.join(eval_results_dir, f"coco_gt_{split}_{path_hash}.json")
    generate_coco_gt(dataset_yaml_path, split, gt_file)

    # Load ground truth mapping
    with open(gt_file, "r") as f:
        gt_data = json.load(f)
    name_to_id = {
        os.path.splitext(img["file_name"])[0]: img["id"] for img in gt_data["images"]
    }

    pred_file = os.path.join(eval_results_dir, f"{run_name}_coco_predictions.json")
    os.makedirs(os.path.dirname(pred_file), exist_ok=True)

    # 3. Retrieve predictions
    mapped_preds = []

    if is_yolo_fw:
        if sahi_enabled:
            # Load SAHI detection model using preloaded model
            from sahi.predict import get_sliced_prediction
            from sahi import AutoDetectionModel

            sahi_model_type = (
                "rtdetr"
                if "rtdetr" in str(model_path_or_model).lower()
                or model.__class__.__name__.startswith("RTDetr")
                else "ultralytics"
            )

            # Convert device to SAHI-compatible device string
            sahi_device = device
            if isinstance(sahi_device, int):
                sahi_device = f"cuda:{sahi_device}"
            elif isinstance(sahi_device, str):
                if sahi_device.isdigit():
                    sahi_device = f"cuda:{sahi_device}"

            model_path_str = (
                model_path_or_model if isinstance(model_path_or_model, str) else None
            )
            sahi_model = AutoDetectionModel.from_pretrained(
                model_type=sahi_model_type,
                model=model,
                model_path=model_path_str,
                confidence_threshold=0.001,
                device=sahi_device,
            )

            # Resolve image paths
            db_info = parse_dataset_yaml(dataset_yaml_path)
            base_path = db_info.get("path", "")
            if not os.path.isabs(base_path):
                yaml_dir = os.path.dirname(os.path.abspath(dataset_yaml_path))
                base_path = os.path.abspath(os.path.join(yaml_dir, base_path))

            split_img_dir = (
                os.path.join(base_path, db_info[split])
                if not os.path.isabs(db_info[split])
                else db_info[split]
            )
            img_paths = sorted(glob.glob(os.path.join(split_img_dir, "*.*")))
            img_paths = [
                p for p in img_paths if p.lower().endswith((".png", ".jpg", ".jpeg"))
            ]

            print(f"Running SAHI inference on {len(img_paths)} images...")
            for img_path in img_paths:
                img_name_clean = os.path.splitext(os.path.basename(img_path))[0]
                image_id = name_to_id.get(img_name_clean)
                if image_id is None:
                    continue

                try:
                    # Resolve SAHI arguments to non-optional local variables to satisfy Pyright
                    sh: int = (
                        sahi_slice_height if sahi_slice_height is not None else 512
                    )
                    sw: int = sahi_slice_width if sahi_slice_width is not None else 512
                    oh: float = (
                        sahi_overlap_height_ratio
                        if sahi_overlap_height_ratio is not None
                        else 0.2
                    )
                    ow: float = (
                        sahi_overlap_width_ratio
                        if sahi_overlap_width_ratio is not None
                        else 0.2
                    )
                    psp: bool = (
                        sahi_perform_standard_pred
                        if sahi_perform_standard_pred is not None
                        else True
                    )
                    pt: str = (
                        sahi_postprocess_type
                        if sahi_postprocess_type is not None
                        else "GREEDYNMM"
                    )
                    pmm: str = (
                        sahi_postprocess_match_metric
                        if sahi_postprocess_match_metric is not None
                        else "IOS"
                    )
                    pmt: float = (
                        sahi_postprocess_match_threshold
                        if sahi_postprocess_match_threshold is not None
                        else 0.5
                    )

                    result = get_sliced_prediction(
                        img_path,
                        sahi_model,
                        slice_height=sh,
                        slice_width=sw,
                        overlap_height_ratio=oh,
                        overlap_width_ratio=ow,
                        perform_standard_pred=psp,
                        postprocess_type=pt,
                        postprocess_match_metric=pmm,
                        postprocess_match_threshold=pmt,
                        verbose=0,
                    )

                    for obj_pred in result.object_prediction_list:
                        bbox = obj_pred.bbox.to_coco_bbox()  # [xmin, ymin, w, h]
                        cls = obj_pred.category.id
                        score = obj_pred.score.value

                        mapped_preds.append(
                            {
                                "image_id": image_id,
                                "category_id": int(cls)
                                + 1,  # 1-indexed for COCO format
                                "bbox": bbox,
                                "score": float(score),
                            }
                        )
                except Exception as e:
                    print(f"Error SAHI predicting on {img_path}: {e}")
                    continue
        else:
            # Ultralytics validation pipeline exports predictions.json automatically
            temp_dir_name = f"temp_val_{run_name}"
            model.val(
                data=dataset_yaml_path,
                split=split,
                save_json=True,
                device=device,
                batch=batch_size,
                imgsz=imgsz,
                workers=workers,
                project=eval_results_dir,
                name=temp_dir_name,
                exist_ok=True,
                plots=False,
            )

            temp_val_path = os.path.join(eval_results_dir, temp_dir_name)
            pred_src_file = os.path.join(temp_val_path, "predictions.json")

            if not os.path.exists(pred_src_file):
                raise FileNotFoundError(
                    f"Ultralytics failed to save predictions.json at {pred_src_file}"
                )

            with open(pred_src_file, "r") as f:
                preds = json.load(f)

            unmapped_count = 0
            for p in preds:
                img_name_clean = os.path.splitext(os.path.basename(str(p["image_id"])))[
                    0
                ]
                if img_name_clean in name_to_id:
                    mapped_preds.append(
                        {
                            "image_id": name_to_id[img_name_clean],
                            "category_id": p["category_id"],
                            "bbox": p["bbox"],
                            "score": p["score"],
                        }
                    )
                else:
                    unmapped_count += 1

            if unmapped_count > 0:
                print(
                    f"Warning: {unmapped_count} predictions could not be mapped to ground-truth image IDs."
                )

            try:
                shutil.rmtree(temp_val_path)
            except Exception as e:
                print(f"Error cleaning up temp directory {temp_val_path}: {e}")

    else:
        # lightly_train validation loop
        db_info = parse_dataset_yaml(dataset_yaml_path)
        base_path = db_info.get("path", "")
        if not os.path.isabs(base_path):
            yaml_dir = os.path.dirname(os.path.abspath(dataset_yaml_path))
            base_path = os.path.abspath(os.path.join(yaml_dir, base_path))

        split_img_dir = os.path.join(base_path, db_info[split])
        img_paths = sorted(glob.glob(os.path.join(split_img_dir, "*.*")))
        img_paths = [
            p for p in img_paths if p.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        from typing import Any as TypeAny

        eval_model: TypeAny = model

        print(
            f"Running {'SAHI ' if sahi_enabled else ''}inference on {len(img_paths)} images..."
        )
        for img_path in img_paths:
            img_name_clean = os.path.splitext(os.path.basename(img_path))[0]
            image_id = name_to_id.get(img_name_clean)
            if image_id is None:
                continue

            try:
                if sahi_enabled:
                    res = eval_model.predict_sahi(
                        image=img_path,
                        threshold=0.001,
                        overlap=sahi_overlap_height_ratio,
                        nms_iou_threshold=sahi_postprocess_match_threshold,
                        global_local_iou_threshold=sahi_global_local_iou_threshold,
                    )
                else:
                    res = eval_model.predict(img_path, threshold=0.001)

                bboxes = res.get("bboxes")
                labels = res.get("labels")
                scores = res.get("scores")

                if bboxes is not None and len(bboxes) > 0:
                    bboxes_np = bboxes.cpu().numpy()
                    labels_np = labels.cpu().numpy()
                    scores_np = scores.cpu().numpy()

                    for box, cls, score in zip(bboxes_np, labels_np, scores_np):
                        x_min, y_min, x_max, y_max = map(float, box)
                        w = x_max - x_min
                        h = y_max - y_min
                        mapped_preds.append(
                            {
                                "image_id": image_id,
                                "category_id": int(cls)
                                + 1,  # 1-indexed for COCO format
                                "bbox": [x_min, y_min, w, h],
                                "score": float(score),
                            }
                        )
            except Exception as e:
                print(f"Error predicting on {img_path}: {e}")
                continue

    # Save mapped predictions file
    with open(pred_file, "w") as f:
        json.dump(mapped_preds, f, indent=4)

    # 4. Run pycocotools COCOeval
    coco_gt = COCO(gt_file)

    if len(mapped_preds) == 0:
        print("Warning: No predictions found, creating dummy evaluator results.")
        stats = [0.0] * 12
        ap30 = 0.0
        ap40 = 0.0
    else:
        coco_dt = coco_gt.loadRes(pred_file)
        coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        stats = list(coco_eval.stats)

        # Compute AP30 and AP40 using custom evaluator
        import numpy as np

        coco_eval_custom = COCOeval(coco_gt, coco_dt, iouType="bbox")
        coco_eval_custom.params.iouThrs = np.array([0.3, 0.4])
        coco_eval_custom.evaluate()
        coco_eval_custom.accumulate()
        precision = coco_eval_custom.eval["precision"]

        # dims: [iouThrs, recThrs, cls, areas, maxDets]
        # area range 'all' index 0, maxDets 100 index 2
        s_30 = precision[0, :, :, 0, 2]
        s_40 = precision[1, :, :, 0, 2]

        ap30 = float(np.mean(s_30[s_30 > -1])) if len(s_30[s_30 > -1]) > 0 else 0.0
        ap40 = float(np.mean(s_40[s_40 > -1])) if len(s_40[s_40 > -1]) > 0 else 0.0

        # Display the custom metrics
        print(
            f" Average Precision  (AP) @[ IoU=0.30      | area=   all | maxDets=100 ] = {ap30:.3f}"
        )
        print(
            f" Average Precision  (AP) @[ IoU=0.40      | area=   all | maxDets=100 ] = {ap40:.3f}"
        )

    # Save metrics JSON in original format
    metrics = {
        "model_variant": getattr(model, "ckpt_path", "") or str(model_path_or_model)
        if isinstance(model_path_or_model, str)
        else "unknown",
        "run_name": run_name,
        "split": split,
        "metrics": {
            "AP": stats[0],
            "AP50": stats[1],
            "AP75": stats[2],
            "AP30": ap30,
            "AP40": ap40,
            "AP_small": stats[3],
            "AP_medium": stats[4],
            "AP_large": stats[5],
            "AR_max1": stats[6],
            "AR_max10": stats[7],
            "AR_max100": stats[8],
            "AR_small": stats[9],
            "AR_medium": stats[10],
            "AR_large": stats[11],
        },
    }

    metrics_file = os.path.join(eval_results_dir, f"{run_name}_coco_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Strict COCO metrics saved to {metrics_file}")

    # GPU Memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return metrics


def build_albumentations_pipeline(config_dict: Dict[str, Any], imgsz: int) -> List[Any]:
    """
    Dynamically constructs an Albumentations augmentation pipeline based on configurations.
    """
    import albumentations as A

    return [
        A.RandomRotate90(p=config_dict.get("albu_rotate90_p", 0.0)),
        A.Transpose(p=config_dict.get("albu_rotate90_p", 0.0)),
        A.HorizontalFlip(p=config_dict.get("albu_spatial_p", 0.0)),
        A.VerticalFlip(p=config_dict.get("albu_spatial_p", 0.0)),
        A.OneOf(
            [
                A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
                A.Affine(shear=(-5, 5), p=1.0),
            ],
            p=config_dict.get("albu_warp_p", 0.0),
        ),
        A.RandomResizedCrop(
            size=(imgsz, imgsz),
            scale=(0.4, 1.0),
            ratio=(0.9, 1.1),
            p=config_dict.get("albu_crop_p", 0.0),
        ),
        A.OneOf(
            [
                A.Sharpen(alpha=(0.2, 0.5), p=1.0),
                A.Blur(blur_limit=3, p=1.0),
            ],
            p=config_dict.get("albu_texture_p", 0.0),
        ),
        A.OneOf(  # type: ignore
            [
                A.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=1.0
                ),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
                A.RandomShadow(  # type: ignore
                    shadow_roi=(0, 0, 1, 1),
                    num_shadows_lower=1,  # type: ignore
                    num_shadows_upper=2,  # type: ignore
                    shadow_dimension=5,
                    p=1.0,
                ),
                A.ToGray(p=0.1),
                A.Solarize(threshold=128, p=0.05),  # type: ignore
            ],
            p=config_dict.get("albu_color_p", 0.0),
        ),
        A.CoarseDropout(
            num_holes_range=(8, 12),
            hole_height_range=(0.02, 0.05),
            hole_width_range=(0.02, 0.05),
            p=config_dict.get("albu_dropout_p", 0.0),
        ),
        A.OneOf(
            [
                A.GaussNoise(std_range=(0.02, 0.08), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
                A.ImageCompression(quality_range=(75, 100), p=1.0),
            ],
            p=config_dict.get("albu_noise_p", 0.0),
        ),
    ]


def get_huggingface_backbone(model_id: str) -> str:
    """
    Downloads a model from Hugging Face if needed, converts its model.safetensors
    to a PyTorch .pt state dict, and returns the path to the converted checkpoint.
    """
    import os
    from dotenv import load_dotenv
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    import torch

    # Load environment variables from .env file
    project_root = os.path.dirname(os.path.abspath(__file__))
    load_dotenv(dotenv_path=os.path.join(project_root, ".env"))
    hf_token = os.getenv("HF_TOKEN")

    # Convert repo ID (e.g. facebook/dinov3-vitl16-pretrain-sat493m) to a filename
    repo_clean = model_id.replace("/", "--")
    cache_dir = os.path.join(project_root, "hf_cache")
    os.makedirs(cache_dir, exist_ok=True)
    pt_path = os.path.join(cache_dir, f"{repo_clean}.pt")

    if os.path.exists(pt_path):
        try:
            # Check if cached checkpoint is complete (e.g., has local_cls_norm keys for sat493m)
            sd_check = torch.load(pt_path, map_location="cpu", weights_only=True)
            if "sat493m" in model_id and "local_cls_norm.weight" not in sd_check:
                print(
                    f"Cached weights at {pt_path} are missing required keys for sat493m. Re-converting..."
                )
            else:
                print(f"Loaded converted PyTorch weights from cache: {pt_path}")
                return pt_path
        except Exception as e:
            print(f"Error loading cached weights: {e}. Re-converting...")

    print(f"Downloading {model_id} from Hugging Face...")
    safetensors_path = os.path.join(cache_dir, f"{repo_clean}.safetensors")

    import requests
    import time

    url = f"https://huggingface.co/{model_id}/resolve/main/model.safetensors"
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}

    try:
        response = requests.get(url, headers=headers, stream=True, timeout=30)
        if response.status_code != 200:
            print(
                f"Requests returned status code {response.status_code}. Falling back to huggingface_hub..."
            )
            safetensors_path = hf_hub_download(
                repo_id=model_id, filename="model.safetensors", token=hf_token
            )
        else:
            total_size = int(response.headers.get("content-length", 0))
            print(f"Total size to download: {total_size / (1024 * 1024):.2f} MB")

            downloaded = 0
            chunk_size = 1024 * 1024  # 1 MB chunks
            last_print = 0
            start_time = time.time()

            with open(safetensors_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

                        if (
                            downloaded - last_print >= 10 * 1024 * 1024
                            or downloaded == total_size
                        ):
                            elapsed = time.time() - start_time
                            speed = downloaded / (elapsed + 1e-6) / (1024 * 1024)
                            pct = (
                                (downloaded / total_size * 100) if total_size > 0 else 0
                            )
                            print(
                                f"Downloaded: {downloaded / (1024 * 1024):.2f} / {total_size / (1024 * 1024):.2f} MB ({pct:.1f}%) | Speed: {speed:.2f} MB/s",
                                flush=True,
                            )
                            last_print = downloaded

            print("Download finished. Converting weights to PyTorch format...")

        print("Converting weights to PyTorch format...")
        state_dict = load_file(safetensors_path)

        # Convert Hugging Face Dinov3Model state_dict to timm/Meta format expected by lightly_train
        print("Mapping Hugging Face state_dict keys to timm/Meta format...")
        new_state_dict = {}
        new_state_dict["cls_token"] = state_dict["embeddings.cls_token"]
        mask_token = state_dict["embeddings.mask_token"]
        if len(mask_token.shape) == 3 and mask_token.shape[0] == 1:
            mask_token = mask_token.squeeze(0)
        new_state_dict["mask_token"] = mask_token
        new_state_dict["patch_embed.proj.weight"] = state_dict[
            "embeddings.patch_embeddings.weight"
        ]
        new_state_dict["patch_embed.proj.bias"] = state_dict[
            "embeddings.patch_embeddings.bias"
        ]
        if "embeddings.register_tokens" in state_dict:
            new_state_dict["storage_tokens"] = state_dict["embeddings.register_tokens"]

        new_state_dict["norm.weight"] = state_dict["norm.weight"]
        new_state_dict["norm.bias"] = state_dict["norm.bias"]
        if "sat493m" in model_id:
            new_state_dict["local_cls_norm.weight"] = state_dict["norm.weight"].clone()
            new_state_dict["local_cls_norm.bias"] = state_dict["norm.bias"].clone()

        layers = set()
        for k in state_dict.keys():
            if k.startswith("layer."):
                parts = k.split(".")
                layers.add(int(parts[1]))

        for layer_idx in sorted(layers):
            src_prefix = f"layer.{layer_idx}."
            dst_prefix = f"blocks.{layer_idx}."

            new_state_dict[f"{dst_prefix}norm1.weight"] = state_dict[
                f"{src_prefix}norm1.weight"
            ]
            new_state_dict[f"{dst_prefix}norm1.bias"] = state_dict[
                f"{src_prefix}norm1.bias"
            ]
            new_state_dict[f"{dst_prefix}norm2.weight"] = state_dict[
                f"{src_prefix}norm2.weight"
            ]
            new_state_dict[f"{dst_prefix}norm2.bias"] = state_dict[
                f"{src_prefix}norm2.bias"
            ]

            if f"{src_prefix}layer_scale1.lambda1" in state_dict:
                new_state_dict[f"{dst_prefix}ls1.gamma"] = state_dict[
                    f"{src_prefix}layer_scale1.lambda1"
                ]
            if f"{src_prefix}layer_scale2.lambda1" in state_dict:
                new_state_dict[f"{dst_prefix}ls2.gamma"] = state_dict[
                    f"{src_prefix}layer_scale2.lambda1"
                ]

            new_state_dict[f"{dst_prefix}mlp.fc1.weight"] = state_dict[
                f"{src_prefix}mlp.up_proj.weight"
            ]
            new_state_dict[f"{dst_prefix}mlp.fc1.bias"] = state_dict[
                f"{src_prefix}mlp.up_proj.bias"
            ]
            new_state_dict[f"{dst_prefix}mlp.fc2.weight"] = state_dict[
                f"{src_prefix}mlp.down_proj.weight"
            ]
            new_state_dict[f"{dst_prefix}mlp.fc2.bias"] = state_dict[
                f"{src_prefix}mlp.down_proj.bias"
            ]

            new_state_dict[f"{dst_prefix}attn.proj.weight"] = state_dict[
                f"{src_prefix}attention.o_proj.weight"
            ]
            new_state_dict[f"{dst_prefix}attn.proj.bias"] = state_dict[
                f"{src_prefix}attention.o_proj.bias"
            ]

            q_w = state_dict[f"{src_prefix}attention.q_proj.weight"]
            k_w = state_dict[f"{src_prefix}attention.k_proj.weight"]
            v_w = state_dict[f"{src_prefix}attention.v_proj.weight"]
            new_state_dict[f"{dst_prefix}attn.qkv.weight"] = torch.cat(
                [q_w, k_w, v_w], dim=0
            )

            q_b = state_dict[f"{src_prefix}attention.q_proj.bias"]
            v_b = state_dict[f"{src_prefix}attention.v_proj.bias"]
            k_b_key = f"{src_prefix}attention.k_proj.bias"
            if k_b_key in state_dict:
                k_b = state_dict[k_b_key]
            else:
                k_b = torch.zeros_like(q_b)
            new_state_dict[f"{dst_prefix}attn.qkv.bias"] = torch.cat(
                [q_b, k_b, v_b], dim=0
            )

        # Copy remaining/constant keys (like rope_embed.periods) from a default backbone model of matching size
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
            print(
                f"Warning: could not load default template model to copy remaining keys: {e}"
            )

        torch.save(new_state_dict, pt_path)
        print(f"Conversion complete. Weights saved to {pt_path}")

        # Clean up temporary safetensors file to save space
        if os.path.exists(safetensors_path) and safetensors_path.endswith(
            ".safetensors"
        ):
            try:
                os.remove(safetensors_path)
                print(f"Cleaned up temporary safetensors file: {safetensors_path}")
            except Exception as e:
                print(
                    f"Warning: Could not remove temporary safetensors file {safetensors_path}: {e}"
                )

        return pt_path
    except Exception as e:
        print(f"Error downloading/converting Hugging Face model {model_id}: {e}")
        raise e
