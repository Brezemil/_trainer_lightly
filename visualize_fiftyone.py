"""
FiftyOne Visualizer for Model Predictions.

This script loads the specified dataset split (default: test) from dataset.yaml,
parses the pycocotools prediction JSON file for a given model,
adds both ground truth labels and predictions to a FiftyOne dataset,
and launches the interactive FiftyOne app.
"""

import os
import sys
import json
import yaml
import glob
from PIL import Image
import fiftyone as fo


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize ground truth and model predictions in FiftyOne."
    )
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to the pycocotools predictions JSON file (e.g., evaluation_results/baseline/yolo11s_seed_42_coco_predictions.json)",
    )
    parser.add_argument(
        "--dataset-yaml",
        type=str,
        default=r"C:\Users\emilb\_data\_smoketest\dataset.yaml",
        help="Path to the dataset.yaml file.",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        default="test",
        help="Split to visualize (default: test).",
    )
    return parser.parse_args()


def load_dataset_info(dataset_yaml_path):
    with open(dataset_yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return data


def main():
    args = parse_args()

    if not os.path.exists(args.predictions):
        print(f"Error: Predictions file not found: {args.predictions}")
        sys.exit(1)

    if not os.path.exists(args.dataset_yaml):
        print(f"Error: Dataset YAML not found: {args.dataset_yaml}")
        sys.exit(1)

    print(f"Loading dataset info from {args.dataset_yaml}...")
    db_info = load_dataset_info(args.dataset_yaml)

    base_path = db_info.get("path", "")
    if not os.path.isabs(base_path):
        yaml_dir = os.path.dirname(os.path.abspath(args.dataset_yaml))
        base_path = os.path.abspath(os.path.join(yaml_dir, base_path))

    split_dir = db_info.get(args.split, "")
    images_dir = (
        os.path.join(base_path, split_dir)
        if not os.path.isabs(split_dir)
        else split_dir
    )
    labels_dir = images_dir.replace("images", "labels")

    names_dict = db_info.get("names", {0: "object"})
    # Convert keys to int and values to string
    class_map = {int(k): str(v) for k, v in names_dict.items()}

    print(f"Searching images in: {images_dir}")
    image_paths = sorted(glob.glob(os.path.join(images_dir, "*.*")))
    image_paths = [
        p for p in image_paths if p.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    if not image_paths:
        print(f"Error: No images found in {images_dir}")
        sys.exit(1)

    print(f"Loading predictions from {args.predictions}...")
    with open(args.predictions, "r") as f:
        predictions = json.load(f)

    # Build a lookup dictionary for predictions: image_id -> list of predictions
    # Predictions in pycocotools format have category_id (1-indexed for COCO)
    # We map them back to class name using class_map[category_id - 1]
    # We also need to match predictions to file names.
    # To do this, let's regenerate the same image name -> image id mapping used during evaluation.
    name_to_id = {}
    id_to_filename = {}
    for idx, img_path in enumerate(image_paths, 1):
        filename = os.path.basename(img_path)
        img_name, _ = os.path.splitext(filename)
        name_to_id[img_name] = idx
        id_to_filename[idx] = img_path

    pred_by_id = {}
    for p in predictions:
        img_id = int(p["image_id"])
        if img_id not in pred_by_id:
            pred_by_id[img_id] = []
        pred_by_id[img_id].append(p)

    print("Initializing FiftyOne Dataset...")
    # Clean up previous dataset with same name if exists
    dataset_name = f"dinov3_benchmark_{args.split}"
    if dataset_name in fo.list_datasets():
        fo.delete_dataset(dataset_name)

    dataset = fo.Dataset(dataset_name)

    samples = []
    print("Parsing images, ground truths, and predictions...")

    for img_path in image_paths:
        filename = os.path.basename(img_path)
        img_name, _ = os.path.splitext(filename)
        img_id = name_to_id[img_name]

        # Load image size for normalization
        with Image.open(img_path) as img:
            w, h = img.size

        sample = fo.Sample(filepath=img_path)

        # 1. Parse and add Ground Truth Detections (YOLO format)
        gt_detections = []
        label_path = os.path.join(labels_dir, img_name + ".txt")

        if os.path.exists(label_path):
            with open(label_path, "r") as lf:
                lines = lf.readlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                class_id = int(parts[0])
                x_c, y_c, box_w, box_h = map(float, parts[1:])

                # Convert normalized center-based [x_c, y_c, box_w, box_h]
                # to FiftyOne [x_min, y_min, box_w, box_h] (still normalized)
                x_min = x_c - box_w / 2.0
                y_min = y_c - box_h / 2.0

                label_name = class_map.get(class_id, f"class_{class_id}")
                gt_detections.append(
                    fo.Detection(
                        label=label_name, bounding_box=[x_min, y_min, box_w, box_h]
                    )
                )
        sample["ground_truth"] = fo.Detections(detections=gt_detections)

        # 2. Parse and add Model Predictions (COCO format)
        pred_detections = []
        img_preds = pred_by_id.get(img_id, [])

        for p in img_preds:
            # category_id is 1-indexed for COCO, mapping back to 0-indexed for class_map
            class_id = int(p["category_id"]) - 1
            label_name = class_map.get(class_id, f"class_{class_id}")
            score = float(p["score"])

            # COCO bbox: absolute pixel [x_min, y_min, box_w, box_h]
            # Convert to FiftyOne normalized format
            bx, by, bw, bh = p["bbox"]
            norm_box = [bx / w, by / h, bw / w, bh / h]

            pred_detections.append(
                fo.Detection(label=label_name, bounding_box=norm_box, confidence=score)
            )
        sample["predictions"] = fo.Detections(detections=pred_detections)
        samples.append(sample)

    dataset.add_samples(samples)
    print(f"Successfully loaded {len(dataset)} samples into FiftyOne.")  # type: ignore

    # Launch FiftyOne App session
    print("Launching FiftyOne Interactive Desktop App/Web Interface...")
    session = fo.launch_app(dataset)
    session.wait()  # Block terminal execution to keep FiftyOne app alive


if __name__ == "__main__":
    main()
