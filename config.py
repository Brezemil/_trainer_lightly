import os
from dataclasses import dataclass


@dataclass
class PipelineConfig:
    """Unified configuration class for the Geospatial CV & YOLO training pipeline.

    Provides a single source of truth for experiment setups, data paths,
    hyperparameter constraints, baseline models, W&B sweeps, and Slicing Aided
    Hyper Inference (SAHI) settings.
    """

    # =========================================================================
    # Weights & Biases (W&B) Core Setup
    # =========================================================================
    entity: str = "brezo-boku-vienna"
    """W&B entity name (user or organization account) under which runs are logged.
    Default: 'brezemil' (Standard workspace account).
    """

    project: str = "_baseline"  # _smoketests or _lightly_train_baseline or _baseline
    """W&B project name to group related experiment runs and sweeps.
    Default: '_trainer' (Central training workspace project).
    """

    # =========================================================================
    # Model & Data Paths
    # =========================================================================
    model_variant: str = "yolo12s.pt"
    """Target baseline model checkpoint filename to train/evaluate.
    Default: 'yolo12s.pt' (Area-Attention optimized YOLO12 Small architecture).
    """

    dataset_path: str = (
        r"C:\Users\emil_brezovsky\Documents\GitHub\_dataset_ail\dataset.yaml"
    )
    """Path to the dataset.yaml file defining the data splits and class mapping.
    Default: 'C:\\Users\\emilb\\_data\\_smoketest\\dataset.yaml' (smoketest dataset).
    Production Standard: Replace with absolute path to full dataset YAML config.
    """

    # =========================================================================
    # Weights & Biases Sweep IDs
    # =========================================================================
    aug_sweep_id: str | None = None
    """W&B Sweep ID for Phase 1 (Augmentation Tuning). If set, training loads
    the best augmentation configuration from the specified sweep.
    Default: None.
    """

    hpo_sweep_id: str | None = None
    """W&B Sweep ID for Phase 2 (Hyperparameter Optimization). If set, training
    loads the best learning hyperparameters from the specified HPO sweep.
    Default: None.
    """

    # =========================================================================
    # Global Training Constraints & Hyperparameters
    # =========================================================================
    image_size: int = 1024
    """Model input image resolution (height and width in pixels).
    Standard Values:
      - 640 (Standard YOLO and RT-DETR default)
      - 1024 or 1280 (High-resolution aerial/satellite detection; improves AP_small
        but increases VRAM usage)
    """

    max_sweep_runs: int = 100
    """Maximum number of Bayesian search trials for W&B Sweep Agents to run.
    Default: 100 (Deep parameter space exploration with Hyperband early stopping).
    """

    sweep_epochs: int = 100
    """Number of training epochs to execute for each sweep trial run.
    Default: 1 (Minimal smoketest limit).
    Production Standard: 10 to 30 epochs for rapid HPO, allowing early convergence indicator.
    """

    prod_epochs: int = 300
    """Number of training epochs to execute for the final baseline or tuned production runs.
    Default: 1 (Smoketest limit).
    Production Standard: 100 to 300 epochs (full convergence of training).
    """

    # -------------------------------------------------------------------------
    # Model-Specific Configuration (YOLO, RT-DETR, DINO)
    # -------------------------------------------------------------------------
    yolo_epochs: int = 300
    """Default training epochs for YOLO-based models."""

    yolo_patience: int = 100
    """Default training patience (early stopping) for YOLO-based models."""

    yolo_batch_size: int = 8
    """Default batch size for YOLO-based models (8 fits 100% inside 16GB VRAM at 1024x1024 without PCIe RAM paging)."""

    rtdetr_epochs: int = 300
    """Default training epochs for RT-DETR models."""

    rtdetr_patience: int = 100
    """Default training patience (early stopping) for RT-DETR models."""

    rtdetr_batch_size: int = 4
    """Default batch size for RT-DETR models."""

    dino_epochs: int = 50
    """Default training epochs for DINO-based models in lightly_train.
    The step-size is calculated automatically through this epoch input by checking the dataset.
    """

    dino_patience: int | None = None
    """Default training patience for DINO-based models (None because lightly_train does not support early stopping)."""

    dino_batch_size: int = 2
    """Default batch size for DINO-based models."""

    device: int = 0
    """Target GPU device index to execute PyTorch training on.
    Standard Values:
      - 0 (GPU device index 0)
      - -1 or 'cpu' (CPU training fallback)
    """

    batch_size: int = 4
    """Number of images processed per training step (global batch size).
    Default: 2 (VRAM smoketest default).
    Production Standard:
      - 16 or 32 (Standard batch size for stable gradient estimation on yolo11s/yolo26s)
      - 8 (For large transformer/DINOv3 models to prevent Out-Of-Memory errors)
    """

    eval_fallback_batch_size: int = 2
    """Fallback batch size used during strict evaluation when the main training
    batch size is configured as auto-tuning ('auto' or -1). 
    Helps prevent CUDA Out-Of-Memory (OOM) errors during evaluation on lower-VRAM GPUs
    (e.g., 16 GB cards) when running high-resolution models.
    Default: 2.
    """

    workers: int = 4
    """Number of CPU dataloader subprocess worker threads for loading/augmenting data.
    Default: 0 (Required on Windows to prevent multi-processing spawn overhead).
    Production Standard: 4 to 8 workers per GPU (usually set to CPU cores / num_gpus).
    """

    fraction: float = 1.0
    """Fraction of the dataset to train on (e.g. 0.05 for 5% dataset subset).
    Default: 1.0 (Uses full dataset).
    Smoketest Standard: 0.05 to 0.1 (Speeds up pipeline validation testing).
    """

    amp: bool = True
    """Enable or disable Automatic Mixed Precision (AMP) training.
    Default: False (Baseline comparability).
    Production Standard: True (Speeds up training and reduces VRAM using FP16/BF16 mixed precision).
    """

    backbone_freeze: bool = True
    """Freeze the DINOv2 or DINOv3 backbone weights during fine-tuning.
    Default: True (Standard setting to significantly reduce VRAM footprint and prevent overfitting).
    """

    # =========================================================================
    # Stock Training Settings & Directories
    # =========================================================================
    models: tuple = ("yolo11s.pt", "yolo26s.pt", "rtdetr-l.pt")
    """Baseline models list to execute inside the baseline benchmark suite run.
    Standard values: yolo11n.pt, yolo26n.pt, rtdetr-l.pt (standard comparison across backends).
    """

    seeds: tuple = (42, 100, 999)
    """List of replication random seed seeds used in baseline benchmarks.
    Default: (42, 100, 999) (Standard 3-seed benchmark for computing variance/SEM).
    """

    runs_dir: str = "runs"
    """Output directory where training runs, checkpoints, and weights are saved.
    Default: 'runs' (relative to workspace).
    """

    wandb_dir: str = "wandb"
    """Directory where Weights & Biases local cache/run logs are saved.
    Default: 'wandb' (relative to workspace).
    """

    eval_results_dir: str = "evaluation_results"
    """Directory where COCO metrics reports and predictions are saved.
    Default: 'evaluation_results' (relative to workspace).
    """

    # =========================================================================
    # Loss Parameters
    # =========================================================================
    fixed_loss: dict | None = None
    """Fixed loss parameters dictionary used during stock baseline training.
    Default: None (resolves to model standard defaults).
    """

    # =========================================================================
    # Decoder Head Options
    # =========================================================================
    decoder_name: str = "rtdetrv2"
    """LTDETR transformer decoder head class selection.
    Standard Options:
      - 'rtdetrv2' (RT-DETRv2 head: continuous L1 + GIoU bounding box regression)
      - 'dfine' (D-FINE head: Discrete Fine-grained Distribution Refinement, treat
        coordinates as bins. Best for strict precision, e.g. AP75 and AP_small in satellite data)
    """

    # =========================================================================
    # Slicing Aided Hyper Inference (SAHI) Configuration
    # =========================================================================
    sahi_enabled: bool = False
    """Toggle Slicing Aided Hyper Inference (SAHI) evaluation on high-resolution data.
    Default: False (Normal full-resolution inference).
    Production Standard: True (For large aerial/satellite imagery to detect small objects).
    """

    sahi_slice_height: int = 512
    sahi_slice_width: int = 512
    """Dimensions of each overlapping tile slice in pixels.
    Standard Values: 512 or 640 (Resolves trade-off between tile context and resolution).
    """

    sahi_overlap_height_ratio: float = 0.2
    sahi_overlap_width_ratio: float = 0.2
    """Percentage of overlap between adjacent tile slices (horizontal/vertical).
    Standard Values: 0.15 to 0.25 (0.2 is default; higher reduces edge truncation but increases slices).
    """

    sahi_perform_standard_pred: bool = True
    """Enable parallel standard full-resolution inference to capture larger objects spanning multiple slices.
    Default: True (recommended).
    """

    sahi_postprocess_type: str = "GREEDYNMM"
    """Deduplication method for merging predictions from overlapping slices.
    Standard Options:
      - 'GREEDYNMM' (Greedy Non-Maximum Suppression, default: recommended in SAHI)
      - 'NMS' (Standard Non-Maximum Suppression)
      - 'NMM' (Non-Maximum Merging)
    """

    sahi_postprocess_match_metric: str = "IOS"
    """Intersection calculation metric for matching overlapping boxes.
    Standard Options:
      - 'IOS' (Intersection over Smaller, default: best for containing boxes and tile edges)
      - 'IOU' (Intersection over Union)
    """

    sahi_postprocess_match_threshold: float = 0.5
    """IoU/IoS threshold above which overlapping bounding boxes are merged/suppressed.
    Standard Values: 0.45 to 0.60 (0.50 is default).
    """

    sahi_global_local_iou_threshold: float = 0.1
    """Intersection over Union threshold to suppress tile predictions if they significantly
    overlap with a global detection of the same class.
    Default: 0.1 (reduces duplicate box clusters on large objects).
    """

    def __post_init__(self):
        # Allow models to start at their standard values by leaving fixed_loss as None by default
        self.fixed_loss = None

        # Resolve logging paths relative to project root (directory of config.py)
        root_dir = os.path.dirname(os.path.abspath(__file__))
        if not os.path.isabs(self.runs_dir):
            self.runs_dir = os.path.abspath(os.path.join(root_dir, self.runs_dir))
        if not os.path.isabs(self.wandb_dir):
            self.wandb_dir = os.path.abspath(os.path.join(root_dir, self.wandb_dir))
        if not os.path.isabs(self.eval_results_dir):
            self.eval_results_dir = os.path.abspath(
                os.path.join(root_dir, self.eval_results_dir)
            )

    @staticmethod
    def get_model_type(model_name: str, backend: str | None = None) -> str:
        """Classify the model type into 'yolo', 'rtdetr', or 'dino' based on model name and backend."""
        name_lower = model_name.lower()
        if "yolo" in name_lower:
            return "yolo"
        elif "rtdetr" in name_lower:
            if backend == "lightly":
                return "dino"
            return "rtdetr"
        elif "dino" in name_lower or name_lower.startswith("facebook/"):
            return "dino"
        return "yolo"  # default fallback

    def get_model_params(
        self, model_name: str, backend: str | None = None
    ) -> tuple[int, int | None, int]:
        """Returns (epochs, patience, batch_size) for the given model and backend."""
        m_type = self.get_model_type(model_name, backend)
        if m_type == "yolo":
            return self.yolo_epochs, self.yolo_patience, self.yolo_batch_size
        elif m_type == "rtdetr":
            return self.rtdetr_epochs, self.rtdetr_patience, self.rtdetr_batch_size
        else:  # dino
            return self.dino_epochs, self.dino_patience, self.dino_batch_size

    @staticmethod
    def get_best_sweep_config(sweep_id: str, project: str, entity: str) -> dict:
        """Fetches the config of the best performing run from a specified W&B sweep."""
        import wandb

        api = wandb.Api()
        sweep = api.sweep(f"{entity}/{project}/{sweep_id}")

        # Try finding the best run by metrics/mAP50-95(B) or standard validation mAP
        metric_name = "metrics/mAP50-95(B)"
        runs = sorted(
            sweep.runs,
            key=lambda run: run.summary.get(
                metric_name, run.summary.get("metrics/AP", 0.0)
            ),
            reverse=True,
        )
        if not runs:
            raise ValueError(f"No runs found in sweep {sweep_id}.")

        # Strip internal W&B keys starting with '_'
        return {k: v for k, v in runs[0].config.items() if not k.startswith("_")}
