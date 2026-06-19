import os
from dataclasses import dataclass


@dataclass
class PipelineConfig:
    # W&B Core Setup
    entity: str = "brezemil"
    project: str = "_trainer"  # Map to standard project

    # Model & Data Paths
    model_variant: str = "yolo26s.pt"
    dataset_path: str = r"C:\Users\emilb\_data\_smoketest\dataset.yaml"

    # Optional Sweep IDs to plug in
    aug_sweep_id: str | None = None
    hpo_sweep_id: str | None = None

    # Global Training Constraints
    image_size: int = 640
    max_sweep_runs: int = 1
    sweep_epochs: int = 1
    prod_epochs: int = 1
    device: int = 0
    batch_size: int = 2
    workers: int = 0
    fraction: float = 1.0
    amp: bool = False

    # Stock Training Settings (Models & Seeds)
    models: tuple = ("yolo11n.pt", "yolo26n.pt", "rtdetr-l.pt")
    seeds: tuple = (42, 100, 999)
    runs_dir: str = "runs"
    wandb_dir: str = "wandb"
    eval_results_dir: str = "evaluation_results"

    # Fixed Loss Parameters (Baseline)
    fixed_loss: dict | None = None

    # Central decoder head config (rtdetrv2 or dfine)
    decoder_name: str = "rtdetrv2"

    # SAHI Configuration
    sahi_enabled: bool = False
    sahi_slice_height: int = 512
    sahi_slice_width: int = 512
    sahi_overlap_height_ratio: float = 0.2
    sahi_overlap_width_ratio: float = 0.2
    sahi_perform_standard_pred: bool = True
    sahi_postprocess_type: str = "GREEDYNMM"
    sahi_postprocess_match_metric: str = "IOS"
    sahi_postprocess_match_threshold: float = 0.5
    sahi_global_local_iou_threshold: float = 0.1

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
