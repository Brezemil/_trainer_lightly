import sys
import os
import torch
from ultralytics.utils.patches import _torch_save

# Restore the original torch.save to bypass the retry mechanism
# and expose the real underlying serialization exception.
torch.save = _torch_save

# Set PYTHONNOUSERSITE=1 to prevent global user package pollution
os.environ["PYTHONNOUSERSITE"] = "1"

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Stub WandbLogger log_image to prevent FileNotFoundError
try:
    from pytorch_lightning.loggers import WandbLogger

    WandbLogger.log_image = lambda *args, **kwargs: None
except ImportError:
    pass

# Mimic the user's CLI arguments
sys.argv = [
    "run_training.py",
    "--model",
    "facebook/dinov3-vitl16-pretrain-sat493m",
    "--backend",
    "lightly",
    "--decoder",
    "rtdetrv2",
    "--device",
    "cpu",
    "--fraction",
    "0.01",
    "--epochs",
    "1",
    "--batch",
    "1",
    "--runs-dir",
    "runs/test_unpatched",
]

import run_training  # noqa: E402

if __name__ == "__main__":
    run_training.main()
