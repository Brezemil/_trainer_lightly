---
source_url: https://docs.lightly.ai/train/stable/installation.html
---

# InstallationÂ¶

## Installation from PyPIÂ¶

Lightly**Train** is available on [PyPI](https://pypi.org/project/lightly-train/) and can be installed via pip or other package managers.

Warning

To successfully install Lightly**Train** the Python version has to be >=3.8 and <=3.13 .
    
    
    pip install lightly-train
    

To update to the latest version, run:
    
    
    pip install --upgrade lightly-train
    

See [Docker](docker.md#docker) for Docker installation instructions.

## Platform CompatibilityÂ¶

Platform | Supported Compute  
---|---  
Linux | CPU or CUDA  
MacOS | CPU (MPS is planned)  
Windows | CPU or CUDA (experimental)  
  
## Version CompatibilityÂ¶

`lightly-train` | `torch` | `torchvision` | `pytorch-lightning` | Python  
---|---|---|---|---  
`>=0.14.1` | `>=2.1` | `>=0.16` | `>=2.1` | `>=3.8`, `<3.14`  
`>=0.12` | `>=2.1` | `>=0.16` | `>=2.1` | `>=3.8`, `<3.13`  
`>=0.6` | `>=2.1`, `<2.6` | `>=0.16` | `>=2.1` | `>=3.8`, `<3.13`  
  
Warning

We recommend installing versions of the `torch`, `torchvision`, and `pytorch-lightning` packages that are compatible with each other.

See the [Torchvision](https://github.com/pytorch/vision?tab=readme-ov-file#installation) and [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/versioning.html#compatibility-matrix) documentation for more information on version compatibility between different PyTorch packages.

## Optional DependenciesÂ¶

Lightly**Train** has optional dependencies that are not installed by default. The following dependencies are available:

### LoggingÂ¶

  * `mlflow`: For logging to [MLflow](https://docs.lightly.ai/train/stable/pretrain_distill/index.html#mlflow)

  * `wandb`: For logging to [Weights & Biases](https://docs.lightly.ai/train/stable/pretrain_distill/index.html#wandb)




### Model SupportÂ¶

  * `rfdetr`: For [RF-DETR](pretrain_distill__models__rfdetr.md#models-rfdetr) models

  * `super-gradients`: For [SuperGradients](pretrain_distill__models__supergradients.md#models-supergradients) models

  * `timm`: For [TIMM](pretrain_distill__models__timm.md#models-timm) models

  * `ultralytics`: For [Ultralytics](pretrain_distill__models__ultralytics.md#models-ultralytics) models




To install optional dependencies, run:
    
    
    pip install "lightly-train[wandb]"
    

Or for multiple optional dependencies:
    
    
    pip install "lightly-train[wandb,timm]"
    

## Hardware RecommendationsÂ¶

An example hardware setup and its performance when using Lightly**Train** is provided in [Hardware Recommendations](performance__hardware_recommendations.md#hardware-recommendations).
