---
source_url: https://obss.github.io/sahi/
---

[ ](https://github.com/obss/sahi/tree/main/docs/index.md "Edit this page")

![SAHI logo](https://raw.githubusercontent.com/obss/sahi/main/docs/images/sahi-logo.svg)

#  SAHI: Slicing Aided Hyper Inference 

####  A lightweight vision library for performing large scale object detection & instance segmentation 

####  ![teaser](https://raw.githubusercontent.com/obss/sahi/main/resources/sliced_inference.gif)

[![downloads](https://pepy.tech/badge/sahi)](https://pepy.tech/project/sahi) [![downloads](https://pepy.tech/badge/sahi/month)](https://pepy.tech/project/sahi) [![License](https://img.shields.io/pypi/l/sahi)](https://github.com/obss/sahi/blob/main/LICENSE.md) [![pypi version](https://badge.fury.io/py/sahi.svg)](https://badge.fury.io/py/sahi) [![conda version](https://anaconda.org/conda-forge/sahi/badges/version.svg)](https://anaconda.org/conda-forge/sahi) [![Continuous Integration](https://github.com/obss/sahi/actions/workflows/ci.yml/badge.svg)](https://github.com/obss/sahi/actions/workflows/ci.yml)   
[![Context7 MCP](https://img.shields.io/badge/Context7%20MCP-Indexed-blue)](https://context7.com/obss/sahi) [![llms.txt](https://img.shields.io/badge/llms.txt-✓-brightgreen)](https://context7.com/obss/sahi/llms.txt) [![ci](https://img.shields.io/badge/DOI-10.1109%2FICIP46576.2022.9897990-orange.svg)](https://ieeexplore.ieee.org/document/9897990) [![arXiv](https://img.shields.io/badge/arXiv-2202.06934-b31b1b.svg)](https://arxiv.org/abs/2202.06934) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_ultralytics.ipynb) [![HuggingFace Spaces](https://raw.githubusercontent.com/obss/sahi/main/resources/hf_spaces_badge.svg)](https://huggingface.co/spaces/fcakyon/sahi-yolox)

## What is SAHI?¶

SAHI (Slicing Aided Hyper Inference) is an open-source framework that provides a generic slicing-aided inference and fine-tuning pipeline for small object detection. Detecting small objects and those far from the camera is a major challenge in surveillance applications, as they are represented by a small number of pixels and lack sufficient detail for conventional detectors.

SAHI addresses this by applying a unique methodology that can be used with any object detector without requiring additional fine-tuning. Experimental evaluations on the Visdrone and xView aerial object detection datasets show that SAHI can increase object detection AP by up to 6.8% for FCOS, 5.1% for VFNet, and 5.3% for TOOD detectors. With slicing-aided fine-tuning, the accuracy can be further improved, resulting in a cumulative increase of 12.7%, 13.4%, and 14.5% AP, respectively. The technique has been successfully integrated with Ultralytics (YOLOv8, YOLO11, YOLO26), HuggingFace Transformers (detection,segmentation), RT-DETR, TorchVision, MMDetection, Detectron2, YOLOv5, YOLOE, YOLO-World, and Roboflow RF-DETR models.

  * **Getting Started**

* * *

Install `sahi` with pip and get up and running in minutes.

[ Quickstart](https://obss.github.io/sahi/quick-start/)

  * **How It Works**

* * *

Understand the slicing algorithm, when to use it, and how to tune it.

[ Sliced Inference](https://obss.github.io/sahi/guides/sliced-inference/)

  * **Model Integrations**

* * *

Use SAHI with Ultralytics, HuggingFace, MMDetection, TorchVision, and more.

[ All models](https://obss.github.io/sahi/guides/models/)

  * **Predict**

* * *

Predict on new images, videos, and streams with SAHI.

[ Learn more](https://obss.github.io/sahi/predict/)

  * **Slicing**

* * *

Learn how to slice large images and datasets for inference.

[ Learn more](https://obss.github.io/sahi/slicing/)

  * **COCO Utilities**

* * *

Work with COCO format datasets, including creation, splitting, and filtering.

[ Learn more](https://obss.github.io/sahi/coco/)

  * **CLI Commands**

* * *

Use SAHI from the command-line for prediction and dataset operations.

[ Learn more](https://obss.github.io/sahi/cli/)

  * **FiftyOne**

* * *

Interactively explore and compare detection results.

[ Learn more](https://obss.github.io/sahi/fiftyone/)

  * **Notebooks**

* * *

Hands-on Colab notebooks for every supported framework.

[ Browse notebooks](https://obss.github.io/sahi/notebooks/)



