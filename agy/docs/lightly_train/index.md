---
source_url: https://docs.lightly.ai/train/stable/
---

# LightlyTrain DocumentationÂ¶

[ ![LightlyTrain Banner](https://docs.lightly.ai/train/stable/_static/lightlyBannerMobile.svg) ](https://www.lightly.ai/lightly-train) [ ![LightlyTrain Banner](https://docs.lightly.ai/train/stable/_static/lightlyBanner.svg) ](https://www.lightly.ai/lightly-train)

![_images/lightly_train_light.svg](_images/lightly_train_light.svg) ![_images/lightly_train_dark.svg](_images/lightly_train_dark.svg)

[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lightly-ai/lightly-train/blob/main/examples/notebooks/object_detection.ipynb) [![Python](https://img.shields.io/badge/Python-3.8%7C3.9%7C3.10%7C3.11%7C3.12%7C3.13-blue.svg)](https://docs.lightly.ai/train/stable/installation.html) [![OS](https://img.shields.io/badge/OS-Linux%7CMacOS%7CWindows-blue.svg)](https://docs.lightly.ai/train/stable/installation.html) [![Docker](https://img.shields.io/badge/Docker-blue?logo=docker&logoColor=fff)](https://docs.lightly.ai/train/stable/docker.html#) [![Documentation](https://img.shields.io/badge/Documentation-blue)](https://docs.lightly.ai/train/stable/) [![Discord](https://img.shields.io/discord/752876370337726585?logo=discord&logoColor=white&label=discord&color=7289da)](https://discord.gg/xvNJW94)

_Train Better Models, Faster_

LightlyTrain is the leading framework for transforming your data into state-of-the-art computer vision models. It covers the entire model development lifecycle from pretraining DINOv2/v3 vision foundation models on your unlabeled data to fine-tuning transformer and YOLO models on detection and segmentation tasks for edge deployment.

[Contact us](https://www.lightly.ai/contact) to request a license for commercial use.

Also check out [LightlyStudio](https://github.com/lightly-ai/lightly-studio) to easily visualize your annotations and predictions.

## NewsÂ¶

  * [[0.15.0](https://docs.lightly.ai/train/stable/changelog.html#changelog-0-15-0)] - 2026-04-14: ð **Distillationv3:** Better generalizing distillation method that performs equally well across dense and global tasks and across all models, from ViTs to hybrids to CNNs (+support for custom teachers!). ð

  * [[0.14.0](https://docs.lightly.ai/train/stable/changelog.html#changelog-0-14-0)] - 2026-01-19: ð£ **PicoDet, Tiny Models, and ONNX/TensorRT FP16 Support:** PicoDet object detection models for low-power embedded devices! All tasks now support tiny DINOv3 models and ONNX/TensorRT export in FP16 precision for faster inference! ð£

  * [[0.13.0](https://docs.lightly.ai/train/stable/changelog.html#changelog-0-13-0)] - 2025-12-15: ð¥ **New Tiny Object Detection Models:** We release tiny DINOv3 models pretrained on COCO for [object detection](https://docs.lightly.ai/train/stable/object_detection.html#coco)! ð¥

  * [[0.12.0](https://docs.lightly.ai/train/stable/changelog.html#changelog-0-12-0)] - 2025-11-06: ð¡ **New DINOv3 Object Detection:** Run inference or fine-tune DINOv3 models for [object detection](https://docs.lightly.ai/train/stable/object_detection.html)! ð¡

  * [[0.11.0](https://docs.lightly.ai/train/stable/changelog.html#changelog-0-11-0)] - 2025-08-15: ð **New DINOv3 Support:** Pretrain your own model with [distillation](https://docs.lightly.ai/train/stable/pretrain_distill/methods/distillation.html#methods-distillation) from DINOv3 weights. Or fine-tune our SOTA [EoMT semantic segmentation model](https://docs.lightly.ai/train/stable/semantic_segmentation.html#semantic-segmentation-eomt-dinov3) with a DINOv3 backbone! ð

  * [[0.10.0](https://docs.lightly.ai/train/stable/changelog.html#changelog-0-10-0)] - 2025-08-04: ð¥ **Train state-of-the-art semantic segmentation models** with our new [**DINOv2 semantic segmentation**](https://docs.lightly.ai/train/stable/semantic_segmentation.html) fine-tuning method! ð¥

  * [[0.9.0](https://docs.lightly.ai/train/stable/changelog.html#changelog-0-9-0)] - 2025-07-21: [**DINOv2 pretraining**](https://docs.lightly.ai/train/stable/pretrain_distill/methods/dinov2.html) is now officially available!




## WorkflowsÂ¶

Object Detection

![](_static/images/tasks/object_detection.png)  
Train LTDETR detection models with DINOv2 or DINOv3 backbones.  


[object_detection.html](object_detection.md)

Instance Segmentation

![](_static/images/tasks/instance_segmentation.png)  
Train EoMT segmentation models with DINOv3 backbones.  


[instance_segmentation.html](instance_segmentation.md)

Panoptic Segmentation

![](_static/images/tasks/panoptic_segmentation.png)  
Train EoMT segmentation models with DINOv2 or DINOv3 backbones.  


[panoptic_segmentation.html](panoptic_segmentation.md)

Semantic Segmentation

![](_static/images/tasks/semantic_segmentation.png)  
Train EoMT segmentation models with DINOv2 or DINOv3 backbones.  


[semantic_segmentation.html](semantic_segmentation.md)

Image Classification

![](_static/images/tasks/image_classification.jpg)  
Train image classification models with any backbone.  


[image_classification.html](image_classification.md)

Distillation

![](_static/images/tasks/distillation.png)  
Distill knowledge from DINOv2 or DINOv3 into any model architecture.  


[pretrain_distill/methods/distillation.html](pretrain_distill__methods__distillation.md)

Pretraining

![](_static/images/tasks/pretraining.png)  
Pretrain DINOv2 foundation models on your domain data.  


[pretrain_distill/methods/dinov2.html](pretrain_distill__methods__dinov2.md)

Autolabeling

![](_static/images/tasks/autolabeling.png)  
Generate high-quality pseudo labels for detection and segmentation tasks.  


[predict_autolabel.html](predict_autolabel.md)

## How It Works [![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lightly-ai/lightly-train/blob/main/examples/notebooks/object_detection.ipynb)Â¶

Install Lightly**Train** on Python 3.8+ for Windows, Linux or MacOS.
    
    
    pip install lightly-train
    

Then train an object detection model with:
    
    
    import lightly_train
    
    if __name__ == "__main__":
        lightly_train.train_object_detection(
            out="out/my_experiment",
            model="dinov3/vitt16-ltdetr-coco",
            data={
                # ... Data configuration
            }
          )
    

And run inference like this:
    
    
    import lightly_train
    
    # Load the model from the best checkpoint
    model = lightly_train.load_model("out/my_experiment/exported_models/exported_best.pt")
    # Or load one of the models hosted by LightlyTrain
    model = lightly_train.load_model("dinov3/vitt16-ltdetr-coco")
    results = model.predict("image.jpg")
    

See the full [quick start guide](quick_start_object_detection.md#quick-start-object-detection) for more details.

## FeaturesÂ¶

  * Python, Command Line, and [Docker](https://docs.lightly.ai/train/stable/docker.html) support

  * Built for [high performance](https://docs.lightly.ai/train/stable/performance/index.html) including [multi-GPU](https://docs.lightly.ai/train/stable/performance/multi_gpu.html) and [multi-node](https://docs.lightly.ai/train/stable/performance/multi_node.html) support

  * [Monitor training progress](https://docs.lightly.ai/train/stable/pretrain_distill.html#logging) with MLflow, TensorBoard, Weights & Biases, and more

  * Runs fully on-premises with no API authentication

  * Export models in their native format for fine-tuning or inference

  * Export models in ONNX or TensorRT format for edge deployment




## ModelsÂ¶

LightlyTrain supports the following model and workflow combinations.

### Fine-tuningÂ¶

Model | Object  
Detection | Instance  
Segmentation | Panoptic  
Segmentation | Semantic  
Segmentation | Image  
Classification  
---|---|---|---|---|---  
DINOv3 | â [ð](https://docs.lightly.ai/train/stable/object_detection.html) | â [ð](https://docs.lightly.ai/train/stable/instance_segmentation.html) | â [ð](https://docs.lightly.ai/train/stable/panoptic_segmentation.html) | â [ð](https://docs.lightly.ai/train/stable/semantic_segmentation.html#use-eomt-with-dinov3) | â [ð](https://docs.lightly.ai/train/stable/image_classification.html)  
DINOv2 | â [ð](https://docs.lightly.ai/train/stable/object_detection.html) | â [ð](https://docs.lightly.ai/train/stable/instance_segmentation.html) | â [ð](https://docs.lightly.ai/train/stable/panoptic_segmentation.html) | â [ð](https://docs.lightly.ai/train/stable/semantic_segmentation.html) | â [ð](https://docs.lightly.ai/train/stable/image_classification.html)  
Any |  |  |  |  | â [ð](https://docs.lightly.ai/train/stable/image_classification.html)  
  
### Distillation & PretrainingÂ¶

Model | Distillation | Pretraining  
---|---|---  
DINOv3 | â [ð](https://docs.lightly.ai/train/stable/pretrain_distill/methods/distillation.html#distill-from-dinov3) |   
DINOv2 | â [ð](https://docs.lightly.ai/train/stable/pretrain_distill/methods/distillation.html) | â [ð](https://docs.lightly.ai/train/stable/pretrain_distill/methods/dinov2.html)  
Torchvision ResNet, ConvNext, ShuffleNetV2 | â [ð](https://docs.lightly.ai/train/stable/pretrain_distill/models/torchvision.html) | â [ð](https://docs.lightly.ai/train/stable/pretrain_distill/models/torchvision.html)  
TIMM models | â [ð](https://docs.lightly.ai/train/stable/pretrain_distill/models/timm.html) | â [ð](https://docs.lightly.ai/train/stable/pretrain_distill/models/timm.html)  
Ultralytics YOLOv5âYOLO12, RT-DETR | â [ð](https://docs.lightly.ai/train/stable/pretrain_distill/models/ultralytics.html) | â [ð](https://docs.lightly.ai/train/stable/pretrain_distill/models/ultralytics.html)  
RT-DETR, RT-DETRv2 | â [ð](https://docs.lightly.ai/train/stable/pretrain_distill/models/rtdetr.html) | â [ð](https://docs.lightly.ai/train/stable/pretrain_distill/models/rtdetr.html)  
RF-DETR | â [ð](https://docs.lightly.ai/train/stable/pretrain_distill/models/rfdetr.html) | â [ð](https://docs.lightly.ai/train/stable/pretrain_distill/models/rfdetr.html)  
YOLOv12 | â [ð](https://docs.lightly.ai/train/stable/pretrain_distill/models/yolov12.html) | â [ð](https://docs.lightly.ai/train/stable/pretrain_distill/models/yolov12.html)  
Custom PyTorch Model | â [ð](https://docs.lightly.ai/train/stable/pretrain_distill/models/custom_models.html) | â [ð](https://docs.lightly.ai/train/stable/pretrain_distill/models/custom_models.html)  
  
[Contact us](https://www.lightly.ai/contact) if you need support for additional models.

## Usage EventsÂ¶

LightlyTrain collects anonymous usage events to help us improve the product. We only track training method, model architecture, and system information (OS, GPU, CI, Container). To opt-out, set the environment variable: `export LIGHTLY_TRAIN_EVENTS_DISABLED=1`

## LicenseÂ¶

Lightly**Train** offers flexible licensing options to suit your specific needs:

  * **AGPL-3.0 License** : Perfect for open-source projects, academic research, and community contributions. Share your innovations with the world while benefiting from community improvements.

  * **Commercial License** : Ideal for businesses and organizations that need proprietary development freedom. Enjoy all the benefits of LightlyTrain while keeping your code and models private. Includes model training and runtime license.

  * **Free Community License** : Available for students, researchers, startups in early stages, or anyone exploring or experimenting with LightlyTrain. Empower the next generation of innovators with full access to the world of pretraining.




Weâre committed to supporting both open-source and commercial users. [Contact us](https://www.lightly.ai/contact) to discuss the best licensing option for your project!

## ContactÂ¶

[![Website](https://img.shields.io/badge/Website-lightly.ai-blue?style=for-the-badge&logo=safari&logoColor=white)](https://www.lightly.ai/lightly-train)   
[![Discord](https://img.shields.io/discord/752876370337726585?style=for-the-badge&logo=discord&logoColor=white&label=discord&color=7289da)](https://discord.gg/xvNJW94)   
[![GitHub](https://img.shields.io/badge/GitHub-lightly--ai-black?style=for-the-badge&logo=github&logoColor=white)](https://github.com/lightly-ai/lightly-train)   
[![X](https://img.shields.io/badge/X-lightlyai-black?style=for-the-badge&logo=x&logoColor=white)](https://x.com/lightlyai)   
[![LinkedIn](https://img.shields.io/badge/LinkedIn-lightly--tech-blue?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/company/lightly-tech)
  *[*]: Keyword-only parameters separator (PEP 3102)
