---
source_url: https://docs.lightly.ai/train/stable/changelog.html
---

# ChangelogÂ¶

All notable changes to Lightly**Train** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.15.1] - 2026-05-28Â¶

### AddedÂ¶

  * Add image logging for all fine-tuning tasks. Sample predictions are saved locally and forwarded to configured loggers (TensorBoard, Weights & Biases, MLflow).

  * Add `predict_batch` for batched inference across all fine-tuning tasks.

  * Configurable `patch_size` for DINOv3 LT-DETR models.

  * Added decoder and losses from [D-FINE](https://github.com/Peterande/D-FINE).

  * Add support for choosing LR scheduler for LTDETR object detection. You can specify the scheduler with `model_args.scheduler_name`, choosing either `linear` or `flat-cosine`.

  * ONNX export for models that support dynamic batch sizes can now specify if the batch size should be dynamic with the `dynamic_batch_size` keyword argument.

  * Explicit support for all YOLO26 variants.




### ChangedÂ¶

### DeprecatedÂ¶

### RemovedÂ¶

### FixedÂ¶

  * Fix PicoDet fine-tuning with mismatched `num_classes`.

  * Fix DINOv3 LT-DETR patch size precedence so `model_args.patch_size` overrides the backbone default.




### SecurityÂ¶

## [0.15.0] - 2026-04-16Â¶

**New Distillation Method and Custom Teacher Models:** We release the new [Distillationv3](https://docs.lightly.ai/train/stable/pretrain_distill/methods/distillation.html) method that achieves better generalization across fine-tuning tasks and works better with DINOv3 teacher models. The new method also supports using [custom teacher models](https://docs.lightly.ai/train/stable/pretrain_distill/methods/distillation.html#methods-distillation-custom-models)

### AddedÂ¶

  * Add distillationv3 method for dense as well as global feature distillation.

  * Add support for custom teacher models with distillationv3.

  * Add support for the new [EUPE models from Meta](https://github.com/facebookresearch/EUPE) for all distillation, pretraining, and fine-tuning tasks. For example, use `dinov3/vits16-eupe` instead of `dinov3/vits16` to load the EUPE pretrained ViT-S/16 model. See the [documentation](https://docs.lightly.ai/train/stable/pretrain_distill/models/dinov3.html#supported-models) for all supported models.

  * Add `Mosaic` augmentation for LTDETR object detection training.

  * Add `CopyBlend` augmentation for LTDETR object detection training.

  * Add `MixUp` augmentation for LTDETR object detection training.

  * Add logging of completed `epoch`s to the console and the loggers.

  * Add support for COCO object detection dataset format.

  * Add support for COCO instance segmentation dataset format.

  * Semantic segmentation now allows one to specify classes from a JSON file.




### ChangedÂ¶

  * Default distillation method is now v3 (previously v2), with a DINOv3 teacher instead of a DINOv2 teacher. Previous default still available with `method="distillationv2"`.

  * Make `ScaleJitter` in LTDETR step-aware. Now you can stop the augmentation by adding a `step_stop` args like the following `transform_args={"scale_jitter": {"step_stop": 10000}}`




### DeprecatedÂ¶

### RemovedÂ¶

  * Remove `StopPolicy` and use `ActivationPolicy` instead for more fine-grained control over the step-aware augmentations.




### FixedÂ¶

### SecurityÂ¶

## [0.14.3] - 2026-03-26Â¶

### AddedÂ¶

  * Add support for DINOv2 [panoptic segmentation](https://docs.lightly.ai/train/stable/panoptic_segmentation.html) inference and fine-tuning.

  * Add support for `metric_args` in all fine-tuning commands to allow configuring the metrics used for validation and best model checkpointing. See the [documentation](https://docs.lightly.ai/train/stable/settings/train_settings.html#metric-args) for details.

  * Add option to freeze the backbone for all EoMT models during training with the `model_args={"backbone_freeze": True}` argument.

  * Add `YOLOOrientedObjectDetectionDataset` for loading YOLO oriented object detection datasets with (cx, cy, w, h, angle) bounding boxes.




### ChangedÂ¶

  * PicoDet switched to O2O NMS-free inference/export, updated L preset to `picodet/l-640`, and improved ONNX/TensorRT export robustness.




### RemovedÂ¶

  * It is no longer possible to set `seed=None`. Instead, an integer seed must be provided for reproducibility. This fixes a bug where recent PyTorch Lightning versions (>=2.2) no longer generate random seeds when `seed=None` is set.

  * LTDETR no longer supports the `detector_weight_decay` and `backbone_weight_decay` arguments. Instead use the general `weight_decay` argument.




### FixedÂ¶

  * Fix incorrect model name format in `export_model()` log example for DINOv2 and DINOv3 packages. The example now shows the correct format (without prefix) that works with `get_model()`.

  * Fix the wrong config of ScaleJitter sizes of LT-DETR.

  * Fix a bug when loading DINOv3 LTDETR checkpoints that were not pretrained on COCO which resulted in backbone weights not being loaded.




### SecurityÂ¶

## [0.14.2] - 2026-02-24Â¶

**New Classification Support:** You can now train image classification models with LightlyTrain! See the [classification documentation](https://docs.lightly.ai/train/stable/image_classification.html) for more information.

### AddedÂ¶

  * Add [classification support](https://docs.lightly.ai/train/stable/image_classification.html)

  * Add support for frozen backbone training in LTDETR and Picodet object detection models. Set `model_args={"backbone_freeze": True}` in `train_object_detection` to freeze the backbone and reduce VRAM usage.

  * Add LTDETR support for DINOv3 ViT-B/L and DINOv2 ViT-L/B/G models. Pretrained weights are not yet available for these models.

  * Add support for fine-tuning DINOv2 models for instance segmentation with the `train_instance_segmentation` command. See the [instance segmentation documentation](https://docs.lightly.ai/train/stable/instance_segmentation.html#model) for more information.




### FixedÂ¶

  * Filter invalid bounding boxes in instance segmentation

  * Fix incorrect logging of training times.




## [0.14.1] - 2026-02-09Â¶

### AddedÂ¶

  * Add Python 3.13 support.

  * Add random rotation transforms for all fine-tuning tasks.

  * Add DistillationV3 preview tailored for ViT models.

  * Skip weight decay for bias/norm/token/etc. layers.

  * Add automatic fine-tuning learning rate scaling based on batch size for all tasks.

  * Log fine-tuning training time breakdown.




### ChangedÂ¶

  * Missing object detection and instance segmentation label files are now treated as images without objects instead of being skipped. This can be configured by setting the `skip_if_label_file_missing` flag in the `data` argument of the `train_object_detection` and `train_instance_segmentation` functions respectively.

  * DINO now updates the teacher temperature and last layer freezing based on the number of training steps instead of epochs.




### FixedÂ¶

  * Fix missing libxcb1 dependency in Dockerfile causing cv2 import errors.

  * Fix issue when fine-tuning panoptic segmentation models with a different number of classes than the pretrained model.




## [0.14.0] - 2026-01-19Â¶

**New PicoDet Models:** We release a preview of PicoDet object detection models for [low-power embedded devices](https://docs.lightly.ai/train/stable/object_detection.html#benchmark-results)!

**New Tiny Models:** We release tiny DINOv3 based models for [instance segmentation](https://docs.lightly.ai/train/stable/instance_segmentation.html#benchmark-results), [panoptic segmentation](https://docs.lightly.ai/train/stable/panoptic_segmentation.html#benchmark-results), and [semantic segmentation](https://docs.lightly.ai/train/stable/semantic_segmentation.html#benchmark-results)!

**New ONNX and TensorRT FP16 Export:** You can now export all supported models to ONNX and TensorRT in FP16 precision for faster inference! [Object detection](https://docs.lightly.ai/train/stable/object_detection.html#exporting-a-checkpoint-to-onnx), [instance segmentation](https://docs.lightly.ai/train/stable/instance_segmentation.html#exporting-a-checkpoint-to-onnx), [panoptic segmentation](https://docs.lightly.ai/train/stable/panoptic_segmentation.html#exporting-a-checkpoint-to-onnx), and [semantic segmentation](https://docs.lightly.ai/train/stable/semantic_segmentation.html#exporting-a-checkpoint-to-onnx) are supported!

### AddedÂ¶

  * Add PicoDet family of object detection models for low-power embedded devices.

  * Add tiny semantic segmentation models.

  * Add tiny instance segmentation models.

  * Add tiny panoptic segmentation models.

  * Add FP16/FP32 ONNX and TensorRT export for object detection models.

  * Add FP16/FP32 ONNX and TensorRT export for instance segmentation models.

  * Add FP16/FP32 ONNX and TensorRT export for panoptic segmentation models.

  * Add FP16/FP32 ONNX and TensorRT export for semantic segmentation models.

  * Add example jupyter notebooks for ONNX and TensorRT export.

  * Add Slicing Aided Hyper Inference (SAHI) for object detection to improve small objects recall at inference.

  * Add pretraining support for Ultralytics RT-DETR models.

  * Add support for different patch size in EoMT and semantic segmentation.

  * Add classwise metrics support for object detection models.

  * Add ignore_classes support for object detection models.




### ChangedÂ¶

  * Change default DINOv3 EoMT semantic segmentation image size from 518x518 to 512x512.

  * New checkpoints for the COCO pretrained DINOv3 EoMT semantic segmentation models.




### FixedÂ¶

  * Fix numerical stability issues in LTDETRâs matcher.

  * Fix issue when resuming from interrupted runs with custom model args.




### SecurityÂ¶

## [0.13.2] - 2025-12-29Â¶

### AddedÂ¶

  * Support for pretraining RF-DETR 1.3 models.




### ChangedÂ¶

  * Export only EMA weights for object detection models. This reduces the exported model size by 2x.




### FixedÂ¶

  * Fix pretrained ViT-small panoptic segmentation model checkpoint.




## [0.13.1] - 2025-12-18Â¶

### FixedÂ¶

  * Fix bug in ONNX export for object detection models.




## [0.13.0] - 2025-12-15Â¶

**New DINOv3 Tiny Object Detection Models:** We release tiny DINOv3 models pretrained on COCO for [object detection](https://docs.lightly.ai/train/stable/object_detection.html#coco)!

**New DINOv3 Panoptic Segmentation:** You can now run inference and fine-tune DINOv3 models for [panoptic segmentation](https://docs.lightly.ai/train/stable/panoptic_segmentation.html)!

### AddedÂ¶

  * New COCO pretrained [tiny LTDETR](https://docs.lightly.ai/train/stable/object_detection.html#coco) models `vitt16` and `vitt16plus`.

  * Support for DINOv3 [panoptic segmentation](https://docs.lightly.ai/train/stable/panoptic_segmentation.html) inference and fine-tuning.

  * Quick start guide for [object detection](https://colab.research.google.com/github/lightly-ai/lightly-train/blob/main/examples/notebooks/object_detection.ipynb).

  * Possibility to load backbone weights in LTDETR.

  * [ONNX export](https://docs.lightly.ai/train/stable/object_detection.html#exporting-a-checkpoint-to-onnx) for LTDETR.

  * Add [Weights & Biases logging support](https://docs.lightly.ai/train/stable/object_detection.html#weights-biases) for all fine-tuning tasks.

  * Log best validation metrics at the end of training.




### ChangedÂ¶

  * Rename `lightly_train.train()` to `lightly_train.pretrain()`. The old name is still available as an alias for backward compatibility but will be removed in a future release.

  * Restructured the documentation to better reflect the different workflows supported by LightlyTrain.




### FixedÂ¶

  * Fix bug in `model.predict()` for object detection models.

  * Fix bug in object detection transforms when using images with dtype float32.

  * Fix bug when running pretraining on an MPS device.

  * Fix bug when resuming training with a recent PyTorch version.

  * Fix bug when resuming a crashed run that was initialized from a pretrained COCO model.




## [0.12.4] - 2025-11-26Â¶

### FixedÂ¶

  * Fix bug in `model.predict()` for object detection models.




## [0.12.3] - 2025-11-26Â¶

### AddedÂ¶

  * Add support for specifying data configs in YAML format.




### ChangedÂ¶

  * Improve the layout of the object detection training logs.




### DeprecatedÂ¶

  * Deprecate `reuse_class_head` argument in the `train`/`pretrain` command. The model will now automatically reuse the classification head only when the number of classes in the data config matches that in the checkpoint. Otherwise, the classification head will be re-initialized.




### FixedÂ¶

  * Fix `image_size` not tuple when training from pretrained model.

  * Fix a bug when fine-tuning a model with `resume_interrupted=True`.

  * Fix `num_classes` not updated when loading an object detection checkpoint with different number of classes.




## [0.12.2] - 2025-11-14Â¶

### FixedÂ¶

  * Fix `image_size` not tuple when training from pretrained model.




## [0.12.1] - 2025-11-13Â¶

### AddedÂ¶

  * Add support for DINOv3 [instance segmentation](https://docs.lightly.ai/train/stable/instance_segmentation.html) inference and fine-tuning.

  * Add support for loading [DICOM images](https://docs.lightly.ai/train/stable/data/dicom.html) as input data for training and inference.

  * Add event tracking, disable with `LIGHTLY_TRAIN_EVENTS_DISABLED=1`

  * Add support for fine-tuning object detection models with custom image resolutions.




## [0.12.0] - 2025-11-06Â¶

ð¡ **New DINOv3 Object Detection:** Run inference or fine-tune DINOv3 models for [object detection](https://docs.lightly.ai/train/stable/object_detection.html)! ð¡

### AddedÂ¶

  * Add support for [DINOv3 object detection](https://docs.lightly.ai/train/stable/object_detection.html) model training.

  * Add [semantic segmentation autolabeling](https://docs.lightly.ai/train/stable/predict_autolabel.html) support with `predict_semantic_segmentation`.

  * Add support for DINOv3 ConvNeXt models.

  * Automatically download DINOv3 weights.

  * Add support for passing pretrained model names or checkpoint paths as `model` argument to the model training functions like `train_semantic_segmentation`.




### ChangedÂ¶

  * Widen PyTorch constraint â remove `<2.6` upper bound to allow PyTorch 2.6 and later that is officially supported by PyTorch Lightning 2.5.

  * Rename `load_model_from_checkpoint` to `load_model`. The function now downloads checkpoints that do not exist locally.




### FixedÂ¶

  * Fix issue with loading DINOv3 SAT493m checkpoints.

  * Fixed an issue where dataset cache files were incorrectly saved.




## [0.11.4] - 2025-10-08Â¶

### AddedÂ¶

  * Add support for saving the best semantic segmentation checkpoints and model weights during training.

  * Expose more arguments for the checkpointing callback in pretraining.

  * Add LT-DETR inference support with DINOv2 and DINOv3 ConvNext backbones.




### ChangedÂ¶

  * Change default precision to `bf16-mixed` for pretraining on GPUs that support it.




### FixedÂ¶

  * Fix warning about too few epochs for DINOv2 which occurs with the default epoch calculation.

  * Fix PyTorch bus errors caused by memory-map file write conflicts when launching multiple runs with `.train_semantic_segmentation()`.




## [0.11.3] - 2025-09-25Â¶

### AddedÂ¶

  * Add EoMT semantic segmentation benchmark results and model weights trained on ADE20k, COCO-Stuff, and Cityscapes datasets.

  * Add support for exporting the semantic segmentation model weights to `exported_models/exported_last.pt`.

  * Add support for allow loading semantic segmentation model weights for training.

  * Add `simplify` flag to ONNX `export_task`.

  * Add support for using DINOv3 models as teacher in distillationv1.




### FixedÂ¶

  * Fix a bug in `model.predict()` with `ignore_index`.

  * Speed up listing of filenames in large datasets.




## [0.11.2] - 2025-09-08Â¶

### AddedÂ¶

  * Add support for using multi-channel masks for the inputs in semantic segmentation.

  * Add support for training models on multi-channel images with `transform_args={"num_channels": 4}`.

  * Add support for using custom mask names for the inputs in semantic segmentation.

  * Add `precision` flag to ONNX export task to specify if we export with float16 or float32 precision.




### FixedÂ¶

  * Fix issue where segmentation fine-tuning could fail when encountering masks containing only unknown classes.

  * Fix issue with mmap cache when multiple runs use the same dataset on the same machine.

  * Speed up logging of datasets with many files.




## [0.11.1] - 2025-08-28Â¶

### AddedÂ¶

  * Add support for DINOv2 linear semantic segmentation models. You can train them with `model="dinov2/vits14-linear"` in the `train_semantic_segmentation` command. Those models are trained with a linear head on top of a frozen backbone and are useful to evaluate the quality of pretrained DINOv2 models.

  * Make fine-tune transform arguments configurable in the `train_semantic_segmentation` command. You can now use the `transform_args` argument like this
        
        transform_args={
          "image_size": (448, 448), # (height, width)
          "normalize": {"mean": (0.0, 0.0, 0.0), "std": (0.5, 0.5, 0.5)}, # (r, g, b) channels
        }
        

to customize the image augmentations used during training and validation. See the [semantic segmentation documentation](https://docs.lightly.ai/train/stable/semantic_segmentation.html#default-image-transform-arguments) for more information.

  * Add support for the channel drop transform in the `train_semantic_segmentation` command.

  * Add support for mapping multiple classes into a single class for semantic segmentation datasets. You can now use a dictionary in the `classes` entry of the `data` argument in the `train_semantic_segmentation` command like this:
        
        data={
          "classes": {
            0: {"name": "background", "values": [0, 255]}, # Map classes 0 and 255 to class 0
            1: {"name": "class 1", "values": [1]},
            2: "class 2",  # Still supported. Equivalent to {"name": "class 2", "values": [2]}
          },
        }
        




### FixedÂ¶

  * Models loaded with `load_model_from_checkpoint` are now automatically moved to the correct device.

  * Loading EoMT models with `load_model_from_checkpoint` no longer raises a missing key error.

  * Fix MLFlow logging on AzureML.




## [0.11.0] - 2025-08-15Â¶

ð **New DINOv3 Support:** Pretrain your own model with [distillation](https://docs.lightly.ai/train/stable/pretrain_distill/methods/distillation.html#methods-distillation) from DINOv3 weights. Or fine-tune our SOTA [EoMT semantic segmentation model](https://docs.lightly.ai/train/stable/semantic_segmentation.html#semantic-segmentation-eomt-dinov3) with a DINOv3 backbone! ð

### AddedÂ¶

  * Distillation now supports [DINOv3 pretrained weights](https://docs.lightly.ai/train/stable/pretrain_distill/methods/distillation.html#methods-distillation) as teacher.

  * Semantic Segmentation now supports [DINOv3 pretrained weights](https://docs.lightly.ai/train/stable/semantic_segmentation.html#semantic-segmentation-eomt-dinov3) as EoMT backbone.




### ChangedÂ¶

  * LightlyTrain now infers the best number of epochs based on the chosen method, dataset size and batch size.




## [0.10.0] - 2025-08-04Â¶

ð¥ **New: Train state-of-the-art semantic segmentation models** with our new [**DINOv2 semantic segmentation**](https://docs.lightly.ai/train/stable/semantic_segmentation.html) fine-tuning method! ð¥

### AddedÂ¶

  * DINOv2 semantic segmentation fine-tuning with the `train_semantic_segmentation` command. See the [semantic segmentation documentation](https://docs.lightly.ai/train/stable/semantic_segmentation.html) for more information.

  * Support for image resolutions that are not a multiple of the patch size in DINOv2.




### ChangedÂ¶

  * DINOv2 model names to be more consistent with the new naming scheme. The model name scheme changed from `dinov2_vit/vits14_pretrain` to `dinov2/vits14`. The pretrained weights are now always loaded by default, making the `pretrain` suffix redundant.

  * DINOv2 models are now using registers by default which increases model performance. You can continue using models without registers with the `-noreg` suffix: `dinov2/vits14-noreg`.

  * DINOv2 and distillation to work with image resolutions that are not a multiple of the patch size.

  * Temporary files are now stored in the `~/.cache/lightly-train` directory be default. The location can be changed with the `LIGHTLY_TRAIN_CACHE_DIR` environment variable.




### DeprecatedÂ¶

  * The `dinov2_vit/vits14_pretrain` model name is deprecated and will be removed in a future release. Use `dinov2/vits14` instead.




## [0.9.0] - 2025-07-21Â¶

### AddedÂ¶

  * Add an extra `teacher_weights` argument in `method_args` to allow loading pretrained DINOv2 teacher weights for distillation methods.

  * Add support for allowing images with different number of channels in the channel drop transform.

  * Add documentation for the [RT-DETRv2 models](https://docs.lightly.ai/train/stable/pretrain_distill/models/rtdetr.html).

  * Add warning for situations where the number of steps is below the recommendation for DINOv2.




### ChangedÂ¶

  * The callback `DeviceStatsMonitor` is now disabled by default.

  * Replace epoch-based warmup hyperparameters with iteration-based ones.




### RemovedÂ¶

  * Remove a note and specific recommendation for using DINOv2 with pretrained weights in the documentation.




## [0.8.1] - 2025-06-23Â¶

### AddedÂ¶

  * Add `student_freeze_backbone_epochs` option to DINOv2 method to control how many epochs the student backbone is frozen during training. We suggest setting it to 1 when starting from DINOv2 pretrained weights. See the [DINOv2 documentation](https://docs.lightly.ai/train/stable/pretrain_distill/methods/dinov2.html) for more information.

  * Add channel drop transform.

  * Add option to load multi-channel images with `LIGHTLY_TRAIN_IMAGE_MODE="UNCHANGED"`.

  * Add option to reuse memmap dataset file via environment variable: `LIGHTLY_TRAIN_MMAP_REUSE_FILE=True`.




## [0.8.0] - 2025-06-10Â¶

### AddedÂ¶

  * **DINOv2 pretraining is now available** with the `method="dinov2"` argument. The method is in beta and further improvements will be released in the coming weeks. See the [DINOv2 documentation](https://docs.lightly.ai/train/stable/pretrain_distill/methods/dinov2.html) for more information.

  * Support for [Torchvision ShuffleNetV2 models](https://docs.lightly.ai/train/stable/pretrain_distill/models/torchvision.html).

  * [RT-DETR](https://docs.lightly.ai/train/stable/pretrain_distill/models/rtdetr.html) has now an integrated model wrapper.




### ChangedÂ¶

  * The [Ultralytics YOLO tutorial](https://docs.lightly.ai/train/stable/tutorials/yolo/index.html) now highlights better how to use YOLO with LightlyTrain.




### DeprecatedÂ¶

  * The `resume` parameter in the `train` command is deprecated in favor of `resume_interrupted` and will be removed in a future release. The new parameter behaves the same as the old one but is more explicit about its purpose. See [the documentation](https://docs.lightly.ai/train/stable/pretrain_distill/index.html#resume-training) for more information.




## [0.7.0] - 2025-05-26Â¶

### AddedÂ¶

  * Add **Distillation v2** method that achieves higher accuracy and trains up to 3x faster than Distillation v1. The new method is selected as default by LightlyTrain with `method="distillation"`.

  * Add MLflow logger to enable system and model metric logging.

  * Add support for lists of files and folders as input to the `embed` and `train` commands.

  * Add faster dataset initialization (SLURM and Windows).

  * Add configurable periodic model export.

  * Add training precision option `float32_matmul_precision`.

  * Add tutorial: [Train embedding models with LightlyTrain](https://docs.lightly.ai/train/stable/tutorials/embedding/index.html).




### ChangedÂ¶

  * Distillation v1 is now selected with `method="distillationv1"`.

  * All commands (`embed`, `export`, and `train`) now require keyword arguments as input.

  * [Custom models](https://docs.lightly.ai/train/stable/pretrain_distill/models/custom_models.html) now require the `get_model` method to be implemented.

  * Distillation methods now use the teacher model from the [official DINOv2 implementation](https://github.com/facebookresearch/dinov2).

  * The RT-DETR example uses RT-DETRv2, imposing fewer constraints on package versions.




### RemovedÂ¶

  * Dependency on the transformers library.




### FixedÂ¶

  * `num_workers="auto"` now limits the number of workers to a maximum of 8 workers/GPU to avoid overloading systems with many CPU cores.




## [0.6.3] - 2025-04-23Â¶

### AddedÂ¶

  * Transforms and methods are now documented on dedicated pages.

  * Add [version compatibility table](https://docs.lightly.ai/train/stable/installation.html#version-compatibility) to the documentation.




### FixedÂ¶

  * Fix image size mismatch issue when using TIMM models and DINO.




## [0.6.2] - 2025-04-09Â¶

### AddedÂ¶

  * Document [RF-DETR models](https://docs.lightly.ai/train/stable/pretrain_distill/models/rtdetr.html).

  * Add [frequently asked questions](https://docs.lightly.ai/train/stable/faq.html) page.

  * Add [Torchvision classification tutorial](https://docs.lightly.ai/train/stable/tutorials/resnet/index.html).

  * Add [depth estimation tutorial](https://docs.lightly.ai/train/stable/tutorials/depth_estimation/index.html).




### ChangedÂ¶

  * Increase minimum Wandb version to `0.17.2` which contains fixes for `numpy>=2.0` support.

  * Limit PyTorch version to `<2.6`. Weâll remove this limitation once PyTorch Lightning 2.6 is released.

  * Limit Python version to `<3.13`. Weâll remove this limitation once PyTorch supports Python 3.13.




### RemovedÂ¶

  * Remove Albumentations versions `1.4.18-1.4.22` support.




## [0.6.1] - 2025-03-31Â¶

### AddedÂ¶

  * Add support for RFDETR models.

  * Document platform compatibility.




### ChangedÂ¶

  * TensorBoard is now automatically installed and no longer an optional dependency.

  * Update the [Models documentation](https://docs.lightly.ai/train/stable/pretrain_distill/models/index.html).

  * Update the [YOLO tutorial](https://docs.lightly.ai/train/stable/tutorials/yolo/index.html)




### RemovedÂ¶

  * Remove DenseCL from the documentation.




## [0.6.0] - 2025-03-24Â¶

### AddedÂ¶

  * Add support for DINOv2 distillation pretraining with the `"distillation"` method.

  * Add support for [YOLO11 and YOLO12 models](https://docs.lightly.ai/train/stable/pretrain_distill/models/ultralytics.html).

  * Add support for [RT-DETR models](https://docs.lightly.ai/train/stable/pretrain_distill/models/rtdetr.html).

  * Add support for [YOLOv12 models](https://docs.lightly.ai/train/stable/pretrain_distill/models/yolov12.html) by the original authors.

  * The Git info (branch name, commit, uncommited changes) for the LightlyTrain package and the directory from where the code runs are now logged in the `train.log` file.




### ChangedÂ¶

  * The default pretraining method is now `"distillation"`.

  * The default embedding format is now `"torch"`.

  * The log messages in the `train.log` file are now more concise.




### FixedÂ¶

  * Ensures proper usage of the `blur_limit` parameter in the `GaussianBlur` transforms.




## [0.5.0] - 2025-03-04Â¶

### AddedÂ¶

  * Add tutorial on how to use [LightlyTrain with YOLO](https://docs.lightly.ai/train/stable/tutorials/yolo/index.html).

  * Show the [`data_wait` percentage](https://docs.lightly.ai/train/stable/performance/index.html#finding-the-performance-bottleneck) in the progress bar to better monitor performance bottlenecks.

  * Add [auto format](https://docs.lightly.ai/train/stable/pretrain_distill/export.html#format) export with example logging, which automatically determines the best export option for your model based on the [used model library](https://docs.lightly.ai/train/stable/pretrain_distill/models/index.html#supported-libraries).

  * Add support for configuring the random rotation transform via `transform_args.random_rotation`.

  * Add support for configuring the color jitter transform via `transform_args.color_jitter`.

  * When using the DINO method and configuring the transforms: Removes `local_view_size`, `local_view_resize` and `n_local_views` from `DINOTransformArgs` in favor of `local_view.view_size`, `local_view.random_resize` and `local_view.num_views`. When using the CLI, replace `transform_args.local_view_size` with `transform_args.local_view.view_size`, â¦ respectively.

  * Allow specifying the precision when using the `embed` command. The loaded checkpoint will be casted to that precision if necessary.




### ChangedÂ¶

  * Increase default DenseCL SGD learning rate to 0.1.

  * Dataset initialization is now faster when using multiple GPUs.

  * Models are now automatically exported at the end of a training.

  * Update the docker image to PyTorch 2.5.1, CUDA 11.8, and cuDNN 9.

  * Switched from using PIL+torchvision to albumentations for the image transformations. This gives a performance boost and allows for more advanced augmentations.

  * The metrics `batch_time` and `data_time` are grouped under `profiling` in the logs.




### FixedÂ¶

  * Fix Ultralytics model export for Ultralytics v8.1 and v8.2

  * Fix that the export command may fail when called in the same script as a train command using DDP.

  * Fix the logging of the `train_loss` to report the batch_size correctly.




## [0.4.0] - 2024-12-05Â¶

### AddedÂ¶

  * Log system information during training

  * Add [Performance Tuning guide](https://docs.lightly.ai/train/stable/performance/index.html) with documentation for [multi-GPU](https://docs.lightly.ai/train/stable/performance/multi_gpu.html) and [multi-node](https://docs.lightly.ai/train/stable/performance/multi_node.html) training

  * Add [Pillow-SIMD support](https://docs.lightly.ai/train/stable/performance/index.html#dataloader-bottleneck-cpu-bound) for faster data processing

    * The docker image now has Pillow-SIMD installed by default

  * Add [`ultralytics`](https://docs.lightly.ai/train/stable/pretrain_distill/export.html#format) export format

  * Add support for DINO weight decay schedule

  * Add support for SGD optimizer with `optim="sgd"`

  * Report final `accelerator`, `num_devices`, and `strategy` in the resolved config

  * Add [Changelog](https://docs.lightly.ai/train/stable/changelog.html) to the documentation




### ChangedÂ¶

  * Various improvements for the DenseCL method

    * Increase default memory bank size

    * Update local loss calculation

  * Custom models have a [new interface](https://docs.lightly.ai/train/stable/pretrain_distill/models/custom_models.html#custom-models)

  * The number of warmup epochs is now set to 10% of the training epochs for runs with less than 100 epochs

  * Update default optimizer settings

    * SGD is now the default optimizer

    * Improve default learning rate and weight decay values

  * Improve automatic `num_workers` calculation

  * The SPPF layer of Ultralytics YOLO models is no longer trained




### RemovedÂ¶

  * Remove DenseCLDINO method

  * Remove DINO `teacher_freeze_last_layer_epochs` argument




## [0.3.2] - 2024-11-06Â¶

### AddedÂ¶

  * Log data loading and forward/backward pass time as `data_time` and `batch_time`

  * Batch size is now more uniformly handled




### ChangedÂ¶

  * The custom model `feature_dim` property is now a method

  * Replace FeatureExtractor base class by the set of Protocols




### FixedÂ¶

  * Datasets support symlinks again




## [0.3.1] - 2024-10-29Â¶

### AddedÂ¶

  * The documentation is now available at https://docs.lightly.ai/train

  * Support loading checkpoint weights with the `checkpoint` argument

  * Log resolved training config to tensorboard and WandB




### FixedÂ¶

  * Support single-channel images by converting them to RGB

  * Log config instead of locals

  * Skip pooling in DenseCLDino




## [0.3.0] - 2024-10-22Â¶

### AddedÂ¶

  * Add Ultralytics model support

  * Add SuperGradients PP-LiteSeg model support

  * Save normalization transform arguments in checkpoints and automatically use them in the embed command

  * Better argument validation

  * Automatically configure `num_workers` based on available CPU cores

  * Add faster and more memory efficient image dataset

  * Log more image augmentations

  * Log resolved config for CallbackArgs, LoggerArgs, MethodArgs, MethodTransformArgs, and OptimizerArgs



  *[*]: Keyword-only parameters separator (PEP 3102)
