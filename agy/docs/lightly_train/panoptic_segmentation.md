---
source_url: https://docs.lightly.ai/train/stable/panoptic_segmentation.html
---

# Panoptic SegmentationÂ¶

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lightly-ai/lightly-train/blob/main/examples/notebooks/eomt_panoptic_segmentation.ipynb)

Note

ð¥ LightlyTrain now supports training **DINOv2** and **DINOv3** -based panoptic segmentation models with the [EoMT architecture](https://arxiv.org/abs/2503.19108) by Kerssies et al.!

## Benchmark ResultsÂ¶

Below we provide the models and report the validation panoptic quality (PQ) and inference latency of different DINOv3 models fine-tuned on COCO with LightlyTrain. You can check here how to use these models for further fine-tuning.

You can also explore running inference and training these models using our Colab notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lightly-ai/lightly-train/blob/main/examples/notebooks/eomt_panoptic_segmentation.ipynb)

### COCOÂ¶

Implementation | Model | Val PQ | Avg. Latency (ms) | Params (M) | Input Size  
---|---|---|---|---|---  
LightlyTrain | dinov3/vitt16-eomt-panoptic-coco | 38.0 | 13.5 | 6.0 | 640Ã640  
LightlyTrain | dinov3/vitt16plus-eomt-panoptic-coco | 41.4 | 14.1 | 7.7 | 640Ã640  
LightlyTrain | dinov3/vits16-eomt-panoptic-coco | 46.8 | 21.2 | 23.4 | 640Ã640  
LightlyTrain | dinov3/vitb16-eomt-panoptic-coco | 53.2 | 39.4 | 92.5 | 640Ã640  
LightlyTrain | dinov3/vitl16-eomt-panoptic-coco | 57.0 | 80.1 | 315.1 | 640Ã640  
LightlyTrain | dinov3/vitl16-eomt-panoptic-coco-1280 | **59.0** | 500.1 | 315.1 | 1280Ã1280  
EoMT (CVPR 2025 paper, current SOTA) | dinov3/vitl16-eomt-panoptic-coco-1280 | 58.9 | - | 315.1 | 1280Ã1280  
  
Training follows the protocol in the original [EoMT paper](https://arxiv.org/abs/2503.19108). Tiny models are trained for 360K steps (48 epochs), small and base models for 180K steps (24 epochs) and large models for 90K steps (12 epochs) on the COCO dataset with batch size `16` and learning rate `2e-4`. The average latency values were measured with model compilation using `torch.compile` on a single NVIDIA T4 GPU with FP16 precision.

## Train a Panoptic Segmentation ModelÂ¶

Training a panoptic segmentation model with LightlyTrain is straightforward and only requires a few lines of code. See data for more details on how to prepare your dataset.
    
    
    import lightly_train
    
    if __name__ == "__main__":
        lightly_train.train_panoptic_segmentation(
            out="out/my_experiment",
            model="dinov3/vitl16-eomt-panoptic-coco", 
            data={
                "train": {
                    "images": "images/train",   # Path to train images
                    "masks": "annotations/train", # Path to train mask images
                    "annotations": "annotations/train.json", # Path to train COCO-style annotations
                },
                "val": {
                    "images": "images/val", # Path to val images
                    "masks": "annotations/val", # Path to val mask images
                    "annotations": "annotations/val.json", # Path to val COCO-style annotations
                },
            },
        )
    

During training, the best and last model weights are exported to `out/my_experiment/exported_models/`, unless disabled in `save_checkpoint_args`:

  * best (highest validation PQ): `exported_best.pt`

  * last: `exported_last.pt`




You can use these weights to continue fine-tuning on another dataset by loading the weights with `model="<checkpoint path>"`:
    
    
    import lightly_train
    
    if __name__ == "__main__":
        lightly_train.train_panoptic_segmentation(
            out="out/my_experiment",
            model="out/my_experiment/exported_models/exported_best.pt",  # Continue training from the best model
            data={...},
        )
    

### Load the Trained Model from Checkpoint and PredictÂ¶

After the training completes, you can load the best model checkpoints for inference like this:
    
    
    import lightly_train
    
    model = lightly_train.load_model("out/my_experiment/exported_models/exported_best.pt")
    results = model.predict("image.jpg")
    results["masks"]    # Masks with (class_label, segment_id) for each pixel, tensor of
                        # shape (height, width, 2). Height and width correspond to the
                        # original image size.
    results["segment_ids"]    # Segment ids, tensor of shape (num_segments,).
    results["scores"]   # Confidence scores, tensor of shape (num_segments,)
    

Or use one of the pretrained models directly from LightlyTrain:
    
    
    import lightly_train
    
    model = lightly_train.load_model("dinov3/vitl16-eomt-panoptic-coco")
    results = model.predict("image.jpg")
    

### Visualize the PredictionsÂ¶

You can visualize the predicted masks like this:
    
    
    import matplotlib.pyplot as plt
    from torchvision.io import read_image
    from torchvision.utils import draw_segmentation_masks
    
    image = read_image("image.jpg")
    masks = results["masks"]
    segment_ids = results["segment_ids"]
    masks = torch.stack([masks[..., 1] == -1] + [masks[..., 1] == segment_id for segment_id in segment_ids])
    colors = [(0, 0, 0)] + [[int(color * 255) for color in plt.cm.tab20c(i / len(segment_ids))[:3]] for i in range(len(segment_ids))]
    image_with_masks = draw_segmentation_masks(image, masks, colors=colors, alpha=1.0)
    plt.imshow(image_with_masks.permute(1, 2, 0))
    

![_images/train.jpg](_images/train.jpg)

## DataÂ¶

Lightly**Train** supports panoptic segmentation datasets in COCO format. Every image must have a corresponding mask image that encodes the segmentation class and segment ID for each pixel. The dataset must also include COCO-style JSON annotation files that define the thing and stuff classes and list the individual segments for each image. See the [COCO Panoptic Segmentation format](https://cocodataset.org/#format-data) for more details.

The following image formats are supported:

  * jpg

  * jpeg

  * png

  * ppm

  * bmp

  * pgm

  * tif

  * tiff

  * webp




Your dataset directory must be organized like this:
    
    
    my_data_dir/
    âââ images
    â   âââ train
    â   â   âââ image1.jpg
    â   â   âââ image2.jpg
    â   â   âââ ...
    â   âââ val
    â       âââ image1.jpg
    â       âââ image2.jpg
    â       âââ ...
    âââ annotations
        âââ train
        â   âââ image1.png
        â   âââ image2.png
        â   âââ ...
        âââ train.json
        âââ val
        â   âââ image1.png
        â   âââ image2.png
        â   âââ ...
        âââ val.json
    

The directories can have any name, as long as the paths are correctly specified in the `data` argument.

See the [Colab notebook](https://colab.research.google.com/github/lightly-ai/lightly-train/blob/main/examples/notebooks/eomt_panoptic_segmentation.ipynb) for an example dataset and how to set up the data for training.

## ModelÂ¶

The `model` argument defines the model used for panoptic segmentation training. The following models are available:

### DINOv3 ModelsÂ¶

  * `dinov3/vitt16-eomt-panoptic-coco` (fine-tuned on COCO)

  * `dinov3/vitt16plus-eomt-panoptic-coco` (fine-tuned on COCO)

  * `dinov3/vits16-eomt-panoptic-coco` (fine-tuned on COCO)

  * `dinov3/vitb16-eomt-panoptic-coco` (fine-tuned on COCO)

  * `dinov3/vitl16-eomt-panoptic-coco` (fine-tuned on COCO)

  * `dinov3/vitl16-eomt-panoptic-coco-1280` (fine-tuned on COCO with 1280x1280 input size)

  * `dinov3/vitt16-eomt`

  * `dinov3/vitt16-eupe-eomt` \- [EUPE weights](https://github.com/facebookresearch/EUPE)

  * `dinov3/vitt16plus-eomt`

  * `dinov3/vits16-eomt`

  * `dinov3/vits16-eupe-eomt` \- [EUPE weights](https://github.com/facebookresearch/EUPE)

  * `dinov3/vits16plus-eomt`

  * `dinov3/vitb16-eomt`

  * `dinov3/vitb16-eupe-eomt` \- [EUPE weights](https://github.com/facebookresearch/EUPE)

  * `dinov3/vitl16-eomt`

  * `dinov3/vitl16plus-eomt`

  * `dinov3/vith16plus-eomt`

  * `dinov3/vit7b16-eomt`




Unless noted otherwise, all DINOv3 backbones are initialized from weights [pretrained by Meta](https://github.com/facebookresearch/dinov3/tree/main?tab=readme-ov-file#pretrained-models). The non-EUPE models with `vitt16` and `vitt16plus` backbones use Lightly-pretrained DINOv3 backbone weights instead. Models marked as EUPE use [EUPE weights](https://github.com/facebookresearch/EUPE). DINOv3 models are under the [DINOv3 license](https://github.com/facebookresearch/dinov3?tab=License-1-ov-file). EUPE models are under the [FAIR Noncommercial Research License](https://github.com/facebookresearch/EUPE?tab=License-1-ov-file).

### DINOv2 ModelsÂ¶

  * `dinov2/vits14-eomt`

  * `dinov2/vitb14-eomt`

  * `dinov2/vitl14-eomt`

  * `dinov2/vitg14-eomt`




All DINOv2 models are [pretrained by Meta](https://github.com/facebookresearch/dinov2?tab=readme-ov-file#pretrained-models).

## Training SettingsÂ¶

See [Train Settings](settings__train_settings.md#train-settings) on how to configure training settings.

## LoggingÂ¶

See [Logging](settings__train_settings.md#train-settings-logging) on how to configure logging.

## Resume TrainingÂ¶

See [Resume Training](settings__train_settings.md#train-settings-resume-training) on how to resume training.

## Default Image Transform ArgumentsÂ¶

The following are the default image transform arguments. See [Transforms](settings__train_settings.md#train-settings-transforms) on how to customize transform settings.

EoMT Panoptic Segmentation DINOv3 Default Transform Arguments

Train
    
    
    {
        "channel_drop": null,
        "color_jitter": null,
        "image_size": "auto",
        "normalize": "auto",
        "num_channels": "auto",
        "random_crop": {
            "fill": 0,
            "height": "auto",
            "pad_if_needed": true,
            "pad_position": "center",
            "prob": 1.0,
            "width": "auto"
        },
        "random_flip": {
            "horizontal_prob": 0.5,
            "vertical_prob": 0.0
        },
        "random_rotate": null,
        "random_rotate_90": null,
        "scale_jitter": {
            "divisible_by": null,
            "max_scale": 2.0,
            "min_scale": 0.1,
            "num_scales": 20,
            "prob": 1.0,
            "sizes": null,
            "step_stop": null
        },
        "smallest_max_size": null
    }
    

Val
    
    
    {
        "channel_drop": null,
        "color_jitter": null,
        "image_size": null,
        "normalize": "auto",
        "num_channels": "auto",
        "random_crop": null,
        "random_flip": null,
        "random_rotate": null,
        "random_rotate_90": null,
        "scale_jitter": null,
        "smallest_max_size": null
    }
    

EoMT Panoptic Segmentation DINOv2 Default Transform Arguments

Train
    
    
    {
        "channel_drop": null,
        "color_jitter": null,
        "image_size": "auto",
        "normalize": "auto",
        "num_channels": "auto",
        "random_crop": {
            "fill": 0,
            "height": "auto",
            "pad_if_needed": true,
            "pad_position": "center",
            "prob": 1.0,
            "width": "auto"
        },
        "random_flip": {
            "horizontal_prob": 0.5,
            "vertical_prob": 0.0
        },
        "random_rotate": null,
        "random_rotate_90": null,
        "scale_jitter": {
            "divisible_by": null,
            "max_scale": 2.0,
            "min_scale": 0.1,
            "num_scales": 20,
            "prob": 1.0,
            "sizes": null,
            "step_stop": null
        },
        "smallest_max_size": null
    }
    

Val
    
    
    {
        "channel_drop": null,
        "color_jitter": null,
        "image_size": null,
        "normalize": "auto",
        "num_channels": "auto",
        "random_crop": null,
        "random_flip": null,
        "random_rotate": null,
        "random_rotate_90": null,
        "scale_jitter": null,
        "smallest_max_size": null
    }
    

## Exporting a Checkpoint to ONNXÂ¶

[Open Neural Network Exchange (ONNX)](https://en.wikipedia.org/wiki/Open_Neural_Network_Exchange) is a standard format for representing machine learning models in a framework independent manner. In particular, it is useful for deploying our models on edge devices where PyTorch is not available.

### RequirementsÂ¶

Exporting to ONNX requires some additional packages to be installed. Namely

  * [onnx](https://pypi.org/project/onnx/)

  * [onnxruntime](https://pypi.org/project/onnxruntime/) if `verify` is set to `True`.

  * [onnxslim](https://pypi.org/project/onnxslim/) if `simplify` is set to `True`.




You can install them with:
    
    
    pip install "lightly-train[onnx,onnxruntime,onnxslim]"
    

The following example shows how to export a previously trained model to ONNX.
    
    
    import lightly_train
    
    # Instantiate the model from a checkpoint.
    model = lightly_train.load_model("out/my_experiment/exported_models/exported_best.pt")
    
    # Export to ONNX.
    model.export_onnx(
        out="out/my_experiment/exported_models/model.onnx",
        # precision="fp16", # Export model with FP16 weights for smaller size and faster inference.
    )
    

See [`export_onnx()`](python_api__lightly_train.md#lightly_train._task_models.dinov3_eomt_panoptic_segmentation.task_model.DINOv3EoMTPanopticSegmentation.export_onnx "lightly_train._task_models.dinov3_eomt_panoptic_segmentation.task_model.DINOv3EoMTPanopticSegmentation.export_onnx") for all available options when exporting to ONNX.

The following notebook shows how to export a model to ONNX in Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lightly-ai/lightly-train/blob/main/examples/notebooks/panoptic_segmentation_export.ipynb)

## Exporting a Checkpoint to TensorRTÂ¶

TensorRT engines are built from an ONNX representation of the model. The `export_tensorrt` method internally exports the model to ONNX (see the ONNX export section above) before building a [TensorRT](https://developer.nvidia.com/tensorrt) engine for fast GPU inference.

### RequirementsÂ¶

TensorRT is not part of LightlyTrainâs dependencies and must be installed separately. Installation depends on your OS, Python version, GPU, and NVIDIA driver/CUDA setup. See the [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html) for more details.

On CUDA 12.x systems you can often install the Python package via:
    
    
    pip install tensorrt-cu12
    
    
    
    import lightly_train
    
    # Instantiate the model from a checkpoint.
    model = lightly_train.load_model("out/my_experiment/exported_models/exported_best.pt")
    
    # Export to TensorRT from an ONNX file.
    model.export_tensorrt(
        out="out/my_experiment/exported_models/model.trt", # TensorRT engine destination.
        # precision="fp16", # Export model with FP16 weights for smaller size and faster inference.
    )
    

See [`export_tensorrt()`](python_api__lightly_train.md#lightly_train._task_models.dinov3_eomt_panoptic_segmentation.task_model.DINOv3EoMTPanopticSegmentation.export_tensorrt "lightly_train._task_models.dinov3_eomt_panoptic_segmentation.task_model.DINOv3EoMTPanopticSegmentation.export_tensorrt") for all available options when exporting to TensorRT.

You can also learn more about exporting EoMT to TensorRT using our Colab notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lightly-ai/lightly-train/blob/main/examples/notebooks/panoptic_segmentation_export.ipynb)
