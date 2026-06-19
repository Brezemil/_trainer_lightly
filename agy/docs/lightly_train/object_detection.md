---
source_url: https://docs.lightly.ai/train/stable/object_detection.html
---

# Object DetectionÂ¶

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lightly-ai/lightly-train/blob/main/examples/notebooks/object_detection.ipynb)

Note

ð¥ LightlyTrain now supports training **LTDETR** : **DINOv3** \- and **DINOv2** -based object detection models with the super fast RT-DETR detection architecture! Our largest model achieves an mAP50:95 of 60.0 on the COCO validation set!

## Benchmark ResultsÂ¶

Below we provide the model checkpoints and report the validation mAP50:95 and inference latency of different DINOv3 and DINOv2-based models, fine-tuned on the COCO dataset. You can check here for how to use these model checkpoints for further fine-tuning. The average latency values were measured using TensorRT version `10.13.3.9` and on a Nvidia T4 GPU with batch size 1.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lightly-ai/lightly-train/blob/main/examples/notebooks/object_detection.ipynb)

### COCOÂ¶

Implementation | Model | Val mAP50:95 | Latency (ms) | Params (M) | Input Size  
---|---|---|---|---|---  
LightlyTrain | picodet-s-coco | 26.7* | 2.2* | 1.17 | 416Ã416  
LightlyTrain | picodet-l-coco | 32.0* | 2.4* | 3.75 | 416Ã416  
LightlyTrain | dinov3/vitt16-ltdetr-coco | 49.8 | 5.4 | 10.1 | 640Ã640  
LightlyTrain | dinov3/vitt16plus-ltdetr-coco | 52.5 | 7.0 | 18.1 | 640Ã640  
LightlyTrain | dinov3/vits16-ltdetr-coco | 55.4 | 10.5 | 36.4 | 640Ã640  
LightlyTrain | dinov3/convnext-tiny-ltdetr-coco | 54.4 | 13.3 | 61.1 | 640Ã640  
LightlyTrain | dinov3/convnext-small-ltdetr-coco | 56.9 | 17.7 | 82.7 | 640Ã640  
LightlyTrain | dinov3/convnext-base-ltdetr-coco | 58.6 | 24.7 | 121.0 | 640Ã640  
LightlyTrain | dinov3/convnext-large-ltdetr-coco | 60.0 | 42.3 | 230.0 | 640Ã640  
  
*Picodet models are in preview and we report preliminary results.

## Object Detection with LTDETRÂ¶

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lightly-ai/lightly-train/blob/main/examples/notebooks/object_detection.ipynb)

Training an object detection model with LightlyTrain is straightforward and only requires a few lines of code. See data for details on how to prepare your dataset.

### Train an Object Detection ModelÂ¶
    
    
    import lightly_train
    
    if __name__ == "__main__":
        lightly_train.train_object_detection(
            out="out/my_experiment",
            model="dinov3/vitt16-ltdetr-coco",
            data={
                "format": "yolo",
                "path": "my_data_dir",
                "train": "images/train2017",
                "val": "images/val2017",
                "names": {
                    0: "person",
                    1: "bicycle",
                    # ...
                },
                # Optional, classes that are in the dataset but should be ignored during
                # training.
                # "ignore_classes": [0],
                #
                # Optional, skip images without label files. By default, these are included
                # as negative samples.
                # "skip_if_label_file_missing": True,
            }
        )
    

During training, both the

  * best (with highest validation mAP50:95) and

  * last (last validation round as determined by `save_checkpoint_args.save_every_num_steps`)




model weights are exported to `out/my_experiment/exported_models/`, unless disabled in `save_checkpoint_args`. You can use these weights to continue fine-tuning on another task by loading the weights via `model="<checkpoint path>"`:
    
    
    import lightly_train
    
    if __name__ == "__main__":
        lightly_train.train_object_detection(
            out="out/my_experiment",
            model="out/my_experiment/exported_models/exported_best.pt", # Use the best model to continue training
            data={...},
        )
    

## Pretrain and Fine-tune an Object Detection ModelÂ¶

To further improve the performance of your object detection model, you can first pretrain a DINOv2 model on unlabeled data using self-supervised learning and then fine-tune it on your object detection dataset. This is especially useful if your dataset is only partially labeled or if you have access to a large amount of unlabeled data.

The following example shows how to pretrain and fine-tune the model. Check out the page on [DINOv2](pretrain_distill__methods__dinov2.md#methods-dinov2) to learn more about pretraining DINOv2 models on unlabeled data.
    
    
    import lightly_train
    
    if __name__ == "__main__":
        # Pretrain a DINOv2 model.
        lightly_train.pretrain(
            out="out/my_pretrain_experiment",
            data="my_pretrain_data_dir",
            model="dinov2/vits14-noreg",
            method="dinov2",
        )
    
        # Fine-tune the DINOv2 model for object detection.
        lightly_train.train_object_detection(
            out="out/my_experiment",
            model="dinov2/vits14-noreg-ltdetr",
            model_args={
                # Path to your pretrained DINOv2 model.
                "backbone_weights": "out/my_pretrain_experiment/exported_models/exported_best.pt",
            },
            data={
                "format": "yolo",
                "path": "my_data_dir",
                "train": "images/train2012",
                "val": "images/val2012",
                "names": {
                    0: "person",
                    1: "bicycle",
                    # ...
                },
            }
        )
    

### Load the Trained Model from Checkpoint and PredictÂ¶

After the training completes, you can load the best model checkpoints for inference like this:
    
    
    import lightly_train
    
    model = lightly_train.load_model("out/my_experiment/exported_models/exported_best.pt")
    results = model.predict("path/to/image.jpg")
    

Or use one of the models provided by LightlyTrain:
    
    
    import lightly_train
    
    model = lightly_train.load_model("dinov3/vitt16-ltdetr-coco")
    results = model.predict("image.jpg")
    results["labels"]   # Class labels, tensor of shape (num_boxes,)
    results["bboxes"]   # Bounding boxes in (xmin, ymin, xmax, ymax) absolute pixel
                        # coordinates of the original image. Tensor of shape (num_boxes, 4).
    results["scores"]   # Confidence scores, tensor of shape (num_boxes,)
    

### Visualize the ResultÂ¶

After making the predictions with the model weights, you can visualize the predicted bounding boxes like this:
    
    
    import matplotlib.pyplot as plt
    from torchvision import io, utils
    
    import lightly_train
    
    model = lightly_train.load_model("dinov3/vitt16-ltdetr-coco")
    results = model.predict_sahi(image="image.jpg")
    results["labels"]   # Class labels, tensor of shape (num_boxes,)
    results["bboxes"]   # Bounding boxes in (xmin, ymin, xmax, ymax) absolute pixel
                        # coordinates of the original image. Tensor of shape (num_boxes, 4).
    results["scores"]   # Confidence scores, tensor of shape (num_boxes,)
    
    # Visualize predictions.
    image_with_boxes = utils.draw_bounding_boxes(
        image=io.read_image("image.jpg"),
        boxes=results["bboxes"],
        labels=[model.classes[i.item()] for i in results["labels"]],
    )
    
    fig, ax = plt.subplots(figsize=(30, 30))
    ax.imshow(image_with_boxes.permute(1, 2, 0))
    fig.savefig("predictions.png")
    

The predicted boxes are in the absolute (x_min, y_min, x_max, y_max) format, i.e. represent the size of the dimension of the bounding boxes in pixels of the original image.

### Improving Small Objects DetectionÂ¶

Detecting small objects in high-resolution images can be challenging because they may occupy only a few pixels when the image is resized to the modelâs input resolution. To address this, we support Slicing Aided Hyper Inference (SAHI) allowing the model to make predictions from overlapping tiles of the original image at full resolution and then merge the predictions.

Using tiled inference requires no extra setup:
    
    
    import lightly_train
    
    model = lightly_train.load_model("dinov3/vitt16-ltdetr-coco")
    results = model.predict_sahi(image="image.jpg")
    results["labels"]   # Class labels, tensor of shape (num_boxes,)
    results["bboxes"]   # Bounding boxes in (xmin, ymin, xmax, ymax) absolute pixel
                        # coordinates of the original image. Tensor of shape (num_boxes, 4).
    results["scores"]   # Confidence scores, tensor of shape (num_boxes,)
    

You can customize the behavior via the following parameters:

  * `overlap`: Fraction of overlap between neighboring tiles. Higher values increase small-object recall but also increase computation.

  * `threshold`: Minimum confidence score required to keep a predicted box.

  * `nms_iou_threshold`: IoU threshold used for non-maximum suppression when merging predictions coming from different tiles.

  * `global_local_iou_threshold`: Our SAHI-style inference combines predictions from both the _global_ (full-image) view and the _local_ tiles. To avoid duplicate detections, tile predictions are suppressed when they significantly overlap (`iou > global_local_iou_threshold`) with a prediction of the same class coming from the global view.


![_images/street.jpg](_images/street.jpg)

## OutÂ¶

The `out` argument specifies the output directory where all training logs, model exports, and checkpoints are saved. It looks like this after training:
    
    
    out/my_experiment
    âââ checkpoints
    â   âââ last.ckpt                                       # Last checkpoint
    âââ exported_models
    |   âââ exported_last.pt                                # Last model exported (unless disabled)
    |   âââ exported_best.pt                                # Best model exported (unless disabled)
    âââ events.out.tfevents.1721899772.host.1839736.0       # TensorBoard logs
    âââ train.log                                           # Training logs
    

The final model checkpoint is saved to `out/my_experiment/checkpoints/last.ckpt`. The last and best model weights are exported to `out/my_experiment/exported_models/` unless disabled in `save_checkpoint_args`.

Tip

Create a new output directory for each experiment to keep training logs, model exports, and checkpoints organized.

## DataÂ¶

Lightly**Train** supports training object detection models with images and bounding boxes. We support inputs in either the YOLO or COCO object detection formats.

We specify the training data with a `data` dictionary:
    
    
    import lightly_train
    
    lightly_train.train_object_detection(
        ...,
        data={
            "format": ...,           # either "yolo" or "coco"
            "ignore_classes": [...], # optional list of class IDs that should be skipped during training
             # format specific options
        },
    )
    

If you would like to skip specific classes during training, add their IDs to the optional `ignore_classes` list. The trainer omits these classes from loss computation and the exported model does not predict them.

### YOLO formatÂ¶

For the [YOLO](https://labelformat.com/formats/object-detection/yolov5/) format, every image has a corresponding label file with the `.txt` extension. Each line in the label file represents one object and contains the class ID followed by 4 normalized bounding box coordinates (x_center, y_center, width, height). An example annotation file for an image with two objects looks like this:
    
    
    0 0.716797 0.395833 0.216406 0.147222
    1 0.687500 0.379167 0.255208 0.175000
    

Your dataset directory should be organized like this:
    
    
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
    âââ labels
        âââ train
        â   âââ image1.txt
        â   âââ image2.txt
        â   âââ ...
        âââ val
            âââ image1.txt
            âââ image2.txt
            âââ ...
    

Alternatively, the splits can also be at the top level:
    
    
    my_data_dir/
    âââ train
    â   âââ images
    â   â   âââ image1.jpg
    â   â   âââ image2.jpg
    â   â   âââ ...
    â   âââ labels
    â       âââ image1.txt
    â       âââ image2.txt
    â       âââ ...
    âââ val
        âââ images
        â   âââ image1.jpg
        â   âââ image2.jpg
        â   âââ ...
        âââ labels
            âââ image1.txt
            âââ image2.txt
            âââ ...
    

Each class in the dataset must be listed in the `names` dictionary. The keys are the class IDs used inside the YOLO annotations and the values are the human-readable class names. Any class IDs that appear in the label files but are not present in the dictionary are silently ignored.

#### Missing LabelsÂ¶

There are three cases in which an image may not have any corresponding labels:

  1. The label file is missing.

  2. The label file is empty.

  3. The label file only contains annotations for classes that are in `ignore_classes`.




LightlyTrain treats all three cases as ânegativeâ samples and includes the images in training with an empty list of bounding boxes.

If you would like to exclude images without label files from training, you can set the `skip_if_label_file_missing` argument in the `data` configuration. This only excludes images without a label file (case 1) but still includes cases 2 and 3 as negative samples.

#### ExampleÂ¶
    
    
    import lightly_train
    
    lightly_train.train_object_detection(
        ...,
        data={
            "format": "yolo",
            "path": "my_data_dir",
            "train": "images/train",
            "val": "images/val",
            "names": {...},
            "skip_if_label_file_missing": True, # Skip images without label files.
        }
    )
    

### COCO formatÂ¶

For the [COCO](https://labelformat.com/formats/object-detection/coco/) format, every split has a separate annotations JSON file. It specifies which images and classes belong to the split and contains the bounding boxes. The structure of such a file is as follows:
    
    
    {
        "images": [
            {
                "id": 1,
                "file_name": "image1.jpg"
            },
            {
                "id": 2,
                "file_name": "image2.jpg"
            }
        ],
        "categories": [
            {
                "id": 0,
                "name": "cat"
            },
            {
                "id": 1,
                "name": "dog"
            }
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 0,
                "bbox": [10, 20, 100, 80]
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 1,
                "bbox": [150, 30, 200, 120]
            },
            {
                "id": 3,
                "image_id": 2,
                "category_id": 0,
                "bbox": [5, 10, 90, 70]
            }
        ]
    }
    

The `file_name` field can also be an absolute or relative path to an image. One can optionally specify the `images` directory so that the paths are resolved relatively to that directory. If it is omitted, the paths are resolved relatively to the annotations file. Furthermore, the `images` path itself is resolved relatively to the annotations file.

It is good practice to have the same categories for all splits but in order to guarantee consistency, we always take them from the train split.

The bounding boxes `bbox` are specified in absolute coordinates (pixels) as follows:
    
    
    [x, y, width, height]
    

#### Missing LabelsÂ¶

There are two cases in which an image may not have any corresponding labels:

  1. There are no bounding boxes specified for an image in the annotations file.

  2. The annotations file only contains annotations for classes that are in `ignore_classes`.




LightlyTrain treats both cases as ânegativeâ samples and includes the images in training with an empty list of bounding boxes.

If you would like to exclude images without bounding boxes from training, you can set the `skip_if_annotations_missing` argument in the `data` configuration. This only excludes images without bounding boxes (case 1) but still includes case 2 as negative samples.

#### ExampleÂ¶
    
    
    import lightly_train
    
    lightly_train.train_object_detection(
        ...,
        data={
            "format": "coco",
            "train": {"annotations": "train_labels.json", "images": "train_images/"},
            "val": {"annotations": "val_labels.json", "images": "val_images/"},
            "skip_if_annotations_missing": True, # Skip images without bounding boxes
        }
    )
    

If in this particular example we specified `file_name` like this in the train annotation file
    
    
    {
        "id": 1,
        "file_name": "train_images/image1.jpg"
    }
    

we could also omit `images`.

### Image FormatsÂ¶

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

  * dcm (DICOM)




For more details on LightlyTrainâs support for data input, please check the [Data Input](https://docs.lightly.ai/train/stable/data/index.html#data-input) page.

## ModelÂ¶

The `model` argument defines the model used for object detection training. The following models are available:

### PicoDet ModelsÂ¶

  * `picodet-s-coco` (pretrained on COCO)

  * `picodet-l-coco` (pretrained on COCO)




### LTDETR DINOv3 ModelsÂ¶

  * `dinov3/vitt16-ltdetr-coco` (pretrained on COCO)

  * `dinov3/vitt16plus-ltdetr-coco` (pretrained on COCO)

  * `dinov3/vits16-ltdetr-coco` (pretrained on COCO)

  * `dinov3/convnext-tiny-ltdetr-coco` (pretrained on COCO)

  * `dinov3/convnext-small-ltdetr-coco` (pretrained on COCO)

  * `dinov3/convnext-base-ltdetr-coco` (pretrained on COCO)

  * `dinov3/convnext-large-ltdetr-coco` (pretrained on COCO)

  * `dinov3/vitt16-ltdetr`

  * `dinov3/vitt16-eupe-ltdetr` \- [EUPE weights](https://github.com/facebookresearch/EUPE)

  * `dinov3/vitt16plus-ltdetr`

  * `dinov3/vits16-ltdetr`

  * `dinov3/vits16-eupe-ltdetr` \- [EUPE weights](https://github.com/facebookresearch/EUPE)

  * `dinov3/vitb16-ltdetr`

  * `dinov3/vitb16-eupe-ltdetr` \- [EUPE weights](https://github.com/facebookresearch/EUPE)

  * `dinov3/vitl16-ltdetr`

  * `dinov3/convnext-tiny-ltdetr`

  * `dinov3/convnext-tiny-eupe-ltdetr` \- [EUPE weights](https://github.com/facebookresearch/EUPE)

  * `dinov3/convnext-small-ltdetr`

  * `dinov3/convnext-small-eupe-ltdetr` \- [EUPE weights](https://github.com/facebookresearch/EUPE)

  * `dinov3/convnext-base-ltdetr`

  * `dinov3/convnext-base-eupe-ltdetr` \- [EUPE weights](https://github.com/facebookresearch/EUPE)

  * `dinov3/convnext-large-ltdetr`




Unless noted otherwise, all DINOv3 backbones are initialized from weights [pretrained by Meta](https://github.com/facebookresearch/dinov3/tree/main?tab=readme-ov-file#pretrained-models). The non-EUPE models with `vitt16` and `vitt16plus` backbones use Lightly-pretrained DINOv3 backbone weights instead. Models marked as EUPE use [EUPE weights](https://github.com/facebookresearch/EUPE). DINOv3 models are under the [DINOv3 license](https://github.com/facebookresearch/dinov3?tab=License-1-ov-file). EUPE models are under the [FAIR Noncommercial Research License](https://github.com/facebookresearch/EUPE?tab=License-1-ov-file).

### LTDETR DINOv2 ModelsÂ¶

  * `dinov2/vits14-ltdetr`

  * `dinov2/vitb14-ltdetr`

  * `dinov2/vitl14-ltdetr`

  * `dinov2/vitg14-ltdetr`




All models are [pretrained by Meta](https://github.com/facebookresearch/dinov2?tab=readme-ov-file#pretrained-models).

## Training SettingsÂ¶

See [Train Settings](settings__train_settings.md#train-settings) on how to configure training settings.

## LoggingÂ¶

See [Logging](settings__train_settings.md#train-settings-logging) on how to configure logging.

## Resume TrainingÂ¶

See [Resume Training](settings__train_settings.md#train-settings-resume-training) on how to resume training.

## Default Image Transform ArgumentsÂ¶

The following are the default image transform arguments. See [Transforms](settings__train_settings.md#train-settings-transforms) on how to customize transforms.

DINOv3 LTDETR Default Transform Arguments

Train
    
    
    {
        "bbox_params": "BboxParams",
        "channel_drop": null,
        "copyblend": {
            "area_threshold": 100,
            "expand_ratios": [
                0.1,
                0.25
            ],
            "num_objects": 3,
            "prob": 0.5,
            "step_start": "auto",
            "step_stop": "auto"
        },
        "image_size": "auto",
        "mixup": {
            "prob": 0.5,
            "step_start": "auto",
            "step_stop": "auto"
        },
        "mosaic": {
            "fill_value": 0,
            "max_cached_images": 50,
            "max_size": null,
            "output_size": 320,
            "prob": 0.5,
            "random_pop": true,
            "rotation_range": 10.0,
            "scaling_range": [
                0.5,
                1.5
            ],
            "step_start": "auto",
            "step_stop": "auto",
            "translation_range": [
                0.1,
                0.1
            ]
        },
        "normalize": "auto",
        "num_channels": "auto",
        "photometric_distort": {
            "brightness": [
                0.875,
                1.125
            ],
            "contrast": [
                0.5,
                1.5
            ],
            "hue": [
                -0.05,
                0.05
            ],
            "prob": 0.5,
            "saturation": [
                0.5,
                1.5
            ],
            "step_start": "auto",
            "step_stop": "auto"
        },
        "random_flip": {
            "horizontal_prob": 0.5,
            "vertical_prob": 0.0
        },
        "random_iou_crop": {
            "crop_trials": 40,
            "iou_trials": 1000,
            "max_aspect_ratio": 2.0,
            "max_scale": 1.0,
            "min_aspect_ratio": 0.5,
            "min_scale": 0.3,
            "prob": 0.8,
            "sampler_options": null,
            "step_start": "auto",
            "step_stop": "auto"
        },
        "random_rotate": null,
        "random_rotate_90": null,
        "random_zoom_out": {
            "fill": 0.0,
            "prob": 0.5,
            "side_range": [
                1.0,
                4.0
            ],
            "step_start": "auto",
            "step_stop": "auto"
        },
        "resize": {
            "height": "auto",
            "width": "auto"
        },
        "scale_jitter": {
            "divisible_by": "auto",
            "max_scale": null,
            "min_scale": null,
            "num_scales": null,
            "prob": 1.0,
            "sizes": [
                [
                    480,
                    480
                ],
                [
                    512,
                    512
                ],
                [
                    544,
                    544
                ],
                [
                    576,
                    576
                ],
                [
                    608,
                    608
                ],
                [
                    640,
                    640
                ],
                [
                    640,
                    640
                ],
                [
                    640,
                    640
                ],
                [
                    640,
                    640
                ],
                [
                    640,
                    640
                ],
                [
                    640,
                    640
                ],
                [
                    640,
                    640
                ],
                [
                    640,
                    640
                ],
                [
                    640,
                    640
                ],
                [
                    640,
                    640
                ],
                [
                    640,
                    640
                ],
                [
                    640,
                    640
                ],
                [
                    640,
                    640
                ],
                [
                    640,
                    640
                ],
                [
                    640,
                    640
                ],
                [
                    640,
                    640
                ],
                [
                    640,
                    640
                ],
                [
                    640,
                    640
                ],
                [
                    640,
                    640
                ],
                [
                    640,
                    640
                ],
                [
                    672,
                    672
                ],
                [
                    704,
                    704
                ],
                [
                    736,
                    736
                ],
                [
                    768,
                    768
                ],
                [
                    800,
                    800
                ]
            ],
            "step_stop": "auto"
        }
    }
    

Val
    
    
    {
        "bbox_params": "BboxParams",
        "channel_drop": null,
        "copyblend": null,
        "image_size": "auto",
        "mixup": null,
        "mosaic": null,
        "normalize": "auto",
        "num_channels": "auto",
        "photometric_distort": null,
        "random_flip": null,
        "random_iou_crop": null,
        "random_rotate": null,
        "random_rotate_90": null,
        "random_zoom_out": null,
        "resize": {
            "height": "auto",
            "width": "auto"
        },
        "scale_jitter": null
    }
    

DINOv2 LTDETR Default Transform Arguments

Train
    
    
    {
        "bbox_params": "BboxParams",
        "channel_drop": null,
        "copyblend": {
            "area_threshold": 100,
            "expand_ratios": [
                0.1,
                0.25
            ],
            "num_objects": 3,
            "prob": 0.5,
            "step_start": "auto",
            "step_stop": "auto"
        },
        "image_size": "auto",
        "mixup": {
            "prob": 0.5,
            "step_start": "auto",
            "step_stop": "auto"
        },
        "mosaic": {
            "fill_value": 0,
            "max_cached_images": 50,
            "max_size": null,
            "output_size": 320,
            "prob": 0.5,
            "random_pop": true,
            "rotation_range": 10.0,
            "scaling_range": [
                0.5,
                1.5
            ],
            "step_start": "auto",
            "step_stop": "auto",
            "translation_range": [
                0.1,
                0.1
            ]
        },
        "normalize": "auto",
        "num_channels": "auto",
        "photometric_distort": {
            "brightness": [
                0.875,
                1.125
            ],
            "contrast": [
                0.5,
                1.5
            ],
            "hue": [
                -0.05,
                0.05
            ],
            "prob": 0.5,
            "saturation": [
                0.5,
                1.5
            ],
            "step_start": "auto",
            "step_stop": "auto"
        },
        "random_flip": {
            "horizontal_prob": 0.5,
            "vertical_prob": 0.0
        },
        "random_iou_crop": {
            "crop_trials": 40,
            "iou_trials": 1000,
            "max_aspect_ratio": 2.0,
            "max_scale": 1.0,
            "min_aspect_ratio": 0.5,
            "min_scale": 0.3,
            "prob": 0.8,
            "sampler_options": null,
            "step_start": "auto",
            "step_stop": "auto"
        },
        "random_rotate": null,
        "random_rotate_90": null,
        "random_zoom_out": {
            "fill": 0.0,
            "prob": 0.5,
            "side_range": [
                1.0,
                4.0
            ],
            "step_start": "auto",
            "step_stop": "auto"
        },
        "resize": {
            "height": "auto",
            "width": "auto"
        },
        "scale_jitter": {
            "divisible_by": null,
            "max_scale": null,
            "min_scale": null,
            "num_scales": null,
            "prob": 1.0,
            "sizes": [
                [
                    476,
                    476
                ],
                [
                    504,
                    504
                ],
                [
                    532,
                    532
                ],
                [
                    560,
                    560
                ],
                [
                    588,
                    588
                ],
                [
                    616,
                    616
                ],
                [
                    644,
                    644
                ],
                [
                    644,
                    644
                ],
                [
                    644,
                    644
                ],
                [
                    644,
                    644
                ],
                [
                    644,
                    644
                ],
                [
                    644,
                    644
                ],
                [
                    644,
                    644
                ],
                [
                    644,
                    644
                ],
                [
                    644,
                    644
                ],
                [
                    644,
                    644
                ],
                [
                    644,
                    644
                ],
                [
                    644,
                    644
                ],
                [
                    644,
                    644
                ],
                [
                    644,
                    644
                ],
                [
                    644,
                    644
                ],
                [
                    644,
                    644
                ],
                [
                    644,
                    644
                ],
                [
                    644,
                    644
                ],
                [
                    644,
                    644
                ],
                [
                    644,
                    644
                ],
                [
                    672,
                    672
                ],
                [
                    700,
                    700
                ],
                [
                    728,
                    728
                ],
                [
                    756,
                    756
                ],
                [
                    784,
                    784
                ],
                [
                    812,
                    812
                ]
            ],
            "step_stop": "auto"
        }
    }
    

Val
    
    
    {
        "bbox_params": "BboxParams",
        "channel_drop": null,
        "copyblend": null,
        "image_size": "auto",
        "mixup": null,
        "mosaic": null,
        "normalize": "auto",
        "num_channels": "auto",
        "photometric_distort": null,
        "random_flip": null,
        "random_iou_crop": null,
        "random_rotate": null,
        "random_rotate_90": null,
        "random_zoom_out": null,
        "resize": {
            "height": "auto",
            "width": "auto"
        },
        "scale_jitter": null
    }
    

PicoDet Default Transform Arguments

Train
    
    
    {
        "bbox_params": "BboxParams",
        "channel_drop": null,
        "copyblend": null,
        "image_size": "auto",
        "mixup": null,
        "mosaic": null,
        "normalize": "auto",
        "num_channels": "auto",
        "photometric_distort": {
            "brightness": [
                0.875,
                1.125
            ],
            "contrast": [
                0.5,
                1.5
            ],
            "hue": [
                -0.05,
                0.05
            ],
            "prob": 0.5,
            "saturation": [
                0.5,
                1.5
            ],
            "step_start": 0,
            "step_stop": null
        },
        "random_flip": {
            "horizontal_prob": 0.5,
            "vertical_prob": 0.0
        },
        "random_iou_crop": {
            "crop_trials": 40,
            "iou_trials": 1000,
            "max_aspect_ratio": 2.0,
            "max_scale": 1.0,
            "min_aspect_ratio": 0.5,
            "min_scale": 0.3,
            "prob": 0.8,
            "sampler_options": null,
            "step_start": 0,
            "step_stop": null
        },
        "random_rotate": null,
        "random_rotate_90": null,
        "random_zoom_out": {
            "fill": 0.0,
            "prob": 0.5,
            "side_range": [
                1.0,
                4.0
            ],
            "step_start": 0,
            "step_stop": null
        },
        "resize": null,
        "scale_jitter": {
            "divisible_by": null,
            "max_scale": null,
            "min_scale": null,
            "num_scales": null,
            "prob": 1.0,
            "sizes": null,
            "step_stop": null
        }
    }
    

Val
    
    
    {
        "bbox_params": "BboxParams",
        "channel_drop": null,
        "copyblend": null,
        "image_size": "auto",
        "mixup": null,
        "mosaic": null,
        "normalize": "auto",
        "num_channels": "auto",
        "photometric_distort": null,
        "random_flip": null,
        "random_iou_crop": null,
        "random_rotate": null,
        "random_rotate_90": null,
        "random_zoom_out": null,
        "resize": {
            "height": "auto",
            "width": "auto"
        },
        "scale_jitter": null
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
        out="out/my_experiment/exported_models/model.onnx"
        # precision="fp16", # Export model with FP16 weights for smaller size and faster inference.
    )
    

See [`export_onnx()`](python_api__lightly_train.md#lightly_train._task_models.dinov3_ltdetr_object_detection.task_model.DINOv3LTDETRObjectDetection.export_onnx "lightly_train._task_models.dinov3_ltdetr_object_detection.task_model.DINOv3LTDETRObjectDetection.export_onnx") for all available options when exporting to ONNX.

The following notebook shows how to export a model to ONNX in Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lightly-ai/lightly-train/blob/main/examples/notebooks/object_detection_export.ipynb)

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
    

See [`export_tensorrt()`](python_api__lightly_train.md#lightly_train._task_models.dinov3_ltdetr_object_detection.task_model.DINOv3LTDETRObjectDetection.export_tensorrt "lightly_train._task_models.dinov3_ltdetr_object_detection.task_model.DINOv3LTDETRObjectDetection.export_tensorrt") for all available options when exporting to TensorRT.

You can also learn more about exporting LTDETR to TensorRT using our Colab notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lightly-ai/lightly-train/blob/main/examples/notebooks/object_detection_export.ipynb)
