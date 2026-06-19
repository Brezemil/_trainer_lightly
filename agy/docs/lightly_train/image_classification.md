---
source_url: https://docs.lightly.ai/train/stable/image_classification.html
---

# Image ClassificationÂ¶

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lightly-ai/lightly-train/blob/main/examples/notebooks/image_classification.ipynb)

Note

LightlyTrain supports training image classification models using any backbone. Both multiclass and multilabel classification are supported.

## Train an Image Classification ModelÂ¶

Training an image classification model with LightlyTrain is straightforward and only requires a few lines of code. See data for more details on how to prepare your dataset.

### Multiclass ClassificationÂ¶

In multiclass classification, each image is assigned to exactly one class. This is the default mode and does not need to be specified explicitly.
    
    
    import lightly_train
    
    if __name__ == "__main__":
        lightly_train.train_image_classification(
            out="out/my_experiment",
            model="dinov3/vitt16",
            data={
                "train": "my_data_dir/train/",
                "val": "my_data_dir/val/",
                "classes": {
                    0: "cat",
                    1: "car",
                    2: "dog",
                    # ...
                },
            },
        )
    

During training, both the

  * best (with highest validation top-1 accuracy) and

  * last (last validation round as determined by `save_checkpoint_args.save_every_num_steps`)




model weights are exported to `out/my_experiment/exported_models/`, unless disabled in `save_checkpoint_args`. You can use these weights to continue fine-tuning on another task by loading the weights via the `model` parameter:
    
    
    import lightly_train
    
    if __name__ == "__main__":
        lightly_train.train_image_classification(
            out="out/my_experiment",
            model="out/my_experiment/exported_models/exported_best.pt",  # Continue training from the best model
            data={...},
        )
    

### Load the Trained Model from Checkpoint and PredictÂ¶

After the training completes, you can load the best model checkpoint for inference like this:
    
    
    import lightly_train
    
    model = lightly_train.load_model("out/my_experiment/exported_models/exported_best.pt")
    results = model.predict("path/to/image.jpg", topk=1, threshold=0.5)
    results["labels"]   # Class labels, tensor of shape (topk,)
    results["scores"]   # Confidence scores, tensor of shape (topk,)
    

The predicted label is the class ID as defined in the `classes` dictionary in the dataset configuration.

## Multilabel ClassificationÂ¶

In multilabel classification, each image can be assigned to multiple classes simultaneously. To enable multilabel classification, set `classification_task` to `"multilabel"`. This requires a CSV-based dataset where multiple labels per image can be specified. See CSV-based Datasets for details on the data format.
    
    
    import lightly_train
    
    if __name__ == "__main__":
        lightly_train.train_image_classification(
            out="out/my_experiment",
            model="dinov3/vitt16",
            classification_task="multilabel",
            data={
                "train_csv": "my_data_dir/train.csv",
                "val_csv": "my_data_dir/val.csv",
                "classes": {
                    0: "cat",
                    1: "car",
                    2: "dog",
                    # ...
                },
            },
        )
    

For multilabel classification, the model is saved based on the best validation F1 score. When running inference, the model returns all labels with a confidence score above 0.5 by default:
    
    
    import lightly_train
    
    model = lightly_train.load_model("out/my_experiment/exported_models/exported_best.pt")
    results = model.predict("path/to/image.jpg")
    results["labels"]   # Class labels, tensor of shape (num_labels,)
    results["scores"]   # Confidence scores, tensor of shape (num_labels,)
    

## OutÂ¶

The `out` argument specifies the output directory where all training logs, model exports, and checkpoints are saved. It looks like this after training:
    
    
    out/my_experiment
    âââ checkpoints
    â   âââ last.ckpt                                       # Last checkpoint
    âââ exported_models
    â   âââ exported_last.pt                                # Last model exported (unless disabled)
    â   âââ exported_best.pt                                # Best model exported (unless disabled)
    âââ events.out.tfevents.1721899772.host.1839736.0       # TensorBoard logs
    âââ train.log                                           # Training logs
    

The final model checkpoint is saved to `out/my_experiment/checkpoints/last.ckpt`. The last and best model weights are exported to `out/my_experiment/exported_models/` unless disabled in `save_checkpoint_args`.

Tip

Create a new output directory for each experiment to keep training logs, model exports, and checkpoints organized.

## DataÂ¶

LightlyTrain supports training image classification models using either a directory-based dataset structure or CSV annotation files. Both single-label and multi-label classification are supported.

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




### Folder-based Datasets (Single-label)Â¶

In the simplest setup, images are organized into subdirectories, where each subdirectory corresponds to one class. The directory name defines the class name, and all images inside that directory are assigned to that class.

Your dataset directory should be organized like this:
    
    
    my_data_dir/
    âââ train
    â   âââ cat
    â   â   âââ img1.jpg
    â   â   âââ img2.jpg
    â   â   âââ ...
    â   âââ car
    â   â   âââ img1.jpg
    â   â   âââ img2.jpg
    â   â   âââ ...
    â   âââ ...
    âââ val
        âââ cat
        â   âââ img1.jpg
        â   âââ ...
        âââ car
        â   âââ img1.jpg
        â   âââ ...
        âââ ...
    

To train with this directory structure, set the `data` argument like this:
    
    
    import lightly_train
    
    if __name__ == "__main__":
        lightly_train.train_image_classification(
            out="out/my_experiment",
            model="dinov3/vitt16",
            data={
                "train": "my_data_dir/train/",
                "val": "my_data_dir/val/",
                "classes": {
                    0: "cat",
                    1: "car",
                    2: "dog",
                    # ...
                },
                # Optional, classes that are in the dataset but should be ignored during
                # training.
                "ignore_classes": [0],
            },
        )
    

In this setup:

  * Each image belongs to exactly one class.

  * Class names are taken from the directory names.

  * Class IDs are assigned according to the passed `classes` dictionary.




### CSV-based Datasets (Single-label and Multi-label)Â¶

For more flexibility, LightlyTrain also supports CSV files that explicitly map image paths to labels. This is required for multi-label classification, and can also be used for single-label datasets.

Each split (train, val, optionally test) must have its own CSV file.

#### CSV formatÂ¶

A CSV file must contain:

  * one column specifying the image path

  * one column specifying the label(s)




The image path can be absolute or relative to the CSV file location.

For example, given the following dataset layout:
    
    
    my_data_dir/
    âââ train
    â   âââ cat
    â   â   âââ img1.jpg
    â   â   âââ img2.jpg
    â   â   âââ ...
    â   âââ car
    â   â   âââ img1.jpg
    â   â   âââ img2.jpg
    â   â   âââ ...
    â   âââ ...
    âââ val
    â   âââ cat
    â   â   âââ img1.jpg
    â   â   âââ ...
    â   âââ car
    â   â   âââ img1.jpg
    â   â   âââ ...
    â   âââ ...
    âââ train.csv
    âââ val.csv
    

A corresponding `train.csv` with class names could look like this:
    
    
    image_path,label
    train/cat/img1.jpg,"cat"
    train/cat/img2.jpg,"cat,dog"
    train/car/img1.jpg,"car"
    

and with class IDs:
    
    
    train/cat/img1.jpg,"0"
    train/cat/img2.jpg,"0,2"
    train/car/img1.jpg,"1"
    

In this case, the image paths are interpreted relative to the directory containing the CSV file, i.e., `my_data_dir/`.

To train with this CSV-based structure, set the `data` argument like this:
    
    
    import lightly_train
    
    if __name__ == "__main__":
        lightly_train.train_image_classification(
            out="out/my_experiment",
            model="dinov3/vitt16",
            data={
                "train_csv": "my_data_dir/train.csv",
                "val_csv": "my_data_dir/val.csv",
                "classes": {
                    0: "cat",
                    1: "car",
                    2: "dog",
                    # ...
                },
                # Optional, classes that are in the dataset but should be ignored during
                # training.
                "ignore_classes": [0],
            },
        )
    

Notes:

  * Image paths must either be absolute or relative to the directory containing the CSV file.

  * Multiple labels are separated by a delimiter (default: `","`).

  * When using commas as label delimiters, the label field must be quoted.

  * Labels can be specified either as class IDs or class names.




#### Supported CSV OptionsÂ¶

The behavior of CSV parsing can be configured via the `data` argument:
    
    
    import lightly_train
    
    if __name__ == "__main__":
        lightly_train.train_image_classification(
            out="out/my_experiment",
            model="dinov3/vitt16",
            data={
                "train_csv": "my_data_dir/train.csv",
                "val_csv": "my_data_dir/val.csv",
                "classes": {
                    0: "cat",
                    1: "car",
                    2: "dog",
                    # ...
                },
                # Optional, classes that are in the dataset but should be ignored during
                # training.
                "ignore_classes": [0],
                # Extra arguments for CSV-based datasets.
                "csv_image_column": "image_path", # Name of the column storing image paths.
                "csv_label_column": "label",      # Name of the column storing labels.
                "csv_label_type": "name",         # Type of labels either "name" or "id".
                "label_delimiter": ",",           # Delimiter used to separate the labels.
            },
        )
    

## ModelÂ¶

The `model` argument defines the backbone model used for image classification. All LightlyTrain models are supported as backbones. For example:

  * `dinov3/vitt16`

  * `dinov2/vitb14`

  * `timm/resnet18`

  * `torchvision/resnet50`




See [Models](https://docs.lightly.ai/train/stable/pretrain_distill/models/index.html) for a full list of supported model backbones.

## Training SettingsÂ¶

See [Train Settings](settings__train_settings.md#train-settings) on how to configure training settings.

## LoggingÂ¶

See [Logging](settings__train_settings.md#train-settings-logging) on how to configure logging.

## Resume TrainingÂ¶

See [Resume Training](settings__train_settings.md#train-settings-resume-training) on how to resume training.

## Default Image Transform ArgumentsÂ¶

The following are the default image transform arguments. See [Transforms](settings__train_settings.md#train-settings-transforms) on how to customize transform settings.

Image Classification Default Transform Arguments

Train
    
    
    {
        "channel_drop": null,
        "color_jitter": {
            "brightness": 0.12549019607843137,
            "contrast": 0.5,
            "hue": 0.05,
            "prob": 0.5,
            "saturation": 0.5,
            "strength": 1.0
        },
        "image_size": "auto",
        "normalize": "auto",
        "num_channels": "auto",
        "random_crop": {
            "max_scale": 1.0,
            "min_scale": 0.2
        },
        "random_flip": {
            "horizontal_prob": 0.5,
            "vertical_prob": 0.0
        },
        "random_rotate": null,
        "random_rotate_90": null,
        "resize_scale": null
    }
    

Val
    
    
    {
        "channel_drop": null,
        "color_jitter": null,
        "image_size": "auto",
        "normalize": "auto",
        "num_channels": "auto",
        "random_crop": null,
        "random_flip": null,
        "random_rotate": null,
        "random_rotate_90": null,
        "resize_scale": 1.143
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
    

See [`export_onnx()`](python_api__lightly_train.md#lightly_train._task_models.image_classification.task_model.ImageClassification.export_onnx "lightly_train._task_models.image_classification.task_model.ImageClassification.export_onnx") for all available options when exporting to ONNX.

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
    

See [`export_tensorrt()`](python_api__lightly_train.md#lightly_train._task_models.image_classification.task_model.ImageClassification.export_tensorrt "lightly_train._task_models.image_classification.task_model.ImageClassification.export_tensorrt") for all available options when exporting to TensorRT.
