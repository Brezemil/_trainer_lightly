---
source_url: https://docs.lightly.ai/train/stable/pretrain_distill/models/yolov12.html
---

# YOLOv12Â¶

This page describes how to use the [YOLOv12 implementation by the original authors](https://github.com/sunsmarterjie/yolov12) with LightlyTrain.

Note

YOLOv12 is a custom fork of a specific version of the Ultralytics package. For this reason, YOLOv12 is not fully integrated with LightlyTrain and has to be installed manually.

## YOLOv12 InstallationÂ¶

Important

Be aware that this will overwrite any current installation of `ultralytics`, so it is best to do in a new virtual environment.

Please install `YOLOv12` directly from GitHub through:
    
    
    pip install git+https://github.com/sunsmarterjie/yolov12
    

In case you are facing a version mismatch issue using CUDA and FlashAttention:
    
    
    FlashAttention is not available on this device. Using scaled_dot_product_attention instead.
    

you can fix it by running the following commands:
    
    
    pip install flash-attn --no-build-isolation
    

You can verify the results by:
    
    
    python -c "import flash_attn; print('FlashAttention version:', flash_attn.__version__)"
    

and a successful installation will give you:
    
    
    FlashAttention version: <some-version>
    

See [this GitHub issue](https://github.com/sunsmarterjie/yolov12/issues/66) for more information.

## Pretrain and Fine-tune a YOLOv12 ModelÂ¶

Pretraining or fine-tuning a YOLOv12 model is the same as doing so with any supported Ultralytics model. The only difference is that the config file is named `yolov12.yaml` instead of `yolo12.yaml` in the official Ultralytics releases.

### PretrainÂ¶

Below we provide the minimum scripts for pretraining using `ultralytics/yolov12s` as an example:

Python
    
    
    import lightly_train
    
    if __name__ == "__main__":
        lightly_train.pretrain(
            out="out/my_experiment",                # Output directory.
            data="my_data_dir",                     # Directory with images.
            model="ultralytics/yolov12s.yaml",      # Pass the YOLO model.
        )
    
    

Or alternatively, pass directly a YOLOv12 model instance:
    
    
    from ultralytics import YOLO
    
    import lightly_train
    
    if __name__ == "__main__":
        model = YOLO("yolov12s.yaml")               # Load the YOLOv12 model.
        lightly_train.pretrain(
            out="out/my_experiment",                # Output directory.
            data="my_data_dir",                     # Directory with images.
            model=model,                            # Pass the YOLOv12 model.
        )
    

Command Line
    
    
    lightly-train pretrain out="out/my_experiment" data="my_data_dir" model="ultralytics/yolov12s.yaml"
    

### Fine-tuneÂ¶

After pretraining, you can load the exported model for fine-tuning with Ultralytics:

Python
    
    
    from pathlib import Path
    
    from ultralytics import YOLO
    
    if __name__ == "__main__":
        # Load the exported model.
        model = YOLO("out/my_experiment/exported_models/exported_last.pt")
    
        # Fine-tune with ultralytics.
        data = Path("my_data_dir/config.yaml").absolute()
        model.train(data=data)
    

Command Line
    
    
    yolo detect train model=out/my_experiment/exported_models/exported_last.pt data="my_data_dir"
    

## Supported ModelsÂ¶

The following YOLOv12 model variants are supported:

  * `ultralytics/yolov12n.yaml`

  * `ultralytics/yolov12n.pt`

  * `ultralytics/yolov12s.yaml`

  * `ultralytics/yolov12s.pt`

  * `ultralytics/yolov12m.yaml`

  * `ultralytics/yolov12m.pt`

  * `ultralytics/yolov12l.yaml`

  * `ultralytics/yolov12l.pt`

  * `ultralytics/yolov12x.yaml`

  * `ultralytics/yolov12x.pt`




Check the [Ultralytics page](pretrain_distill__models__ultralytics.md) for more information on our support of earlier YOLO variants.
