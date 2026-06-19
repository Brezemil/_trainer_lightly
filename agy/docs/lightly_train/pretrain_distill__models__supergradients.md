---
source_url: https://docs.lightly.ai/train/stable/pretrain_distill/models/supergradients.html
---

# SuperGradientsÂ¶

Important

[SuperGradients](https://github.com/Deci-AI/super-gradients) must be installed with `pip install "lightly-train[super-gradients]"`.

Warning

SuperGradients support is still experimental. There might be unexpected warnings in the logs.

## Pretrain and Fine-tune a SuperGradients ModelÂ¶

### PretrainÂ¶

Pretraining a SuperGradients models with LightlyTrain is straightforward. Below we provide the minimum scripts for pretraining using `super_gradients/yolo_nas_s` as an example:

Python
    
    
    import lightly_train
    
    if __name__ == "__main__":
        lightly_train.pretrain(
            out="out/my_experiment",                # Output directory.
            data="my_data_dir",                     # Directory with images.
            model="super_gradients/yolo_nas_s",     # Pass the supergradient model.
        )
    
    

Or alternatively, pass directly a SuperGradients model instance:
    
    
    from super_gradients.training import models
    
    import lightly_train
    
    if __name__ == "__main__":
        model = models.get(model_name="yolo_nas_s", num_classes=3)  # Load the model.
        lightly_train.pretrain(
            out="out/my_experiment",                # Output directory.
            data="my_data_dir",                     # Directory with images.
            model=model,                            # Pass the SuperGradients model.
        )
    

Command Line
    
    
    lightly-train pretrain out="out/my_experiment" data="my_data_dir" model="super_gradients/yolo_nas_s"
    

### Fine-tuneÂ¶

After pretraining, you can load the exported model for fine-tuning with SuperGradients:
    
    
    from super_gradients.training import models
    
    model = models.get(
      model_name="yolo_nas_s",
      num_classes=3,
      checkpoint_path="out/my_experiment/exported_models/exported_last.pt",
    )
    

## Supported ModelsÂ¶

  * PP-LiteSeg

    * `super_gradients/pp_lite_b_seg`

    * `super_gradients/pp_lite_b_seg50`

    * `super_gradients/pp_lite_b_seg75`

    * `super_gradients/pp_lite_t_seg`

    * `super_gradients/pp_lite_t_seg50`

    * `super_gradients/pp_lite_t_seg75`

  * SSD

    * `super_gradients/ssd_lite_mobilenet_v2`

    * `super_gradients/ssd_mobilenet_v1`

  * YOLO-NAS

    * `super_gradients/yolo_nas_l`

    * `super_gradients/yolo_nas_m`

    * `super_gradients/yolo_nas_pose_l`

    * `super_gradients/yolo_nas_pose_m`

    * `super_gradients/yolo_nas_pose_n`

    * `super_gradients/yolo_nas_pose_s`

    * `super_gradients/yolo_nas_s`



