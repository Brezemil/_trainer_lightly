---
source_url: https://docs.lightly.ai/train/stable/pretrain_distill/methods/dino.html
---

# DINOÂ¶

[DINO (Distillation with No Labels)](https://arxiv.org/abs/2104.14294) is a self-supervised learning framework for visual representation learning using knowledge distillation but without the need for labels. Similar to knowledge distillation, DINO uses a teacher-student setup where the student learns to mimic the teacherâs outputs. The major difference is that DINO uses an exponential moving average of the student as teacher. DINO achieves strong performance on image clustering, segmentation, and zero-shot transfer tasks.

## Use DINO in LightlyTrainÂ¶

Python
    
    
    import lightly_train
    
    if __name__ == "__main__":
        lightly_train.pretrain(
            out="out/my_experiment", 
            data="my_data_dir",
            model="torchvision/resnet18",
            method="dino",
        )
    

Command Line
    
    
    lightly-train pretrain out=out/my_experiment data=my_data_dir model="torchvision/resnet18" method="dino"
    

## Whatâs under the HoodÂ¶

DINO trains a student network to match the output of a momentum-averaged teacher network without labels. It employs a self-distillation objective with a cross-entropy loss between the student and teacher outputs. DINO uses random cropping, resizing, color jittering, and Gaussian blur to create diverse views of the same image. In particular, DINO employs a multi-crop augmentation strategy to generate two global views and multiple local views that are smaller crops of the original image. Additionally, centering and sharpening of the teacher pseudo labels is used to stabilize the training.

## Lightly RecommendationsÂ¶

  * **Models** : DINO works well with both ViT and CNN.

  * **Batch Size** : We recommend somewhere between 256 and 1024 for DINO as the original paper suggested.

  * **Number of Epochs** : We recommend somewhere between 100 to 300 epochs. However, DINO benefits from longer schedules and may still improve after training for more than 300 epochs.




## Default Method ArgumentsÂ¶

The following are the default method arguments for DINO. To learn how you can override these settings, see [Method Arguments](https://docs.lightly.ai/train/stable/pretrain_distill/index.html#method-args).

Default Method Arguments
    
    
    {
        "batch_norm": false,
        "bottleneck_dim": 256,
        "center_momentum": 0.9,
        "hidden_dim": 2048,
        "lr_scale_method": "linear",
        "momentum_end": 1.0,
        "momentum_start": "auto",
        "norm_last_layer": true,
        "output_dim": "auto",
        "reference_batch_size": 256,
        "student_freeze_last_layer_epochs": null,
        "student_freeze_last_layer_steps": "auto",
        "student_temp": 0.1,
        "teacher_temp": "auto",
        "warmup_max_steps_fraction": 0.1,
        "warmup_steps": 12500,
        "warmup_teacher_temp": "auto",
        "warmup_teacher_temp_epochs": null,
        "warmup_teacher_temp_max_steps_fraction": 0.3,
        "warmup_teacher_temp_steps": "auto",
        "weight_decay_end": "auto",
        "weight_decay_start": "auto"
    }
    

## Default Image Transform ArgumentsÂ¶

The following are the default transform arguments for DINO. To learn how you can override these settings, see [Transforms](settings__pretrain_settings.md#method-transform-args).

Default Image Transforms
    
    
    {
        "channel_drop": null,
        "color_jitter": {
            "brightness": 0.8,
            "contrast": 0.8,
            "hue": 0.2,
            "prob": 0.8,
            "saturation": 0.4,
            "strength": 0.5
        },
        "gaussian_blur": {
            "blur_limit": 0,
            "prob": 1.0,
            "sigmas": [
                0.1,
                2.0
            ]
        },
        "global_view_1": {
            "gaussian_blur": {
                "blur_limit": 0,
                "prob": 0.1,
                "sigmas": [
                    0.1,
                    2.0
                ]
            },
            "solarize": {
                "prob": 0.2,
                "threshold": 0.5
            }
        },
        "image_size": [
            224,
            224
        ],
        "local_view": {
            "gaussian_blur": {
                "blur_limit": 0,
                "prob": 0.5,
                "sigmas": [
                    0.1,
                    2.0
                ]
            },
            "num_views": 6,
            "random_resize": {
                "max_scale": 0.14,
                "min_scale": 0.05
            },
            "view_size": [
                96,
                96
            ]
        },
        "normalize": {
            "mean": [
                0.485,
                0.456,
                0.406
            ],
            "std": [
                0.229,
                0.224,
                0.225
            ]
        },
        "num_channels": 3,
        "random_flip": {
            "horizontal_prob": 0.5,
            "vertical_prob": 0.0
        },
        "random_gray_scale": 0.2,
        "random_resize": {
            "max_scale": 1.0,
            "min_scale": 0.14
        },
        "random_rotation": null,
        "solarize": null
    }
    
