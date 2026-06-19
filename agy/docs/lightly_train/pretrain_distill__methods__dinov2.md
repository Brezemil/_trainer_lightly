---
source_url: https://docs.lightly.ai/train/stable/pretrain_distill/methods/dinov2.html
---

# DINOv2Â¶

DINOv2 is a state-of-the-art self-supervised learning method for training vision foundation models. It is optimized for large-scale models and datasets. DINOv2 pretrained models are effective across a wide range of tasks, including image classification, object detection, and segmentation. They are also known to generate high-quality features that can be used without fine-tuning the model.

Implementation | Model | Val ImageNet k-NN  
---|---|---  
LightlyTrain | ViT-L/16 | 81.9%  
[Official](https://github.com/facebookresearch/dinov2) | ViT-L/16 | 81.6%  
  
_The LightlyTrain DINOv2 implementation matches or outperforms the official implementation. All models are trained from scratch on ImageNet-1K._

## Use DINOv2 in LightlyTrainÂ¶

Python
    
    
    import lightly_train
    
    if __name__ == "__main__":
        lightly_train.pretrain(
            out="out/my_experiment", 
            data="my_data_dir",
            model="dinov2/vitb14",
            method="dinov2",
        )
    

Command Line
    
    
    lightly-train pretrain out=out/my_experiment data=my_data_dir model="dinov2/vitb14" method="dinov2"
    

The following models are available for DINOv2 pretraining. Pretrained models initialize from Metaâs weights and are recommended when fine-tuning on domain-specific data. Not-pretrained models start from random initialization and are useful for training from scratch on large custom datasets.

  * Pretrained (Meta weights)

    * `dinov2/vits14`

    * `dinov2/vitb14`

    * `dinov2/vitl14`

    * `dinov2/vitg14`

  * Not pretrained (random initialization)

    * `dinov2/vits14-notpretrained`

    * `dinov2/vitb14-notpretrained`

    * `dinov2/vitl14-notpretrained`

    * `dinov2/vitg14-notpretrained`




See [DINOv2 models](pretrain_distill__models__dinov2.md#models-dinov2) for a full list including models without register tokens.

## Whatâs under the HoodÂ¶

DINOv2 combines the strengths of DINO and iBOT, two previous self-supervised learning methods. Following DINO, it trains a student network to match the output of a momentum-averaged teacher network without labels. It also incorporates the masked image modeling loss from iBOT, which helps the model learn strong local semantic features.

## Lightly RecommendationsÂ¶

  * **Models** : DINOv2 can only be used with ViTs. If you want to use a different model, we recommend first pretraining a ViT with DINOv2 and then distilling the knowledge of the ViT into your model of choice with the [distillation method](pretrain_distill__methods__distillation.md#methods-distillation).

  * **Batch Size** : We recommend somewhere around 3072 for DINOv2 as the original paper suggested.

  * **Number of Epochs** : We recommend somewhere between 100 to 300 epochs. However, DINOv2 benefits from longer schedules and may still improve after training for more than 300 epochs.

  * **Large Datasets** : DINOv2 is optimized for large datasets. We recommend at least 1 million images for training from scratch.




## Default Method ArgumentsÂ¶

The following are the default method arguments for DINOv2. To learn how you can override these settings, see [Method Arguments](https://docs.lightly.ai/train/stable/pretrain_distill/index.html#method-args).

Default Method Arguments
    
    
    {
        "batch_norm": false,
        "center_method": "softmax",
        "center_momentum": 0.9,
        "dino_bottleneck_dim": 256,
        "dino_loss_weight": 1.0,
        "gradient_clip_val": 3.0,
        "hidden_dim": 2048,
        "ibot_bottleneck_dim": 256,
        "ibot_loss_weight": 1.0,
        "ibot_separate_head": false,
        "koleo_loss_weight": 0.1,
        "layerwise_decay": 0.9,
        "lr_scale_method": "sqrt",
        "mask_probability": 0.5,
        "mask_ratio_max": 0.5,
        "mask_ratio_min": 0.1,
        "min_lr": 1e-06,
        "momentum_end": 1.0,
        "momentum_start": 0.992,
        "output_dim": 65536,
        "patch_embed_lr_multiplier": 0.2,
        "reference_batch_size": 1024,
        "student_freeze_backbone_steps": 0,
        "student_freeze_last_layer_steps": 1250,
        "student_temp": 0.1,
        "teacher_temp_end": 0.07,
        "teacher_temp_start": 0.04,
        "teacher_temp_warmup_steps": 37500,
        "warmup_steps": 12500,
        "weight_decay_end": 0.4,
        "weight_decay_start": "auto"
    }
    

## Default Image Transform ArgumentsÂ¶

The following are the default transform arguments for DINOv2. To learn how you can override these settings, see [Transforms](settings__pretrain_settings.md#method-transform-args).

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
            "num_views": 8,
            "random_resize": {
                "max_scale": 0.32,
                "min_scale": 0.05
            },
            "view_size": [
                98,
                98
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
            "min_scale": 0.32
        },
        "random_rotation": null,
        "solarize": null
    }
    
