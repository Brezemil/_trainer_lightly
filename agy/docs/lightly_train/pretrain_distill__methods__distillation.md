---
source_url: https://docs.lightly.ai/train/stable/pretrain_distill/methods/distillation.html
---

# DistillationÂ¶

[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lightly-ai/lightly-train/blob/main/examples/notebooks/distillation.ipynb)

Knowledge distillation involves transferring knowledge from a large, compute-intensive teacher model to a smaller, efficient student model by encouraging similarity between the student and teacher representations. It addresses the challenge of bridging the gap between state-of-the-art large-scale vision models and smaller, more computationally efficient models suitable for practical applications.

Note

Three distillation versions are available. Choose based on your downstream task:

  * **`distillation`** (alias for `distillationv3`, default from **LightlyTrain 0.15.0**): Best compromise â strong on both global tasks (e.g., classification) and dense tasks (e.g., detection, segmentation). Recommended for most use cases.

  * **`distillationv1`** : Best for purely global tasks (e.g., image classification).

  * **`distillationv2`** : Best for purely dense tasks (e.g., object detection, segmentation) and we advise to only use it with DINOv2 teacher models (use `distillationv3` for DINOv3 teachers instead).




## Use Distillation in LightlyTrainÂ¶

[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lightly-ai/lightly-train/blob/main/examples/notebooks/distillation.ipynb)

Follow the code below to distill the knowledge of the default DINOv3 ViT-B/16 teacher model into your model architecture. The example uses a `torchvision/resnet18` model as the student:

Note

DINOv3 models are released under the [DINOv3 license](https://github.com/lightly-ai/lightly-train/blob/main/licences/DINOv3_LICENSE.md). Use DINOv2 models instead for a more permissive Apache 2.0 license.

Python
    
    
    import lightly_train
    
    if __name__ == "__main__":
        lightly_train.pretrain(
            out="out/my_experiment",
            data="my_data_dir",
            model="torchvision/resnet18",
            method="distillation",
        )
    

Command Line
    
    
    lightly-train pretrain out=out/my_experiment data=my_data_dir model="torchvision/resnet18" method="distillation"
    

### Custom Teacher and Student ModelsÂ¶

Besides the built-in support for many popular models, all distillation versions also support using custom student models, by implementing the interface specified on the [Custom Models](pretrain_distill__models__custom_models.md#custom-models) page.

With distillationv3, LightlyTrain now also supports custom teacher models, by implementing the same interface for the teacher.

**Option A: String-based** â use any supported model string with optional constructor args:
    
    
    import lightly_train
    
    if __name__ == "__main__":
        lightly_train.pretrain(
            out="out/my_experiment",
            data="my_data_dir",
            model="timm/resnet18",           # student as string
            model_args={"num_classes": 120}, # optional student constructor args
            method="distillationv3",
            method_args={
                "teacher": "timm/vit_base_patch16_224",  # any supported model string
                "teacher_args": {"pretrained": False},    # optional teacher constructor args
                "teacher_weights": "path/to/teacher_weights.pt", # optional teacher weights path
            }
        )
    

**Option B: Pre-instantiated wrapper** â implement the interface on the [Custom Models](pretrain_distill__models__custom_models.md#custom-models) page and pass an instance of your wrapper directly to the `model` (for student) and `teacher` (for teacher) arguments:
    
    
    import lightly_train
    from my_module import MyStudentWrapper, MyTeacherWrapper
    
    if __name__ == "__main__":
        student = MyStudentWrapper(my_student_model)
        teacher = MyTeacherWrapper(my_teacher_model)
    
        lightly_train.pretrain(
            out="out/my_experiment",
            data="my_data_dir",
            model=student,
            method="distillationv3",
            method_args={"teacher": teacher},
        )
    

When passing pre-instantiated wrappers, `model_args` and `teacher_args` are ignored since the models are already constructed.

### Pretrain and Distill Your Own DINOv2 WeightsÂ¶

LightlyTrain also supports [DINOv2 pretraining](pretrain_distill__methods__dinov2.md#methods-dinov2), which can help you adjust the DINOv2 weights to your own domain data. Starting from **LightlyTrain 0.9.0** , after pretraining a ViT with DINOv2, you can distill your own pretrained model to your target model architecture with the distillation method. This is done by setting an optional `teacher_weights` argument in `method_args`.

The following example shows how to pretrain a ViT-B/14 model with DINOv2 and then distill the pretrained model to a ResNet-18 student model. Check out the [DINOv2 pretraining documentation](pretrain_distill__methods__dinov2.md#methods-dinov2) for more details on how to pretrain a DINOv2 model.

Python
    
    
    import lightly_train
    
    if __name__ == "__main__":
        # Pretrain a DINOv2 ViT-B/14 model.
        lightly_train.pretrain(
            out="out/my_dinov2_pretrain_experiment",
            data="my_dinov2_pretrain_data_dir",
            model="dinov2/vitb14",
            method="dinov2",
        )
    
        # Distill the pretrained DINOv2 model to a ResNet-18 student model.
        lightly_train.pretrain(
            out="out/my_distillation_pretrain_experiment",
            data="my_distillation_pretrain_data_dir",
            model="torchvision/resnet18",
            method="distillation",
            method_args={
                "teacher": "dinov2/vitb14",
                "teacher_weights": "out/my_dinov2_pretrain_experiment/exported_models/exported_last.pt", # pretrained `dinov2/vitb14` weights 
            }
        )
    

### Supported Teacher ModelsÂ¶

For distillation v1/v2, the following models for `teacher` are supported:

  * DINOv3

    * `dinov3/vits16`

    * `dinov3/vits16plus`

    * `dinov3/vitb16`

    * `dinov3/vitl16`

    * `dinov3/vitl16-sat493m`

    * `dinov3/vitl16plus`

    * `dinov3/vith16plus`

    * `dinov3/vit7b16`

    * `dinov3/vit7b16-sat493m`

    * `dinov3/convnext-tiny`

    * `dinov3/convnext-small`

    * `dinov3/convnext-base`

    * `dinov3/convnext-large`

  * DINOv2

    * `dinov2/vits14`

    * `dinov2/vitb14`

    * `dinov2/vitl14`

    * `dinov2/vitg14`




For distillationv3, any model supported by LightlyTrain can be used (including custom models). You can find the full list of supported models on the [Models](https://docs.lightly.ai/train/stable/pretrain_distill/models/index.html#models) page.

## Whatâs under the HoodÂ¶

All versions apply a loss between the features of the student and teacher networks when processing the same image, using strong identical augmentations on both inputs for consistency. The different versions draw heavy inspiration from:

  * [_Knowledge Distillation: A Good Teacher is Patient and Consistent_](https://arxiv.org/abs/2106.05237).

  * The [_AM-RADIO_](https://arxiv.org/pdf/2304.07193) series of papers.




The versions differ in how the loss is computed:

  * **v1** uses a queue of teacher embeddings to compute pseudo labels (global loss). Best for global tasks such as classification or distilling your own embedding model.

  * **v2** directly applies MSE loss on the spatial features (dense loss). Best for dense tasks such as detection and segmentation.

  * **v3** combines both the queue-based pseudo label loss and loss on the spatial features, making it a strong general-purpose choice.




## Lightly RecommendationsÂ¶

  * **Models** : Knowledge distillation is agnostic to the choice of student backbone networks.

  * **Batch Size** : We recommend somewhere between 128 and 2048 for knowledge distillation.

  * **Number of Epochs** : We recommend somewhere between 100 and 3000. However, distillation benefits from longer schedules and models still improve after pretraining for more than 3000 epochs. For small datasets (<100k images) it can also be beneficial to pretrain up to 10000 epochs.




## Default Method ArgumentsÂ¶

The following are the default method arguments for distillation. To learn how you can override these settings, see [Method Arguments](https://docs.lightly.ai/train/stable/pretrain_distill/index.html#method-args).

distillation (v3)
    
    
    {
        "loss_local_weight": 1.0,
        "lr_scale_method": "sqrt",
        "queue_size": "auto",
        "reference_batch_size": 1536,
        "teacher": "dinov3/vitb16",
        "teacher_args": null,
        "teacher_weights": null,
        "temperature_global": 0.07,
        "temperature_local": 0.07
    }
    

distillationv1
    
    
    {
        "lr_scale_method": "sqrt",
        "queue_size": "auto",
        "reference_batch_size": 1536,
        "teacher": "dinov2/vitb14-noreg",
        "teacher_url": null,
        "teacher_weights": null,
        "temperature": 0.07
    }
    

distillationv2
    
    
    {
        "lr_scale_method": "sqrt",
        "n_projection_layers": 1,
        "n_teacher_blocks": 2,
        "projection_hidden_dim": 2048,
        "reference_batch_size": 1536,
        "teacher": "dinov2/vitb14-noreg",
        "teacher_url": null,
        "teacher_weights": null
    }
    

## Default Image Transform ArgumentsÂ¶

The following are the default transform arguments for distillation. To learn how you can override these settings, see [Transforms](settings__pretrain_settings.md#method-transform-args).

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
                0.0,
                0.1
            ]
        },
        "image_size": [
            224,
            224
        ],
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
    
