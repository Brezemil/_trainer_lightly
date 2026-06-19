---
source_url: https://docs.lightly.ai/train/stable/pretrain_distill/models/dinov3.html
---

# DINOv3Â¶

This page describes how to use DINOv3 models with LightlyTrain.

[DINOv3](https://github.com/facebookresearch/dinov3) models are Vision Transformers (ViTs) and ConvNeXt models pretrained by Meta using the DINOv3 self-supervised learning method on the large-scale LVD-1689M dataset. They are state-of-the-art vision foundation models and serve as strong backbones for downstream tasks such as object detection, segmentation, and image classification.

Note

DINOv3 models are released under the [DINOv3 license](https://github.com/lightly-ai/lightly-train/blob/main/licences/DINOv3_LICENSE.md). Use [DINOv2](pretrain_distill__models__dinov2.md#models-dinov2) models instead for a more permissive Apache 2.0 license.

## Pretrain and Fine-tune a DINOv3 ModelÂ¶

### PretrainÂ¶

DINOv3 ViT-T/16 models (`dinov3/vitt16` and `dinov3/vitt16plus`) are efficient tiny models trained by Lightly using the [distillation method](pretrain_distill__methods__distillation.md#methods-distillation) with DINOv3 ViT-L/16 as the teacher on ImageNet-1K. They are not part of Metaâs official DINOv3 release but follow the same architecture. The ViT-T architecture is based on the design proposed in [Touvron et al., 2022](https://arxiv.org/abs/2207.10666).

You can distill your own DINOv3 ViT-T/16 model from DINOv3 ViT-L/16 on your custom dataset as follows:

Python
    
    
    import lightly_train
    
    if __name__ == "__main__":
        lightly_train.pretrain(
            out="out/my_experiment",                # Output directory.
            data="my_data_dir",                     # Directory with images.
            model="dinov3/vitt16",                  # Student: DINOv3 ViT-T/16.
            method="distillation",
            method_args={
                "teacher": "dinov3/vitl16",         # Teacher: DINOv3 ViT-L/16.
            },
        )
    

Command Line
    
    
    lightly-train pretrain out="out/my_experiment" data="my_data_dir" model="dinov3/vitt16" method="distillation" method_args.teacher="dinov3/vitl16"
    

See [Distillation method](pretrain_distill__methods__distillation.md#methods-distillation) for more details on the pretraining method and its configuration options.

### Fine-tuneÂ¶

DINOv3 models come with high-quality pretrained weights from Meta and can be used directly as fine-tuning backbones without additional pretraining. After pretraining on a custom dataset, the exported backbone can also be loaded via the `backbone_weights` argument. Refer to the following pages for fine-tuning instructions and example code:

  * [Object Detection](object_detection.md#object-detection) â fine-tune a DINOv3-based LTDETR model; supports loading custom pretrained backbone weights via `backbone_weights` (see [Pretrain and Fine-tune](object_detection.md#object-detection-pretrain-finetune)).

  * [Semantic Segmentation](semantic_segmentation.md#semantic-segmentation) â fine-tune a DINOv3-based EoMT model; supports loading custom pretrained backbone weights via `backbone_weights` (see [Pretrain and Fine-tune](semantic_segmentation.md#semantic-segmentation-pretrain-finetune)).

  * [Instance Segmentation](instance_segmentation.md#instance-segmentation) â fine-tune a DINOv3-based EoMT model.

  * [Panoptic Segmentation](panoptic_segmentation.md#panoptic-segmentation) â fine-tune a DINOv3-based EoMT model.

  * [Image Classification](image_classification.md#image-classification) â fine-tune a DINOv3 backbone for classification.




## Supported ModelsÂ¶

### ViT ModelsÂ¶

The following ViT models are supported. The LVD-1689M and SAT-493M models are [pretrained by Meta](https://github.com/facebookresearch/dinov3/tree/main?tab=readme-ov-file#pretrained-models) and are under the [DINOv3 license](https://github.com/facebookresearch/dinov3?tab=License-1-ov-file). The EUPE models are pretrained by Meta using the [EUPE method](https://github.com/facebookresearch/EUPE) and are under the [FAIR Noncommercial Research License](https://github.com/facebookresearch/EUPE?tab=License-1-ov-file). The ViT-T/16 models, except the EUPE one, are trained by Lightly using knowledge distillation from DINOv3 ViT-L/16.

  * ViT-T (Lightly, distilled from DINOv3 ViT-L/16 on ImageNet-1K)

    * `dinov3/vitt16` â distillationv2 weights; recommended for dense tasks (object detection, segmentation)

    * `dinov3/vitt16plus` â distillationv2 weights; recommended for dense tasks

    * `dinov3/vitt16-distillationv1` â distillationv1 weights; recommended for global tasks (image classification)

    * `dinov3/vitt16plus-distillationv1` â distillationv1 weights; recommended for global tasks

    * `dinov3/vitt16-notpretrained` â random initialization; for training from scratch

    * `dinov3/vitt16plus-notpretrained` â random initialization; for training from scratch

  * ViT-T (Meta, LVD-1689M)

    * `dinov3/vitt16-eupe` \- [EUPE weights](https://github.com/facebookresearch/EUPE)

  * ViT-S (Meta, LVD-1689M)

    * `dinov3/vits16`

    * `dinov3/vits16-eupe` \- [EUPE weights](https://github.com/facebookresearch/EUPE)

    * `dinov3/vits16plus`

  * ViT-B (Meta, LVD-1689M)

    * `dinov3/vitb16`

    * `dinov3/vitb16-eupe` \- [EUPE weights](https://github.com/facebookresearch/EUPE)

  * ViT-L (Meta)

    * `dinov3/vitl16` (LVD-1689M)

    * `dinov3/vitl16-sat493m` (SAT-493M)

  * ViT-H (Meta, LVD-1689M)

    * `dinov3/vith16plus`

  * ViT-7B (Meta)

    * `dinov3/vit7b16` (LVD-1689M)

    * `dinov3/vit7b16-sat493m` (SAT-493M)




For ViT models, the patch size is encoded in the model name by default (for example, `vitt16` or `vits16`). You can still override it via `model_args["patch_size"]` when loading the model. If you provide `model_args["patch_size"]` for a ViT model whose name already encodes a patch size, LightlyTrain will log a warning. See the [Training Settings](settings__train_settings.md#train-settings) page for details.

### ConvNeXt ModelsÂ¶

The following ConvNeXt models are supported. All are [pretrained by Meta](https://github.com/facebookresearch/dinov3/tree/main?tab=readme-ov-file#pretrained-models) on the LVD-1689M dataset. The DINOv3 models are under the [DINOv3 license](https://github.com/facebookresearch/dinov3?tab=License-1-ov-file). The EUPE models are pretrained by Meta using the [EUPE method](https://github.com/facebookresearch/EUPE) and are under the [FAIR Noncommercial Research License](https://github.com/facebookresearch/EUPE?tab=License-1-ov-file).

  * `dinov3/convnext-tiny`

  * `dinov3/convnext-tiny-eupe` \- [EUPE weights](https://github.com/facebookresearch/EUPE)

  * `dinov3/convnext-small`

  * `dinov3/convnext-small-eupe` \- [EUPE weights](https://github.com/facebookresearch/EUPE)

  * `dinov3/convnext-base`

  * `dinov3/convnext-base-eupe` \- [EUPE weights](https://github.com/facebookresearch/EUPE)

  * `dinov3/convnext-large`



