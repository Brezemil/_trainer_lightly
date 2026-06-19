---
source_url: https://docs.lightly.ai/train/stable/pretrain_distill/models/dinov2.html
---

# DINOv2Â¶

This page describes how to use DINOv2 models with LightlyTrain.

[DINOv2](https://github.com/facebookresearch/dinov2) models are Vision Transformers (ViTs) pretrained by Meta using the DINOv2 self-supervised learning method on large image datasets. They serve as high-quality feature extractors and strong backbones for downstream tasks such as object detection, segmentation, and image classification.

Note

DINOv2 models are released under the [Apache 2.0 license](https://github.com/facebookresearch/dinov2/blob/main/LICENSE).

## Pretrain and Fine-tune a DINOv2 ModelÂ¶

### PretrainÂ¶

DINOv2 models can be pretrained from scratch or starting from Metaâs pretrained weights using the [DINOv2 method](pretrain_distill__methods__dinov2.md#methods-dinov2). Below we provide the minimum scripts using `dinov2/vitb14` as an example:

Python
    
    
    import lightly_train
    
    if __name__ == "__main__":
        lightly_train.pretrain(
            out="out/my_experiment",                # Output directory.
            data="my_data_dir",                     # Directory with images.
            model="dinov2/vitb14",                  # Pass the DINOv2 model.
            method="dinov2",                        # Use the DINOv2 pretraining method.
        )
    

Command Line
    
    
    lightly-train pretrain out="out/my_experiment" data="my_data_dir" model="dinov2/vitb14" method="dinov2"
    

See [DINOv2 method](pretrain_distill__methods__dinov2.md#methods-dinov2) for more details on the pretraining method and its configuration options.

### Fine-tuneÂ¶

After pretraining, the exported DINOv2 backbone can be loaded into downstream task models via the `backbone_weights` argument. Refer to the following pages for fine-tuning instructions and example code:

  * [Object Detection](object_detection.md#object-detection-pretrain-finetune) â fine-tune a DINOv2-based LTDETR model; supports loading custom pretrained backbone weights via `backbone_weights`.

  * [Semantic Segmentation](semantic_segmentation.md#semantic-segmentation-pretrain-finetune) â fine-tune a DINOv2-based EoMT model; supports loading custom pretrained backbone weights via `backbone_weights`.

  * [Instance Segmentation](instance_segmentation.md#instance-segmentation) â fine-tune a DINOv2-based EoMT model.

  * [Panoptic Segmentation](panoptic_segmentation.md#panoptic-segmentation) â fine-tune a DINOv2-based EoMT model.

  * [Image Classification](image_classification.md#image-classification) â fine-tune a DINOv2 backbone for classification.




## Supported ModelsÂ¶

### Pretrained ModelsÂ¶

The following models are [pretrained by Meta](https://github.com/facebookresearch/dinov2?tab=readme-ov-file#pretrained-models) and loaded automatically when used.

  * `dinov2/vits14`

  * `dinov2/vitb14`

  * `dinov2/vitl14`

  * `dinov2/vitg14`




### Not Pretrained ModelsÂ¶

The following models start from random initialization and are useful when pretraining from scratch with the [DINOv2 method](pretrain_distill__methods__dinov2.md#methods-dinov2) on a custom dataset without starting from Metaâs weights.

  * `dinov2/vits14-notpretrained`

  * `dinov2/vitb14-notpretrained`

  * `dinov2/vitl14-notpretrained`

  * `dinov2/vitg14-notpretrained`



