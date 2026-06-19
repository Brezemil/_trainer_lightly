---
source_url: https://docs.lightly.ai/train/stable/pretrain_distill/methods/simclr.html
---

# SimCLRÂ¶

[SimCLR](https://arxiv.org/abs/2002.05709) is a self-supervised learning method that employs contrastive learning. More specifically, it enforces similarity between the representations of two augmented views of the same image and dissimilarity w.r.t. to the other instances in the batch. Using strong data augmentations and large batch sizes, it achieves classification performance on ImageNet-1k that is comparable to that of supervised learning approaches.

## Use SimCLR in LightlyTrainÂ¶

Python
    
    
    import lightly_train
    
    if __name__ == "__main__":
        lightly_train.pretrain(
            out="out/my_experiment", 
            data="my_data_dir",
            model="torchvision/resnet18",
            method="simclr",
        )
    

Command Line
    
    
    lightly-train pretrain out=out/my_experiment data=my_data_dir model="torchvision/resnet18" method="simclr"
    

## Whatâs under the HoodÂ¶

SimCLR learns representations by creating two augmented views of the same imageâusing techniques like random cropping, resizing, color jittering, and Gaussian blurâand then training the model to maximize agreement between these augmented views while distinguishing them from other images. It employs the normalized temperature-scaled cross-entropy loss (NT-Xent) to encourage similar pairs to align and dissimilar pairs to diverge. The method benefits from large batch sizes, enabling it to achieve performance comparable to supervised learning on benchmarks like ImageNet-1k.

## Lightly RecommendationsÂ¶

  * **Models** : SimCLR is specifically optimized for convolutional neural networks, with a focus on ResNet architectures. Using transformer-based models is doable but less common.

  * **Batch Size** : We recommend a minimum of 256, though somewhere between 1024 and 4096 is ideal since SimCLR usually benefits from large batch sizes.

  * **Number of Epochs** : We recommend a minimum of 800 epochs based on the top-5 linear evaluation results using ResNet-50 on ImageNet-1k reported by the original paper. The top-1 results continues to increase even after 3200 epochs. Also, using a large number of epochs compensates for using a relatively smaller batch size.




## Default Method ArgumentsÂ¶

The following are the default method arguments for SimCLR. To learn how you can override these settings, see [Method Arguments](https://docs.lightly.ai/train/stable/pretrain_distill/index.html#method-args).

Default Method Arguments
    
    
    {
        "batch_norm": true,
        "hidden_dim": 2048,
        "lr_scale_method": "linear",
        "num_layers": 2,
        "output_dim": 128,
        "reference_batch_size": 256,
        "temperature": 0.1
    }
    

## Default Image Transform ArgumentsÂ¶

The following are the default transform arguments for SimCLR. To learn how you can override these settings, see [Transforms](settings__pretrain_settings.md#method-transform-args).

Default Image Transforms
    
    
    {
        "channel_drop": null,
        "color_jitter": {
            "brightness": 0.8,
            "contrast": 0.8,
            "hue": 0.2,
            "prob": 0.8,
            "saturation": 0.8,
            "strength": 1.0
        },
        "gaussian_blur": {
            "blur_limit": 0,
            "prob": 0.5,
            "sigmas": [
                0.1,
                2.0
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
            "min_scale": 0.08
        },
        "random_rotation": null,
        "solarize": null
    }
    
