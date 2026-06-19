---
source_url: https://docs.lightly.ai/train/stable/faq.html
---

# FAQÂ¶

## GeneralÂ¶

###### What is LightlyTrain?Â¶

LightlyTrain is a production-ready framework to pretrain computer vision models on unlabeled data. This allows you to get started with model training immediately as you donât have to wait for your data to get labeled and it significantly reduces the amount of labeled data needed to reach a high model performance. This decreases model deployment time and costs, and allows you to use your unlabeled data to its full potential.

Training a model with LightlyTrain only requires a single line of code. It is available as a Python package or Docker container and runs fully on-premise.

###### Who is LightlyTrain for?Â¶

LightlyTrain is designed for engineers and teams who want to use their unlabeled data to its full potential. It is ideal if any of the following applies to you:

  * You want to speedup model development cycles

  * You have limited labeled data but abundant unlabeled data

  * You have slow and expensive labeling processes

  * You want to build your own foundation model

  * You work with domain-specific datasets (video analytics, robotics, medical, agriculture, etc.)

  * You cannot use public pretrained models

  * No pretrained models are available for your specific architecture

  * You want to leverage the latest research in self-supervised learning and distillation




###### How do I install LightlyTrain?Â¶

You can install LightlyTrain using pip:
    
    
    pip install lightly-train
    

LightlyTrain is also available as a Docker container. For full details see the [installation page](installation.md#installation).

###### What are the system requirements?Â¶

LightlyTrain requires:

  * Linux (CUDA & CPU), MacOS (CPU only) or Windows (CUDA & CPU).

  * Python 3.8 or higher

  * PyTorch 2.1 or higher

  * CUDA-compatible GPU(s) for fast training

  * Sufficient storage for your dataset and model




LightlyTrain scales from single GPU setups to multi-GPU and multi-node configurations automatically, depending on your available hardware.

We recommend the following minimal setup for training:

  * NVIDIA GPU with 12GB+ memory

  * 16GB+ RAM

  * 50GB+ disk space for datasets and models




###### Is LightlyTrain free to use?Â¶

Lightly**Train** offers flexible licensing options to suit your specific needs:

  * **AGPL-3.0 License** : Perfect for open-source projects, academic research, and community contributions. Share your innovations with the world while benefiting from community improvements.

  * **Commercial License** : Ideal for businesses and organizations that need proprietary development freedom. Enjoy all the benefits of LightlyTrain while keeping your code and models private.




Weâre committed to supporting both open-source and commercial users. Please [contact us](https://www.lightly.ai/contact) to discuss the best licensing option for your project!

###### Whatâs the difference between LightlyTrain and LightlySSL?Â¶

  * **LightlyTrain** : A production-ready framework designed for engineers and teams who want to pretrain models with minimal code and configuration. It integrates directly with many popular model training libraries.

  * **LightlySSL** : A research-oriented package targeting self-supervised learning (SSL) researchers who want to modify and develop new SSL methods. It is not designed for production use and requires expertise in writing model training code.




###### Whatâs the difference between LightlyTrain and other open-source self-supervised learning implementations?Â¶

LightlyTrain offers several advantages over other self-supervised learning (SSL) implementations:

  * **User-friendly** : You donât need to be an SSL expert - focus on training your model instead of implementation details. Existing SSL frameworks require deep model training knowledge and focus on research instead of industry applications.

  * **Works with various model architectures** : Popular SSL frameworks are usually limited to few model architectures like ResNet or ViT. LightlyTrain directly integrates with different libraries such as Torchvision, Ultralytics, etc. and allows pretraining of their models out of the box.

  * **Handles complexity** : LightlyTrain manages many complexities around model training and SSL, such as scaling from single GPU to multi-GPU training and optimizing hyperparameters.

  * **Seamless workflow** : LightlyTrain automatically pretrains the correct layers and exports models in the right format for fine-tuning, reducing risks when moving from pretraining to fine-tuning.

  * **DINOv2 distillation** : Lightly has developed a unique distillation method that allows you to pretrain smaller models with the knowledge of larger DINOv2 models without the need for large compute resources.

  * **DINOv2 pretraining** : LightlyTrain supports DINOv2 pretraining out of the box, allowing you to train state-of-the-art vision foundation models on your own datasets.




## Capabilities & Use CasesÂ¶

###### Which tasks does LightlyTrain support?Â¶

LightlyTrain supports pretraining for various downstream computer vision tasks such as:

  * Image classification

  * Object detection

  * Semantic segmentation

  * Instance segmentation

  * Pose estimation

  * Depth estimation

  * Image embedding

  * Image retrieval




The framework is designed to create general-purpose visual representations that can be fine-tuned for any visual task, especially when you have limited labeled data or domain-specific requirements.

###### Which models does LightlyTrain support?Â¶

LightlyTrain supports the latest model architectures in computer vision such as:

  * YOLO (YOLOv5, YOLOv6, YOLOv8, YOLOv11, YOLOv12, YOLO26)

  * RF-DETR

  * RT-DETR

  * Vision Transformers

  * ResNets

  * Many others




LightlyTrain supports models from popular model training libraries:

  * Torchvision

  * TIMM

  * Ultralytics

  * RT-DETR

  * RF-DETR

  * YOLOv12

  * SuperGradients

  * Custom Models




Check the [Models](https://docs.lightly.ai/train/stable/pretrain_distill/models/index.html#models) documentation for details on specific models and their supported configurations.

###### Can I train an embedding model with LightlyTrain?Â¶

Yes, LightlyTrain is the perfect choice to train image embedding models. Its algorithms are optimized to learn strong image representations for tasks such as image visualization, clustering, and retrieval. With LightlyTrain you can train your own image embedding model on your unlabeled data. LightlyTrain also comes with an embedding function that lets you create image embeddings of your data.

###### Which data types does LightlyTrain support?Â¶

LightlyTrain currently supports training on images and video frames. It works with standard image formats such as JPG, PNG, etc. The framework handles image loading, preprocessing, and transformation automatically.

See [the documentation](https://docs.lightly.ai/train/stable/pretrain_distill/index.html#pretrain-data) for all the supported data formats.

###### Which datasets and domains does LightlyTrain support?Â¶

LightlyTrain is designed to work with any RGB image or video dataset and domain. It works especially well for domains such as:

  * Video Analytics

  * Robotics

  * ADAS

  * Agriculture

  * Medical Imaging

  * Visual Inspection




As LightlyTrain doesnât need labeled data it is particularly valuable in domains where labeled data is rare or expensive to get.

###### How much data do I need?Â¶

LightlyTrain supports use cases from thousands to millions of images. We recommend a minimum of a several thousand unlabeled images for training with LightlyTrain and 100+ labeled images for the fine-tuning afterwards. The larger the difference in dataset size between the unlabeled and labeled data, the larger the benefit of LightlyTrain. For best results we recommend at least 5x more unlabeled than labeled data. However, for most cases 2x more unlabeled than labeled data yields already strong improvements.

An example use case looks like this:

  * 100â000 unlabeled images

  * 10â000 labeled training images

  * 1â000 labeled validation images




The model is pretrained on the 100â000 unlabeled images with LightlyTrain. The pretrained model is then fine-tuned on the 10â000 labeled images with your favorite fine-tuning library and evaluated on the 1â000 validation images.

You can include the 10â000 labeled images in the unlabeled images for pretraining (it will make your model better). But do not include the validation images in the unlabeled data as this will lead to data leakage.

###### Can I train on labeled images?Â¶

Yes but LightlyTrain will ignore the labels. In fact, we recommend to add the training split of your labeled dataset to the dataset used for pretraining with LightlyTrain. This will make your model better. However, do not include the validation images in dataset used for pretraining with LightlyTrain as this leads to data leakage when you later evaluate the model on those images. The unlabeled dataset must always be treated like a training split and any images in the unlabeled set must not be used for evaluation.

###### Is there a limit on the number of images I can use?Â¶

No, LightlyTrain does not impose any limits on the number of images you can use for training. The framework is designed to scale to millions of images and you can use as many images as your hardware can handle. In fact, the training method in LightlyTrain performs better the more images you use as long as they are similar to the images you want to use the model on.

###### Can I pretrain a custom model on my own dataset?Â¶

Yes you can! LightlyTrain supports any model implemented in PyTorch. See the documentation on [custom models](pretrain_distill__models__custom_models.md#custom-models) on how to pretrain your model.

There are no restrictions on the dataset you use, except that it must contain images stored in a directory. See [the documentation](https://docs.lightly.ai/train/stable/pretrain_distill/index.html#pretrain-data) for all the supported images formats and the dataset structure.

###### How can I fine-tune a model?Â¶

The focus of LightlyTrain is on pretraining models on unlabeled data. It doesnât support fine-tuning. Instead, LightlyTrain integrates with popular fine-tuning libraries and supports pretraining models from those libraries out of the box. At the end of the pretraining, LightlyTrain exports the model in the correct format for the fine-tuning library which allows you to directly fine-tune the model without having to worry about converting and copying model weights.

See the documentation for a list of all [supported libraries](https://docs.lightly.ai/train/stable/pretrain_distill/models/index.html#models-supported-libraries).

###### How can I improve the performance of my model?Â¶

If your model performance is not satisfactory after fine-tuning consider the following options:

  * Check that the fine-tuning runs smoothly. Make sure there are no large spikes in the fine-tuning loss curves as those can negate the effect of the LightlyTrain pretraining. Pretrained models might need a lower fine-tuning learning rate than randomly initialized models.

  * Check that the pretraining runs smoothly. Make sure there are no large spikes in the pretraining loss curves. If you observe unstable training then lower the learning rate.

  * Increase the number of epochs for pretraining with LightlyTrain. Pretraining benefits a lot from long training schedules and doesnât suffer as much from overfitting as fine-tuning. Large datasets (>100â000 images) benefit from pretraining up to 3000 epochs. Smaller datasets (<100â000 images) benefit from even longer pretraining for up to 10â000 epochs.

  * Increase the number of unlabeled images in your pretraining dataset. The more unlabeled images you have the better. If you have more labeled than unlabeled data then pretraining is unlikely to help.

  * If pretraining and fine-tuning are stable and you donât observe improvements then increase the learning rate during pretraining. This is especially recommended for smaller models with ~10Mio or fewer parameters.




###### Do I have to tune hyperparameters?Â¶

No, LightlyTrain is designed to work well with the default hyperparameters for most use cases. If you want to customize specific aspects of the training, LightlyTrain provides configuration options through its API. However, most users can achieve good results without any manual hyperparameter tuning.

You might want to adjust the following parameters based on your hardware and training budget:

  * `batch_size`: We recommend a batch size between 128-1536.

  * `epochs`: We recommend to pretrain for 100-3000 epochs for large datasets (>100â000 images). For small datasets (<100â000 images) we recommend to pretrain up to 10â000 epochs.

  * `num_devices`: The number of GPUs to use for training. LightlyTrain automatically uses all available GPUs.

  * `precision`: We recommend training with âbf16-mixedâ precision for faster training speed.




Other parameters should only be adjusted for special use cases:

  * `learning_rate`: Change only if you want to further optimize model performance. Increase learning rate for small models.

  * `transform_args`: Change only if you need to customize the data augmentation pipeline in case you have a special dataset.




###### Why should I use LightlyTrain instead of other already pretrained models?Â¶

Pretrained models are often trained on curated, object-centric datasets that donât reflect the reality of real-world applications. In practice, these models can struggle when faced with raw, unlabeled data from production environments. Fortunately, LightlyTrain offers a scalable way to learn tailored representations directly from your own unlabeled data. Off-the-shelf models can still be a good starting point, and LightlyTrain fully supports leveraging existing pretrained weights. The benefit is that you can adapt representations to your domain without requiring any labels. Whether you start from scratch or fine-tune existing weights is entirely up to you.

LightlyTrain is most beneficial in these scenarios:

  * **Different data domains** : When working with data that has a very different distribution from the data existing pretrained models are trained on. For example, models pretrained on COCO or ImageNet often donât transfer well to medical images or industrial data. In these cases, LightlyTrain can help you pretrain a model that is better suited for your specific domain.

  * **Policies or license restrictions** : Models trained on popular datasets such as ImageNet have oftentimes unclear licensing policies, making it difficult to use them in production. If you are restricted by policies or licenses, LightlyTrain allows you to pretrain your own model without relying on existing pretrained models.

  * **Limited labeled data** : When you have access to a lot of unlabeled data but limited labeled data for your specific task.

  * **Custom architectures** : When using custom model architectures where no pretrained checkpoints are available.




For domains similar to COCO or natural images, supervised pretrained models like YOLO or Vision Transformers are very strong, and you may not see significant benefits from SSL pretraining.

## Technical BackgroundÂ¶

###### What is pretraining?Â¶

In the context of LightlyTrain, pretraining is the process of training a model on an unlabeled dataset to learn semantic representations of the images in the dataset. A pretrained model cannot classify or detect objects yet. But it has already a very good internal representation of the dataset and requires only a few labeled example images to become a strong model for a specific task.

Pretrained model serves as a much better starting point for fine-tuning than randomly initialized weights, typically leading to:

  * Higher accuracy

  * Faster convergence

  * Better generalization

  * Less labeled data needed for downstream tasks




###### What is fine-tuning?Â¶

Fine-tuning is the process of taking a pretrained model and further training it on a specific task with labeled data. The pretrained model already has learned useful visual features, and fine-tuning adapts these features to a specific task such as classification, object detection, or segmentation.

With LightlyTrain, you first pretrain your model on unlabeled data, then fine-tune it on a smaller labeled dataset for your specific task.

###### What is the difference between pretraining and fine-tuning?Â¶

Pretraining and fine-tuning are two distinct phases in the model training process:

**Pretraining:**

  * Uses unlabeled data

  * Usually performed on larger datasets

  * Uses self-supervised learning

  * Resulting model is general-purpose




**Fine-tuning:**

  * Uses labeled data for a specific task

  * Can be performed on smaller datasets

  * Uses supervised learning

  * Resulting model is task specific (classification, object detection, etc.)




Pretraining happens before fine-tuning and is essential for learning useful representations from unlabeled data. LightlyTrain focuses on the pretraining phase, while fine-tuning is typically done using standard supervised learning frameworks appropriate for your downstream task.

###### Which pretraining methods are supported?Â¶

LightlyTrain supports different methods such as:

  * DINOv2 distillation

  * DINOv2

  * DINO

  * SimCLR




See [the documentation on methods](https://docs.lightly.ai/train/stable/pretrain_distill/methods/index.html#methods) for more information.

###### Will you add more pretraining methods?Â¶

Yes, we will add more pretraining methods in the future. That being said, there are hundreds of different pretraining methods and we will not add all of them. Instead, the goal of LightlyTrain is to focus on the best pretraining methods for industry relevant tasks.

###### What is self-supervised learning (SSL)?Â¶

Self-supervised learning (SSL) is a machine learning approach where models learn useful representations from unlabeled data without requiring manual annotations. Instead of being trained on explicit labels, SSL methods create âproxy tasksâ where the model learns by solving objectives derived from the data itself.

In computer vision, SSL typically involves training models to match different views of the same image, predict masked regions, or understand relationships between different parts of images. This allows models to develop rich, general-purpose representations that can be fine-tuned for downstream tasks with much less labeled data.

###### What is distillation?Â¶

In the context of LightlyTrain, distillation refers to knowledge distillation where a teacher model guides the training of your student model.

During distillation, the student model learns to produce similar feature representations as the teacher model for the same input images. This allows smaller models to benefit from the knowledge of larger, more powerful models like vision transformers pretrained with DINOv2, without requiring the same computational resources for training.

###### What is DINOv2?Â¶

DINOv2 is a self-supervised learning method developed by Facebook AI Research (FAIR) that produces state-of-the-art visual representations. It builds upon the original DINO (Self-Distillation with No Labels) method and uses a Vision Transformer (ViT) architecture.

DINOv2 models are pretrained on a diverse set of image data and have shown remarkable performance across various visual tasks, from classification to dense prediction tasks. The resulting representations exhibit strong semantic understanding and work well for zero-shot and few-shot learning scenarios.

In LightlyTrain, DINOv2-pretrained ViT models serve as teacher models for the distillation method, transferring their knowledge to your chosen backbone architecture.

## DeploymentÂ¶

###### Does LightlyTrain run on the cloud?Â¶

Lightly does not offer a service to train models with LightlyTrain in the cloud. However, you can run the LightlyTrain Python package on your own cloud instances.

###### Does LightlyTrain need an internet connection?Â¶

LightlyTrain does not require an internet connection to run. However, it may need to download model weights on the first run. All subsequent runs will use the cached weights. The model weights can also be downloaded beforehand for deployments without an internet connection.

LightlyTrain does not send any telemetry data and does not require authetification over the internet.

###### How many GPUs do I need?Â¶

LightlyTrain works well with a single GPU. Multi-GPU and multi-node training is supported for faster training speeds. See our [documentation](https://docs.lightly.ai/train/stable/performance/index.html#performance) for more information.

###### Can I train on multiple GPUs and Nodes?Â¶

Yes, LightlyTrain supports multi-GPU and multi-node training. LightlyTrain automatically trains on multiple GPUs if available. By default it uses only a single node. You can control the number of GPUs and nodes with:

  * `num_devices`: The number of GPUs to use for training.

  * `num_nodes`: The number of nodes to use for training.




See our [documentation](https://docs.lightly.ai/train/stable/performance/index.html#performance) for more information.

## Support & ResourcesÂ¶

###### How can I get a license?Â¶

Please [contact us](https://www.lightly.ai/contact) to discuss the best licensing option for your project!

###### How can I get support?Â¶

If you need help with LightlyTrain, you have several options:

  1. **Documentation** : Check the [official documentation](https://docs.lightly.ai/train) for guides and reference material.

  2. **Discord Community** : Join our [Discord server](https://discord.gg/xvNJW94) to ask questions and interact with other users.

  3. **Email Support** : Contact us at [sales@lightly.ai](mailto:sales%40lightly.ai) for licensing questions.

  4. **GitHub Issues** : Report bugs or request features on [GitHub](https://github.com/lightly-ai/lightly-train/issues).




For commercial users with a license, additional support options are available.

###### How can I report a bug?Â¶

Report bugs on our issue tracker on [GitHub](https://github.com/lightly-ai/lightly-train/issues).

###### How can I request a feature?Â¶

Request features on our issue tracker on [GitHub](https://github.com/lightly-ai/lightly-train/issues).

###### How can I contribute to LightlyTrain?Â¶

We welcome contributions to LightlyTrain!

To contribute, please head to our [GitHub repo](https://github.com/lightly-ai/lightly-train) and join our [Discord](https://discord.gg/xvNJW94) for any questions.
  *[*]: Keyword-only parameters separator (PEP 3102)
