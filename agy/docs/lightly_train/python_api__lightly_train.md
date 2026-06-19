---
source_url: https://docs.lightly.ai/train/stable/python_api/lightly_train.html
---

# lightly_trainÂ¶

Documentation of the public API of the `lightly_train` package.

## FunctionsÂ¶

lightly_train.embed(_*_ , _out : str | Path_, _data : str | Path | Sequence[str | Path]_, _checkpoint : str | Path_, _format : str | EmbeddingFormat = 'torch'_, _image_size : int | tuple[int, int] = (224, 224)_, _batch_size : int = 128_, _num_workers : int | Literal['auto'] = 'auto'_, _accelerator : str | Accelerator = 'auto'_, _overwrite : bool = False_, _precision : Literal[64, 32, 16, 'transformer-engine', 'transformer-engine-float16', '16-true', '16-mixed', 'bf16-true', 'bf16-mixed', '32-true', '64-true', '64', '32', '16', 'bf16'] = '32-true'_) → NoneÂ¶
    

Embed images from a model checkpoint.

See the documentation for more information: <https://docs.lightly.ai/train/stable/embed.html>

Args:
    

out:
    

Filepath where the embeddings will be saved. For example âembeddings.csvâ.

data:
    

Directory containing the images to embed or a sequence of image directories and files.

checkpoint:
    

Path to the LightlyTrain checkpoint file used for embedding. The location of the checkpoint depends on the train command. If training was run with `out="out/my_experiment"`, then the last LightlyTrain checkpoint is saved to `out/my_experiment/checkpoints/last.ckpt`.

format:
    

Format of the embeddings. Supported formats are [âcsvâ, âlightly_csvâ, âtorchâ]. âtorchâ is the recommended and most efficient format. Torch embeddings can be loaded with `torch.load(out, weigths_only=True)`. Choose âlightly_csvâ if you want to use the embeddings as custom embeddings with the Lightly Worker.

image_size:
    

Size to which the images are resized before embedding. If a single integer is provided, the image is resized to a square with the given side length. If a (height, width) tuple is provided, the image is resized to the given height and width. Note that not all models support all image sizes.

batch_size:
    

Number of images per batch.

num_workers:
    

Number of workers for the dataloader. âautoâ automatically sets the number of workers based on the available CPU cores.

accelerator:
    

Hardware accelerator. Can be one of [âcpuâ, âgpuâ, âtpuâ, âipuâ, âhpuâ, âmpsâ, âautoâ]. âautoâ will automatically select the best accelerator available.

overwrite:
    

Overwrite the output file if it already exists.

precision:
    

Embedding precision. Select â32-trueâ for full 32-bit precision, or âbf16-mixedâ/â16-mixedâ for mixed precision.

lightly_train.export(_*_ , _out : str | Path_, _checkpoint : str | Path_, _part : str | ModelPart = 'model'_, _format : str | ModelFormat = 'package_default'_, _overwrite : bool = False_) → NoneÂ¶
    

Export a model from a checkpoint.

See the documentation for more information: <https://docs.lightly.ai/train/stable/pretrain_distill/export.html>

Args:
    

out:
    

Path where the exported model will be saved.

checkpoint:
    

Path to the LightlyTrain checkpoint file to export the model from. The location of the checkpoint depends on the train command. If training was run with `out="out/my_experiment"`, then the last LightlyTrain checkpoint is saved to `out/my_experiment/checkpoints/last.ckpt`.

part:
    

Part of the model to export. Valid options are âmodelâ and âembedding_modelâ. âmodelâ is the default option and exports the model that was passed as `model` argument to the train function. âembedding_modelâ exports the embedding model. This includes the model passed with the model argument in the train function and an extra embedding layer if the `embed_dim` argument was set during training. This is useful if you want to use the exported model for embedding images.

format:
    

Format to save the model in. Valid options are [âpackage_defaultâ, âtorch_modelâ, âtorch_state_dictâ]. âpackage_defaultâ is the default option and exports the model in the default format of the package that was used for training. This ensures compatibility with the package and is the most flexible option. âtorch_state_dictâ exports the modelâs state dict which can be loaded with model.load_state_dict(torch.load(out, weights_only=True)). âtorch_modelâ exports the model as a torch module which can be loaded with model = torch.load(out). This requires that the same LightlyTrain version is installed when the model is exported and when it is loaded again.

overwrite:
    

Overwrite the output file if it already exists.

lightly_train.export_onnx(_*_ , _out : str | Path_, _checkpoint : str | Path_, _batch_size : int = 1_, _height : int | None = None_, _width : int | None = None_, _precision : Literal['32-true', '16-true'] = '32-true'_, _simplify : bool = True_, _verify : bool = True_, _overwrite : bool = False_, _format_args : dict[str, Any] | None = None_) → NoneÂ¶
    

Export a model as ONNX from a checkpoint.

Args:
    

out:
    

Path where the exported model will be saved.

checkpoint:
    

Path to the LightlyTrain checkpoint file to export the model from.

batch_size:
    

Batch size of the input tensor.

height:
    

Height of the input tensor.

width:
    

Width of the input tensor.

precision:
    

â32-trueâ for float32 precision or â16-trueâ for float16 precision. Choosing â16-trueâ can lead to less memory consumption and faster inference times on GPUs but might lead to slightly more inaccuracies. Default is â32-trueâ.

simplify:
    

Simplify the ONNX model with onnxslim after the export. Default is True.

verify:
    

Check the exported model for errors. With recommend to enable this.

overwrite:
    

Overwrite the output file if it already exists.

format_args:
    

Arguments that are passed to torch.onnx.export. Only use this if you know what you are doing.

lightly_train.list_methods() → list[str]Â¶
    

Lists all available self-supervised learning methods.

See the documentation for more information: <https://docs.lightly.ai/train/stable/pretrain_distill/methods/index.html>

lightly_train.list_models() → list[str]Â¶
    

Lists all models in `<package_name>/<model_name>` format.

See the documentation for more information: <https://docs.lightly.ai/train/stable/pretrain_distill/models/>

lightly_train.load_model(_model : str | Path_, _device : Literal['cpu', 'cuda', 'mps'] | device | None = None_) → TaskModelÂ¶
    

Either load model from an exported model file (in .pt format) or a checkpoint file (in .ckpt format) or download it from the Lightly model repository.

First check if model points to a valid file. If not and model is a str try to match that name to one of the models in the Lightly model repository and download it. Downloaded models are cached under the location specified by the environment variable LIGHTLY_TRAIN_MODEL_CACHE_DIR.

Args:
    

model:
    

Either a path to the exported model/checkpoint file or the name of a model in the Lightly model repository.

device:
    

Device to load the model on. If None, the model will be loaded onto a GPU (âcudaâ or âmpsâ) if available, and otherwise fall back to CPU.

Returns:
    

The loaded model.

lightly_train.pretrain(_*_ , _out : str | Path_, _data : str | Path | Sequence[str | Path]_, _model : str | Module | ModelWrapper | Any_, _method : str = 'distillation'_, _method_args : dict[str, Any] | None = None_, _embed_dim : int | None = None_, _epochs : int | Literal['auto'] = 'auto'_, _batch_size : int = 128_, _num_workers : int | Literal['auto'] = 'auto'_, _devices : int | str | list[int] = 'auto'_, _num_nodes : int = 1_, _resume_interrupted : bool = False_, _checkpoint : str | Path | None = None_, _overwrite : bool = False_, _accelerator : str | Accelerator = 'auto'_, _strategy : str | Strategy = 'auto'_, _precision : Literal[64, 32, 16, 'transformer-engine', 'transformer-engine-float16', '16-true', '16-mixed', 'bf16-true', 'bf16-mixed', '32-true', '64-true', '64', '32', '16', 'bf16', 'auto'] = 'auto'_, _float32_matmul_precision : Literal['auto', 'highest', 'high', 'medium'] = 'auto'_, _seed : int = 0_, _loggers : dict[str, dict[str, Any] | None] | None = None_, _callbacks : dict[str, dict[str, Any] | None] | None = None_, _optim : str = 'auto'_, _optim_args : dict[str, Any] | None = None_, _transform_args : dict[str, Any] | None = None_, _loader_args : dict[str, Any] | None = None_, _trainer_args : dict[str, Any] | None = None_, _model_args : dict[str, Any] | None = None_, _resume : bool | None = None_) → NoneÂ¶
    

Pretrain a self-supervised model.

See the documentation for more information: <https://docs.lightly.ai/train/stable/pretrain_distill.html>

The pretraining process can be monitored with TensorBoard:
    
    
    tensorboard --logdir out
    

After pretraining, the model is exported in the library default format to `out/exported_models/exported_last.pt`. It can be exported to different formats using the `lightly_train.export` command.

Args:
    

out:
    

Output directory to save logs, checkpoints, and other artifacts.

data:
    

Path to a directory containing images or a sequence of image directories and files.

model:
    

Model name or instance to use for pretraining / distillation.

method:
    

Method name for pretraining / distillation.

method_args:
    

Arguments for the pretraining / distillation method. The available arguments depend on the `method` parameter.

embed_dim:
    

Embedding dimension. Set this if you want to pretrain an embedding model with a specific dimension. If None, the output dimension of `model` is used.

epochs:
    

Number of training epochs. Set to âautoâ to automatically determine the number of epochs based on the dataset size and batch size.

batch_size:
    

Global batch size. The batch size per device/GPU is inferred from this value and the number of devices and nodes.

num_workers:
    

Number of workers for the dataloader per device/GPU. âautoâ automatically sets the number of workers based on the available CPU cores.

devices:
    

Number of devices/GPUs for training. âautoâ automatically selects all available devices. The device type is determined by the `accelerator` parameter.

num_nodes:
    

Number of nodes for distributed training.

checkpoint:
    

Use this parameter to further pretrain a model from a previous run. The checkpoint must be a path to a checkpoint file created by a previous training run, for example âout/my_experiment/checkpoints/last.ckptâ. This will only load the model weights from the previous run. All other training state (e.g. optimizer state, epochs) from the previous run are not loaded. Instead, a new run is started with the model weights from the checkpoint.

If you want to resume training from an interrupted or crashed run, use the `resume_interrupted` parameter instead. See <https://docs.lightly.ai/train/stable/pretrain_distill/index.html#resume-training> for more information.

resume_interrupted:
    

Set this to True if you want to resume training from an **interrupted or crashed** training run. This will pick up exactly where the training left off, including the optimizer state and the current epoch.

  * You must use the same `out` directory as the interrupted run.

  * You must **NOT** change any training parameters (e.g., learning rate, batch size, data, etc.).

  * This is intended for continuing the same run without modification.




If you want to further pretrain a model or change the training parameters, use the `checkpoint` parameter instead. See <https://docs.lightly.ai/train/stable/pretrain_distill/index.html#resume-training> for more information.

overwrite:
    

Overwrite the output directory if it already exists. Warning, this might overwrite existing files in the directory!

accelerator:
    

Hardware accelerator. Can be one of [âcpuâ, âgpuâ, âtpuâ, âipuâ, âhpuâ, âmpsâ, âautoâ]. âautoâ will automatically select the best accelerator available.

strategy:
    

Training strategy. For example âddpâ or âautoâ. âautoâ automatically selects the best strategy available.

precision:
    

Training precision. Select â16-mixedâ for mixed 16-bit precision, â32-trueâ for full 32-bit precision, or âbf16-mixedâ for mixed bfloat16 precision.

float32_matmul_precision:
    

Precision for float32 matrix multiplication. Can be one of [âautoâ, âhighestâ, âhighâ, âmediumâ]. See <https://docs.pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision> for more information.

seed:
    

Random seed for reproducibility.

loggers:
    

Loggers for training. Either None or a dictionary of logger names to either None or a dictionary of logger arguments. None uses the default loggers. To disable a logger, set it to None: `loggers={"tensorboard": None}`. To configure a logger, pass the respective arguments: `loggers={"wandb": {"project": "my_project"}}`.

callbacks:
    

Callbacks for training. Either None or a dictionary of callback names to either None or a dictionary of callback arguments. None uses the default callbacks. To disable a callback, set it to None: `callbacks={"model_checkpoint": None}`. To configure a callback, pass the respective arguments: `callbacks={"model_checkpoint": {"every_n_epochs": 5}}`.

optim:
    

Optimizer name. Must be one of [âautoâ, âadamwâ, âsgdâ]. âautoâ automatically selects the optimizer based on the method.

optim_args:
    

Optimizer arguments. Available arguments depend on the optimizer.

AdamW:
    

`optim_args={"lr": float, "betas": (float, float), "weight_decay": float}`

SGD:
    

`optim_args={"lr": float, "momentum": float, "weight_decay": float}`

transform_args:
    

Arguments for the image transform. The available arguments depend on the method parameter. The following arguments are always available:
    
    
    transform_args={
        "image_size": (int, int),
        "random_resize": {
            "min_scale": float,
            "max_scale": float,
        },
        "random_flip": {
            "horizonal_prob": float,
            "vertical_prob": float,
        },
        "random_rotation": {
            "prob": float,
            "degrees": int,
        },
        "random_gray_scale": float,
        "normalize": {
            "mean": (float, float, float),
            "std": (float, float, float),
        }
    }
    

loader_args:
    

Arguments for the PyTorch DataLoader. Should only be used in special cases as default values are automatically set. Prefer to use the batch_size and num_workers arguments instead. For details, see: <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>

trainer_args:
    

Arguments for the PyTorch Lightning Trainer. Should only be used in special cases as default values are automatically set. For details, see: <https://lightning.ai/docs/pytorch/stable/common/trainer.html>

model_args:
    

Arguments for the model. The available arguments depend on the `model` parameter. For example, if `model='torchvision/<model_name>'`, the arguments are passed to `torchvision.models.get_model(model_name, **model_args)`.

resume:
    

Deprecated. Use `resume_interrupted` instead.

lightly_train.train(_*_ , _out : str | Path_, _data : str | Path | Sequence[str | Path]_, _model : str | Module | ModelWrapper | Any_, _method : str = 'distillation'_, _method_args : dict[str, Any] | None = None_, _embed_dim : int | None = None_, _epochs : int | Literal['auto'] = 'auto'_, _batch_size : int = 128_, _num_workers : int | Literal['auto'] = 'auto'_, _devices : int | str | list[int] = 'auto'_, _num_nodes : int = 1_, _resume_interrupted : bool = False_, _checkpoint : str | Path | None = None_, _overwrite : bool = False_, _accelerator : str | Accelerator = 'auto'_, _strategy : str | Strategy = 'auto'_, _precision : Literal[64, 32, 16, 'transformer-engine', 'transformer-engine-float16', '16-true', '16-mixed', 'bf16-true', 'bf16-mixed', '32-true', '64-true', '64', '32', '16', 'bf16', 'auto'] = 'auto'_, _float32_matmul_precision : Literal['auto', 'highest', 'high', 'medium'] = 'auto'_, _seed : int = 0_, _loggers : dict[str, dict[str, Any] | None] | None = None_, _callbacks : dict[str, dict[str, Any] | None] | None = None_, _optim : str = 'auto'_, _optim_args : dict[str, Any] | None = None_, _transform_args : dict[str, Any] | None = None_, _loader_args : dict[str, Any] | None = None_, _trainer_args : dict[str, Any] | None = None_, _model_args : dict[str, Any] | None = None_, _resume : bool | None = None_) → NoneÂ¶
    

Deprecated. Use `pretrain()` instead.

lightly_train.train_image_classification(_*_ , _out : str | Path_, _data : dict[str, Any] | str_, _model : str_, _classification_task : Literal['multiclass', 'multilabel'] = 'multiclass'_, _steps : int | Literal['auto'] = 'auto'_, _batch_size : int | Literal['auto'] = 'auto'_, _num_workers : int | Literal['auto'] = 'auto'_, _devices : int | str | list[int] = 'auto'_, _num_nodes : int = 1_, _resume_interrupted : bool = False_, _checkpoint : str | Path | None = None_, _reuse_class_head : bool = False_, _overwrite : bool = False_, _accelerator : str = 'auto'_, _strategy : str = 'auto'_, _precision : Literal[64, 32, 16, 'transformer-engine', 'transformer-engine-float16', '16-true', '16-mixed', 'bf16-true', 'bf16-mixed', '32-true', '64-true', '64', '32', '16', 'bf16'] = 'bf16-mixed'_, _float32_matmul_precision : Literal['auto', 'highest', 'high', 'medium'] = 'auto'_, _seed : int = 0_, _logger_args : dict[str, Any] | None = None_, _model_args : dict[str, Any] | None = None_, _transform_args : dict[str, Any] | None = None_, _metric_args : dict[str, Any] | None = None_, _loader_args : dict[str, Any] | None = None_, _save_checkpoint_args : dict[str, Any] | None = None_, _torch_compile_args : dict[str, Any] | None = None_, _gradient_accumulation_steps : int | Literal['auto'] = 'auto'_) → NoneÂ¶
    

Train an image classification model.

See the documentation for more information: <https://docs.lightly.ai/train/stable/image_classification.html>

> The training process can be monitored with TensorBoard:
    
    
    tensorboard --logdir out
    

After training, the last model checkpoint is saved in the out directory to: `out/checkpoints/last.ckpt` and also exported to `out/exported_models/exported_last.pt`.

Args:
    

out:
    

The output directory where the model checkpoints and logs are saved.

data:
    

The dataset configuration or path to a YAML file with the configuration. See the documentation for more information: <https://docs.lightly.ai/train/stable/image_classification.html#data>

model:
    

The model to train. For example, âdinov3/vitt16â, or a path to a local model checkpoint.

If you want to resume training from an interrupted or crashed run, use the `resume_interrupted` parameter.

classification_task:
    

The type of image classification task. Can be either âmulticlassâ or âmultilabelâ. In a multiclass classification task, each image is assigned to exactly one class. In a multilabel classification task, each image can be assigned to multiple classes.

steps:
    

The number of training steps.

batch_size:
    

Global batch size. The batch size per device/GPU is inferred from this value and the number of devices and nodes.

num_workers:
    

Number of workers for the dataloader per device/GPU. âautoâ automatically sets the number of workers based on the available CPU cores.

devices:
    

Number of devices/GPUs for training. âautoâ automatically selects all available devices. The device type is determined by the `accelerator` parameter.

num_nodes:
    

Number of nodes for distributed training.

checkpoint:
    

Use this parameter to further fine-tune a model from a previous fine-tuned checkpoint. The checkpoint must be a path to a checkpoint file, for example âcheckpoints/model.ckptâ. This will only load the model weights from the previous run. All other training state (e.g. optimizer state, epochs) from the previous run are not loaded.

This option is equivalent to setting `model="<path_to_checkpoint>"`.

If you want to resume training from an interrupted or crashed run, use the `resume_interrupted` parameter instead.

resume_interrupted:
    

Set this to True if you want to resume training from an **interrupted or crashed** training run. This will pick up exactly where the training left off, including the optimizer state and the current step.

  * You must use the same `out` directory as the interrupted run.

  * You must **NOT** change any training parameters (e.g., learning rate, batch size, data, etc.).

  * This is intended for continuing the same run without modification.



overwrite:
    

Overwrite the output directory if it already exists. Warning, this might overwrite existing files in the directory!

accelerator:
    

Hardware accelerator. Can be one of [âcpuâ, âgpuâ, âmpsâ, âautoâ]. âautoâ will automatically select the best accelerator available.

strategy:
    

Training strategy. For example âddpâ or âautoâ. âautoâ automatically selects the best strategy available.

precision:
    

Training precision. Select â16-mixedâ for mixed 16-bit precision, â32-trueâ for full 32-bit precision, or âbf16-mixedâ for mixed bfloat16 precision.

float32_matmul_precision:
    

Precision for float32 matrix multiplication. Can be one of [âautoâ, âhighestâ, âhighâ, âmediumâ]. See <https://docs.pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision> for more information.

seed:
    

Random seed for reproducibility.

logger_args:
    

Logger arguments. Either None or a dictionary of logger names to either None or a dictionary of logger arguments. None uses the default loggers. To disable a logger, set it to None: `logger_args={"tensorboard": None}`. To configure a logger, pass the respective arguments: `logger_args={"mlflow": {"experiment_name": "my_experiment", ...}}`. See <https://docs.lightly.ai/train/stable/image_classification.html#logging> for more information.

model_args:
    

Model training arguments. Either None or a dictionary of model arguments.

transform_args:
    

Transform arguments. Either None or a dictionary of transform arguments. The image size and normalization parameters can be set with `transform_args={"image_size": (height, width), "normalize": {"mean": (r, g, b), "std": (r, g, b)}}`

metric_args:
    

Metric arguments. Either None or a dictionary of metric arguments. Set `metric_args={"train": True}` to also log metrics on training data. Set `metric_args={"classwise": True}` to log per-class metrics. Set `metric_args={"watch_metric": "val_metric/top1_acc_micro"}` to configure the metric used to select the best checkpoint.

loader_args:
    

Arguments for the PyTorch DataLoader. Should only be used in special cases as default values are automatically set. Prefer to use the batch_size and num_workers arguments instead. For details, see: <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>

save_checkpoint_args:
    

Arguments to configure the saving of checkpoints. The checkpoint frequency can be set with `save_checkpoint_args={"save_every_num_steps": 100}`.

torch_compile_args:
    

Arguments to configure model compilation with torch.compile. The arguments are directly passed to torch.compile. Model compilation is disabled for most models by default. Set `torch_compile_args={"disable": True}` to disable it if you encounter any issues.

gradient_accumulation_steps:
    

Number of gradient accumulation steps. âautoâ automatically enables gradient accumulation when batch_size is smaller than the modelâs default batch size, using `max(1, default_batch_size // batch_size)` steps to keep the effective batch size and learning rate close to the model defaults. Set to 1 to explicitly disable gradient accumulation.

lightly_train.train_instance_segmentation(_*_ , _out : str | Path_, _data : dict[str, Any] | str_, _model : str_, _steps : int | Literal['auto'] = 'auto'_, _batch_size : int | Literal['auto'] = 'auto'_, _num_workers : int | Literal['auto'] = 'auto'_, _devices : int | str | list[int] = 'auto'_, _num_nodes : int = 1_, _resume_interrupted : bool = False_, _checkpoint : str | Path | None = None_, _reuse_class_head : bool = False_, _overwrite : bool = False_, _accelerator : str = 'auto'_, _strategy : str = 'auto'_, _precision : Literal[64, 32, 16, 'transformer-engine', 'transformer-engine-float16', '16-true', '16-mixed', 'bf16-true', 'bf16-mixed', '32-true', '64-true', '64', '32', '16', 'bf16'] = 'bf16-mixed'_, _float32_matmul_precision : Literal['auto', 'highest', 'high', 'medium'] = 'auto'_, _seed : int = 0_, _logger_args : dict[str, Any] | None = None_, _model_args : dict[str, Any] | None = None_, _transform_args : dict[str, Any] | None = None_, _metric_args : dict[str, Any] | None = None_, _loader_args : dict[str, Any] | None = None_, _save_checkpoint_args : dict[str, Any] | None = None_, _torch_compile_args : dict[str, Any] | None = None_, _gradient_accumulation_steps : int | Literal['auto'] = 'auto'_) → NoneÂ¶
    

Train an instance segmentation model.

See the documentation for more information: <https://docs.lightly.ai/train/stable/instance_segmentation.html>

> The training process can be monitored with TensorBoard:
    
    
    tensorboard --logdir out
    

After training, the last model checkpoint is saved in the out directory to: `out/checkpoints/last.ckpt` and also exported to `out/exported_models/exported_last.pt`.

Args:
    

out:
    

The output directory where the model checkpoints and logs are saved.

data:
    

The dataset configuration or path to a YAML file with the configuration. See the documentation for more information: <https://docs.lightly.ai/train/stable/instance_segmentation.html#data>

model:
    

The model to train. For example, âdinov2/vits14-eomtâ, âdinov3/vits16-eomt-cocoâ, or a path to a local model checkpoint.

If you want to resume training from an interrupted or crashed run, use the `resume_interrupted` parameter.

steps:
    

The number of training steps.

batch_size:
    

Global batch size. The batch size per device/GPU is inferred from this value and the number of devices and nodes.

num_workers:
    

Number of workers for the dataloader per device/GPU. âautoâ automatically sets the number of workers based on the available CPU cores.

devices:
    

Number of devices/GPUs for training. âautoâ automatically selects all available devices. The device type is determined by the `accelerator` parameter.

num_nodes:
    

Number of nodes for distributed training.

checkpoint:
    

Use this parameter to further fine-tune a model from a previous fine-tuned checkpoint. The checkpoint must be a path to a checkpoint file, for example âcheckpoints/model.ckptâ. This will only load the model weights from the previous run. All other training state (e.g. optimizer state, epochs) from the previous run are not loaded.

This option is equivalent to setting `model="<path_to_checkpoint>"`.

If you want to resume training from an interrupted or crashed run, use the `resume_interrupted` parameter instead.

reuse_class_head:
    

Deprecated. Now the model will reuse the classification head by default only when the num_classes in the data config matches that in the checkpoint. Otherwise, the classification head will be re-initialized.

resume_interrupted:
    

Set this to True if you want to resume training from an **interrupted or crashed** training run. This will pick up exactly where the training left off, including the optimizer state and the current step.

  * You must use the same `out` directory as the interrupted run.

  * You must **NOT** change any training parameters (e.g., learning rate, batch size, data, etc.).

  * This is intended for continuing the same run without modification.



overwrite:
    

Overwrite the output directory if it already exists. Warning, this might overwrite existing files in the directory!

accelerator:
    

Hardware accelerator. Can be one of [âcpuâ, âgpuâ, âmpsâ, âautoâ]. âautoâ will automatically select the best accelerator available.

strategy:
    

Training strategy. For example âddpâ or âautoâ. âautoâ automatically selects the best strategy available.

precision:
    

Training precision. Select â16-mixedâ for mixed 16-bit precision, â32-trueâ for full 32-bit precision, or âbf16-mixedâ for mixed bfloat16 precision.

float32_matmul_precision:
    

Precision for float32 matrix multiplication. Can be one of [âautoâ, âhighestâ, âhighâ, âmediumâ]. See <https://docs.pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision> for more information.

seed:
    

Random seed for reproducibility.

logger_args:
    

Logger arguments. Either None or a dictionary of logger names to either None or a dictionary of logger arguments. None uses the default loggers. To disable a logger, set it to None: `logger_args={"tensorboard": None}`. To configure a logger, pass the respective arguments: `logger_args={"mlflow": {"experiment_name": "my_experiment", ...}}`. See <https://docs.lightly.ai/train/stable/instance_segmentation.html#logging> for more information.

model_args:
    

Model training arguments. Either None or a dictionary of model arguments.

transform_args:
    

Transform arguments. Either None or a dictionary of transform arguments. The image size and normalization parameters can be set with `transform_args={"image_size": (height, width), "normalize": {"mean": (r, g, b), "std": (r, g, b)}}`

metric_args:
    

Metric arguments. Either None or a dictionary of metric arguments. Set `metric_args={"train": True}` to also log metrics on training data. Set `metric_args={"classwise": True}` to log per-class metrics. Set `metric_args={"watch_metric": "val_metric/map"}` to configure the metric used to select the best checkpoint.

loader_args:
    

Arguments for the PyTorch DataLoader. Should only be used in special cases as default values are automatically set. Prefer to use the batch_size and num_workers arguments instead. For details, see: <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>

save_checkpoint_args:
    

Arguments to configure the saving of checkpoints. The checkpoint frequency can be set with `save_checkpoint_args={"save_every_num_steps": 100}`.

torch_compile_args:
    

Arguments to configure model compilation with torch.compile. The arguments are directly passed to torch.compile. Model compilation is disabled for most models by default. Set `torch_compile_args={"disable": True}` to disable it if you encounter any issues.

gradient_accumulation_steps:
    

Number of gradient accumulation steps. âautoâ automatically enables gradient accumulation when batch_size is smaller than the modelâs default batch size, using `max(1, default_batch_size // batch_size)` steps to keep the effective batch size and learning rate close to the model defaults. Set to 1 to explicitly disable gradient accumulation.

lightly_train.train_object_detection(_*_ , _out : str | Path_, _data : dict[str, Any] | str_, _model : str_, _steps : int | Literal['auto'] = 'auto'_, _batch_size : int | Literal['auto'] = 'auto'_, _num_workers : int | Literal['auto'] = 'auto'_, _devices : int | str | list[int] = 'auto'_, _num_nodes : int = 1_, _resume_interrupted : bool = False_, _checkpoint : str | Path | None = None_, _reuse_class_head : bool = False_, _overwrite : bool = False_, _accelerator : str = 'auto'_, _strategy : str = 'auto'_, _precision : Literal[64, 32, 16, 'transformer-engine', 'transformer-engine-float16', '16-true', '16-mixed', 'bf16-true', 'bf16-mixed', '32-true', '64-true', '64', '32', '16', 'bf16'] = 'bf16-mixed'_, _float32_matmul_precision : Literal['auto', 'highest', 'high', 'medium'] = 'auto'_, _seed : int = 0_, _logger_args : dict[str, Any] | None = None_, _model_args : dict[str, Any] | None = None_, _transform_args : dict[str, Any] | None = None_, _metric_args : dict[str, Any] | None = None_, _loader_args : dict[str, Any] | None = None_, _save_checkpoint_args : dict[str, Any] | None = None_, _torch_compile_args : dict[str, Any] | None = None_, _gradient_accumulation_steps : int | Literal['auto'] = 'auto'_) → NoneÂ¶
    

Train an object detection model.

See the documentation for more information: <https://docs.lightly.ai/train/stable/object_detection.html>

> The training process can be monitored with TensorBoard:
    
    
    tensorboard --logdir out
    

After training, the last model checkpoint is saved in the out directory to: `out/checkpoints/last.ckpt` and also exported to `out/exported_models/exported_last.pt`.

Args:
    

out:
    

The output directory where the model checkpoints and logs are saved.

data:
    

The dataset configuration or path to a YAML file with the configuration. See the documentation for more information: <https://docs.lightly.ai/train/stable/object_detection.html#data>

model:
    

The model to train. For example, âdinov3/convnext-tiny-ltdetr-cocoâ, âdinov2/vits14-ltdetrâ, or a path to a local model checkpoint.

If you want to resume training from an interrupted or crashed run, use the `resume_interrupted` parameter.

steps:
    

The number of training steps.

batch_size:
    

Global batch size. The batch size per device/GPU is inferred from this value and the number of devices and nodes.

num_workers:
    

Number of workers for the dataloader per device/GPU. âautoâ automatically sets the number of workers based on the available CPU cores.

devices:
    

Number of devices/GPUs for training. âautoâ automatically selects all available devices. The device type is determined by the `accelerator` parameter.

num_nodes:
    

Number of nodes for distributed training.

checkpoint:
    

Use this parameter to further fine-tune a model from a previous fine-tuned checkpoint. The checkpoint must be a path to a checkpoint file, for example âcheckpoints/model.ckptâ. This will only load the model weights from the previous run. All other training state (e.g. optimizer state, epochs) from the previous run are not loaded.

This option is equivalent to setting `model="<path_to_checkpoint>"`.

If you want to resume training from an interrupted or crashed run, use the `resume_interrupted` parameter instead.

reuse_class_head:
    

Deprecated. Now the model will reuse the classification head by default only when the num_classes in the data config matches that in the checkpoint. Otherwise, the classification head will be re-initialized.

resume_interrupted:
    

Set this to True if you want to resume training from an **interrupted or crashed** training run. This will pick up exactly where the training left off, including the optimizer state and the current step.

  * You must use the same `out` directory as the interrupted run.

  * You must **NOT** change any training parameters (e.g., learning rate, batch size, data, etc.).

  * This is intended for continuing the same run without modification.



overwrite:
    

Overwrite the output directory if it already exists. Warning, this might overwrite existing files in the directory!

accelerator:
    

Hardware accelerator. Can be one of [âcpuâ, âgpuâ, âmpsâ, âautoâ]. âautoâ will automatically select the best accelerator available.

strategy:
    

Training strategy. For example âddpâ or âautoâ. âautoâ automatically selects the best strategy available.

precision:
    

Training precision. Select â16-mixedâ for mixed 16-bit precision, â32-trueâ for full 32-bit precision, or âbf16-mixedâ for mixed bfloat16 precision.

float32_matmul_precision:
    

Precision for float32 matrix multiplication. Can be one of [âautoâ, âhighestâ, âhighâ, âmediumâ]. See <https://docs.pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision> for more information.

seed:
    

Random seed for reproducibility.

logger_args:
    

Logger arguments. Either None or a dictionary of logger names to either None or a dictionary of logger arguments. None uses the default loggers. To disable a logger, set it to None: `logger_args={"tensorboard": None}`. To configure a logger, pass the respective arguments: `logger_args={"mlflow": {"experiment_name": "my_experiment", ...}}`. See <https://docs.lightly.ai/train/stable/semantic_segmentation.html#logging> for more information.

model_args:
    

Model training arguments. Either None or a dictionary of model arguments.

transform_args:
    

Transform arguments. Either None or a dictionary of transform arguments. The image size and normalization parameters can be set with `transform_args={"image_size": (height, width), "normalize": {"mean": (r, g, b), "std": (r, g, b)}}`

metric_args:
    

Metric arguments. Either None or a dictionary of metric arguments. Set `metric_args={"train": True}` to also log metrics on training data. Set `metric_args={"classwise": True}` to log per-class metrics. Set `metric_args={"watch_metric": "val_metric/map"}` to configure the metric used to select the best checkpoint.

loader_args:
    

Arguments for the PyTorch DataLoader. Should only be used in special cases as default values are automatically set. Prefer to use the batch_size and num_workers arguments instead. For details, see: <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>

save_checkpoint_args:
    

Arguments to configure the saving of checkpoints. The checkpoint frequency can be set with `save_checkpoint_args={"save_every_num_steps": 100}`.

torch_compile_args:
    

Arguments to configure model compilation with torch.compile. The arguments are directly passed to torch.compile. Model compilation is disabled for most models by default. Set `torch_compile_args={"disable": True}` to disable it if you encounter any issues.

gradient_accumulation_steps:
    

Number of gradient accumulation steps. âautoâ automatically enables gradient accumulation when batch_size is smaller than the modelâs default batch size, using `max(1, default_batch_size // batch_size)` steps to keep the effective batch size and learning rate close to the model defaults. Set to 1 to explicitly disable gradient accumulation.

lightly_train.train_panoptic_segmentation(_*_ , _out : str | Path_, _data : dict[str, Any]_, _model : str_, _steps : int | Literal['auto'] = 'auto'_, _batch_size : int | Literal['auto'] = 'auto'_, _num_workers : int | Literal['auto'] = 'auto'_, _devices : int | str | list[int] = 'auto'_, _num_nodes : int = 1_, _resume_interrupted : bool = False_, _checkpoint : str | Path | None = None_, _reuse_class_head : bool = False_, _overwrite : bool = False_, _accelerator : str = 'auto'_, _strategy : str = 'auto'_, _precision : Literal[64, 32, 16, 'transformer-engine', 'transformer-engine-float16', '16-true', '16-mixed', 'bf16-true', 'bf16-mixed', '32-true', '64-true', '64', '32', '16', 'bf16'] = 'bf16-mixed'_, _float32_matmul_precision : Literal['auto', 'highest', 'high', 'medium'] = 'auto'_, _seed : int = 0_, _logger_args : dict[str, Any] | None = None_, _model_args : dict[str, Any] | None = None_, _transform_args : dict[str, Any] | None = None_, _metric_args : dict[str, Any] | None = None_, _loader_args : dict[str, Any] | None = None_, _save_checkpoint_args : dict[str, Any] | None = None_, _torch_compile_args : dict[str, Any] | None = None_, _gradient_accumulation_steps : int | Literal['auto'] = 'auto'_) → NoneÂ¶
    

Train a panoptic segmentation model.

See the documentation for more information: <https://docs.lightly.ai/train/stable/panoptic_segmentation.html>

> The training process can be monitored with TensorBoard:
    
    
    tensorboard --logdir out
    

After training, the last model checkpoint is saved in the out directory to: `out/checkpoints/last.ckpt` and also exported to `out/exported_models/exported_last.pt`.

Args:
    

out:
    

The output directory where the model checkpoints and logs are saved.

data:
    

The dataset configuration or path to a YAML file with the configuration. See the documentation for more information: <https://docs.lightly.ai/train/stable/panoptic_segmentation.html#data>

model:
    

The model to train. For example âdinov3/vits16-eomt-cocoâ or a path to a local model checkpoint.

If you want to resume training from an interrupted or crashed run, use the `resume_interrupted` parameter.

steps:
    

The number of training steps.

batch_size:
    

Global batch size. The batch size per device/GPU is inferred from this value and the number of devices and nodes.

num_workers:
    

Number of workers for the dataloader per device/GPU. âautoâ automatically sets the number of workers based on the available CPU cores.

devices:
    

Number of devices/GPUs for training. âautoâ automatically selects all available devices. The device type is determined by the `accelerator` parameter.

num_nodes:
    

Number of nodes for distributed training.

checkpoint:
    

Use this parameter to further fine-tune a model from a previous fine-tuned checkpoint. The checkpoint must be a path to a checkpoint file, for example âcheckpoints/model.ckptâ. This will only load the model weights from the previous run. All other training state (e.g. optimizer state, epochs) from the previous run are not loaded.

This option is equivalent to setting `model="<path_to_checkpoint>"`.

If you want to resume training from an interrupted or crashed run, use the `resume_interrupted` parameter instead.

reuse_class_head:
    

Set this to True if you want to keep the class head from the provided checkpoint. The default behavior removes the class head before loading so that a new head can be initialized for the current task.

resume_interrupted:
    

Set this to True if you want to resume training from an **interrupted or crashed** training run. This will pick up exactly where the training left off, including the optimizer state and the current step.

  * You must use the same `out` directory as the interrupted run.

  * You must **NOT** change any training parameters (e.g., learning rate, batch size, data, etc.).

  * This is intended for continuing the same run without modification.



overwrite:
    

Overwrite the output directory if it already exists. Warning, this might overwrite existing files in the directory!

accelerator:
    

Hardware accelerator. Can be one of [âcpuâ, âgpuâ, âmpsâ, âautoâ]. âautoâ will automatically select the best accelerator available.

strategy:
    

Training strategy. For example âddpâ or âautoâ. âautoâ automatically selects the best strategy available.

precision:
    

Training precision. Select â16-mixedâ for mixed 16-bit precision, â32-trueâ for full 32-bit precision, or âbf16-mixedâ for mixed bfloat16 precision.

float32_matmul_precision:
    

Precision for float32 matrix multiplication. Can be one of [âautoâ, âhighestâ, âhighâ, âmediumâ]. See <https://docs.pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision> for more information.

seed:
    

Random seed for reproducibility.

logger_args:
    

Logger arguments. Either None or a dictionary of logger names to either None or a dictionary of logger arguments. None uses the default loggers. To disable a logger, set it to None: `logger_args={"tensorboard": None}`. To configure a logger, pass the respective arguments: `logger_args={"mlflow": {"experiment_name": "my_experiment", ...}}`. See <https://docs.lightly.ai/train/stable/panoptic_segmentation.html#logging> for more information.

model_args:
    

Model training arguments. Either None or a dictionary of model arguments.

transform_args:
    

Transform arguments. Either None or a dictionary of transform arguments. The image size and normalization parameters can be set with `transform_args={"image_size": (height, width), "normalize": {"mean": (r, g, b), "std": (r, g, b)}}`

metric_args:
    

Metric arguments. Either None or a dictionary of metric arguments. Set `metric_args={"train": True}` to also log metrics on training data. Set `metric_args={"classwise": True}` to log per-class metrics. Set `metric_args={"watch_metric": "val_metric/pq"}` to configure the metric used to select the best checkpoint.

loader_args:
    

Arguments for the PyTorch DataLoader. Should only be used in special cases as default values are automatically set. Prefer to use the batch_size and num_workers arguments instead. For details, see: <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>

save_checkpoint_args:
    

Arguments to configure the saving of checkpoints. The checkpoint frequency can be set with `save_checkpoint_args={"save_every_num_steps": 100}`.

torch_compile_args:
    

Arguments to configure model compilation with torch.compile. The arguments are directly passed to torch.compile. Model compilation is disabled for most models by default. Set `torch_compile_args={"disable": True}` to disable it if you encounter any issues.

gradient_accumulation_steps:
    

Number of gradient accumulation steps. âautoâ automatically enables gradient accumulation when batch_size is smaller than the modelâs default batch size, using `max(1, default_batch_size // batch_size)` steps to keep the effective batch size and learning rate close to the model defaults. Set to 1 to explicitly disable gradient accumulation.

lightly_train.train_semantic_segmentation(_*_ , _out : str | Path_, _data : dict[str, Any]_, _model : str_, _steps : int | Literal['auto'] = 'auto'_, _batch_size : int | Literal['auto'] = 'auto'_, _num_workers : int | Literal['auto'] = 'auto'_, _devices : int | str | list[int] = 'auto'_, _num_nodes : int = 1_, _resume_interrupted : bool = False_, _checkpoint : str | Path | None = None_, _reuse_class_head : bool = False_, _overwrite : bool = False_, _accelerator : str = 'auto'_, _strategy : str = 'auto'_, _precision : Literal[64, 32, 16, 'transformer-engine', 'transformer-engine-float16', '16-true', '16-mixed', 'bf16-true', 'bf16-mixed', '32-true', '64-true', '64', '32', '16', 'bf16'] = 'bf16-mixed'_, _float32_matmul_precision : Literal['auto', 'highest', 'high', 'medium'] = 'auto'_, _seed : int = 0_, _logger_args : dict[str, Any] | None = None_, _model_args : dict[str, Any] | None = None_, _transform_args : dict[str, Any] | None = None_, _metric_args : dict[str, Any] | None = None_, _loader_args : dict[str, Any] | None = None_, _save_checkpoint_args : dict[str, Any] | None = None_, _torch_compile_args : dict[str, Any] | None = None_, _gradient_accumulation_steps : int | Literal['auto'] = 'auto'_) → NoneÂ¶
    

Train a semantic segmentation model.

See the documentation for more information: <https://docs.lightly.ai/train/stable/semantic_segmentation.html>

> The training process can be monitored with TensorBoard:
    
    
    tensorboard --logdir out
    

After training, the last model checkpoint is saved in the out directory to: `out/checkpoints/last.ckpt` and also exported to `out/exported_models/exported_last.pt`.

Args:
    

out:
    

The output directory where the model checkpoints and logs are saved.

data:
    

The dataset configuration or path to a YAML file with the configuration. See the documentation for more information: <https://docs.lightly.ai/train/stable/semantic_segmentation.html#data>

model:
    

The model to train. For example, âdinov2/vits14-eomtâ, âdinov3/vits16-eomt-cocoâ, or a path to a local model checkpoint.

If you want to resume training from an interrupted or crashed run, use the `resume_interrupted` parameter.

steps:
    

The number of training steps.

batch_size:
    

Global batch size. The batch size per device/GPU is inferred from this value and the number of devices and nodes.

num_workers:
    

Number of workers for the dataloader per device/GPU. âautoâ automatically sets the number of workers based on the available CPU cores.

devices:
    

Number of devices/GPUs for training. âautoâ automatically selects all available devices. The device type is determined by the `accelerator` parameter.

num_nodes:
    

Number of nodes for distributed training.

checkpoint:
    

Use this parameter to further fine-tune a model from a previous fine-tuned checkpoint. The checkpoint must be a path to a checkpoint file, for example âcheckpoints/model.ckptâ. This will only load the model weights from the previous run. All other training state (e.g. optimizer state, epochs) from the previous run are not loaded.

This option is equivalent to setting `model="<path_to_checkpoint>"`.

If you want to resume training from an interrupted or crashed run, use the `resume_interrupted` parameter instead.

reuse_class_head:
    

Deprecated. Now the model will reuse the classification head by default only when the num_classes in the data config matches that in the checkpoint. Otherwise, the classification head will be re-initialized.

resume_interrupted:
    

Set this to True if you want to resume training from an **interrupted or crashed** training run. This will pick up exactly where the training left off, including the optimizer state and the current step.

  * You must use the same `out` directory as the interrupted run.

  * You must **NOT** change any training parameters (e.g., learning rate, batch size, data, etc.).

  * This is intended for continuing the same run without modification.



overwrite:
    

Overwrite the output directory if it already exists. Warning, this might overwrite existing files in the directory!

accelerator:
    

Hardware accelerator. Can be one of [âcpuâ, âgpuâ, âmpsâ, âautoâ]. âautoâ will automatically select the best accelerator available.

strategy:
    

Training strategy. For example âddpâ or âautoâ. âautoâ automatically selects the best strategy available.

precision:
    

Training precision. Select â16-mixedâ for mixed 16-bit precision, â32-trueâ for full 32-bit precision, or âbf16-mixedâ for mixed bfloat16 precision.

float32_matmul_precision:
    

Precision for float32 matrix multiplication. Can be one of [âautoâ, âhighestâ, âhighâ, âmediumâ]. See <https://docs.pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision> for more information.

seed:
    

Random seed for reproducibility.

logger_args:
    

Logger arguments. Either None or a dictionary of logger names to either None or a dictionary of logger arguments. None uses the default loggers. To disable a logger, set it to None: `logger_args={"tensorboard": None}`. To configure a logger, pass the respective arguments: `logger_args={"mlflow": {"experiment_name": "my_experiment", ...}}`. See <https://docs.lightly.ai/train/stable/semantic_segmentation.html#logging> for more information.

model_args:
    

Model training arguments. Either None or a dictionary of model arguments.

transform_args:
    

Transform arguments. Either None or a dictionary of transform arguments. The image size and normalization parameters can be set with `transform_args={"image_size": (height, width), "normalize": {"mean": (r, g, b), "std": (r, g, b)}}`

metric_args:
    

Metric arguments. Either None or a dictionary of metric arguments. Set `metric_args={"train": True}` to also log metrics on training data. Set `metric_args={"classwise": True}` to log per-class metrics. Set `metric_args={"watch_metric": "val_metric/miou"}` to configure the metric used to select the best checkpoint.

loader_args:
    

Arguments for the PyTorch DataLoader. Should only be used in special cases as default values are automatically set. Prefer to use the batch_size and num_workers arguments instead. For details, see: <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>

save_checkpoint_args:
    

Arguments to configure the saving of checkpoints. The checkpoint frequency can be set with `save_checkpoint_args={"save_every_num_steps": 100}`.

torch_compile_args:
    

Arguments to configure model compilation with torch.compile. The arguments are directly passed to torch.compile. Model compilation is disabled for most models by default. Set `torch_compile_args={"disable": True}` to disable it if you encounter any issues.

gradient_accumulation_steps:
    

Number of gradient accumulation steps. âautoâ automatically enables gradient accumulation when batch_size is smaller than the modelâs default batch size, using `max(1, default_batch_size // batch_size)` steps to keep the effective batch size and learning rate close to the model defaults. Set to 1 to explicitly disable gradient accumulation.

## ModelsÂ¶

class lightly_train._task_models.image_classification.task_model.ImageClassificationÂ¶
    

export_onnx(_out : str | Path_, _*_ , _precision : Literal['auto', 'fp32', 'fp16'] = 'auto'_, _batch_size : int = 1_, _dynamic_batch_size : bool = True_, _height : int | None = None_, _width : int | None = None_, _opset_version : int | None = None_, _simplify : bool = True_, _verify : bool = True_, _format_args : dict[str, Any] | None = None_) → NoneÂ¶
    

Exports the model to ONNX for inference.

The export uses a dummy input of shape (batch_size, C, H, W) where C is inferred from the first model parameter and (H, W) come from self.image_size. If dynamic_batch_size is True, the ONNX graph will have a dynamic batch dimension for the input. The graph produces two outputs: labels and scores.

Optionally simplifies the exported model in-place using onnxslim and verifies numerical closeness against a float32 CPU reference via ONNX Runtime.

Args:
    

out:
    

Path where the ONNX model will be written.

precision:
    

Precision for the ONNX model. Either âautoâ, âfp32â, or âfp16â. âautoâ uses the modelâs current precision.

batch_size:
    

Batch size for the ONNX input.

dynamic_batch_size:
    

If True, the ONNX graph will have a dynamic batch dimension for the input. If False, the batch dimension is fixed to batch_size.

height:
    

Height of the ONNX input. If None, will be taken from self.image_size.

width:
    

Width of the ONNX input. If None, will be taken from self.image_size.

opset_version:
    

ONNX opset version to target. If None, PyTorchâs default opset is used.

simplify:
    

If True, run onnxslim to simplify and overwrite the exported model.

verify:
    

If True, validate the ONNX file and compare outputs to a float32 CPU reference forward pass.

format_args:
    

Optional extra keyword arguments forwarded to torch.onnx.export.

Returns:
    

None. Writes the ONNX model to out.

export_tensorrt(_out : str | Path_, _*_ , _precision : Literal['auto', 'fp32', 'fp16'] = 'auto'_, _onnx_args : dict[str, Any] | None = None_, _max_batchsize : int = 1_, _opt_batchsize : int = 1_, _min_batchsize : int = 1_, _verbose : bool = False_) → NoneÂ¶
    

Build a TensorRT engine from an ONNX model.

Note

TensorRT is not part of LightlyTrainâs dependencies and must be installed separately. Installation depends on your OS, Python version, GPU, and NVIDIA driver/CUDA setup. See the [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html) for more details. On CUDA 12.x systems you can often install the Python package via pip install tensorrt-cu12.

This loads the ONNX file, parses it with TensorRT, infers the static input shape (C, H, W) from the âimagesâ input, and creates an engine with a dynamic batch dimension in the range [min_batchsize, opt_batchsize, max_batchsize]. Spatial dimensions must be static in the ONNX model (dynamic H/W are not yet supported).

The engine is serialized and written to out.

Args:
    

out:
    

Path where the TensorRT engine will be saved.

precision:
    

Precision for ONNX export and TensorRT engine building. Either âautoâ, âfp32â, or âfp16â. âautoâ uses the modelâs current precision.

onnx_args:
    

Optional arguments to pass to export_onnx when exporting the ONNX model prior to building the TensorRT engine. If None, default arguments are used and the ONNX file is saved alongside the TensorRT engine with the same name but .onnx extension.

max_batchsize:
    

Maximum supported batch size.

opt_batchsize:
    

Batch size TensorRT optimizes for.

min_batchsize:
    

Minimum supported batch size.

verbose:
    

Enable verbose TensorRT logging.

Raises:
    

FileNotFoundError: If the ONNX file does not exist. RuntimeError: If the ONNX cannot be parsed or engine building fails. ValueError: If batch size constraints are invalid or H/W are dynamic.

predict(_image : str | Path | Image | Tensor_, _topk : int = 1_, _threshold : float = 0.5_) → dict[str, Tensor]Â¶
    

Returns the predicted labels and scores for the input image.

Args:
    

image:
    

The input image as a path, URL, PIL image, or tensor. Tensors must have shape (C, H, W).

Returns:
    

A {âlabelsâ: Tensor, âscoresâ: Tensor} dict. Labels has shape (topk,) for multiclass and (num_labels,) for multilabel where num_labels is the number of labels with score > threshold. Scores has the same shape as labels and contains the corresponding scores.

class lightly_train._task_models.dinov3_eomt_instance_segmentation.task_model.DINOv3EoMTInstanceSegmentationÂ¶
    

export_onnx(_out : str | Path_, _*_ , _precision : Literal['auto', 'fp32', 'fp16'] = 'auto'_, _batch_size : int = 1_, _dynamic_batch_size : bool = True_, _height : int | None = None_, _width : int | None = None_, _opset_version : int | None = None_, _simplify : bool = True_, _verify : bool = True_, _format_args : dict[str, Any] | None = None_) → NoneÂ¶
    

Exports the model to ONNX for inference.

The export uses a dummy input of shape (batch_size, C, H, W) where C is inferred from the first model parameter and (H, W) come from self.image_size. If dynamic_batch_size is True, the ONNX graph will have a dynamic batch dimension for the input. The graph produces three outputs: labels, masks, and scores.

Optionally simplifies the exported model in-place using onnxslim and verifies numerical closeness against a float32 CPU reference via ONNX Runtime.

Args:
    

out:
    

Path where the ONNX model will be written.

precision:
    

Precision for the ONNX model. Either âautoâ, âfp32â, or âfp16â. âautoâ uses the modelâs current precision.

batch_size:
    

Batch size for the ONNX input.

dynamic_batch_size:
    

If True, the ONNX graph will have a dynamic batch dimension for the input. If False, the batch dimension is fixed to batch_size.

height:
    

Height of the ONNX input. If None, will be taken from self.image_size.

width:
    

Width of the ONNX input. If None, will be taken from self.image_size.

opset_version:
    

ONNX opset version to target. If None, PyTorchâs default opset is used.

simplify:
    

If True, run onnxslim to simplify and overwrite the exported model.

verify:
    

If True, validate the ONNX file and compare outputs to a float32 CPU reference forward pass.

format_args:
    

Optional extra keyword arguments forwarded to torch.onnx.export.

Returns:
    

None. Writes the ONNX model to out.

export_tensorrt(_out : str | Path_, _*_ , _precision : Literal['auto', 'fp32', 'fp16'] = 'auto'_, _onnx_args : dict[str, Any] | None = None_, _max_batchsize : int = 1_, _opt_batchsize : int = 1_, _min_batchsize : int = 1_, _verbose : bool = False_) → NoneÂ¶
    

Build a TensorRT engine from an ONNX model.

Note

TensorRT is not part of LightlyTrainâs dependencies and must be installed separately. Installation depends on your OS, Python version, GPU, and NVIDIA driver/CUDA setup. See the [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html) for more details. On CUDA 12.x systems you can often install the Python package via pip install tensorrt-cu12.

This loads the ONNX file, parses it with TensorRT, infers the static input shape (C, H, W) from the âimagesâ input, and creates an engine with a dynamic batch dimension in the range [min_batchsize, opt_batchsize, max_batchsize]. Spatial dimensions must be static in the ONNX model (dynamic H/W are not yet supported).

The engine is serialized and written to out.

Args:
    

out:
    

Path where the TensorRT engine will be saved.

precision:
    

Precision for ONNX export and TensorRT engine building. Either âautoâ, âfp32â, or âfp16â. âautoâ uses the modelâs current precision.

onnx_args:
    

Optional arguments to pass to export_onnx when exporting the ONNX model prior to building the TensorRT engine. If None, default arguments are used and the ONNX file is saved alongside the TensorRT engine with the same name but .onnx extension.

max_batchsize:
    

Maximum supported batch size.

opt_batchsize:
    

Batch size TensorRT optimizes for.

min_batchsize:
    

Minimum supported batch size.

verbose:
    

Enable verbose TensorRT logging.

Raises:
    

FileNotFoundError: If the ONNX file does not exist. RuntimeError: If the ONNX cannot be parsed or engine building fails. ValueError: If batch size constraints are invalid or H/W are dynamic.

predict(_image : str | Path | Image | Tensor_, _threshold : float = 0.8_) → dict[str, Tensor]Â¶
    

Returns the predicted mask for the given image.

Args:
    

image:
    

The input image as a path, URL, PIL image, or tensor. Tensors must have shape (C, H, W).

threshold:
    

The confidence threshold for the predicted masks. Only masks with a confidence score above this threshold are returned.

Returns:
    

A {âlabelsâ: Tensor, âmasksâ: Tensor, âscoresâ: Tensor} dict. Labels is a tensor of shape (Q,) containing the predicted class for each query. Masks is a tensor of shape (Q, H, W) containing the predicted mask for each query. Scores is a tensor of shape (Q,) containing the confidence score for each query.

class lightly_train._task_models.dinov2_ltdetr_object_detection.task_model.DINOv2LTDETRObjectDetectionÂ¶
    

predict(_image : str | Path | Image | Tensor_, _threshold : float = 0.6_) → dict[str, Tensor]Â¶
    

Run inference on a single image and return detected boxes, labels and scores.

Args:
    

image:
    

Input image. Can be a path, a PIL image, or a tensor of shape (C, H, W).

threshold:
    

Score threshold to filter low-confidence predictions. Predictions with scores <= threshold are discarded.

Returns:
    

A dictionary with:
    

  * âlabelsâ: Tensor of shape (N,) with predicted class indices.

  * âbboxesâ: Tensor of shape (N, 4) with bounding boxes in (x_min, y_min, x_max, y_max) in absolute pixel coordinates of the original image.

  * âscoresâ: Tensor of shape (N,) with confidence scores for each prediction.




predict_sahi(_image : str | Path | Image | Tensor_, _threshold : float = 0.6_, _overlap : float = 0.2_, _nms_iou_threshold : float = 0.3_, _global_local_iou_threshold : float = 0.1_) → dict[str, Tensor]Â¶
    

Run Slicing Aided Hyper Inference (SAHI) inference on the input image.

The image is first converted to a tensor, then:

  * Tiled into overlapping crops of size self.image_size.

  * A resized full-image version is added as a âglobalâ tile.

  * All tiles (global + local) are passed through the model in parallel.

  * Predictions are filtered by score and merged using NMS and a global/local consistency heuristic. NMS is only applied on tiles predictions. The heuristic discards tiles predictions that heavily overlaps with global predictions.




Args:
    

image:
    

Input image. Can be a path, a PIL image, or a tensor of shape (C, H, W).

threshold:
    

Score threshold for filtering low-confidence predictions.

overlap:
    

Fractional overlap between tiles in [0, 1). 0.0 means no overlap.

nms_iou_threshold:
    

IoU threshold used for non-maximum suppression when merging predictions from tiles and global image. A lower nms_iou_threshold value yields less predictions.

global_local_iou_threshold:
    

Minimum IoU required to consider a tile prediction as matching a global prediction when combining them. A lower global_local_iou_threshold yields less predictions.

Returns:
    

A dictionary with:
    

  * âlabelsâ: Tensor of shape (N,) with predicted class indices.

  * âbboxesâ: Tensor of shape (N, 4) with bounding boxes in
    

(x_min, y_min, x_max, y_max) in absolute pixel coordinates of the original image.

  * âscoresâ: Tensor of shape (N,) with confidence scores for each prediction.




class lightly_train._task_models.dinov3_ltdetr_object_detection.task_model.DINOv3LTDETRObjectDetectionÂ¶
    

export_onnx(_out : str | Path_, _*_ , _precision : Literal['auto', 'fp32', 'fp16'] = 'auto'_, _batch_size : int = 1_, _dynamic_batch_size : bool = True_, _opset_version : int | None = None_, _simplify : bool = True_, _verify : bool = True_, _format_args : dict[str, Any] | None = None_, _num_channels : int | None = None_) → NoneÂ¶
    

Exports the model to ONNX for inference.

The export uses a dummy input of shape (batch_size, C, H, W) where C is inferred from the first model parameter and (H, W) come from self.image_size. If dynamic_batch_size is True, the ONNX graph will have a dynamic batch dimension for the input. The graph produces three outputs: labels, boxes, and scores.

Optionally simplifies the exported model in-place using onnxslim and verifies numerical closeness against a float32 CPU reference via ONNX Runtime.

Args:
    

out:
    

Path where the ONNX model will be written.

precision:
    

Precision for the ONNX model. Either âautoâ, âfp32â, or âfp16â. âautoâ uses the modelâs current precision.

batch_size:
    

Batch size for the ONNX input.

dynamic_batch_size:
    

If True, the ONNX graph will have a dynamic batch dimension for the input. If False, the batch dimension is fixed to batch_size.

opset_version:
    

ONNX opset version to target. If None, PyTorchâs default opset is used.

simplify:
    

If True, run onnxslim to simplify and overwrite the exported model.

verify:
    

If True, validate the ONNX file and compare outputs to a float32 CPU reference forward pass.

format_args:
    

Optional extra keyword arguments forwarded to torch.onnx.export.

num_channels:
    

Number of input channels. If None, will be inferred.

Returns:
    

None. Writes the ONNX model to out.

export_tensorrt(_out : str | Path_, _*_ , _precision : Literal['auto', 'fp32', 'fp16'] = 'auto'_, _onnx_args : dict[str, Any] | None = None_, _max_batchsize : int = 1_, _opt_batchsize : int = 1_, _min_batchsize : int = 1_, _verbose : bool = False_) → NoneÂ¶
    

Build a TensorRT engine from an ONNX model.

Note

TensorRT is not part of LightlyTrainâs dependencies and must be installed separately. Installation depends on your OS, Python version, GPU, and NVIDIA driver/CUDA setup. See the [TensorRT documentation](https://docs.lightly.ai/train/stable/python_api/<https:/docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html>) for more details. On CUDA 12.x systems you can often install the Python package via pip install tensorrt-cu12.

This loads the ONNX file, parses it with TensorRT, infers the static input shape (C, H, W) from the âimagesâ input, and creates an engine with a dynamic batch dimension in the range [min_batchsize, opt_batchsize, max_batchsize]. Spatial dimensions must be static in the ONNX model (dynamic H/W are not yet supported).

The engine is serialized and written to out.

Args:
    

out:
    

Path where the TensorRT engine will be saved.

precision:
    

Precision for ONNX export and TensorRT engine building. Either âautoâ, âfp32â, or âfp16â. âautoâ uses the modelâs current precision.

onnx_args:
    

Optional arguments to pass to export_onnx when exporting the ONNX model prior to building the TensorRT engine. If None, default arguments are used and the ONNX file is saved alongside the TensorRT engine with the same name but .onnx extension.

max_batchsize:
    

Maximum supported batch size.

opt_batchsize:
    

Batch size TensorRT optimizes for.

min_batchsize:
    

Minimum supported batch size.

verbose:
    

Enable verbose TensorRT logging.

Raises:
    

FileNotFoundError: If the ONNX file does not exist. RuntimeError: If the ONNX cannot be parsed or engine building fails. ValueError: If batch size constraints are invalid or H/W are dynamic.

predict(_image : str | Path | Image | Tensor_, _threshold : float = 0.6_) → dict[str, Tensor]Â¶
    

Run inference on a single image and return detected boxes, labels and scores.

Args:
    

image:
    

Input image. Can be a path, a PIL image, or a tensor of shape (C, H, W).

threshold:
    

Score threshold to filter low-confidence predictions. Predictions with scores <= threshold are discarded.

Returns:
    

A dictionary with:
    

  * âlabelsâ: Tensor of shape (N,) with predicted class indices.

  * âbboxesâ: Tensor of shape (N, 4) with bounding boxes in (x_min, y_min, x_max, y_max) in absolute pixel coordinates of the original image.

  * âscoresâ: Tensor of shape (N,) with confidence scores for each prediction.




predict_sahi(_image : str | Path | Image | Tensor_, _threshold : float = 0.6_, _overlap : float = 0.2_, _nms_iou_threshold : float = 0.3_, _global_local_iou_threshold : float = 0.1_) → dict[str, Tensor]Â¶
    

Run Slicing Aided Hyper Inference (SAHI) inference on the input image.

The image is first converted to a tensor, then:

  * Tiled into overlapping crops of size self.image_size.

  * A resized full-image version is added as a âglobalâ tile.

  * All tiles (global + local) are passed through the model in parallel.

  * Predictions are filtered by score and merged using NMS and a global/local consistency heuristic. NMS is only applied on tiles predictions. The heuristic discards tiles predictions that heavily overlaps with global predictions.




Args:
    

image:
    

Input image. Can be a path, a PIL image, or a tensor of shape (C, H, W).

threshold:
    

Score threshold for filtering low-confidence predictions.

overlap:
    

Fractional overlap between tiles in [0, 1). 0.0 means no overlap.

nms_iou_threshold:
    

IoU threshold used for non-maximum suppression when merging predictions from tiles and global image. A lower nms_iou_threshold value yields less predictions.

global_local_iou_threshold:
    

Minimum IoU required to consider a tile prediction as matching a global prediction when combining them. A lower global_local_iou_threshold yields less predictions.

Returns:
    

A dictionary with:
    

  * âlabelsâ: Tensor of shape (N,) with predicted class indices.

  * âbboxesâ: Tensor of shape (N, 4) with bounding boxes in (x_min, y_min, x_max, y_max) in the coordinates of the original image.

  * âscoresâ: Tensor of shape (N,) with confidence scores for each prediction.




class lightly_train._task_models.picodet_object_detection.task_model.PicoDetObjectDetectionÂ¶
    

PicoDet-S object detection model.

PicoDet is a lightweight anchor-free object detector designed for mobile and edge deployment. It uses an Enhanced ShuffleNet backbone, CSP-PAN neck, and GFL-style detection head.

export_onnx(_out : str | Path_, _*_ , _precision : Literal['auto', 'fp32', 'fp16'] = 'auto'_, _batch_size : int = 1_, _dynamic_batch_size : bool = True_, _opset_version : int | None = None_, _simplify : bool = True_, _verify : bool = True_, _format_args : dict[str, Any] | None = None_, _num_channels : int | None = None_) → NoneÂ¶
    

Exports the model to ONNX for inference.

The export uses a dummy input of shape (batch_size, C, H, W) where C is inferred from the first model parameter and (H, W) come from self.image_size. If dynamic_batch_size is True, the ONNX graph will have a dynamic batch dimension for the input. The graph outputs labels, boxes, and scores.

Optionally simplifies the exported model in-place using onnxslim and verifies numerical closeness against a float32 CPU reference via ONNX Runtime.

Args:
    

out:
    

Path where the ONNX model will be written.

precision:
    

Precision for the ONNX model. Either âautoâ, âfp32â, or âfp16â. âautoâ uses the modelâs current precision.

batch_size:
    

Batch size for the ONNX input.

dynamic_batch_size:
    

If True, the ONNX graph will have a dynamic batch dimension for the input. If False, the batch dimension is fixed to batch_size.

opset_version:
    

ONNX opset version to target. If None, PyTorchâs default opset is used.

simplify:
    

If True, run onnxslim to simplify and overwrite the exported model.

verify:
    

If True, validate the ONNX file and compare outputs to a float32 CPU reference forward pass.

format_args:
    

Optional extra keyword arguments forwarded to torch.onnx.export.

num_channels:
    

Number of input channels. If None, will be inferred.

export_tensorrt(_out : str | Path_, _*_ , _precision : Literal['auto', 'fp32', 'fp16'] = 'auto'_, _onnx_args : dict[str, Any] | None = None_, _max_batchsize : int = 1_, _opt_batchsize : int = 1_, _min_batchsize : int = 1_, _verbose : bool = False_) → NoneÂ¶
    

Build a TensorRT engine from an ONNX model.

Note

TensorRT is not part of LightlyTrainâs dependencies and must be installed separately. Installation depends on your OS, Python version, GPU, and NVIDIA driver/CUDA setup. See the [TensorRT documentation](https://docs.lightly.ai/train/stable/python_api/<https:/docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html>) for more details. On CUDA 12.x systems you can often install the Python package via pip install tensorrt-cu12.

This loads the ONNX file, parses it with TensorRT, infers the static input shape (C, H, W) from the âimagesâ input, and creates an engine with a dynamic batch dimension in the range [min_batchsize, opt_batchsize, max_batchsize]. Spatial dimensions must be static in the ONNX model (dynamic H/W are not yet supported).

The engine is serialized and written to out.

Args:
    

out:
    

Path where the TensorRT engine will be saved.

precision:
    

Precision for ONNX export and TensorRT engine building. Either âautoâ, âfp32â, or âfp16â. âautoâ uses the modelâs current precision.

onnx_args:
    

Optional arguments to pass to export_onnx when exporting the ONNX model prior to building the TensorRT engine. If None, default arguments are used and the ONNX file is saved alongside the TensorRT engine with the same name but .onnx extension.

max_batchsize:
    

Maximum supported batch size.

opt_batchsize:
    

Batch size TensorRT optimizes for.

min_batchsize:
    

Minimum supported batch size.

verbose:
    

Enable verbose TensorRT logging.

Raises:
    

FileNotFoundError: If the ONNX file does not exist. RuntimeError: If the ONNX cannot be parsed or engine building fails. ValueError: If batch size constraints are invalid or H/W are dynamic.

predict(_image : str | Path | Image | Tensor_, _threshold : float = 0.6_) → dict[str, Tensor]Â¶
    

Run inference on a single image.

Args:
    

image: Input image as path, PIL image, or tensor (C, H, W). threshold: Score threshold for detections.

Returns:
    

Dictionary with: \- labels: Tensor of shape (N,) with class indices. \- bboxes: Tensor of shape (N, 4) with boxes in xyxy format. \- scores: Tensor of shape (N,) with confidence scores.

class lightly_train._task_models.dinov3_eomt_panoptic_segmentation.task_model.DINOv3EoMTPanopticSegmentationÂ¶
    

export_onnx(_out : str | Path_, _*_ , _precision : Literal['auto', 'fp32', 'fp16'] = 'auto'_, _batch_size : int = 1_, _height : int | None = None_, _width : int | None = None_, _threshold : float = 0.8_, _mask_threshold : float = 0.5_, _mask_overlap_threshold : float = 0.8_, _opset_version : int | None = None_, _simplify : bool = True_, _verify : bool = True_, _format_args : dict[str, Any] | None = None_) → NoneÂ¶
    

Exports the model to ONNX for inference.

The export uses a dummy input of shape (batch_size, C, H, W) where C is inferred from the first model parameter and (H, W) come from self.image_size. The ONNX graph uses dynamic batch size for input images. The output masks, segment_ids, and scores have dynamic shapes depending on the number of detected segments.

Optionally simplifies the exported model in-place using onnxslim and verifies numerical closeness against a float32 CPU reference via ONNX Runtime.

Args:
    

out:
    

Path where the ONNX model will be written.

precision:
    

Precision for the ONNX model. Either âautoâ, âfp32â, or âfp16â. âautoâ uses the modelâs current precision.

batch_size:
    

Batch size for the ONNX input. Only batch size 1 is supported.

height:
    

Height of the ONNX input. If None, will be taken from self.image_size.

width:
    

Width of the ONNX input. If None, will be taken from self.image_size.

threshold:
    

Confidence threshold to keep predicted masks. Will be folded into the ONNX graph as a constant.

mask_threshold:
    

Threshold to convert predicted mask logits to binary masks. Will be folded into the ONNX graph as a constant.

mask_overlap_threshold:
    

Overlap area threshold for the predicted masks. Used to filter out or merge disconnected mask regions for every instance. Will be folded into the ONNX graph as a constant.

opset_version:
    

ONNX opset version to target. If None, PyTorchâs default opset is used.

simplify:
    

If True, run onnxslim to simplify and overwrite the exported model.

verify:
    

If True, validate the ONNX file and compare outputs to a float32 CPU reference forward pass.

format_args:
    

Optional extra keyword arguments forwarded to torch.onnx.export.

Returns:
    

None. Writes the ONNX model to out.

export_tensorrt(_out : str | Path_, _*_ , _precision : Literal['auto', 'fp32', 'fp16'] = 'auto'_, _onnx_args : dict[str, Any] | None = None_, _max_batchsize : int = 1_, _opt_batchsize : int = 1_, _min_batchsize : int = 1_, _verbose : bool = False_) → NoneÂ¶
    

Build a TensorRT engine from an ONNX model.

Note

TensorRT is not part of LightlyTrainâs dependencies and must be installed separately. Installation depends on your OS, Python version, GPU, and NVIDIA driver/CUDA setup. See the [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html) for more details. On CUDA 12.x systems you can often install the Python package via pip install tensorrt-cu12.

This loads the ONNX file, parses it with TensorRT, infers the static input shape (C, H, W) from the âimagesâ input, and creates an engine with a dynamic batch dimension in the range [min_batchsize, opt_batchsize, max_batchsize]. Spatial dimensions must be static in the ONNX model (dynamic H/W are not yet supported).

The engine is serialized and written to out.

Args:
    

out:
    

Path where the TensorRT engine will be saved.

precision:
    

Precision for ONNX export and TensorRT engine building. Either âautoâ, âfp32â, or âfp16â. âautoâ uses the modelâs current precision.

onnx_args:
    

Optional arguments to pass to export_onnx when exporting the ONNX model prior to building the TensorRT engine. If None, default arguments are used and the ONNX file is saved alongside the TensorRT engine with the same name but .onnx extension.

max_batchsize:
    

Maximum supported batch size.

opt_batchsize:
    

Batch size TensorRT optimizes for.

min_batchsize:
    

Minimum supported batch size.

verbose:
    

Enable verbose TensorRT logging.

predict(_image : str | Path | Image | Tensor_, _threshold : float = 0.8_, _mask_threshold : float = 0.5_, _mask_overlap_threshold : float = 0.8_) → dict[str, Tensor]Â¶
    

Returns the predicted mask for the given image.

Args:
    

image:
    

The input image as a path, URL, PIL image, or tensor. Tensors must have shape (C, H, W).

threshold:
    

The confidence threshold to keep predicted masks.

mask_threshold:
    

The threshold to convert predicted mask logits to binary masks.

mask_overlap_threshold:
    

The overlap area threshold for the predicted masks. Used to filter out or merge disconnected mask regions for every instance.

Returns:
    

A {âmasksâ: Tensor, âsegment_idsâ: Tensor, âscoresâ: Tensor} dict. Mask is a tensor of shape (H, W, 2) where the last dimension has two channels: \- Channel 0: class label per pixel \- Channel 1: segment id per pixel Segment ids are in [-1, num_unique_segment_ids - 1]. There can be multiple segments with the same id if they belong to the same stuff class. Id -1 indicates pixels without an assigned segment. Scores is a tensor of shape (num_segments,) containing the confidences score for each segment.

class lightly_train._task_models.dinov2_eomt_semantic_segmentation.task_model.DINOv2EoMTSemanticSegmentationÂ¶
    

export_onnx(_out : str | Path_, _*_ , _precision : Literal['auto', 'fp32', 'fp16'] = 'auto'_, _batch_size : int = 1_, _dynamic_batch_size : bool = True_, _height : int | None = None_, _width : int | None = None_, _opset_version : int | None = None_, _simplify : bool = True_, _verify : bool = True_, _format_args : dict[str, Any] | None = None_) → NoneÂ¶
    

Exports the model to ONNX for inference.

The export uses a dummy input of shape (batch_size, C, H, W) where C is inferred from the first model parameter and (H, W) come from self.image_size. If dynamic_batch_size is True, the ONNX graph will have a dynamic batch dimension for the input. The graph produces two outputs: masks and logits.

Optionally simplifies the exported model in-place using onnxslim and verifies numerical closeness against a float32 CPU reference via ONNX Runtime.

Args:
    

out:
    

Path where the ONNX model will be written.

precision:
    

Precision for the ONNX model. Either âautoâ, âfp32â, or âfp16â. âautoâ uses the modelâs current precision.

batch_size:
    

Batch size for the ONNX input.

dynamic_batch_size:
    

If True, the ONNX graph will have a dynamic batch dimension for the input. If False, the batch dimension is fixed to batch_size.

height:
    

Height of the ONNX input. If None, will be taken from self.image_size.

width:
    

Width of the ONNX input. If None, will be taken from self.image_size.

opset_version:
    

ONNX opset version to target. If None, PyTorchâs default opset is used.

simplify:
    

If True, run onnxslim to simplify and overwrite the exported model.

verify:
    

If True, validate the ONNX file and compare outputs to a float32 CPU reference forward pass.

format_args:
    

Optional extra keyword arguments forwarded to torch.onnx.export.

Returns:
    

None. Writes the ONNX model to out.

export_tensorrt(_out : str | Path_, _*_ , _precision : Literal['auto', 'fp32', 'fp16'] = 'auto'_, _onnx_args : dict[str, Any] | None = None_, _max_batchsize : int = 1_, _opt_batchsize : int = 1_, _min_batchsize : int = 1_, _verbose : bool = False_) → NoneÂ¶
    

Build a TensorRT engine from an ONNX model.

Note

TensorRT is not part of LightlyTrainâs dependencies and must be installed separately. Installation depends on your OS, Python version, GPU, and NVIDIA driver/CUDA setup. See the [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html) for more details. On CUDA 12.x systems you can often install the Python package via pip install tensorrt-cu12.

This loads the ONNX file, parses it with TensorRT, infers the static input shape (C, H, W) from the âimagesâ input, and creates an engine with a dynamic batch dimension in the range [min_batchsize, opt_batchsize, max_batchsize]. Spatial dimensions must be static in the ONNX model (dynamic H/W are not yet supported).

The engine is serialized and written to out.

Args:
    

out:
    

Path where the TensorRT engine will be saved.

precision:
    

Precision for ONNX export and TensorRT engine building. Either âautoâ, âfp32â, or âfp16â. âautoâ uses the modelâs current precision.

onnx_args:
    

Optional arguments to pass to export_onnx when exporting the ONNX model prior to building the TensorRT engine. If None, default arguments are used and the ONNX file is saved alongside the TensorRT engine with the same name but .onnx extension.

max_batchsize:
    

Maximum supported batch size.

opt_batchsize:
    

Batch size TensorRT optimizes for.

min_batchsize:
    

Minimum supported batch size.

verbose:
    

Enable verbose TensorRT logging.

Raises:
    

FileNotFoundError: If the ONNX file does not exist. RuntimeError: If the ONNX cannot be parsed or engine building fails. ValueError: If batch size constraints are invalid or H/W are dynamic.

predict(_image : str | Path | Image | Tensor_) → TensorÂ¶
    

Returns the predicted mask for the given image.

Args:
    

image:
    

The input image as a path, URL, PIL image, or tensor. Tensors must have shape (C, H, W).

Returns:
    

The predicted mask as a tensor of shape (H, W). The values represent the class IDs as defined in the classes argument of your dataset. These classes are also stored in the classes attribute of the model. The model will always predict the pixels as one of the known classes even when your dataset contains ignored classes defined by the ignore_classes argument.

class lightly_train._task_models.dinov3_eomt_semantic_segmentation.task_model.DINOv3EoMTSemanticSegmentationÂ¶
    

export_onnx(_out : str | Path_, _*_ , _precision : Literal['auto', 'fp32', 'fp16'] = 'auto'_, _batch_size : int = 1_, _dynamic_batch_size : bool = True_, _height : int | None = None_, _width : int | None = None_, _opset_version : int | None = None_, _simplify : bool = True_, _verify : bool = True_, _format_args : dict[str, Any] | None = None_) → NoneÂ¶
    

Exports the model to ONNX for inference.

The export uses a dummy input of shape (batch_size, C, H, W) where C is inferred from the first model parameter and (H, W) come from self.image_size. If dynamic_batch_size is True, the ONNX graph will have a dynamic batch dimension for the input. The graph produces two outputs: masks and logits.

Optionally simplifies the exported model in-place using onnxslim and verifies numerical closeness against a float32 CPU reference via ONNX Runtime.

Args:
    

out:
    

Path where the ONNX model will be written.

precision:
    

Precision for the ONNX model. Either âautoâ, âfp32â, or âfp16â. âautoâ uses the modelâs current precision.

batch_size:
    

Batch size for the ONNX input.

dynamic_batch_size:
    

If True, the ONNX graph will have a dynamic batch dimension for the input. If False, the batch dimension is fixed to batch_size.

height:
    

Height of the ONNX input. If None, will be taken from self.image_size.

width:
    

Width of the ONNX input. If None, will be taken from self.image_size.

opset_version:
    

ONNX opset version to target. If None, PyTorchâs default opset is used.

simplify:
    

If True, run onnxslim to simplify and overwrite the exported model.

verify:
    

If True, validate the ONNX file and compare outputs to a float32 CPU reference forward pass.

format_args:
    

Optional extra keyword arguments forwarded to torch.onnx.export.

Returns:
    

None. Writes the ONNX model to out.

export_tensorrt(_out : str | Path_, _*_ , _precision : Literal['auto', 'fp32', 'fp16'] = 'auto'_, _onnx_args : dict[str, Any] | None = None_, _max_batchsize : int = 1_, _opt_batchsize : int = 1_, _min_batchsize : int = 1_, _verbose : bool = False_) → NoneÂ¶
    

Build a TensorRT engine from an ONNX model.

Note

TensorRT is not part of LightlyTrainâs dependencies and must be installed separately. Installation depends on your OS, Python version, GPU, and NVIDIA driver/CUDA setup. See the [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html) for more details. On CUDA 12.x systems you can often install the Python package via pip install tensorrt-cu12.

This loads the ONNX file, parses it with TensorRT, infers the static input shape (C, H, W) from the âimagesâ input, and creates an engine with a dynamic batch dimension in the range [min_batchsize, opt_batchsize, max_batchsize]. Spatial dimensions must be static in the ONNX model (dynamic H/W are not yet supported).

The engine is serialized and written to out.

Args:
    

out:
    

Path where the TensorRT engine will be saved.

precision:
    

Precision for ONNX export and TensorRT engine building. Either âautoâ, âfp32â, or âfp16â. âautoâ uses the modelâs current precision.

onnx_args:
    

Optional arguments to pass to export_onnx when exporting the ONNX model prior to building the TensorRT engine. If None, default arguments are used and the ONNX file is saved alongside the TensorRT engine with the same name but .onnx extension.

max_batchsize:
    

Maximum supported batch size.

opt_batchsize:
    

Batch size TensorRT optimizes for.

min_batchsize:
    

Minimum supported batch size.

verbose:
    

Enable verbose TensorRT logging.

Raises:
    

FileNotFoundError: If the ONNX file does not exist. RuntimeError: If the ONNX cannot be parsed or engine building fails. ValueError: If batch size constraints are invalid or H/W are dynamic.

predict(_image : str | Path | Image | Tensor_) → TensorÂ¶
    

Returns the predicted mask for the given image.

Args:
    

image:
    

The input image as a path, URL, PIL image, or tensor. Tensors must have shape (C, H, W).

Returns:
    

The predicted mask as a tensor of shape (H, W). The values represent the class IDs as defined in the classes argument of your dataset. These classes are also stored in the classes attribute of the model. The model will always predict the pixels as one of the known classes even when your dataset contains ignored classes defined by the ignore_classes argument.
  *[*]: Keyword-only parameters separator (PEP 3102)
