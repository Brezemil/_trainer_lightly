---
source_url: https://docs.lightly.ai/train/stable/pretrain_distill/models/rtdetr.html
---

# RT-DETRÂ¶

This page describes how to use the [official RT-DETR implementation](https://github.com/lyuwenyu/RT-DETR) with LightlyTrain.

Note

RT-DETR is not a pip-installable Python package. For this reason, RT-DETR is not fully integrated with LightlyTrain and has to be installed manually, however LightlyTrain provides a model wrapper for the RT-DETR models.

Warning

Due to an incompatibility between the torchvision version requirements of RT-DETRv1, we support the RT-DETRv2 implementation that is backward compatible and allows the training of both RT-DETR(v1) and RT-DETRv2 models.

## RT-DETR InstallationÂ¶

To install RT-DETR for use with LightlyTrain, first clone the repository:
    
    
    git clone https://github.com/lyuwenyu/RT-DETR.git
    

We assume that all commands are executed from the `RT-DETR/rtdetrv2_pytorch` directory unless otherwise stated:
    
    
    cd RT-DETR/rtdetrv2_pytorch
    

Next, create a Python environment using your preferred tool and install the required dependencies. Some dependencies are fixed to specific versions to ensure compatibility with RT-DETR:

Note

Due to recent changes in the RT-DETR repository, RT-DETRâs dependencies are compatible with Python >=3.8 and <=3.10. However, if you donât need TensorRT support, you can simply comment out the `tensorrt` dependency in the `requirements.txt` file and you should be able to use Python 3.11 or 3.12 as well.
    
    
    pip install lightly-train -r requirements.txt
    

## Pretrain and Fine-tune an RT-DETR ModelÂ¶

### PretrainÂ¶

Run the following script to pretrain an RT-DETR model. The script must either be executed from inside the `RT-DETR/rtdetrv2_pytorch` directory or the `RT-DETR/rtdetrv2_pytorch` directory must be added to the Python path.
    
    
    from src.core import YAMLConfig
    
    import lightly_train
    from lightly_train.model_wrappers import RTDETRModelWrapper
    
    if __name__ == "__main__":
        # Load the RT-DETR model
        # Change the config file to the desired RT-DETR model
        config = YAMLConfig("configs/rtdetr/rtdetr_r50vd_6x_coco.yml")
        config.yaml_cfg["PResNet"]["pretrained"] = False
        config.yaml_cfg["PResNet"]["freeze_norm"] = False
        config.yaml_cfg["PResNet"]["freeze_at"] = -1
        model = config.model
    
        # Pretrain the model
        wrapped_model = RTDETRModelWrapper(model)
        lightly_train.pretrain(
            out="out/my_experiment",
            data="my_data_dir", # Replace with your dataset path.
            model=wrapped_model,
        )
    

### Fine-TuneÂ¶

After the pretraining completes, the model can be fine-tuned using the default RT-DETR training script by providing the path to the pretrained model:
    
    
    # Training on single-gpu
    export CUDA_VISIBLE_DEVICES=0
    python tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml --resume out/my_experiment/exported_models/exported_last.pt
    

See the [RT-DETR repository](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch) for more information on how to fine-tune a model.

## Supported ModelsÂ¶

The following RT-DETR model variants are supported:

  * RT-DETR

    * `rtdetr/rtdetr_r18vd_6x_coco.yml`

    * `rtdetr/rtdetr_r34vd_6x_coco.yml`

    * `rtdetr/rtdetr_r50vd_6x_coco.yml`

    * `rtdetr/rtdetr_r50vd_m_6x_coco.yml`

    * `rtdetr/rtdetr_r101vd_6x_coco.yml`

  * RT-DETRv2

    * `rtdetrv2/rtdetrv2_r18vd_120e_coco.yml`

    * `rtdetrv2/rtdetrv2_r18vd_120e_voc.yml`

    * `rtdetrv2/rtdetrv2_r18vd_dsp_3x_coco.yml`

    * `rtdetrv2/rtdetrv2_r18vd_sp1_120e_coco.yml`

    * `rtdetrv2/rtdetrv2_r18vd_sp2_120e_coco.yml`

    * `rtdetrv2/rtdetrv2_r18vd_sp3_120e_coco.yml`

    * `rtdetrv2/rtdetrv2_r34vd_120e_coco.yml`

    * `rtdetrv2/rtdetrv2_r34vd_dsp_1x_coco.yml`

    * `rtdetrv2/rtdetrv2_r50vd_6x_coco.yml`

    * `rtdetrv2/rtdetrv2_r50vd_dsp_1x_coco.yml`

    * `rtdetrv2/rtdetrv2_r50vd_m_7x_coco.yml`

    * `rtdetrv2/rtdetrv2_r50vd_m_dsp_3x_coco.yml`

    * `rtdetrv2/rtdetrv2_r101vd_6x_coco.yml`



