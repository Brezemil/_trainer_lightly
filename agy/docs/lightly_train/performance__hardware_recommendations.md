---
source_url: https://docs.lightly.ai/train/stable/performance/hardware_recommendations.html
---

# Hardware RecommendationsÂ¶

We tested Lightly**Train** in various setups, the following training examples with hardware setups are provided as a performance reference:

  * **Distillation YOLOv8l on 2 Ã NVIDIA RTX 4090**

    * `method='distillation'`

    * `batch_size=512`

    * `model='ultralytics/yolov8l.yaml'`

    * `precision='bf16-mixed'`

    * Dataset Size: 1 million images

    * GPUs: 2 Ã NVIDIA RTX 4090

    * GPU Memory: ~16GB per GPU

    * Time per Epoch: ~19 minutes

  * **Distillation YOLO11x on 4 Ã NVIDIA H100**

    * `method='distillation'`

    * `batch_size=2048`

    * `model='ultralytics/yolov11x.yaml'`

    * `precision='32-true'`

    * Dataset Size: 1 million images

    * GPUs: 4 Ã NVIDIA H100

    * GPU Memory: ~80GB per GPU

    * Time per Epoch: ~6 minutes




Those setups deliver efficient training while handling large-scale datasets with high throughput. While smaller setups can be used, training times may increase significantly and batch size might need to be reduced to fit GPU memory.
