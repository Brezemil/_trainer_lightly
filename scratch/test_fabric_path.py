import os
import io
import torch
import lightning_fabric
import ultralytics

print("Initializing Fabric...")
fabric = lightning_fabric.Fabric(accelerator="cpu", precision="bf16-mixed")
fabric.launch()

state = {"model": {"weight": torch.randn(10, dtype=torch.bfloat16)}}

ckpt_path = r"C:\Users\emilb\_trainer_lightly\runs\facebook\dinov3-vitl16-pretrain-sat493m_seed_42_rtdetrv2\checkpoints\last_test.ckpt"
os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
try:
    fabric.save(ckpt_path, state)
    print("Success saving to long path!")
except Exception as e:
    import traceback
    traceback.print_exc()
