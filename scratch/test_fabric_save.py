import os
import io
import torch
import lightning_fabric
import ultralytics  # This applies the monkeypatch to torch.save!

print("Initializing Fabric...")
fabric = lightning_fabric.Fabric(accelerator="cpu", precision="bf16-mixed")
fabric.launch()

print("Creating dummy state (600MB)...")
state = {
    "model": {f"weight_{i}": torch.randn(1000, 1000, dtype=torch.bfloat16) for i in range(300)}
}

print("Saving checkpoint via fabric.save...")
ckpt_path = "runs/test_fabric_save.ckpt"
os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
try:
    fabric.save(ckpt_path, state)
    print("Success saving via Fabric!")
except Exception as e:
    import traceback
    traceback.print_exc()
