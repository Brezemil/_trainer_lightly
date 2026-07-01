import io
import torch

state = {"test": torch.randn(10)}
buf = io.BytesIO()
try:
    torch.save(state, buf)
    print("Success")
except Exception:
    import traceback

    traceback.print_exc()
