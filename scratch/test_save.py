import io
import torch
import ultralytics  # This applies the monkeypatch to torch.save!

state = {"test": torch.randn(10)}
buf = io.BytesIO()
try:
    torch.save(state, buf)
    print("Success")
except Exception as e:
    import traceback
    traceback.print_exc()
