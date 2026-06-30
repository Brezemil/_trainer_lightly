import io
import torch
import lightly_train
import ultralytics  # This applies the monkeypatch to torch.save!

print("Creating model...")
model = lightly_train.load_model("dinov3/vitl16-ltdetr")
print("Model created.")

print("Saving model state_dict...")
buf = io.BytesIO()
try:
    torch.save(model.state_dict(), buf)
    print("Success saving state_dict!")
except Exception as e:
    import traceback
    traceback.print_exc()
