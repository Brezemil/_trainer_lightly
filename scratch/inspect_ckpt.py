import lightly_train as lt

ckpt_path = r"C:\Users\emilb\Documents\GitHub\_trainer_lightly\runs\facebook\dinov3-vitl16-pretrain-sat493m_seed_42_rtdetrv2\exported_models\exported_best.pt"

print("Attempting to load model using lightly_train.load_model...")
model = lt.load_model(ckpt_path)
print("Successfully loaded model!")
