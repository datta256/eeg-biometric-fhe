import inspect
import torch
from eeg_model import TemporalSpatialEEGEncoder

print("=== FORWARD METHOD ===")
print(inspect.getsource(TemporalSpatialEEGEncoder.forward))

print("\n=== MODEL MODULES ===")
model = TemporalSpatialEEGEncoder(channels=8, embedding_dim=32)
print(model)

print("\n=== CHECKPOINT COMPATIBILITY ===")
ckpt = torch.load(
    r"eeg_runs\arcface_custom_v1\best_checkpoint.pt",
    map_location="cpu",
    weights_only=False
)
result = model.load_state_dict(ckpt["encoder"], strict=True)
print("Encoder weights loaded:", result)

model.eval()
print("\n=== INPUT SHAPE TEST ===")

candidates = [
    (1, 8, 1500),
    (1, 1, 8, 1500),
    (1, 8, 1500, 1),
]

for shape in candidates:
    try:
        x = torch.zeros(shape, dtype=torch.float32)
        with torch.no_grad():
            y = model(x)
        print(
            "PASS:", shape,
            "-> output:", tuple(y.shape),
            "dtype:", y.dtype
        )
    except Exception as e:
        print("FAIL:", shape, "->", type(e).__name__, str(e)[:220])
