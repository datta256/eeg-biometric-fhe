import os
import inspect
import importlib.util
import torch
import eeg_model

print("=== MODEL ===")
print("File:", eeg_model.__file__)

if hasattr(eeg_model, "TemporalSpatialEEGEncoder"):
    cls = eeg_model.TemporalSpatialEEGEncoder
    print("Constructor:", inspect.signature(cls))
else:
    print("ERROR: TemporalSpatialEEGEncoder not found")

print("\n=== CHECKPOINT ===")
path = r"eeg_runs\arcface_custom_v1\best_checkpoint.pt"
print("Exists:", os.path.exists(path), path)

if os.path.exists(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    print("Checkpoint type:", type(ckpt).__name__)

    if isinstance(ckpt, dict):
        print("Top-level keys:", list(ckpt.keys()))

        for key, value in ckpt.items():
            if isinstance(value, dict) and value:
                tensor_items = [
                    (k, tuple(v.shape))
                    for k, v in value.items()
                    if isinstance(v, torch.Tensor)
                ]
                if tensor_items:
                    print(f"\nPossible state dict: {key}")
                    print("Tensor count:", len(tensor_items))
                    print("First 12 tensors:", tensor_items[:12])

print("\n=== DEPENDENCIES ===")
for name in ["onnx", "onnxruntime", "ezkl"]:
    print(name, "INSTALLED" if importlib.util.find_spec(name) else "MISSING")

print("\nPyTorch:", torch.__version__)
