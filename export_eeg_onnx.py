import os
import numpy as np
import torch
import onnx
import onnxruntime as ort
from eeg_model import TemporalSpatialEEGEncoder

torch.manual_seed(2026)
np.random.seed(2026)

checkpoint_path = r"eeg_runs\arcface_custom_v1\best_checkpoint.pt"
onnx_path = r"eeg_encoder.onnx"

ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
model = TemporalSpatialEEGEncoder(channels=8, embedding_dim=32)
model.load_state_dict(ckpt["encoder"], strict=True)
model.eval()

# Fixed batch size for the first proof experiment.
dummy = torch.randn(1, 8, 1500, dtype=torch.float32)

print("Exporting ONNX...")
with torch.no_grad():
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["eeg"],
        output_names=["embedding"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False
    )

onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)
print("ONNX checker: PASS")
print("ONNX file:", os.path.abspath(onnx_path))
print("ONNX size:", round(os.path.getsize(onnx_path) / 1024, 1), "KB")

session = ort.InferenceSession(
    onnx_path,
    providers=["CPUExecutionProvider"]
)

# Test several channel-wise z-scored inputs, matching your preprocessing.
max_abs_errors = []
for seed in [10, 20, 30]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(1, 8, 1500)).astype(np.float32)
    x = (x - x.mean(axis=-1, keepdims=True)) / np.maximum(
        x.std(axis=-1, keepdims=True), 1e-6
    )

    with torch.no_grad():
        torch_y = model(torch.from_numpy(x)).cpu().numpy()

    onnx_y = session.run(["embedding"], {"eeg": x})[0]
    max_err = float(np.max(np.abs(torch_y - onnx_y)))
    mean_err = float(np.mean(np.abs(torch_y - onnx_y)))
    norm = float(np.linalg.norm(onnx_y))

    max_abs_errors.append(max_err)
    print(
        f"seed={seed}: max_abs_error={max_err:.8g}, "
        f"mean_abs_error={mean_err:.8g}, "
        f"onnx_embedding_norm={norm:.8f}"
    )

print("\nMaximum error across tests:", max(max_abs_errors))
print("Parity check:", "PASS" if max(max_abs_errors) < 1e-4 else "REVIEW")
print("Expected input: [1, 8, 1500]")
print("Expected output: [1, 32]")
