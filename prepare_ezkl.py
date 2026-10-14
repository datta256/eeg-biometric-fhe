import json
import traceback
from pathlib import Path

import numpy as np
import ezkl

root = Path("zkml_eeg")
root.mkdir(exist_ok=True)

model_path = str(Path("eeg_encoder.onnx").resolve())
settings_path = str((root / "settings.json").resolve())
cal_data_path = str((root / "calibration.json").resolve())

if not Path(model_path).exists():
    raise FileNotFoundError(f"ONNX model not found: {model_path}")

# Representative normalized inputs; these are synthetic, not real EEG.
rng = np.random.default_rng(20261014)
samples = []

for _ in range(3):
    x = rng.normal(size=(1, 8, 1500)).astype(np.float32)
    x = (x - x.mean(axis=-1, keepdims=True)) / np.maximum(
        x.std(axis=-1, keepdims=True), 1e-6
    )
    samples.append(x.reshape(-1).astype(float).tolist())

with open(cal_data_path, "w", encoding="utf-8") as f:
    json.dump({"input_data": samples}, f)

print("ONNX:", model_path)
print("Calibration inputs:", len(samples))
print("Values per input:", len(samples[0]))
print("Settings:", settings_path)

try:
    print("\n=== GENERATE SETTINGS ===")
    result = ezkl.gen_settings(model_path, settings_path)
    print("gen_settings returned:", result)
    if result is False:
        raise RuntimeError("EZKL settings generation returned False")

    print("\n=== CALIBRATE ===")
    result = ezkl.calibrate_settings(
        cal_data_path,
        model_path,
        settings_path,
        target="resources",
        lookup_safety_margin=4,
    )
    print("calibrate_settings returned:", result)
    if result is False:
        raise RuntimeError("EZKL calibration returned False")

    print("\nSETTINGS AND CALIBRATION COMPLETED")
    print("Next artifact:", settings_path)

except Exception:
    print("\nEZKL FAILED:")
    traceback.print_exc()
    raise
