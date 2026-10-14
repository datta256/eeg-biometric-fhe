from pathlib import Path
import argparse, json
import numpy as np
import tenseal as ts

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--public-context", default="fhe_keys/public_context.bin")
    p.add_argument("--input", required=True, help="JSON list of 32 floats")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    ctx = ts.context_from(Path(args.public_context).read_bytes())
    x = np.asarray(json.loads(Path(args.input).read_text(encoding="utf-8")), dtype=np.float64)
    if x.shape != (32,,) or not np.all(np.isfinite(x)):
        raise SystemExit(f"Expected exactly 32 finite values; got shape {x.shape}")
    norm = float(np.linalg.norm(x))
    if not np.isclose(norm, 1.0, atol=2e-3):
        raise SystemExit(f"Expected an L2-normalized embedding; norm={norm:.6f}")
    encrypted = ts.ckks_vector(ctx, x.tolist())
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(encrypted.serialize())
    print(f"Encrypted 32-D embedding -> {out}")

if __name__ == "__main__":
    main()
