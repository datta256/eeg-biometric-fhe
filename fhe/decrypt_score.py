from pathlib import Path
import argparse
import tenseal as ts

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--private-context", default="fhe_keys/private_context.bin")
    p.add_argument("--input", required=True)
    p.add_argument("--threshold", type=float, default=0.617122)
    args = p.parse_args()
    ctx = ts.context_from(Path(args.private_context).read_bytes())
    encrypted = ts.ckks_vector_from(ctx, Path(args.input).read_bytes())
    values = encrypted.decrypt()
    if not values:
        raise SystemExit("Decrypted score was empty")
    score = float(values[0])
    print(f"Cosine similarity: {score:.6f}")
    print(f"Threshold:         {args.threshold:.6f}")
    print(f"Decision:          {'ACCEPT' if score >= args.threshold else 'REJECT'}")
    print("Client-side decision after decrypting the score.")

if __name__ == "__main__":
    main()
