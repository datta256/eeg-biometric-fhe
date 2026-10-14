from pathlib import Path
import argparse
import tenseal as ts

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--public-context", default="fhe_keys/public_context.bin")
    p.add_argument("--template", required=True)
    p.add_argument("--probe", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    ctx = ts.context_from(Path(args.public_context).read_bytes())
    template = ts.ckks_vector_from(ctx, Path(args.template).read_bytes())
    probe = ts.ckks_vector_from(ctx, Path(args.probe).read_bytes())
    # For normalized vectors, dot product is cosine similarity.
    score = probe.dot(template)
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(score.serialize())
    print(f"Encrypted similarity computed -> {out}")
    print("Score remains encrypted; evaluator cannot decrypt it.")

if __name__ == "__main__":
    main()
