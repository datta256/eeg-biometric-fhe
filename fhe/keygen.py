from pathlib import Path
import argparse
import tenseal as ts

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="fhe_keys")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192,
                     coeff_mod_bit_sizes=[60, 40, 40, 60])
    ctx.global_scale = 2**40
    ctx.generate_relin_keys()
    ctx.generate_galois_keys()
    (out / "private_context.bin").write_bytes(ctx.serialize(
        save_public_key=True, save_secret_key=True,
        save_galois_keys=True, save_relin_keys=True))
    ctx.make_context_public()
    (out / "public_context.bin").write_bytes(ctx.serialize(
        save_public_key=True, save_secret_key=False,
        save_galois_keys=True, save_relin_keys=True))
    print(f"Created keys in {out.resolve()}")
    print("Keep private_context.bin secret; share only public_context.bin.")

if __name__ == "__main__":
    main()
