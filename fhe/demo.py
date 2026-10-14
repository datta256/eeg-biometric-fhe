import numpy as np
import tenseal as ts

def main():
    rng = np.random.default_rng(20261014)
    template = rng.normal(size=32); template /= np.linalg.norm(template)
    probe = template + 0.12 * rng.normal(size=32); probe /= np.linalg.norm(probe)
    plain = float(np.dot(probe, template))
    ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192,
                     coeff_mod_bit_sizes=[60, 40, 40, 60])
    ctx.global_scale = 2**40
    ctx.generate_relin_keys(); ctx.generate_galois_keys()
    private_bytes = ctx.serialize(save_public_key=True, save_secret_key=True,
                                  save_galois_keys=True, save_relin_keys=True)
    ctx.make_context_public()
    public_bytes = ctx.serialize(save_public_key=True, save_secret_key=False,
                                 save_galois_keys=True, save_relin_keys=True)
    public = ts.context_from(public_bytes)
    encrypted_score = ts.ckks_vector(public, probe.tolist()).dot(
        ts.ckks_vector(public, template.tolist()))
    private = ts.context_from(private_bytes)
    actual = float(ts.ckks_vector_from(private, encrypted_score.serialize()).decrypt()[0])
    error = abs(plain - actual)
    print(f"Plaintext cosine similarity: {plain:.8f}")
    print(f"FHE-decrypted similarity:    {actual:.8f}")
    print(f"Absolute error:              {error:.3e}")
    if error > 1e-3:
        raise SystemExit("FAIL: CKKS error exceeded 1e-3")
    print("PASS: encrypted dot product agrees with plaintext within tolerance.")

if __name__ == "__main__":
    main()
