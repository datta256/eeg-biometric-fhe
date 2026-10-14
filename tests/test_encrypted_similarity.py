import numpy as np
import tenseal as ts

def test_encrypted_dot_product_close_to_plaintext():
    rng = np.random.default_rng(7)
    a = rng.normal(size=32); a /= np.linalg.norm(a)
    b = rng.normal(size=32); b /= np.linalg.norm(b)
    expected = float(np.dot(a, b))
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
    result = ts.ckks_vector(public, a.tolist()).dot(ts.ckks_vector(public, b.tolist()))
    private = ts.context_from(private_bytes)
    actual = float(ts.ckks_vector_from(private, result.serialize()).decrypt()[0])
    assert abs(expected - actual) < 1e-3
