"""Validate CKKS cosine similarity against the real EEG ArcFace encoder/API preprocessing."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import tenseal as ts

from eeg_model import TemporalSpatialEEGEncoder

CHANNELS = ["POZ", "PO3", "PO4", "PO7", "PO8", "OZ", "O1", "O2"]


def load_metadata(cache: Path):
    metadata = json.loads((cache / "metadata.json").read_text(encoding="utf-8"))
    if isinstance(metadata, dict):
        for key in ("trials", "records", "items", "data"):
            if isinstance(metadata.get(key), list):
                metadata = metadata[key]
                break
    if not metadata:
        raise RuntimeError("metadata.json contains no channel metadata")
    names = metadata[0]["channel_names"]
    return [names.index(c) for c in CHANNELS]


def load_embedding(subject, session, frequency, records_index, cache, channel_indices, encoder, device):
    key = (int(subject), int(session), round(float(frequency), 4))
    record = records_index.get(key)
    if record is None:
        raise RuntimeError(f"No manifest record for subject={subject}, session={session}, frequency={frequency}")
    path = cache / record["file"]
    if not path.is_file():
        raise FileNotFoundError(path)
    x = np.load(path, allow_pickle=False).astype(np.float32, copy=False)
    if x.shape != (64, 1500):
        raise RuntimeError(f"Unexpected trial shape {x.shape} in {path}; expected (64, 1500)")
    x = x[channel_indices, :]
    # Exactly mirror api.py preprocessing.
    x = (x - x.mean(axis=-1, keepdims=True)) / np.maximum(x.std(axis=-1, keepdims=True), 1e-6)
    tensor = torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0).to(device)
    with torch.no_grad():
        z = encoder(tensor).squeeze(0).detach().cpu().numpy().astype(np.float64)
    norm = float(np.linalg.norm(z))
    if not np.isfinite(norm) or norm < 1e-12:
        raise RuntimeError(f"Invalid embedding norm {norm}")
    return z / norm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--enroll-subject", type=int, default=None)
    p.add_argument("--probe-subject", type=int, default=None)
    p.add_argument("--enroll-session", type=int, default=3)
    p.add_argument("--probe-session", type=int, default=4)
    p.add_argument("--frequency", type=float, default=10.0)
    p.add_argument("--threshold", type=float, default=0.617122)
    p.add_argument("--tolerance", type=float, default=1e-3)
    args = p.parse_args()

    cache = Path(args.cache)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    records_index = {}
    for r in manifest["records"]:
        key = (int(r["subject"]), int(r["session"]), round(float(r["frequency"]), 4))
        if key in records_index:
            raise RuntimeError(f"Duplicate manifest record for {key}")
        records_index[key] = r

    available = sorted({
        int(r["subject"]) for r in manifest["records"]
        if int(r["session"]) in (args.enroll_session, args.probe_session)
        and round(float(r["frequency"]), 4) == round(float(args.frequency), 4)
    })
    if not available:
        raise RuntimeError(
            f"No subjects found for sessions {args.enroll_session}/{args.probe_session} "
            f"at {args.frequency} Hz. Check manifest and arguments."
        )
    if args.enroll_subject is None:
        args.enroll_subject = available[0]
    if args.probe_subject is None:
        args.probe_subject = args.enroll_subject
    print(f"Available subjects for this session/frequency: {available}")
    if args.enroll_subject not in available or args.probe_subject not in available:
        raise RuntimeError("Requested subject is unavailable for the selected session/frequency.")

    channel_indices = load_metadata(cache)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    encoder = TemporalSpatialEEGEncoder(channels=8, embedding_dim=32).to(device)
    encoder.load_state_dict(checkpoint["encoder"], strict=True)
    encoder.eval()

    enroll = load_embedding(args.enroll_subject, args.enroll_session, args.frequency,
                            records_index, cache, channel_indices, encoder, device)
    probe = load_embedding(args.probe_subject, args.probe_session, args.frequency,
                           records_index, cache, channel_indices, encoder, device)

    plaintext = float(np.dot(enroll, probe))
    ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192,
                     coeff_mod_bit_sizes=[60, 40, 40, 60])
    ctx.global_scale = 2**40
    ctx.generate_relin_keys()
    ctx.generate_galois_keys()
    private_bytes = ctx.serialize(save_public_key=True, save_secret_key=True,
                                  save_galois_keys=True, save_relin_keys=True)
    ctx.make_context_public()
    public_bytes = ctx.serialize(save_public_key=True, save_secret_key=False,
                                 save_galois_keys=True, save_relin_keys=True)
    public_ctx = ts.context_from(public_bytes)
    enc_enroll = ts.ckks_vector(public_ctx, enroll.tolist())
    enc_probe = ts.ckks_vector(public_ctx, probe.tolist())
    encrypted_score = enc_probe.dot(enc_enroll)

    private_ctx = ts.context_from(private_bytes)
    fhe_score = float(ts.ckks_vector_from(private_ctx, encrypted_score.serialize()).decrypt()[0])
    error = abs(plaintext - fhe_score)
    label = "genuine/same identity" if args.enroll_subject == args.probe_subject else "impostor/different identities"

    print("REAL EEG FHE VALIDATION")
    print(f"Device:                    {device}")
    print(f"Enrollment:                subject={args.enroll_subject}, session={args.enroll_session}")
    print(f"Probe:                     subject={args.probe_subject}, session={args.probe_session}")
    print(f"Pair type:                 {label}")
    print(f"Frequency:                 {args.frequency} Hz")
    print(f"Plaintext cosine:          {plaintext:.10f}")
    print(f"FHE-decrypted cosine:      {fhe_score:.10f}")
    print(f"Absolute error:            {error:.3e}")
    print(f"Decision at threshold:     {'ACCEPT' if fhe_score >= args.threshold else 'REJECT'}")
    print(f"Threshold (demo default):  {args.threshold:.6f}")
    if error <= args.tolerance:
        print(f"PASS: FHE score matches plaintext within {args.tolerance:g}.")
    else:
        print(f"FAIL: error exceeds tolerance {args.tolerance:g}.")
        raise SystemExit(2)
    print("Note: this single-pair test validates arithmetic, not biometric security or FAR/FRR.")

if __name__ == "__main__":
    main()
