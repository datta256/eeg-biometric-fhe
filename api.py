import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from eeg_model import TemporalSpatialEEGEncoder

CHANNELS = ["POZ", "PO3", "PO4", "PO7", "PO8", "OZ", "O1", "O2"]
FREQUENCIES = [8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0]

parser = argparse.ArgumentParser()
parser.add_argument("--cache", required=True)
parser.add_argument("--manifest", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--threshold", type=float, default=0.617122,
                    help="Default calibrated threshold from seen-identity 1%% FAR calibration.")
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=8000)
args = parser.parse_args()

CACHE = Path(args.cache)
MANIFEST_PATH = Path(args.manifest)
CHECKPOINT_PATH = Path(args.checkpoint)
THRESHOLD = float(args.threshold)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
records = manifest["records"]
with (CACHE / "metadata.json").open("r", encoding="utf-8") as f:
    metadata = json.load(f)
if isinstance(metadata, dict):
    for key in ("trials", "records", "items", "data"):
        if isinstance(metadata.get(key), list):
            metadata = metadata[key]
            break
channel_names = metadata[0]["channel_names"]
channel_indices = [channel_names.index(c) for c in CHANNELS]

records_index = {}
for r in records:
    key = (int(r["subject"]), int(r["session"]), round(float(r["frequency"]), 4))
    if key in records_index:
        raise RuntimeError(f"Duplicate manifest record for {key}")
    records_index[key] = r

checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
encoder = TemporalSpatialEEGEncoder(channels=8, embedding_dim=32).to(DEVICE)
encoder.load_state_dict(checkpoint["encoder"], strict=True)
encoder.eval()

split = manifest.get("identity_split", {})
seen_subjects = sorted(map(int, split.get("seen_subjects", [])))
unseen_subjects = sorted(map(int, split.get("unseen_subjects", [])))
all_subjects = sorted({int(r["subject"]) for r in records})

app = FastAPI(title="EEG ArcFace Wallet Demo", version="0.1.0")
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/api/config")
def get_config():
    return {
        "device": str(DEVICE),
        "threshold": THRESHOLD,
        "threshold_source": "Seen-identity calibration threshold previously measured for nominal 1% FAR",
        "subjects": all_subjects,
        "seen_subjects": seen_subjects,
        "unseen_subjects": unseen_subjects,
        "sessions": sorted({int(r["session"]) for r in records}),
        "frequencies": FREQUENCIES,
        "channels": CHANNELS,
        "embedding_dim": 32,
        "checkpoint_epoch": checkpoint.get("epoch"),
    }


def get_embedding(subject: int, session: int, frequency: float) -> np.ndarray:
    key = (int(subject), int(session), round(float(frequency), 4))
    r = records_index.get(key)
    if r is None:
        raise HTTPException(404, f"No trial for subject={subject}, session={session}, frequency={frequency}")
    path = CACHE / r["file"]
    if not path.is_file():
        raise HTTPException(404, f"Cached EEG trial not found: {path}")
    x = np.load(path, allow_pickle=False).astype(np.float32, copy=False)
    if x.shape != (64, 1500):
        raise HTTPException(500, f"Unexpected trial shape {x.shape}; expected (64, 1500)")
    x = x[channel_indices, :]
    x = (x - x.mean(axis=-1, keepdims=True)) / np.maximum(x.std(axis=-1, keepdims=True), 1e-6)
    tensor = torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        z = encoder(tensor).squeeze(0).detach().cpu().numpy().astype(np.float64)
    return z


class VerifyRequest(BaseModel):
    enroll_subject: int
    enroll_session: int = Field(3, ge=0)
    probe_subject: int
    probe_session: int = Field(4, ge=0)
    frequency: float
    threshold: float | None = None


@app.post("/api/verify")
def verify(req: VerifyRequest):
    if req.enroll_session == req.probe_session:
        raise HTTPException(400, "Choose a different session for the unlock attempt.")
    if round(float(req.frequency), 4) not in {round(f, 4) for f in FREQUENCIES}:
        raise HTTPException(400, "Unsupported SSVEP frequency.")
    enrollment = get_embedding(req.enroll_subject, req.enroll_session, req.frequency)
    probe = get_embedding(req.probe_subject, req.probe_session, req.frequency)
    denom = max(float(np.linalg.norm(enrollment) * np.linalg.norm(probe)), 1e-12)
    score = float(np.dot(enrollment, probe) / denom)
    threshold = THRESHOLD if req.threshold is None else float(req.threshold)
    passed = score >= threshold
    return {
        "score": score,
        "threshold": threshold,
        "passed": passed,
        "decision": "UNLOCK ALLOWED" if passed else "ACCESS DENIED",
        "enroll_subject": req.enroll_subject,
        "enroll_session": req.enroll_session,
        "probe_subject": req.probe_subject,
        "probe_session": req.probe_session,
        "frequency": req.frequency,
        "device": str(DEVICE),
        "score_meaning": "Cosine similarity between frozen EEG embeddings; higher means more similar.",
        "security_note": "This is a research demo gate, not production authentication or a cryptographic biometric key.",
    }


@app.post("/api/verify-fhe")
def verify_fhe(req: VerifyRequest):
    """Research demo: CKKS-encrypt the two embeddings and evaluate their dot product.

    IMPORTANT: this endpoint creates and retains the secret key in the same API
    process so it can decrypt the score for numerical comparison. It demonstrates
    CKKS arithmetic, NOT a distributed-key or production privacy boundary.
    The EEG encoder itself runs in plaintext before the embeddings are encrypted.
    """
    try:
        import tenseal as ts
    except ImportError as exc:
        raise HTTPException(503, "TenSEAL is not installed in this Python environment.") from exc

    if req.enroll_session == req.probe_session:
        raise HTTPException(400, "Choose a different session for the unlock attempt.")
    if round(float(req.frequency), 4) not in {round(f, 4) for f in FREQUENCIES}:
        raise HTTPException(400, "Unsupported SSVEP frequency.")

    started = time.perf_counter()
    enrollment = get_embedding(req.enroll_subject, req.enroll_session, req.frequency)
    probe = get_embedding(req.probe_subject, req.probe_session, req.frequency)
    plain_score = float(np.dot(enrollment, probe))
    inference_seconds = time.perf_counter() - started

    fhe_started = time.perf_counter()
    try:
        # Demo-only CKKS setup. The secret key remains in this process and is
        # deliberately NOT returned by this API. A real validator network needs
        # threshold CKKS key generation/decryption, not this single-key setup.
        ctx = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60],
        )
        ctx.global_scale = 2**40
        ctx.generate_relin_keys()
        ctx.generate_galois_keys()
        private_bytes = ctx.serialize(
            save_public_key=True, save_secret_key=True,
            save_galois_keys=True, save_relin_keys=True,
        )
        ctx.make_context_public()
        public_bytes = ctx.serialize(
            save_public_key=True, save_secret_key=False,
            save_galois_keys=True, save_relin_keys=True,
        )
        public_ctx = ts.context_from(public_bytes)
        enc_enrollment = ts.ckks_vector(public_ctx, enrollment.tolist())
        enc_probe = ts.ckks_vector(public_ctx, probe.tolist())
        encrypted_score = enc_probe.dot(enc_enrollment)
        encrypted_score_bytes = encrypted_score.serialize()
        private_ctx = ts.context_from(private_bytes)
        fhe_score = float(
            ts.ckks_vector_from(private_ctx, encrypted_score_bytes).decrypt()[0]
        )
    except Exception as exc:
        raise HTTPException(500, f"CKKS demo failed: {type(exc).__name__}: {exc}") from exc

    fhe_seconds = time.perf_counter() - fhe_started
    threshold = THRESHOLD if req.threshold is None else float(req.threshold)
    error = abs(plain_score - fhe_score)
    passed = fhe_score >= threshold
    return {
        "mode": "CKKS FHE demonstration",
        "score": fhe_score,
        "plaintext_score": plain_score,
        "fhe_score": fhe_score,
        "absolute_error": error,
        "threshold": threshold,
        "passed": passed,
        "decision": "UNLOCK ALLOWED" if passed else "ACCESS DENIED",
        "enroll_subject": req.enroll_subject,
        "enroll_session": req.enroll_session,
        "probe_subject": req.probe_subject,
        "probe_session": req.probe_session,
        "frequency": req.frequency,
        "device": str(DEVICE),
        "inference_seconds": inference_seconds,
        "fhe_seconds": fhe_seconds,
        "encrypted_score_bytes": len(encrypted_score_bytes),
        "privacy_note": (
            "EEG encoder inference ran in plaintext; only the resulting embeddings "
            "were CKKS-encrypted. This demo decrypts the score inside the same API "
            "process using a single secret key. No threshold validators, ZK proof, "
            "or distributed key shares are implemented."
        ),
        "security_note": (
            "Research demo only. This endpoint validates CKKS arithmetic and does "
            "not establish end-to-end biometric privacy or production authentication."
        ),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)
