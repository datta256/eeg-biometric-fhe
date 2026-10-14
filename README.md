# EEG Biometrics with ArcFace-Style Embeddings and CKKS Homomorphic Encryption

**A research prototype exploring whether an EEG-based biometric comparison can be performed on encrypted embeddings.** The repository combines a compact temporal-spatial EEG encoder, cross-session verification experiments, a FastAPI browser demo, and a CKKS encrypted-dot-product demonstration using TenSEAL.

> **Status: research prototype — not production authentication.** The browser demo uses pre-recorded research trials. It is not a live EEG acquisition system, a cryptographic identity wallet, a ZKML proof system, or a distributed validator network.

## Why this project?

Biometric systems usually turn a measurement into a template and compare that template during verification. Templates can become sensitive data: if exposed, they may be difficult or impossible to replace. This project investigates one narrow part of a privacy-preserving alternative: compute a similarity score over encrypted neural embeddings rather than exposing the embedding values to the arithmetic layer.

The experiment has two separate parts:

1. **EEG representation and verification:** preprocess eight posterior channels and map a trial to a 32-dimensional embedding; compare enrollment and probe embeddings across sessions.
2. **Encrypted similarity arithmetic:** use CKKS approximate homomorphic encryption to calculate a dot product on encrypted embeddings, then compare its decrypted result with the plaintext result.

These are not the same as end-to-end private inference. In this prototype, the EEG encoder runs in plaintext, and the demo API holds the CKKS secret key so it can decrypt the result. Those limitations are deliberate and documented.

## What is included

- EEG encoder and inference code (`eeg_model.py`, `eeg_dataset.py`)
- Training, export, smoke-test, and evaluation scripts
- FastAPI demo and browser UI (`api.py`, `static/index.html`)
- CKKS utilities and encrypted-match examples (`fhe/`)
- Held-out identity and multi-frequency / multi-session evaluation scripts
- Experiment manifest and selected result artifacts
- `MEDIUM_ARTICLE.md`, a publication-ready article draft describing the design and caveats

## Experiment setup

- **Input:** one preprocessed trial with 8 channels × 1,500 samples
- **Channels:** POZ, PO3, PO4, PO7, PO8, OZ, O1, O2
- **Frequencies:** 8–12 Hz in 0.5 Hz increments (9 frequencies)
- **Representation:** 32-dimensional EEG embedding
- **Similarity:** dot product / cosine similarity, depending on the evaluation path; embeddings used for cosine verification are normalized as documented in the scripts
- **Encryption:** TenSEAL CKKS, polynomial modulus degree 8192, coefficient modulus bits `[60, 40, 40, 60]`, scale `2**40`
- **Web API:** FastAPI with plaintext baseline and CKKS demo modes

The project expects a local EEG cache and manifest. The dataset itself is not bundled as raw recordings. See **Setup** below.

## Results snapshot

The recorded multi-condition evaluation covered **9 frequencies × 10 session-pair conditions = 90 conditions** on held-out identities, with up to 200 sampled impostor comparisons per condition for the FHE arithmetic check.

| Metric | Recorded result |
|---|---:|
| Mean AUC across 90 conditions | 0.9882 |
| Mean descriptive EER across 90 conditions | 4.89% |
| FHE pair evaluations | 21,600 |
| Mean absolute plaintext-vs-FHE score error | 2.412 × 10⁻⁶ |
| Maximum absolute score error | 2.609 × 10⁻⁶ |
| FHE decision flips in tested pairs | 0 |

These are **experimental results, not a claim of production-grade biometric accuracy**. Subjects and session/frequency conditions recur across evaluations, so the rows are not independent samples. The threshold was not independently calibrated for every condition; some conditions have materially higher false-reject rates. The evaluation should be repeated with a pre-registered protocol, subject-independent splits, confidence intervals, and an independently calibrated operating threshold before making comparative or deployment claims.

The FHE numbers show numerical agreement for the tested encrypted dot-product operation. They do **not** show that raw EEG stays encrypted during model inference, that the API operator cannot decrypt the score, or that the system is secure against malicious servers or validators.

## Setup

Use Python 3.10–3.12 in a virtual environment. The full training/evaluation stack may require additional dependencies beyond the minimal FHE demo requirements.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install fastapi uvicorn torch
```

For the web demo, you need a compatible EEG cache, `metadata.json`, the experiment manifest, and a trained checkpoint. These paths are supplied explicitly at startup; update them to match your machine:

```powershell
python api.py `
  --cache "E:\path\to\ssvep-64ch-cache" `
  --manifest ".\eeg_experiment_manifest.json" `
  --checkpoint ".\eeg_runs\arcface_custom_v1\best_checkpoint.pt" `
  --host 127.0.0.1 `
  --port 8000
```

Open <http://127.0.0.1:8000>. Choose **Plaintext baseline** or **CKKS FHE demo** in the interface.

### Important setup notes

- `requirements.txt` currently captures the minimal numerical / FHE dependencies. Install the compatible PyTorch and FastAPI packages separately as shown above.
- The manifest must point to trials available in your local cache. Each trial is expected to have shape `(64, 1500)` and metadata must include the channel names listed above.
- Do not commit private keys, raw EEG, licensed datasets, personal data, or local cache files.
- The demo generates CKKS keys at request time for demonstration, which adds latency. This is not a recommended production key-management design.

## API

- `GET /api/config` — reports demo configuration and available manifest metadata.
- `POST /api/verify` — runs the plaintext embedding comparison.
- `POST /api/verify-fhe` — encrypts the generated embeddings, computes an encrypted dot product, decrypts the result inside the same API process, and reports numerical error and timing.

Example request body:

```json
{
  "enroll_subject": 1,
  "enroll_session": 3,
  "probe_subject": 1,
  "probe_session": 4,
  "frequency": 10.0
}
```

Use only with the included demo data or data you are authorized to process. This API is intended for local research demonstrations.

## Security model and limitations

| Component | Current prototype | Not yet provided |
|---|---|---|
| EEG input | Pre-recorded trial selected by ID | Live sensor capture / secure device pipeline |
| Encoder | Runs in plaintext on API server | Encrypted inference or a verified private-inference protocol |
| Similarity | CKKS encrypted dot product | Robust end-to-end encrypted threshold comparison |
| Decryption | Secret key used inside the same API process | Threshold CKKS / distributed key generation |
| Correctness | Plaintext-vs-FHE numerical comparison | ZK proof that the full model and protocol were executed correctly |
| Access control | Demo response gates a browser mockup | Real wallet integration, hardware-backed keys, audited authorization |

CKKS is approximate arithmetic. The error values reported here are empirical for these test vectors and parameters; they are not a universal bound. A deployable system would need a documented threat model, key lifecycle, secure threshold-decryption/comparison design, malicious-participant handling, rate limits, audit logs, independent security review, and a substantially stronger evaluation.

## Reproducibility

The repository includes scripts and result artifacts for inspection, but a fully independent reproduction requires the compatible dataset/cache, checkpoint, package versions, and exact split configuration. Preserve the identity split and threshold provenance when rerunning experiments. Do not tune a threshold on the final test set and then report the same set as an untouched evaluation.

## Roadmap

- [ ] Publish exact dependency lockfile and environment details
- [ ] Separate development, calibration, and final test identities
- [ ] Report confidence intervals and subject-level bootstrap results
- [ ] Evaluate a fixed threshold across all frequencies and session pairs
- [ ] Replace single-process decryption with a well-specified threshold CKKS protocol
- [ ] Investigate proof systems for model execution and encrypted arithmetic separately
- [ ] Replace the mock browser gate with a security-reviewed authorization flow

## Citation

If you build on this prototype, cite the repository URL and the commit you used. The current implementation is exploratory engineering work; it should not be described as a validated medical, identity, or wallet-security product.

## License

Check the upstream ArcFace project and every included dependency / dataset license before redistributing. No license grant is implied by this README. Add an explicit license only after confirming you have the right to license all code you own and have included.
