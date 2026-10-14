# EEG ArcFace CKKS FHE demo patch

## What changed
- `api.py` adds `POST /api/verify-fhe`, which loads the same EEG encoder, creates normalized embeddings, CKKS-encrypts both 32-D embeddings, computes an encrypted dot product, decrypts the score inside the local API process, and returns the plaintext-vs-FHE numerical error.
- `static/index.html` adds a computation-mode selector and displays the CKKS score/error.

## Run
Use the same PowerShell command and paths you already use for `api.py`, with TenSEAL installed in that Python environment. Open `http://127.0.0.1:8000` and select **CKKS FHE demo**.

## Security limitations
This is a local arithmetic demonstration, not the final architecture. The encoder still runs in plaintext. The API creates and retains a single secret key and decrypts the score itself, so it does not implement threshold decryption, validator key shares, ZKML, or on-chain proof verification. No secret key or private context is returned to the browser. Do not expose this API to the internet.
