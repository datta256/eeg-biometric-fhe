"""Batch evaluation of real EEG ArcFace embeddings and CKKS encrypted similarity.

This evaluates all available subject pairs in plaintext, then validates CKKS on
all genuine pairs plus a deterministic sample of impostor pairs by default.
It is a research validation script, not a production authentication system.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import tenseal as ts

from eeg_model import TemporalSpatialEEGEncoder

CHANNELS = ["POZ", "PO3", "PO4", "PO7", "PO8", "OZ", "O1", "O2"]


def load_channel_indices(cache: Path):
    metadata = json.loads((cache / "metadata.json").read_text(encoding="utf-8"))
    if isinstance(metadata, dict):
        for key in ("trials", "records", "items", "data"):
            if isinstance(metadata.get(key), list):
                metadata = metadata[key]
                break
    if not metadata:
        raise RuntimeError("metadata.json contains no channel metadata")
    names = metadata[0]["channel_names"]
    missing = [c for c in CHANNELS if c not in names]
    if missing:
        raise RuntimeError(f"Missing expected EEG channels in metadata: {missing}")
    return [names.index(c) for c in CHANNELS]


def get_record_index(manifest):
    result = {}
    for r in manifest["records"]:
        key = (int(r["subject"]), int(r["session"]), round(float(r["frequency"]), 4))
        if key in result:
            raise RuntimeError(f"Duplicate manifest record for {key}; expected one cached trial per key.")
        result[key] = r
    return result


def load_embedding(subject, session, frequency, record_index, cache,
                   channel_indices, encoder, device):
    key = (int(subject), int(session), round(float(frequency), 4))
    record = record_index.get(key)
    if record is None:
        raise RuntimeError(f"No manifest record for subject/session/frequency={key}")
    path = cache / record["file"]
    if not path.is_file():
        raise FileNotFoundError(path)
    x = np.load(path, allow_pickle=False).astype(np.float32, copy=False)
    if x.shape != (64, 1500):
        raise RuntimeError(f"Unexpected trial shape {x.shape} in {path}; expected (64, 1500)")
    x = x[channel_indices, :]
    # Mirror api.py and validate_real_eeg_fhe.py preprocessing exactly.
    x = (x - x.mean(axis=-1, keepdims=True)) / np.maximum(x.std(axis=-1, keepdims=True), 1e-6)
    tensor = torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0).to(device)
    with torch.no_grad():
        z = encoder(tensor).squeeze(0).detach().cpu().numpy().astype(np.float64)
    norm = float(np.linalg.norm(z))
    if not np.isfinite(norm) or norm < 1e-12:
        raise RuntimeError(f"Invalid embedding norm {norm} for {key}")
    return z / norm


def make_fhe_context():
    private_ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60],
    )
    private_ctx.global_scale = 2**40
    private_ctx.generate_relin_keys()
    private_ctx.generate_galois_keys()
    private_bytes = private_ctx.serialize(
        save_public_key=True, save_secret_key=True,
        save_galois_keys=True, save_relin_keys=True,
    )
    public_source = ts.context_from(private_bytes)
    public_source.make_context_public()
    public_bytes = public_source.serialize(
        save_public_key=True, save_secret_key=False,
        save_galois_keys=True, save_relin_keys=True,
    )
    return ts.context_from(private_bytes), ts.context_from(public_bytes)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--enroll-session", type=int, default=3)
    p.add_argument("--probe-session", type=int, default=4)
    p.add_argument("--frequency", type=float, default=10.0)
    p.add_argument("--threshold", type=float, default=0.617122)
    p.add_argument("--fhe-impostor-sample", type=int, default=1000,
                   help="Number of impostor pairs to test with FHE; use 0 for all impostor pairs.")
    p.add_argument("--seed", type=int, default=20261014)
    p.add_argument("--subject-split", default=None,
                   help="Optional identity split JSON; when supplied, evaluates only subjects listed under --split-key.")
    p.add_argument("--split-key", default="test_subjects",
                   help="List key in --subject-split JSON (default: test_subjects).")
    p.add_argument("--csv", default="eeg_fhe_heldout_results.csv")
    args = p.parse_args()

    start = time.time()
    cache = Path(args.cache)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    record_index = get_record_index(manifest)
    freq_key = round(float(args.frequency), 4)
    available = sorted({
        int(r["subject"]) for r in manifest["records"]
        if int(r["session"]) in (args.enroll_session, args.probe_session)
        and round(float(r["frequency"]), 4) == freq_key
        and (int(r["subject"]), args.enroll_session, freq_key) in record_index
        and (int(r["subject"]), args.probe_session, freq_key) in record_index
    })
    if args.subject_split:
        split_data = json.loads(Path(args.subject_split).read_text(encoding="utf-8"))
        if args.split_key not in split_data or not isinstance(split_data[args.split_key], list):
            raise RuntimeError(
                f"Split file must contain a list at key {args.split_key!r}; "
                f"available keys: {list(split_data.keys())}"
            )
        requested_subjects = {int(s) for s in split_data[args.split_key]}
        available = [s for s in available if s in requested_subjects]
        missing = sorted(requested_subjects - set(available))
        print(f"Applied subject split: {args.subject_split}")
        print(f"Split key: {args.split_key}; requested subjects: {len(requested_subjects)}")
        if missing:
            print(f"WARNING: {len(missing)} split subjects lack one of the requested session/frequency records: {missing[:20]}")
    if len(available) < 2:
        raise RuntimeError(f"Need at least two subjects with both sessions at {args.frequency} Hz; found {len(available)}.")

    channel_indices = load_channel_indices(cache)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    encoder = TemporalSpatialEEGEncoder(channels=8, embedding_dim=32).to(device)
    encoder.load_state_dict(checkpoint["encoder"], strict=True)
    encoder.eval()

    print("Loading normalized EEG embeddings...")
    embeddings_enroll, embeddings_probe = [], []
    for i, subject in enumerate(available, 1):
        embeddings_enroll.append(load_embedding(
            subject, args.enroll_session, args.frequency, record_index,
            cache, channel_indices, encoder, device))
        embeddings_probe.append(load_embedding(
            subject, args.probe_session, args.frequency, record_index,
            cache, channel_indices, encoder, device))
        if i % 10 == 0 or i == len(available):
            print(f"  embedded {i}/{len(available)} subjects")
    enroll_mat = np.stack(embeddings_enroll)
    probe_mat = np.stack(embeddings_probe)

    # All pair scores: each enrolled subject vs every probe subject.
    scores = probe_mat @ enroll_mat.T
    pair_rows = []
    genuine_indices = []
    impostor_indices = []
    for probe_i, probe_subject in enumerate(available):
        for enroll_i, enroll_subject in enumerate(available):
            is_genuine = probe_subject == enroll_subject
            pair_rows.append({
                "enroll_subject": enroll_subject,
                "probe_subject": probe_subject,
                "enroll_session": args.enroll_session,
                "probe_session": args.probe_session,
                "frequency": args.frequency,
                "pair_type": "genuine" if is_genuine else "impostor",
                "plaintext_score": float(scores[probe_i, enroll_i]),
                "fhe_score": "",
                "absolute_error": "",
                "plaintext_decision": "",
                "fhe_decision": "",
                "decision_flip": "",
            })
            idx = len(pair_rows) - 1
            (genuine_indices if is_genuine else impostor_indices).append(idx)

    # FHE check includes every genuine pair and a seeded impostor sample.
    if args.fhe_impostor_sample == 0 or args.fhe_impostor_sample >= len(impostor_indices):
        chosen_impostors = impostor_indices
    else:
        rng = np.random.default_rng(args.seed)
        chosen_impostors = sorted(rng.choice(
            impostor_indices, size=args.fhe_impostor_sample, replace=False
        ).tolist())
    fhe_indices = sorted(genuine_indices + chosen_impostors)
    print(f"\nSubjects: {len(available)}")
    print(f"Plaintext pair scores: {len(pair_rows)} ({len(genuine_indices)} genuine, {len(impostor_indices)} impostor)")
    print(f"FHE pairs to evaluate: {len(fhe_indices)} ({len(genuine_indices)} genuine, {len(chosen_impostors)} impostor)")
    print("Creating CKKS keys and encrypting each enrollment/probe embedding once...")
    private_ctx, public_ctx = make_fhe_context()
    enc_enroll = [ts.ckks_vector(public_ctx, v.tolist()) for v in enroll_mat]
    enc_probe = [ts.ckks_vector(public_ctx, v.tolist()) for v in probe_mat]

    # Recover pair matrix indices from each flat row.
    n = len(available)
    row_by_pair = {}
    for idx, row in enumerate(pair_rows):
        row_by_pair[(row["probe_subject"], row["enroll_subject"])] = idx
    t_fhe = time.time()
    for count, flat_idx in enumerate(fhe_indices, 1):
        row = pair_rows[flat_idx]
        probe_i = available.index(row["probe_subject"])
        enroll_i = available.index(row["enroll_subject"])
        encrypted_score = enc_probe[probe_i].dot(enc_enroll[enroll_i])
        decrypted_score = float(ts.ckks_vector_from(
            private_ctx, encrypted_score.serialize()
        ).decrypt()[0])
        plaintext_score = row["plaintext_score"]
        error = abs(plaintext_score - decrypted_score)
        p_decision = plaintext_score >= args.threshold
        f_decision = decrypted_score >= args.threshold
        row["fhe_score"] = decrypted_score
        row["absolute_error"] = error
        row["plaintext_decision"] = "ACCEPT" if p_decision else "REJECT"
        row["fhe_decision"] = "ACCEPT" if f_decision else "REJECT"
        row["decision_flip"] = bool(p_decision != f_decision)
        if count % 100 == 0 or count == len(fhe_indices):
            print(f"  FHE pairs {count}/{len(fhe_indices)}")
    elapsed_fhe = time.time() - t_fhe

    all_plain_scores = np.array([r["plaintext_score"] for r in pair_rows], dtype=np.float64)
    labels = np.array([r["pair_type"] == "genuine" for r in pair_rows], dtype=bool)
    plain_decisions = all_plain_scores >= args.threshold
    genuine_mask = labels
    impostor_mask = ~labels
    far = float(np.mean(plain_decisions[impostor_mask]))
    frr = float(np.mean(~plain_decisions[genuine_mask]))
    accuracy = float(np.mean(plain_decisions == labels))

    try:
        from sklearn.metrics import roc_auc_score, roc_curve
        auc = float(roc_auc_score(labels.astype(np.uint8), all_plain_scores))
        fpr, tpr, roc_thresholds = roc_curve(labels.astype(np.uint8), all_plain_scores)
        fnr = 1.0 - tpr
        eer_idx = int(np.argmin(np.abs(fpr - fnr)))
        eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
        eer_threshold = float(roc_thresholds[eer_idx])
    except Exception as exc:
        auc, eer, eer_threshold = float("nan"), float("nan"), float("nan")
        print(f"ROC-AUC/EER unavailable (scikit-learn issue): {exc}")

    fhe_rows = [pair_rows[i] for i in fhe_indices]
    errors = np.array([float(r["absolute_error"]) for r in fhe_rows], dtype=np.float64)
    flips = sum(bool(r["decision_flip"]) for r in fhe_rows)
    fhe_genuine = [r for r in fhe_rows if r["pair_type"] == "genuine"]
    fhe_impostor = [r for r in fhe_rows if r["pair_type"] == "impostor"]
    fhe_far = (sum(float(r["fhe_score"]) >= args.threshold for r in fhe_impostor) / len(fhe_impostor)
               if fhe_impostor else float("nan"))
    fhe_frr = (sum(float(r["fhe_score"]) < args.threshold for r in fhe_genuine) / len(fhe_genuine)
               if fhe_genuine else float("nan"))

    out_path = Path(args.csv)
    import csv
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(pair_rows[0].keys()))
        writer.writeheader()
        writer.writerows(pair_rows)

    print("\n========== BATCH EEG / FHE RESULTS ==========")
    print(f"Device:                          {device}")
    print(f"Sessions:                        {args.enroll_session} -> {args.probe_session}")
    print(f"Frequency:                       {args.frequency} Hz")
    print(f"Subjects:                        {len(available)}")
    print(f"All pair comparisons:             {len(pair_rows)}")
    print(f"Genuine pairs:                    {len(genuine_indices)}")
    print(f"Impostor pairs:                   {len(impostor_indices)}")
    print(f"Plaintext ROC-AUC:                {auc:.6f}")
    print(f"Plaintext EER (descriptive):      {eer:.4%}")
    print(f"EER threshold (descriptive):      {eer_threshold:.6f}")
    print(f"Plaintext FAR at {args.threshold:.6f}:       {far:.4%}")
    print(f"Plaintext FRR at {args.threshold:.6f}:       {frr:.4%}")
    print(f"Pairwise classification accuracy: {accuracy:.4%} (not a balanced metric)")
    print(f"FHE pairs checked:                {len(fhe_rows)}")
    print(f"FHE mean absolute error:          {float(np.mean(errors)):.3e}")
    print(f"FHE max absolute error:           {float(np.max(errors)):.3e}")
    print(f"FHE decision flips:               {flips}/{len(fhe_rows)}")
    print(f"FHE-sample FAR:                   {fhe_far:.4%}")
    print(f"FHE-sample FRR:                   {fhe_frr:.4%}")
    print(f"FHE evaluation seconds:           {elapsed_fhe:.1f}")
    print(f"Total elapsed seconds:            {time.time() - start:.1f}")
    print(f"CSV:                              {out_path.resolve()}")
    print("\nInterpretation notes:")
    print("- Plaintext FAR/FRR are for this selected session/frequency pair protocol only.")
    print("- The threshold is carried over as a provisional threshold, not recalibrated here.")
    print("- All impostor pairs are evaluated in plaintext; FHE runs on all genuine pairs plus the configured impostor sample.")
    print("- Pairwise scores are dependent because subjects/trials recur; do not treat pair count as independent sample size.")
    print("- This checks CKKS arithmetic and score decisions, not protection of the plaintext encoder or production security.")


if __name__ == "__main__":
    main()
