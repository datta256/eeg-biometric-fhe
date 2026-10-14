"""Multi-frequency, multi-session held-out EEG ArcFace + CKKS evaluation.

Uses the checkpoint's recorded seen_subjects to exclude training identities from
the test set, unless an explicit split JSON is supplied. Computes plaintext
scores for every available pair per condition and CKKS scores for every genuine
pair plus a deterministic sample of impostor pairs.

Research evaluation only; not a production authentication/security assessment.
"""
import argparse
import csv
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch
import tenseal as ts

from eeg_model import TemporalSpatialEEGEncoder

CHANNELS = ["POZ", "PO3", "PO4", "PO7", "PO8", "OZ", "O1", "O2"]
DEFAULT_FREQUENCIES = [8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0]


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
        raise RuntimeError(f"Missing expected EEG channels: {missing}")
    return [names.index(c) for c in CHANNELS]


def get_record_index(manifest):
    result = {}
    for r in manifest["records"]:
        key = (int(r["subject"]), int(r["session"]), round(float(r["frequency"]), 4))
        if key in result:
            raise RuntimeError(f"Duplicate manifest record for {key}")
        result[key] = r
    return result


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
    return private_ctx, ts.context_from(public_bytes)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--subject-split", default=None,
                   help="Optional split JSON; by default test subjects are manifest subjects absent from checkpoint seen_subjects.")
    p.add_argument("--split-key", default="test_subjects")
    p.add_argument("--frequencies", nargs="+", type=float, default=DEFAULT_FREQUENCIES)
    p.add_argument("--sessions", nargs="+", type=int, default=[0, 1, 2, 3, 4],
                   help="Session IDs to combine in all unique enrollment/probe pairs.")
    p.add_argument("--threshold", type=float, default=0.617122)
    p.add_argument("--fhe-impostor-sample", type=int, default=200,
                   help="Impostor pairs per condition for FHE; 0 means all impostor pairs.")
    p.add_argument("--seed", type=int, default=20261014)
    p.add_argument("--summary-csv", default="eeg_fhe_multicondition_summary.csv")
    p.add_argument("--pairs-csv", default="eeg_fhe_multicondition_pairs.csv")
    args = p.parse_args()

    start = time.time()
    cache = Path(args.cache)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    record_index = get_record_index(manifest)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    seen = set(map(int, checkpoint.get("seen_subjects", [])))
    all_subjects = {int(r["subject"]) for r in manifest["records"]}

    if args.subject_split:
        split_data = json.loads(Path(args.subject_split).read_text(encoding="utf-8"))
        if args.split_key not in split_data or not isinstance(split_data[args.split_key], list):
            raise RuntimeError(f"Split JSON has no list key {args.split_key!r}")
        requested = set(map(int, split_data[args.split_key]))
        test_subjects = sorted(all_subjects & requested)
        overlap = sorted(set(test_subjects) & seen)
        if overlap:
            raise RuntimeError(
                f"Provided test split overlaps checkpoint seen_subjects: {overlap}. "
                "Use a split whose test subjects are absent from the checkpoint."
            )
    else:
        if not seen:
            raise RuntimeError("Checkpoint has no seen_subjects metadata; provide a verified --subject-split.")
        test_subjects = sorted(all_subjects - seen)

    if len(test_subjects) < 2:
        raise RuntimeError(f"Need >=2 held-out subjects, found {len(test_subjects)}")
    print(f"Checkpoint training identities recorded: {len(seen)}")
    print(f"Held-out identities selected: {len(test_subjects)}")
    print(f"Any overlap: {bool(set(test_subjects) & seen)}")
    if set(test_subjects) & seen:
        raise RuntimeError("Training/test identity overlap detected.")

    channel_indices = load_channel_indices(cache)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = TemporalSpatialEEGEncoder(channels=8, embedding_dim=32).to(device)
    encoder.load_state_dict(checkpoint["encoder"], strict=True)
    encoder.eval()

    # Cache each embedding once per subject/session/frequency.
    embedding_cache = {}

    def embedding(subject, session, frequency):
        key = (int(subject), int(session), round(float(frequency), 4))
        if key in embedding_cache:
            return embedding_cache[key]
        record = record_index.get(key)
        if record is None:
            return None
        path = cache / record["file"]
        if not path.is_file():
            raise FileNotFoundError(path)
        x = np.load(path, allow_pickle=False).astype(np.float32, copy=False)
        if x.shape != (64, 1500):
            raise RuntimeError(f"Unexpected shape {x.shape} in {path}; expected (64,1500)")
        x = x[channel_indices, :]
        x = (x - x.mean(axis=-1, keepdims=True)) / np.maximum(x.std(axis=-1, keepdims=True), 1e-6)
        tensor = torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0).to(device)
        with torch.no_grad():
            z = encoder(tensor).squeeze(0).detach().cpu().numpy().astype(np.float64)
        norm = float(np.linalg.norm(z))
        if not np.isfinite(norm) or norm < 1e-12:
            raise RuntimeError(f"Invalid embedding norm for {key}: {norm}")
        result = z / norm
        embedding_cache[key] = result
        return result

    session_pairs = list(itertools.combinations(sorted(set(args.sessions)), 2))
    conditions = [(float(f), int(s1), int(s2))
                  for f in args.frequencies for s1, s2 in session_pairs]
    print(f"Conditions requested: {len(conditions)} "
          f"({len(args.frequencies)} frequencies x {len(session_pairs)} session pairs)")
    print("Embedding trials (this may take a few minutes)...")

    private_ctx, public_ctx = make_fhe_context()
    summary_rows, all_pair_rows = [], []
    rng = np.random.default_rng(args.seed)
    total_fhe_pairs = 0
    total_conditions = len(conditions)

    for cond_i, (freq, enroll_session, probe_session) in enumerate(conditions, 1):
        subjects = [s for s in test_subjects
                    if (s, enroll_session, round(freq, 4)) in record_index
                    and (s, probe_session, round(freq, 4)) in record_index]
        if len(subjects) < 2:
            print(f"[{cond_i}/{total_conditions}] SKIP {freq} Hz {enroll_session}->{probe_session}: only {len(subjects)} subjects")
            continue

        enroll_mat = np.stack([embedding(s, enroll_session, freq) for s in subjects])
        probe_mat = np.stack([embedding(s, probe_session, freq) for s in subjects])
        score_matrix = probe_mat @ enroll_mat.T
        rows = []
        genuine_idx, impostor_idx = [], []
        for pi, ps in enumerate(subjects):
            for ei, es in enumerate(subjects):
                is_genuine = ps == es
                rows.append({
                    "frequency_hz": freq,
                    "enroll_session": enroll_session,
                    "probe_session": probe_session,
                    "enroll_subject": es,
                    "probe_subject": ps,
                    "pair_type": "genuine" if is_genuine else "impostor",
                    "plaintext_score": float(score_matrix[pi, ei]),
                    "fhe_score": "",
                    "absolute_error": "",
                    "plaintext_decision": "",
                    "fhe_decision": "",
                    "decision_flip": "",
                })
                (genuine_idx if is_genuine else impostor_idx).append(len(rows) - 1)

        if args.fhe_impostor_sample == 0 or args.fhe_impostor_sample >= len(impostor_idx):
            chosen_imp = impostor_idx
        else:
            chosen_imp = sorted(rng.choice(
                impostor_idx, size=args.fhe_impostor_sample, replace=False
            ).tolist())
        fhe_indices = sorted(genuine_idx + chosen_imp)

        # Encrypt each enrollment and probe vector once for this condition.
        enc_enroll = [ts.ckks_vector(public_ctx, v.tolist()) for v in enroll_mat]
        enc_probe = [ts.ckks_vector(public_ctx, v.tolist()) for v in probe_mat]
        for flat_idx in fhe_indices:
            row = rows[flat_idx]
            pi = subjects.index(row["probe_subject"])
            ei = subjects.index(row["enroll_subject"])
            enc_score = enc_probe[pi].dot(enc_enroll[ei])
            dec_score = float(ts.ckks_vector_from(private_ctx, enc_score.serialize()).decrypt()[0])
            pscore = float(row["plaintext_score"])
            row["fhe_score"] = dec_score
            row["absolute_error"] = abs(pscore - dec_score)
            row["plaintext_decision"] = "ACCEPT" if pscore >= args.threshold else "REJECT"
            row["fhe_decision"] = "ACCEPT" if dec_score >= args.threshold else "REJECT"
            row["decision_flip"] = bool((pscore >= args.threshold) != (dec_score >= args.threshold))

        scores = np.array([r["plaintext_score"] for r in rows], dtype=np.float64)
        labels = np.array([r["pair_type"] == "genuine" for r in rows], dtype=bool)
        pred = scores >= args.threshold
        far = float(np.mean(pred[~labels]))
        frr = float(np.mean(~pred[labels]))
        try:
            from sklearn.metrics import roc_auc_score, roc_curve
            auc = float(roc_auc_score(labels.astype(np.uint8), scores))
            fpr, tpr, roc_thresholds = roc_curve(labels.astype(np.uint8), scores)
            fnr = 1.0 - tpr
            ix = int(np.argmin(np.abs(fpr - fnr)))
            eer = float((fpr[ix] + fnr[ix]) / 2)
            eer_thr = float(roc_thresholds[ix])
        except Exception:
            auc, eer, eer_thr = float("nan"), float("nan"), float("nan")

        fhe_rows = [rows[i] for i in fhe_indices]
        errors = np.array([float(r["absolute_error"]) for r in fhe_rows])
        flips = sum(bool(r["decision_flip"]) for r in fhe_rows)
        fg = [r for r in fhe_rows if r["pair_type"] == "genuine"]
        fi = [r for r in fhe_rows if r["pair_type"] == "impostor"]
        fhe_far = float(np.mean([float(r["fhe_score"]) >= args.threshold for r in fi])) if fi else float("nan")
        fhe_frr = float(np.mean([float(r["fhe_score"]) < args.threshold for r in fg])) if fg else float("nan")

        summary_rows.append({
            "frequency_hz": freq,
            "enroll_session": enroll_session,
            "probe_session": probe_session,
            "subjects": len(subjects),
            "genuine_pairs": len(genuine_idx),
            "impostor_pairs": len(impostor_idx),
            "roc_auc": auc,
            "eer": eer,
            "eer_threshold": eer_thr,
            "threshold": args.threshold,
            "plaintext_far": far,
            "plaintext_frr": frr,
            "fhe_pairs": len(fhe_rows),
            "fhe_impostor_pairs": len(fi),
            "fhe_mean_abs_error": float(np.mean(errors)),
            "fhe_max_abs_error": float(np.max(errors)),
            "fhe_decision_flips": flips,
            "fhe_far_sample": fhe_far,
            "fhe_frr": fhe_frr,
        })
        all_pair_rows.extend(rows)
        total_fhe_pairs += len(fhe_rows)
        print(
            f"[{cond_i}/{total_conditions}] {freq:>4g} Hz {enroll_session}->{probe_session} | "
            f"n={len(subjects)} AUC={auc:.4f} EER={eer:.2%} "
            f"FAR={far:.2%} FRR={frr:.2%} | "
            f"FHE n={len(fhe_rows)} err_max={np.max(errors):.2e} flips={flips}"
        )

    summary_path = Path(args.summary_csv)
    pairs_path = Path(args.pairs_csv)
    if summary_rows:
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)
        if all_pair_rows:
            with pairs_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(all_pair_rows[0].keys()))
                w.writeheader()
                w.writerows(all_pair_rows)

    print("\n========== MULTI-CONDITION SUMMARY ==========")
    print(f"Conditions completed:              {len(summary_rows)}/{len(conditions)}")
    print(f"FHE pair evaluations:              {total_fhe_pairs}")
    if summary_rows:
        print(f"Mean ROC-AUC across conditions:    {np.mean([r['roc_auc'] for r in summary_rows]):.6f}")
        print(f"Mean descriptive EER:              {np.mean([r['eer'] for r in summary_rows]):.4%}")
        print(f"Conditions with zero FHE flips:    {sum(r['fhe_decision_flips'] == 0 for r in summary_rows)}/{len(summary_rows)}")
        print(f"Maximum FHE error overall:         {max(r['fhe_max_abs_error'] for r in summary_rows):.3e}")
        print(f"Mean FHE error across conditions:  {np.mean([r['fhe_mean_abs_error'] for r in summary_rows]):.3e}")
    print(f"Summary CSV: {summary_path.resolve()}")
    print(f"Pair CSV:    {pairs_path.resolve()}")
    print(f"Elapsed:     {time.time() - start:.1f}s")
    print("\nCaveats:")
    print("- Test identities are derived from manifest subjects absent from checkpoint seen_subjects.")
    print("- Threshold is fixed at the supplied value; it is not independently calibrated here.")
    print("- Scores are dependent because each subject/trial appears in many pair comparisons.")
    print("- Frequencies/sessions are repeated measurements, not independent datasets.")
    print("- FHE protects encrypted similarity computation, not plaintext EEG preprocessing or encoder inference.")


if __name__ == "__main__":
    main()
