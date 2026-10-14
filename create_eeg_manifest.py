import json
import random
from pathlib import Path
from collections import Counter

cache = Path(r"E:\ssvep-64ch-cache-retry")
out = Path(__file__).parent

with (cache / "metadata.json").open(encoding="utf-8") as f:
    records = json.load(f)

if isinstance(records, dict):
    for key in ("trials", "records", "items", "data"):
        if isinstance(records.get(key), list):
            records = records[key]
            break

if not isinstance(records, list) or len(records) != 4500:
    raise SystemExit(f"Unexpected metadata structure/count: {type(records)}, "
                     f"{len(records) if isinstance(records, list) else 'not a list'}")

required_channels = ["POZ", "PO3", "PO4", "PO7", "PO8", "OZ", "O1", "O2"]
subjects = sorted({int(r["subject"]) for r in records})
sessions = sorted({int(r["session"]) for r in records})
frequencies = sorted({float(r["frequency"]) for r in records})

if len(subjects) != 100 or sessions != [0, 1, 2, 3, 4]:
    raise SystemExit(f"Unexpected subjects/sessions: {len(subjects)}, {sessions}")

for i, r in enumerate(records):
    if not all(k in r for k in ("file", "subject", "session", "frequency",
                                "shape", "channel_names")):
        raise SystemExit(f"Incomplete record at index {i}")
    if not (cache / r["file"]).is_file():
        raise SystemExit(f"Missing file: {r['file']}")
    if list(r["shape"]) != [64, 1500]:
        raise SystemExit(f"Unexpected shape: {r['file']} {r['shape']}")
    if any(c not in r["channel_names"] for c in required_channels):
        raise SystemExit(f"Required channel missing: {r['file']}")

# Verify exactly one trial per subject/session/frequency.
triples = Counter(
    (int(r["subject"]), int(r["session"]), float(r["frequency"]))
    for r in records
)
if len(triples) != 4500 or any(n != 1 for n in triples.values()):
    raise SystemExit("Unexpected duplicate or missing subject/session/frequency trials")

# Fixed, reproducible identity-disjoint split.
seed = 20261014
shuffled = subjects.copy()
random.Random(seed).shuffle(shuffled)
seen = sorted(shuffled[:60])
unseen = sorted(shuffled[60:])

assert len(set(seen) & set(unseen)) == 0

manifest_records = []
counts = Counter()

for r in records:
    subject = int(r["subject"])
    session = int(r["session"])
    frequency = float(r["frequency"])

    if subject in seen:
        if session in (0, 1, 2):
            role = "train"
        elif session == 3:
            role = "seen_validation_and_calibration_enrollment"
        else:
            role = "seen_threshold_calibration_test"
    else:
        if session == 3:
            role = "unseen_enrollment"
        elif session == 4:
            role = "unseen_final_test"
        else:
            role = "unused_unseen_identity_record"

    counts[role] += 1
    manifest_records.append({
        "file": r["file"],
        "subject": subject,
        "session": session,
        "frequency": frequency,
        "shape": list(r["shape"]),
        "role": role
    })

manifest = {
    "protocol": "EEG identity-disjoint cross-session verification",
    "version": 1,
    "seed": seed,
    "dataset": {
        "cache": str(cache),
        "trials": len(records),
        "subjects": 100,
        "sessions": sessions,
        "frequencies_hz": frequencies,
        "recorded_shape": [64, 1500]
    },
    "channels_in_order": required_channels,
    "identity_split": {
        "seen_subjects": seen,
        "unseen_subjects": unseen
    },
    "roles": {
        "train": {
            "subjects": "seen",
            "sessions": [0, 1, 2],
            "frequencies_hz": frequencies
        },
        "validation": {
            "subjects": "seen",
            "session": 3,
            "purpose": "checkpoint and hyperparameter selection"
        },
        "threshold_calibration": {
            "subjects": "seen",
            "enrollment_session": 3,
            "test_session": 4,
            "purpose": "set threshold without using unseen identities"
        },
        "unseen_final": {
            "subjects": "unseen",
            "enrollment_session": 3,
            "test_session": 4,
            "purpose": "locked final verification"
        }
    },
    "frequency_protocol": {
        "primary": "all nine frequencies; genuine/impostor comparisons matched by frequency",
        "report_per_frequency": True,
        "report_aggregate": True
    },
    "leakage_rules": [
        "Unseen identities never used for training or model selection",
        "Unseen identities never used for threshold calibration",
        "Session 4 never used for training or checkpoint selection",
        "Fit preprocessing statistics on training data only",
        "Keep model weights frozen during enrollment and verification",
        "Match genuine and impostor comparisons by frequency"
    ],
    "role_counts": dict(counts),
    "records": manifest_records
}

manifest_path = out / "eeg_experiment_manifest.json"
split_path = out / "eeg_identity_split.json"

manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
split_path.write_text(json.dumps({
    "seed": seed,
    "seen_subjects": seen,
    "unseen_subjects": unseen
}, indent=2), encoding="utf-8")

print("FIXED SPLIT CREATED")
print(f"Trials: {len(records)}")
print(f"Seen identities: {len(seen)}")
print(f"Unseen identities: {len(unseen)}")
print(f"Frequencies: {frequencies}")
print("\nRole counts:")
for role, count in sorted(counts.items()):
    print(f"  {role}: {count}")
print(f"\nManifest: {manifest_path}")
print(f"Identity split: {split_path}")
