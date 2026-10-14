import json
from pathlib import Path
from collections import Counter

root = Path(__file__).parent
path = root / "eeg_experiment_manifest.json"
manifest = json.loads(path.read_text(encoding="utf-8"))

split = manifest["identity_split"]
seen = set(map(int, split["seen_subjects"]))
unseen = set(map(int, split["unseen_subjects"]))

if len(seen) != 60 or len(unseen) != 40 or seen & unseen:
    raise SystemExit("Invalid identity split; refusing to continue.")

counts = Counter()

for r in manifest["records"]:
    subject = int(r["subject"])
    session = int(r["session"])

    if subject in seen:
        if session in (0, 1):
            role = "train"
        elif session == 2:
            role = "seen_validation"
        elif session == 3:
            role = "seen_threshold_enrollment"
        elif session == 4:
            role = "seen_threshold_test"
        else:
            raise SystemExit(f"Unexpected session: {session}")
    elif subject in unseen:
        if session == 3:
            role = "unseen_enrollment"
        elif session == 4:
            role = "unseen_final_test"
        else:
            role = "unused_unseen_identity_record"

    r["role"] = role
    counts[role] += 1

manifest["roles"] = {
    "train": {
        "subjects": "60 seen identities",
        "sessions": [0, 1],
        "frequencies_hz": manifest["dataset"]["frequencies_hz"]
    },
    "validation": {
        "subjects": "same 60 seen identities",
        "session": 2,
        "purpose": "checkpoint selection only"
    },
    "threshold_calibration": {
        "subjects": "same 60 seen identities",
        "enrollment_session": 3,
        "test_session": 4,
        "purpose": "set global threshold using development identities only"
    },
    "unseen_final": {
        "subjects": "40 identities excluded from training",
        "enrollment_session": 3,
        "test_session": 4,
        "purpose": "locked final verification"
    }
}

manifest["evaluation"] = {
    "primary_frequencies_hz": manifest["dataset"]["frequencies_hz"],
    "genuine_impostor_frequency_matching": True,
    "threshold_calibration_subjects": "seen identities only",
    "final_test_subjects_used_for_training": False,
    "final_test_used_for_checkpoint_selection": False,
    "preprocessing_statistics_fit_on_training_data_only": True,
    "encoder_frozen_during_enrollment_and_verification": True,
    "sessions_5_and_6_required": False
}

manifest["role_counts"] = dict(counts)
manifest["protocol_version"] = 2

path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

print("CORRECTED MANIFEST")
print("Train sessions: 0, 1")
print("Validation session: 2")
print("Threshold calibration: seen identities, sessions 3 -> 4")
print("Final verification: unseen identities, sessions 3 -> 4")
print("\nRole counts:")
for role, count in sorted(counts.items()):
    print(f"  {role}: {count}")
print(f"\nSaved: {path}")
