import json
import sys
from pathlib import Path
from collections import Counter

cache = Path(sys.argv[1])
metadata_path = cache / "metadata.json"

if not metadata_path.exists():
    raise SystemExit(f"ERROR: Metadata not found: {metadata_path}")

with metadata_path.open("r", encoding="utf-8") as f:
    metadata = json.load(f)

if isinstance(metadata, dict):
    for key in ("trials", "records", "items", "data"):
        if isinstance(metadata.get(key), list):
            metadata = metadata[key]
            break

if not isinstance(metadata, list) or not metadata:
    raise SystemExit("ERROR: Expected a non-empty list of trial records.")

print("=" * 65)
print("EEG DATASET AUDIT")
print("=" * 65)
print(f"Cache: {cache}")
print(f"Metadata records: {len(metadata)}")

required = ("file", "subject", "session", "frequency")
missing_fields = Counter()
subjects, sessions, frequencies = set(), Counter(), Counter()
channel_orders = Counter()
shapes = Counter()
seen_paths = Counter()
missing_files = []
invalid_records = []
sample_checks = []

for i, row in enumerate(metadata):
    if not isinstance(row, dict):
        invalid_records.append((i, "record is not an object"))
        continue

    for field in required:
        if field not in row:
            missing_fields[field] += 1

    if not all(field in row for field in required):
        invalid_records.append((i, f"missing required field: {row}"))
        continue

    subjects.add(str(row["subject"]))
    sessions[str(row["session"])] += 1
    frequencies[str(row["frequency"])] += 1

    if row.get("shape") is not None:
        shapes[tuple(row["shape"])] += 1

    if row.get("channel_names"):
        channel_orders[tuple(row["channel_names"])] += 1

    relative_path = Path(str(row["file"]))
    full_path = cache / relative_path
    seen_paths[str(relative_path).replace("\\", "/")] += 1

    if not full_path.is_file():
        missing_files.append(str(relative_path))

print(f"Unique subjects: {len(subjects)}")
print(f"Subject IDs: {', '.join(sorted(subjects, key=lambda x: int(x)))}")

print("\nTRIAL COUNTS BY SESSION")
for session, count in sorted(sessions.items(), key=lambda x: float(x[0])):
    print(f"  Session {session}: {count}")

print("\nTRIAL COUNTS BY FREQUENCY")
for frequency, count in sorted(frequencies.items(), key=lambda x: float(x[0])):
    print(f"  {frequency} Hz: {count}")

print("\nRECORDED SAMPLE SHAPES")
for shape, count in shapes.most_common():
    print(f"  {shape}: {count} trials")

print("\nCHANNEL METADATA")
print(f"Distinct channel orders: {len(channel_orders)}")
for channels, count in channel_orders.most_common(3):
    print(f"  {count} records; {len(channels)} channels")
    print("  " + ", ".join(channels))

print("\nFILE INTEGRITY")
print(f"Unique trial paths: {len(seen_paths)}")
print(f"Duplicate path entries: {sum(n - 1 for n in seen_paths.values() if n > 1)}")
print(f"Missing files: {len(missing_files)}")
for path in missing_files[:20]:
    print(f"  MISSING: {path}")

print(f"Invalid metadata records: {len(invalid_records)}")
for index, reason in invalid_records[:10]:
    print(f"  Record {index}: {reason}")

# Load only a few samples to independently verify actual array shapes.
try:
    import numpy as np

    checked = 0
    for row in metadata:
        if not isinstance(row, dict) or not row.get("file"):
            continue
        sample_path = cache / row["file"]
        if not sample_path.is_file():
            continue
        try:
            arr = np.load(sample_path, mmap_mode="r", allow_pickle=False)
            sample_checks.append((row["file"], tuple(arr.shape), row.get("shape")))
            checked += 1
            if checked >= 5:
                break
        except Exception as exc:
            sample_checks.append((row["file"], "LOAD ERROR", str(exc)))

    print("\nACTUAL ARRAY SHAPE CHECK")
    for item in sample_checks:
        print(f"  {item}")
except ImportError:
    print("\nNumPy unavailable; skipped loading sample arrays.")

report = {
    "cache": str(cache),
    "metadata_records": len(metadata),
    "unique_subjects": len(subjects),
    "sessions": dict(sessions),
    "frequencies": dict(frequencies),
    "shapes": {str(k): v for k, v in shapes.items()},
    "missing_file_count": len(missing_files),
    "duplicate_path_count": sum(n - 1 for n in seen_paths.values() if n > 1),
    "invalid_record_count": len(invalid_records),
    "sample_checks": [
        {"file": a, "actual_shape": str(b), "metadata_shape": str(c)}
        for a, b, c in sample_checks
    ],
}

output_path = Path(__file__).with_name("dataset_audit.json")
output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(f"\nAudit report saved: {output_path}")
