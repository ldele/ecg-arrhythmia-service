"""Count AAMI class distribution across all MIT-BIH records."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from src.data.labels import AAMI_CLASSES, map_to_aami
from src.data.load import load_record

# Standard MIT-BIH record IDs (48 records, excluding paced-only 102/104/107/217
# is common in literature) — but for inspection we look at all of them).
MITBIH_RECORDS = [
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119", "121",
    "122", "123", "124", "200", "201", "202", "203", "205", "207", "208",
    "209", "210", "212", "213", "214", "215", "217", "219", "220", "221",
    "222", "223", "228", "230", "231", "232", "233", "234",
]


def main() -> None:
    data_dir = Path("data/raw/mitdb")
    aami_counts: Counter[str] = Counter()
    unmapped_counts: Counter[str] = Counter()

    for record_id in MITBIH_RECORDS:
        record = load_record(record_id, data_dir)
        for symbol in record.beat_symbols:
            aami_class = map_to_aami(symbol)
            if aami_class is None:
                unmapped_counts[symbol] += 1
            else:
                aami_counts[aami_class] += 1

    total = sum(aami_counts.values())
    print(f"Total beats: {total}\n")
    print("AAMI class distribution:")
    for cls in AAMI_CLASSES:
        count = aami_counts[cls]
        pct = 100 * count / total if total else 0
        print(f"  {cls}: {count:>7} ({pct:5.2f}%)")

    print(f"\nUnmapped (non-beat) annotations: {sum(unmapped_counts.values())}")
    print("Top unmapped symbols:")
    for symbol, count in unmapped_counts.most_common(10):
        print(f"  {symbol!r}: {count}")


if __name__ == "__main__":
    main()