import argparse
import json
import os
import re
from typing import Dict, List

from utils import DATASET_PATH


TYPE_PATTERN = re.compile(r"questions_(.+)_filtered_dedup_normalized\.json$")


def load_questions(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}")
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Merge normalized, filtered questions into a single dataset."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=DATASET_PATH,
        help="Directory containing questions_*_filtered_dedup_normalized.json files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="questions.json",
        help="Output filename (relative to data-path unless absolute).",
    )
    args = parser.parse_args()

    data_path = args.data_path
    if not data_path:
        raise SystemExit("DATASET_PATH is empty. Provide --data-path explicitly.")
    if not os.path.isdir(data_path):
        raise SystemExit(f"Data path not found: {data_path}")

    files = [
        os.path.join(data_path, name)
        for name in os.listdir(data_path)
        if TYPE_PATTERN.match(name)
    ]
    files.sort()
    if not files:
        raise SystemExit(
            f"No files matching questions_*_filtered_dedup_normalized.json in {data_path}"
        )

    merged: List[Dict[str, str]] = []
    next_id = 0
    for path in files:
        match = TYPE_PATTERN.search(os.path.basename(path))
        if not match:
            continue
        q_type = match.group(1)
        items = load_questions(path)
        for item in items:
            question = item.get("question", "")
            if not question:
                continue
            merged.append(
                {
                    "id": next_id,
                    "question": question,
                    "answer": item.get("answer", ""),
                    "type": q_type,
                }
            )
            next_id += 1

    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = os.path.join(data_path, output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=4, ensure_ascii=False)

    print(f"Merged {len(merged)} questions → {output_path}")


if __name__ == "__main__":
    main()
