import argparse
import json
import os
import re
import sys
from typing import Dict, List

sys.path.append(os.path.abspath(".."))

from utils import DATASET_PATH


TYPE_PATTERN = re.compile(r"questions_(.+)_filtered_dedup_normalized\.json$")


def load_questions(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}")
    return data


def load_groundtruth(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}")
    return data


def validate_groundtruth_alignment(
    question_path: str,
    questions: List[Dict[str, str]],
    groundtruth_path: str,
    groundtruth: List[Dict[str, str]],
) -> None:
    question_ids = {item.get("id") for item in questions}
    groundtruth_ids = {item.get("id") for item in groundtruth}

    missing_groundtruth = sorted(question_ids - groundtruth_ids)
    extra_groundtruth = sorted(groundtruth_ids - question_ids)

    if missing_groundtruth or extra_groundtruth:
        details = [
            f"Question file: {question_path}",
            f"Groundtruth file: {groundtruth_path}",
            f"question_count={len(questions)}",
            f"groundtruth_count={len(groundtruth)}",
        ]
        if missing_groundtruth:
            details.append(f"missing_groundtruth_ids={missing_groundtruth[:20]}")
        if extra_groundtruth:
            details.append(f"extra_groundtruth_ids={extra_groundtruth[:20]}")
        raise ValueError("Groundtruth mismatch detected.\n" + "\n".join(details))


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
    parser.add_argument(
        "--groundtruth-output",
        type=str,
        default="groundtruth.json",
        help="Groundtruth output filename (relative to data-path unless absolute).",
    )
    parser.add_argument(
        "--allow-missing-groundtruth",
        action="store_true",
        help="Allow merging questions even if some per-type groundtruth files are missing or incomplete.",
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
    merged_groundtruth: List[Dict[str, str]] = []
    next_id = 0
    for path in files:
        match = TYPE_PATTERN.search(os.path.basename(path))
        if not match:
            continue
        q_type = match.group(1)
        items = load_questions(path)
        groundtruth_path = os.path.join(
            data_path,
            os.path.basename(path).replace("questions_", "groundtruth_", 1),
        )
        groundtruth_items_by_id = {}
        if os.path.exists(groundtruth_path):
            gt_items = load_groundtruth(groundtruth_path)
            if not args.allow_missing_groundtruth:
                validate_groundtruth_alignment(path, items, groundtruth_path, gt_items)
            groundtruth_items_by_id = {item.get("id"): item for item in gt_items}
        elif not args.allow_missing_groundtruth:
            raise FileNotFoundError(
                f"Missing groundtruth file for {path}: expected {groundtruth_path}"
            )

        for item in items:
            question = item.get("question", "")
            if not question:
                continue
            merged_id = next_id
            merged.append(
                {
                    "id": merged_id,
                    "question": question,
                    "answer": item.get("answer", ""),
                    "type": q_type,
                }
            )
            gt_item = groundtruth_items_by_id.get(item.get("id"))
            if gt_item is not None:
                merged_groundtruth.append(
                    {
                        **gt_item,
                        "id": merged_id,
                        "type": q_type,
                    }
                )
            next_id += 1

    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = os.path.join(data_path, output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=4, ensure_ascii=False)

    groundtruth_output_path = args.groundtruth_output
    if not os.path.isabs(groundtruth_output_path):
        groundtruth_output_path = os.path.join(data_path, groundtruth_output_path)
    with open(groundtruth_output_path, "w", encoding="utf-8") as f:
        json.dump(merged_groundtruth, f, indent=4, ensure_ascii=False)

    print(f"Merged {len(merged)} questions → {output_path}")
    print(f"Merged {len(merged_groundtruth)} groundtruth entries → {groundtruth_output_path}")


if __name__ == "__main__":
    main()
