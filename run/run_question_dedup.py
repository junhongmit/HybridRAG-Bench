# python run_question_dedup.py \
#     /Users/bing.z/git/Bidirection/Data/100Arxiv_on_KG/questions_open_ended_version2.json --method exact
# python run_question_dedup.py \
#     /Users/bing.z/git/Bidirection/Data/100Arxiv_on_KG/questions_open_ended_version2.json --method normalized
# run both:
# python run_question_dedup.py \
#     /Users/bing.z/git/Bidirection/Data/100Arxiv_on_KG/questions_open_ended_version2.json

import json
import re
import sys
import os
import argparse

def normalize(text: str) -> str:
    """Normalize text by lowercasing, removing punctuation, and collapsing spaces."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
    text = re.sub(r"\s+", " ", text)     # collapse multiple spaces
    return text.strip()

def dedup_exact(data):
    """Deduplicate based on exact string match."""
    seen = set()
    unique = []
    for item in data:
        q = item["question"]
        if q not in seen:
            seen.add(q)
            unique.append(item)
    return unique

def dedup_normalized(data):
    """Deduplicate based on normalized text."""
    seen = set()
    unique = []
    for item in data:
        q_norm = normalize(item["question"])
        if q_norm not in seen:
            seen.add(q_norm)
            unique.append(item)
    return unique

def save_output(data, input_file, suffix):
    """Save deduplicated data with appropriate suffix."""
    base, ext = os.path.splitext(input_file)
    output_file = f"{base}_dedup_{suffix}{ext}"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"Saved {len(data)} unique items → {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Deduplicate questions in a JSON file by exact or normalized text match."
    )
    parser.add_argument("input_file", help="Path to the input JSON file")
    parser.add_argument(
        "--method",
        choices=["exact", "normalized", "both"],
        default="both",
        help="Deduplication method: 'exact', 'normalized', or 'both' (default)"
    )

    args = parser.parse_args()
    input_file = args.input_file

    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)

    # Load JSON data
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Run deduplication(s)
    if args.method in ("exact", "both"):
        result_exact = dedup_exact(data)
        save_output(result_exact, input_file, "exact")

    if args.method in ("normalized", "both"):
        result_norm = dedup_normalized(data)
        save_output(result_norm, input_file, "normalized")

if __name__ == "__main__":
    main()
