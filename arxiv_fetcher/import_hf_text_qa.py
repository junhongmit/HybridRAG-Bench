#!/usr/bin/env python3
"""
Reconstruct local text+QA dataset folders from HF parquet package.

Input (packaged parquet):
  <text_qa_root>/<domain>/papers.parquet
  <text_qa_root>/<domain>/qa.parquet

Output (local DATASET_PATH-like layout):
  <out_data_root>/<DomainName>/md/*.md
  <out_data_root>/<DomainName>/questions.json

Domain mapping defaults:
  arxiv_ai -> arxiv_AI
  arxiv_qm -> arxiv_QM
  arxiv_cy -> arxiv_CY
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


DEFAULT_DOMAIN_MAP = {
    "arxiv_ai": "arxiv_AI",
    "arxiv_qm": "arxiv_QM",
    "arxiv_cy": "arxiv_CY",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Import packaged text_qa parquet into local DATASET_PATH layout")
    ap.add_argument("--text-qa-root", required=True, help="Root folder containing per-domain parquet files")
    ap.add_argument("--out-data-root", required=True, help="Target DATASET_PATH root")
    ap.add_argument(
        "--domains",
        nargs="*",
        default=["arxiv_ai", "arxiv_qm", "arxiv_cy"],
        help="Domain folder names under text_qa root",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing md/json files",
    )
    return ap.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_int(value, fallback: int) -> int:
    try:
        return int(value)
    except Exception:
        return fallback


def to_frontmatter(row: dict, paper_id: str) -> str:
    lines = [
        "---",
        f"id: {paper_id}",
        f"title: {row.get('title', '')}",
        f"authors: {row.get('authors', '')}",
        f"published: {row.get('published', '')}",
        f"updated: {row.get('updated', '')}",
        f"categories: {row.get('categories', '')}",
        f"abs_url: {row.get('abs_url', '')}",
        f"pdf_url: {row.get('pdf_url', '')}",
        "---",
    ]
    return "\n".join(lines)


def paper_filename_from_id(paper_id: str) -> str:
    """
    Normalize arXiv id to a flat markdown filename under md/.
    Example:
      q-bio/0405015v1 -> 0405015v1.md
      2501.12345v1 -> 2501.12345v1.md
    """
    pid = (paper_id or "").strip().strip("/")
    if not pid:
        return ""
    stem = pid.rsplit("/", 1)[-1]
    return f"{stem}.md"


def write_markdown_files(papers_df: pd.DataFrame, md_dir: Path, overwrite: bool) -> int:
    count = 0
    for _, r in papers_df.iterrows():
        row = r.to_dict()
        paper_id = str(row.get("arxiv_id", "")).strip()
        if not paper_id:
            continue
        filename = paper_filename_from_id(paper_id)
        if not filename:
            continue
        md_path = md_dir / filename
        if md_path.exists() and (not overwrite):
            continue
        fm = to_frontmatter(row, paper_id)
        body = str(row.get("md_text", ""))
        content = fm + "\n" + body.strip() + "\n"
        md_path.write_text(content, encoding="utf-8")
        count += 1
    return count


def build_qa_records(df: pd.DataFrame) -> List[dict]:
    records: List[dict] = []
    for i, r in enumerate(df.to_dict("records")):
        q = str(r.get("question", "")).strip()
        a = str(r.get("answer", "")).strip()
        if not q or not a:
            continue
        qid = safe_int(r.get("question_id"), i)
        qtype = str(r.get("question_type", "")).strip() or "unknown"
        rec = {
            "id": qid,
            "question": q,
            "answer": a,
            "type": qtype,
        }
        records.append(rec)
    return records


def write_json(path: Path, payload: List[dict], overwrite: bool) -> bool:
    if path.exists() and (not overwrite):
        return False
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=4), encoding="utf-8")
    return True


def write_question_files(qa_df: pd.DataFrame, domain_dir: Path, overwrite: bool) -> Dict[str, int]:
    all_records = build_qa_records(qa_df)
    write_json(domain_dir / "questions.json", all_records, overwrite=overwrite)
    return {"questions.json": len(all_records)}


def import_one_domain(text_qa_root: Path, out_data_root: Path, in_domain: str, overwrite: bool) -> None:
    in_domain_dir = text_qa_root / in_domain
    out_domain_name = DEFAULT_DOMAIN_MAP.get(in_domain, in_domain)
    out_domain_dir = out_data_root / out_domain_name
    md_dir = out_domain_dir / "md"

    papers_path = in_domain_dir / "papers.parquet"
    qa_path = in_domain_dir / "qa.parquet"
    if not papers_path.exists() or not qa_path.exists():
        print(f"[warn] skip {in_domain}: missing papers.parquet or qa.parquet")
        return

    ensure_dir(out_domain_dir)
    ensure_dir(md_dir)

    papers_df = pd.read_parquet(papers_path)
    qa_df = pd.read_parquet(qa_path)

    md_written = write_markdown_files(papers_df, md_dir, overwrite=overwrite)
    q_stats = write_question_files(qa_df, out_domain_dir, overwrite=overwrite)

    print(
        f"[domain] {in_domain} -> {out_domain_name}: "
        f"md_written={md_written}, total_questions={q_stats.get('questions.json', 0)}"
    )


def main() -> None:
    args = parse_args()

    text_qa_root = Path(args.text_qa_root).expanduser().resolve()
    out_data_root = Path(args.out_data_root).expanduser().resolve()

    if not text_qa_root.exists():
        raise FileNotFoundError(f"text_qa_root not found: {text_qa_root}")

    ensure_dir(out_data_root)

    for domain in args.domains:
        import_one_domain(
            text_qa_root=text_qa_root,
            out_data_root=out_data_root,
            in_domain=domain,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
