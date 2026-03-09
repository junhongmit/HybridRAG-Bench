#!/usr/bin/env python3
"""
Package HybridRAG text + QA into Hugging Face-friendly parquet files.

Inputs:
- Domain folders under data root (for example: arxiv_AI, arxiv_QM, arxiv_CY)
- Markdown papers under <domain>/md
- Question JSON files (default: questions*.json, excluding *rejected*)
- Existing license report JSON (offline) with hf_publish_decision fields

Outputs:
- <out-dir>/<domain>/papers.parquet
- <out-dir>/<domain>/qa.parquet
- <out-dir>/papers_all.parquet
- <out-dir>/qa_all.parquet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Package HybridRAG text/QA into parquet")
    ap.add_argument("--data-root", required=True, help="Root dir containing arxiv_* domain folders")
    ap.add_argument(
        "--domains",
        nargs="*",
        default=["arxiv_AI", "arxiv_QM", "arxiv_CY"],
        help="Domain directories under data root",
    )
    ap.add_argument(
        "--license-report-json",
        required=True,
        help="Path to arxiv_license_report.json with hf_publish_decision",
    )
    ap.add_argument(
        "--out-dir",
        default="HybridRAG-Bench/release/hf_text_qa",
        help="Output directory",
    )
    ap.add_argument(
        "--question-glob",
        default="questions*.json",
        help="Glob for question files under each domain dir",
    )
    ap.add_argument(
        "--include-rejected",
        action="store_true",
        help="Include question files containing 'rejected'",
    )
    ap.add_argument(
        "--license-allow-prefix",
        default="allow_full_text",
        help="Only include papers where hf_publish_decision starts with this prefix",
    )
    return ap.parse_args()


def normalize_domain(domain: str) -> str:
    return domain.strip().lower()


def parse_frontmatter(md_path: Path) -> Tuple[Dict[str, str], str]:
    try:
        text = md_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return {}, ""

    if not text.startswith("---\n"):
        return {}, text

    end_idx = text.find("\n---\n", 4)
    if end_idx == -1:
        return {}, text

    header_block = text[4:end_idx]
    body = text[end_idx + 5 :]

    meta: Dict[str, str] = {}
    for line in header_block.splitlines():
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        meta[key.strip()] = val.strip()
    return meta, body.strip()


def arxiv_id_from_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""
    # Typical forms:
    # https://arxiv.org/abs/2501.12345v1
    # https://arxiv.org/abs/q-bio/0405015v1
    # https://arxiv.org/pdf/q-bio/0405015v1
    for token in ("/abs/", "/pdf/"):
        idx = url.find(token)
        if idx >= 0:
            rid = url[idx + len(token) :]
            if rid.endswith(".pdf"):
                rid = rid[:-4]
            return rid.strip("/")
    return ""


def load_license_map(report_json: Path) -> Dict[str, dict]:
    payload = json.loads(report_json.read_text(encoding="utf-8"))
    out: Dict[str, dict] = {}
    for item in payload:
        if not isinstance(item, dict):
            continue
        arxiv_id = str(item.get("arxiv_id", "")).strip()
        if not arxiv_id:
            continue
        out[arxiv_id] = item
        out[arxiv_id.rsplit("/", 1)[-1]] = item
    return out


def infer_question_type(source_filename: str, row: dict) -> str:
    explicit = str(row.get("type", "")).strip()
    if explicit:
        return explicit

    name = source_filename.lower()
    if "single_hop_w_conditions" in name:
        return "single_hop_w_conditions"
    if "single_hop" in name:
        return "single_hop"
    if "multi_hop_difficult" in name:
        return "multi_hop_difficult"
    if "multi_hop" in name:
        return "multi_hop"
    if "counterfactual" in name:
        return "counterfactual"
    if "open_ended" in name:
        return "open_ended"
    return "unknown"


def paper_key_variants(raw_id: str) -> List[str]:
    raw_id = (raw_id or "").strip()
    if not raw_id:
        return []
    return [raw_id, raw_id.rsplit("/", 1)[-1]]


def build_papers_for_domain(
    domain_dir: Path,
    domain_norm: str,
    license_map: Dict[str, dict],
    allow_prefix: str,
) -> pd.DataFrame:
    md_dir = domain_dir / "md"
    rows: List[dict] = []

    if not md_dir.exists():
        return pd.DataFrame()

    for md_path in sorted(md_dir.glob("*.md")):
        meta, md_text = parse_frontmatter(md_path)

        raw_id = meta.get("id", md_path.stem)
        abs_url = meta.get("abs_url", "")
        url_id = arxiv_id_from_url(abs_url)
        arxiv_id = url_id or raw_id

        lic = None
        for key in paper_key_variants(arxiv_id):
            lic = license_map.get(key)
            if lic:
                break

        hf_decision = str((lic or {}).get("hf_publish_decision", "metadata_only_review"))
        if not hf_decision.startswith(allow_prefix):
            continue

        rows.append(
            {
                "domain": domain_norm,
                "split": "test",
                "arxiv_id": arxiv_id,
                "title": meta.get("title", ""),
                "authors": meta.get("authors", ""),
                "published": meta.get("published", ""),
                "updated": meta.get("updated", ""),
                "categories": meta.get("categories", ""),
                "abs_url": abs_url,
                "pdf_url": meta.get("pdf_url", ""),
                "md_text": md_text,
                "license_url": str((lic or {}).get("license_url", "")),
                "hf_publish_decision": hf_decision,
            }
        )

    return pd.DataFrame(rows)


def build_qa_for_domain(
    domain_dir: Path,
    domain_norm: str,
    include_rejected: bool,
    question_glob: str,
) -> pd.DataFrame:
    rows: List[dict] = []
    qfiles = sorted(domain_dir.glob(question_glob))

    for qf in qfiles:
        lname = qf.name.lower()
        if (not include_rejected) and ("rejected" in lname):
            continue

        try:
            payload = json.loads(qf.read_text(encoding="utf-8"))
        except Exception:
            continue

        if not isinstance(payload, list):
            continue

        for idx, item in enumerate(payload):
            if not isinstance(item, dict):
                continue
            q = str(item.get("question", "")).strip()
            a = str(item.get("answer", "")).strip()
            if not q or not a:
                continue

            qid = item.get("id", idx)

            rows.append(
                {
                    "domain": domain_norm,
                    "split": "test",
                    "question_id": str(qid),
                    "question": q,
                    "answer": a,
                    "question_type": infer_question_type(qf.name, item)
                }
            )

    return pd.DataFrame(rows)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    ensure_parent(path)
    df.to_parquet(path, index=False)


def main() -> None:
    args = parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    report_json = Path(args.license_report_json).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    if not data_root.exists():
        raise FileNotFoundError(f"data root does not exist: {data_root}")
    if not report_json.exists():
        raise FileNotFoundError(f"license report does not exist: {report_json}")

    license_map = load_license_map(report_json)

    all_papers: List[pd.DataFrame] = []
    all_qa: List[pd.DataFrame] = []

    for domain in args.domains:
        domain_dir = data_root / domain
        domain_norm = normalize_domain(domain)

        if not domain_dir.exists():
            print(f"[warn] skip missing domain dir: {domain_dir}")
            continue

        papers_df = build_papers_for_domain(
            domain_dir=domain_dir,
            domain_norm=domain_norm,
            license_map=license_map,
            allow_prefix=args.license_allow_prefix,
        )
        qa_df = build_qa_for_domain(
            domain_dir=domain_dir,
            domain_norm=domain_norm,
            include_rejected=args.include_rejected,
            question_glob=args.question_glob,
        )

        domain_out = out_dir / domain_norm
        write_parquet(papers_df, domain_out / "papers.parquet")
        write_parquet(qa_df, domain_out / "qa.parquet")

        all_papers.append(papers_df)
        all_qa.append(qa_df)

        print(f"[domain] {domain_norm}: papers={len(papers_df)} qa={len(qa_df)}")

    # papers_all = pd.concat(all_papers, ignore_index=True) if all_papers else pd.DataFrame()
    # qa_all = pd.concat(all_qa, ignore_index=True) if all_qa else pd.DataFrame()

    # write_parquet(papers_all, out_dir / "papers_all.parquet")
    # write_parquet(qa_all, out_dir / "qa_all.parquet")

    # print(f"[write] {out_dir / 'papers_all.parquet'} rows={len(papers_all)}")
    # print(f"[write] {out_dir / 'qa_all.parquet'} rows={len(qa_all)}")


if __name__ == "__main__":
    main()
