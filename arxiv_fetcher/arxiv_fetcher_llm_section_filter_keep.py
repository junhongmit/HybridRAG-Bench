"""
ArXiv fetcher + Docling + LLM section filtering (KEEP/EXCLUDE) → Markdown.


Quick start (RITS)
------------------
export RITS_API_KEY=********
# (optional overrides)
# export RITS_MODEL_ENDPOINT="https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/llama-3-3-70b-instruct"
# export RITS_MODEL_NAME="meta-llama/llama-3-3-70b-instruct"

python arxiv_fetcher_llm_section_filter_keep.py \
  --cats cs.AI --keywords "reinforcement learning" \
  --keep "Abstract, Introduction" \
  --rits-endpoint "$RITS_MODEL_ENDPOINT" \
  --rits-model "$RITS_MODEL_NAME" \
  --limit 5 --outdir pdfs --md-outdir mds_llm --frontmatter

Notes
-----
- If both --keep and --exclude are set, --keep takes precedence.
- Matching is case-insensitive and robust to numbered headings (e.g., "1 Introduction").
- The LLM step (via RITS) classifies blocks into canonical labels (abstract, introduction,
  methods, experiments/results, related_work, discussion, conclusion, acknowledgments,
  references, appendix, authors, ethics, front_matter, other). We map many synonyms.
- Install: pip install docling requests

"""

from __future__ import annotations

import argparse
import backoff
import csv
import json
import os
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

import requests  # RITS HTTP
from docling.document_converter import DocumentConverter

from urllib.error import URLError
from openai import RateLimitError, APIConnectionError, APITimeoutError
RETRYABLE_ERRORS = (URLError, RateLimitError, APIConnectionError, APITimeoutError)
def default_on_backoff(details):
    print(f"Backing off {details['wait']:0.1f}s after {details['tries']} tries due to {details['exception']}")

ARXIV_API = "http://export.arxiv.org/api/query"

# ============================== RITS defaults =================================
# You can override via env: RITS_MODEL_ENDPOINT / RITS_MODEL_NAME
RITS_API_KEY = os.getenv("RITS_API_KEY")
RITS_MODEL_ENDPOINT = os.getenv(
    "RITS_MODEL_ENDPOINT",
    "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/llama-3-3-70b-instruct",
)
RITS_MODEL_NAME = os.getenv("RITS_MODEL_NAME", "meta-llama/llama-3-3-70b-instruct")
RITS_CHAT_PATH = "/v1/chat/completions"

# ============================== Utilities =====================================

def _iso_date(s: str | None):
    if not s:
        return None
    fmts = ("%Y-%m-%d", "%Y-%m")
    for f in fmts:
        try:
            return datetime.strptime(s, f).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    raise ValueError(f"Invalid date: {s}. Use YYYY-MM-DD or YYYY-MM")


def build_search_query(categories, keywords):
    cat_part = None
    if categories:
        cat_terms = [f"cat:{c.strip()}" for c in categories]
        cat_part = "(" + " OR ".join(cat_terms) + ")"
    kw_part = None
    if keywords:
        if (" " not in keywords and "OR" not in keywords and "AND" not in keywords):
            kw_part = f'all:"{keywords}"'
        else:
            kw_part = f"all:{keywords}"
    if cat_part and kw_part:
        return f"{cat_part} AND {kw_part}"
    return cat_part or kw_part or "all:*"


@backoff.on_exception(
        backoff.expo,
        RETRYABLE_ERRORS,
        max_tries=8,
        on_backoff=default_on_backoff,
    )
def fetch_page(search_query, start=0, max_results=100):
    params = {"search_query": search_query, "start": str(start), "max_results": str(max_results)}
    url = ARXIV_API + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url) as resp:
        data = resp.read()
    return data


def parse_atom(atom_xml_bytes):
    ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
    root = ET.fromstring(atom_xml_bytes)
    out = []
    for e in root.findall("atom:entry", ns):
        eid = e.findtext("atom:id", default="", namespaces=ns)
        title = (e.findtext("atom:title", default="", namespaces=ns) or "").strip().replace("\n", " ")
        summary = (e.findtext("atom:summary", default="", namespaces=ns) or "").strip().replace("\n", " ")
        published = e.findtext("atom:published", default="", namespaces=ns)
        updated = e.findtext("atom:updated", default="", namespaces=ns)
        cats = [c.get("term", "") for c in e.findall("atom:category", ns)]
        authors = [a.findtext("atom:name", default="", namespaces=ns) for a in e.findall("atom:author", ns)]
        pdf_link, html_link = None, None
        for link in e.findall("atom:link", ns):
            rel = link.get("rel", ""); href = link.get("href", ""); title_attr = link.get("title", "")
            if title_attr == "pdf" or (href and href.endswith(".pdf")):
                pdf_link = href
            if rel == "alternate" and href.startswith("http"):
                html_link = href
        arxiv_id = eid.rsplit("/", 1)[-1]
        out.append({
            "id": arxiv_id,
            "title": title,
            "summary": summary,
            "published": published,
            "updated": updated,
            "categories": ";".join(cats),
            "authors": ";".join(authors),
            "pdf_url": pdf_link or f"https://arxiv.org/pdf/{arxiv_id}.pdf",
            "abs_url": html_link or f"https://arxiv.org/abs/{arxiv_id}",
        })
    return out


def within_date_window(iso_str, start_dt, end_dt):
    if not (start_dt or end_dt):
        return True
    if not iso_str:
        return False
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    except ValueError:
        return False
    if start_dt and dt < start_dt:
        return False
    if end_dt and dt > end_dt:
        return False
    return True


def save_csv(rows, path):
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cols = ["id","title","published","updated","categories","authors","abs_url","pdf_url","summary"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})


def safe_filename(s):
    bad = '<>:"/\\|?*'
    for ch in bad:
        s = s.replace(ch, "_")
    return s


def download_pdf(item, outdir="pdfs", sleep_sec=3.0):
    os.makedirs(outdir, exist_ok=True)
    arxiv_id = item["id"]; pdf_url = item["pdf_url"]
    fn = safe_filename(arxiv_id) + ".pdf"; path = os.path.join(outdir, fn)
    if os.path.exists(path):
        return path
    req = urllib.request.Request(pdf_url, headers={"User-Agent": "arxiv-llm-filter/1.0"})
    with urllib.request.urlopen(req) as resp, open(path, "wb") as f:
        f.write(resp.read())
    time.sleep(sleep_sec)
    return path

# ============================== Markdown parsing ==============================

_HEADING_RE = re.compile(r"^(#{1,6})\s*(.+?)\s*$", re.MULTILINE)
_ABSTRACT_BOLD_RE = re.compile(r"(?:^|\n)\s*(\*\*|__)\s*abstract\s*(\*\*|__)\s*[:.\-–—]?\s*\n", re.IGNORECASE)


def _normalize_title(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^(appendix\s+[a-z]\s*[:.\-–—]\s*|\d+(?:\.\d+)*\s*[:.\-–—]?\s*)", "", s, flags=re.IGNORECASE)
    return s.lower()


@dataclass
class Block:
    idx: int
    start: int
    end: int
    level: int
    heading: Optional[str]
    text: str


def md_to_blocks(md: str) -> List[Block]:
    heads = [(m.start(), m.end(), len(m.group(1)), m.group(2)) for m in _HEADING_RE.finditer(md)]
    blocks: List[Block] = []
    if not heads:
        # No headings; split by double newlines as pseudo-blocks
        parts = re.split(r"\n\s*\n", md)
        pos = 0; idx = 0
        for p in parts:
            if not p.strip():
                pos += len(p) + 2
                continue
            start = md.find(p, pos)
            end = start + len(p)
            blocks.append(Block(idx=idx, start=start, end=end, level=0, heading=None, text=p))
            pos = end; idx += 1
        return blocks

    # With headings: each block is a section from heading i to before heading j
    for i, (s, e, lvl, title) in enumerate(heads):
        end_idx = len(md) if i == len(heads) - 1 else heads[i + 1][0]
        text = md[e:end_idx]
        blocks.append(Block(idx=i, start=s, end=end_idx, level=lvl, heading=title, text=text))
    return blocks

# ============================== Label space ===================================

LABEL_SYNONYMS: Dict[str, List[str]] = {
    "abstract": ["abstract", "summary"],
    "introduction": ["introduction", "background", "overview", "motivation"],
    "related_work": ["related work", "literature review", "prior work", "background and related work"],
    "methods": ["methods", "method", "approach", "model", "methodology", "architecture"],
    "experiments": ["experiments", "results", "evaluation", "empirical study", "analysis"],
    "discussion": ["discussion", "insights", "ablation", "error analysis"],
    "conclusion": ["conclusion", "conclusions", "summary and conclusions", "future work"],
    "acknowledgments": ["acknowledgments", "acknowledgements", "funding"],
    "references": ["references", "bibliography"],
    "appendix": ["appendix", "supplementary", "supplemental"],
    "authors": ["authors", "author contributions", "affiliations", "biography"],
    "ethics": ["ethics", "impact statement"],
    "front_matter": ["front matter", "title block", "metadata"],
    "other": ["other"],
}

CANONICAL_LABELS = list(LABEL_SYNONYMS.keys())


def _normalize_label(s: str) -> str:
    s = s.strip().lower()
    for canon, syns in LABEL_SYNONYMS.items():
        if s == canon or any(s == x for x in syns):
            return canon
    for canon, syns in LABEL_SYNONYMS.items():
        if any(x in s for x in [canon] + syns):
            return canon
    return "other"


def choose_labels_from_titles(heading: Optional[str]) -> str:
    if not heading:
        return "other"
    norm = _normalize_title(heading)
    for canon, syns in LABEL_SYNONYMS.items():
        if canon in norm or any(x in norm for x in syns):
            return canon
    return "other"

# ============================== RITS chat =====================================

def rits_chat(system_prompt: str, user_prompt: str, *, endpoint: str, model_name: str, timeout: int = 60) -> str:
    if not RITS_API_KEY:
        raise RuntimeError("Set RITS_API_KEY in your environment.")
    url = f"{endpoint}{RITS_CHAT_PATH}"
    headers = {
        "accept": "application/json",
        "RITS_API_KEY": RITS_API_KEY,
        "Content-Type": "application/json",
    }
    body = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "model": model_name,
    }
    resp = requests.post(url, json=body, headers=headers, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        raise RuntimeError(f"Unexpected RITS response: {data}") from e

# ============================== LLM classification ============================

def llm_classify_blocks(
    blocks: List[Block],
    keep_targets: List[str],
    endpoint: str,
    model_name: str,
    temperature: float = 0.0,
    batch_size: int = 8,
    max_chars_per_block: int = 1500,
) -> Dict[int, str]:
    """Return mapping block_idx -> canonical_label using RITS provider."""
    out: Dict[int, str] = {}

    keep_lower = [k.strip().lower() for k in keep_targets if k.strip()]

    # Compose label list emphasizing keep targets first
    preferred_order = []
    for canon in CANONICAL_LABELS:
        if canon in keep_lower:
            preferred_order.append(canon)
    for canon in CANONICAL_LABELS:
        if canon not in preferred_order:
            preferred_order.append(canon)

    def to_spec(b: Block) -> Dict[str, str]:
        content = b.text.strip()
        if len(content) > max_chars_per_block:
            content = content[:max_chars_per_block] + "\n..."
        return {
            "index": b.idx,
            "heading": (b.heading or ""),
            "level": b.level,
            "content": content,
        }

    for i in range(0, len(blocks), batch_size):
        batch = blocks[i:i + batch_size]
        spec = [to_spec(b) for b in batch]
        sys_prompt = (
            "You are a scholarly document segment labeler.\n"
            "Given blocks from a scientific paper (heading + content snippet), classify each block into ONE of these canonical labels: "
            + ", ".join(preferred_order) + ".\n"
            "Rules:\n"
            "- Use 'abstract' for the Abstract paragraph(s) even if not formatted as a heading.\n"
            "- Use 'introduction' for opening motivation/overview sections.\n"
            "- 'methods' includes approach/model/architecture.\n"
            "- 'experiments' includes evaluation/results/analysis.\n"
            "- 'related_work' for literature surveys.\n"
            "- 'acknowledgments' for funding thanks.\n"
            "- 'references' for bibliography/citations list.\n"
            "- 'appendix' for supplementary material.\n"
            "- 'authors' for author/affiliation/contact blocks.\n"
            "- 'front_matter' for title/metadata boilerplate before the first section.\n"
            "Output STRICT JSON: {\n  \"labels\": [{\"index\": int, \"label\": str}]\n}\n"
        )
        user_prompt = (
            "Keep targets (lowercased): " + json.dumps(keep_lower) + "\n\n"
            "Blocks to classify: " + json.dumps(spec, ensure_ascii=False)
        )

        content = rits_chat(
            system_prompt=sys_prompt,
            user_prompt=user_prompt,
            endpoint=endpoint,
            model_name=model_name,
        )

        # Parse JSON
        try:
            data = json.loads(content)
        except Exception:
            m = re.search(r"\{[\s\S]*\}", content)
            if not m:
                raise RuntimeError(f"LLM returned non-JSON: {content[:200]}")
            data = json.loads(m.group(0))
        labels = data.get("labels", [])
        for item in labels:
            idx = int(item.get("index"))
            lbl = _normalize_label(str(item.get("label", "other")))
            out[idx] = lbl
    return out

# ============================== Filtering pipeline ============================

def apply_keep_exclude(
    md: str,
    keep: List[str],
    exclude: List[str],
    rits_endpoint: str,
    rits_model_name: str,
    temperature: float,
    batch_size: int,
    max_chars_per_block: int,
) -> str:
    keep_norm = [_normalize_label(k) for k in keep]
    exclude_norm = [_normalize_label(k) for k in exclude]

    blocks = md_to_blocks(md)

    # Heuristic heading-based match first
    matched: Dict[int, str] = {}
    for b in blocks:
        lbl = choose_labels_from_titles(b.heading)
        matched[b.idx] = lbl

    # If keep is specified, call LLM to ensure recall
    if keep_norm:
        llm_labels = llm_classify_blocks(
            blocks=blocks,
            keep_targets=keep_norm,
            endpoint=rits_endpoint,
            model_name=rits_model_name,
            temperature=temperature,
            batch_size=batch_size,
            max_chars_per_block=max_chars_per_block,
        )
        matched.update(llm_labels)

    def want(lbl: str) -> bool:
        if keep_norm:
            return lbl in keep_norm
        if exclude_norm:
            return lbl not in exclude_norm
        return True

    kept_chunks: List[str] = []
    for b in blocks:
        lbl = matched.get(b.idx, "other")
        if want(lbl):
            heading = f"{'#'*max(1,b.level)} {b.heading}\n\n" if b.heading else ""
            kept_chunks.append(heading + b.text.strip() + "\n\n")

    out_md = "".join(kept_chunks).strip() or md

    # Abstract fallback: if requested but missing and bold **Abstract** exists
    if "abstract" in keep_norm and "#" not in out_md:
        m = _ABSTRACT_BOLD_RE.search(md)
        if m:
            start = m.start()
            next_h = _HEADING_RE.search(md, pos=start)
            end = next_h.start() if next_h else len(md)
            out_md = md[start:end].strip() + "\n\n" + out_md
    return out_md

# ============================== PDF → Markdown ================================

def pdf_to_markdown_llm(
    pdf_path: str,
    md_outdir: str = "mds_llm",
    keep_sections: Optional[List[str]] = None,
    exclude_sections: Optional[List[str]] = None,
    add_frontmatter: bool = False,
    meta: Optional[dict] = None,
    rits_endpoint: Optional[str] = None,
    rits_model_name: Optional[str] = None,
    llm_temperature: float = 0.0,
    llm_batch_size: int = 8,
    llm_max_chars_per_block: int = 1500,
) -> str:
    os.makedirs(md_outdir, exist_ok=True)
    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    md_text = result.document.export_to_markdown()

    keep_sections = keep_sections or []
    exclude_sections = exclude_sections or []

    md_text = apply_keep_exclude(
        md_text,
        keep=keep_sections,
        exclude=exclude_sections,
        rits_endpoint=rits_endpoint or RITS_MODEL_ENDPOINT,
        rits_model_name=rits_model_name or RITS_MODEL_NAME,
        temperature=llm_temperature,
        batch_size=llm_batch_size,
        max_chars_per_block=llm_max_chars_per_block,
    )

    if add_frontmatter and meta:
        fm_lines = ["---"]
        for k in ["id", "title", "authors", "published", "updated", "categories", "abs_url", "pdf_url"]:
            v = meta.get(k, "") if meta else ""
            if isinstance(v, str):
                v = v.replace("\n", " ")
            fm_lines.append(f"{k}: {v}")
        fm_lines.append("---\n")
        md_text = "\n".join(fm_lines) + md_text

    md_path = os.path.join(md_outdir, os.path.basename(pdf_path).replace(".pdf", ".md"))
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)
    return md_path

# ============================== Main ==========================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cats", nargs="*", default=[], help="arXiv subject categories (e.g., cs.AI cs.CL stat.ML)")
    ap.add_argument("--keywords", type=str, default="", help="Keyword query (supports boolean)")
    ap.add_argument("--start-date", type=str, default="", help="YYYY-MM-DD or YYYY-MM")
    ap.add_argument("--end-date", type=str, default="", help="YYYY-MM-DD or YYYY-MM")
    ap.add_argument("--limit", type=int, default=200, help="Max total results to fetch")
    ap.add_argument("--per-page", type=int, default=100, help="Items per API page (<=300)")
    ap.add_argument("--sleep", type=float, default=3.0, help="Seconds to sleep between requests/downloads")
    ap.add_argument("--outdir", type=str, default="pdfs", help="Where to save PDFs")
    ap.add_argument("--csv", type=str, default="arxiv_results.csv", help="Where to save metadata CSV")
    ap.add_argument("--no-download", action="store_true", help="Only save CSV; do not download PDFs")

    # Filtering controls
    ap.add_argument("--keep", type=str, default="", help="Comma-separated section titles to KEEP (e.g., 'Abstract, Introduction')")
    ap.add_argument("--exclude", type=str, default="", help="Comma-separated section titles to drop (ignored if --keep is set)")
    ap.add_argument("--md-outdir", type=str, default="mds_llm", help="Output directory for Markdown files")
    ap.add_argument("--frontmatter", action="store_true", help="Prepend YAML front matter with arXiv metadata")

    # RITS parameters (with env defaults)
    ap.add_argument("--rits-endpoint", type=str, default=os.getenv("RITS_MODEL_ENDPOINT", RITS_MODEL_ENDPOINT), help="RITS model endpoint base (without /v1/chat/completions)")
    ap.add_argument("--rits-model", type=str, default=os.getenv("RITS_MODEL_NAME", RITS_MODEL_NAME), help="RITS model name")

    # LLM misc params
    ap.add_argument("--llm-temp", type=float, default=0.0, help="LLM temperature for classification")
    ap.add_argument("--llm-batch", type=int, default=8, help="Blocks per LLM request")
    ap.add_argument("--llm-max-chars", type=int, default=1500, help="Max characters per block sent to LLM")

    args = ap.parse_args()

    start_dt = _iso_date(args.start_date)
    end_dt = _iso_date(args.end_date)

    search_query = build_search_query(args.cats, args.keywords)
    print(f"[query] {search_query}")

    all_rows = []
    start = 0
    per_page = max(1, min(args.per_page, 300))
    while len(all_rows) < args.limit:
        xml = fetch_page(search_query, start=start, max_results=per_page)
        entries = parse_atom(xml)
        if not entries:
            break
        for e in entries:
            if within_date_window(e["published"], start_dt, end_dt):
                all_rows.append(e)
                if len(all_rows) >= args.limit:
                    break
        print(f"[page] got {len(entries)} entries, kept {len(all_rows)} so far")
        start += per_page
        time.sleep(args.sleep)
        if len(entries) < per_page:
            break

    save_csv(all_rows, args.csv)
    print(f"[csv] wrote {len(all_rows)} rows to {args.csv}")

    if args.no_download:
        return

    keep_sections = [s.strip() for s in args.keep.split(",") if s.strip()] if args.keep else []
    exclude_sections = [s.strip() for s in args.exclude.split(",") if s.strip()] if args.exclude else []

    for i, row in enumerate(all_rows, 1):
        try:
            pdf_path = download_pdf(row, outdir=args.outdir, sleep_sec=args.sleep)
            print(f"[pdf {i}/{len(all_rows)}] {row['id']} -> {pdf_path}")

            md_path = pdf_to_markdown_llm(
                pdf_path,
                md_outdir=args.md_outdir,
                keep_sections=keep_sections,
                exclude_sections=exclude_sections if not keep_sections else [],
                add_frontmatter=args.frontmatter,
                meta=row,
                rits_endpoint=args.rits_endpoint,
                rits_model_name=args.rits_model,
                llm_temperature=args.llm_temp,
                llm_batch_size=args.llm_batch,
                llm_max_chars_per_block=args.llm_max_chars,
            )
            print(f"[md {i}/{len(all_rows)}] {row['id']} -> {md_path}")
        except Exception as ex:
            print(f"[warn] failed {row['id']}: {ex}")


if __name__ == "__main__":
    main()
