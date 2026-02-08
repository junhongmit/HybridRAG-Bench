"""
ArXiv fetcher + Docling converter with section KEEP filtering.

Usage examples:
  # Keep only Abstract and Introduction in the generated .md files
  python arxiv_fetcher_docling_section_keep.py --cats cs.AI --keywords "reinforcement learning" \
    --keep "Abstract, Introduction" --limit 10 --outdir pdfs --md-outdir mds_keep

  # With a date range and YAML front matter
  python arxiv_fetcher_docling_section_keep.py --cats cs.AI cs.LG --keywords "graph neural network" \
    --start-date 2024-01-01 --end-date 2025-09-30 --limit 5 --keep "Abstract, Introduction" \
    --frontmatter

Notes:
- Requires `docling` (pip install docling).
- If both --keep and --exclude are provided, --keep takes precedence.
- Matching is case-insensitive and robust to numbered headings (e.g., "1 Introduction").
- For Abstracts that are not emitted as headings, we apply a small fallback heuristic to capture
  a bolded **Abstract** block at the top.
"""

import argparse
import csv
import os
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import List, Tuple
import re

from docling.document_converter import DocumentConverter

ARXIV_API = "http://export.arxiv.org/api/query"

# --------------------------- Utility: dates & query ---------------------------

def _iso_date(s: str):
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
        # Allow boolean phrases; otherwise quote a single token
        if (" " not in keywords and "OR" not in keywords and "AND" not in keywords):
            kw_part = f'all:"{keywords}"'
        else:
            kw_part = f"all:{keywords}"
    if cat_part and kw_part:
        return f"{cat_part} AND {kw_part}"
    return cat_part or kw_part or "all:*"


def fetch_page(search_query, start=0, max_results=100):
    params = {
        "search_query": search_query,
        "start": str(start),
        "max_results": str(max_results),
    }
    url = ARXIV_API + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url) as resp:
        data = resp.read()
    return data


def parse_atom(atom_xml_bytes):
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }
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
            rel = link.get("rel", "")
            href = link.get("href", "")
            title_attr = link.get("title", "")
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


# --------------------------- IO helpers --------------------------------------

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
    arxiv_id = item["id"]
    pdf_url = item["pdf_url"]
    fn = safe_filename(arxiv_id) + ".pdf"
    path = os.path.join(outdir, fn)
    if os.path.exists(path):
        return path
    req = urllib.request.Request(pdf_url, headers={"User-Agent": "arxiv-fetcher/1.0"})
    with urllib.request.urlopen(req) as resp, open(path, "wb") as f:
        f.write(resp.read())
    time.sleep(sleep_sec)
    return path


# --------------------------- Markdown filtering ------------------------------

_HEADING_RE = re.compile(r'^(#{1,6})\s*(.+?)\s*$', re.MULTILINE)
_ABSTRACT_BOLD_RE = re.compile(r'(?:^|\n)\s*(\*\*|__)\s*abstract\s*(\*\*|__)\s*[:.\-–—]?\s*\n', re.IGNORECASE)


def _normalize_title(s: str) -> str:
    # Drop numbering and punctuation, lowercase
    s = s.strip()
    s = re.sub(r'^(appendix\s+[a-z]\s*[:.\-–—]\s*|\d+(?:\.\d+)*\s*[:.\-–—]?\s*)', '', s, flags=re.IGNORECASE)
    return s.lower()


def _find_headings(md: str) -> List[Tuple[int, int, int, str]]:
    """Return list of (start_idx, end_idx, level, title)."""
    out = []
    for m in _HEADING_RE.finditer(md):
        hashes = m.group(1)
        title = m.group(2)
        out.append((m.start(), m.end(), len(hashes), title))
    return out


def _merge_ranges(ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not ranges:
        return []
    ranges = sorted(ranges)
    merged = [ranges[0]]
    for s, e in ranges[1:]:
        ps, pe = merged[-1]
        if s <= pe:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def _slice(md: str, keep_ranges: List[Tuple[int, int]]) -> str:
    if not keep_ranges:
        return ""
    keep_ranges = _merge_ranges(keep_ranges)
    parts = []
    for s, e in keep_ranges:
        parts.append(md[s:e])
    return "".join(parts)


def keep_markdown_sections(md: str, keep: List[str]) -> str:
    """
    Keep only sections whose heading matches any token in 'keep'.
    - Matching is case-insensitive and robust to numbering like "1 Introduction".
    - A section spans from its heading to (but not including) the next heading
      of the same or higher level.
    - If 'abstract' is requested but isn't a heading, we also try to keep a
      bolded **Abstract** block at the top.
    """
    tokens = [t.strip().lower() for t in keep if t.strip()]
    if not tokens:
        return md

    heads = _find_headings(md)
    keep_ranges: List[Tuple[int, int]] = []

    # 1) Capture explicit heading-based sections
    for i, (s, e, lvl, title) in enumerate(heads):
        norm = _normalize_title(title)
        if any(tok in norm for tok in tokens):
            # find next heading with level <= lvl
            end_idx = len(md)
            for j in range(i + 1, len(heads)):
                s2, e2, lvl2, _ = heads[j]
                if lvl2 <= lvl:
                    end_idx = s2
                    break
            keep_ranges.append((s, end_idx))

    # 2) Fallback for **Abstract** block if requested & not found above
    if any(tok in ("abstract",) for tok in tokens):
        if not any("abstract" in _normalize_title(h[3]) for h in heads):
            m = _ABSTRACT_BOLD_RE.search(md)
            if m:
                start = m.start()
                # end at first heading after this position
                next_h = _HEADING_RE.search(md, pos=start)
                end = next_h.start() if next_h else len(md)
                keep_ranges.append((start, end))

    if not keep_ranges:
        # Nothing matched; fall back to original to avoid empty outputs
        return md

    return _slice(md, keep_ranges)


def exclude_markdown_sections(md: str, exclude: List[str]) -> str:
    tokens = [t.strip().lower() for t in exclude if t.strip()]
    if not tokens:
        return md
    heads = _find_headings(md)
    if not heads:
        return md
    kill_ranges: List[Tuple[int, int]] = []
    for i, (s, e, lvl, title) in enumerate(heads):
        norm = _normalize_title(title)
        if any(tok in norm for tok in tokens):
            end_idx = len(md)
            for j in range(i + 1, len(heads)):
                s2, e2, lvl2, _ = heads[j]
                if lvl2 <= lvl:
                    end_idx = s2
                    break
            kill_ranges.append((s, end_idx))
    # Build complement
    if not kill_ranges:
        return md
    kill_ranges = _merge_ranges(kill_ranges)
    parts = []
    last = 0
    for s, e in kill_ranges:
        if s > last:
            parts.append(md[last:s])
        last = max(last, e)
    parts.append(md[last:])
    return "".join(parts)


# --------------------------- PDF -> Markdown ---------------------------------

def pdf_to_markdown_filtered(
    pdf_path: str,
    md_outdir: str = "mds",
    keep_sections: List[str] | None = None,
    exclude_sections: List[str] | None = None,
    add_frontmatter: bool = False,
    meta: dict | None = None,
) -> str:
    os.makedirs(md_outdir, exist_ok=True)
    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    md_text = result.document.export_to_markdown()

    # Apply KEEP first (if provided). If keep is set, we ignore exclude.
    if keep_sections:
        md_text = keep_markdown_sections(md_text, keep_sections)
    elif exclude_sections:
        md_text = exclude_markdown_sections(md_text, exclude_sections)

    if add_frontmatter and meta:
        # Minimal YAML front matter
        fm_lines = ["---"]
        for k in ["id", "title", "authors", "published", "updated", "categories", "abs_url", "pdf_url"]:
            v = meta.get(k, "") if meta else ""
            # Escape newlines in title/others
            if isinstance(v, str):
                v = v.replace("\n", " ")
            fm_lines.append(f"{k}: {v}")
        fm_lines.append("---\n")
        md_text = "\n".join(fm_lines) + md_text

    md_path = os.path.join(md_outdir, os.path.basename(pdf_path).replace(".pdf", ".md"))
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)
    return md_path


# --------------------------- Main --------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cats", nargs="*", default=[], help="arXiv categories (e.g., cs.AI cs.CL stat.ML)")
    ap.add_argument("--keywords", type=str, default="", help="Keyword query (supports boolean phrases)")
    ap.add_argument("--start-date", type=str, default="", help="YYYY-MM-DD or YYYY-MM")
    ap.add_argument("--end-date", type=str, default="", help="YYYY-MM-DD or YYYY-MM")
    ap.add_argument("--limit", type=int, default=200, help="Max total results to fetch")
    ap.add_argument("--per-page", type=int, default=100, help="Items per API page (<=300)")
    ap.add_argument("--sleep", type=float, default=3.0, help="Seconds to sleep between requests/downloads")
    ap.add_argument("--outdir", type=str, default="pdfs", help="Where to save PDFs")
    ap.add_argument("--csv", type=str, default="arxiv_results.csv", help="Where to save metadata CSV")
    ap.add_argument("--no-download", action="store_true", help="Only save CSV; do not download PDFs")

    # Filtering options
    ap.add_argument("--keep", type=str, default="", help="Comma-separated section titles to KEEP (e.g., 'Abstract, Introduction')")
    ap.add_argument("--exclude", type=str, default="", help="Comma-separated section titles to drop (ignored if --keep is set)")
    ap.add_argument("--md-outdir", type=str, default="mds_keep", help="Output directory for Markdown files")
    ap.add_argument("--frontmatter", action="store_true", help="Prepend YAML front matter with arXiv metadata")

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

            md_path = pdf_to_markdown_filtered(
                pdf_path,
                md_outdir=args.md_outdir,
                keep_sections=keep_sections,
                exclude_sections=exclude_sections,
                add_frontmatter=args.frontmatter,
                meta=row,
            )
            print(f"[md {i}/{len(all_rows)}] {row['id']} -> {md_path}")
        except Exception as ex:
            print(f"[warn] failed {row['id']}: {ex}")


if __name__ == "__main__":
    main()
