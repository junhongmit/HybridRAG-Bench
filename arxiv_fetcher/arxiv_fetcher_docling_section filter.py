"""
arxiv_fetch_docling_section_filtering.py — search arXiv by category & keywords, save metadata, download PDFs, filter section, convert to .md.

Usage examples:
# Example1: 100 outputs
python arxiv_fetcher_docling_section_filter.py --keywords "reinforcement learning" --limit 100 --outdir pdfs
# Example2: other filter options
python arxiv_fetcher_docling_section_filter.py --cats cs.AI cs.LG stat.ML \
      --keywords "(graph neural network) OR (knowledge graph)" \
      --start-date 2024-01-01 --end-date 2025-09-18 \
      --limit 200 --per-page 100 --outdir pdfs
# Example3: fetch, convert, and save markdown without the author section or references
python arxiv_fetcher_docling_section_filter.py --cats cs.AI --keywords "reinforcement learning" \
  --exclude "Authors, Author Contributions, Acknowledgments, References" --strip-top-authors

"""
# pip install docling
import argparse, csv, os, time, urllib.parse, urllib.request, xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import List, Tuple
import re
from docling.document_converter import DocumentConverter


ARXIV_API = "http://export.arxiv.org/api/query"

def _iso_date(s: str):
    # Accept YYYY-MM-DD or YYYY-MM
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
        kw_part = f'all:"{keywords}"' if (' ' not in keywords and 'OR' not in keywords and 'AND' not in keywords) else f'all:{keywords}'
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
            if title_attr == "pdf" or href.endswith(".pdf"):
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

# ----------------------- NEW: Markdown filtering helpers -----------------------

_HEADING_RE = re.compile(r'^(#{1,6})\s*(.+?)\s*$', re.MULTILINE)

def _normalize_title(s: str) -> str:
    """Lowercase, drop leading numbering and punctuation for robust matching."""
    s = s.strip()
    s = re.sub(r'^(appendix\s+[a-z]\s*[:.\-–—]\s*|\d+(\.\d+)*\s*[:.\-–—]?\s*)', '', s, flags=re.IGNORECASE)
    return s.lower()

def _find_headings(md: str) -> List[Tuple[int, int, int, str]]:
    """
    Return a list of (start_idx, end_idx, level, title) for each heading in md.
    """
    out = []
    for m in _HEADING_RE.finditer(md):
        hashes = m.group(1)
        title = m.group(2)
        out.append((m.start(), m.end(), len(hashes), title))
    return out

def _remove_ranges(md: str, ranges: List[Tuple[int, int]]) -> str:
    if not ranges:
        return md
    ranges = sorted(ranges)  # by start
    parts = []
    last = 0
    for a, b in ranges:
        if a > last:
            parts.append(md[last:a])
        last = max(last, b)
    parts.append(md[last:])
    return "".join(parts)

def filter_markdown_sections(md: str, exclude: List[str]) -> str:
    """
    Remove sections whose heading matches any token in 'exclude'.
    Matching is case-insensitive and robust to numbering like "1 Authors".
    We remove from the matched heading up to (but not including) the next heading
    at the same or higher level.
    """
    if not exclude:
        return md
    tokens = [t.strip().lower() for t in exclude if t.strip()]
    if not tokens:
        return md

    heads = _find_headings(md)
    if not heads:
        return md

    kill_ranges = []
    for i, (s, e, lvl, title) in enumerate(heads):
        norm = _normalize_title(title)
        if any(tok in norm for tok in tokens):
            # end at next heading of level <= lvl or end of doc
            if i + 1 < len(heads):
                # find the next heading with level <= lvl
                end_idx = len(md)
                for j in range(i + 1, len(heads)):
                    s2, e2, lvl2, _ = heads[j]
                    if lvl2 <= lvl:
                        end_idx = s2
                        break
            else:
                end_idx = len(md)
            kill_ranges.append((s, end_idx))

    return _remove_ranges(md, kill_ranges)

def strip_top_authors_block(md: str) -> str:
    """
    Remove an initial 'author block' above the first major heading (e.g., Abstract).
    Heuristics: drop the top lines before the first heading if they contain emails,
    affiliations, ORCID, or look like a dense author list.
    """
    m = _HEADING_RE.search(md)
    if not m:
        return md
    top = md[:m.start()]
    rest = md[m.start():]

    lines = [ln for ln in top.splitlines()]
    keep = []
    for ln in lines:
        ln_stripped = ln.strip()
        # Heuristics for author block
        has_email = "@" in ln_stripped
        has_affil_keywords = bool(re.search(r'\b(university|institute|laborator|dept\.?|department|school|college|csiro|cnrs|tsinghua|mit|stanford|ibm|google|microsoft|amazon|meta|alibaba|bytedance|deepmind|ai lab)\b', ln_stripped, re.IGNORECASE))
        looks_author_line = bool(re.search(r'^[A-Z][A-Za-z\-\'\.]+( [A-Z][A-Za-z\-\'\.]+){0,3}(,| and )', ln_stripped))
        has_orcid = 'orcid' in ln_stripped.lower()
        if not (has_email or has_affil_keywords or looks_author_line or has_orcid):
            keep.append(ln)
    cleaned_top = "\n".join(keep).strip()
    if cleaned_top:
        cleaned_top += "\n\n"
    return cleaned_top + rest

# ----------------------- /NEW -----------------------

def pdf_to_markdown(pdf_path, md_outdir="mds", exclude_sections=None, strip_top_authors=False):
    os.makedirs(md_outdir, exist_ok=True)
    converter = DocumentConverter()
    result = converter.convert(pdf_path)  # result is a DoclingResult object
    md_text = result.document.export_to_markdown()

    # Apply optional filters
    if strip_top_authors:
        md_text = strip_top_authors_block(md_text)
    if exclude_sections:
        md_text = filter_markdown_sections(md_text, exclude_sections)

    md_path = os.path.join(md_outdir, os.path.basename(pdf_path).replace(".pdf", ".md"))
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)
    return md_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cats", nargs="*", default=[], help="arXiv subject categories (e.g., cs.AI cs.CL stat.ML)")
    ap.add_argument("--keywords", type=str, default="", help="Keyword query (you can use AND/OR and quotes)")
    ap.add_argument("--start-date", type=str, default="", help="YYYY-MM-DD or YYYY-MM")
    ap.add_argument("--end-date", type=str, default="", help="YYYY-MM-DD or YYYY-MM")
    ap.add_argument("--limit", type=int, default=200, help="Max total results to fetch")
    ap.add_argument("--per-page", type=int, default=100, help="Items per API page (<=300 recommended)")
    ap.add_argument("--sleep", type=float, default=3.0, help="Seconds to sleep between requests/downloads")
    ap.add_argument("--outdir", type=str, default="pdfs", help="Where to save PDFs")
    ap.add_argument("--csv", type=str, default="arxiv_results.csv", help="Where to save metadata CSV")
    ap.add_argument("--no-download", action="store_true", help="Do not download PDFs; just save CSV")

    # ----------------------- NEW CLI args -----------------------
    ap.add_argument(
        "--exclude",
        type=str,
        default="",
        help="Comma-separated section titles to remove from Markdown (e.g., 'Authors, Acknowledgments, References')."
    )
    ap.add_argument(
        "--strip-top-authors",
        action="store_true",
        help="Strip the pre-Abstract author block at the very top of the markdown."
    )
    # ----------------------- /NEW -----------------------

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

    if not args.no_download:
        # Parse exclude list once
        exclude_sections = [s.strip() for s in args.exclude.split(",")] if args.exclude else []
        for i, row in enumerate(all_rows, 1):
            try:
                pdf_path = download_pdf(row, outdir=args.outdir, sleep_sec=args.sleep)
                print(f"[pdf {i}/{len(all_rows)}] {row['id']} -> {pdf_path}")

                md_path = pdf_to_markdown(
                    pdf_path,
                    md_outdir="mds",
                    exclude_sections=exclude_sections,
                    strip_top_authors=args.strip_top_authors,
                )
                print(f"[md {i}/{len(all_rows)}] {row['id']} -> {md_path}")

            except Exception as ex:
                print(f"[warn] failed {row['id']}: {ex}")


if __name__ == "__main__":
    main()
