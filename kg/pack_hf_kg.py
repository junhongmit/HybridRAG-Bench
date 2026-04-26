#!/usr/bin/env python3
"""
Export Neo4j KGs to Hugging Face-friendly portable files.

Per database output:
- nodes.parquet
- edges.parquet
- node_properties.parquet (long format)
- edge_properties.parquet (long format)
- schema.json
- constraints.cypher
- indexes.cypher

Example:
  python HybridRAG-Bench/kg/pack_hf_kg.py \
      --uri bolt://localhost:7687 --user neo4j --password password \
      --databases arxiv.ai arxiv.qm arxiv.cy \
      --out-dir HybridRAG-Bench/release/hf_kg
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


@dataclass
class Neo4jAuth:
    uri: str
    user: str
    password: str


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Export Neo4j KG to HF-ready parquet files")
    ap.add_argument("--uri", default=os.getenv("NEO4J_URI", "bolt://localhost:7687"), help="Neo4j bolt URI")
    ap.add_argument("--user", default=os.getenv("NEO4J_USER", "neo4j"), help="Neo4j user")
    ap.add_argument("--password", default=os.getenv("NEO4J_PASSWORD", "password"), help="Neo4j password")
    ap.add_argument(
        "--databases",
        nargs="*",
        default=["arxiv.ai", "arxiv.qm", "arxiv.cy"],
        help="Database names to export",
    )
    ap.add_argument("--batch-size", type=int, default=50000, help="Rows per batch query")
    ap.add_argument("--out-dir", default="HybridRAG-Bench/release/hf_kg", help="Output directory")
    ap.add_argument(
        "--drop-internal-labels",
        action="store_true",
        help="Exclude labels that start with '_' from primary_label selection",
    )
    return ap.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def json_default(obj: Any) -> Any:
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    return str(obj)


def props_to_json(props: Dict[str, Any]) -> str:
    return json.dumps(props or {}, ensure_ascii=False, sort_keys=True, default=json_default)


def normalize_scalar(v: Any) -> Any:
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    if isinstance(v, (list, tuple, dict)):
        return json.dumps(v, ensure_ascii=False, sort_keys=True, default=json_default)
    return v


def normalize_prop_value_text(v: Any) -> str:
    """
    Force property values into a single parquet-compatible text column.
    This avoids mixed object dtype issues in pyarrow (int/str/list/dict in one column).
    """
    if v is None:
        return ""
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    if isinstance(v, (list, tuple, dict)):
        return json.dumps(v, ensure_ascii=False, sort_keys=True, default=json_default)
    return str(v)


def chunk_offsets(total: int, batch_size: int) -> Iterable[int]:
    for offset in range(0, total, batch_size):
        yield offset


def run_query(session, query: str, params: Optional[dict] = None) -> List[dict]:
    result = session.run(query, params or {})
    return [dict(r) for r in result]


def safe_name(db_name: str) -> str:
    return db_name.replace(".", "_")


def first_non_internal_label(labels: List[str], drop_internal: bool) -> str:
    if not labels:
        return ""
    if not drop_internal:
        return labels[0]
    for lb in labels:
        if not lb.startswith("_"):
            return lb
    return labels[0]


def fetch_count(session, query: str) -> int:
    rows = run_query(session, query)
    if not rows:
        return 0
    return int(rows[0].get("cnt", 0) or 0)


def export_nodes(session, batch_size: int, drop_internal_labels: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    total = fetch_count(session, "MATCH (n) RETURN count(n) AS cnt")
    nodes_rows: List[dict] = []
    node_prop_rows: List[dict] = []

    q = (
        "MATCH (n) "
        "RETURN id(n) AS node_id, elementId(n) AS element_id, labels(n) AS labels, properties(n) AS props "
        "SKIP $offset LIMIT $limit"
    )

    for offset in chunk_offsets(total, batch_size):
        rows = run_query(session, q, {"offset": offset, "limit": batch_size})
        for r in rows:
            props = r.get("props") or {}
            labels = r.get("labels") or []
            primary_label = first_non_internal_label(labels, drop_internal_labels)
            nodes_rows.append(
                {
                    "node_id": int(r["node_id"]),
                    "element_id": str(r.get("element_id", "")),
                    "labels": labels,
                    "primary_label": primary_label,
                    "display_name": normalize_scalar(props.get("name", "")),
                    "properties_json": props_to_json(props),
                }
            )

            for k, v in props.items():
                node_prop_rows.append(
                    {
                        "node_id": int(r["node_id"]),
                        "key": str(k),
                        "value": normalize_prop_value_text(v),
                    }
                )

    return pd.DataFrame(nodes_rows), pd.DataFrame(node_prop_rows)


def export_edges(session, batch_size: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    total = fetch_count(session, "MATCH ()-[r]->() RETURN count(r) AS cnt")
    edge_rows: List[dict] = []
    edge_prop_rows: List[dict] = []

    q = (
        "MATCH ()-[r]->() "
        "RETURN id(r) AS edge_id, elementId(r) AS element_id, "
        "id(startNode(r)) AS src_id, id(endNode(r)) AS dst_id, "
        "type(r) AS rel_type, properties(r) AS props "
        "SKIP $offset LIMIT $limit"
    )

    for offset in chunk_offsets(total, batch_size):
        rows = run_query(session, q, {"offset": offset, "limit": batch_size})
        for r in rows:
            props = r.get("props") or {}
            edge_rows.append(
                {
                    "edge_id": int(r["edge_id"]),
                    "element_id": str(r.get("element_id", "")),
                    "src_id": int(r["src_id"]),
                    "dst_id": int(r["dst_id"]),
                    "rel_type": str(r.get("rel_type", "")),
                    "properties_json": props_to_json(props),
                }
            )

            for k, v in props.items():
                edge_prop_rows.append(
                    {
                        "edge_id": int(r["edge_id"]),
                        "key": str(k),
                        "value": normalize_prop_value_text(v),
                    }
                )

    return pd.DataFrame(edge_rows), pd.DataFrame(edge_prop_rows)


def _escape_label_or_type(name: str) -> str:
    return "`" + name.replace("`", "``") + "`"


def collect_schema(session, db_name: str) -> dict:
    labels = [r["label"] for r in run_query(session, "CALL db.labels() YIELD label RETURN label ORDER BY label")]
    rel_types = [
        r["relationshipType"]
        for r in run_query(session, "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType ORDER BY relationshipType")
    ]

    triples = run_query(
        session,
        """
        MATCH (a)-[r]->(b)
        RETURN DISTINCT labels(a) AS source_labels, type(r) AS rel_type, labels(b) AS target_labels
        ORDER BY rel_type
        """,
    )

    node_prop_keys: Dict[str, List[str]] = {}
    for lb in labels:
        q = f"MATCH (n:{_escape_label_or_type(lb)}) UNWIND keys(n) AS k RETURN DISTINCT k ORDER BY k"
        node_prop_keys[lb] = [r["k"] for r in run_query(session, q)]

    rel_prop_keys: Dict[str, List[str]] = {}
    for rt in rel_types:
        q = f"MATCH ()-[r:{_escape_label_or_type(rt)}]->() UNWIND keys(r) AS k RETURN DISTINCT k ORDER BY k"
        rel_prop_keys[rt] = [r["k"] for r in run_query(session, q)]

    return {
        "database": db_name,
        "labels": labels,
        "relationship_types": rel_types,
        "relation_schema": triples,
        "node_property_keys": node_prop_keys,
        "relationship_property_keys": rel_prop_keys,
    }


def collect_create_statements(session, what: str) -> List[str]:
    if what == "constraints":
        q = "SHOW CONSTRAINTS YIELD createStatement RETURN createStatement ORDER BY createStatement"
    elif what == "indexes":
        q = "SHOW INDEXES YIELD createStatement RETURN createStatement ORDER BY createStatement"
    else:
        return []

    try:
        return [r["createStatement"] for r in run_query(session, q) if r.get("createStatement")]
    except Exception:
        return []


def write_lines(path: Path, lines: List[str], header: str) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        f.write(header.rstrip() + "\n")
        for line in lines:
            f.write(line.rstrip() + ";\n")


def export_database(driver, db_name: str, out_dir: Path, batch_size: int, drop_internal_labels: bool) -> None:
    print(f"[export] database={db_name}")
    db_out = out_dir / safe_name(db_name)
    ensure_dir(db_out)

    with driver.session(database=db_name) as session:
        nodes_df, node_props_df = export_nodes(session, batch_size=batch_size, drop_internal_labels=drop_internal_labels)
        edges_df, edge_props_df = export_edges(session, batch_size=batch_size)

        nodes_df.to_parquet(db_out / "nodes.parquet", index=False)
        edges_df.to_parquet(db_out / "edges.parquet", index=False)
        node_props_df.to_parquet(db_out / "node_properties.parquet", index=False)
        edge_props_df.to_parquet(db_out / "edge_properties.parquet", index=False)

        schema = collect_schema(session, db_name=db_name)
        with (db_out / "schema.json").open("w", encoding="utf-8") as f:
            json.dump(schema, f, ensure_ascii=False, indent=2)

        constraints = collect_create_statements(session, "constraints")
        indexes = collect_create_statements(session, "indexes")

        write_lines(
            db_out / "constraints.cypher",
            constraints,
            header="// Recreate constraints (exported via SHOW CONSTRAINTS)",
        )
        write_lines(
            db_out / "indexes.cypher",
            indexes,
            header="// Recreate indexes (exported via SHOW INDEXES)",
        )

    print(
        f"[done] {db_name}: nodes={len(nodes_df)} edges={len(edges_df)} "
        f"node_props={len(node_props_df)} edge_props={len(edge_props_df)}"
    )


def main() -> None:
    args = parse_args()

    try:
        from neo4j import GraphDatabase
    except ModuleNotFoundError as ex:
        raise ModuleNotFoundError(
            "Missing dependency 'neo4j'. Install with: pip install neo4j"
        ) from ex

    auth = Neo4jAuth(uri=args.uri, user=args.user, password=args.password)
    out_dir = Path(args.out_dir).expanduser().resolve()
    ensure_dir(out_dir)

    driver = GraphDatabase.driver(auth.uri, auth=(auth.user, auth.password))
    try:
        for db in args.databases:
            export_database(
                driver=driver,
                db_name=db,
                out_dir=out_dir,
                batch_size=args.batch_size,
                drop_internal_labels=args.drop_internal_labels,
            )
    finally:
        driver.close()


if __name__ == "__main__":
    main()
