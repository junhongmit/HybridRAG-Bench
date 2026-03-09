#!/usr/bin/env python3
"""
Import packaged KG parquet files into a Neo4j database.

Expected per-database folder contents:
- nodes.parquet
- edges.parquet
- constraints.cypher (optional)
- indexes.cypher (optional)
- schema.json (optional, used for db name mapping)

Notes:
- This importer adds helper properties for reproducible mapping/comparison:
  - node._orig_node_id
  - relationship._orig_edge_id
- Use an empty target DB (or pass --clear-db).
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Import HF KG parquet into Neo4j")
    ap.add_argument("--uri", default=os.getenv("NEO4J_URI", "bolt://localhost:7687"), help="Neo4j bolt URI")
    ap.add_argument("--user", default=os.getenv("NEO4J_USER", "neo4j"), help="Neo4j user")
    ap.add_argument("--password", default=os.getenv("NEO4J_PASSWORD", "password"), help="Neo4j password")
    ap.add_argument("--kg-root", required=True, help="Root folder containing per-db exported KG folders")
    ap.add_argument(
        "--databases",
        nargs="*",
        default=[],
        help="Database names to import. If omitted, infer from schema.json or folder names",
    )
    ap.add_argument("--batch-size", type=int, default=5000, help="UNWIND batch size")
    ap.add_argument("--clear-db", action="store_true", help="Delete all existing nodes/relationships before import")
    ap.add_argument("--apply-schema", action="store_true", help="Apply constraints.cypher and indexes.cypher if present")
    ap.add_argument(
        "--no-vector-indexes",
        action="store_true",
        help="Do not create vector indexes used by retrieval",
    )
    ap.add_argument(
        "--vector-dimensions",
        type=int,
        default=768,
        help="Vector index dimensions",
    )
    ap.add_argument(
        "--vector-similarity",
        default="cosine",
        choices=["cosine", "euclidean"],
        help="Vector index similarity function",
    )
    return ap.parse_args()


def chunked(items: List[dict], size: int) -> Iterable[List[dict]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def parse_json_cell(v: Any) -> Dict[str, Any]:
    if v is None:
        return {}
    if isinstance(v, dict):
        return v
    if isinstance(v, float) and pd.isna(v):
        return {}
    s = str(v).strip()
    if not s:
        return {}
    try:
        obj = json.loads(s)
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def parse_labels_cell(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) for x in v]
    if isinstance(v, tuple):
        return [str(x) for x in v]
    # Handle ndarray / Arrow array / other iterable containers without forcing string parsing.
    if hasattr(v, "__iter__") and not isinstance(v, (str, bytes, dict)):
        try:
            return [str(x) for x in list(v)]
        except Exception:
            pass
    if isinstance(v, float) and pd.isna(v):
        return []
    s = str(v).strip()
    if not s:
        return []
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return [str(x) for x in obj]
    except Exception:
        pass
    # Python literal list forms.
    try:
        lit = ast.literal_eval(s)
        if isinstance(lit, (list, tuple)):
            return [str(x) for x in lit]
    except Exception:
        pass
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        if not inner:
            return []
        # np.ndarray-like repr uses whitespace separator without commas:
        # "['_Embeddable' 'Concept']"
        quoted = re.findall(r"""['"]([^'"]+)['"]""", inner)
        if quoted:
            return [str(x) for x in quoted]
        return [x.strip().strip("'\"") for x in inner.split(",") if x.strip()]
    return [s]


def esc_ident(name: str) -> str:
    return "`" + name.replace("`", "``") + "`"


def load_statements(path: Path) -> List[str]:
    if not path.exists():
        return []
    raw = path.read_text(encoding="utf-8", errors="ignore")
    lines: List[str] = []
    for ln in raw.splitlines():
        s = ln.strip()
        if not s or s.startswith("//"):
            continue
        lines.append(ln)
    content = "\n".join(lines)
    parts = [p.strip() for p in content.split(";")]
    return [p for p in parts if p]


def infer_db_folder_map(kg_root: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for child in sorted(kg_root.iterdir()):
        if not child.is_dir():
            continue
        schema_path = child / "schema.json"
        db_name = ""
        if schema_path.exists():
            try:
                payload = json.loads(schema_path.read_text(encoding="utf-8"))
                db_name = str(payload.get("database", "")).strip()
            except Exception:
                db_name = ""
        if not db_name:
            db_name = child.name.replace("_", ".")
        out[db_name] = child
    return out


def prepare_node_rows(nodes_df: pd.DataFrame) -> Dict[tuple[str, ...], List[dict]]:
    groups: Dict[tuple[str, ...], List[dict]] = {}

    for _, row in nodes_df.iterrows():
        node_id = int(row["node_id"])
        labels = tuple(sorted(parse_labels_cell(row.get("labels"))))
        props = parse_json_cell(row.get("properties_json"))
        props["_orig_node_id"] = node_id

        groups.setdefault(labels, []).append(
            {
                "node_id": node_id,
                "props": props,
            }
        )

    return groups


def prepare_edge_rows(edges_df: pd.DataFrame) -> Dict[str, List[dict]]:
    groups: Dict[str, List[dict]] = {}
    for _, row in edges_df.iterrows():
        edge_id = int(row["edge_id"])
        rel_type = str(row.get("rel_type", "")).strip()
        if not rel_type:
            continue
        props = parse_json_cell(row.get("properties_json"))
        props["_orig_edge_id"] = edge_id

        groups.setdefault(rel_type, []).append(
            {
                "edge_id": edge_id,
                "src_id": int(row["src_id"]),
                "dst_id": int(row["dst_id"]),
                "props": props,
            }
        )
    return groups


def create_vector_indexes(session, dimensions: int, similarity: str) -> None:
    stmts = [
        (
            "entityVector",
            f"""CREATE VECTOR INDEX entityVector IF NOT EXISTS
FOR (n:_Embeddable)
ON n._embedding
OPTIONS {{indexConfig: {{
`vector.dimensions`: {dimensions},
`vector.similarity_function`: '{similarity}'
}}}}""",
        ),
        (
            "entitySchemaVector",
            f"""CREATE VECTOR INDEX entitySchemaVector IF NOT EXISTS
FOR (s:_EntitySchema)
ON s._embedding
OPTIONS {{indexConfig: {{
`vector.dimensions`: {dimensions},
`vector.similarity_function`: '{similarity}'
}}}}""",
        ),
        (
            "relationSchemaVector",
            f"""CREATE VECTOR INDEX relationSchemaVector IF NOT EXISTS
FOR (s:_RelationSchema)
ON s._embedding
OPTIONS {{indexConfig: {{
`vector.dimensions`: {dimensions},
`vector.similarity_function`: '{similarity}'
}}}}""",
        ),
    ]

    for name, stmt in stmts:
        try:
            session.run(stmt)
            print(f"[index] ensured vector index: {name}")
        except Exception as ex:
            print(f"[warn] failed creating vector index {name}: {ex}")


def import_one_database(
    driver,
    db_name: str,
    db_dir: Path,
    batch_size: int,
    clear_db: bool,
    apply_schema: bool,
    no_vector_indexes: bool,
    vector_dimensions: int,
    vector_similarity: str,
) -> None:
    nodes_path = db_dir / "nodes.parquet"
    edges_path = db_dir / "edges.parquet"
    if not nodes_path.exists() or not edges_path.exists():
        print(f"[warn] skip {db_name}: missing nodes.parquet or edges.parquet in {db_dir}")
        return

    nodes_df = pd.read_parquet(nodes_path)
    edges_df = pd.read_parquet(edges_path)
    node_groups = prepare_node_rows(nodes_df)
    edge_groups = prepare_edge_rows(edges_df)

    with driver.session(database=db_name) as session:
        if clear_db:
            session.run("MATCH (n) DETACH DELETE n")
            print(f"[clear] {db_name}: existing graph deleted")

        for labels, rows in node_groups.items():
            label_clause = "".join([":" + esc_ident(lb) for lb in labels])
            q = (
                "UNWIND $rows AS row "
                f"MERGE (n{label_clause} {{_orig_node_id: row.node_id}}) "
                "SET n += row.props"
            )
            for batch in chunked(rows, batch_size):
                session.run(q, {"rows": batch})

        for rel_type, rows in edge_groups.items():
            rt = esc_ident(rel_type)
            q = (
                "UNWIND $rows AS row "
                "MATCH (s {_orig_node_id: row.src_id}) "
                "MATCH (t {_orig_node_id: row.dst_id}) "
                f"MERGE (s)-[r:{rt} {{_orig_edge_id: row.edge_id}}]->(t) "
                "SET r += row.props"
            )
            for batch in chunked(rows, batch_size):
                session.run(q, {"rows": batch})

        if apply_schema:
            constraints = load_statements(db_dir / "constraints.cypher")
            indexes = load_statements(db_dir / "indexes.cypher")
            for stmt in constraints + indexes:
                try:
                    session.run(stmt)
                except Exception as ex:
                    print(f"[warn] schema statement failed on {db_name}: {ex}")

        if not no_vector_indexes:
            create_vector_indexes(
                session,
                dimensions=vector_dimensions,
                similarity=vector_similarity,
            )

        node_cnt = session.run("MATCH (n) RETURN count(n) AS cnt").single()["cnt"]
        edge_cnt = session.run("MATCH ()-[r]->() RETURN count(r) AS cnt").single()["cnt"]

    print(
        f"[done] {db_name}: imported nodes={len(nodes_df)} edges={len(edges_df)} "
        f"db_nodes={node_cnt} db_edges={edge_cnt}"
    )


def main() -> None:
    args = parse_args()
    try:
        from neo4j import GraphDatabase
    except ModuleNotFoundError as ex:
        raise ModuleNotFoundError("Missing dependency 'neo4j'. Install with: pip install neo4j") from ex

    kg_root = Path(args.kg_root).expanduser().resolve()
    if not kg_root.exists():
        raise FileNotFoundError(f"kg root not found: {kg_root}")

    db_map = infer_db_folder_map(kg_root)
    if not db_map:
        raise RuntimeError(f"No database folders found under {kg_root}")

    target_dbs = args.databases if args.databases else sorted(db_map.keys())

    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    try:
        for db in target_dbs:
            db_dir = db_map.get(db)
            if db_dir is None:
                print(f"[warn] no folder found for database '{db}', skipping")
                continue
            import_one_database(
                driver=driver,
                db_name=db,
                db_dir=db_dir,
                batch_size=args.batch_size,
                clear_db=args.clear_db,
                apply_schema=args.apply_schema,
                no_vector_indexes=args.no_vector_indexes,
                vector_dimensions=args.vector_dimensions,
                vector_similarity=args.vector_similarity,
            )
    finally:
        driver.close()


if __name__ == "__main__":
    main()
