import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from kg.kg_rep import KGRelation, relation_to_dict, relation_to_text


TYPE_PREFIX_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_ ]{0,40}:\s*")


def normalize_entity_name(value: str) -> str:
    text = str(value or "").strip()
    text = TYPE_PREFIX_PATTERN.sub("", text)
    return " ".join(text.split())


def relation_signature(relation: KGRelation) -> Tuple[str, str, str]:
    head = normalize_entity_name(relation.source.name if relation.source else "")
    tail = normalize_entity_name(relation.target.name if relation.target else "")
    return (head, relation.name or "", tail)


def reasoning_path_signature(reasoning_path: Iterable[Dict[str, Any]]) -> Tuple[Tuple[str, str, str], ...]:
    signature: List[Tuple[str, str, str]] = []
    for item in reasoning_path or []:
        signature.append(
            (
                normalize_entity_name(item.get("head", "")),
                str(item.get("relation", "")),
                normalize_entity_name(item.get("tail", "")),
            )
        )
    return tuple(signature)


def verbalize_path(relations: Sequence[KGRelation], include_paragraphs: bool = True) -> str:
    return "\n".join(
        relation_to_text(relation, include_par=include_paragraphs)
        for relation in relations
    )


def collect_text_paragraphs(relations: Sequence[KGRelation]) -> List[str]:
    paragraphs: List[str] = []
    seen = set()
    for relation in relations:
        candidates = [
            relation.source.paragraph if relation.source else None,
            relation.paragraph,
            relation.target.paragraph if relation.target else None,
        ]
        for paragraph in candidates:
            if not paragraph:
                continue
            normalized = paragraph.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            paragraphs.append(normalized)
    return paragraphs


def build_graph_path(relations: Sequence[KGRelation]) -> Dict[str, Any]:
    return {
        "verbalized_path": verbalize_path(relations, include_paragraphs=True),
        "triples": [
            {
                "head": relation.source.name if relation.source else "",
                "relation": relation.name or "",
                "tail": relation.target.name if relation.target else "",
            }
            for relation in relations
        ],
        "relations": [relation_to_dict(relation) for relation in relations],
    }


def build_groundtruth_entry(
    question_id: int,
    question: str,
    answer: str,
    question_type: str,
    relation_paths: Optional[Sequence[Sequence[KGRelation]]] = None,
    text_paragraphs: Optional[Sequence[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    relation_paths = relation_paths or []
    paragraphs = list(text_paragraphs or [])
    if not paragraphs:
        for path in relation_paths:
            for paragraph in collect_text_paragraphs(path):
                if paragraph not in paragraphs:
                    paragraphs.append(paragraph)

    return {
        "id": question_id,
        "type": question_type,
        "question": question,
        "answer": answer,
        "evidence": {
            "text": {
                "paragraphs": paragraphs,
            },
            "graph": {
                "paths": [build_graph_path(path) for path in relation_paths],
            },
        },
        "metadata": metadata or {},
    }


def write_questions_and_groundtruth(
    output_dir: str,
    question_filename: str,
    questions: Sequence[Dict[str, Any]],
    groundtruth: Sequence[Dict[str, Any]],
) -> Tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    question_path = os.path.join(output_dir, question_filename)
    groundtruth_filename = question_filename.replace("questions", "groundtruth", 1)
    groundtruth_path = os.path.join(output_dir, groundtruth_filename)

    with open(question_path, "w", encoding="utf-8") as f:
        json.dump(list(questions), f, indent=4, ensure_ascii=False)
    with open(groundtruth_path, "w", encoding="utf-8") as f:
        json.dump(list(groundtruth), f, indent=4, ensure_ascii=False)

    return question_path, groundtruth_path


def load_questions_and_groundtruth(
    output_dir: str,
    question_filename: str,
    logger=None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    question_path = os.path.join(output_dir, question_filename)
    groundtruth_filename = question_filename.replace("questions", "groundtruth", 1)
    groundtruth_path = os.path.join(output_dir, groundtruth_filename)

    questions: List[Dict[str, Any]] = []
    groundtruth: List[Dict[str, Any]] = []

    if os.path.exists(question_path):
        try:
            with open(question_path, "r", encoding="utf-8") as f:
                existing_questions = json.load(f)
            if isinstance(existing_questions, list):
                questions = existing_questions
                if logger:
                    logger.info(f"Resumed from existing output with {len(questions)} items.")
        except Exception:
            if logger:
                logger.warning("Failed to load existing output file; starting fresh.")

    if os.path.exists(groundtruth_path):
        try:
            with open(groundtruth_path, "r", encoding="utf-8") as f:
                existing_groundtruth = json.load(f)
            if isinstance(existing_groundtruth, list):
                groundtruth = existing_groundtruth
                if logger:
                    logger.info(f"Resumed from existing groundtruth with {len(groundtruth)} items.")
        except Exception:
            if logger:
                logger.warning("Failed to load existing groundtruth file; starting fresh.")

    return questions, groundtruth


def build_row_path_index(
    rows: Sequence[Dict[str, Any]],
    instantiate_from_path_rows,
) -> Dict[Tuple[Tuple[str, str, str], ...], List[KGRelation]]:
    index: Dict[Tuple[Tuple[str, str, str], ...], List[KGRelation]] = {}
    for row in rows:
        _, relations = instantiate_from_path_rows([row])
        if not relations:
            continue
        index[tuple(relation_signature(relation) for relation in relations)] = relations
    return index


def find_relation_path(
    path_index: Dict[Tuple[Tuple[str, str, str], ...], List[KGRelation]],
    reasoning_path: Optional[Iterable[Dict[str, Any]]],
) -> List[KGRelation]:
    signature = reasoning_path_signature(reasoning_path or [])
    if signature in path_index:
        return path_index[signature]

    if len(signature) == 1:
        target = signature[0]
        for candidate_signature, relations in path_index.items():
            if target in candidate_signature:
                return relations

    return []
