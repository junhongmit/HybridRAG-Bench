import os
import random
from typing import Any, Dict, Optional

from kg.kg_driver import *
from kg.kg_rep import *
from utils.utils import *
from utils.logger import *

# ======================== Hyperparameters ======================== #
_OUTPUT_FILENAME = "questions_counterfactual_CWQstyle.json"
_BATCHES = 50
_BATCH_LIMIT = 10
_MAX_ENTITY_CANDIDATES = 5
_MAX_RELATION_CANDIDATES = 5
_RNG = random.Random(42)
# ======================== Hyperparameters ======================== #


def instantiate_from_path_rows(rows: List[Dict[str, Any]]) -> Tuple[List[KGEntity], List[KGRelation]]:
    """
    Convert the Cypher-returned rows into KGEntity and KGRelation objects.
    Same as original script's helper.
    """
    entities_by_id: Dict[str, KGEntity] = {}
    relations: List[KGRelation] = []

    for row in rows:
        nodes = row["nodes"]
        rels = row["rels"]

        for n in nodes:
            nid = n["id"]
            props = n.get("properties") or {}
            if nid not in entities_by_id:
                ent = KGEntity(
                    id=nid,
                    type=kg_driver.get_label(n.get("labels", [])),
                    name=n.get("name"),
                    description=props.get(PROP_DESCRIPTION),
                    paragraph=props.get(PROP_PARAGRAPH),
                    created_at=props.get(PROP_CREATED),
                    modified_at=props.get(PROP_MODIFIED),
                    properties=kg_driver.get_properties(props),
                    ref=props.get(PROP_REFERENCE),
                )
                entities_by_id[nid] = ent

        node_ids = [n["id"] for n in nodes]
        for idx, r in enumerate(rels):
            rid = r["id"]
            rtype = r["type"]
            rprops = r.get("properties") or {}
            head = node_ids[idx]
            tail = node_ids[idx + 1]
            relations.append(
                KGRelation(
                    id=rid,
                    name=rtype,
                    source=entities_by_id[head],
                    target=entities_by_id[tail],
                    description=rprops.get(PROP_DESCRIPTION),
                    paragraph=rprops.get(PROP_PARAGRAPH),
                    properties=kg_driver.get_properties(rprops),
                )
            )

    return list(entities_by_id.values()), relations


# ----------------------------
# Helper KG-lookup functions
# ----------------------------
def find_entity_candidates(base_entity: KGEntity, limit: int = _MAX_ENTITY_CANDIDATES) -> List[Dict[str, Any]]:
    """
    Return candidate replacement entities from the KG.
    Strategy:
      1) If base_entity.type (label) is present, prefer other nodes with same label.
      2) Fallback: sample nodes with non-empty name.
    Returns list of dicts with at least 'id' and 'name' (and optionally 'type', 'paragraph').
    """
    candidates = []
    try:
        label = base_entity.type or None
        params = {"name": base_entity.name, "limit": limit}
        if label:
            # label may contain spaces; wrap in backticks in Cypher
            cypher = f"""
                MATCH (c:{label})
                WHERE c.name IS NOT NULL AND c.name <> $name AND c._paragraph IS NOT NULL
                RETURN elementId(c) AS id, c.name AS name, labels(c) AS labels, c._paragraph AS paragraph
                LIMIT $limit
            """
        else:
            cypher = """
                MATCH (c)
                WHERE c.name IS NOT NULL AND c.name <> $name AND c._paragraph IS NOT NULL
                RETURN elementId(c) AS id, c.name AS name, labels(c) AS labels, c._paragraph AS paragraph
                LIMIT $limit
            """
        rows = kg_driver.run_query(cypher, params)
        for r in rows:
            # depending on run_query return shape: sometimes rows are dictionaries
            if isinstance(r, dict):
                candidates.append({"id": r.get("id"), "name": r.get("name"), "labels": r.get("labels"), "paragraph": r.get("paragraph")})
            else:
                # fallback: r may be a list
                candidates.append({"id": r[0], "name": r[1]})
    except Exception as e:
        logger.warn(f"Entity candidate lookup failed for {base_entity.name}: {e}")
    # Shuffle to add variety
    RNG.shuffle(candidates)
    return candidates[:limit]


def find_relation_candidates(head_label: Optional[str], tail_label: Optional[str], limit: int = _MAX_RELATION_CANDIDATES) -> List[str]:
    """
    Find relation types that appear between nodes with given labels.
    If labels are None, then return some frequent relation types.
    """
    try:
        params = {"limit": limit}
        if head_label and tail_label:
            cypher = f"""
                MATCH (a:{head_label})-[r]->(b:{tail_label})
                RETURN distinct type(r) AS rel
                LIMIT $limit
            """
        elif head_label:
            cypher = f"""
                MATCH (a:{head_label})-[r]->(b)
                RETURN distinct type(r) AS rel
                LIMIT $limit
            """
        else:
            cypher = """
                MATCH ()-[r]->()
                RETURN distinct type(r) AS rel
                LIMIT $limit
            """
        rows = kg_driver.run_query(cypher, params)
        rels = []
        for r in rows:
            if isinstance(r, dict):
                rels.append(r.get("rel"))
            else:
                rels.append(r[0])
        RNG.shuffle(rels)
        return rels[:limit]
    except Exception as e:
        logger.warn(f"Relation candidate lookup failed for {head_label}->{tail_label}: {e}")
        return []


# ----------------------------
# LLM prompts
# ----------------------------
SYSTEM_PROMPT_ORIGINAL = """\
You will produce a single, high-quality OPEN-ENDED question grounded ONLY in the provided KG path evidence.

The path evidence is provided as a short structured context containing nodes, relation labels, and short 'paragraph' evidence. Generate ONE exploratory question that:
- is not a simple factual lookup,
- encourages explanation, implication, or causal reasoning,
- uses explicit entity names (do not use "this paper" or pronouns),
- is faithful to the evidence (do not hallucinate),
- is concise (one or two sentences).

Output ONLY JSON array with one object:
[{"question": "<original question>"}]
"""

SYSTEM_PROMPT_PERTURB = """\
You are given:
1) an ORIGINAL natural-language question (open-ended) that is grounded in a KG path.
2) the ORIGINAL path (list of head-relation-tail items).
3) a PROPOSED perturbation: either (A) replace an entity with another entity, or (B) replace a relation with another relation.

Produce TWO items in JSON:
1) a COUNTERFACTUAL question that explicitly frames the proposed perturbation as a hypothetical (e.g., "If <perturbation>, how would ...?"). The question must be grounded in the path evidence and must not assert the hypothetical as fact.
2) a CONTRASTIVE question (a distractor) that is minimally different from the original but changes the key element (the distractor should be realistic and could lead to a different answer).

Requirements:
- Keep both questions short and well-formed.
- Use explicit entity and relation names.
- Do NOT include any answers or extra commentary.
- Output ONLY JSON (array of one object):
[
  {
    "counterfactual_question": "...",
    "contrastive_question": "...",
    "perturbation_type": "entity" or "relation",
    "perturbed_element": "<name or relation>"
  }
]

"""

# ----------------------------
# Utility to convert relation objects into simple path lists
# ----------------------------
def relation_objs_to_path(relations: List[KGRelation]) -> List[Dict[str, str]]:
    """
    Convert a list of KGRelation objects along a path into list of {head, relation, tail}.
    """
    path = []
    for rel in relations:
        head = rel.source.name if rel.source else ""
        tail = rel.target.name if rel.target else ""
        path.append({"head": head, "relation": rel.name, "tail": tail})
    return path


async def generate(path: str, config: Optional[Dict[str, Any]] = {},
                   logger: Optional[BaseProgressLogger] = None):
    """Generate CWQ-style counterfactual QA pairs."""
    logger = logger or DefaultProgressLogger()

    global OUTPUT_FILENAME, BATCHES, BATCH_LIMIT, MAX_ENTITY_CANDIDATES, MAX_RELATION_CANDIDATES, RNG
    OUTPUT_FILENAME = config.get("output_filename", _OUTPUT_FILENAME)
    BATCHES = config.get("batches", _BATCHES)
    BATCH_LIMIT = config.get("batch_limit", _BATCH_LIMIT)
    MAX_ENTITY_CANDIDATES = config.get("max_entity_candidates", _MAX_ENTITY_CANDIDATES)
    MAX_RELATION_CANDIDATES = config.get("max_relation_candidates", _MAX_RELATION_CANDIDATES)
    RNG = config.get("rng", _RNG)

    output_path = os.path.join(path, OUTPUT_FILENAME)
    output: List[Dict[str, Any]] = []

    for batch_idx in range(BATCHES):
        logger.info(f"Batch {batch_idx+1}/{BATCHES} — querying KG for paths...")

        # Default: use 2-hop paths to provide rich context (a)-[r1]->(b)-[r2]->(c)
        results = kg_driver.run_query(
            """
            MATCH p=(a)-[r1]->(b)-[r2]->(c)
            WHERE a._paragraph IS NOT NULL AND b._paragraph IS NOT NULL AND c._paragraph IS NOT NULL
              AND r1._paragraph IS NOT NULL AND r2._paragraph IS NOT NULL
              AND a <> b AND b <> c AND a <> c
            WITH p, rand() AS rorder
            RETURN
               p,
               [n IN nodes(p) | {id: elementId(n), labels: labels(n), name: n.name, properties: apoc.map.removeKey(properties(n), "_embedding"), paragraph: n._paragraph}] AS nodes,
               [r IN relationships(p) | {id: elementId(r), type: type(r), properties: properties(r), paragraph: r._paragraph}] AS rels
            ORDER BY rorder
            LIMIT $limit;
            """,
            {"limit": BATCH_LIMIT}
        )

        # instantiate objects
        entities, relations = instantiate_from_path_rows(results)

        # build contexts per path (the run_query returned multiple paths; instantiate_from_path_rows merges entities and relations
        # so we need to rebuild per-row context. For simplicity, reconstruct contexts from results directly:
        for row in results:
            try:
                nodes = row.get("nodes", [])
                rels = row.get("rels", [])
            except Exception:
                logger.warn("Unexpected row format; skipping")
                continue

            # Construct KGRelation-like objects for this path so relation_objs_to_path works
            path_rel_objs: List[KGRelation] = []
            # We'll reuse instantiate_from_path_rows for each single-row mini-path for consistent KGRelation objects.
            single_entities, single_relations = instantiate_from_path_rows([row])
            path_rel_objs = single_relations

            # Create a short context for the LLM using relation_to_text if available; include paragraphs
            context_text = []
            for rel in path_rel_objs:
                try:
                    txt = relation_to_text(rel, include_par=True)
                except Exception:
                    head = rel.source.name if rel.source else ""
                    tail = rel.target.name if rel.target else ""
                    par = (rel.paragraph or "")[:400]
                    txt = f"({head}) -[{rel.name}]-> ({tail})\nEVIDENCE: {par}"
                context_text.append(txt)
            context_blob = "\n\n".join(context_text)

            # 1) Ask the LLM to write an original open-ended question grounded in the path
            # prompts_original = [
            #     {"role": "system", "content": SYSTEM_PROMPT_ORIGINAL},
            #     {"role": "user", "content": f"CONTEXT:\n{context_blob}\n\nProduce one open-ended question."}
            # ]
            # try:
            #     response_orig = await generate_response(prompts_original, max_tokens=512, temperature=0.6, logger=logger)
            #     original_qs = json.loads(response_orig)
            #     if isinstance(original_qs, list) and len(original_qs) > 0:
            #         original_question = original_qs[0].get("question", "").strip()
            #     else:
            #         logger.warn("Original question response not list or empty; skipping this path.")
            #         continue
            # except Exception as e:
            #     logger.error(f"Failed to generate original question: {e}")
            #     continue

            # 1) Ask the LLM to write an original open-ended question grounded in the path
            prompts_original = [
                {"role": "system", "content": SYSTEM_PROMPT_ORIGINAL},
                {"role": "user", "content": f"CONTEXT:\n{context_blob}\n\nProduce one open-ended question."}
            ]
            try:
                response_orig = await generate_response(prompts_original, max_tokens=512, temperature=0.2,
                                                        logger=logger)
                if not response_orig.strip():
                    logger.warn("LLM returned empty response; skipping this path")
                    continue

                try:
                    original_qs = json.loads(response_orig)
                    if isinstance(original_qs, list) and len(original_qs) > 0:
                        original_question = original_qs[0].get("question", "").strip()
                        if not original_question:
                            logger.warn(
                                f"Original question is empty after parsing; skipping. Raw LLM output:\n{response_orig}")
                            continue
                    else:
                        logger.warn(
                            f"Original question response not a non-empty list; skipping. Raw LLM output:\n{response_orig}")
                        continue
                except json.JSONDecodeError:
                    logger.warn(
                        f"Failed to parse JSON from LLM output; skipping this path.\nRAW OUTPUT:\n{response_orig}")
                    continue

            except Exception as e:
                logger.error(f"Failed to generate original question: {e}")
                continue

            # Build original_path
            original_path = relation_objs_to_path(path_rel_objs)

            # 2) Generate candidate perturbations (entity replacements and relation swaps)
            # We'll attempt: replace the middle node (b) first (common in CWQ), and also propose relation swaps on r1 and r2
            middle_rel = None
            if len(path_rel_objs) >= 2:
                # choose middle relation as second relation if available
                middle_rel = path_rel_objs[1]  # r2 (b->c)
            else:
                middle_rel = path_rel_objs[0]

            # entity candidates for middle node (b)
            middle_node = path_rel_objs[0].target if path_rel_objs else None  # node b
            entity_candidates = []
            if middle_node:
                entity_candidates = find_entity_candidates(middle_node, limit=MAX_ENTITY_CANDIDATES)

            # relation candidates for r1 and r2 (relation swaps)
            # use labels if possible
            head_label = path_rel_objs[0].source.type if path_rel_objs and path_rel_objs[0].source else None
            tail_label = path_rel_objs[-1].target.type if path_rel_objs and path_rel_objs[-1].target else None
            relation_candidates = find_relation_candidates(head_label, tail_label, limit=MAX_RELATION_CANDIDATES)

            # We'll construct perturbations: try up to N entity perturbations and up to M relation perturbations
            perturbations_to_try = []

            # Entity perturbations (type "entity"): replace middle node b with each candidate
            for cand in entity_candidates:
                perturbations_to_try.append({
                    "type": "entity",
                    "target_node": "middle",
                    "candidate": cand
                })

            # Relation perturbations (type "relation"): replace r1 or r2 with each candidate relation type
            for cand_rel in relation_candidates:
                perturbations_to_try.append({
                    "type": "relation",
                    "which_relation": "r2",  # default attempt: swap r2 (b->c); could also try r1
                    "candidate_relation": cand_rel
                })

            # If no perturbations found, do a fallback (simple relation-based swap using most common relations)
            if not perturbations_to_try:
                # simple fallback: attempt to change r2 to "RELATED_TO" or similar generic relation (if exists)
                perturbations_to_try.append({
                    "type": "relation",
                    "which_relation": "r2",
                    "candidate_relation": "RELATED_TO"
                })

            # For each perturbation, ask the LLM to generate counterfactual + contrastive
            for pert in perturbations_to_try:
                if pert["type"] == "entity":
                    cand = pert["candidate"]
                    candidate_name = cand.get("name")
                    perturbed_path = [p.copy() for p in original_path]
                    # replace middle node's name in path representation
                    for p in perturbed_path:
                        if p["head"] == middle_node.name:
                            # if middle node appears as head for second relation, replace only tail/head occurrences accordingly
                            pass
                    # More simply: replace any occurrence of the middle node name with candidate_name
                    perturbed_path = [{"head": (candidate_name if x["head"] == middle_node.name else x["head"]),
                                       "relation": x["relation"],
                                       "tail": (candidate_name if x["tail"] == middle_node.name else x["tail"])}
                                      for x in perturbed_path]

                    # Build user prompt for perturbation LLM call
                    perturb_user_content = (
                        "ORIGINAL CONTEXT:\n" + context_blob + "\n\n"
                        "ORIGINAL QUESTION:\n" + original_question + "\n\n"
                        "PERTURBATION:\nReplace the middle entity "
                        f"'{middle_node.name}' with '{candidate_name}'.\n\n"
                        "Produce (A) a COUNTERFACTUAL QUESTION that frames this replacement as a hypothetical (use 'If ...' or similar), "
                        "(B) a CONTRASTIVE QUESTION that is a plausible distractor with minimal change.\n"
                        "Output only JSON per the system instruction."
                    )
                    prompts_pert = [
                        {"role": "system", "content": SYSTEM_PROMPT_PERTURB},
                        {"role": "user", "content": perturb_user_content}
                    ]
                    try:
                        response_pert = await generate_response(prompts_pert, max_tokens=512, temperature=0.7, logger=logger)
                        parsed = json.loads(response_pert)
                        if isinstance(parsed, list) and parsed:
                            frag = parsed[0]
                            counterfactual_q = frag.get("counterfactual_question", "").strip()
                            contrastive_q = frag.get("contrastive_question", "").strip()
                            pert_entry = {
                                "original_question": original_question,
                                "counterfactual_question": counterfactual_q,
                                "contrastive_question": contrastive_q,
                                "perturbation_type": "entity",
                                "perturbed_element": candidate_name,
                                "original_path": original_path,
                                "perturbed_path": perturbed_path,
                                "metadata": {
                                    "middle_node_id": middle_node.id if middle_node else None,
                                    "candidate_id": cand.get("id")
                                }
                            }
                            output.append(pert_entry)
                        else:
                            logger.warn("Perturbation LLM did not return expected JSON list; skipping this perturbation.")
                    except Exception as e:
                        logger.error(f"Error generating perturbed questions for entity {candidate_name}: {e}")

                elif pert["type"] == "relation":
                    cand_rel = pert.get("candidate_relation")
                    which_rel = pert.get("which_relation", "r2")
                    # Build perturbed path: replace relation string in the appropriate relation slot.
                    perturbed_path = [p.copy() for p in original_path]
                    # If which_rel == 'r2' we replace the last relation in original_path (typical)
                    idx_to_replace = -1 if which_rel == "r2" else 0
                    # do replacement
                    if 0 <= idx_to_replace + len(perturbed_path) <= len(perturbed_path):
                        # safe index: replace by position
                        actual_idx = idx_to_replace if idx_to_replace >= 0 else len(perturbed_path) + idx_to_replace
                        if 0 <= actual_idx < len(perturbed_path):
                            perturbed_path[actual_idx]["relation"] = cand_rel

                    perturb_user_content = (
                        "ORIGINAL CONTEXT:\n" + context_blob + "\n\n"
                        "ORIGINAL QUESTION:\n" + original_question + "\n\n"
                        f"PERTURBATION:\nReplace relation '{original_path[actual_idx]['relation']}' with '{cand_rel}'.\n\n"
                        "Produce (A) a COUNTERFACTUAL QUESTION that frames this replacement as a hypothetical, "
                        "(B) a CONTRASTIVE QUESTION that is a plausible distractor. Output only JSON per the system instruction."
                    )
                    prompts_pert = [
                        {"role": "system", "content": SYSTEM_PROMPT_PERTURB},
                        {"role": "user", "content": perturb_user_content}
                    ]
                    try:
                        response_pert = await generate_response(prompts_pert, max_tokens=512, temperature=0.7, logger=logger)
                        parsed = json.loads(response_pert)
                        if isinstance(parsed, list) and parsed:
                            frag = parsed[0]
                            counterfactual_q = frag.get("counterfactual_question", "").strip()
                            contrastive_q = frag.get("contrastive_question", "").strip()
                            pert_entry = {
                                "original_question": original_question,
                                "counterfactual_question": counterfactual_q,
                                "contrastive_question": contrastive_q,
                                "perturbation_type": "relation",
                                "perturbed_element": cand_rel,
                                "original_path": original_path,
                                "perturbed_path": perturbed_path,
                                "metadata": {
                                    "replaced_relation_index": actual_idx
                                }
                            }
                            output.append(pert_entry)
                        else:
                            logger.warn("Perturbation LLM did not return expected JSON list; skipping this perturbation.")
                    except Exception as e:
                        logger.error(f"Error generating perturbed questions for relation {cand_rel}: {e}")

            # Incremental save after each path processed
            try:
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(output, f, indent=2, ensure_ascii=False)
            except Exception as e:
                logger.error(f"Failed to write output file: {e}")

        logger.info(f"Batch {batch_idx+1}/{BATCHES} done — total items collected: {len(output)}")

    logger.info(f"Finished generating perturbations. Total entries: {len(output)}")
    logger.info(f"Saved to {output_path}")
    return output_path
