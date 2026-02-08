import os
from typing import Any, Dict, Optional

from kg.kg_driver import *
from kg.kg_rep import *
from utils.utils import *
from utils.logger import *

"""
- LLM must generate BOTH:
    * original_question (a factual question grounded in the KG path)
    * question (a small, explicit hypothetical change)
    * answer (a concise, evidence-grounded answer to the counterfactual question)
- Output schema (exact):
[
  {
    "id": <int>,
    "question": "<original question>",
    "question": "<counterfactual question>",
    "answer": "<concise answer to the counterfactual question>",
    "grounding_path": [
      {"head": "<node_name>", "relation": "<REL_TYPE>", "tail": "<node_name>"},
      ...
    ]
  }
]

"""

# ======================== Configuration ======================== #
_OUTPUT_FILENAME = "questions_counterfactual.json"
_RAW_LOG_DIRNAME = "raw_model_outputs"
_GEN_ROUND = 20
_BATCH_LIMIT = 10
_MAX_RETRIES = 2  # retry LLM call + salvage attempts
# ======================== Hyperparameters ======================== #


# # ensure output directory exists
# os.makedirs(DATA_PATH, exist_ok=True)
# raw_log_dir = os.path.join(DATA_PATH, RAW_LOG_DIRNAME)
# os.makedirs(raw_log_dir, exist_ok=True)

# --- main adapter: from path query rows to objects ---
def instantiate_from_path_rows(rows: List[Dict[str, Any]]) -> Tuple[List[KGEntity], List[KGRelation]]:
    entities_by_id: Dict[str, KGEntity] = {}
    relations: List[KGRelation] = []

    for row in rows:
        nodes = row.get("nodes", [])  # [{id, labels, name, properties}, ...]
        rels  = row.get("rels", [])   # [{id, type, properties}, ...]

        # Build / dedupe entities
        for i, n in enumerate(nodes):
            nid   = n["id"]
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

        # Build relations with head/tail using the path order:
        node_ids = [n["id"] for n in nodes]
        for idx, r in enumerate(rels):
            rid   = r["id"]
            rtype = r["type"]
            rprops = r.get("properties") or {}
            # safe guard if nodes shorter than rels
            if idx + 1 >= len(node_ids):
                continue
            head  = node_ids[idx]
            tail  = node_ids[idx + 1]
            relations.append(
                KGRelation(
                    id=rid,
                    name=rtype,
                    source=entities_by_id.get(head),
                    target=entities_by_id.get(tail),
                    description=rprops.get(PROP_DESCRIPTION),
                    paragraph=rprops.get(PROP_PARAGRAPH),
                    properties=kg_driver.get_properties(rprops),
                )
            )

    return list(entities_by_id.values()), relations


# ----------------------------
# Strict system prompt
# ----------------------------
system_prompt = """\
*NEVER include ANY EXPLANATION or NOTE in the output, ONLY OUTPUT JSON* 
The following is the return JSON schema, with the key being the question ID, and the value stores the question/answer pair:
{
  "1": {
    "original_question": "<string>",
    "question": "<string>",
    "answer": "<string>",
    "grounding_path": [
      {"head": "<string>", "relation": "<string>", "tail": "<string>"}
    ]
  },
  "2": {
    ...
  }
  ...
}

INSTRUCTIONS:
1) For the provided KG path evidence, produce EXACTLY ONE object in the array.
2) original_question: a concise factual question derived from the KG path facts.
3) question: a concise counterfactual variant that changes one relation or a short property in the path (e.g., "If <head> had NOT <relation> <tail>..., how would ...?").
4) answer: a concise (1-3 sentence) grounded answer that explains plausible consequences or implications of the counterfactual using ONLY the provided evidence (phrase as cautious inference: "This could reduce...", "This might imply...", etc.). Do NOT invent new facts.
5) grounding_path: list the path edges used, each with head, relation, tail (plain strings).
6) If unsure about specifics, keep strings empty but KEEP THE KEYS.
7) DO NOT include extraneous commentary, markdown, or backticks. Return only the JSON array.

If you cannot satisfy the schema, still return the array with the required keys and empty strings.
"""

USER_PROMPT_TEMPLATE = """\
KG PATH CONTEXT (evidence paragraphs are included inline). Use these facts to create:
1) an original factual question,
2) a counterfactual question that modifies one relation/property,
3) a concise grounded answer to the counterfactual question.

Context:
{text}

Return EXACTLY ONE JSON array with ONE object following the required schema. No explanation.
"""


# ----------------------------
# JSON salvage helpers
# ----------------------------
def salvage_json_from_text(text: str) -> Any:
    """
    Try to extract a JSON array from model output by searching for bracketed arrays,
    trying candidates from longest to shortest until json.loads succeeds.
    Returns parsed JSON if successful, else None.
    """
    import re
    candidates = re.findall(r'\[[\s\S]*\]', text)
    if not candidates:
        return None
    # sort candidates by length descending (prefer longest)
    candidates = sorted(candidates, key=len, reverse=True)
    for c in candidates:
        try:
            parsed = json.loads(c)
            return parsed
        except Exception:
            continue
    return None


def normalize_grounding_path(raw_path: Any) -> List[Dict[str, str]]:
    normalized = []
    if not isinstance(raw_path, list):
        return normalized
    for g in raw_path:
        if not isinstance(g, dict):
            continue
        h = g.get("head") or g.get("source") or g.get("head_name") or ""
        rel = g.get("relation") or g.get("rel") or g.get("type") or ""
        t = g.get("tail") or g.get("target") or g.get("tail_name") or ""
        normalized.append({
            "head": (h if isinstance(h, str) else str(h))[:500],
            "relation": (rel if isinstance(rel, str) else str(rel))[:200],
            "tail": (t if isinstance(t, str) else str(t))[:500]
        })
    return normalized


async def generate(path: str, config: Optional[Dict[str, Any]] = {},
                   logger: Optional[BaseProgressLogger] = None):
    """Generate counterfactual QA pairs grounded on KG paths."""
    logger = logger or DefaultProgressLogger()

    OUTPUT_FILENAME = config.get("output_filename", _OUTPUT_FILENAME)
    gen_round = config.get("gen_round", _GEN_ROUND)
    BATCH_LIMIT = config.get("batch_limit", _BATCH_LIMIT)
    MAX_RETRIES = config.get("max_retries", _MAX_RETRIES)
    RAW_LOG_DIRNAME = config.get("raw_log_dirname", _RAW_LOG_DIRNAME)
    raw_log_dir = os.path.join(path, _RAW_LOG_DIRNAME)
    os.makedirs(path, exist_ok=True)
    os.makedirs(raw_log_dir, exist_ok=True)

    output_path = os.path.join(path, OUTPUT_FILENAME)
    output: List[Dict[str, Any]] = []

    # resume existing output if present
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
                if isinstance(existing, list):
                    output = existing
                    logger.info(f"Resumed from existing output with {len(output)} items.")
        except Exception:
            logger.warning("Failed to load existing output file; starting fresh.")

    for batch_idx in range(gen_round):
        logger.info(f"Batch {batch_idx+1}/{gen_round}: querying KG for paths...")

        results = kg_driver.run_query(
            """
            MATCH p=(a)-[r1]->(b)-[r2]->(c)
            WHERE a._paragraph IS NOT NULL AND b._paragraph IS NOT NULL AND c._paragraph IS NOT NULL
              AND r1._paragraph IS NOT NULL AND r2._paragraph IS NOT NULL
              AND a <> b AND b <> c AND a <> c
            WITH p, rand() AS rorder
            RETURN
              p,
              [n IN nodes(p) | {
                id: elementId(n),
                labels: labels(n),
                name: n.name,
                properties: apoc.map.removeKey(properties(n), "_embedding")
              }] AS nodes,
              [rel IN relationships(p) | {
                id: elementId(rel),
                type: type(rel),
                properties: properties(rel)
              }] AS rels
            ORDER BY rorder
            LIMIT $limit;
            """,
            {"limit": BATCH_LIMIT}
        )

        # instantiate objects
        entities, relations = instantiate_from_path_rows(results)

        if not relations:
            logger.info("No relations returned for this batch; continuing.")
            continue

        # build a context describing the path (use relation_to_text if available)
        context_lines = []
        for rel in relations:
            try:
                text = relation_to_text(rel, include_par=True)
            except Exception:
                head = (rel.source.name if rel.source else "UNKNOWN")
                tail = (rel.target.name if rel.target else "UNKNOWN")
                par = (rel.source.paragraph or "")[:500]
                text = f"({head}) -[{rel.name}]-> ({tail})\nEVIDENCE: {par}"
            context_lines.append(text)

        combined_context = "\n\n".join(context_lines)
        user_prompt = USER_PROMPT_TEMPLATE.format(text=combined_context)

        prompts = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Attempt LLM call with retries and salvage
        raw_response = None
        parsed_json = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                logger.info(f"Calling LLM (attempt {attempt+1})...")
                raw_response = await generate_response(
                    prompts,
                    max_tokens=2048,
                    temperature=0.3 + 0.15 * attempt,  # slightly higher temp on retries
                    response_format={"type": "json_object"},
                    logger=logger
                )
            except Exception as e:
                raw_response = f""  # keep it simple
                logger.error(f"LLM call raised exception: {e}")
                raw_response = raw_response or ""

            # save raw model output for debugging
            ts = int(time.time())
            raw_log_file = os.path.join(raw_log_dir, f"batch{batch_idx+1}_attempt{attempt+1}_{ts}.txt")
            try:
                with open(raw_log_file, "w", encoding="utf-8") as rf:
                    rf.write(raw_response or "")
            except Exception:
                logger.warning("Could not write raw model output file.")

            # Try direct parse
            try:
                logger.info("Question extraction..." + raw_response)
                parsed_json = json.loads(raw_response)
            except Exception:
                parsed_json = salvage_json_from_text(raw_response)

            if parsed_json is not None:
                logger.info("Successfully parsed JSON from model output.")
                break
            else:
                logger.warning(f"JSON parse failed on attempt {attempt+1}.")
                # Slightly modify the last system prompt to be stricter on the next retry
                prompts[0]["content"] = system_prompt + "\nReminder: RETURN ONLY THE REQUIRED JSON ARRAY. NOTHING ELSE."
                # continue to next attempt

        if parsed_json is None:
            logger.error("JSON parse failed after retries. Skipping this batch.")
            # continue with next batch without adding anything
            continue

        # Validate and append to output
        for qid, item in parsed_json.items():
            if not isinstance(item, dict):
                continue

            original_q = item.get("original_question") or ""
            cf_q = item.get("question") or item.get("counterfactual_question") or ""
            cf_ans = item.get("answer") or item.get("counterfactual_answer") or ""

            grounding_raw = item.get("grounding_path") or item.get("path") or []
            grounding = normalize_grounding_path(grounding_raw)

            # Basic validation: at least original question present
            if not isinstance(original_q, str):
                original_q = str(original_q) if original_q is not None else ""
            if not isinstance(cf_q, str):
                cf_q = str(cf_q) if cf_q is not None else ""
            if not isinstance(cf_ans, str):
                cf_ans = str(cf_ans) if cf_ans is not None else ""

            # If grounding is empty, attempt to construct it from relations we provided to the prompt
            if not grounding:
                # Attempt to build grounding from available relations list (best-effort)
                grounding = []
                for rel in relations:
                    h = rel.source.name if (rel.source and rel.source.name) else ""
                    rname = rel.name if rel.name else ""
                    t = rel.target.name if (rel.target and rel.target.name) else ""
                    if h or rname or t:
                        grounding.append({"head": h, "relation": rname, "tail": t})

            # Append to output
            out_item = {
                "id": len(output),
                "original_question": original_q.strip(),
                "question": cf_q.strip(),
                "answer": cf_ans.strip(),
                "grounding_path": grounding
            }

            # Minimal schema normalization: ensure keys exist
            for k in ["original_question", "question", "answer", "grounding_path"]:
                if k not in out_item:
                    out_item[k] = "" if k != "grounding_path" else []

            output.append(out_item)

        # write incremental output so we don't lose work
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to write output file: {e}")

        logger.info(f"Batch {batch_idx+1} complete. Total items so far: {len(output)}")

    logger.info(f"Finished. Wrote {len(output)} items to {output_path}")
    return output_path
