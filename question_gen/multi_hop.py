import os
from typing import Any, Dict, Optional

from kg.kg_driver import *
from kg.kg_rep import *
from utils.utils import *
from utils.logger import *

logger = DefaultProgressLogger()

# ======================== Hyperparameters ======================== #
_GEN_ROUND = 20
_DEFAULT_MAX_TOKENS = 8192
_DEFAULT_TEMPERATURE = 0.5
_DEFAULT_BATCH_QUESTION_INPUT_SIZE = 5
# ======================== Hyperparameters ======================== #

# --- main adapter: from path query rows to objects ---
def instantiate_from_path_rows(rows: List[Dict[str, Any]]) -> Tuple[List[KGEntity], List[KGRelation]]:
    entities_by_id: Dict[str, KGEntity] = {}
    relations: List[KGRelation] = []

    for row in rows:
        nodes = row["nodes"]  # [{id, labels, name, properties}, ...]
        rels  = row["rels"]   # [{id, type, properties}, ...]
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
        # relationships(p) aligns with consecutive node pairs: (nodes[0] -> nodes[1]), (nodes[1] -> nodes[2]), ...
        node_ids = [n["id"] for n in nodes]
        rel_ids  = []
        for idx, r in enumerate(rels):
            rid   = r["id"]
            rtype = r["type"]
            rprops = r.get("properties") or {}
            head  = node_ids[idx]
            tail  = node_ids[idx + 1]
            rel_ids.append(rid)
            relations.append(
                KGRelation(
                    id=rid,
                    name=rtype,
                    source=entities_by_id[head],
                    target=entities_by_id[tail],
                    description=props.get(PROP_DESCRIPTION),
                    paragraph=rprops.get(PROP_PARAGRAPH),   # or anchors if that’s your schema
                    properties=kg_driver.get_properties(rprops),
                )
            )

    return list(entities_by_id.values()), relations

system_prompt = """\
You are generating question–answer (QA) pairs grounded ONLY in the provided paths from a knowledge graph.
Each path supplies:
(entity_type: entity_name, desc: "<description>", paragraph: "<paragraph>", props: {{key: [val_1 (70%, ctx:"context"), val_2 (30%, ctx:"context")], ...}})-[relation_type, desc: "<description>", paragraph: "<paragraph>", props: {{key: [val_1 (70%, ctx:"context"), val_2 (30%, ctx:"context")], ...}}]->(entity_type: entity_name, desc: "<description>", paragraph: "<paragraph>", props: {{key: [val_1 (70%, ctx:"context"), val_2 (30%, ctx:"context")], ...}}).
"paragraph" is the evidence text where the node/relationship is stated.

Requirements:
1) Grounding: Every answer must be directly supported by the supplied paragraphs. Do NOT invent facts.
2) Multi-hop: For each path, produce at least one multi-hop QA whose answer requires chaining ALL relationships in that path in order.
3) Self-contained entities: Never use "this paper" or pronouns. Always use explicit names from "name".
4) Concise answers: Prefer a single entity, number, or short phrase.
5) No outside knowledge or normalization beyond what evidence states (e.g., keep reported years as given).

Output JSON only. No explanations.

For each input path, produce 1–2 QA pairs:
- If the path length is 2 (three nodes): produce 1 multi-hop QA using both relationships.
- If the path length is 3 (four nodes): produce 1–2 multi-hop QAs that require all three relations.

*NEVER include ANY EXPLANATION or NOTE in the output, ONLY OUTPUT JSON*  
The following is the return JSON schema, with the key being the question ID, and the value stores the question/answer pair:
{
  "1": {
    "question": "<multi-hop question text>",
    "answer": "<concise answer>",
    "reasoning_path": [
      {"head": "<node_name>", "relation": "<REL_TYPE>", "tail": "<node_name>"},
      ...
    ]
  },
  "2": {
    ...
  }
  ...
}
"""
USER_PROMPT = """\
Now, based on the following paths extracted from KG, generate question–answer pairs that satisfy the above requirements:

{text}

Please output the extracted questions and answers in the previously mentioned JSON format.
*NEVER include ANY EXPLANATION or NOTE in the output, ONLY OUTPUT FLAT JSON*
"""


QUERY_REGULAR = """\
MATCH p=(a)-[r1]->(b)-[r2]->(c)
WHERE a._paragraph IS NOT NULL AND b._paragraph IS NOT NULL AND c._paragraph IS NOT NULL
AND r1._paragraph IS NOT NULL AND r2._paragraph IS NOT NULL
AND a <> b AND b <> c AND a <> c
WITH p, rand() AS r
RETURN
p,
[n IN nodes(p) | {id: elementId(n), labels: labels(n), name: n.name, properties: apoc.map.removeKey(properties(n), "_embedding")}] AS nodes,
[r IN relationships(p) | {id: elementId(r), type: type(r), properties: properties(r)}] AS rels
ORDER BY r
LIMIT $limit;
"""


# query to retrieve relation whose nodes have high degree.
QUERY_DIFFICULT = """\
MATCH (n)
WITH n, COUNT { (n)--() } AS deg
ORDER BY deg DESC
LIMIT 20
WITH collect(n) AS topNodes

MATCH p=(a)-[r1]->(b)-[r2]->(c)
WHERE b IN topNodes
  AND elementId(a) < elementId(b) AND elementId(b) < elementId(c)
  AND a._paragraph IS NOT NULL AND b._paragraph IS NOT NULL AND c._paragraph IS NOT NULL
  AND r1._paragraph IS NOT NULL AND r2._paragraph IS NOT NULL
  AND a <> b AND b <> c AND a <> c
WITH p, rand() AS r
RETURN
p,
[n IN nodes(p) | {id: elementId(n), labels: labels(n), name: n.name, properties: apoc.map.removeKey(properties(n), "_embedding")}] AS nodes,
[r IN relationships(p) | {id: elementId(r), type: type(r), properties: properties(r)}] AS rels
ORDER BY r
LIMIT $limit;
"""


async def generate(path: str, config: Optional[Dict[str, Any]] = {},
                   logger: Optional[BaseProgressLogger] = None):
    """Generate multi-hop QA pairs grounded on KG paths."""
    logger = logger or DefaultProgressLogger()

    logger.info(f"Use Config: {config}")

    fn_suffix = f"_difficult" if config.get("difficulty", "regular") == "difficult" else ""
    gen_round = config.get("gen_round") or _GEN_ROUND
    limit = config.get("batch_question_input_size") or _DEFAULT_BATCH_QUESTION_INPUT_SIZE
    max_tokens = config.get("max_tokens") or _DEFAULT_MAX_TOKENS
    temperature = config.get("temperature") or _DEFAULT_TEMPERATURE
    
    output_path = os.path.join(path, f"questions_multi_hop{fn_suffix}.json")
    
    output = []
    for idx in range(gen_round):
        logger.info(f"Working on {idx} batch of question:")
        results = kg_driver.run_query(
            QUERY_DIFFICULT if config.get("difficulty", "regular") == "difficult" else QUERY_REGULAR
            , {"limit": limit}
        )
        entities, relations = instantiate_from_path_rows(results)
        context = [relation_to_text(relation, include_par=False) for relation in relations]
        
        user_prompt = USER_PROMPT.format(
            text="\n".join(context)
        )

        prompts = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = await generate_response(
            prompts, 
            max_tokens=max_tokens,
            temperature=temperature,
            response_format={"type": "json_object"},
            logger=logger
        )

        logger.info(f"Question extraction..." + response)
        questions = json.loads(response)
    
        for qid, question in questions.items():
            output.append({
                "id": len(output),
                "question": question["question"],
                "answer": question["answer"]
            })

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=4, ensure_ascii=False)

    return output_path
