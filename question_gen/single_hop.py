import openai
import os, sys; sys.path.append(os.path.abspath('..'))
from kg.kg_driver import *
from kg.kg_rep import *
from utils.utils import *
from utils.logger import *

from typing import Any, Dict, Optional

# ======================== Hyperparameters ======================== #
_GEN_ROUND = 20
# default query limit for relation used for each batch of question generation
_DEFAULT_BATCH_QUESTION_INPUT_SIZE = 10
_DEFAULT_MAX_TOKENS = 4096
_DEFAULT_TEMPERATURE = 0.4
# ======================== Hyperparameters ======================== #

# --- main adapter: from path query rows to objects ---
def instantiate_from_path_rows(rows: List[Dict[str, Any]]) -> Tuple[List[KGEntity], List[KGRelation]]:
    entities_by_id: Dict[str, KGEntity] = {}
    relations: List[KGRelation] = []

    for row in rows:
        nodes = row["nodes"]  # [{id, labels, name, properties}, ...]
        rels  = row["rels"]   # [{id, type, properties}, ...]

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

        node_ids = [n["id"] for n in nodes]
        for idx, r in enumerate(rels):
            rid   = r["id"]
            rtype = r["type"]
            rprops = r.get("properties") or {}
            head  = node_ids[idx]
            tail  = node_ids[idx + 1]
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


system_prompt = """\
You are generating SINGLE-HOP question–answer (QA) pairs grounded ONLY in the provided knowledge graph relation.

Each input contains:
(entity_type: entity_name, paragraph: "<paragraph>")-[relation_type]->(entity_type: entity_name, paragraph: "<paragraph>")

Requirements:
1) Single-hop grounding: The question must be answerable using ONLY this one relation and its supporting paragraphs.
2) Relational focus: The question should reflect the meaning of the relation_type (e.g., “contributed_to,” “advised_by,” “collaborated_with”), not just ask “Who” or “What.”
3) Evidence-based: Both question and answer must be directly supported by the paragraph text.
4) No trivial fact lookups: Avoid generic forms like “Who authored…?” or “What organization is X affiliated with?"
5) Explicit entities: Always name entities explicitly — do NOT use pronouns like “this paper."
6) Concise answer: The answer must be a short entity or phrase directly drawn from the paragraph.
7) No hallucination: Do NOT invent or infer any facts beyond what’s in the text.

Examples of good single-hop question styles:
- “What relationship is described between <entity A> and <entity B>?”
- “How does <entity A> contribute to <entity B> according to the paragraph?”
- “What role does <entity A> have in relation to <entity B>?”

*NEVER include ANY EXPLANATION or NOTE in the output, ONLY OUTPUT JSON*  
The following is the return JSON schema, with the key being the question ID, and the value stores the question/answer pair:
{
  "1": {
    "question": "<relation-grounded question>",
    "answer": "<concise, text-supported answer>",
    "reasoning_path": [
      {"head": "<node_name>", "relation": "<REL_TYPE>", "tail": "<node_name>"}
    ]
  },
  "2": {
    ...
  }
  ...
}
"""

USER_PROMPT = """\
Generate SINGLE-HOP QA pairs based on the following KG relation:

{text}

Only output JSON. Do NOT add explanations.
"""


QUERY_REGULAR = """
MATCH p=(a)-[r]->(b)
WHERE a._paragraph IS NOT NULL AND b._paragraph IS NOT NULL AND r._paragraph IS NOT NULL
AND a <> b
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
"""

# query to retrieve relation whose nodes have high degree.
QUERY_DIFFICULT = """
MATCH (n)
WITH n, COUNT { (n)--() } AS deg
ORDER BY deg DESC
LIMIT 20
WITH collect(n) AS topNodes

MATCH p=(a)-[r]-(b)
WHERE (a IN topNodes OR b IN topNodes)
  AND elementId(a) < elementId(b)
  AND a._paragraph IS NOT NULL
  AND b._paragraph IS NOT NULL
  AND r._paragraph IS NOT NULL

WITH p, rand() AS k
ORDER BY k
LIMIT $limit

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
  }] AS rels;
"""


async def generate(
    path: str, 
    config: Optional[Dict[str, Any]] = {},
    logger: Optional[BaseProgressLogger] = None
):
    """Generate single-hop QA pairs grounded on KG relations."""
    logger = logger or DefaultProgressLogger()

    logger.info(f"Use Config: {config}")

    gen_round = config.get("gen_round") or _GEN_ROUND
    limit = config.get("batch_question_input_size") or _DEFAULT_BATCH_QUESTION_INPUT_SIZE
    max_tokens = config.get("max_tokens") or _DEFAULT_MAX_TOKENS
    temperature = config.get("temperature") or _DEFAULT_TEMPERATURE

    fn_suffix = f"_difficult" if config.get("difficulty", "regular") == "difficult" else ""
    output_path = os.path.join(path, f"questions_single_hop{fn_suffix}.json")

    output = []
    for idx in range(gen_round):
        logger.info(f"Working on {idx} batch of question:")

        # Query modified to return a SINGLE EDGE (one-hop relation)
        results = kg_driver.run_query(
            QUERY_DIFFICULT if config.get("difficulty", "regular") == "difficult" else QUERY_REGULAR,
            {"limit": limit}
        )

        entities, relations = instantiate_from_path_rows(results)
        context = [relation_to_text(relation, include_par=False) for relation in relations]

        user_prompt = USER_PROMPT.format(text="\n".join(context))
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

        logger.info("Question extraction..." + response)
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
