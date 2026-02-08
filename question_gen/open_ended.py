import os
from typing import Any, Dict, Optional

from kg.kg_driver import *
from kg.kg_rep import *
from utils.utils import *
from utils.logger import *

# ======================== Hyperparameters ======================== #
_GEN_ROUND = 20
# ======================== Hyperparameters ======================== #

# --- main adapter: from path query rows to objects ---
def instantiate_from_path_rows(rows: List[Dict[str, Any]]) -> Tuple[List[KGEntity], List[KGRelation]]:
    entities_by_id: Dict[str, KGEntity] = {}
    relations: List[KGRelation] = []

    for row in rows:
        nodes = row["nodes"]  # [{id, labels, name, properties}, ...]
        rels = row["rels"]  # [{id, type, properties}, ...]

        for i, n in enumerate(nodes):
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


# NEW SYSTEM PROMPT — generate open-ended questions AND answers
# --- EDITED SYSTEM PROMPT ---
system_prompt = """\
You are generating OPEN-ENDED QUESTIONS and corresponding ANSWERS grounded ONLY in the provided knowledge graph paths.

Each path contains entities and relations with evidence ("paragraph").
Use ONLY the information in the path to form a question and its answer.

Requirements for the QUESTION:
1) The question must be **open-ended**, encouraging explanation or reasoning, not a fact lookup.
2) The question must stay **strictly within the evidence** — do NOT imply causation, influence, or significance unless those concepts are explicitly stated in the paragraph.
3) Do NOT speculate or assume relationships beyond what is given.
4) Use explicit entity names; avoid pronouns like “this paper” or “their work”.
5) Keep the question neutral — do not presuppose importance, impact, or outcomes.

Requirements for the ANSWER:
1) The answer must be a **small paragraph of text** that directly and sufficiently answers the open-ended question based **only** on the path evidence.
2) The answer should provide a comprehensive explanation or reasoning requested by the question.

Examples of acceptable question styles:
- “What is the relationship between <entity> and <entity> described in the paragraph?”
- “In what ways does <entity> relate to <entity> through <relation>?”
- “How does the paragraph describe the connection between <entity> and <entity>?”

*NEVER include ANY EXPLANATION or NOTE in the output, ONLY OUTPUT JSON*  
The following is the return JSON schema, with the key being the question ID, and the value stores the question/answer pair:
{
  "1": {
    "question": "<open-ended question>",
    "answer": "<paragraph-long answer grounded in evidence>"
  },
  "2": {
    ...
  }
  ...
}
"""
# ----------------------------

USER_PROMPT = """\
Generate ONE OPEN-ENDED QUESTION and ONE corresponding ANSWER for each KG path in the following list:

{text}

Output ONLY JSON as specified above.
"""


async def generate(path: str, config: Optional[Dict[str, Any]] = {},
                   logger: Optional[BaseProgressLogger] = None):
    """Generate open-ended QA pairs grounded on KG paths."""
    logger = logger or DefaultProgressLogger()

    GEN_ROUND = config.get("gen_round", _GEN_ROUND)

    # Changed output file name to reflect inclusion of answers
    output_path = os.path.join(path, "questions_open_ended.json")

    output = []
    for idx in range(GEN_ROUND):
        logger.info(f"Working on {idx} batch of question/answer generation:")

        # The query logic remains the same
        results = kg_driver.run_query(
            """
            MATCH p=(a)-[r1]->(b)-[r2]->(c)
            WHERE a._paragraph IS NOT NULL AND b._paragraph IS NOT NULL AND c._paragraph IS NOT NULL
            AND r1._paragraph IS NOT NULL AND r2._paragraph IS NOT NULL
            AND a <> b AND b <> c AND a <> c
            WITH p, rand() AS r
            RETURN
                p,
                [n IN nodes(p) | {
                    id: elementId(n),
                    labels: labels(n),
                    name: n.name,
                    properties: apoc.map.removeKey(properties(n), "_embedding")
                }] AS nodes,
                [r IN relationships(p) | {
                    id: elementId(r),
                    type: type(r),
                    properties: properties(r)
                }] AS rels
            ORDER BY r
            LIMIT $limit;
            """,
            {"limit": 10}
        )

        entities, relations = instantiate_from_path_rows(results)
        # Include paragraph evidence in the context to ground the answer generation
        context = [relation_to_text(relation, include_par=True) for relation in relations]

        user_prompt = USER_PROMPT.format(text="\n".join(context))
        prompts = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = await generate_response(
            prompts,
            max_tokens=10240,
            temperature=0.7,
            response_format={"type": "json_object"},
            logger=logger
        )

        logger.info(f"Extracted questions and answers..." + response)
        questions_answers = json.loads(response)

        # --- EDITED OUTPUT HANDLING ---
        for qid, qa in questions_answers.items():
            output.append({
                "id": len(output),
                "question": qa.get("question", ""),
                "answer": qa.get("answer", "")  # Added 'answer' field
            })
        # ----------------------------

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=4, ensure_ascii=False)

    return output_path
