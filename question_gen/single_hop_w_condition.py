import json
import os, sys; sys.path.append(os.path.abspath('..'))
from typing import Any, Dict, Optional, List

from kg.kg_driver import *
from kg.kg_rep import *
from utils.utils import *
from utils.logger import *

# ======================== Hyperparameters ======================== #
_GEN_ROUND = 20
# default query limit for relation used for each batch of question generation
_DEFAULT_BATCH_QUESTION_INPUT_SIZE = 20
_DEFAULT_MAX_TOKENS = 8192
_DEFAULT_TEMPERATURE = 0.4
# ======================== Hyperparameters ======================== #

QUERY_REGULAR = """
MATCH p=(a)-[r]->(b)
// WHERE a._paragraph IS NOT NULL AND b._paragraph IS NOT NULL AND r._paragraph IS NOT NULL
WHERE r._description IS NOT NULL
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
        properties: apoc.map.removeKey(properties(rel), "_embedding")
    }] AS rels
ORDER BY rorder
LIMIT $limit;
"""

SYSTEM_PROMPT = """
You are an intelligent dataset generator. Your task is to generate a simple, fact-based Question and Answer pair based *strictly* on the provided structured information about a Relation, a Head Entity, and a Tail Entity.

### **Input Format**
You will be provided with information about a relation and its entities in a knowledge graph, in the following format:

- "relation":
	- "type": <type_of_relation>
    	- "name": <name_of_relation>
    	- "description": <description_of_relation>
    	- "paragraph": <paragraph_of_relation>

- "head_entity":
	- "type": <type_of_head_entity>
    	- "name": <name_of_head_entity>
    	- "description": <description_of_head_entity>
    	- "paragraph": <paragraph_of_head_entity>

- "tail_entity":
	- "type": <type_of_tail_entity>
    	- "name": <name_of_tail_entity>
    	- "description": <description_of_tail_entity>
    	- "paragraph": <paragraph_of_tail_entity>

### **Task Instructions**
- **Analyze the Data:** Read the "paragraph" and "description" fields carefully.
- **Formulate a Question:** Create a simple natural language question that connects the entities using the relation with a restrictive condition.
- * **Conditions:** Incorporate specific conditions found in the text, such as **time** (years, dates), **location**, or **specific attributes** (e.g., "in the second movie", "stock price last week").
- **Provide the Answer:** Extract the direct answer from the text.
- **Strict Constraints:**
    * **NO HALLUCINATION:** Do not make up information. The question must be answerable *solely* using the provided "paragraph" and "description" fields.
    * **Insufficient Information:** If If neither the paragraph nor the description provides enough specific details to form a valid question or if the relation is unclear, you must output "no proper question" in the question field.
- ** The **Restrictive Condition**: It is critical to incorporate specific attributes, roles, or conditions found in the text into the question. Do not ask "Where does X work?" if you can ask "Where does X, the author of the Yellow Paper, work?". Use:
		- Time/Date: (e.g., "In 2010...", "during the second quarter...")
		- Specific Attributes/Roles: (e.g., "As the lead architect...", "the purple version of...")
		- Location/Context: (e.g., "At the Melbourne circuit...", "In the second film...")

### **Output Format**
You must respond with a JSON object containing exactly these fields:

*NEVER include ANY EXPLANATION or NOTE in the output, ONLY OUTPUT JSON*  
The following is the return JSON schema, with the key being the question ID, and the value stores the question/answer pair:
{
  "1": {
    "question": "<The generated question string OR 'no proper question'>",
    "answer": "<The answer string OR ''>"
  },
  "2": {
    ...
  }
  ...
}

---

### **Few-Shot Examples**


**Example 1: Time Condition**
*Input:*
* **Relation:** Director
* **Head Entity:** Christopher Nolan (Person) - "Christopher Nolan directed the sci-fi thriller Inception which was released to critical acclaim in 2010."
* **Tail Entity:** Inception (Film)
* **Output:**
- "question": "What 2010 film was directed by Christopher Nolan?",
- "answer": "Inception"


**Example 2: Attribute/Specific Condition**
*Input:*
* **Relation:** Driven By
* **Head Entity:** Roman Pearce (Character) - "In the 2003 film *2 Fast 2 Furious*, Roman Pearce is seen driving a purple 2003 Mitsubishi Eclipse Spyder."
* **Tail Entity:** Mitsubishi Eclipse Spyder (Car)
* **Output:**
- "question": "What car did Roman Pearce drive in the second Fast and Furious movie?",
- "answer": "A 2003 Mitsubishi Eclipse Spyder"


**Example 3: Location/Event Condition**
*Input:*
* **Relation:** Location
* **Head Entity:** 2011 F1 Season (Event) - "The 2011 Formula One season kicked off with the Australian Grand Prix held in Melbourne."
* **Tail Entity:** Melbourne (City)
* **Output:**
- "question": "Where was the first race of the 2011 F1 season?",
- "answer": "Melbourne"


**Example 4: Financial/Value Condition**
*Input:*
* **Relation:** Stock Performance
* **Head Entity:** Eli Lilly and Company (Company) - "As of the recent report, Eli Lilly and Company's stock price has seen a steady increase and is currently up from its yearly open."
* **Tail Entity:** Yearly Open (Financial Metric)
* **Output:**
- "question": "Is Eli Lilly and Company's stock price up from its yearly open?",
- "answer": "Yes"


**Example 5: Insufficient Information (Negative Case)**
*Input:*
* **Relation:** Member of
* **Head Entity:** John Smith - "John Smith is a software engineer living in Seattle."
* **Tail Entity:** Unknown Club - [Empty Paragraph]
* **Output:**
- "question": "no proper question",
- "answer": ""


** Example 6: Ambiguous Reference (Negative Case)
*Input:*
* **Relation**: Financial Support
* **Head Entity**: Office of Naval Research (ONR)
* **Tail Entity**: N00014-23-1-2345 (Grant Number) - "ONR provided grant N00014-23-1-2345 to support the research in this work."
* **Output**:
- "question": "no proper question",
- "answer": "",
- "explain": "The phrase 'this work' is an ambiguous reference that lacks specific context outside of the source document."


** Example 7: Redundant Condition (Negative Case)
*Input:*
* **Relation**: Developed By
* **Head Entity**: MEMIT (Method) - "Kevin Meng developed the method MEMIT, which was published in 2023."
* **Tail Entity**: Kevin Meng (Person)
* **Output**:
- "question": "no proper question",
- "answer": "",
- "explain": "The condition 'published in 2023' is redundant because it is a fixed, inherent attribute of the entity MEMIT."


** Example 8: Missing Restrictive Condition (Negative Case)
*Input:*
* **Relation**: Evaluation Benchmark
* **Head Entity**: CausalPlan (Algorithm) - "CausalPlan was evaluated on the Blocksworld benchmark."
* **Tail Entity**: Blocksworld (Benchmark)
* **Output**:
- "question": "no proper question",
- "answer": "",
- "explain": "The contexts lacks a specific restrictive condition in addition to the main relation `evaluated on`."

"""
USER_TEMPLATE = """
- "relation":
	- "type": "{e_type}"
    	- "name": "{e_name}"
    	- "description": "{e_description}"
    	- "paragraph": "{e_paragraph}"

- "head_entity":
	- "type": "{h_type}"
	- "name": "{h_name}"
	- "description": "{h_description}"
	- "paragraph": "{h_paragraph}"

- "tail_entity":
	- "type": "{t_type}"
	- "name": "{t_name}"
	- "description": "{t_description}"
	- "paragraph": "{t_paragraph}"
"""
USER_PROMPT = """
**Now, generate the QA pair based on the following input: 

{text}

Only output JSON. Do NOT add explanations.
"""


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

    output_path = os.path.join(path, f"questions_single_hop_w_conditions.json")

    output = []
    for idx in range(gen_round):
        logger.info(f"Working on {idx} batch of question:")
        results = kg_driver.run_query(QUERY_REGULAR, {"limit": limit})
        _, relations = instantiate_from_path_rows(results)
        contexts = [relation_to_dict(rel) for rel in relations]
        contexts = _filter_pure_conceptual_relations(contexts)
        contexts = [_format_context(context) for context in contexts]
        
        user_prompt = USER_PROMPT.format(text="\n".join(contexts))
        prompts = [
            {"role": "system", "content": SYSTEM_PROMPT},
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


def _format_context(parsed_relation: dict):
    """
    Given a parsed_relation (output of relation_to_dict) and a template string,
    fill the template with relation/head/tail attributes.
    """
    # Map parsed_relation keys to template placeholders
    variables = {
        # Relation fields
        "e_type": parsed_relation.get("e_name", "") or "",
        "e_name": parsed_relation.get("e_name", "") or "",
        "e_description": parsed_relation.get("e_description", "") or "",
        "e_paragraph": parsed_relation.get("e_paragraph", "") or "",
        # Head entity fields
        "h_type": parsed_relation.get("h_type", "") or "",
        "h_name": parsed_relation.get("h_name", "") or "",
        "h_description": parsed_relation.get("h_description", "") or "",
        "h_paragraph": parsed_relation.get("h_paragraph", "") or "",
        # Tail entity fields (template uses e_* for tail)
        "t_type": parsed_relation.get("t_type", "") or "",
        "t_name": parsed_relation.get("t_name", "") or "",
        "t_description": parsed_relation.get("t_description", "") or "",
        "t_paragraph": parsed_relation.get("t_paragraph", "") or "",
    }
    # Assume the template only contains well-formed placeholders corresponding to keys in `variables`
    return USER_TEMPLATE.format(**variables)


def _filter_pure_conceptual_relations(relations: List[Dict[str, Any]]):
    """Filter relations based on the condition.
    
    We think these conditions suggest the relation cannot generate a proper question.
    """
    relations_filtered = []
    for rel in relations:
        h_type = str(rel.get("h_type", "") or "").lower()
        t_type = str(rel.get("t_type", "") or "").lower()
        if h_type == "concept" and t_type == "concept":
            h_name = rel.get("h_name", "")
            t_name = rel.get("t_name", "")
            logger.info(
                f"Filter out {h_name} - {h_type}, {t_name} - {t_type} due to all entities are conceptual."
            )
        else:
            relations_filtered.append(rel)
    return relations_filtered