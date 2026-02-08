import json
import os
import re
from typing import Any, Dict, Iterable, Optional

from utils.logger import BaseProgressLogger, DefaultProgressLogger
from utils.utils import generate_response

GEN_ROUND = 1
DATA_PATH = "/Users/junhonglin/Data/10Arxiv_on_KG"

SYSTEM_PROMPT = """\
You are generating question–answer (QA) pairs grounded ONLY in the provided paths from a knowledge graph.
Each path supplies:
- nodes: {id, labels, name, paragraph}
- relationships: {id, type, paragraph}
"paragraph" is the evidence text where the node/relationship is stated.

Requirements:
1) Grounding: Every answer must be directly supported by the supplied paragraphs. Do NOT invent facts.
2) Multi-hop: For each path, produce at least one multi-hop QA whose answer requires chaining ALL relationships in that path in order.
3) Self-contained entities: Never use "this paper" or pronouns. Always use explicit names from "name".
4) Concise answers: Prefer a single entity, number, or short phrase.
5) No outside knowledge or normalization beyond what evidence states (e.g., keep reported years as given).
6) Provenance: For each QA, include a "provenance" listing the ordered triple chain and the IDs (node/rel) used.

Output JSON only. No explanations.

For each input path, produce 1–2 QA pairs:
- If the path length is 2 (three nodes): produce 1 multi-hop QA using both relationships.
- If the path length is 3 (four nodes): produce 1–2 multi-hop QAs that require all three relations.

JSON schema:
[
  {
    "question": "<multi-hop question text>",
    "answer": "<concise answer>",
    "reasoning_path": [
      {"head": "<node_name>", "relation": "<REL_TYPE>", "tail": "<node_name>"},
      ...
    ],
    "provenance": {
      "node_ids": [<neo4j-id>, ...],
      "rel_ids": [<neo4j-id>, ...]
    }
  }
]
"""

USER_PROMPT = """\
Now, based on the following paper text, generate 10–20 diverse factual question–answer pairs that satisfy the above requirements:
{text}

Please output the extracted questions and answers in the previously mentioned JSON format.
*NEVER include ANY EXPLANATION or NOTE in the output, ONLY OUTPUT JSON*
"""

DEFAULT_CONFIG = {
    "data_path": DATA_PATH,
    "gen_round": GEN_ROUND,
}


def load_documents(directory_path: str, start_idx: int = 0) -> Iterable[Dict[str, Any]]:
    filepaths = [
        os.path.join(directory_path, filename)
        for filename in os.listdir(directory_path)
        if filename.endswith(".md")
    ]
    filepaths.sort(
        key=lambda x: int(re.search(r"(\d+)", x).group(1))
        if re.search(r"(\d+)", x)
        else float("inf")
    )

    for idx, filepath in enumerate(filepaths):
        if idx < start_idx:
            continue
        entry = {
            "id": os.path.splitext(os.path.basename(filepath))[0],
            "context": None,
            "doc_path": filepath,
        }
        try:
            with open(filepath, "r", encoding="utf-8") as file:
                content = file.read().strip()
                entry["context"] = content
        except Exception as exc:  # noqa: BLE001
            print(f"Error reading file {filepath}: {exc}")
            continue
        yield entry


async def generate(path: str,
                   config: Optional[Dict[str, Any]] = None,
                   logger: Optional[BaseProgressLogger] = None):
    """Generate QA pairs directly from paper text files."""
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    logger = logger or DefaultProgressLogger()

    data_path = path or cfg["data_path"]
    gen_round = cfg.get("gen_round", GEN_ROUND)
    output_path = os.path.join(data_path, "questions.json")

    output = []
    for idx, text in enumerate(load_documents(data_path)):
        logger.info(f"Working on {idx} paper: {text['id']}")
        user_prompt = USER_PROMPT.format(text=text["context"])
        prompts = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        results = []
        for i in range(gen_round):
            logger.info(f"Round {i} generation...")
            response = await generate_response(
                prompts,
                max_tokens=10240,
                temperature=0.5,
                response_format={"type": "json_object"},
                logger=logger,
            )
            results.append(response)
            prompts.extend(
                [
                    {"role": "assistant", "content": response},
                    {"role": "user", "content": "Great! Could you generate more such questions?"},
                ]
            )

        logger.info("Question extraction...")
        for result in results:
            try:
                questions = json.loads(result)
            except Exception:  # noqa: BLE001
                continue

            for question in questions:
                output.append(
                    {
                        "id": len(output),
                        "paper": text["id"],
                        "question": question["question"],
                        "answer": question["answer"],
                    }
                )
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(output, file, indent=4, ensure_ascii=False)

    return output_path
