import asyncio
import re
import textwrap
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from inference import *
from kg.kg_driver import *
from kg.kg_rep import *
from utils.prompt_list import *
from utils.utils import *
from utils.logger import *

######################################################################################################
######################################################################################################
###
# Chain-of-Knowledge style baseline adapted to the Bidirection KG interface.
# This version uses only KG retrieval (no external web sources).
######################################################################################################

# CONFIG PARAMETERS ---
SC_NUM_SAMPLES = 5
SC_THRESHOLD = 0.5
TOP_K_ENTITIES = 8
TOP_K_RELATIONS = 20
MAX_KNOWLEDGE_TRIPLETS = 40
MAX_KNOWLEDGE_CHARS = 4000
# CONFIG PARAMETERS END---

PROMPTS = get_default_prompts()

PROMPTS["cok_s1_prompt"] = {
    "system": textwrap.dedent(
        """\
        You are provided with a question in the {domain} domain and its query time.
        Strictly follow the format: "First, ... Second, ... The answer is ...".
        Provide two rationales before answering. If you do not know, answer "I don't know".
        """
    ),
    "user": textwrap.dedent(
        """\
        Question: {query}
        Query Time: {query_time}
        Answer:
        """
    ),
}

PROMPTS["cok_s2_edit_prompt"] = {
    "system": textwrap.dedent(
        """\
        You are given a sentence and knowledge from a knowledge graph.
        The sentence may contain factual errors. Correct the sentence using the knowledge.
        If the knowledge does not help, keep the sentence unchanged.
        Return only the edited sentence without extra commentary.
        """
    ),
    "user": textwrap.dedent(
        """\
        Sentence: {sentence}
        Knowledge:
        {knowledge}
        Edited sentence:
        """
    ),
}

PROMPTS["cok_s3_prompt"] = {
    "system": textwrap.dedent(
        """\
        You are given a question and two rationales. Answer the question succinctly.
        If the rationales are insufficient, answer "I don't know".
        Return only the answer.
        """
    ),
    "user": textwrap.dedent(
        """\
        Question: {query}
        Query Time: {query_time}
        Rationales:
        First, {rationale_1}
        Second, {rationale_2}
        Answer:
        """
    ),
}


class CoK_Model:
    """
    Chain-of-Knowledge baseline adapted to use the Bidirection KG.
    """

    def __init__(
        self,
        domain: str = None,
        logger: BaseProgressLogger = DefaultProgressLogger(),
        num_samples: int = SC_NUM_SAMPLES,
        sc_threshold: float = SC_THRESHOLD,
        top_k_entities: int = TOP_K_ENTITIES,
        top_k_relations: int = TOP_K_RELATIONS,
        max_knowledge_triplets: int = MAX_KNOWLEDGE_TRIPLETS,
        max_knowledge_chars: int = MAX_KNOWLEDGE_CHARS,
        step_by_step: bool = True,
        **kwargs,
    ):
        self.name = "cok"
        self.domain = domain
        self.logger = logger
        self.num_samples = num_samples
        self.sc_threshold = sc_threshold
        self.top_k_entities = top_k_entities
        self.top_k_relations = top_k_relations
        self.max_knowledge_triplets = max_knowledge_triplets
        self.max_knowledge_chars = max_knowledge_chars
        self.step_by_step = step_by_step

    def _extract_answer(self, text: str) -> str:
        cleaned = strip_code_fence(text or "").strip()
        maybe_json = maybe_load_json(cleaned, force_load=False)
        if isinstance(maybe_json, dict) and "answer" in maybe_json:
            return str(maybe_json["answer"]).strip()
        marker = "The answer is"
        if marker in cleaned:
            return cleaned.split(marker, 1)[1].strip().strip(".")
        if "Answer:" in cleaned:
            return cleaned.split("Answer:", 1)[1].strip().strip(".")
        return cleaned.strip()

    def _extract_rationales(self, text: str) -> Tuple[str, str]:
        cleaned = strip_code_fence(text or "").strip()
        match = re.search(
            r"First,\s*(.*?)\s*Second,\s*(.*?)(?:\s*The answer is|$)",
            cleaned,
            re.DOTALL,
        )
        if not match:
            return "", ""
        rationale_1 = match.group(1).strip().strip(".")
        rationale_2 = match.group(2).strip().strip(".")
        return rationale_1, rationale_2

    async def _sample_cot(
        self,
        query: str,
        query_time: datetime,
    ) -> str:
        system_prompt = PROMPTS["cok_s1_prompt"]["system"].format(domain=self.domain)
        user_message = PROMPTS["cok_s1_prompt"]["user"].format(
            query=query,
            query_time=query_time,
        )
        response = await generate_response(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=512,
            temperature=0.7,
            logger=self.logger,
        )
        self.logger.debug(system_prompt + "\n" + user_message + "\n" + response)
        return response

    async def _get_cot_sc_results(
        self,
        query: str,
        query_time: datetime,
    ) -> Dict[str, Optional[str]]:
        responses = await asyncio.gather(
            *[self._sample_cot(query, query_time) for _ in range(self.num_samples)]
        )
        answers = [self._extract_answer(r) for r in responses if r]
        answers = [a for a in answers if a]

        if not answers:
            return {
                "cot_response": responses[0] if responses else "",
                "cot_answer": "",
                "cot_sc_score": 0.0,
                "cot_sc_response": responses[0] if responses else "",
                "cot_sc_answer": "",
                "cot_sc_rationales": ("", ""),
            }

        most_common_answer, count = Counter(answers).most_common(1)[0]
        sc_score = float(count) / float(len(answers))
        best_index = next(
            (i for i, r in enumerate(responses) if self._extract_answer(r) == most_common_answer),
            0,
        )
        best_response = responses[best_index] if responses else ""
        rationale_1, rationale_2 = self._extract_rationales(best_response)

        return {
            "cot_response": responses[0] if responses else "",
            "cot_answer": answers[0] if answers else "",
            "cot_sc_score": sc_score,
            "cot_sc_response": best_response,
            "cot_sc_answer": most_common_answer,
            "cot_sc_rationales": (rationale_1, rationale_2),
        }

    async def _retrieve_kg_knowledge(
        self,
        text: str,
    ) -> str:
        embeddings = await generate_embedding([text])
        embedding = embeddings[0] if embeddings else None
        if embedding is None:
            return "None"

        relevant_entities = kg_driver.get_entities(
            embedding=embedding,
            top_k=self.top_k_entities,
            return_score=True,
        )

        entity_lines = []
        relations: List[KGRelation] = []
        for idx, relevant in enumerate(relevant_entities):
            entity_lines.append(
                f"ent_{idx}: {entity_to_text(relevant.entity, include_des=False)}"
            )
            relations.extend(
                kg_driver.get_relations(
                    source=relevant.entity
                )
            )

        seen = set()
        relation_lines = []
        for relation in relations:
            if relation.id in seen:
                continue
            seen.add(relation.id)
            relation_lines.append(
                f"rel_{len(relation_lines)}: {relation_to_text(relation, include_des=False, include_src_des=False, include_dst_des=False)}"
            )
            if len(relation_lines) >= self.max_knowledge_triplets:
                break

        knowledge_sections = []
        if entity_lines:
            knowledge_sections.append("Entities:\n" + "\n".join(entity_lines))
        if relation_lines:
            knowledge_sections.append("Triplets:\n" + "\n".join(relation_lines))
        knowledge = "\n".join(knowledge_sections) if knowledge_sections else "None"
        return knowledge[: self.max_knowledge_chars]

    async def _edit_rationale(
        self,
        sentence: str,
        knowledge: str,
    ) -> str:
        system_prompt = PROMPTS["cok_s2_edit_prompt"]["system"]
        user_message = PROMPTS["cok_s2_edit_prompt"]["user"].format(
            sentence=sentence,
            knowledge=knowledge,
        )
        response = await generate_response(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=256,
            temperature=0,
            logger=self.logger,
        )
        self.logger.debug(system_prompt + "\n" + user_message + "\n" + response)
        return response.strip()

    async def _regenerate_rationale_2(
        self,
        query: str,
        query_time: datetime,
        edited_rationale_1: str,
    ) -> str:
        system_prompt = PROMPTS["cok_s1_prompt"]["system"].format(domain=self.domain)
        user_message = (
            f"Question: {query}\n"
            f"Query Time: {query_time}\n"
            f"Answer: First, {edited_rationale_1} Second, "
        )
        response = await generate_response(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=256,
            temperature=0,
            logger=self.logger,
        )
        self.logger.debug(system_prompt + "\n" + user_message + "\n" + response)
        rationale = response.strip()
        if "Second," in rationale:
            rationale = rationale.split("Second,", 1)[1]
        if "The answer is" in rationale:
            rationale = rationale.split("The answer is", 1)[0]
        return rationale.strip().strip(".")

    async def _consolidate_answer(
        self,
        query: str,
        query_time: datetime,
        rationale_1: str,
        rationale_2: str,
    ) -> str:
        system_prompt = PROMPTS["cok_s3_prompt"]["system"]
        user_message = PROMPTS["cok_s3_prompt"]["user"].format(
            query=query,
            query_time=query_time,
            rationale_1=rationale_1,
            rationale_2=rationale_2,
        )
        response = await generate_response(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=256,
            temperature=0,
            logger=self.logger,
        )
        self.logger.debug(system_prompt + "\n" + user_message + "\n" + response)
        return response.strip()

    async def generate_answer(
        self,
        query: str,
        query_time: datetime = None,
        **kwargs,
    ) -> str:
        results = await self._get_cot_sc_results(query, query_time)
        rationale_1, rationale_2 = results["cot_sc_rationales"]

        if results["cot_sc_score"] >= self.sc_threshold:
            return results["cot_sc_answer"]

        if not rationale_1 or not rationale_2:
            return results["cot_sc_answer"] or "I don't know."

        if self.step_by_step:
            knowledge_1 = await self._retrieve_kg_knowledge(rationale_1)
            edited_rationale_1 = await self._edit_rationale(rationale_1, knowledge_1)
            regenerated_rationale_2 = await self._regenerate_rationale_2(
                query, query_time, edited_rationale_1
            )
            knowledge_2 = await self._retrieve_kg_knowledge(regenerated_rationale_2)
            edited_rationale_2 = await self._edit_rationale(
                regenerated_rationale_2, knowledge_2
            )
        else:
            knowledge_1 = await self._retrieve_kg_knowledge(rationale_1)
            knowledge_2 = await self._retrieve_kg_knowledge(rationale_2)
            edited_rationale_1 = await self._edit_rationale(rationale_1, knowledge_1)
            edited_rationale_2 = await self._edit_rationale(rationale_2, knowledge_2)

        final_answer = await self._consolidate_answer(
            query,
            query_time,
            edited_rationale_1,
            edited_rationale_2,
        )
        return final_answer
