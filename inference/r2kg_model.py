import re
import textwrap
from datetime import datetime
from typing import Dict, List, Tuple

from inference import *
from kg.kg_driver import *
from kg.kg_rep import *
from utils.prompt_list import *
from utils.utils import *
from utils.logger import *

PROMPTS = get_default_prompts()

PROMPTS["r2kg_operator"] = {
    "system": textwrap.dedent("""\
    You are an operator agent that solves a claim by walking a knowledge graph using helper functions.
    Only use these helpers and wait for execution results before continuing:
    1) getRelation[entity]: list relations connected to the entity.
    2) exploreKG[entity]=[rel_1, rel_2]: return triples (entity, relation, neighbor) for the chosen relations. Use ~rel to denote incoming direction if needed.
    3) Verification[answer]: call when you can answer; put the final answer inside the brackets.
    You may chain multiple helpers in one turn separated by '##'. Keep statements brief and rely on the helper outputs to decide next steps.
    """),
    "user": textwrap.dedent("""\
    Claim: {claim}
    Given entities: {entities}
    Start by inspecting the given entities with getRelation before exploring further.
    """),
}

PROMPTS["r2kg_answer"] = {
    "system": "You are an NLP expert in answering questions using provided triples.",
    "user": textwrap.dedent("""\
    Question: {query}
    Query Time: {query_time}
    Entities: {entities}
    Triples:\n{triplets}
    Use only the triples and common sense; if insufficient, say you don't know. Respond with the final answer only.
    """),
}


def _clean_token(token: str) -> str:
    token = token.strip()
    if len(token) >= 2 and ((token[0] == token[-1] == '"') or (token[0] == token[-1] == "'")):
        token = token[1:-1]
    return token.strip()


def _parse_helper_calls(text: str) -> Tuple[List[str], List[Tuple[str, List[str]]], List[str]]:
    """
    Returns:
        relations_calls: list of entity names requested via getRelation
        explore_calls: list of (entity, [relations])
        verifications: list of answers passed to Verification[]
    """
    relations_calls = re.findall(r"getRelation\[(.*?)\]", text)
    explore_matches = re.findall(r"exploreKG\[(.*?)\]\s*=\s*\[(.*?)\]", text)
    verification_calls = re.findall(r"Verification\[(.*?)\]", text)

    relations_calls = [_clean_token(ent) for ent in relations_calls]

    explore_calls: List[Tuple[str, List[str]]] = []
    for ent_raw, rel_raw in explore_matches:
        ent = _clean_token(ent_raw)
        rels = [_clean_token(r) for r in rel_raw.split(",") if r.strip()]
        explore_calls.append((ent, rels))

    verification_calls = [ans.strip() for ans in verification_calls]
    return relations_calls, explore_calls, verification_calls


class R2KG_Model:
    """
    A lightweight R2-KG-style agent using our unified KG driver.
    The operator LLM issues helper calls; we execute them on the Neo4j-backed KG.
    """

    def __init__(
        self,
        domain: str = None,
        iter_limit: int = 10,
        logger: BaseProgressLogger = DefaultProgressLogger(),
        **kwargs,
    ):
        self.name = "r2kg"
        self.domain = domain
        self.iter_limit = iter_limit
        self.logger = logger

    @llm_retry(max_retries=10, default_output=[])
    async def extract_entity(self, query: str, query_time: datetime) -> List[str]:
        system_prompt = PROMPTS["kg_topic_entity"]["system"]
        user_message = PROMPTS["kg_topic_entity"]["user"].format(
            query=query,
            query_time=query_time,
        )

        response = await generate_response(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=256,
            response_format={"type": "json_object"} if "deepseek" not in MODEL_NAME.lower() else None,
            logger=self.logger,
        )
        self.logger.debug(system_prompt + "\n" + user_message + "\n" + response)
        result = maybe_load_json(response)

        entities_list: List[str] = []
        if result.get("domain") == "movie":
            for key in ["movie_name", "person", "year"]:
                val = result.get(key)
                if val:
                    items = val.split(",") if isinstance(val, str) else val
                    entities_list.extend([normalize_entity(str(v)) for v in items])
        elif result.get("domain") == "sports":
            for key in ["tournament", "team"]:
                val = result.get(key)
                if val:
                    items = val.split(",") if isinstance(val, str) else val
                    entities_list.extend([normalize_entity(str(v)) for v in items])
        else:
            val = result.get("main_entity")
            if val:
                items = val.split(",") if isinstance(val, str) else val
                entities_list.extend([normalize_entity(str(v)) for v in items])

        return entities_list

    async def align_topic(self, topic_entities: List[str]) -> List[KGEntity]:
        aligned: List[KGEntity] = []
        for topic in topic_entities:
            match = kg_driver.get_entities(name=topic, top_k=1, fuzzy=True)
            if match:
                aligned.append(match[0])
        return aligned

    def _format_relations(self, entity_name: str, relations: List[str]) -> str:
        rels = "', '".join(relations)
        return f'Relations_list["{entity_name}"] = [\'{rels}\']' if relations else f'Do not change the format of entity {entity_name} in helper function.'

    def _get_relation_names(self, entity: KGEntity) -> List[str]:
        rels = kg_driver.get_relations(source=entity, unique_relation=True)
        names = []
        for rel in rels:
            prefix = "~" if rel.direction == "reverse" else ""
            names.append(f"{prefix}{rel.name}")
        return sorted(set(names))

    def _explore_relations(self, entity: KGEntity, requested: List[str], max_triples: int = 50) -> Tuple[List[List[str]], str]:
        triples: List[List[str]] = []
        all_rels = kg_driver.get_relations(source=entity)
        for rel in all_rels:
            key = f"~{rel.name}" if rel.direction == "reverse" else rel.name
            if key not in requested:
                continue
            head = entity.name
            tail = rel.target.name
            triples.append([head, key, tail])
            if len(triples) >= max_triples:
                break

        msg = ", ".join(str(t) for t in triples) if triples else f"Choose other relations based on Relations_list for {entity.name}"
        return triples, msg

    async def generate_answer(self, query: str, query_time: datetime = None, **kwargs):
        topic_entities = await self.extract_entity(query, query_time)
        aligned_entities = await self.align_topic(topic_entities)
        if not aligned_entities:
            return "I don't know."

        entity_cache: Dict[str, KGEntity] = {(ent.name or "None"): ent for ent in aligned_entities}
        collected_triples: List[List[str]] = []
        final_answer = None

        system_prompt = PROMPTS["r2kg_operator"]["system"]
        user_prompt = PROMPTS["r2kg_operator"]["user"].format(
            claim=query,
            entities=", ".join(entity_cache.keys()),
        )

        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        for _ in range(self.iter_limit):
            response = await generate_response(
                conversation,
                max_tokens=512,
                logger=self.logger,
            )
            self.logger.debug("\n".join([msg["content"] for msg in conversation]) + "\n" + response)

            relations_calls, explore_calls, verification_calls = _parse_helper_calls(response)

            if verification_calls:
                final_answer = verification_calls[-1] if verification_calls[-1] else "I don't know."
                break

            helper_outputs: List[str] = []

            for ent_name in relations_calls:
                ent_name = ent_name or "None"
                ent = entity_cache.get(ent_name)
                if not ent:
                    matches = kg_driver.get_entities(name=ent_name, top_k=1, fuzzy=True)
                    ent = matches[0] if matches else None
                    if ent:
                        entity_cache[ent.name or "None"] = ent
                if ent:
                    rels = self._get_relation_names(ent)
                    helper_outputs.append(self._format_relations(ent.name or None, rels))
                else:
                    helper_outputs.append(f"Do not change the format of entity {ent_name} in helper function.")

            for ent_name, rel_list in explore_calls:
                ent_name = ent_name or "None"
                ent = entity_cache.get(ent_name)
                if not ent:
                    matches = kg_driver.get_entities(name=ent_name, top_k=1, fuzzy=True)
                    ent = matches[0] if matches else None
                    if ent:
                        entity_cache[ent.name or "None"] = ent
                if ent:
                    triples, msg = self._explore_relations(ent, rel_list)
                    collected_triples.extend(triples)
                    for _, _, tail in triples:
                        if tail not in entity_cache:
                            matches = kg_driver.get_entities(name=tail, top_k=1, fuzzy=True)
                            if matches:
                                entity_cache[matches[0].name or "None"] = matches[0]
                    helper_outputs.append(msg)
                else:
                    helper_outputs.append(f"Choose other relations based on Relations_list for {ent_name}")

            if not helper_outputs:
                final_answer = "I don't know."
                break

            conversation.append({"role": "assistant", "content": response})
            conversation.append({"role": "user", "content": "\n".join(helper_outputs)})

        if final_answer:
            return final_answer

        # Fallback: ask model to answer from collected triples
        triplets_str = "\n".join(str(t) for t in collected_triples) if collected_triples else "None"
        system_prompt = PROMPTS["r2kg_answer"]["system"]
        print(entity_cache)
        user_message = PROMPTS["r2kg_answer"]["user"].format(
            query=query,
            query_time=query_time,
            entities=", ".join(entity_cache.keys()),
            triplets=triplets_str,
        )

        response = await generate_response(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=512,
            logger=self.logger,
        )
        self.logger.debug(user_message + "\n" + response)
        return response
