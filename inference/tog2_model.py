import asyncio
import json
import random
import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from inference import *  # noqa: F401,F403
from kg.kg_driver import *  # noqa: F401,F403
from kg.kg_rep import *  # noqa: F401,F403
from utils.prompt_list import *  # noqa: F401,F403
from utils.utils import *  # noqa: F401,F403
from utils.logger import *  # noqa: F401,F403


PROMPTS = get_default_prompts()

PROMPTS["tog2_topic_prune"] = {
    "system": textwrap.dedent(
        """\
        You help a question answering agent decide which candidate topic entities
        are worth exploring in a knowledge graph before starting search.

        Always respond in JSON with the following structure:
        {
          "keep": [<entity_id_or_name>, ...],
          "reason": "short explanation (optional)"
        }

        Do not include any additional keys. Provide at least one entity.
        """
    ),
    "user": textwrap.dedent(
        """\
        Question: {query}
        Query Time: {query_time}

        Candidate Topic Entities:
        {entity_descriptions}

        Please return JSON as specified.
        """
    ),
}

PROMPTS["tog2_relation_pruning"] = {
    "system": textwrap.dedent(
        """\
        You are expanding a knowledge graph for question answering.
        Choose up to {width} relations that are most helpful for the next hop.
        Return JSON: {{"selected_relations": {{"rel_0": score, ...}}}}
        Scores must be between 0 and 1 and sum to 1 (use uniform scores if unsure).
        """
    ),
    "user": textwrap.dedent(
        """\
        Question: {query}
        Query Time: {query_time}
        Current Entity: {entity_text}

        Candidate Relations:
        {relation_descriptions}

        JSON Response:
        """
    ),
}

PROMPTS["tog2_relation_combination"] = {
    "system": textwrap.dedent(
        """\
        You are selecting the best relations to explore across multiple entities.
        Return a JSON object on the form:
        {{
          "selected_relations": {{
             "candidate_key": score,
             ...
          }}
        }}
        Scores >= 0.0. You may keep more than {width} relations if useful,
        but prefer the strongest {width}. If no relations look promising,
        return an empty dictionary.
        """
    ),
    "user": textwrap.dedent(
        """\
        Question: {query}
        Query Time: {query_time}

        Candidate Relations Grouped by Entity:
        {grouped_relation_text}

        JSON Response:
        """
    ),
}

PROMPTS["tog2_triplet_pruning"] = {
    "system": textwrap.dedent(
        """\
        You are ranking knowledge graph triplets (entity, relation, entity) for expansion.
        Select triplets that most likely advance the reasoning toward answering the question.
        Respond in JSON:
        {{
          "selected_triplets": {{
             "rel_0": score,
             ...
          }}
        }}
        Scores must be >= 0 and should sum to 1 (normalize if necessary).
        """
    ),
    "user": textwrap.dedent(
        """\
        Question: {query}
        Query Time: {query_time}
        Current Path: {path_hint}

        Candidate Triplets:
        {triplet_descriptions}

        JSON Response:
        """
    ),
}

PROMPTS["tog2_reasoning"] = {
    "system": textwrap.dedent(
        """\
        You are a careful assistant that must decide whether the collected graph
        evidence is sufficient to answer the question.
        Always respond in JSON with keys:
        - "confidence": "yes" or "no" (yes means the answer is well-supported)
        - "answer": the best possible answer string (even if confidence is "no")
        - "explanation": short justification referencing the evidence (optional)
        """
    ),
    "user": textwrap.dedent(
        """\
        Question: {query}
        Query Time: {query_time}

        Explored Triplet Chains:
        {chain_text}

        Retrieved Evidence Sentences:
        {evidence_text}

        JSON Response:
        """
    ),
}

PROMPTS["tog2_generate_direct"] = {
    "system": textwrap.dedent(
        """\
        You are a question answering assistant. Provide your best possible answer
        to the question using your parametric knowledge when graph exploration is not helpful.
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


@dataclass
class RelevantTriplet:
    relation: KGRelation
    score: float
    evidence: List[str] = field(default_factory=list)


def _split_sentences(text: str) -> List[str]:
    if not text:
        return []
    sentences = []
    for chunk in text.replace("\n", " ").split(". "):
        chunk = chunk.strip()
        if chunk:
            sentences.append(chunk)
    return sentences


class ToG2_Model:
    """
    A reproduction of Think-on-Graph 2.0 style model adapted to the
    Bidirection framework (entity extraction from raw questions).
    """

    def __init__(
        self,
        domain: Optional[str] = None,
        logger: BaseProgressLogger = DefaultProgressLogger(),
        width: int = 3,
        depth: int = 3,
        topic_prune: bool = True,
        relation_prune_combination: bool = True,
        num_sents_for_reasoning: int = 10,
        **kwargs: Any,
    ):
        self.name = "tog2"
        self.domain = domain
        self.logger = logger
        self.width = width
        self.depth = depth
        self.topic_prune_enabled = topic_prune
        self.relation_prune_combination_enabled = relation_prune_combination
        self.num_sents_for_reasoning = num_sents_for_reasoning

    @llm_retry(max_retries=10, default_output=[])
    async def extract_entity(
        self,
        query: str,
        query_time: datetime,
    ) -> List[str]:
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
            response_format={"type": "json_object"}
            if "deepseek" not in MODEL_NAME.lower()
            else None,
            logger=self.logger,
        )
        self.logger.debug(system_prompt + "\n" + user_message + "\n" + response)

        result = maybe_load_json(response)
        if not isinstance(result, dict):
            return []

        entities_list: List[str] = []
        domain = result.get("domain")

        def extract_list(value: Any) -> List[str]:
            if isinstance(value, list):
                return [normalize_entity(v) for v in value]
            if isinstance(value, str):
                return [normalize_entity(v.strip()) for v in value.split(",") if v.strip()]
            if isinstance(value, (int, float)):
                return [normalize_entity(str(value))]
            return []

        if domain == "movie":
            for key in ("movie_name", "person", "year"):
                entities_list.extend(extract_list(result.get(key)))
        elif domain == "sports":
            for key in ("tournament", "team", "datetime", "sport_type"):
                entities_list.extend(extract_list(result.get(key)))
        else:
            entities_list.extend(extract_list(result.get("main_entity")))

        # Deduplicate while preserving order
        seen = set()
        unique_entities = []
        for ent in entities_list:
            if ent not in seen:
                seen.add(ent)
                unique_entities.append(ent)
        return unique_entities

    @llm_retry(max_retries=10, default_output=[])
    async def align_topic(
        self,
        query: str,
        query_time: datetime,
        topic_entities: List[str],
    ) -> List[RelevantEntity]:
        norm_coeff = 1 / len(topic_entities) if topic_entities else 1.0
        results: List[RelevantEntity] = []
        for topic in topic_entities:
            matches = kg_driver.get_entities(name=topic, top_k=1, fuzzy=True)
            if not matches:
                continue
            results.append(RelevantEntity(matches[0], norm_coeff))
        return results

    def _format_entity_descriptions(self, entities: List[KGEntity]) -> str:
        lines = []
        for entity in entities:
            desc = entity.description or ""
            lines.append(
                f"- {entity.id or entity.name}: {entity_to_text(entity, include_prop=False)}"
                + (f' | "{desc}"' if desc else "")
            )
        return "\n".join(lines)

    @llm_retry(max_retries=5, default_output=None)
    async def topic_entity_prune(
        self,
        query: str,
        query_time: datetime,
        entities: List[KGEntity],
    ) -> Optional[List[str]]:
        if len(entities) <= self.width:
            return [entity.id or entity.name for entity in entities]

        system_prompt = PROMPTS["tog2_topic_prune"]["system"]
        user_message = PROMPTS["tog2_topic_prune"]["user"].format(
            query=query,
            query_time=query_time,
            entity_descriptions=self._format_entity_descriptions(entities),
        )

        response = await generate_response(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=256,
            response_format={"type": "json_object"}
            if "deepseek" not in MODEL_NAME.lower()
            else None,
            logger=self.logger,
        )
        self.logger.debug(system_prompt + "\n" + user_message + "\n" + response)
        result = maybe_load_json(response)
        if isinstance(result, dict):
            keep = result.get("keep")
            if isinstance(keep, list) and keep:
                return [str(item) for item in keep]
        elif isinstance(result, list):
            return [str(item) for item in result if item]
        return None

    def _format_relation_descriptions(self, relations: Dict[str, KGRelation]) -> str:
        return "\n".join(
            [
                f"{key}: {relation_to_text(rel, include_dst_prop=False, include_dst_des=False)}"
                for key, rel in relations.items()
            ]
        )

    def _format_grouped_relations(
        self, grouped: Dict[str, Dict[str, KGRelation]]
    ) -> str:
        sections = []
        for entity_label, rels in grouped.items():
            if not rels:
                continue
            rel_text = "\n".join(
                [
                    f"  {key}: {relation_to_text(rel, include_dst_prop=False, include_dst_des=False)}"
                    for key, rel in rels.items()
                ]
            )
            sections.append(f"Entity {entity_label}:\n{rel_text}")
        return "\n\n".join(sections)

    @llm_retry(max_retries=5, default_output={})
    async def relation_search_prune(
        self,
        query: str,
        query_time: datetime,
        entity: KGEntity,
        visited_relation_ids: Optional[set] = None,
    ) -> Dict[str, RelevantRelation]:
        relation_list = kg_driver.get_relations(entity, unique_relation=True)
        if not relation_list:
            return {}

        unique_relations: Dict[str, KGRelation] = {}
        for idx, relation in enumerate(relation_list):
            if visited_relation_ids and relation.id in visited_relation_ids:
                continue
            key = f"rel_{idx}"
            # Clone without target details to focus on relation semantics
            clean_relation = KGRelation(
                id=relation.id,
                name=relation.name,
                source=relation.source,
                target=KGEntity(
                    id="",
                    type=relation.target.type,
                    name=relation.target.name,
                    description=relation.target.description,
                ),
                description=relation.description,
                properties=relation.properties,
            )
            unique_relations[key] = clean_relation

        if not unique_relations:
            return {}

        system_prompt = PROMPTS["tog2_relation_pruning"]["system"].format(width=self.width)
        user_message = PROMPTS["tog2_relation_pruning"]["user"].format(
            query=query,
            query_time=query_time,
            entity_text=entity_to_text(entity, include_prop=False),
            relation_descriptions=self._format_relation_descriptions(unique_relations),
        )

        response = await generate_response(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=512,
            response_format={"type": "json_object"}
            if "deepseek" not in MODEL_NAME.lower()
            else None,
            logger=self.logger,
        )
        self.logger.debug(system_prompt + "\n" + user_message + "\n" + response)
        result = maybe_load_json(response)
        if not isinstance(result, dict):
            result = {}

        selected = result.get("selected_relations", {})
        if not isinstance(selected, dict) or not selected:
            # fall back to uniform selection of top-K
            selected = {}
            for key in list(unique_relations.keys())[: self.width]:
                selected[key] = 1.0 / min(len(unique_relations), self.width)

        total_score = sum(float(value) for value in selected.values() if isinstance(value, (int, float)))
        norm = total_score if total_score > 0 else 1.0

        relevant: Dict[str, RelevantRelation] = {}
        for key, score in selected.items():
            if key not in unique_relations:
                continue
            safe_score = float(score) / norm
            relevant[key] = RelevantRelation(unique_relations[key], safe_score)
        return relevant

    @llm_retry(max_retries=5, default_output={})
    async def relation_prune_combination(
        self,
        query: str,
        query_time: datetime,
        grouped_relations: Dict[str, Dict[str, RelevantRelation]],
    ) -> Dict[str, RelevantRelation]:
        if not grouped_relations:
            return {}

        # Flatten for prompt readability
        grouped_clean: Dict[str, Dict[str, KGRelation]] = {
            entity_label: {key: rel.relation for key, rel in relations.items()}
            for entity_label, relations in grouped_relations.items()
            if relations
        }
        if not grouped_clean:
            return {}

        system_prompt = PROMPTS["tog2_relation_combination"]["system"].format(width=self.width)
        user_message = PROMPTS["tog2_relation_combination"]["user"].format(
            query=query,
            query_time=query_time,
            grouped_relation_text=self._format_grouped_relations(grouped_clean),
        )

        response = await generate_response(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=512,
            response_format={"type": "json_object"}
            if "deepseek" not in MODEL_NAME.lower()
            else None,
            logger=self.logger,
        )
        self.logger.debug(system_prompt + "\n" + user_message + "\n" + response)
        result = maybe_load_json(response)
        if not isinstance(result, dict):
            return {}

        selected = result.get("selected_relations", {})
        if not isinstance(selected, dict):
            return {}

        flattened: Dict[str, RelevantRelation] = {}
        total_score = 0.0
        for key, score in selected.items():
            if key not in flattened:
                total_score += float(score) if isinstance(score, (int, float)) else 0.0

        norm = total_score if total_score > 0 else 1.0
        for key, score in selected.items():
            if key not in flattened and isinstance(score, (int, float)):
                # Candidate key format: "{entity_label}::{relation_key}"
                if "::" not in key:
                    continue
                entity_label, rel_key = key.split("::", 1)
                if entity_label not in grouped_relations:
                    continue
                relation_dict = grouped_relations[entity_label]
                if rel_key not in relation_dict:
                    continue
                flattened[key] = RelevantRelation(
                    relation_dict[rel_key].relation,
                    float(score) / norm,
                )

        # If combination prompt fails, fall back to individual top selections
        if not flattened:
            for entity_label, relations in grouped_relations.items():
                for rel_key, rel_val in relations.items():
                    combined_key = f"{entity_label}::{rel_key}"
                    flattened[combined_key] = rel_val

        return flattened

    def _collect_entity_context(self, entity: KGEntity, max_sentences: int = 3) -> List[str]:
        sentences: List[str] = []
        sentences.extend(_split_sentences(entity.paragraph or ""))
        sentences.extend(_split_sentences(entity.description or ""))

        for prop, values in entity.properties.items():
            if prop in RESERVED_KEYS:
                continue
            if isinstance(values, dict):
                sentences.extend(
                    [
                        f"{entity.name} - {prop}: {val}"
                        for val in list(values.keys())[: max(1, max_sentences - len(sentences))]
                    ]
                )
            else:
                sentences.append(f"{entity.name} - {prop}: {values}")
        # Deduplicate while preserving order
        seen = set()
        deduped: List[str] = []
        for sent in sentences:
            if sent in seen:
                continue
            seen.add(sent)
            deduped.append(sent)
            if len(deduped) >= max_sentences:
                break
        return deduped

    def _format_triplet_descriptions(self, triplet_dict: Dict[str, KGRelation]) -> str:
        lines = []
        for key, relation in triplet_dict.items():
            relation_text = relation_to_text(
                relation,
                include_dst_prop=True,
                include_dst_des=True,
                include_src_prop=False,
                include_src_des=False,
            )
            lines.append(f"{key}: {relation_text}")
        return "\n".join(lines)

    @llm_retry(max_retries=5, default_output={})
    async def triplet_prune(
        self,
        query: str,
        query_time: datetime,
        relevant_relation: RelevantRelation,
        triplet_candidates: List[KGRelation],
        path_hint: str,
    ) -> Dict[str, RelevantTriplet]:
        if not triplet_candidates:
            return {}

        triplet_dict: Dict[str, KGRelation] = {
            f"rel_{idx}": triplet for idx, triplet in enumerate(triplet_candidates)
        }

        system_prompt = PROMPTS["tog2_triplet_pruning"]["system"]
        user_message = PROMPTS["tog2_triplet_pruning"]["user"].format(
            query=query,
            query_time=query_time,
            path_hint=path_hint,
            triplet_descriptions=self._format_triplet_descriptions(triplet_dict),
        )

        response = await generate_response(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=512,
            response_format={"type": "json_object"}
            if "deepseek" not in MODEL_NAME.lower()
            else None,
            logger=self.logger,
        )
        self.logger.debug(system_prompt + "\n" + user_message + "\n" + response)
        result = maybe_load_json(response)
        if not isinstance(result, dict):
            result = {}
        selected = result.get("selected_triplets", {})
        if not isinstance(selected, dict) or not selected:
            selected = {}
            for key in list(triplet_dict.keys())[: self.width]:
                selected[key] = 1.0 / min(len(triplet_dict), self.width)

        total_score = sum(float(value) for value in selected.values() if isinstance(value, (int, float)))
        norm = total_score if total_score > 0 else 1.0

        relevant_triplets: Dict[str, RelevantTriplet] = {}
        for key, score in selected.items():
            if key not in triplet_dict:
                continue
            kg_relation = triplet_dict[key]
            safe_score = float(score) / norm
            evidence = self._collect_entity_context(kg_relation.target)
            relevant_triplets[key] = RelevantTriplet(
                relation=kg_relation,
                score=safe_score * relevant_relation.score,
                evidence=evidence,
            )
        return relevant_triplets

    def triplet_sort(
        self,
        total_relevant_triplets: List[RelevantTriplet],
    ) -> Tuple[bool, List[str], List[RelevantTriplet], List[str]]:
        if not total_relevant_triplets:
            return False, [], [], []

        sorted_triplets = sorted(
            total_relevant_triplets,
            key=lambda item: item.score,
            reverse=True,
        )[: self.width]

        cluster_chain = [
            relation_to_text(triplet.relation, include_dst_prop=False)
            for triplet in sorted_triplets
        ]
        evidence_sentences: List[str] = []
        for triplet in sorted_triplets:
            evidence_sentences.extend(triplet.evidence)

        return True, cluster_chain, sorted_triplets, evidence_sentences

    @llm_retry(max_retries=5, default_output={"confidence": "no", "answer": ""})
    async def reasoning(
        self,
        query: str,
        query_time: datetime,
        initial_entities: List[KGEntity],
        cluster_chain_of_entities: List[List[str]],
        evidence_sentences: List[str],
    ) -> Tuple[bool, str]:
        chain_lines: List[str] = []
        for depth, chain in enumerate(cluster_chain_of_entities, start=1):
            if not chain:
                continue
            chain_lines.append(f"Depth {depth}:")
            for item in chain:
                chain_lines.append(f"  - {item}")
        chain_text = "\n".join(chain_lines) if chain_lines else "None"

        limited_evidence = evidence_sentences[: self.num_sents_for_reasoning]
        evidence_text = "\n".join(f"- {sent}" for sent in limited_evidence) or "None"

        system_prompt = PROMPTS["tog2_reasoning"]["system"]
        user_message = PROMPTS["tog2_reasoning"]["user"].format(
            query=query,
            query_time=query_time,
            chain_text=chain_text,
            evidence_text=evidence_text,
        )

        response = await generate_response(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=512,
            response_format={"type": "json_object"}
            if "deepseek" not in MODEL_NAME.lower()
            else None,
            logger=self.logger,
        )
        self.logger.debug(system_prompt + "\n" + user_message + "\n" + response)
        result = maybe_load_json(response)
        if not isinstance(result, dict):
            return False, ""

        confidence = str(result.get("confidence", "")).strip().lower()
        answer = result.get("answer", "")

        return confidence == "yes", answer

    @llm_retry(max_retries=5, default_output="I am not sure.")
    async def generate_only_with_gpt(
        self,
        query: str,
        query_time: datetime,
    ) -> str:
        system_prompt = PROMPTS["tog2_generate_direct"]["system"]
        user_message = PROMPTS["tog2_generate_direct"]["user"].format(
            query=query,
            query_time=query_time,
        )
        response = await generate_response(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=512,
            logger=self.logger,
        )
        self.logger.debug(system_prompt + "\n" + user_message + "\n" + response)
        return response

    @llm_retry(max_retries=5, default_output="I am not sure.")
    async def generate_answer(
        self,
        query: str,
        query_time: datetime = None,
        **kwargs: Any,
    ) -> str:
        query_time = query_time or datetime.now()

        topic_strings = await self.extract_entity(query, query_time)
        self.logger.info(f"Extracted topic strings: {topic_strings}")
        if not topic_strings:
            return await self.generate_only_with_gpt(query, query_time)

        topic_entities_scores = await self.align_topic(query, query_time, topic_strings)
        if not topic_entities_scores:
            return await self.generate_only_with_gpt(query, query_time)

        topic_entities = [rel.entity for rel in topic_entities_scores]
        if self.topic_prune_enabled:
            kept = await self.topic_entity_prune(query, query_time, topic_entities)
            if kept:
                topic_entities_scores = [
                    rel for rel in topic_entities_scores if (rel.entity.id or rel.entity.name) in kept
                ]

        if not topic_entities_scores:
            return await self.generate_only_with_gpt(query, query_time)

        initial_topics = [rel.entity for rel in topic_entities_scores]
        cluster_chain_of_entities: List[List[str]] = []
        evidence_sentences: List[str] = []
        visited_relation_ids: set = set()
        visited_entities: set = {entity.id for entity in initial_topics if entity.id}
        entity_paths: Dict[str, List[str]] = {
            entity.id or entity.name: [entity.name] for entity in initial_topics
        }

        for depth in range(1, self.depth + 1):
            self.logger.info(f"Depth {depth} exploration start.")
            per_entity_relations: Dict[str, Dict[str, RelevantRelation]] = {}
            tasks = []
            for entity_score in topic_entities_scores:
                entity = entity_score.entity
                tasks.append(
                    self.relation_search_prune(
                        query,
                        query_time,
                        entity,
                        visited_relation_ids=visited_relation_ids,
                    )
                )

            relation_results = await asyncio.gather(*tasks)
            for rel_entity, rels in zip(topic_entities_scores, relation_results):
                entity_label = rel_entity.entity.id or rel_entity.entity.name
                per_entity_relations[entity_label] = rels

            if self.relation_prune_combination_enabled:
                combined = await self.relation_prune_combination(
                    query,
                    query_time,
                    per_entity_relations,
                )
                flattened_items = list(combined.values())
            else:
                flattened_items = []
                for entity_label, rels in per_entity_relations.items():
                    flattened_items.extend(rels.values())

            if not flattened_items:
                self.logger.info("No relations selected; falling back to direct answer.")
                return await self.generate_only_with_gpt(query, query_time)

            triplet_tasks = []
            relation_sources: List[RelevantRelation] = []
            path_hints: List[str] = []
            for relation in flattened_items:
                source_id = relation.relation.source.id or relation.relation.source.name
                current_path = entity_paths.get(source_id, [relation.relation.source.name])
                path_hint = " -> ".join(current_path + [relation.relation.name])
                path_hints.append(path_hint)
                relation_sources.append(relation)
                candidate_triplets = kg_driver.get_relations(
                    source=relation.relation.source,
                    relation=relation.relation.name,
                    target_type=relation.relation.target.type or None,
                )

                if len(candidate_triplets) > 120:
                    candidate_triplets = random.sample(candidate_triplets, 120)

                triplet_tasks.append(
                    self.triplet_prune(
                        query,
                        query_time,
                        relation,
                        candidate_triplets,
                        path_hint=path_hint,
                    )
                )

            triplet_results_dicts = await asyncio.gather(*triplet_tasks)
            relevant_triplets: List[RelevantTriplet] = []
            for result_dict in triplet_results_dicts:
                relevant_triplets.extend(result_dict.values())

            if not relevant_triplets:
                self.logger.info("No triplets selected; breaking exploration.")
                break

            flag, chain_of_entities, sorted_triplets, collected_evidence = self.triplet_sort(
                relevant_triplets
            )

            if collected_evidence:
                evidence_sentences.extend(collected_evidence)

            cluster_chain_of_entities.append(chain_of_entities)

            if not flag:
                self.logger.info("Triplet sort indicated no useful expansion; stopping.")
                break

            # Prepare next layer entities
            next_entities: Dict[str, RelevantEntity] = {}
            for triplet in sorted_triplets:
                relation = triplet.relation
                visited_relation_ids.add(relation.id)
                target = relation.target
                target_id = target.id or target.name
                if target_id not in visited_entities:
                    visited_entities.add(target_id)
                    source_id = relation.source.id or relation.source.name
                    parent_path = entity_paths.get(source_id, [relation.source.name])
                    entity_paths[target_id] = parent_path + [relation.name, target.name]
                prev = next_entities.get(target_id)
                if prev:
                    prev.score += triplet.score
                else:
                    next_entities[target_id] = RelevantEntity(target, triplet.score)

            if not next_entities:
                self.logger.info("No new entities discovered; breaking.")
                break

            topic_entities_scores = list(next_entities.values())

            stop, answer = await self.reasoning(
                query,
                query_time,
                initial_topics,
                cluster_chain_of_entities,
                evidence_sentences,
            )
            if stop:
                self.logger.info(f"Stopped at depth {depth} with confident answer.")
                return answer

        self.logger.info("Falling back to parametric answer generation.")
        return await self.generate_only_with_gpt(query, query_time)
