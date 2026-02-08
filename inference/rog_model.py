import textwrap
from collections import defaultdict, deque
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

from inference import *
from kg.kg_driver import *
from kg.kg_rep import *
from utils.logger import *
from utils.prompt_list import *
from utils.utils import *

######################################################################################################
######################################################################################################
###
# Please pay special attention to the comments that start with "TUNE THIS VARIABLE"
# as they depend on your model and the available GPU resources.
###
# DISCLAIMER: This baseline has NOT been tuned for performance
# or efficiency, and is provided as is for demonstration.
######################################################################################################

# CONFIG PARAMETERS ---

PROMPTS = get_default_prompts()

# From CRAG Benchmark: https://github.com/facebookresearch/CRAG/blob/main/models/rag_knowledge_graph_baseline.py
PROMPTS['kg_topic_entity'] = {
    "system": textwrap.dedent("""\
    You are given a Query and Query Time. Do the following: 

    1) Determine the domain the query is about. The domain should be one of the following: "sports", "movie", and "other". If none of the domain applies, use "other". Use "domain" as the key in the result json. 

    2) Extract structured information from the query. Include different keys into the result json depending on the domains, amd put them DIRECTLY in the result json. Here are the rules:

    For `movie` queries, these are possible keys:
    - `movie_name`: name of the movie
    - `person`: person name related to moves
    - `year`: if the query is about movies released in a specific year, extract the year

    For `sports` queries, these are possible keys:
    - `sport_type`: one of `basketball`, `soccer`, `other`
    - `tournament`: such as NBA, World Cup, Olympic.
    - `team`: teams that user interested in.
    - `datetime`: time frame that user interested in. When datetime is not explicitly mentioned, use `Query Time` as default. 

    For `other` queries, these are possible keys:
    -  `main_entity`: extract the main entity of the query. 

    Return the results in a FLAT json. 

    *NEVER include ANY EXPLANATION or NOTE in the output, ONLY OUTPUT JSON*
    
    EXAMPLE JSON OUTPUT:
    {"domain": "movie", "movie_name": "Mount Everest"}
    """),

    "user": textwrap.dedent("""\
    Question: {query}
    Query Time: {query_time}
    EXAMPLE JSON OUTPUT:
    {{"domain": "movie", "movie_name": "Mount Everest"}}
    Output:
    """)
}

PROMPTS["rog_plan"] = {
    "system": textwrap.dedent("""\
    You are an expert planner that composes relation paths on a knowledge graph.
    Given a natural language question, the current time, the grounded entities, and
    the relations available around those entities, produce at most {max_paths} promising
    relation paths (1 to {max_hops} hops) that, if traversed from any grounded entity,
    could reveal the answer. Use ONLY the relation names that appear in the provided list.
    Return valid JSON with the following structure:
    {{
        "relation_paths": [
            ["relation_a", "relation_b"],
            ["relation_c"]
        ]
    }}
    If no reasonable path exists, return {{"relation_paths": []}}.
    """),
    "user": textwrap.dedent("""\
    Question: {query}
    Query Time: {query_time}
    Grounded entities:
    {entities}

    Available relation types: {relations}

    Produce up to {max_paths} candidate relation paths.
    """),
}

PROMPTS["rog_answer"] = {
    "system": textwrap.dedent("""\
    You answer questions using faithful reasoning paths grounded in a knowledge graph.
    Read the question and the provided reasoning paths. Use them as the primary evidence
    and explicitly explain how the answer follows from them. If the paths contradict the answer,
    say you cannot answer.
    """),
    "user": textwrap.dedent("""\
    Question: {query}
    Query Time: {query_time}
    Reasoning Paths:
    {reasoning_paths}

    Provide the final answer and a brief explanation that references the paths.
    """),
}

PROMPTS["rog_answer_direct"] = {
    "system": textwrap.dedent("""\
    You are a knowledgeable assistant. No grounded reasoning paths were available,
    so rely on general world knowledge to answer conservatively. If unsure, say you do not know.
    """),
    "user": textwrap.dedent("""\
    Question: {query}
    Query Time: {query_time}
    """),
}


class RoG_Model:
    """
    A lightweight implementation of the Reasoning-on-Graphs baseline that mirrors
    the Think-on-Graph pipeline while following the RoG planning + execution recipe.
    """

    def __init__(
        self,
        domain: str = None,
        config: Optional[Dict] = None,
        logger: BaseProgressLogger = DefaultProgressLogger(),
        **kwargs
    ):
        self.name = "rog"
        self.domain = domain
        self.logger = logger

        config = config or {}
        defaults = {
            "max_subgraph_hops": 2,
            "max_subgraph_edges": 256,
            "max_planner_paths": 4,
            "max_paths_per_rule": 3,
            "max_context_paths": 8,
        }
        self.config = defaults | {key: value for key, value in config.items() if key in defaults}

        self.max_hops = self.config["max_subgraph_hops"]
        self.max_edges = self.config["max_subgraph_edges"]
        self.max_planner_paths = self.config["max_planner_paths"]
        self.max_paths_per_rule = self.config["max_paths_per_rule"]
        self.max_context_paths = self.config["max_context_paths"]

    @staticmethod
    def _format_time(query_time: Optional[datetime]) -> str:
        return query_time.isoformat() if isinstance(query_time, datetime) else "Unknown"

    # ----------------------------------------------------------------------
    # Shared entity extraction utilities (mirrors template/tog implementations)
    # ----------------------------------------------------------------------
    @llm_retry(max_retries=10, default_output=[])
    async def extract_entity(
        self,
        query: str,
        query_time: datetime
    ) -> List[str]:
        system_prompt = PROMPTS["kg_topic_entity"]["system"]
        user_message = PROMPTS["kg_topic_entity"]["user"].format(
            query=query,
            query_time=query_time
        )

        response = await generate_response(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=256,
            response_format={
                "type": "json_object"} if "deepseek" not in MODEL_NAME.lower() else None,
            logger=self.logger
        )
        self.logger.debug(system_prompt + '\n' + user_message + '\n' + response)

        result = maybe_load_json(response)

        entities_list = []
        if result['domain'] == "movie":
            if result.get("movie_name"):
                if isinstance(result["movie_name"], str):
                    movie_names = result["movie_name"].split(",")
                else:
                    movie_names = result["movie_name"]
                for movie_name in movie_names:
                    entities_list.append(normalize_entity(movie_name))
            if result.get("person"):
                if isinstance(result["person"], str):
                    person_list = result["person"].split(",")
                else:
                    person_list = result["person"]
                for person in person_list:
                    entities_list.append(normalize_entity(person))
            if result.get("year"):
                if isinstance(result["year"], str) or isinstance(result["year"], int):
                    years = str(result["year"]).split(",")
                else:
                    years = result["year"]
                for year in years:
                    entities_list.append(normalize_entity(str(year)))
        elif result['domain'] == "sports":
            if result.get("tournament"):
                if isinstance(result["tournament"], str):
                    matches = result["tournament"].split(",")
                else:
                    matches = result["tournament"]
                for match in matches:
                    entities_list.append(normalize_entity(match))
            if result.get("team"):
                if isinstance(result["team"], str):
                    teams = result["team"].split(",")
                else:
                    teams = result["team"]
                for team in teams:
                    entities_list.append(normalize_entity(team))
        elif result['domain'] == "other":
            if result.get("main_entity"):
                if isinstance(result["main_entity"], str):
                    entities = result["main_entity"].split(",")
                else:
                    entities = result["main_entity"]
                for entity in entities:
                    entities_list.append(normalize_entity(entity))

        return entities_list

    @llm_retry(max_retries=10, default_output=[])
    async def align_topic(
        self,
        query: str,
        query_time: datetime,
        topic_entities: List[str]
    ) -> List[RelevantEntity]:
        norm_coeff = 1 / len(topic_entities) if len(topic_entities) > 0 else 1
        results = []

        for topic in topic_entities:
            exact_match = kg_driver.get_entities(
                name=topic, top_k=1, fuzzy=True)
            if not exact_match:
                continue
            results.append(RelevantEntity(exact_match[0], norm_coeff))

        return results

    # ----------------------------------------------------------------------
    # RoG-specific helpers
    # ----------------------------------------------------------------------
    def _entity_list_to_text(self, entities: Sequence[KGEntity]) -> str:
        if not entities:
            return "None"
        return "\n".join(
            f"- {entity.name} ({entity.type})"
            for entity in entities if entity is not None
        )

    def _format_relations_for_prompt(self, relations: Sequence[str]) -> str:
        if not relations:
            return "None"
        return ", ".join(sorted(relations))

    def _build_local_subgraph(self, seeds: List[KGEntity]):
        adjacency: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        relation_pool = set()
        visited = {}
        queue = deque()

        for entity in seeds:
            if entity is None:
                continue
            visited[entity.id] = 0
            queue.append((entity, 0))

        edges = 0
        while queue and edges < self.max_edges:
            current, depth = queue.popleft()
            if depth >= self.max_hops:
                continue

            relations = kg_driver.get_relations(current)
            for relation in relations:
                if relation.source.id != current.id:
                    continue
                target = relation.target
                adjacency[current.name].append((relation.name, target.name))
                relation_pool.add(relation.name)
                edges += 1

                if edges >= self.max_edges:
                    break

                if target.id not in visited and depth + 1 < self.max_hops:
                    visited[target.id] = depth + 1
                    queue.append((target, depth + 1))
            if edges >= self.max_edges:
                break

        return adjacency, relation_pool

    def _bfs_with_rule(self,
                       adjacency: Dict[str, List[Tuple[str, str]]],
                       start_nodes: Sequence[str],
                       rule: Sequence[str]) -> List[List[Tuple[str, str, str]]]:
        if not rule:
            return []

        results = []
        for start in start_nodes:
            queue = deque([(start, [])])
            while queue and len(results) < self.max_paths_per_rule:
                node, path = queue.popleft()
                if len(path) == len(rule):
                    results.append(path)
                    continue

                relation_name = rule[len(path)]
                for rel, neighbor in adjacency.get(node, []):
                    if rel != relation_name:
                        continue
                    queue.append(
                        (neighbor, path + [(node, rel, neighbor)])
                    )
        return results

    def _paths_to_strings(self, paths: List[List[Tuple[str, str, str]]]) -> List[str]:
        result = []
        for path in paths:
            if not path:
                continue
            tokens = []
            for idx, (head, rel, tail) in enumerate(path):
                if idx == 0:
                    tokens.append(f"{head} -> {rel} -> {tail}")
                else:
                    tokens.append(f" -> {rel} -> {tail}")
            result.append("".join(tokens))
        return result

    @llm_retry(max_retries=6, default_output=[])
    async def plan_relation_paths(
        self,
        query: str,
        query_time: datetime,
        entities: List[KGEntity],
        relation_pool: Sequence[str]
    ) -> List[List[str]]:
        if not relation_pool:
            return []

        system_prompt = PROMPTS["rog_plan"]["system"].format(
            max_paths=self.max_planner_paths,
            max_hops=self.max_hops
        )
        user_message = PROMPTS["rog_plan"]["user"].format(
            query=query,
            query_time=query_time,
            entities=self._entity_list_to_text(entities),
            relations=self._format_relations_for_prompt(relation_pool),
            max_paths=self.max_planner_paths
        )

        response = await generate_response(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=512,
            response_format={
                "type": "json_object"} if "deepseek" not in MODEL_NAME.lower() else None,
            logger=self.logger
        )
        self.logger.debug(system_prompt + "\n" + user_message + "\n" + response)

        data = maybe_load_json(response)
        raw_paths = data.get("relation_paths", []) if isinstance(data, dict) else []

        cleaned_paths: List[List[str]] = []
        for path in raw_paths:
            if isinstance(path, str):
                candidate = [segment.strip()
                             for segment in path.split("->") if segment.strip()]
            elif isinstance(path, list):
                candidate = [str(segment).strip()
                             for segment in path if str(segment).strip()]
            else:
                continue
            if not candidate:
                continue
            if any(rel not in relation_pool for rel in candidate):
                continue
            if len(candidate) > self.max_hops:
                candidate = candidate[:self.max_hops]
            cleaned_paths.append(candidate)
            if len(cleaned_paths) >= self.max_planner_paths:
                break

        return cleaned_paths

    def _apply_rules(
        self,
        adjacency: Dict[str, List[Tuple[str, str]]],
        start_entities: Sequence[KGEntity],
        rules: List[List[str]]
    ) -> List[str]:
        start_names = [entity.name for entity in start_entities if entity]
        collected: List[str] = []

        for rule in rules:
            matched = self._bfs_with_rule(adjacency, start_names, rule)
            collected.extend(self._paths_to_strings(matched))
            if len(collected) >= self.max_context_paths:
                break

        return collected[:self.max_context_paths]

    @llm_retry(max_retries=10, default_output="I don't know.")
    async def answer_with_paths(
        self,
        query: str,
        query_time: datetime,
        reasoning_paths: List[str]
    ) -> str:
        reasoning_text = "\n".join(
            [f"path_{idx + 1}: {path}" for idx, path in enumerate(reasoning_paths)]
        ) if reasoning_paths else "None"

        system_prompt = PROMPTS["rog_answer"]["system"]
        user_message = PROMPTS["rog_answer"]["user"].format(
            query=query,
            query_time=query_time,
            reasoning_paths=reasoning_text,
        )

        response = await generate_response(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=512,
            logger=self.logger
        )
        self.logger.debug(user_message + "\n" + response)

        return response

    @llm_retry(max_retries=10, default_output="I don't know.")
    async def generate_without_paths(
        self,
        query: str,
        query_time: datetime
    ) -> str:
        system_prompt = PROMPTS["rog_answer_direct"]["system"]
        user_message = PROMPTS["rog_answer_direct"]["user"].format(
            query=query,
            query_time=query_time,
        )

        response = await generate_response(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=512,
            logger=self.logger
        )
        self.logger.debug(user_message + "\n" + response)
        return response

    # ----------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------
    @llm_retry(max_retries=10, default_output="I don't know.")
    async def generate_answer(
        self,
        query: str,
        query_time: datetime = None,
        **kwargs
    ):
        topic_entities = await self.extract_entity(query, query_time)
        self.logger.info(f"[RoG] Extracted topic entities: {topic_entities}")

        aligned_entities = await self.align_topic(query, query_time, topic_entities)
        if not aligned_entities:
            self.logger.info("[RoG] No aligned entities; falling back to direct answer.")
            return await self.generate_without_paths(query, query_time)

        seed_entities = [entity.entity for entity in aligned_entities if entity.entity]
        adjacency, relation_pool = self._build_local_subgraph(seed_entities)
        if not relation_pool:
            self.logger.info("[RoG] Empty relation pool; falling back to direct answer.")
            return await self.generate_without_paths(query, query_time)

        relation_paths = await self.plan_relation_paths(
            query,
            query_time,
            seed_entities,
            relation_pool
        )

        if not relation_paths:
            self.logger.info("[RoG] Planner returned no relation paths; answering directly.")
            return await self.generate_without_paths(query, query_time)

        reasoning_paths = self._apply_rules(adjacency, seed_entities, relation_paths)
        if not reasoning_paths:
            self.logger.info("[RoG] Unable to ground planner rules; answering directly.")
            return await self.generate_without_paths(query, query_time)

        return await self.answer_with_paths(query, query_time, reasoning_paths)
