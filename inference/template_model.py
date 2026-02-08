import asyncio
import json
import random
import textwrap
from typing import Any, Dict, List

from inference import *
from kg.kg_driver import *
from kg.kg_rep import *
from utils.prompt_list import *
from utils.utils import *
from utils.logger import *

######################################################################################################
######################################################################################################
###
# Please pay special attention to the comments that start with "TUNE THIS VARIABLE"
# as they depend on your model and the available GPU resources.
###
# DISCLAIMER: This baseline has NOT been tuned for performance
# or efficiency, and is provided as is for demonstration.
######################################################################################################


# Load the environment variable that specifies the URL of the MockAPI. This URL is essential
# for accessing the correct API endpoint in Task 2 and Task 3. The value of this environment variable
# may vary across different evaluation settings, emphasizing the importance of dynamically obtaining
# the API URL to ensure accurate endpoint communication.

# CONFIG PARAMETERS ---

# Define the number of context sentences to consider for generating an answer.
YOUR_PARAMETER_HERE = None

PROMPTS = get_default_prompts()

PROMPTS['your_custom_prompt'] = {
    "system": textwrap.dedent("""\
    You are a helpful assistant.
    """),

    "user": textwrap.dedent("""\
    This is my question.
    """)
}

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

# CONFIG PARAMETERS END---


class Template_Model:
    """
    A model template.
    """

    def __init__(
        self,
        domain: str = None,
        logger: BaseProgressLogger = DefaultProgressLogger(),
        **kwargs
    ):
        self.name = "template"
        self.domain = domain
        self.logger = logger

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

        # Run all requests asynchronously
        response = await generate_response(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=256,
            # DeepSeek-V3 generates endless JSON with json_object enforcement, has to turn it off
            response_format={
                "type": "json_object"} if "deepseek" not in MODEL_NAME.lower() else None,
            logger=self.logger
        )
        self.logger.debug(system_prompt + '\n' +
                          user_message + '\n' + response)

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
        """
        Perform exact match in KG to align a list of topic entity strings of a query to KG entities.

        Args:
            query (str): The query itself.
            topic_entities (List[str]): A list of topic entity strings.
            top_k (int): Specify the top-k entities assessed in KG. Note that the search is based on approximate nearest-neighbor (ANN) search,
                        so, in general, a larger top_k retreive more accurate results.

        Returns:
            List[RelevantEntity]: A list of relevant KG entities with their relevant scores.
        """
        norm_coeff = 1 / len(topic_entities) if len(
            topic_entities) > 0 else 1  # Assuming all the topic entities are equally important
        results = []

        for idx, topic in enumerate(topic_entities):
            exact_match = kg_driver.get_entities(
                name=topic, top_k=1, fuzzy=True)
            results.append(RelevantEntity(exact_match[0], norm_coeff))

        return results

    @llm_retry(max_retries=10, default_output=("I don't know."))
    async def generate_answer(
        self,
        query: str,
        query_time: datetime = None,
        **kwargs
    ):
        topic_entities = await self.extract_entity(query, query_time)
        self.logger.info(f"Extracted topic entities: {topic_entities}")

        topic_entities_scores = await self.align_topic(query, query_time, topic_entities)
        
        ans = ""
        
        # ******** Your code logic goes here ********
        # You can use the following method to interact with the KG:
        # - kg_driver.get_entities()
        # - kg_driver.get_relations()
        # Both of them provide the exact match and vector search approach.

        ans = "I don't know."

        # ******** Your code logic ends here ********

        return ans
