import asyncio
from datetime import datetime
from typing import List, Optional, Tuple

from inference import *
from utils.prompt_list import *
from utils.utils import *

PROMPTS = get_default_prompts()

PROMPTS["io_prompt"] = {
    "system": textwrap.dedent(
        """\
        You are provided with a question in the {domain} domain, and its query time. Your task is to answer the question succinctly, using the fewest words possible. 
        If you don't have enough knowledge to answer the question, respond with 'I don't know'.
        """),

    "user": textwrap.dedent(
        """\
        Question: {query}
        Query Time: {query_time}
        """
    )
}


class IO_Model:
    def __init__(
        self,
        domain: str = None,
        logger: BaseProgressLogger = DefaultProgressLogger(),
        **kwargs
    ):
        self.name = "io"
        self.domain = domain
        self.logger = logger

    @llm_retry(max_retries=10, default_output=("I don't know."))
    async def _sample_once(
        self,
        query: str,
        query_time: datetime = None,
        max_tokens: int = 75,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        system_prompt, user_message = self._build_prompt(query, query_time)
        response = await generate_response([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p)
        self.logger.debug(system_prompt + '\n' +
                          user_message + '\n' + response)
        return response

    @llm_retry(max_retries=10, default_output=("I don't know."))
    async def generate_answer(
        self,
        query: str,
        query_time: datetime = None,
        **kwargs
    ) -> str:
        """
        Generates answers for a query using associated (pre-cached) search results and query times.

        Parameters:
            query (str): User queries.
            query_time (str): timestamps (represented as a string), each corresponding to when a query was made.

        Returns:
            str: A plain text responses for each query in the batch. Each response is limited to 75 tokens.
            If the generated response exceeds 75 tokens, it will be truncated to fit within this limit.

        Notes:
        - If the correct answer is uncertain, it's preferable to respond with "I don't know" to avoid
          the penalty for hallucination.
        - Response Time: Ensure that your model processes and responds to each query within 30 seconds.
          Failing to adhere to this time constraint **will** result in a timeout during evaluation.
        """
        system_prompt = PROMPTS["io_prompt"]["system"].format(
            domain=self.domain
        )
        user_message = PROMPTS["io_prompt"]["user"].format(
            query=query,
            query_time=query_time
        )
        response = await generate_response(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=75)
        self.logger.debug(system_prompt + '\n' +
                          user_message + '\n' + response)

        return response

    async def sample_answers(
        self,
        query: str,
        query_time: datetime = None,
        samples: int = 3,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 75,
    ) -> list[str]:
        """
        Draws multiple IO samples to gauge uncertainty.
        """
        tasks = [
            self._sample_once(
                query=query,
                query_time=query_time,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            for _ in range(max(samples, 1))
        ]
        return await asyncio.gather(*tasks)

    def _avg_logprob_from_choice(self, choice) -> Optional[float]:
        """
        Extract average token logprob from an OpenAI chat completion choice.
        Returns None if logprobs are unavailable.
        """
        logprobs = getattr(choice, "logprobs", None)
        if not logprobs or not getattr(logprobs, "content", None):
            return None

        token_logprobs = []
        for block in logprobs.content:
            token_logprob = getattr(block, "logprob", None)
            if token_logprob is not None:
                token_logprobs.append(float(token_logprob))

        if not token_logprobs:
            return None
        print(token_logprobs)
        return sum(token_logprobs) / len(token_logprobs)

    @llm_retry(max_retries=10, default_output=[("I don't know.", None)])
    async def sample_with_logprobs(
        self,
        query: str,
        query_time: datetime = None,
        samples: int = 1,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 75,
    ) -> List[Tuple[str, Optional[float]]]:
        """
        Sample IO answers and return (answer, avg_token_logprob) for each sample.
        If logprobs are not supported by the backend, logprob will be None.
        """
        system_prompt, user_message = self._build_prompt(query, query_time)

        async def _sample():
            raw = await generate_response(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                logprobs=True,
                top_logprobs=5,
                return_raw=True,
            )
            choice = raw.choices[0]
            text = choice.message.content
            avg_lp = self._avg_logprob_from_choice(choice)
            self.logger.debug(system_prompt + '\n' +
                              user_message + '\n' + text + f"\n[avg_logprob={avg_lp}]")
            return text, avg_lp

        tasks = [_sample() for _ in range(max(samples, 1))]
        return await asyncio.gather(*tasks)
