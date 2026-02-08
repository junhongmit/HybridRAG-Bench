import argparse
from typing import Any, Optional
import json
from pathlib import Path
from typing import Dict, List

from utils.logger import *
from utils.utils import generate_response, strip_code_fence, always_get_an_event_loop

_DEFAULT_MAX_TOKENS = 512
_DEFAULT_TEMPERATURE = 0.0
_DEFAULT_NUM_PARAPHRASES = 3


def _build_system_prompt(num_paraphrases: int) -> str:
    """Construct the system prompt with the requested paraphrase count."""
    placeholders = ",\n".join(
        [f'      "<paraphrase {i}>"' for i in range(1, num_paraphrases + 1)]
    )
    return f"""
You are a helpful assistant. Provide {num_paraphrases} different paraphrases of the
question below that keep the same meaning and can be answered with the same
factual answer. Keep each clear and concise. Return only the rewritten
questions, nothing else.

Output JSON ONLY:

[
  {{
    "paraphrased_question": [
{placeholders}
    ]
  }}
]
"""

USER_PROMPT = """
Now paraphrase the question below:

{text}

Only output JSON.
"""


async def _paraphrase_single_question(question: str, config: Dict[str, Any]) -> List[str] | str:
    """Call the shared LLM client to generate paraphrases for a single question."""
    num_paraphrases = config.get("num_paraphrases", _DEFAULT_NUM_PARAPHRASES)
    prompts = [
        {"role": "system", "content": _build_system_prompt(num_paraphrases)},
        {"role": "user", "content": USER_PROMPT.format(text=question)},
    ]

    max_tokens = config.get("max_tokens") or _DEFAULT_MAX_TOKENS
    temperature = config.get("temperature") or _DEFAULT_TEMPERATURE

    response = await generate_response(
        prompts,
        max_tokens=max_tokens,
        temperature=temperature,
        logger=logger,
    )

    cleaned = strip_code_fence(response)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list) and parsed:
            return parsed[0].get("paraphrased_question", cleaned)
        if isinstance(parsed, dict):
            return parsed.get("paraphrased_question", cleaned)
    except Exception as e:
        logger.error(f"Error parsing JSON: {e}")
    return cleaned


async def generate(
    path: str, 
    config: Optional[Dict[str, Any]] = {},
    logger: Optional[BaseProgressLogger] = None
):
    """
    Read a questions JSON in `input_dir`, paraphrase each question via LLM,
    and write `<original>_paraphrased.json` next to the input.

    Assumes input JSON is a list of objects with at least a `question` field
    (like `simple_w_condition_questions_500.json`).
    """
    logger = logger or DefaultProgressLogger()
    input_path_raw = config.get("source_questions_path") or path
    if not input_path_raw:
        raise ValueError("A source questions path must be provided.")

    input_path = Path(input_path_raw)
    out_path = input_path.with_name(input_path.stem + "_paraphrased.json")
    
    num_paraphrases = config.get("num_paraphrases")
    if not isinstance(num_paraphrases, int) or num_paraphrases < 1:
        raise ValueError("num_paraphrases must be a positive integer")
    config = {**config, "num_paraphrases": num_paraphrases}

    logger.info(f"Use Config: {config}")

    if input_path.is_dir():
        # Pick the first .json file in the directory (common case with one file).
        candidates = sorted(input_path.glob("*.json"))
        if not candidates:
            raise FileNotFoundError(f"No .json file found under {input_path}")
        input_path = candidates[0]
    elif not input_path.is_file():
        raise FileNotFoundError(f"{input_path} is not a file")

    logger.info(f"Loading questions from '{input_path}'")
    with open(input_path, "r", encoding="utf-8") as f:
        data: List[Dict] = json.load(f)

    paraphrased: List[Dict] = []
    total = len(data)
    processed = 0
    current_id = 0

    for item in data:
        q = item.get("question", "")
        if not q:
            continue

        processed += 1
        logger.info(f"Paraphrasing {processed}/{total} (id={item.get('id')})")

        paraphrased_question = await _paraphrase_single_question(q, config)

        paraphrases = (
            paraphrased_question
            if isinstance(paraphrased_question, list)
            else [paraphrased_question]
        )
        for para in paraphrases:
            paraphrased.append(
                {
                    "id": current_id,
                    "question": para,
                    "answer": item.get("answer"),
                }
            )
            current_id += 1
    
        logger.info(f"Writing paraphrased questions to '{out_path}'")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(paraphrased, f, ensure_ascii=False, indent=2)

    return out_path


def main():
    parser = argparse.ArgumentParser(description="Paraphrase questions JSON with an LLM.")
    parser.add_argument(
        "--source-questions-path",
        required=True,
        help="Path to a questions JSON file or a directory containing one.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help=f"Max tokens for paraphrase generation (default: {_DEFAULT_MAX_TOKENS}).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help=f"Temperature for paraphrase generation (default: {_DEFAULT_TEMPERATURE}).",
    )
    parser.add_argument(
        "--num-paraphrases",
        type=int,
        default=_DEFAULT_NUM_PARAPHRASES,
        help=f"Number of paraphrases to generate per question (default: {_DEFAULT_NUM_PARAPHRASES}).",
    )
    args = parser.parse_args()

    config = {
        "source_questions_path": args.source_questions_path,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "num_paraphrases": args.num_paraphrases,
    }

    run_logger = DefaultProgressLogger()
    loop = always_get_an_event_loop()
    out_path = loop.run_until_complete(
        generate(
            path=args.source_questions_path,
            config=config,
            logger=run_logger,
        )
    )
    run_logger.info(f"Paraphrased questions written to '{out_path}'")


if __name__ == "__main__":
    main()
