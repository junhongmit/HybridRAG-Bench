import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

sys.path.append(os.path.abspath(".."))

from question_gen import QUESTION_GEN_MAP
from utils.logger import BaseProgressLogger, DefaultProgressLogger
from utils.utils import *


EVAL_SYSTEM_PROMPT = """\
You are judging whether a question is answerable by a typical human without seeing the source paragraph.

Mark KEEP if:
- the question is self-contained and unambiguous (no missing references like "this paragraph", "Definition 13" without context),
- a human could reasonably attempt to answer using only the question wording (and optionally the provided answer as a sanity check).

Mark DROP if:
- the question refers to unspecified context, figures, definitions, or paragraphs,
- uses pronouns or deictic words without antecedents,
- is too vague/underspecified to answer.

Always respond with JSON: {"reason": "<short explanation>", "verdict": "keep" | "drop"}.
"""

EVAL_USER_PROMPT = """\
Question: {question}

Provided answer: {answer}

Should this question be kept or dropped? Output JSON as instructed."""


def parse_key_value(arg):
    """Parses key=value string into a (key, value) pair, converting value to int/float if needed."""
    if "=" not in arg:
        raise argparse.ArgumentTypeError("Arguments must be in key=value format")
    key, value = arg.split("=", 1)
    try:
        value = float(value) if "." in value else int(value)
    except ValueError:
        pass
    return key, value


# ---- Dedup helpers ----
def _normalize(text: str) -> str:
    import re

    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _dedup_normalized(data: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    unique = []
    for item in data:
        q_norm = _normalize(item.get("question", ""))
        if q_norm not in seen:
            seen.add(q_norm)
            unique.append(item)
    return unique


def _parse_eval_response(raw: str) -> Tuple[bool, str]:
    """
    Parse the eval model response. Defaults to drop on parse failure.
    """
    try:
        data = json.loads(raw)
        verdict = str(data.get("verdict", "")).lower()
        reason = str(data.get("reason", ""))
    except Exception:
        verdict = "drop"
        reason = raw.strip()[:200]

    keep = verdict.startswith("keep")
    if not reason:
        reason = "No reason provided."
    return keep, reason


async def judge_question(question: str, answer: str,
                         logger: BaseProgressLogger) -> Tuple[bool, str]:
    prompts = [
        {"role": "system", "content": EVAL_SYSTEM_PROMPT},
        {"role": "user", "content": EVAL_USER_PROMPT.format(question=question, answer=answer or "[no answer provided]")},
    ]
    raw = await generate_eval_response(
        prompt=prompts,
        max_tokens=256,
        temperature=0.0,
        logger=logger,
    )
    keep, reason = _parse_eval_response(raw)
    return keep, reason


async def evaluate_file(input_path: str,
                        logger: BaseProgressLogger,
                        suffix: str = "filtered",
                        max_items: Optional[int] = None) -> Optional[str]:
    if not os.path.exists(input_path):
        logger.warning(f"Eval skipped: file not found {input_path}")
        return None

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        logger.warning(f"Eval skipped: expected list in {input_path}")
        return None

    filtered = []
    rejected = []
    for idx, item in enumerate(data):
        if max_items and idx >= max_items:
            filtered.append(item)
            continue

        question = item.get("question", "")
        answer = item.get("answer", "")
        keep, reason = await judge_question(question, answer, logger)
        if keep:
            filtered.append(item)
        else:
            rejected.append({**item, "rejection_reason": reason})
            logger.info(f"Dropped question id={item.get('id')} reason={reason}")

    base, ext = os.path.splitext(input_path)
    filtered_path = f"{base}_{suffix}{ext}"
    rejected_path = f"{base}_{suffix}_rejected{ext}"

    with open(filtered_path, "w", encoding="utf-8") as f:
        json.dump(filtered, f, indent=4, ensure_ascii=False)
    with open(rejected_path, "w", encoding="utf-8") as f:
        json.dump(rejected, f, indent=4, ensure_ascii=False)

    logger.info(
        f"Eval complete for {input_path}: kept {len(filtered)}/{len(data)} "
        f"→ {filtered_path} (rejected logged to {rejected_path})"
    )
    return filtered_path


def deduplicate_file(input_path: str,
                     logger: BaseProgressLogger) -> List[str]:
    if not os.path.exists(input_path):
        logger.warning(f"Dedup skipped: file not found {input_path}")
        return []

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        logger.warning(f"Dedup skipped: expected list in {input_path}")
        return []

    methods = [("normalized", _dedup_normalized)] # More other dedup methods can be added here.

    output_paths = []
    base, ext = os.path.splitext(input_path)
    for suffix, fn in methods:
        deduped = fn(data)
        out_path = f"{base}_dedup_{suffix}{ext}"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(deduped, f, indent=4, ensure_ascii=False)
        logger.info(f"Dedup ({suffix}) complete: kept {len(deduped)}/{len(data)} → {out_path}")
        output_paths.append(out_path)

    return output_paths


async def run_generators(path: str, question_types: List[str],
                         config: Dict,
                         logger: BaseProgressLogger,
                         evaluate: bool,
                         eval_suffix: str,
                         max_eval: Optional[int],
                         skip_dedup: bool):
    for question_type in question_types:
        generator = QUESTION_GEN_MAP[question_type]
        logger.info(f"Starting question generation for type: {question_type}")
        output_path = await generator(path, config=config, logger=logger)
        logger.info(f"Finished question generation for type: {question_type} → {output_path}")

        if evaluate and output_path:
            processed_path = await evaluate_file(output_path, logger=logger, suffix=eval_suffix, max_items=max_eval)
        else:
            processed_path = output_path

        if not skip_dedup and processed_path:
            deduplicate_file(processed_path, logger=logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default=DATASET_PATH,
        help="Directory to store generated question files (default: DATASET_PATH set in env file).")
    parser.add_argument("--question-type", type=str, default="all",
        choices=[*QUESTION_GEN_MAP.keys(), "all"],
        help="Question type to generate or 'all' to run every generator.",
    )
    parser.add_argument("--config", nargs="*",
        type=parse_key_value,
        help="Override generator config as key=value; applied to each selected generator.",
    )
    parser.add_argument("--skip-eval", action="store_true",
        help="Skip LLM-based answerability check (default: run evaluation).")
    parser.add_argument("--eval-suffix", type=str, default="filtered",
        help="Suffix to append to evaluated files (default: filtered).")
    parser.add_argument("--max-eval", type=int, default=None,
        help="Maximum number of items to judge (useful for smoke tests).")
    parser.add_argument("--skip-dedup", action="store_true",
        help="Skip deduplication post-processing (default: run dedup).")
    parser.add_argument("--difficulty", type=str, choices=["regular", "difficult"], default="regular",
        help="Choose 'difficult' to sample high-degree relations, or 'regular' (default).")
    parser.add_argument("--source_questions_path", type=str, default="",
        help="Use source questions instead of generated questions.")
    parser.add_argument("--gen_round", type=int, default=20,
        help="Number of batches to generate questions.")
    parser.add_argument("--temperature", type=float, default=None, 
        help="Temperature for question generation.")
    parser.add_argument("--max_tokens", type=int, default=None, 
        help="Maximum number of tokens to generate for each question generation request.")
    parser.add_argument("--batch_question_input_size", type=int, default=None, 
        help="Query limit for relations used as inputs to question generation.")
    args = parser.parse_args()

    config = dict(args.config) if args.config else {}
    config["batch_question_input_size"] = args.batch_question_input_size
    config["difficulty"] = args.difficulty
    config["gen_round"] = args.gen_round
    config["max_tokens"] = args.max_tokens
    config["source_questions_path"] = args.source_questions_path
    config["temperature"] = args.temperature
    
    selected_types = list(QUESTION_GEN_MAP.keys()) if args.question_type == "all" else [args.question_type]

    logger = DefaultProgressLogger()
    loop = always_get_an_event_loop()
    loop.run_until_complete(
        run_generators(
            args.data_path,
            selected_types,
            config,
            logger,
            not args.skip_eval,
            args.eval_suffix,
            args.max_eval,
            args.skip_dedup
        )
    )
