import argparse
from datetime import datetime
from copy import deepcopy
import functools
import json
import os
import time

from dataset import *
from inference import *
from utils.eval import evaluate_predictions
from utils.logger import *
from utils.utils import *


def parse_key_value(arg):
    """Parses key=value string into a (key, value) pair, converting value to int/float if needed."""
    if '=' not in arg:
        raise argparse.ArgumentTypeError(
            "Arguments must be in key=value format")
    key, value = arg.split('=', 1)
    try:
        # Try to cast to int or float
        if '.' in value:
            value = float(value)
        else:
            value = int(value)
    except ValueError:
        pass  # Keep as string if it can't be converted
    return key, value


def _snapshot_token_usage():
    return deepcopy(token_counter.get_token_usage()) if token_counter else {}


def _compute_token_usage_delta(start_usage):
    if not token_counter:
        return {}
    end_usage = token_counter.get_token_usage()
    keys = set(start_usage.keys()) | set(end_usage.keys())
    return {key: end_usage.get(key, 0) - start_usage.get(key, 0) for key in keys}


async def generate_prediction(id: str = "",
                              query: str = "",
                              query_time: datetime = None,
                              ans: str = "",
                              logger: BaseProgressLogger = DefaultProgressLogger(),
                              **kwargs):
    start_time = time.perf_counter()
    token_usage_start = _snapshot_token_usage()

    prediction = None
    meta = None
    # Capture per-question token usage and LLM wait intervals to avoid cross-talk in concurrent runs.
    with capture_token_usage() as local_token_usage, capture_llm_intervals() as local_llm_intervals:
        try:
            if getattr(participant_model, "name", "") == "router":
                prediction = await participant_model.generate_answer(
                    query=query,
                    query_time=query_time,
                    return_meta=True,
                    **kwargs
                )
            else:
                prediction = await participant_model.generate_answer(
                    query=query,
                    query_time=query_time,
                    **kwargs
                )
        except TypeError:
            prediction = await participant_model.generate_answer(
                query=query,
                query_time=query_time,
                **kwargs
            )

    if isinstance(prediction, tuple) and len(prediction) == 2:
        prediction, meta = prediction
        try:
            json.dumps(meta)
        except Exception:
            meta = str(meta)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    # Prefer the local per-question counter; fall back to global delta if missing.
    token_usage_delta = local_token_usage or _compute_token_usage_delta(token_usage_start)
    # Compute raw sum and merged (non-overlapping) LLM wait time.
    llm_time_raw = sum((end - start) for start, end in (local_llm_intervals or []))
    merged_llm_time = 0.0
    if local_llm_intervals:
        intervals = sorted(local_llm_intervals, key=lambda x: x[0])
        merged_start, merged_end = intervals[0]
        for s, e in intervals[1:]:
            if s <= merged_end:
                merged_end = max(merged_end, e)
            else:
                merged_llm_time += merged_end - merged_start
                merged_start, merged_end = s, e
        merged_llm_time += merged_end - merged_start
    llm_time = min(merged_llm_time, elapsed_time)
    non_llm_time = max(elapsed_time - llm_time, 0.0)

    logger.add_stat({
        "id": id,
        "query": query,
        "query_time": query_time,
        "ans": ans,
        "prediction": prediction,
        "meta": meta,
        "processing_time": round(elapsed_time, 2),
        "llm_time": round(llm_time, 2),
        "llm_time_raw": round(llm_time_raw, 2),
        "non_llm_time": round(non_llm_time, 2),
        "token_usage": token_usage_delta
    })
    print(len(logger.processed_questions))
    logger.update_progress({"last_question_total": round(elapsed_time, 2)})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        required=True, help="Evaluation dataset")
    parser.add_argument("--model", type=str, required=True,
                        choices=MODEL_MAP.keys(), help="Model to run inference with")
    parser.add_argument("--num-workers", type=int, default=128,
                        help="Number of workers generating the answers")
    parser.add_argument("--queue-size", type=int, default=128,
                        help="Queue size of data loading")
    parser.add_argument("--split", type=int, default=0,
                        help="Dataset split index")
    parser.add_argument("--prefix", type=str,
                        help="Prefix added to the result file name")
    parser.add_argument("--postfix", type=str,
                        help="Postfix added to the result file name")
    parser.add_argument("--keep", action='store_true',
                        help="Keep the progress file")
    parser.add_argument('--config', nargs='*', type=parse_key_value,
                        help="Override model config as key=value")
    args = parser.parse_args()

    config = {
        "num_workers": args.num_workers,
        "queue_size": args.queue_size,
        "split": args.split,
    }

    progress_path = f"results/{args.dataset}/{args.model}{f"_{args.prefix}" if args.prefix else ""}_{args.dataset}_progress{f"_{args.postfix}" if args.postfix else ""}.json"
    result_path = f"results/{args.dataset}/{args.model}{f"_{args.prefix}" if args.prefix else ""}_{args.dataset}_results{f"_{args.postfix}" if args.postfix else ""}.json"
    os.makedirs(f"results/{args.dataset}", exist_ok=True)
    logger = QAProgressLogger(progress_path=progress_path)
    print(logger.processed_questions)

    if args.dataset.lower() == "movie":
        domain = "movie"
        loader = MovieDatasetLoader(
            os.path.join(DATASET_PATH, "crag_movie_dev.jsonl.bz2"),
            config, "qa", logger,
            processor=functools.partial(generate_prediction, logger=logger)
        )
    elif args.dataset.lower() == "movie_2024":
        domain = "movie"
        loader = MovieDatasetLoader(
            os.path.join(DATASET_PATH, "crag_movie_2024_dev.jsonl.bz2"),
            config, "qa", logger,
            processor=functools.partial(generate_prediction, logger=logger)
        )
    elif args.dataset.lower() == "sports":
        domain = "sports"
        loader = SportsDatasetLoader(
            os.path.join(DATASET_PATH, "crag_sports_dev.jsonl.bz2"),
            config, "qa", logger,
            processor=functools.partial(generate_prediction, logger=logger)
        )
    elif args.dataset.lower() == "sports_2024":
        domain = "sports"
        loader = SportsDatasetLoader(
            os.path.join(DATASET_PATH, "crag_sports_2024_dev.jsonl.bz2"),
            config, "qa", logger,
            processor=functools.partial(generate_prediction, logger=logger)
        )
    elif args.dataset.lower() == "music":
        domain = "music"
        loader = MusicDatasetLoader(
            os.path.join(DATASET_PATH, "crag_music_dev.jsonl.bz2"),
            config, "qa", logger,
            processor=functools.partial(generate_prediction, logger=logger)
        )
    elif args.dataset.lower() == "finance":
        domain = "finance"
        loader = ArxivDatasetLoader(
            os.path.join(DATASET_PATH, "finance"),
            config, "qa", logger, 
            processor=functools.partial(generate_prediction, logger=logger)
        )
        # loader = FinanceDatasetLoader(
        #     os.path.join(DATASET_PATH, "crag_finance_dev.jsonl.bz2"),
        #     config, "qa", logger,
        #     processor=functools.partial(generate_prediction, logger=logger)
        # )
    elif args.dataset.lower() == "multitq":
        domain = "open"
        loader = MultiTQDatasetLoader(
            "dataset/MultiTQ",
            config, "qa", logger,
            processor=functools.partial(generate_prediction, logger=logger)
        )
    elif args.dataset.lower() == "timequestions":
        domain = "yearly question"
        extra_config = dict(args.config) if args.config else {}
        loader = TimeQuestionsDatasetLoader(
            "dataset/TimeQuestions",
            config, "qa", extra_config.get("split", "test"), logger,
            processor=functools.partial(generate_prediction, logger=logger)
        )
    elif args.dataset.lower() == "arxiv_ai":
        domain = "arxiv AI paper"
        loader = ArxivDatasetLoader(
            os.path.join(DATASET_PATH, "arxiv_AI"),
            config, "qa", logger, 
            processor=functools.partial(generate_prediction, logger=logger)
        )
    elif args.dataset.lower() == "arxiv_cy":
        domain = "arxiv CY paper"
        loader = ArxivDatasetLoader(
            os.path.join(DATASET_PATH, "arxiv_CY"),
            config, "qa", logger, 
            processor=functools.partial(generate_prediction, logger=logger)
        )
    elif args.dataset.lower() == "arxiv_qm":
        domain = "arxiv QM paper"
        loader = ArxivDatasetLoader(
            os.path.join(DATASET_PATH, "arxiv_QM"),
            config, "qa", logger, 
            processor=functools.partial(generate_prediction, logger=logger)
        )
    else:
        raise NotImplementedError(f"Dataset {args.dataset} is not supported.")

    participant_model = MODEL_MAP[args.model](
        dataset=args.dataset.lower(),
        domain=domain,
        config=dict(args.config) if args.config else None,
        logger=logger
    )

    loop = always_get_an_event_loop()
    loop.run_until_complete(
        loader.run()
    )

    inf_token_usage = token_counter.get_token_usage()
    token_counter.reset_token_usage()
    results = [
        {"id": int(stat["id"]), "query": stat["query"], "query_time": stat["query_time"],
         "ans": stat["ans"], "prediction": stat["prediction"], "processing_time": stat["processing_time"],
         "llm_time": stat.get("llm_time"), "non_llm_time": stat.get("non_llm_time"),
         "token_usage": stat.get("token_usage", {}), "meta": stat.get("meta")}
        for stat in logger.progress_data["stats"]
    ]
    results = sorted(results, key=lambda x: x["id"])
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    queries = [item["query"] for item in results]
    ground_truths_list = [[str(item["ans"])] for item in results]
    predictions = [str(item["prediction"]) for item in results]

    stats, history = evaluate_predictions(
        queries, ground_truths_list, predictions, 'llama', batch_size=128
    )
    eval_token_usage = token_counter.get_token_usage()
    stats.update({
        "inf_prompt_tokens": inf_token_usage.get("prompt_tokens"),
        "inf_completion_tokens": inf_token_usage.get("completion_tokens"),
        "inf_total_tokens": inf_token_usage.get("total_tokens"),
        "eval_prompt_tokens": eval_token_usage.get("prompt_tokens"),
        "eval_completion_tokens": eval_token_usage.get("completion_tokens"),
        "eval_total_tokens": eval_token_usage.get("total_tokens")
    })
    for idx in range(len(results)):
        id = results[idx]['id']
        results[idx]['score'] = history[idx]['score']
        results[idx]['explanation'] = history[idx]['explanation']
    results.insert(0, stats)
    # Save to a JSON file
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    if not args.keep:
        os.remove(progress_path)

    logger.info(
        f"Done inference in {args.dataset} dataset on {args.model}_model ✅")
    logger.info(
        f"Inference token usage: {inf_token_usage}; Eval token usage: {eval_token_usage}")
