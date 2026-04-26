import argparse
import json
import os

from utils.eval import evaluate_predictions
from utils.logger import DefaultProgressLogger, QAProgressLogger
from utils.utils import token_counter


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="arxiv_ai",
        help="Evaluation dataset",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--reeval",
        type=str,
        help="Specify the .json file you want to re-evaluate",
    )
    group.add_argument("--model", type=str, help="Model to evaluate")
    parser.add_argument("--prefix", type=str, help="Prefix added to the result file name")
    parser.add_argument("--postfix", type=str, help="Postfix added to the result file name")
    args = parser.parse_args()

    other_stat = {}
    inf_token_usage = {}
    logger = DefaultProgressLogger()

    if not args.reeval:
        progress_path = (
            f"results/{args.dataset}/"
            f"{args.model}{f'_{args.prefix}' if args.prefix else ''}_{args.dataset}"
            f"_progress{f'_{args.postfix}' if args.postfix else ''}.json"
        )
        result_path = (
            f"results/{args.dataset}/"
            f"{args.model}{f'_{args.prefix}' if args.prefix else ''}_{args.dataset}"
            f"_results{f'_{args.postfix}' if args.postfix else ''}.json"
        )

        logger = QAProgressLogger(progress_path=progress_path)
        if len(logger.progress_data["stats"]) == 0:
            logger.error(f"No progress found for {args.model}_model ❌")
            raise SystemExit(1)

        inf_token_usage = token_counter.get_token_usage()
        token_counter.reset_token_usage()
        results = [
            {
                "id": int(stat["id"]),
                "query": stat["query"],
                "query_time": stat["query_time"],
                "ans": stat["ans"],
                "prediction": stat["prediction"],
                "processing_time": stat["processing_time"],
                "llm_time": stat.get("llm_time"),
                "non_llm_time": stat.get("non_llm_time"),
                "token_usage": stat.get("token_usage", {}),
                "meta": stat.get("meta"),
            }
            for stat in logger.progress_data["stats"]
        ]
    else:
        with open(args.reeval, "r", encoding="utf-8") as f:
            temp_results = json.load(f)

        results = []
        for result in temp_results:
            if "id" in result:
                results.append(result)
            else:
                other_stat = result
                other_stat.pop("eval_llm", None)

        result_path = args.reeval

    results = sorted(results, key=lambda x: x["id"])
    result_dir = os.path.dirname(result_path)
    if result_dir:
        os.makedirs(result_dir, exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    queries = [item["query"] for item in results]
    ground_truths_list = [[str(item["ans"])] for item in results]
    predictions = [str(item["prediction"]) for item in results]

    stats, history = evaluate_predictions(
        queries, ground_truths_list, predictions, "llama", batch_size=128
    )
    logger.info(stats)
    eval_token_usage = token_counter.get_token_usage()
    for key, value in other_stat.items():
        if key not in stats:
            stats[key] = value

    stats.update(
        {
            "inf_prompt_tokens": inf_token_usage.get("prompt_tokens"),
            "inf_completion_tokens": inf_token_usage.get("completion_tokens"),
            "inf_total_tokens": inf_token_usage.get("total_tokens"),
            "eval_prompt_tokens": eval_token_usage.get("prompt_tokens"),
            "eval_completion_tokens": eval_token_usage.get("completion_tokens"),
            "eval_total_tokens": eval_token_usage.get("total_tokens"),
        }
    )
    for idx in range(len(results)):
        results[idx]["score"] = history[idx]["score"]
        results[idx]["explanation"] = history[idx]["explanation"]
    results.insert(0, stats)

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    logger.info(f"Done evaluation on {args.model or 'reeval'} ✅")
    logger.info(f"Token usage: {token_counter.get_token_usage()}")
