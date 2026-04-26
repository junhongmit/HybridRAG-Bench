import argparse

from kg.kg_updater import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset used to update the KG")
    parser.add_argument("--num-workers", type=int, default=128, help="Number of workers generating the answers")
    parser.add_argument("--queue-size", type=int, default=128, help="Queue size of data loading")
    parser.add_argument("--progress-path", type=str, default="results/update_{dataset}_kg_progress.json", help="Progress log path")
    
    args = parser.parse_args()
    # DATA_PATH = "/Users/junhonglin/Data/arxiv"

    config = {
        "num_workers": args.num_workers,
        "queue_size": args.queue_size,
    }
    logger = KGProgressLogger(progress_path=args.progress_path.format(dataset=args.dataset))
    updater = KG_Updater(config=config, logger=logger)
    if args.dataset.lower() == "arxiv_ai":
        domain = "arxiv AI paper"
        loader = ArxivDatasetLoader(
            os.path.join(DATASET_PATH, "arxiv_AI"),
            config, "doc", logger, 
            processor=functools.partial(updater.process_doc, domain=domain)
        )
    elif args.dataset.lower() == "arxiv_cy":
        domain = "arxiv CY paper"
        loader = ArxivDatasetLoader(
            os.path.join(DATASET_PATH, "arxiv_CY"),
            config, "doc", logger, 
            processor=functools.partial(updater.process_doc, domain=domain)
        )
    elif args.dataset.lower() == "arxiv_qm":
        domain = "arxiv QM paper"
        loader = ArxivDatasetLoader(
            os.path.join(DATASET_PATH, "arxiv_QM"),
            config, "doc", logger, 
            processor=functools.partial(updater.process_doc, domain=domain)
        )
    else:
        raise NotImplementedError(
            f"Dataset {args.dataset} is not supported in the public release. "
            "Use one of: arxiv_ai, arxiv_cy, arxiv_qm."
        )
    print(logger.processed_docs)

    loop = always_get_an_event_loop()
    loop.run_until_complete(
        loader.run()
        # parse_dataset(DATA_PATH, logger=logger)
    )

    logger.info(f"Done updating KG using provided corpus ✅")
    logger.info(f"Token usage: {token_counter.get_token_usage()}")
