import argparse
import functools
import os

from dataset import *
from inference import HippoRAG_Model
from utils.logger import KGProgressLogger, BaseProgressLogger
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


def build_loader(dataset: str, config: dict, logger: BaseProgressLogger, processor):
    if dataset.lower() == "movie":
        domain = "movie"
        loader = MovieDatasetLoader(
            os.path.join(DATASET_PATH, "crag_movie_dev.jsonl.bz2"),
            config, "doc", logger,
            processor=functools.partial(processor, domain=domain)
        )
    elif dataset.lower() == "sports":
        domain = "sports"
        loader = SportsDatasetLoader(
            os.path.join(DATASET_PATH, "crag_sports_dev.jsonl.bz2"),
            config, "doc", logger,
            processor=functools.partial(processor, domain=domain)
        )
    elif dataset.lower() == "music":
        domain = "music"
        loader = MusicDatasetLoader(
            os.path.join(DATASET_PATH, "crag_music_dev.jsonl.bz2"),
            config, "doc", logger,
            processor=functools.partial(processor, domain=domain)
        )
    elif dataset.lower() == "arxiv_ai":
        domain = "arxiv AI paper"
        loader = ArxivDatasetLoader(
            os.path.join(DATASET_PATH, "arxiv_AI"),
            config, "doc", logger, 
            processor=functools.partial(processor, domain=domain)
        )
    elif dataset.lower() == "arxiv_cy":
        domain = "arxiv CY paper"
        loader = ArxivDatasetLoader(
            os.path.join(DATASET_PATH, "arxiv_CY"),
            config, "doc", logger, 
            processor=functools.partial(processor, domain=domain)
        )
    elif dataset.lower() == "arxiv_qm":
        domain = "arxiv QM paper"
        loader = ArxivDatasetLoader(
            os.path.join(DATASET_PATH, "arxiv_QM"),
            config, "doc", logger, 
            processor=functools.partial(processor, domain=domain)
        )
    else:
        raise NotImplementedError(f"Dataset {dataset} is not supported.")
    return loader, domain


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset used to index the HippoRAG store")
    parser.add_argument("--num-workers", type=int, default=64,
                        help="Number of workers generating the answers")
    parser.add_argument("--queue-size", type=int, default=64,
                        help="Queue size of data loading")
    parser.add_argument("--progress-path", type=str, default="results/hipporag_{dataset}_index_progress.json",
                        help="Progress log path")
    parser.add_argument('--config', nargs='*', type=parse_key_value,
                        help="Override HippoRAG config as key=value")
    args = parser.parse_args()

    config = {
        "num_workers": args.num_workers,
        "queue_size": args.queue_size,
    }

    hipporag_config = dict(args.config) if args.config else {}
    logger = KGProgressLogger(
        progress_path=args.progress_path.format(dataset=args.dataset))

    async def _noop(**kwargs):
        return None

    loader, domain = build_loader(
        dataset=args.dataset, config=config, logger=logger, processor=_noop)

    model = HippoRAG_Model(dataset=args.dataset,
                           domain=domain,
                           config=hipporag_config,
                           logger=logger)

    loader.processor = functools.partial(model.process_doc, domain=domain)

    loop = always_get_an_event_loop()
    loop.run_until_complete(loader.run())

    logger.info(
        f"Collected {len(model._pending_docs)} docs. Building HippoRAG index...")
    model.finalize_index()

    logger.info("HippoRAG indexing complete ✅")
