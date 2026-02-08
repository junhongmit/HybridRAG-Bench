import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from utils.logger import BaseProgressLogger, DefaultProgressLogger


class HippoRAGStore:
    """
    Thin wrapper around the upstream HippoRAG implementation to keep its
    indexing/retrieval logic inside our `kg` namespace.

    The class defers heavy imports to runtime so other models are unaffected
    if HippoRAG is not installed. Use `ensure_index` before calling
    `retrieve`/`rag_qa` if you are not relying on auto indexing.
    """

    def __init__(
        self,
        repo_path: Optional[str] = None,
        save_dir: Optional[str] = None,
        llm_model_name: Optional[str] = None,
        embedding_model_name: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        embedding_base_url: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        azure_embedding_endpoint: Optional[str] = None,
        retrieval_top_k: Optional[int] = None,
        qa_top_k: Optional[int] = None,
        force_reindex: bool = False,
        logger: BaseProgressLogger = DefaultProgressLogger(),
    ):
        repo_path = Path(repo_path).expanduser() if repo_path else None
        if repo_path:
            sys.path.append(str(repo_path))

        try:
            from hipporag import HippoRAG  # type: ignore
            from hipporag.utils.config_utils import BaseConfig  # type: ignore

        except Exception as exc:
            raise ImportError(
                "HippoRAG is required for the hipporag baseline. "
                "Point `repo_path` to the cloned HippoRAG repo or install it with `pip install hipporag`."
            ) from exc

        self.logger = logger
        self.save_dir = str(Path(save_dir).expanduser()
                            ) if save_dir else None

        config = BaseConfig()
        if self.save_dir:
            config.save_dir = self.save_dir
        if llm_base_url:
            config.llm_base_url = llm_base_url
        if llm_model_name:
            config.llm_name = llm_model_name
        if embedding_model_name:
            config.embedding_model_name = embedding_model_name
        if retrieval_top_k:
            config.retrieval_top_k = retrieval_top_k
        if qa_top_k:
            config.qa_top_k = qa_top_k
        config.force_index_from_scratch = force_reindex

        self.engine = HippoRAG(
            global_config=config,
            save_dir=self.save_dir,
            llm_model_name=llm_model_name,
            llm_base_url=llm_base_url,
            embedding_model_name=embedding_model_name,
            embedding_base_url=embedding_base_url,
            azure_endpoint=azure_endpoint,
            azure_embedding_endpoint=azure_embedding_endpoint,
            logger=logger
        )
        self._index_ready = False

    @property
    def index_ready(self) -> bool:
        return self._index_ready

    def load_corpus(self, corpus_path: str) -> List[str]:
        """
        Load a corpus file into a list of strings. Supports JSON/JSONL with either a
        raw list of strings or objects containing one of the keys:
        ['text', 'content', 'doc', 'paragraph', 'chunk'].
        """
        path = Path(corpus_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Corpus file not found: {path}")

        records: List[str] = []
        if path.suffix.lower() == ".jsonl":
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    text = self._extract_text_field(obj)
                    if text:
                        records.append(text)
        elif path.suffix.lower() == ".json":
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, list):
                for item in payload:
                    if isinstance(item, str):
                        records.append(item)
                    else:
                        text = self._extract_text_field(item)
                        if text:
                            records.append(text)
        else:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        records.append(line.strip())

        return records

    def ensure_index(
        self,
        docs: Optional[Sequence[str]] = None,
        force: bool = False,
    ):
        """
        Build the HippoRAG index if nothing is cached on disk. If a saved graph
        already exists, this simply warms up retrieval objects.
        """
        if self._index_ready and not force:
            return

        if not force and self._has_cached_index():
            # Warm up retrieval objects using existing cache
            self.engine.prepare_retrieval_objects()
            self._index_ready = True
            return

        if not docs:
            raise ValueError(
                "No documents provided for indexing and no existing HippoRAG cache found."
            )

        self.logger.info(f"Indexing {len(docs)} documents with HippoRAG.")
        self.engine.index(list(docs))
        self.engine.prepare_retrieval_objects()
        self._index_ready = True

    def retrieve(self, queries: List[str], num_to_retrieve: Optional[int] = None):
        self._check_ready()
        return self.engine.retrieve(queries=queries, num_to_retrieve=num_to_retrieve)

    def rag_qa(self, queries: List[str]):
        self._check_ready()
        return self.engine.rag_qa(queries=queries)

    def _check_ready(self):
        if not self._index_ready:
            raise RuntimeError(
                "HippoRAG index is not ready. Call `ensure_index` with a corpus first."
            )

    def _has_cached_index(self) -> bool:
        """
        Determine whether a previously built HippoRAG index exists on disk.
        This checks for the graph pickle and non-empty embedding stores.
        """
        working_dir = Path(getattr(self.engine, "working_dir", "")).expanduser()
        graph_pickle_attr = getattr(self.engine, "_graph_pickle_filename", None)
        graph_pickle = Path(graph_pickle_attr) if graph_pickle_attr else working_dir / "graph.pickle"

        try:
            chunk_rows = self.engine.chunk_embedding_store.get_all_id_to_rows()
            fact_rows = self.engine.fact_embedding_store.get_all_id_to_rows()
            entity_rows = self.engine.entity_embedding_store.get_all_id_to_rows()
        except Exception:
            return False

        has_embeddings = any([chunk_rows, fact_rows, entity_rows])
        return graph_pickle.exists() and has_embeddings

    @staticmethod
    def _extract_text_field(obj: dict) -> Optional[str]:
        if not isinstance(obj, dict):
            return None
        for key in ["text", "content", "doc", "paragraph", "chunk"]:
            if key in obj and isinstance(obj[key], str):
                return obj[key]
        return None
