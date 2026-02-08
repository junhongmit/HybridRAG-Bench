import asyncio
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from kg.hipporag_store import HippoRAGStore
from utils.logger import BaseProgressLogger, DefaultProgressLogger

# Defaults are deliberately conservative to avoid triggering heavy indexing
# automatically. Override via `--config key=value` when running `run/run_qa.py`.
DEFAULT_CONFIG: Dict[str, Any] = {
    "hipporag_repo_path": "~/LLM/HippoRAG/src",          # Path to cloned HippoRAG repo; defaults to ../HippoRAG
    "hipporag_save_dir": None,           # Where HippoRAG persists its graph/embeddings
    "corpus_path": None,                 # Path to corpus used for indexing
    "auto_index": False,                 # Set True to build the HippoRAG index on startup
    "force_index": False,                # Rebuild even if cache exists
    "llm_model_name": os.getenv("MODEL_NAME", "gpt-4o-mini"),
    "llm_base_url": os.getenv("API_BASE"),
    "embedding_model_name": os.getenv("EMB_MODEL_NAME", "nvidia/NV-Embed-v2"),
    "embedding_base_url": os.getenv("EMB_API_BASE"),
    "azure_endpoint": os.getenv("HIPPORAG_AZURE_ENDPOINT"),
    "azure_embedding_endpoint": os.getenv("HIPPORAG_AZURE_EMBED_ENDPOINT"),
    "retrieval_top_k": 50,
    "qa_top_k": 5,
}


class HippoRAG_Model:
    """
    Wraps the upstream HippoRAG 2 codebase inside our inference interface.

    Usage:
        python run/run_qa.py --model hipporag --dataset movie --config corpus_path=/path/to/docs auto_index=True
    """

    def __init__(
        self,
        dataset: str = "",
        domain: str = None,
        config: Optional[Dict[str, Any]] = None,
        logger: BaseProgressLogger = DefaultProgressLogger(),
        **kwargs
    ):
        self.name = "hipporag"
        self.domain = domain
        self.logger = logger

        repo_root = Path(__file__).resolve().parents[1]

        self.config = DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)

        # Fill in defaults after user overrides to allow custom absolute paths.
        if self.config["hipporag_repo_path"] is None:
            self.config["hipporag_repo_path"] = str(
                repo_root.parent / "HippoRAG")
        if self.config["hipporag_save_dir"] is None:
            self.config["hipporag_save_dir"] = str(
                repo_root / "results" / "hipporag_cache" / dataset)

        self.store = HippoRAGStore(
            repo_path=self.config["hipporag_repo_path"],
            save_dir=self.config["hipporag_save_dir"],
            llm_model_name=self.config["llm_model_name"],
            embedding_model_name=self.config["embedding_model_name"],
            llm_base_url=self.config.get("llm_base_url"),
            embedding_base_url=self.config.get("embedding_base_url"),
            azure_endpoint=self.config.get("azure_endpoint"),
            azure_embedding_endpoint=self.config.get(
                "azure_embedding_endpoint"),
            retrieval_top_k=self.config.get("retrieval_top_k"),
            qa_top_k=self.config.get("qa_top_k"),
            force_reindex=self.config.get("force_index", False),
            logger=self.logger,
        )

        if self.config.get("auto_index"):
            docs = self._maybe_load_corpus()
            if docs:
                self.store.ensure_index(
                    docs, force=self.config.get("force_index", False))
            else:
                self.logger.warning(
                    "auto_index=True but no corpus_path was provided; skipping indexing.")

        self._pending_docs: List[str] = []
        self._pending_doc_ids = set()

    async def generate_answer(
        self,
        query: str = "",
        query_time: datetime = None,
        **kwargs
    ) -> str:
        """
        Runs HippoRAG retrieval + QA. If the index is not ready and auto_index
        is enabled, the corpus will be indexed before answering.
        """

        if not self.store.index_ready:
            docs: List[str] = []
            if self.config.get("auto_index"):
                docs = self._maybe_load_corpus()
            try:
                self.store.ensure_index(
                    docs if docs else None,
                    force=self.config.get("force_index", False)
                )
            except ValueError as exc:
                raise RuntimeError(
                    "HippoRAG index is not ready. Run run/run_hipporag_index.py first "
                    "or enable auto_index with a valid corpus_path."
                ) from exc

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._run_sync, query)

    def _run_sync(self, query: str) -> str:
        results = self.store.rag_qa([query])
        # rag_qa returns (List[QuerySolution], all_response_message, all_metadata)
        query_solutions = results[0]
        if query_solutions and getattr(query_solutions[0], "answer", None):
            return query_solutions[0].answer
        return ""

    async def process_doc(
        self,
        id: str = "",
        doc: str = "",
        **kwargs
    ):
        """
        Collect documents for offline HippoRAG indexing (used by run_hipporag_index.py).
        """
        if not doc or id in self._pending_doc_ids:
            return
        self._pending_doc_ids.add(id)
        self._pending_docs.append(doc)

    def finalize_index(self):
        """
        Index all collected documents and prepare retrieval objects.
        """
        if self.store.index_ready and not self.config.get("force_index", False):
            return
        if not self._pending_docs:
            raise ValueError(
                "No documents collected for HippoRAG indexing. "
                "Provide docs via process_doc or set auto_index with corpus_path."
            )
        self.store.ensure_index(
            self._pending_docs, force=self.config.get("force_index", False))
        self._pending_docs = []
        self._pending_doc_ids = set()

    def _maybe_load_corpus(self) -> List[str]:
        corpus_path = self.config.get("corpus_path")
        if not corpus_path:
            return []
        try:
            return self.store.load_corpus(corpus_path)
        except Exception as exc:
            self.logger.warning(
                f"Failed to load corpus from {corpus_path}: {exc}")
            return []
