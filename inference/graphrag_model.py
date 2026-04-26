import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from utils import (
    API_BASE,
    API_KEY,
    EMB_API_BASE,
    EMB_API_KEY,
    EMB_CONTEXT_LENGTH,
    EMB_MODEL_NAME,
    MODEL_NAME,
)
from utils.logger import BaseProgressLogger, DefaultProgressLogger
from utils.utils import (
    always_get_an_event_loop,
    generate_embedding,
    get_tokenizer,
)


DEFAULT_CONFIG: Dict[str, Any] = {
    "graphrag_repo_path": None,
    "graphrag_working_dir": None,
    "query_method": "local",
    "response_type": "Single Paragraph",
    "community_level": 2,
    "dynamic_community_selection": False,
    "auto_index": False,
    "force_index": False,
    "llm_model_name": MODEL_NAME,
    "llm_api_base": API_BASE,
    "llm_api_key": API_KEY,
    "embedding_model_name": EMB_MODEL_NAME,
    "embedding_api_base": EMB_API_BASE if EMB_API_BASE else API_BASE,
    "embedding_api_key": EMB_API_KEY if EMB_API_KEY else API_KEY,
    "embedding_dim": None,
    "embedding_max_tokens": EMB_CONTEXT_LENGTH,
    "indexing_method": "standard",
    "concurrent_requests": 32,
    "chunk_size": 1200,
    "chunk_overlap": 100,
    "extract_graph_max_gleanings": 1,
    "entity_types": None,
    "verbose": False,
    "local_top_k_entities": 10,
    "local_top_k_relationships": 10,
    "local_max_context_tokens": 12000,
    "local_text_unit_prop": 0.5,
    "local_community_prop": 0.25,
    "global_max_context_tokens": 12000,
    "global_data_max_tokens": 12000,
    "global_map_max_length": 1000,
    "global_reduce_max_length": 2000,
    "drift_n_depth": 3,
    "drift_k_followups": 20,
    "drift_primer_folds": 5,
    "basic_k": 10,
    "basic_max_context_tokens": 12000,
}


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)


class _HFTokenizerAdapter:
    def __init__(self, hf_tokenizer):
        self._hf_tokenizer = hf_tokenizer

    def encode(self, text: str) -> list[int]:
        return self._hf_tokenizer.encode(text)

    def decode(self, tokens: list[int]) -> str:
        return self._hf_tokenizer.decode(tokens, skip_special_tokens=True)

    def num_tokens(self, text: str) -> int:
        return len(self.encode(text))


class GraphRAG_Model:
    """
    Wrap Microsoft GraphRAG under the benchmark inference interface.

    The lifecycle mirrors LightRAG in this repo:
    1. collect documents through `process_doc`
    2. build the GraphRAG index through `finalize_index`
    3. answer questions through GraphRAG's query API
    """

    def __init__(
        self,
        dataset: str = "",
        domain: str = None,
        config: Optional[Dict[str, Any]] = None,
        logger: BaseProgressLogger = DefaultProgressLogger(),
        **kwargs,
    ):
        self.name = "graphrag"
        self.dataset = dataset
        self.domain = domain
        self.logger = logger

        repo_root = Path(__file__).resolve().parents[1]
        self.config = DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)

        if self.config["graphrag_repo_path"] is None:
            self.config["graphrag_repo_path"] = str(repo_root.parent / "graphrag")
        if self.config["graphrag_working_dir"] is None:
            llm_name = _safe_name(str(self.config.get("llm_model_name", MODEL_NAME)))
            emb_name = _safe_name(
                str(self.config.get("embedding_model_name", EMB_MODEL_NAME))
            )
            self.config["graphrag_working_dir"] = str(
                repo_root / "results" / "graphrag_cache" / dataset / f"{llm_name}_{emb_name}"
            )

        self.repo_path = Path(self.config["graphrag_repo_path"]).expanduser().resolve()
        self.working_dir = Path(self.config["graphrag_working_dir"]).expanduser().resolve()
        self.working_dir.mkdir(parents=True, exist_ok=True)

        self._init_lock = asyncio.Lock()
        self._query_lock = asyncio.Lock()
        self._pending_docs: List[Dict[str, Any]] = []
        self._pending_doc_ids = set()

        self._graphrag_api = None
        self._graphrag_load_config = None
        self._graphrag_indexing_method = None
        self._graphrag_modules_ready = False
        self._graphrag_embedding_patch_applied = False
        self._graphrag_config = None
        self._query_tables: Optional[Dict[str, Optional[pd.DataFrame]]] = None

    async def generate_answer(
        self,
        query: str = "",
        query_time: datetime = None,
        **kwargs,
    ) -> str:
        if not query:
            return ""

        await self._ensure_ready_for_query()

        method = str(self.config.get("query_method", "local")).lower()
        response_type = str(self.config.get("response_type", "Single Paragraph"))
        verbose = bool(self.config.get("verbose", False))

        if method == "local":
            response, _ = await self._graphrag_api.local_search(
                config=self._graphrag_config,
                entities=self._query_tables["entities"],
                communities=self._query_tables["communities"],
                community_reports=self._query_tables["community_reports"],
                text_units=self._query_tables["text_units"],
                relationships=self._query_tables["relationships"],
                covariates=self._query_tables["covariates"],
                community_level=int(self.config.get("community_level", 2)),
                response_type=response_type,
                query=query,
                verbose=verbose,
            )
        elif method == "global":
            response, _ = await self._graphrag_api.global_search(
                config=self._graphrag_config,
                entities=self._query_tables["entities"],
                communities=self._query_tables["communities"],
                community_reports=self._query_tables["community_reports"],
                community_level=int(self.config.get("community_level", 2)),
                dynamic_community_selection=bool(
                    self.config.get("dynamic_community_selection", False)
                ),
                response_type=response_type,
                query=query,
                verbose=verbose,
            )
        elif method == "drift":
            response, _ = await self._graphrag_api.drift_search(
                config=self._graphrag_config,
                entities=self._query_tables["entities"],
                communities=self._query_tables["communities"],
                community_reports=self._query_tables["community_reports"],
                text_units=self._query_tables["text_units"],
                relationships=self._query_tables["relationships"],
                community_level=int(self.config.get("community_level", 2)),
                response_type=response_type,
                query=query,
                verbose=verbose,
            )
        elif method == "basic":
            response, _ = await self._graphrag_api.basic_search(
                config=self._graphrag_config,
                text_units=self._query_tables["text_units"],
                response_type=response_type,
                query=query,
                verbose=verbose,
            )
        else:
            raise ValueError(
                f"Unsupported GraphRAG query_method={method}. "
                "Use one of: local, global, drift, basic."
            )

        return response if isinstance(response, str) else str(response)

    async def process_doc(
        self,
        id: str = "",
        doc: str = "",
        ref: str = "",
        **kwargs,
    ):
        if not doc or id in self._pending_doc_ids:
            return

        metadata = self._extract_doc_metadata(id=id, ref=ref)
        self._pending_doc_ids.add(id)
        self._pending_docs.append(
            {
                "id": id,
                "title": metadata["title"],
                "text": doc,
                "creation_date": metadata["creation_date"],
                "raw_data": metadata["raw_data"],
            }
        )

    def finalize_index(self):
        loop = always_get_an_event_loop()
        loop.run_until_complete(self.afinalize_index())

    async def afinalize_index(self):
        await self._ensure_graphrag_modules()
        await self._ensure_workspace_config()

        if self._index_ready() and not self.config.get("force_index", False):
            self.logger.info(
                f"GraphRAG index already available at {self.working_dir}; skipping rebuild."
            )
            await self._load_query_assets(force_reload=True)
            return

        if not self._pending_docs:
            raise ValueError(
                "No documents collected for GraphRAG indexing. "
                "Provide docs via process_doc before finalize_index()."
            )

        input_documents = pd.DataFrame(self._pending_docs).loc[
            :, ["id", "title", "text", "creation_date", "raw_data"]
        ]
        input_documents["human_readable_id"] = range(len(input_documents))

        self.logger.info(
            f"Building GraphRAG index with {len(input_documents)} documents into {self.working_dir}"
        )

        indexing_method = self._parse_indexing_method(
            self.config.get("indexing_method", "standard")
        )
        outputs = await self._graphrag_api.build_index(
            config=self._graphrag_config,
            method=indexing_method,
            input_documents=input_documents,
            verbose=bool(self.config.get("verbose", False)),
        )
        errors = [output for output in outputs if getattr(output, "error", None) is not None]
        if errors:
            first_error = errors[0].error
            error_msg = (
                f"{type(first_error).__name__}: {first_error}"
                if first_error is not None
                else "unknown error"
            )
            raise RuntimeError(
                "GraphRAG indexing completed with workflow errors. "
                f"First failing workflow: {errors[0].workflow}. "
                f"Underlying error: {error_msg}"
            )

        self._write_manifest(
            {
                "dataset": self.dataset,
                "doc_count": len(input_documents),
                "llm_model_name": self.config.get("llm_model_name", MODEL_NAME),
                "embedding_model_name": self.config.get(
                    "embedding_model_name", EMB_MODEL_NAME
                ),
                "query_method": self.config.get("query_method", "local"),
                "community_level": int(self.config.get("community_level", 2)),
                "indexing_method": str(self.config.get("indexing_method", "standard")),
            }
        )
        self._pending_docs = []
        self._pending_doc_ids = set()
        await self._load_query_assets(force_reload=True)

    async def _ensure_ready_for_query(self):
        await self._ensure_graphrag_modules()
        await self._ensure_workspace_config()

        if self._pending_docs:
            await self.afinalize_index()
            return

        if not self._index_ready():
            raise RuntimeError(
                "GraphRAG index is not ready. Run run/run_graphrag_index.py first "
                "or collect documents and call finalize_index()."
            )

        await self._load_query_assets()

    async def _ensure_graphrag_modules(self):
        if self._graphrag_modules_ready:
            return

        async with self._init_lock:
            if self._graphrag_modules_ready:
                return

            for path in self._graphrag_python_paths():
                if path not in sys.path:
                    sys.path.insert(0, path)

            try:
                import graphrag.api as graphrag_api
                from graphrag.config.enums import IndexingMethod
                from graphrag.config.load_config import load_config
            except Exception as exc:
                raise RuntimeError(
                    "Failed to import GraphRAG. Install the GraphRAG Python dependencies "
                    "for the active environment, or run this baseline inside a GraphRAG-ready env."
                ) from exc

            self._graphrag_api = graphrag_api
            self._graphrag_load_config = load_config
            self._graphrag_indexing_method = IndexingMethod
            self._patch_graphrag_embedding_tokenization()
            self._graphrag_modules_ready = True

    def _graphrag_python_paths(self) -> List[str]:
        packages_dir = self.repo_path / "packages"
        return [
            str(packages_dir / "graphrag"),
            str(packages_dir / "graphrag-cache"),
            str(packages_dir / "graphrag-chunking"),
            str(packages_dir / "graphrag-common"),
            str(packages_dir / "graphrag-input"),
            str(packages_dir / "graphrag-llm"),
            str(packages_dir / "graphrag-storage"),
            str(packages_dir / "graphrag-vectors"),
        ]

    def _patch_graphrag_embedding_tokenization(self):
        if self._graphrag_embedding_patch_applied:
            return

        from graphrag.index.operations.embed_text import run_embed_text as embed_module

        exact_embedding_tokenizer = _HFTokenizerAdapter(
            get_tokenizer(str(self.config.get("embedding_model_name", EMB_MODEL_NAME)))
        )
        embedding_max_tokens = int(
            self.config.get("embedding_max_tokens", EMB_CONTEXT_LENGTH)
        )
        if embedding_max_tokens <= 512:
            exact_max_tokens = max(int(embedding_max_tokens * 0.75), 64)
        else:
            exact_max_tokens = max(embedding_max_tokens - 128, 64)

        original_prepare = embed_module._prepare_embed_texts
        original_create_batches = embed_module._create_text_batches

        def _prepare_embed_texts_exact(
            input: list[str],
            tokenizer,
            batch_max_tokens: int = 8191,
            chunk_overlap: int = 100,
        ) -> tuple[list[str], list[int]]:
            effective_tokens = min(batch_max_tokens, exact_max_tokens)
            overlap = min(chunk_overlap, max(min(effective_tokens // 4, 64), 0))
            sizes: list[int] = []
            snippets: list[str] = []

            for text in input:
                split_texts = embed_module.split_text_on_tokens(
                    text,
                    chunk_size=effective_tokens,
                    chunk_overlap=overlap,
                    encode=exact_embedding_tokenizer.encode,
                    decode=exact_embedding_tokenizer.decode,
                )
                normalized_splits: list[str] = []
                for split_text in split_texts:
                    if not split_text or not split_text.strip():
                        continue
                    token_ids = exact_embedding_tokenizer.encode(split_text)
                    if not token_ids:
                        continue
                    if len(token_ids) > effective_tokens:
                        split_text = exact_embedding_tokenizer.decode(
                            token_ids[:effective_tokens]
                        )
                    if split_text.strip():
                        normalized_splits.append(split_text)
                split_texts = normalized_splits
                sizes.append(len(split_texts))
                snippets.extend(split_texts)

            return snippets, sizes

        def _create_text_batches_exact(
            texts: list[str],
            tokenizer,
            max_batch_size: int,
            max_batch_tokens: int,
        ) -> list[list[str]]:
            effective_max_tokens = min(max_batch_tokens, exact_max_tokens)
            result: list[list[str]] = []
            current_batch: list[str] = []
            current_batch_tokens = 0

            for text in texts:
                if not text or not text.strip():
                    continue

                token_ids = exact_embedding_tokenizer.encode(text)
                if not token_ids:
                    continue
                if len(token_ids) > effective_max_tokens:
                    text = exact_embedding_tokenizer.decode(
                        token_ids[:effective_max_tokens]
                    )
                    token_ids = exact_embedding_tokenizer.encode(text)
                    if not token_ids:
                        continue

                token_count = len(token_ids)
                if current_batch and (
                    len(current_batch) >= max_batch_size
                    or current_batch_tokens + token_count > effective_max_tokens
                ):
                    result.append(current_batch)
                    current_batch = []
                    current_batch_tokens = 0

                current_batch.append(text)
                current_batch_tokens += token_count

            if current_batch:
                result.append(current_batch)

            return result

        embed_module._prepare_embed_texts = _prepare_embed_texts_exact
        embed_module._create_text_batches = _create_text_batches_exact
        self._graphrag_embedding_patch_applied = True

    async def _ensure_workspace_config(self):
        settings_path = self._settings_path()
        if (
            not settings_path.exists()
            or self.config.get("force_index", False)
            or bool(self._pending_docs)
        ):
            self._write_settings()
        if self._graphrag_config is None:
            self._graphrag_config = self._graphrag_load_config(self.working_dir)
            await self._sync_embedding_dimensions()

    async def _load_query_assets(self, force_reload: bool = False):
        if self._query_tables is not None and not force_reload:
            return

        async with self._query_lock:
            if self._query_tables is not None and not force_reload:
                return

            output_dir = self.working_dir / "output"
            tables = {
                "entities": pd.read_parquet(output_dir / "entities.parquet"),
                "communities": pd.read_parquet(output_dir / "communities.parquet"),
                "community_reports": pd.read_parquet(output_dir / "community_reports.parquet"),
                "text_units": pd.read_parquet(output_dir / "text_units.parquet"),
                "relationships": pd.read_parquet(output_dir / "relationships.parquet"),
                "covariates": None,
            }
            covariates_path = output_dir / "covariates.parquet"
            if covariates_path.exists():
                tables["covariates"] = pd.read_parquet(covariates_path)

            self._query_tables = tables

    def _settings_path(self) -> Path:
        return self.working_dir / "settings.json"

    def _manifest_path(self) -> Path:
        return self.working_dir / "bidirection_graphrag_manifest.json"

    def _required_output_files(self) -> List[Path]:
        output_dir = self.working_dir / "output"
        return [
            output_dir / "entities.parquet",
            output_dir / "communities.parquet",
            output_dir / "community_reports.parquet",
            output_dir / "text_units.parquet",
            output_dir / "relationships.parquet",
        ]

    def _index_ready(self) -> bool:
        return self._manifest_path().exists() and all(
            path.exists() for path in self._required_output_files()
        )

    def _write_manifest(self, payload: Dict[str, Any]):
        manifest = {
            **payload,
            "working_dir": str(self.working_dir),
            "created_at": datetime.utcnow().isoformat() + "Z",
        }
        self._manifest_path().write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

    def _write_settings(self):
        self.working_dir.mkdir(parents=True, exist_ok=True)
        (self.working_dir / "input").mkdir(parents=True, exist_ok=True)
        (self.working_dir / "output").mkdir(parents=True, exist_ok=True)
        (self.working_dir / "cache").mkdir(parents=True, exist_ok=True)
        (self.working_dir / "reporting").mkdir(parents=True, exist_ok=True)
        self._graphrag_config = None

        entity_types = self.config.get("entity_types")
        if not entity_types:
            entity_types = ["organization", "person", "geo", "event"]

        llm_api_key = self.config.get("llm_api_key") or "EMPTY"
        embedding_api_key = self.config.get("embedding_api_key") or "EMPTY"
        llm_call_args = self._build_model_call_args(
            api_base=self.config.get("llm_api_base")
        )
        embedding_call_args = self._build_model_call_args(
            api_base=self.config.get("embedding_api_base")
        )

        embedding_max_tokens = int(
            self.config.get("embedding_max_tokens", EMB_CONTEXT_LENGTH)
        )
        # Leave substantial headroom because GraphRAG's tokenizer accounting can
        # diverge from provider-side tokenization, especially for non-OpenAI
        # embedding models exposed through OpenAI-compatible gateways.
        if embedding_max_tokens <= 512:
            safe_embedding_tokens = max(int(embedding_max_tokens * 0.75), 64)
        else:
            safe_embedding_tokens = max(embedding_max_tokens - 128, 64)
        chunk_size = min(int(self.config.get("chunk_size", 1200)), safe_embedding_tokens)
        chunk_overlap = min(
            int(self.config.get("chunk_overlap", 100)),
            max(min(chunk_size // 4, 64), 0),
        )

        settings: Dict[str, Any] = {
            "completion_models": {
                "default_completion_model": {
                    "model_provider": "openai",
                    "model": self.config.get("llm_model_name", MODEL_NAME),
                    "api_key": llm_api_key,
                    "auth_method": "api_key",
                    "call_args": llm_call_args,
                    "retry": {"type": "exponential_backoff"},
                }
            },
            "embedding_models": {
                "default_embedding_model": {
                    "model_provider": "openai",
                    "model": self.config.get("embedding_model_name", EMB_MODEL_NAME),
                    "api_key": embedding_api_key,
                    "auth_method": "api_key",
                    "call_args": embedding_call_args,
                    "retry": {"type": "exponential_backoff"},
                }
            },
            "concurrent_requests": int(self.config.get("concurrent_requests", 32)),
            "input": {"type": "text"},
            "input_storage": {
                "type": "file",
                "base_dir": str((self.working_dir / "input").resolve()),
            },
            "output_storage": {
                "type": "file",
                "base_dir": str((self.working_dir / "output").resolve()),
            },
            "reporting": {
                "type": "file",
                "base_dir": str((self.working_dir / "reporting").resolve()),
            },
            "cache": {
                "type": "json",
                "storage": {
                    "type": "file",
                    "base_dir": str((self.working_dir / "cache").resolve()),
                },
            },
            "vector_store": {
                "type": "lancedb",
                "db_uri": str((self.working_dir / "output" / "lancedb").resolve()),
            },
            "chunking": {
                "type": "tokens",
                "size": chunk_size,
                "overlap": chunk_overlap,
                "encoding_model": "o200k_base",
            },
            "embed_text": {
                "embedding_model_id": "default_embedding_model",
                "batch_max_tokens": safe_embedding_tokens,
            },
            "extract_graph": {
                "completion_model_id": "default_completion_model",
                "entity_types": list(entity_types),
                "max_gleanings": int(
                    self.config.get("extract_graph_max_gleanings", 1)
                ),
            },
            "summarize_descriptions": {
                "completion_model_id": "default_completion_model",
            },
            "community_reports": {
                "completion_model_id": "default_completion_model",
            },
            "local_search": {
                "completion_model_id": "default_completion_model",
                "embedding_model_id": "default_embedding_model",
                "top_k_entities": int(self.config.get("local_top_k_entities", 10)),
                "top_k_relationships": int(
                    self.config.get("local_top_k_relationships", 10)
                ),
                "max_context_tokens": int(
                    self.config.get("local_max_context_tokens", 12000)
                ),
                "text_unit_prop": float(
                    self.config.get("local_text_unit_prop", 0.5)
                ),
                "community_prop": float(
                    self.config.get("local_community_prop", 0.25)
                ),
            },
            "global_search": {
                "completion_model_id": "default_completion_model",
                "max_context_tokens": int(
                    self.config.get("global_max_context_tokens", 12000)
                ),
                "data_max_tokens": int(
                    self.config.get("global_data_max_tokens", 12000)
                ),
                "map_max_length": int(
                    self.config.get("global_map_max_length", 1000)
                ),
                "reduce_max_length": int(
                    self.config.get("global_reduce_max_length", 2000)
                ),
            },
            "drift_search": {
                "completion_model_id": "default_completion_model",
                "embedding_model_id": "default_embedding_model",
                "n_depth": int(self.config.get("drift_n_depth", 3)),
                "drift_k_followups": int(
                    self.config.get("drift_k_followups", 20)
                ),
                "primer_folds": int(self.config.get("drift_primer_folds", 5)),
            },
            "basic_search": {
                "completion_model_id": "default_completion_model",
                "embedding_model_id": "default_embedding_model",
                "k": int(self.config.get("basic_k", 10)),
                "max_context_tokens": int(
                    self.config.get("basic_max_context_tokens", 12000)
                ),
            },
        }

        llm_api_base = self.config.get("llm_api_base")
        if llm_api_base:
            settings["completion_models"]["default_completion_model"]["api_base"] = (
                llm_api_base
            )
        embedding_api_base = self.config.get("embedding_api_base")
        if embedding_api_base:
            settings["embedding_models"]["default_embedding_model"]["api_base"] = (
                embedding_api_base
            )

        self._settings_path().write_text(
            json.dumps(settings, indent=2), encoding="utf-8"
        )

    async def _sync_embedding_dimensions(self):
        detected_dim = self.config.get("embedding_dim")
        if not detected_dim:
            probe = await generate_embedding(["dimension probe"], logger=self.logger)
            if not probe or not probe[0]:
                raise RuntimeError("Failed to infer GraphRAG embedding dimension.")
            detected_dim = len(probe[0])
            self.config["embedding_dim"] = detected_dim

        vector_store = self._graphrag_config.vector_store
        if getattr(vector_store, "vector_size", None) != detected_dim:
            self.logger.info(
                "Syncing GraphRAG vector_store.vector_size from %s to %s",
                getattr(vector_store, "vector_size", None),
                detected_dim,
            )
            vector_store.vector_size = detected_dim

        for schema in vector_store.index_schema.values():
            if getattr(schema, "vector_size", None) != detected_dim:
                schema.vector_size = detected_dim

    def _build_model_call_args(self, api_base: Optional[str] = None) -> Dict[str, Any]:
        call_args: Dict[str, Any] = {}
        extra_headers: Dict[str, str] = {}

        # HybridRAG-Bench commonly routes through IBM's OpenAI-compatible gateway.
        # GraphRAG reaches that endpoint through LiteLLM, so propagate the same
        # custom auth header that the rest of Bidirection uses.
        rits_api_key = os.environ.get("RITS_API_KEY")
        if rits_api_key:
            extra_headers["RITS_API_KEY"] = rits_api_key

        if extra_headers:
            call_args["extra_headers"] = extra_headers

        if api_base and "3scale-apicast" in api_base:
            # Some gateway deployments reject requests without an explicit content type.
            call_args.setdefault("extra_headers", {})
            call_args["extra_headers"].setdefault("Content-Type", "application/json")
            # IBM's OpenAI-compatible embedding gateway rejects `encoding_format=null`.
            # Force the standard numeric response format instead.
            call_args.setdefault("encoding_format", "float")

        return call_args

    def _parse_indexing_method(self, value: Any):
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized == "standard":
                return self._graphrag_indexing_method.Standard
            if normalized == "fast":
                return self._graphrag_indexing_method.Fast
        return value

    def _extract_doc_metadata(self, id: str, ref: str) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {
            "title": id or "untitled",
            "creation_date": None,
            "raw_data": None,
        }
        if not ref:
            return metadata

        try:
            parsed = json.loads(ref)
        except Exception:
            return metadata

        metadata["raw_data"] = parsed
        if isinstance(parsed, dict):
            if isinstance(parsed.get("title"), str):
                metadata["title"] = parsed["title"]
            elif isinstance(parsed.get("name"), str):
                metadata["title"] = parsed["name"]
            elif isinstance(parsed.get("path"), str):
                metadata["title"] = Path(parsed["path"]).stem

            if isinstance(parsed.get("creation_date"), str):
                metadata["creation_date"] = parsed["creation_date"]
            elif isinstance(parsed.get("date"), str):
                metadata["creation_date"] = parsed["date"]

        return metadata
