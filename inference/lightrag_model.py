import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import networkx as nx

from utils import EMB_CONTEXT_LENGTH, EMB_MODEL_NAME, MODEL_NAME
from utils.logger import BaseProgressLogger, DefaultProgressLogger
from utils.utils import always_get_an_event_loop, generate_embedding, generate_response


DEFAULT_CONFIG: Dict[str, Any] = {
    "lightrag_repo_path": None,
    "lightrag_working_dir": None,
    "query_mode": "mix",
    "response_type": "Single Paragraph",
    "top_k": 60,
    "chunk_top_k": 20,
    "max_entity_tokens": 6000,
    "max_relation_tokens": 8000,
    "max_total_tokens": 30000,
    "enable_rerank": True,
    "include_references": False,
    "auto_index": False,
    "force_index": False,
    "corpus_path": None,
    "llm_max_tokens": 2048,
    "llm_temperature": 0.1,
    "embedding_dim": None,
    "embedding_model_name": EMB_MODEL_NAME,
    "llm_model_name": MODEL_NAME,
    "llm_model_max_async": 8,
    "embedding_func_max_async": 8,
    "max_parallel_insert": 8,
    "chunk_token_size": 1200,
    "chunk_overlap_token_size": 100,
    "kv_storage": "JsonKVStorage",
    "vector_storage": "NanoVectorDBStorage",
    "graph_storage": "NetworkXStorage",
    "doc_status_storage": "JsonDocStatusStorage",
}


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)


class LightRAG_Model:
    """
    Wrap the upstream LightRAG core under the benchmark inference interface.

    This follows the same index lifecycle used by HippoRAG in this codebase:
    documents are collected via `process_doc`, persisted via `finalize_index`,
    and then queried natively through LightRAG.
    """

    def __init__(
        self,
        dataset: str = "",
        domain: str = None,
        config: Optional[Dict[str, Any]] = None,
        logger: BaseProgressLogger = DefaultProgressLogger(),
        **kwargs,
    ):
        self.name = "lightrag"
        self.dataset = dataset
        self.domain = domain
        self.logger = logger

        repo_root = Path(__file__).resolve().parents[1]

        self.config = DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)

        if self.config["lightrag_repo_path"] is None:
            self.config["lightrag_repo_path"] = str(repo_root.parent / "LightRAG")
        if self.config["lightrag_working_dir"] is None:
            llm_name = _safe_name(str(self.config.get("llm_model_name", MODEL_NAME)))
            emb_name = _safe_name(
                str(self.config.get("embedding_model_name", EMB_MODEL_NAME))
            )
            self.config["lightrag_working_dir"] = str(
                repo_root / "results" / "lightrag_cache" / dataset / f"{llm_name}_{emb_name}"
            )

        self.repo_path = Path(self.config["lightrag_repo_path"]).expanduser().resolve()
        self.working_dir = Path(self.config["lightrag_working_dir"]).expanduser().resolve()
        self.working_dir.mkdir(parents=True, exist_ok=True)

        self._rag = None
        self._QueryParam = None
        self._init_lock = asyncio.Lock()
        self._pending_docs: List[Dict[str, Any]] = []
        self._pending_doc_ids = set()

    async def generate_answer(
        self,
        query: str = "",
        query_time: datetime = None,
        **kwargs,
    ) -> str:
        if not query:
            return ""

        await self._ensure_ready_for_query()

        query_param = self._QueryParam(
            mode=self.config.get("query_mode", "mix"),
            response_type=self.config.get("response_type", "Single Paragraph"),
            top_k=int(self.config.get("top_k", 60)),
            chunk_top_k=int(self.config.get("chunk_top_k", 20)),
            max_entity_tokens=int(self.config.get("max_entity_tokens", 6000)),
            max_relation_tokens=int(self.config.get("max_relation_tokens", 8000)),
            max_total_tokens=int(self.config.get("max_total_tokens", 30000)),
            enable_rerank=bool(self.config.get("enable_rerank", True)),
            include_references=bool(self.config.get("include_references", False)),
        )

        response = await self._rag.aquery(query, param=query_param)
        return response if isinstance(response, str) else ""

    async def process_doc(
        self,
        id: str = "",
        doc: str = "",
        ref: str = "",
        **kwargs,
    ):
        if not doc or id in self._pending_doc_ids:
            return

        file_path = self._extract_file_path(id=id, ref=ref)
        self._pending_doc_ids.add(id)
        self._pending_docs.append(
            {
                "id": id,
                "doc": doc,
                "file_path": file_path,
            }
        )

    def finalize_index(self):
        loop = always_get_an_event_loop()
        loop.run_until_complete(self.afinalize_index())

    async def afinalize_index(self):
        await self._ensure_rag()

        if self.config.get("force_index", False):
            self.logger.info(
                f"Force rebuild requested for LightRAG working dir {self.working_dir}."
            )
        elif self._index_ready():
            resumed = await self._resume_incomplete_docs_if_any()
            if resumed:
                self._pending_docs = []
                self._pending_doc_ids = set()
                self._write_manifest(
                    {
                        "dataset": self.dataset,
                        "doc_count": await self._count_indexed_docs(),
                        "query_mode": self.config.get("query_mode", "mix"),
                        "llm_model_name": self.config.get("llm_model_name", MODEL_NAME),
                        "embedding_model_name": self.config.get(
                            "embedding_model_name", EMB_MODEL_NAME
                        ),
                    }
                )
                return
            self.logger.info(
                f"LightRAG index already available at {self.working_dir}; skipping rebuild."
            )
            return
        elif self._has_existing_doc_status_store():
            resumed = await self._resume_incomplete_docs_if_any()
            if resumed:
                self.logger.info(
                    "Detected existing LightRAG doc_status state without a completed manifest; "
                    "resumed the existing queue instead of re-enqueuing documents."
                )
                self._pending_docs = []
                self._pending_doc_ids = set()
                self._write_manifest(
                    {
                        "dataset": self.dataset,
                        "doc_count": await self._count_indexed_docs(),
                        "query_mode": self.config.get("query_mode", "mix"),
                        "llm_model_name": self.config.get("llm_model_name", MODEL_NAME),
                        "embedding_model_name": self.config.get(
                            "embedding_model_name", EMB_MODEL_NAME
                        ),
                    }
                )
                return

        if not self._pending_docs:
            if self.config.get("auto_index"):
                self._pending_docs = self._maybe_load_corpus()
                self._pending_doc_ids = {doc["id"] for doc in self._pending_docs}

            if not self._pending_docs:
                raise ValueError(
                    "No documents collected for LightRAG indexing. "
                    "Provide docs via process_doc or set auto_index with corpus_path."
                )

        docs = [item["doc"] for item in self._pending_docs]
        ids = [item["id"] for item in self._pending_docs]
        file_paths = [item["file_path"] for item in self._pending_docs]

        self.logger.info(
            f"Building LightRAG index with {len(docs)} documents into {self.working_dir}"
        )
        await self._rag.ainsert(docs, ids=ids, file_paths=file_paths)
        self._write_manifest(
            {
                "dataset": self.dataset,
                "doc_count": len(docs),
                "query_mode": self.config.get("query_mode", "mix"),
                "llm_model_name": self.config.get("llm_model_name", MODEL_NAME),
                "embedding_model_name": self.config.get(
                    "embedding_model_name", EMB_MODEL_NAME
                ),
            }
        )
        self._pending_docs = []
        self._pending_doc_ids = set()

    async def _ensure_ready_for_query(self):
        await self._ensure_rag()

        if self._pending_docs:
            await self.afinalize_index()
            return

        if self._index_ready():
            return

        if self.config.get("auto_index"):
            await self.afinalize_index()
            return

        raise RuntimeError(
            "LightRAG index is not ready. Run run/run_lightrag_index.py first "
            "or enable auto_index with a valid corpus_path."
        )

    async def _ensure_rag(self):
        if self._rag is not None:
            return

        async with self._init_lock:
            if self._rag is not None:
                return

            lightrag_root = str(self.repo_path)
            if lightrag_root not in sys.path:
                sys.path.insert(0, lightrag_root)

            try:
                from lightrag import LightRAG, QueryParam
                from lightrag.base import DocStatus
                from lightrag.utils import wrap_embedding_func_with_attrs
            except Exception as exc:
                raise RuntimeError(
                    "Failed to import LightRAG. Ensure the local LightRAG repo and its "
                    "Python dependencies are available."
                ) from exc

            embedding_dim = self.config.get("embedding_dim")
            if embedding_dim is None:
                embedding_dim = await self._infer_embedding_dim()
                self.config["embedding_dim"] = embedding_dim

            async def lightrag_llm_model_func(
                prompt: str,
                system_prompt: str | None = None,
                history_messages: Optional[List[Dict[str, str]]] = None,
                keyword_extraction: bool = False,
                stream: bool = False,
                **kwargs,
            ) -> str:
                if stream:
                    raise NotImplementedError(
                        "Streaming LightRAG queries are not enabled in this benchmark wrapper."
                    )

                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                if history_messages:
                    messages.extend(history_messages)
                messages.append({"role": "user", "content": prompt})

                response_kwargs = {}
                if keyword_extraction and "deepseek" not in MODEL_NAME.lower():
                    response_kwargs["response_format"] = {"type": "json_object"}

                return await generate_response(
                    messages,
                    max_tokens=int(
                        self.config.get(
                            "llm_max_tokens_keyword"
                            if keyword_extraction
                            else "llm_max_tokens",
                            self.config.get("llm_max_tokens", 2048),
                        )
                    ),
                    temperature=float(self.config.get("llm_temperature", 0.1)),
                    custom_model=self.config.get("llm_model_name", MODEL_NAME),
                    logger=self.logger,
                    **response_kwargs,
                )

            @wrap_embedding_func_with_attrs(
                embedding_dim=int(embedding_dim),
                max_token_size=int(self.config.get("embedding_max_tokens", EMB_CONTEXT_LENGTH)),
                model_name=self.config.get("embedding_model_name", EMB_MODEL_NAME),
            )
            async def lightrag_embedding_func(texts: List[str], **kwargs):
                embeddings = await generate_embedding(list(texts), logger=self.logger)
                return np.array(embeddings, dtype=np.float32)

            self._rag = LightRAG(
                working_dir=str(self.working_dir),
                llm_model_func=lightrag_llm_model_func,
                embedding_func=lightrag_embedding_func,
                llm_model_name=self.config.get("llm_model_name", MODEL_NAME),
                llm_model_max_async=int(
                    self.config.get("llm_model_max_async", 128)
                ),
                embedding_func_max_async=int(
                    self.config.get("embedding_func_max_async", 128)
                ),
                max_parallel_insert=int(
                    self.config.get("max_parallel_insert", 128)
                ),
                chunk_token_size=int(self.config.get("chunk_token_size", 1200)),
                chunk_overlap_token_size=int(
                    self.config.get("chunk_overlap_token_size", 100)
                ),
                kv_storage=self.config.get("kv_storage", "JsonKVStorage"),
                vector_storage=self.config.get("vector_storage", "NanoVectorDBStorage"),
                graph_storage=self.config.get("graph_storage", "NetworkXStorage"),
                doc_status_storage=self.config.get(
                    "doc_status_storage", "JsonDocStatusStorage"
                ),
                auto_manage_storages_states=False,
            )
            self._QueryParam = QueryParam
            self._DocStatus = DocStatus
            await self._rag.initialize_storages()

    async def _infer_embedding_dim(self) -> int:
        probe = await generate_embedding(["dimension probe"], logger=self.logger)
        if not probe or not probe[0]:
            raise RuntimeError("Failed to infer embedding dimension for LightRAG.")
        return len(probe[0])

    async def _resume_incomplete_docs_if_any(self) -> bool:
        pending = await self._rag.doc_status.get_docs_by_statuses(
            [
                self._DocStatus.PROCESSING,
                self._DocStatus.FAILED,
                self._DocStatus.PENDING,
            ]
        )
        if not pending:
            return False

        self.logger.info(
            f"Resuming {len(pending)} LightRAG documents left in processing/failed/pending states."
        )
        await self._rag.apipeline_process_enqueue_documents()
        return True

    async def _count_indexed_docs(self) -> int:
        docs = await self._rag.doc_status.get_docs_by_status(
            self._DocStatus.PROCESSED
        )
        return len(docs or {})

    def export_doc_graphs(
        self,
        output_dir: str | Path,
        doc_id_to_filename: Optional[Dict[str, str]] = None,
        overwrite: bool = True,
    ) -> Dict[str, Dict[str, int]]:
        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            self.aexport_doc_graphs(
                output_dir=output_dir,
                doc_id_to_filename=doc_id_to_filename,
                overwrite=overwrite,
            )
        )

    async def aexport_doc_graphs(
        self,
        output_dir: str | Path,
        doc_id_to_filename: Optional[Dict[str, str]] = None,
        overwrite: bool = True,
    ) -> Dict[str, Dict[str, int]]:
        await self._ensure_rag()

        if self._pending_docs:
            await self.afinalize_index()

        output_path = Path(output_dir).expanduser().resolve()
        output_path.mkdir(parents=True, exist_ok=True)

        doc_status_path = self.working_dir / "kv_store_doc_status.json"
        full_entities_path = self.working_dir / "kv_store_full_entities.json"
        graph_path = self.working_dir / "graph_chunk_entity_relation.graphml"

        if not doc_status_path.exists() or not full_entities_path.exists() or not graph_path.exists():
            raise RuntimeError(
                "LightRAG graph export requires kv_store_doc_status.json, "
                "kv_store_full_entities.json, and graph_chunk_entity_relation.graphml."
            )

        doc_status = json.loads(doc_status_path.read_text(encoding="utf-8"))
        full_entities = json.loads(full_entities_path.read_text(encoding="utf-8"))
        graph = nx.read_graphml(graph_path)

        doc_id_to_filename = doc_id_to_filename or {}
        exported: Dict[str, Dict[str, int]] = {}
        processed_status = self._DocStatus.PROCESSED.value

        for doc_id, status in self._iter_doc_status_items(doc_status):
            if status.get("status") != processed_status:
                continue

            chunk_ids = set(status.get("chunks_list") or [])
            entities = set(full_entities.get(doc_id, {}).get("entity_names", []))
            relations = set()

            for node_name, node_data in graph.nodes(data=True):
                if self._has_chunk_overlap(node_data.get("source_id"), chunk_ids):
                    entities.add(str(node_name))

            for source, target, edge_data in graph.edges(data=True):
                descriptions = self._select_descriptions_for_doc(edge_data, chunk_ids)
                if not descriptions:
                    continue

                entities.add(str(source))
                entities.add(str(target))
                for description in descriptions:
                    relation_text = self._normalize_relation_text(description)
                    if relation_text:
                        relations.add((str(source), relation_text, str(target)))

            file_name = doc_id_to_filename.get(doc_id, f"{doc_id}.json")
            target_path = output_path / file_name
            if target_path.exists() and not overwrite:
                self.logger.info(f"Skipping existing exported KG: {target_path}")
                continue

            payload = {
                "entities": sorted(entities),
                "relations": [list(rel) for rel in sorted(relations)],
            }
            target_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            exported[doc_id] = {
                "entities": len(payload["entities"]),
                "relations": len(payload["relations"]),
            }

        self.logger.info(
            f"Exported {len(exported)} document-level LightRAG KGs into {output_path}"
        )
        return exported

    def _manifest_path(self) -> Path:
        return self.working_dir / "bidirection_lightrag_manifest.json"

    def _doc_status_path(self) -> Path:
        return self.working_dir / "kv_store_doc_status.json"

    def _index_ready(self) -> bool:
        return self._manifest_path().exists()

    def _has_existing_doc_status_store(self) -> bool:
        doc_status_path = self._doc_status_path()
        if not doc_status_path.exists():
            return False
        try:
            data = json.loads(doc_status_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        return bool(data)

    def _write_manifest(self, payload: Dict[str, Any]):
        manifest = {
            **payload,
            "working_dir": str(self.working_dir),
            "created_at": datetime.utcnow().isoformat() + "Z",
        }
        self._manifest_path().write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def _extract_file_path(self, id: str, ref: str) -> str:
        if not ref:
            return id or "unknown_source"
        try:
            parsed = json.loads(ref)
        except Exception:
            return id or "unknown_source"

        if isinstance(parsed, dict):
            if "path" in parsed and isinstance(parsed["path"], str):
                return parsed["path"]
            if id in parsed and isinstance(parsed[id], dict):
                nested = parsed[id]
                if isinstance(nested.get("path"), str):
                    return nested["path"]
                if isinstance(nested.get("link"), str):
                    return nested["link"]
                if isinstance(nested.get("name"), str):
                    return nested["name"]
        return id or "unknown_source"

    def _maybe_load_corpus(self) -> List[Dict[str, str]]:
        corpus_path = self.config.get("corpus_path")
        if not corpus_path:
            return []

        corpus = Path(str(corpus_path)).expanduser()
        if not corpus.exists():
            self.logger.warning(f"LightRAG corpus_path does not exist: {corpus}")
            return []

        docs: List[Dict[str, str]] = []
        if corpus.is_file():
            text = corpus.read_text(encoding="utf-8")
            docs.append({"id": corpus.stem, "doc": text, "file_path": str(corpus)})
            return docs

        for path in sorted(corpus.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".md", ".txt"}:
                continue
            try:
                docs.append(
                    {
                        "id": path.stem,
                        "doc": path.read_text(encoding="utf-8"),
                        "file_path": str(path),
                    }
                )
            except Exception as exc:
                self.logger.warning(f"Failed to read corpus file {path}: {exc}")
        return docs

    @staticmethod
    def _iter_doc_status_items(doc_status: Dict[str, Dict[str, Any]]):
        def sort_key(item):
            doc_id = str(item[0])
            return (0, int(doc_id)) if doc_id.isdigit() else (1, doc_id)

        return sorted(doc_status.items(), key=sort_key)

    @staticmethod
    def _split_sep(value: Any) -> List[str]:
        if value is None:
            return []
        text = str(value).strip()
        if not text:
            return []
        return [part.strip() for part in text.split("<SEP>") if part.strip()]

    @classmethod
    def _has_chunk_overlap(cls, source_id: Any, chunk_ids: set[str]) -> bool:
        if not chunk_ids:
            return False
        return any(chunk_id in chunk_ids for chunk_id in cls._split_sep(source_id))

    @classmethod
    def _select_descriptions_for_doc(
        cls,
        edge_data: Dict[str, Any],
        chunk_ids: set[str],
    ) -> List[str]:
        source_ids = cls._split_sep(edge_data.get("source_id"))
        descriptions = cls._split_sep(edge_data.get("description"))

        if not source_ids or not descriptions or not chunk_ids:
            return []

        if len(descriptions) == len(source_ids):
            return [
                description
                for source_id, description in zip(source_ids, descriptions)
                if source_id in chunk_ids and description
            ]

        if any(source_id in chunk_ids for source_id in source_ids):
            return descriptions

        return []

    @staticmethod
    def _normalize_relation_text(description: str) -> str:
        text = " ".join(str(description).split())
        return text[:-1] if text.endswith(".") else text
