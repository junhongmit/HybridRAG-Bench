import json
import os
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Iterable, Optional

from utils.data import BaseDatasetLoader
from utils.logger import BaseProgressLogger, DefaultProgressLogger


def iter_text_documents(
    directory_path: str,
    suffixes: tuple[str, ...] = (".txt", ".md"),
) -> Iterable[Dict[str, str]]:
    """
    Minimal helper for plain-text corpora.

    It walks a directory recursively and yields one record per file:
    {
        "id": "<relative path without suffix>",
        "context": "<full text>",
        "doc_path": "<absolute path>"
    }
    """
    root = Path(directory_path)
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        yield {
            "id": str(path.relative_to(root).with_suffix("")),
            "context": path.read_text(encoding="utf-8").strip(),
            "doc_path": str(path.resolve()),
        }


class TemplateTextDatasetLoader(BaseDatasetLoader):
    """
    Template loader for user-provided plain-text corpora.

    Expected layout:
      my_dataset/
        docs/
          0001.txt
          0002.txt
        questions.json   # optional, only needed for QA mode

    `questions.json` should be a list of objects like:
    [
      {"question": "...", "answer": "..."},
      {"question": "...", "answer": "..."}
    ]

    To adapt this template:
    1. Point `doc_dir` at your text directory.
    2. Adjust `iter_text_documents()` if your metadata lives elsewhere.
    3. Optionally customize `load_query()` if your QA file has a different schema.
    """

    def __init__(
        self,
        data_path: str,
        config: Dict[str, Any],
        mode: str = "doc",
        logger: BaseProgressLogger = DefaultProgressLogger(),
        question_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(config, mode, **kwargs)
        self.data_path = data_path
        self.logger = logger
        self.doc_dir = os.path.join(self.data_path, "docs")
        self.question_path = question_path or os.path.join(self.data_path, "questions.json")

    async def load_doc(self) -> AsyncGenerator[Dict[str, Any], None]:
        for item in iter_text_documents(self.doc_dir):
            doc_id = item["id"]
            if doc_id in self.logger.processed_docs:
                continue
            yield {
                "id": doc_id,
                "doc": item["context"],
                "created_at": None,
                "modified_at": None,
                "ref": json.dumps({"id": doc_id, "path": item["doc_path"]}),
            }

    async def load_query(self) -> AsyncGenerator[Dict[str, Any], None]:
        if not os.path.exists(self.question_path):
            raise FileNotFoundError(
                f"questions.json not found at {self.question_path}. "
                "Provide a QA file or override load_query()."
            )

        with open(self.question_path, "r", encoding="utf-8") as f:
            questions = json.load(f)

        for idx, item in enumerate(questions):
            query_id = str(idx)
            if query_id in self.logger.processed_questions:
                continue
            yield {
                "id": query_id,
                "interaction_id": query_id,
                "query": item["question"],
                "query_time": None,
                "docs": [],
                "ans": item.get("answer", ""),
            }
