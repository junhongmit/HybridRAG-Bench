import csv
import os
import re
from typing import AsyncGenerator, Any, Dict, List

from utils.data import *
from utils.logger import *
from utils.utils import parse_timestamp

class ArxivDatasetLoader(BaseDatasetLoader):
    
    def __init__(self,
                 data_path: str,
                 config: Dict[str, Any], 
                 mode: str = "doc",
                 logger: BaseProgressLogger = DefaultProgressLogger(),
                 **kwargs):
        super().__init__(config, mode, **kwargs)

        self.data_path = data_path
        self.logger = logger
        self.input_question_path = os.path.join(self.data_path, "questions.json")
        self.data_generator = load_documents(
            os.path.join(data_path, "md")
        )
    
    async def load_doc(self) -> AsyncGenerator[Dict[str, Any], None]:
        while True:
            try:
                item = next(self.data_generator)
            except StopIteration:
                break  # Exit the loop when there is no more data.

            # Transform each record into a document item with necessary fields
            doc_id = item["id"]
            if doc_id in self.logger.processed_docs:
                continue
            
            doc = item["context"]
            ref = json.dumps({"id": doc_id, "path": item["doc_path"]})
            # modified_at = parse_timestamp(modified_time)
            # created_at = parse_timestamp(query_time)
            yield {
                "id": doc_id,
                "doc": doc,
                "created_at": None,
                "modified_at": None,
                "ref": ref
            }

    async def load_query(self) -> AsyncGenerator[Dict[str, Any], None]:
        contents = [item["context"] for item in list(self.data_generator)]
        with open(self.input_question_path, "r", encoding="utf-8") as f:
            questions = json.load(f)
        
        for idx, question in enumerate(questions):
            query_id = f"{idx}"
            if query_id in self.logger.processed_questions:
                    continue
            # with open(os.path.join(self.data_path, f"{question["paper"]}.md"), 'r', encoding='utf-8') as file:
            #     content = file.read().strip()
            yield {
                "id": query_id,
                "interaction_id": query_id, # TODO: Will deprecate this in the near future
                "query": question['question'],
                "query_time": None,
                "docs": contents, # [content],
                "ans": question['answer']
            }
    
    # async def load_query(self) -> AsyncGenerator[Dict[str, Any], None]:
    #     with open(self.input_csv_path, 'r', newline='', encoding='utf-8') as csvfile:
    #         reader = csv.DictReader(csvfile)
    #         for idx, row in enumerate(reader):
    #             if 'question' in row:  # Assuming the column name is 'question'
    #                 query_id = f"{idx}"
    #                 yield {
    #                     "id": query_id,
    #                     "interaction_id": query_id, # TODO: Will deprecate this in the near future
    #                     "query": row['question'],
    #                     "query_time": None,
    #                     "docs": [doc], # TODO: Requires a paired doc
    #                     "ans": row['answer']
    #                 }

def load_documents(directory_path, start_idx=0):
    dataset = []

    filepaths = [os.path.join(directory_path, filename) for filename in os.listdir(directory_path)
                 if filename.endswith(".md")]

    filepaths.sort(key=lambda x: int(re.search(r'(\d+)', x).group(1)) if re.search(r'(\d+)', x) else float('inf'))

    for idx, filepath in enumerate(filepaths):
        if idx < start_idx: continue
        entry = {"id": os.path.splitext(os.path.basename(filepath))[0], 
                 "context": None,
                 "doc_path": filepath}
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read().strip()
                print(f"File: {filepath}, Content length: {len(content)}")
                entry["context"] = content
        except Exception as e:
            print(f"Error reading file {filepath}: {e}")

        yield entry
    print(f"Dataset before batching: {dataset}") #added print statement

