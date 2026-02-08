import os
from dotenv import load_dotenv

env_file = os.getenv("ENV_FILE", ".env")
dotenv_path = os.path.join(os.path.dirname(__file__), "..", env_file)
load_dotenv(dotenv_path)

############### LLM Settings ###############
# If API base url is not set, fallback to use vLLM based local server
API_KEY = os.environ.get("API_KEY", "")
API_BASE = os.environ.get("API_BASE")
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
CONTEXT_LENGTH = int(os.environ.get("CONTEXT_LENGTH", "131072"))
TIME_OUT = int(os.environ.get("TIME_OUT", "-1"))
TIME_OUT = TIME_OUT if TIME_OUT > 0 else None

EMB_API_KEY = os.environ.get("API_KEY", "")
EMB_API_BASE = os.environ.get("EMB_API_BASE")
EMB_MODEL_NAME = os.environ.get("EMB_MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct-e")
EMB_CONTEXT_LENGTH = int(os.environ.get("EMB_CONTEXT_LENGTH", "512"))
EMB_TIME_OUT = int(os.environ.get("EMB_TIME_OUT", "-1"))
EMB_TIME_OUT = EMB_TIME_OUT if EMB_TIME_OUT > 0 else None

EVAL_API_KEY = os.environ.get("EVAL_API_KEY", API_KEY)
EVAL_API_BASE = os.environ.get("EVAL_API_BASE", API_BASE)
EVAL_MODEL_NAME = os.environ.get("EVAL_MODEL_NAME", MODEL_NAME)
EVAL_CONTEXT_LENGTH = int(os.environ.get("EVAL_CONTEXT_LENGTH", "131072"))
EVAL_TIME_OUT = int(os.environ.get("EVAL_TIME_OUT", "-1"))
EVAL_TIME_OUT = EVAL_TIME_OUT if EVAL_TIME_OUT > 0 else None


############### KG Updater Settings ###############
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")

DATASET_PATH = os.environ.get("DATASET_PATH", "")

MAX_CHUNK = int(os.environ.get("MAX_CHUNK", "8_000"))
MIN_CHUNK = int(os.environ.get("MIN_CHUNK", "4_000"))
MAX_GENERATION_TOKENS = int(os.environ.get("MAX_GENERATION_TOKENS", "10_000"))
EXPECTED_TIME = int(os.environ.get("EXPECTED_TIME", "600"))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "5"))
STAGES = int(os.environ.get("STAGES", "1"))

ALIGN_TOPK = int(os.environ.get("ALIGN_TOPK", "5"))

ALIGN_ENTITY = eval(os.environ.get("ALIGN_ENTITY", "False"))
ALIGN_ENTITY_BATCH_SIZE = int(os.environ.get("ALIGN_ENTITY_BATCH_SIZE", "32"))

MERGE_ENTITY = eval(os.environ.get("MERGE_ENTITY", "False"))
MERGE_ENTITY_BATCH_SIZE = int(os.environ.get("MERGE_ENTITY_BATCH_SIZE", "32"))

ALIGN_RELATION = eval(os.environ.get("ALIGN_RELATION", "False"))
ALIGN_RELATION_BATCH_SIZE = int(os.environ.get("ALIGN_RELATION_BATCH_SIZE", "32"))

MERGE_RELATION = eval(os.environ.get("MERGE_RELATION", "False"))
MERGE_RELATION_BATCH_SIZE = int(os.environ.get("MERGE_RELATION_BATCH_SIZE", "32"))

SELF_REFLECTION = eval(os.environ.get("SELF_REFLECTION", "False"))
