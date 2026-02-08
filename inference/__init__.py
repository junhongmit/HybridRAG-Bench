from inference.io_model import IO_Model
from inference.cot_model import CoT_Model
from inference.sc_model import SC_Model
from inference.cok_model import CoK_Model
from inference.rag_model import RAG_Model
from inference.one_hop_kg_model import OneHopKG_Model
from inference.one_hop_kg_rag_model import OneHopKG_RAG_Model
from inference.tog_model import ToG_Model
from inference.tog2_model import ToG2_Model
from inference.pog_model import PoG_Model
from inference.rog_model import RoG_Model
from inference.hipporag_model import HippoRAG_Model
from inference.evoreasoner_model import EvoReasoner_Model
from inference.r2kg_model import R2KG_Model

MODEL_MAP = {
    "io": IO_Model,
    "cot": CoT_Model,
    "sc": SC_Model,
    "cok": CoK_Model,
    "rag": RAG_Model,
    "one-hop-kg": OneHopKG_Model,
    "one-hop-kg-rag": OneHopKG_RAG_Model,
    "tog": ToG_Model,
    "tog2": ToG2_Model,
    "pog": PoG_Model,
    "rog": RoG_Model,
    "hipporag": HippoRAG_Model,
    "evoresoner": EvoReasoner_Model,
    "r2kg": R2KG_Model,
}
