from question_gen.multi_hop import generate as generate_multi_hop
from question_gen.single_hop import generate as generate_single_hop 
from question_gen.single_hop_w_condition import generate as generate_single_hop_w_condition
from question_gen.open_ended import generate as generate_open_ended
from question_gen.counterfactual import generate as generate_counterfactual
from question_gen.counterfactual_cwqstyle import generate as generate_counterfactual_cwqstyle
from question_gen.paper_text import generate as generate_paper_text

QUESTION_GEN_MAP = {
    "paper": generate_paper_text,
    "multi_hop": generate_multi_hop,
    "single_hop": generate_single_hop,
    "single_hop_w_condition": generate_single_hop_w_condition,
    "open_ended": generate_open_ended,
    "counterfactual": generate_counterfactual,
    "counterfactual_cwqstyle": generate_counterfactual_cwqstyle,
}

__all__ = [
    "QUESTION_GEN_MAP",
    "generate_paper_text",
    "generate_multi_hop",
    "generate_single_hop",
    "generate_single_hop_w_condition",
    "generate_open_ended",
    "generate_counterfactual",
    "generate_counterfactual_cwqstyle",
]
