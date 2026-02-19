HybridRAG-Bench Documentation
=============================

.. raw:: html

   <a class="paper-link-card" href="https://www.arxiv.org/abs/2602.10210" target="_blank" rel="noopener noreferrer">
     <span class="paper-icon" aria-hidden="true">&#128196;</span>
     <span class="paper-text">
       <span class="paper-label">Read The Paper</span>
       <span class="paper-title">How Much Reasoning Do Retrieval-Augmented Models Add beyond LLMs?</span>
     </span>
   </a>
   <br>

Large language models (LLMs) continue to struggle with knowledge-intensive questions that require up-to-date
information and multi-hop reasoning. Augmenting LLMs with hybrid external knowledge, such as unstructured text
and structured knowledge graphs, offers a promising alternative to costly continual pretraining. As such,
reliable evaluation of their retrieval and reasoning capabilities becomes critical. However, many existing
benchmarks increasingly overlap with LLM pretraining data, which means answers or supporting knowledge may
already be encoded in model parameters, making it difficult to distinguish genuine retrieval and reasoning
from parametric recall.

We introduce :hybridrag:`null` **HybridRAG-Bench**, a framework for constructing benchmarks to evaluate retrieval-intensive,
multi-hop reasoning over hybrid knowledge. :hybridrag:`null` HybridRAG-Bench automatically couples unstructured text and structured
knowledge graph representations derived from recent scientific literature on arXiv, and generates
knowledge-intensive question-answer pairs grounded in explicit reasoning paths. The framework supports flexible
domain and time-frame selection, enabling contamination-aware and customizable evaluation as models and knowledge
evolve. Experiments across three domains (artificial intelligence, governance and policy, and bioinformatics)
demonstrate that :hybridrag:`null` HybridRAG-Bench rewards genuine retrieval and reasoning rather than parametric recall, offering
a principled testbed for evaluating hybrid knowledge-augmented reasoning systems.

.. image:: ./_static/framework.png
  :align: center

The framework supports:

- Time-framed corpus collection from scientific sources.
- Hybrid knowledge construction from aligned text chunks and KG structure.
- Reasoning-grounded QA generation across multiple reasoning types.
- Automated quality control and reproducible benchmark evaluation.

.. toctree::
   :maxdepth: 1
   :caption: Documentation

   get_started/overview
   install/installation
   leaderboard
