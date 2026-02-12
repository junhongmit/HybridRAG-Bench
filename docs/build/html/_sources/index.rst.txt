HybridRAG-Bench Documentation
=============================

**HybridRAG-Bench** is a benchmarking framework for retrieval-intensive, multi-hop reasoning over hybrid knowledge.
It evaluates how retrieval-augmented methods use both unstructured text and structured knowledge graphs under realistic settings.

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
