Overview
========

HybridRAG-Bench is designed to separate retrieval quality from reasoning quality.

Pipeline
--------

1. Collect a time-framed corpus that is external to model pretraining.
2. Build hybrid knowledge from documents and extracted KG relations.
3. Generate reasoning-grounded QA pairs from explicit reasoning paths.
4. Validate benchmark quality with automated checks.

Repository Structure
--------------------

- ``arxiv_fetcher/``: Corpus acquisition and processing.
- ``dataset/``: Dataset construction utilities.
- ``kg/``: Knowledge graph storage, preprocessing, and updates.
- ``question_gen/``: Multi-type reasoning question generation.
- ``inference/``: Baselines and retrieval-augmented inference methods.
- ``run/``: End-to-end pipeline and evaluation entry points.

Quick Start
-----------

Run the main pipeline modules from the project root:

.. code-block:: bash

   python -m run.run_kg_preprocess
   python -m run.run_kg_embed
   python -m run.run_kg_update
   python -m run.run_qa --dataset [movie, sports] --model [io, rag, kg, our]
