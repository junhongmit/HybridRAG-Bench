Leaderboard
===========

This page tracks benchmark results for HybridRAG-Bench.

Summary
-------

- Metric columns can be adapted to your final evaluation protocol.
- Higher is better unless a column explicitly states lower-is-better.
- Update tables by editing:
  ``docs/source/_static/leaderboard_arxiv_ai.csv``,
  ``docs/source/_static/leaderboard_arxiv_cy.csv``,
  ``docs/source/_static/leaderboard_arxiv_bio.csv``.

Arxiv_AI
--------

.. csv-table::
   :file: _static/leaderboard_arxiv_ai.csv
   :header-rows: 1

Arxiv_CY
--------

.. csv-table::
   :file: _static/leaderboard_arxiv_cy.csv
   :header-rows: 1

Arxiv_BIO
---------

.. csv-table::
   :file: _static/leaderboard_arxiv_bio.csv
   :header-rows: 1

Submission Format
-----------------

Use this schema when adding new results:

- ``Date``: YYYY-MM-DD
- ``Model``: Model name and size
- ``Method``: IO, RAG, KG-RAG, HybridRAG, etc.
- ``Acc``: Primary accuracy metric
- ``Notes``: Optional details (retriever, hops, or config)
