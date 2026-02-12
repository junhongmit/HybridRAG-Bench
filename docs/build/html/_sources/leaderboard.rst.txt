Leaderboard
===========

This page tracks benchmark results for HybridRAG-Bench.

Summary
-------

- Metric columns can be adapted to your final evaluation protocol.
- Higher is better unless a column explicitly states lower-is-better.
- Update the table by editing ``docs/source/_static/leaderboard.csv``.

Current Results
---------------

.. csv-table::
   :file: _static/leaderboard.csv
   :header-rows: 1

Submission Format
-----------------

Use this schema when adding new results:

- ``Date``: YYYY-MM-DD
- ``Model``: Model name and size
- ``Method``: IO, RAG, KG-RAG, HybridRAG, etc.
- ``Dataset``: Evaluated split/domain
- ``EM``: Exact match
- ``F1``: Token-level F1
- ``Faithfulness``: Attribution/grounding consistency
- ``Notes``: Optional details (retriever, hops, or config)
