Installation
============

Environment
-----------

.. code-block:: bash

   conda create -n vllm python=3.12 -y
   conda activate vllm
   pip install vllm
   pip install -r requirements.txt

Then create your runtime config:

.. code-block:: bash

   cp .env_template .env

Set ``API_BASE``, ``API_KEY``, and dataset/model-specific paths in ``.env``.

Neo4j Setup
-----------

.. code-block:: bash

   wget "https://neo4j.com/artifact.php?name=neo4j-community-5.26.3-unix.tar.gz" -O neo4j.tar.gz
   tar -xvzf neo4j.tar.gz
   mv neo4j-community-*/ neo4j/
   cd neo4j
   bin/neo4j-admin dbms set-initial-password password

Copy APOC jar to plugins:

.. code-block:: bash

   cp neo4j/labs/apoc-5.26.3-core.jar neo4j/plugins/apoc-5.26.3-core.jar

Start or stop Neo4j:

.. code-block:: bash

   bin/neo4j start
   bin/neo4j stop
