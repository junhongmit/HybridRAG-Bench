# KG Update
python -m run.run_question_gen_v2
python -m run.run_kg_embed
python -m run.run_kg_update --dataset arxiv_ai
python -m run.run_qa --dataset arxiv --model rag --postfix -0
python -m run.run_qa --dataset arxiv --model our --postfix -0
python -m run.run_eval --dataset arxiv --model our --postfix -0

# Question gen
ENV_FILE=.env.deepseek python -m run.run_question_gen --question-type all
ENV_FILE=.env.deepseek.updated python -m run.run_question_gen --question-type single_hop --gen_round 40
ENV_FILE=.env.deepseek.updated python -m run.run_question_gen --question-type single_hop_w_condition --gen_round 20
ENV_FILE=.env.deepseek.updated python -m run.run_question_gen --question-type multi_hop --gen_round 40
ENV_FILE=.env.deepseek.updated python -m run.run_question_gen --question-type multi_hop --difficulty difficult --gen_round 40
ENV_FILE=.env.deepseek.updated python -m run.run_question_gen --question-type open_ended --gen_round 60
ENV_FILE=.env.deepseek.updated python -m run.run_question_gen --question-type counterfactual --gen_round 60
python -m run.run_question_merge

# HippoRAG
ENV_FILE=.env.updated python -m run.run_hipporag_index --dataset movie
python -m run.run_qa --dataset movie --model hipporag --postfix -0


python -m run.run_qa --dataset arxiv_qm --model io --postfix -1
python -m run.run_qa --dataset arxiv_qm --model cot --postfix -1
python -m run.run_qa --dataset arxiv_qm --model sc --postfix -1
python -m run.run_qa --dataset arxiv_qm --model rag --postfix -1 --num-workers=8
python -m run.run_qa --dataset arxiv_qm --model one-hop-kg --postfix -1
python -m run.run_qa --dataset arxiv_qm --model one-hop-kg-rag --postfix -0 --num-workers=8
python -m run.run_qa --dataset arxiv_qm --model cok --postfix -1
python -m run.run_qa --dataset arxiv_qm --model tog --postfix -1
python -m run.run_qa --dataset arxiv_qm --model tog2 --postfix -1
python -m run.run_qa --dataset arxiv_qm --model pog --postfix -1
python -m run.run_qa --dataset arxiv_qm --model rog --postfix -1
python -m run.run_qa --dataset arxiv_qm --model our --postfix -1 --num-workers=16
python -m run.run_hipporag_index --dataset arxiv_qm --config force_index=True
python -m run.run_qa --dataset arxiv_qm --model hipporag --postfix -1

ENV_FILE=.env.qwen python -m run.run_qa --dataset arxiv_qm --model io --postfix qwen-1
ENV_FILE=.env.qwen python -m run.run_qa --dataset arxiv_qm --model cot --postfix qwen-0
ENV_FILE=.env.qwen python -m run.run_qa --dataset arxiv_qm --model sc --postfix qwen-0
ENV_FILE=.env.qwen python -m run.run_qa --dataset arxiv_qm --model rag --postfix qwen-0 --num-workers=8
ENV_FILE=.env.qwen python -m run.run_qa --dataset arxiv_qm --model one-hop-kg --postfix qwen-1
ENV_FILE=.env.qwen python -m run.run_qa --dataset arxiv_qm --model one-hop-kg-rag --postfix qwen-0 --num-workers=8
ENV_FILE=.env.qwen python -m run.run_qa --dataset arxiv_qm --model cok --postfix qwen-0
ENV_FILE=.env.qwen python -m run.run_qa --dataset arxiv_qm --model tog --postfix qwen-1
ENV_FILE=.env.qwen python -m run.run_qa --dataset arxiv_qm --model tog2 --postfix qwen-0
ENV_FILE=.env.qwen python -m run.run_qa --dataset arxiv_qm --model pog --postfix qwen-0
ENV_FILE=.env.qwen python -m run.run_qa --dataset arxiv_qm --model rog --postfix qwen-1
ENV_FILE=.env.qwen python -m run.run_qa --dataset arxiv_qm --model r2kg --postfix qwen-0
ENV_FILE=.env.qwen python -m run.run_qa --dataset arxiv_qm --model our --postfix qwen-0 --num-workers=16
ENV_FILE=.env.qwen python -m run.run_hipporag_index --dataset arxiv_qm --config force_index=True
ENV_FILE=.env.qwen python -m run.run_qa --dataset arxiv_qm --model hipporag --postfix qwen-0

ENV_FILE=.env.llama8 python -m run.run_qa --dataset arxiv_qm --model io --postfix llama8-1
ENV_FILE=.env.llama8 python -m run.run_qa --dataset arxiv_qm --model cot --postfix llama8-1
ENV_FILE=.env.llama8 python -m run.run_qa --dataset arxiv_qm --model sc --postfix llama8-0
ENV_FILE=.env.llama8 python -m run.run_qa --dataset arxiv_qm --model rag --postfix llama8-0 --num-workers=8
ENV_FILE=.env.llama8 python -m run.run_qa --dataset arxiv_qm --model one-hop-kg --postfix llama8-0
ENV_FILE=.env.llama8 python -m run.run_qa --dataset arxiv_qm --model one-hop-kg-rag --postfix llama8-0 --num-workers=8
ENV_FILE=.env.llama8 python -m run.run_qa --dataset arxiv_qm --model tog --postfix llama8-0
ENV_FILE=.env.llama8 python -m run.run_qa --dataset arxiv_qm --model tog2 --postfix llama8-1
ENV_FILE=.env.llama8 python -m run.run_qa --dataset arxiv_qm --model pog --postfix llama8-1
ENV_FILE=.env.llama8 python -m run.run_qa --dataset arxiv_qm --model rog --postfix llama8-1
ENV_FILE=.env.llama8 python -m run.run_qa --dataset arxiv_qm --model r2kg --postfix llama8-1
ENV_FILE=.env.llama8 python -m run.run_qa --dataset arxiv_qm --model our --postfix llama8-0 --num-workers=16
ENV_FILE=.env.llama8 python -m run.run_hipporag_index --dataset arxiv_qm --config force_index=True
ENV_FILE=.env.llama8 python -m run.run_qa --dataset arxiv_qm --model hipporag --postfix llama8-1

ENV_FILE=.env.deepseek python -m run.run_qa --dataset arxiv_qm --model io --postfix deepseek-0
ENV_FILE=.env.deepseek python -m run.run_qa --dataset arxiv_qm --model cot --postfix deepseek-0
ENV_FILE=.env.deepseek python -m run.run_qa --dataset arxiv_qm --model sc --postfix deepseek-0
ENV_FILE=.env.deepseek python -m run.run_qa --dataset arxiv_qm --model rag --postfix deepseek-0 --num-workers=8
ENV_FILE=.env.deepseek python -m run.run_qa --dataset arxiv_qm --model one-hop-kg --postfix deepseek-0
ENV_FILE=.env.deepseek python -m run.run_qa --dataset arxiv_qm --model one-hop-kg-rag --postfix deepseek-0 --num-workers=8
ENV_FILE=.env.deepseek python -m run.run_qa --dataset arxiv_qm --model cok --postfix deepseek-0
ENV_FILE=.env.deepseek python -m run.run_qa --dataset arxiv_qm --model tog --postfix deepseek-0
ENV_FILE=.env.deepseek python -m run.run_qa --dataset arxiv_qm --model tog2 --postfix deepseek-0
ENV_FILE=.env.deepseek python -m run.run_qa --dataset arxiv_qm --model pog --postfix deepseek-0
ENV_FILE=.env.deepseek python -m run.run_qa --dataset arxiv_qm --model rog --postfix deepseek-0
ENV_FILE=.env.deepseek python -m run.run_qa --dataset arxiv_qm --model r2kg --postfix deepseek-0
ENV_FILE=.env.deepseek python -m run.run_qa --dataset arxiv_qm --model our --postfix deepseek-0 --num-workers=16
ENV_FILE=.env.deepseek python -m run.run_hipporag_index --dataset arxiv_qm --config force_index=True
ENV_FILE=.env.deepseek python -m run.run_qa --dataset arxiv_qm --model hipporag --postfix deepseek-0

ENV_FILE=.env.updated python -m run.run_qa --dataset arxiv_cy --model io --postfix -1
ENV_FILE=.env.updated python -m run.run_qa --dataset arxiv_cy --model cot --postfix -1
ENV_FILE=.env.updated python -m run.run_qa --dataset arxiv_cy --model sc --postfix -1
ENV_FILE=.env.updated python -m run.run_qa --dataset arxiv_cy --model rag --postfix -1 --num-workers=8
ENV_FILE=.env.updated python -m run.run_qa --dataset arxiv_cy --model one-hop-kg --postfix -1
ENV_FILE=.env.updated python -m run.run_qa --dataset arxiv_cy --model one-hop-kg-rag --postfix -1 --num-workers=8
ENV_FILE=.env.updated python -m run.run_qa --dataset arxiv_cy --model cok --postfix -1
ENV_FILE=.env.updated python -m run.run_qa --dataset arxiv_cy --model tog --postfix -1
ENV_FILE=.env.updated python -m run.run_qa --dataset arxiv_cy --model tog2 --postfix -1
ENV_FILE=.env.updated python -m run.run_qa --dataset arxiv_cy --model pog --postfix -1
ENV_FILE=.env.updated python -m run.run_qa --dataset arxiv_cy --model rog --postfix -1
ENV_FILE=.env.updated python -m run.run_qa --dataset arxiv_cy --model r2kg --postfix -1
ENV_FILE=.env.updated python -m run.run_qa --dataset arxiv_cy --model our --postfix -1 --num-workers=16
ENV_FILE=.env.updated python -m run.run_hipporag_index --dataset arxiv_cy --config force_index=True
ENV_FILE=.env.updated python -m run.run_qa --dataset arxiv_cy --model hipporag --postfix -1

ENV_FILE=.env.qwen.updated python -m run.run_qa --dataset arxiv_cy --model io --postfix qwen-0
ENV_FILE=.env.qwen.updated python -m run.run_qa --dataset arxiv_cy --model cot --postfix qwen-0
ENV_FILE=.env.qwen.updated python -m run.run_qa --dataset arxiv_cy --model sc --postfix qwen-0
ENV_FILE=.env.qwen.updated python -m run.run_qa --dataset arxiv_cy --model rag --postfix qwen-0 --num-workers=8
ENV_FILE=.env.qwen.updated python -m run.run_qa --dataset arxiv_cy --model one-hop-kg --postfix qwen-1
ENV_FILE=.env.qwen.updated python -m run.run_qa --dataset arxiv_cy --model one-hop-kg-rag --postfix qwen-0 --num-workers=8
ENV_FILE=.env.qwen.updated python -m run.run_qa --dataset arxiv_cy --model cok --postfix qwen-0
ENV_FILE=.env.qwen.updated python -m run.run_qa --dataset arxiv_cy --model tog --postfix qwen-0
ENV_FILE=.env.qwen.updated python -m run.run_qa --dataset arxiv_cy --model tog2 --postfix qwen-0
ENV_FILE=.env.qwen.updated python -m run.run_qa --dataset arxiv_cy --model pog --postfix qwen-0
ENV_FILE=.env.qwen.updated python -m run.run_qa --dataset arxiv_cy --model rog --postfix qwen-0
ENV_FILE=.env.qwen.updated python -m run.run_qa --dataset arxiv_cy --model r2kg --postfix qwen-1
ENV_FILE=.env.qwen.updated python -m run.run_qa --dataset arxiv_cy --model our --postfix qwen-1 --num-workers=16
ENV_FILE=.env.qwen.updated python -m run.run_hipporag_index --dataset arxiv_cy --config force_index=True
ENV_FILE=.env.qwen.updated python -m run.run_qa --dataset arxiv_cy --model hipporag --postfix qwen-0

ENV_FILE=.env.llama8.updated python -m run.run_qa --dataset arxiv_cy --model io --postfix llama8-1
ENV_FILE=.env.llama8.updated python -m run.run_qa --dataset arxiv_cy --model cot --postfix llama8-1
ENV_FILE=.env.llama8.updated python -m run.run_qa --dataset arxiv_cy --model sc --postfix llama8-1
ENV_FILE=.env.llama8.updated python -m run.run_qa --dataset arxiv_cy --model rag --postfix llama8-1 --num-workers=8
ENV_FILE=.env.llama8.updated python -m run.run_qa --dataset arxiv_cy --model one-hop-kg --postfix llama8-1
ENV_FILE=.env.llama8.updated python -m run.run_qa --dataset arxiv_cy --model one-hop-kg-rag --postfix llama8-1 --num-workers=8
ENV_FILE=.env.llama8.updated python -m run.run_qa --dataset arxiv_cy --model cok --postfix llama8-1
ENV_FILE=.env.llama8.updated python -m run.run_qa --dataset arxiv_cy --model tog --postfix llama8-1
ENV_FILE=.env.llama8.updated python -m run.run_qa --dataset arxiv_cy --model tog2 --postfix llama8-1
ENV_FILE=.env.llama8.updated python -m run.run_qa --dataset arxiv_cy --model pog --postfix llama8-1
ENV_FILE=.env.llama8.updated python -m run.run_qa --dataset arxiv_cy --model rog --postfix llama8-1
ENV_FILE=.env.llama8.updated python -m run.run_qa --dataset arxiv_cy --model r2kg --postfix llama8-1
ENV_FILE=.env.llama8.updated python -m run.run_qa --dataset arxiv_cy --model our --postfix llama8-1 --num-workers=16
ENV_FILE=.env.llama8.updated python -m run.run_hipporag_index --dataset arxiv_cy --config force_index=True
ENV_FILE=.env.llama8.updated python -m run.run_qa --dataset arxiv_cy --model hipporag --postfix llama8-1

ENV_FILE=.env.deepseek.updated python -m run.run_qa --dataset arxiv_cy --model io --postfix deepseek-0
ENV_FILE=.env.deepseek.updated python -m run.run_qa --dataset arxiv_cy --model cot --postfix deepseek-0
ENV_FILE=.env.deepseek.updated python -m run.run_qa --dataset arxiv_cy --model sc --postfix deepseek-0
ENV_FILE=.env.deepseek.updated python -m run.run_qa --dataset arxiv_cy --model rag --postfix deepseek-0 --num-workers=8
ENV_FILE=.env.deepseek.updated python -m run.run_qa --dataset arxiv_cy --model one-hop-kg --postfix deepseek-0
ENV_FILE=.env.deepseek.updated python -m run.run_qa --dataset arxiv_cy --model one-hop-kg-rag --postfix deepseek-0 --num-workers=8
ENV_FILE=.env.deepseek.updated python -m run.run_qa --dataset arxiv_cy --model cok --postfix deepseek-0
ENV_FILE=.env.deepseek.updated python -m run.run_qa --dataset arxiv_cy --model tog --postfix deepseek-0
ENV_FILE=.env.deepseek.updated python -m run.run_qa --dataset arxiv_cy --model tog2 --postfix deepseek-0
ENV_FILE=.env.deepseek.updated python -m run.run_qa --dataset arxiv_cy --model pog --postfix deepseek-0
ENV_FILE=.env.deepseek.updated python -m run.run_qa --dataset arxiv_cy --model rog --postfix deepseek-0
ENV_FILE=.env.deepseek.updated python -m run.run_qa --dataset arxiv_cy --model r2kg --postfix deepseek-0
ENV_FILE=.env.deepseek.updated python -m run.run_qa --dataset arxiv_cy --model our --postfix deepseek-0 --num-workers=16
ENV_FILE=.env.deepseek.updated python -m run.run_hipporag_index --dataset arxiv_cy --config force_index=True
ENV_FILE=.env.deepseek.updated python -m run.run_qa --dataset arxiv_cy --model hipporag --postfix deepseek-0

python -m run.run_qa --dataset arxiv_ai --model io --postfix -1
python -m run.run_qa --dataset arxiv_ai --model cot --postfix -1
python -m run.run_qa --dataset arxiv_ai --model sc --postfix -1
python -m run.run_qa --dataset arxiv_ai --model rag --postfix -1 --num-workers=8
python -m run.run_qa --dataset arxiv_ai --model one-hop-kg --postfix -1
python -m run.run_qa --dataset arxiv_ai --model one-hop-kg-rag --postfix -1 --num-workers=8
python -m run.run_qa --dataset arxiv_ai --model cok --postfix -1
python -m run.run_qa --dataset arxiv_ai --model rog --postfix -1
python -m run.run_qa --dataset arxiv_ai --model tog --postfix -1
python -m run.run_qa --dataset arxiv_ai --model tog2 --postfix -1
python -m run.run_qa --dataset arxiv_ai --model pog --postfix -1
python -m run.run_qa --dataset arxiv_ai --model r2kg --postfix -1
python -m run.run_qa --dataset arxiv_ai --model our --postfix -1 --num-workers=16
python -m run.run_hipporag_index --dataset arxiv_ai --config force_index=True
python -m run.run_qa --dataset arxiv_ai --model hipporag --postfix -1

ENV_FILE=.env.qwen python -m run.run_qa --dataset arxiv_ai --model io --postfix qwen-1
ENV_FILE=.env.qwen python -m run.run_qa --dataset arxiv_ai --model cot --postfix qwen-1
ENV_FILE=.env.qwen python -m run.run_qa --dataset arxiv_ai --model sc --postfix qwen-1
ENV_FILE=.env.qwen python -m run.run_qa --dataset arxiv_ai --model rag --postfix qwen-1 --num-workers=8
ENV_FILE=.env.qwen python -m run.run_qa --dataset arxiv_ai --model one-hop-kg --postfix qwen-1
ENV_FILE=.env.qwen python -m run.run_qa --dataset arxiv_ai --model one-hop-kg-rag --postfix qwen-1 --num-workers=8
ENV_FILE=.env.qwen python -m run.run_qa --dataset arxiv_ai --model cok --postfix qwen-1
ENV_FILE=.env.qwen python -m run.run_qa --dataset arxiv_ai --model tog --postfix qwen-1
ENV_FILE=.env.qwen python -m run.run_qa --dataset arxiv_ai --model tog2 --postfix qwen-1
ENV_FILE=.env.qwen python -m run.run_qa --dataset arxiv_ai --model pog --postfix qwen-1
ENV_FILE=.env.qwen python -m run.run_qa --dataset arxiv_ai --model rog --postfix qwen-1
ENV_FILE=.env.qwen python -m run.run_qa --dataset arxiv_ai --model r2kg --postfix qwen-1
ENV_FILE=.env.qwen python -m run.run_qa --dataset arxiv_ai --model our --postfix qwen-1 --num-workers=16
ENV_FILE=.env.qwen python -m run.run_hipporag_index --dataset arxiv_ai --config force_index=True
ENV_FILE=.env.qwen python -m run.run_qa --dataset arxiv_ai --model hipporag --postfix qwen-1

ENV_FILE=.env.llama8 python -m run.run_qa --dataset arxiv_ai --model io --postfix llama8-1
ENV_FILE=.env.llama8 python -m run.run_qa --dataset arxiv_ai --model cot --postfix llama8-1
ENV_FILE=.env.llama8 python -m run.run_qa --dataset arxiv_ai --model sc --postfix llama8-1
ENV_FILE=.env.llama8 python -m run.run_qa --dataset arxiv_ai --model rag --postfix llama8-1 --num-workers=8
ENV_FILE=.env.llama8 python -m run.run_qa --dataset arxiv_ai --model one-hop-kg --postfix llama8-1
ENV_FILE=.env.llama8 python -m run.run_qa --dataset arxiv_ai --model one-hop-kg-rag --postfix llama8-1 --num-workers=8
ENV_FILE=.env.llama8 python -m run.run_qa --dataset arxiv_ai --model cok --postfix llama8-1
ENV_FILE=.env.llama8 python -m run.run_qa --dataset arxiv_ai --model tog --postfix llama8-1
ENV_FILE=.env.llama8 python -m run.run_qa --dataset arxiv_ai --model tog2 --postfix llama8-1
ENV_FILE=.env.llama8 python -m run.run_qa --dataset arxiv_ai --model pog --postfix llama8-1
ENV_FILE=.env.llama8 python -m run.run_qa --dataset arxiv_ai --model rog --postfix llama8-1
ENV_FILE=.env.llama8 python -m run.run_qa --dataset arxiv_ai --model r2kg --postfix llama8-1
ENV_FILE=.env.llama8 python -m run.run_qa --dataset arxiv_ai --model our --postfix llama8-1 --num-workers=16
ENV_FILE=.env.llama8 python -m run.run_hipporag_index --dataset arxiv_ai --config force_index=True
ENV_FILE=.env.llama8 python -m run.run_qa --dataset arxiv_ai --model hipporag --postfix llama8-1

ENV_FILE=.env.deepseek python -m run.run_qa --dataset arxiv_ai --model io --postfix deepseek-1
ENV_FILE=.env.deepseek python -m run.run_qa --dataset arxiv_ai --model cot --postfix deepseek-1
ENV_FILE=.env.deepseek python -m run.run_qa --dataset arxiv_ai --model sc --postfix deepseek-1
ENV_FILE=.env.deepseek python -m run.run_qa --dataset arxiv_ai --model rag --postfix deepseek-1 --num-workers=8
ENV_FILE=.env.deepseek python -m run.run_qa --dataset arxiv_ai --model one-hop-kg --postfix deepseek-1
ENV_FILE=.env.deepseek python -m run.run_qa --dataset arxiv_ai --model one-hop-kg-rag --postfix deepseek-1 --num-workers=8
ENV_FILE=.env.deepseek python -m run.run_qa --dataset arxiv_ai --model cok --postfix deepseek-1
ENV_FILE=.env.deepseek python -m run.run_qa --dataset arxiv_ai --model tog --postfix deepseek-1
ENV_FILE=.env.deepseek python -m run.run_qa --dataset arxiv_ai --model tog2 --postfix deepseek-1
ENV_FILE=.env.deepseek python -m run.run_qa --dataset arxiv_ai --model pog --postfix deepseek-1
ENV_FILE=.env.deepseek python -m run.run_qa --dataset arxiv_ai --model rog --postfix deepseek-1
ENV_FILE=.env.deepseek python -m run.run_qa --dataset arxiv_ai --model r2kg --postfix deepseek-1
ENV_FILE=.env.deepseek python -m run.run_qa --dataset arxiv_ai --model our --postfix deepseek-1 --num-workers=16
ENV_FILE=.env.deepseek python -m run.run_hipporag_index --dataset arxiv_ai --config force_index=True
ENV_FILE=.env.deepseek python -m run.run_qa --dataset arxiv_ai --model hipporag --postfix deepseek-1
