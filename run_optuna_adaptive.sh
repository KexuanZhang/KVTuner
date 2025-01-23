# WS-12: optuna search adaptive, llama3
python3 optuna_search_adaptive.py --limit=200 --n_trials=1000 --device=cuda:0 --max_per_layer_scale=6 | tee optuna_llama3_adaptive_new_1k_per_token_limit6.log
python3 optuna_search_adaptive.py --limit=200 --n_trials=1000 --device=cuda:1 --max_per_layer_scale=4 | tee optuna_llama3_adaptive_new_1k_per_token_limit4.log

# WS-15: optuna search adaptive, qwen2 3b
python3 optuna_search_adaptive.py --model_name='Qwen/Qwen2.5-3B-Instruct-AWQ' --limit=200 --n_trials=1000 --device=cuda:0 --max_per_layer_scale=6 | tee optuna_qwen2_adaptive_new_1k_per_token_limit6.log
python3 optuna_search_adaptive.py --model_name='Qwen/Qwen2.5-3B-Instruct-AWQ' --limit=200 --n_trials=1000 --device=cuda:1 --max_per_layer_scale=4 | tee optuna_qwen2_adaptive_new_1k_per_token_limit4.log

# WS-9 optuna search adaptive, qwen2 7b
python3 optuna_search_adaptive.py --model_name='Qwen/Qwen2.5-7B-Instruct' --limit=200 --n_trials=1000 --device=cuda:0 --max_per_layer_scale=6 | tee optuna_qwen2_7b_adaptive_new_1k_per_token_limit6.log
python3 optuna_search_adaptive.py --model_name='Qwen/Qwen2.5-7B-Instruct' --limit=200 --n_trials=1000 --device=cuda:1 --max_per_layer_scale=4 | tee optuna_qwen2_7b_adaptive_new_1k_per_token_limit4.log

# WS-3 optuna search adaptive, mistralai/Mistral-7B-Instruct-v0.3
python3 optuna_search_adaptive.py --model_name='mistralai/Mistral-7B-Instruct-v0.3' --limit=200 --n_trials=1000 --device=cuda:0 --max_per_layer_scale=6 | tee optuna_mistral_7b_adaptive_new_1k_per_token_limit6.log
python3 optuna_search_adaptive.py --model_name='mistralai/Mistral-7B-Instruct-v0.3' --limit=200 --n_trials=1000 --device=cuda:1 --max_per_layer_scale=4 | tee optuna_mistral_7b_adaptive_new_1k_per_token_limit4.log