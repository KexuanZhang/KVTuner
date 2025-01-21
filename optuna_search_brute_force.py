import warnings
warnings.filterwarnings("ignore")
import torch
import random
import argparse
import torch
import optuna
import lm_eval
from lm_eval.models.huggingface_quant import HFLM_Quant
import logging
import sys

# For reproducibility
random.seed(0)
torch.manual_seed(0)
CACHE_DIR = "./models_storage"

TEMPLATE_KV_QUANT_CONFIG = [
    {'nbits_key': 8, 'nbits_value': 8},
    {'nbits_key': 8, 'nbits_value': 4},
    {'nbits_key': 4, 'nbits_value': 4},
    {'nbits_key': 4, 'nbits_value': 2},
    {'nbits_key': 2, 'nbits_value': 2},
]

# THIS IS FOR PER_TOKEN QUANTIZATION
LLAMA3_IMPORTANT_LAYERS = [0, 3, 5, 7, 12, 15, 22, 26, 30, 31]
LLAMA3_MEDIUM_LAYERS = [6, 8, 9, 10, 11, 13, 14, 25, 27, 28, 29]

QWEN_IMPORTANT_LAYERS = [0, 18, 20, 27, 29, 35]
QWEN_MEDIUM_LAYERS = [3, 4, 5]

global_args = {}
model = None
tokenizer = None
dataset = None

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model_name', type=str, default="meta-llama/Llama-2-7b-hf")
    # parser.add_argument('--model_name', type=str, default="Qwen/Qwen2.5-3B-Instruct-AWQ")
    # parser.add_argument('--model_name', type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument('--model_name', type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument('--residual_length', type=int, default=0)
    parser.add_argument('--group_size', type=int, default=-1)
    parser.add_argument('--asym', type=bool, default=True)
    # in Vanilla, 0 for per-token, 1 for per-channel, we have to use per-channel there as residual_length is 0
    parser.add_argument('--axis_key', type=int, default=0)
    parser.add_argument('--axis_value', type=int, default=0)
    parser.add_argument('--limit', type=int, default=20)
    parser.add_argument('--num_fewshots', type=int, default=4)
    parser.add_argument('--max_per_layer_scale', type=int, default=8)
    parser.add_argument('--n_trials', type=int, default=100)
    parser.add_argument('--device', type=str, default="cuda")
    return parser.parse_args(args)



def run_gsm8k(residual_length: int, group_size: int, asym: bool, axis_key: int, axis_value: int, per_layer_config: dict, model_name: str, num_fewshots: int, limit: int, device: str):
    results = lm_eval.simple_evaluate(
        model='hf-quant',
        model_args={
            'pretrained': model_name,
            'nbits_key': -1,
            'nbits_value': -1,
            'residual_length': residual_length,
            'q_group_size': group_size,
            'asym': asym,
            'axis_key': axis_key,
            'axis_value': axis_value,
            'dtype': torch.bfloat16,
            'force_quant': True,
            'per_layer_quant': True,
            'per_layer_config': per_layer_config,
            'quantilizer': 'vanilla',
        },
        tasks=["gsm8k"],
        num_fewshot=num_fewshots,
        limit=limit,
        device=device
    )
    print(results['results']['gsm8k']['exact_match,flexible-extract'])
    return float(results['results']['gsm8k']['exact_match,flexible-extract'])

def objective(trial):
    tot_layers = 32 if 'llama' in model.lower() else 36
    
    per_layer_config = {}
    tot_scale = 0
    
    for layer in range(0, tot_layers):
        config_current_layer = trial.suggest_int('layer_{}'.format(layer), 0, len(TEMPLATE_KV_QUANT_CONFIG) - 1)
        per_layer_config[layer] = TEMPLATE_KV_QUANT_CONFIG[config_current_layer]
        tot_scale += per_layer_config[layer]['nbits_key'] + per_layer_config[layer]['nbits_value']
    
    # Constraints which are considered feasible if less than or equal to zero.
    tot_scale /= tot_layers * 2
    c = tot_scale - global_args['max_per_layer_scale']
    
    print('constraints:', c)
    
    trial.set_user_attr('constraints', (c, ))
    
    accuracy = run_gsm8k(global_args['residual_length'], global_args['group_size'], global_args['asym'], global_args['axis_key'], global_args['axis_value'], per_layer_config, 
                        global_args['model_name'], global_args['num_fewshots'], global_args['limit'], global_args['device'])
    
    return accuracy, tot_scale

def constraints(trial):
    return trial.user_attrs["constraints"]

if __name__ == "__main__":
    args = parse_args()
    model = args.model_name
    
    global_args['model_name'] = args.model_name
    global_args['residual_length'] = args.residual_length
    global_args['group_size'] = args.group_size
    global_args['asym'] = args.asym
    global_args['axis_key'] = args.axis_key
    global_args['axis_value'] = args.axis_value
    global_args['limit'] = args.limit
    global_args['num_fewshots'] = args.num_fewshots
    global_args['device'] = args.device
    global_args['max_per_layer_scale'] = args.max_per_layer_scale
    
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "{}_gsm8k_l{}_search_{}_m{}_brute_force_{}".format(model.replace("/", "_"), args.limit, args.device.replace(":", ""), args.max_per_layer_scale, 'per_token' if args.group_size else 'kivi')
    storage_name = "sqlite:///{}.db".format(study_name)
    sampler = optuna.samplers.NSGAIISampler(constraints_func=constraints)
    study = optuna.create_study(directions=["maximize", "minimize"], study_name=study_name, storage=storage_name, sampler=sampler)
    study.optimize(objective, n_trials=args.n_trials)

    # print(study.best_params)
    # print(study.best_value)
