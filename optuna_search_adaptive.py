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



LAYER_GROUPING_CONFIG = {
    'Meta-Llama-3.1-8B-Instruct': {
        'per-token-asym': [[0], [1, 2, 3, 4, 7, 13, 18, 25, 27, 31], [5, 6, 12, 21, 26, 28], [8, 9, 10, 11, 14, 15, 16, 17, 20, 30], [19, 22], [23, 24, 29]],
        'per-channel-asym': [[0], [1, 2, 3, 7, 29, 31], [4, 25, 27], [5, 21, 23, 24], [6, 8, 9, 10, 11, 12, 14, 15, 16, 18, 19, 20, 22, 26, 28, 30], [13, 17]],
    },
    'Mistral-7B-Instruct-v0.3': {
        'per-token-asym': [[0], [1, 2], [3, 4, 23, 31], [5, 6], [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 29, 30]],
        'per-channel-asym': [[0, 1, 31], [2, 3, 4], [6, 27, 29], [7, 8, 10, 18], [9, 14], [5, 21, 22, 23, 24, 25, 26, 28, 30], [11, 12, 13, 15, 17, 19, 20], [16]],
    },
    'Qwen2.5-3B-Instruct': {
        'per-token-asym': [[0], [1, 3, 4, 5, 6, 8, 9, 12, 13, 15, 20], [2, 14, 23, 35], [7, 11, 16, 25, 28, 32], [10, 19, 24, 26, 33], [17, 30, 31, 34], [21, 22], [18, 27, 29]],
        'per-channel-asym': [[0, 1], [2, 4], [34, 35], [3, 6, 11, 13, 23], [5, 7, 25, 32, 33], [8, 16, 18, 21, 22, 24, 26, 27, 30], [9, 10, 14, 15, 17, 19, 20, 29, 31], [12, 28]],
    },
    'Qwen2.5-7B-Instruct': {
        'per-token-asym': [[0], [1, 2, 4, 5, 25], [6, 19], [7, 10, 11, 15, 23], [8, 24], [9, 12, 16, 17, 18, 21, 22, 26], [14, 20], [3, 13, 27]],
        'per-channel-asym': [[0, 2], [1, 3], [4, 5, 12, 22, 23, 24, 25], [7, 9, 10, 13, 14, 16, 18, 19, 20, 21, 27], [8, 26], [11, 15, 17], [6]],
    },
    'Qwen2.5-14B-Instruct': {
        'per-token-asym': [[0, 1, 2, 6, 11, 12, 19, 23, 24, 25, 41], [3, 4, 5, 8], [7, 10, 15], [9, 13, 14, 31, 38, 39], [16, 17, 18, 20, 21, 27, 28, 30, 32, 33, 34, 35, 36, 37, 40, 42, 43, 44, 46, 47], [22, 26, 29, 45]],
        'per-channel-asym': [[0, 2], [1, 3, 4], [5, 6, 8, 9, 12], [7, 10, 13, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35, 36, 37, 38, 44, 45, 46, 47], [11, 25, 41, 42], [14, 39, 40, 43], [22, 34]],
    },
    'Qwen2.5-32B-Instruct': {
        'per-token-asym': [[0, 2, 11, 12, 15, 33, 54, 57], [1, 5, 7, 8, 9, 10, 13, 14, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 55, 56, 58, 59, 60, 61, 62, 63], [3, 4], [6, 16]],
        'per-channel-asym': [[0, 1, 2, 3, 4], [11], [5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 32], [13, 15, 17, 22, 24, 25, 29, 30, 31, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62], [63]]
    }
}

SPECIAL_LAYERS = {
    'Meta-Llama-3.1-8B-Instruct': {
        'per-token-asym': {
            (0,): ['K4V8'],
        },
        'per-channel-asym': {
            (0,): ['K4V8', 'K2V4'],
            (1, 2, 3, 7, 29, 31): ['K4V8'],
        },
    },
    'Mistral-7B-Instruct-v0.3': {
        'per-token-asym': {
            (0,): ['K4V8', 'K2V4'],
        },
        'per-channel-asym': {
            (0,): ['K4V8', 'K2V4'],
            (2, 3, 4, 6, 7, 8, 9, 10, 14, 18, 27, 29): ['K4V8'],
        },
    },
    'Qwen2.5-3B-Instruct': {
        'per-token-asym': {
            (0,): ['K8V2'],
            (18, 27, 29): ['K8V2'],
        },
        'per-channel-asym': {
            (0, 1, 2, 4, 34, 35): ['K4V8', 'K2V4'],
            (3, 6, 11, 13, 23): ['K4V8'],
        },
    },
    'Qwen2.5-7B-Instruct': {
        'per-token-asym': {
            (0,): ['K8V2'],
            (3, 13, 27): ['K8V2'],
        },
        'per-channel-asym': {
            (0, 1, 2, 3): ['K4V8', 'K2V4'],
            (6,): ['K4V8'],
        },
    },
    'Qwen2.5-14B-Instruct': {
        'per-channel-asym': {
            (0, 1, 2, 3, 4): ['K4V8', 'K2V4'],
            (5, 6, 8, 9, 12): ['K4V8'],
        },
    },
    'Qwen2.5-32B-Instruct': {
        'per-channel-asym': {
            (0, 1, 2, 3, 4, 11): ['K4V8', 'K2V4'],
            (5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 19, 20, 21, 23, 26, 27, 28, 32): ['K4V8'],
            (63,): ['K2V4'],
        },
    },
}

TOT_LAYER = {
    'Meta-Llama-3.1-8B-Instruct': 32,
    'Mistral-7B-Instruct-v0.3': 36,
    'Qwen2.5-3B-Instruct': 36,
    'Qwen2.5-7B-Instruct': 28,
    'Qwen2.5-14B-Instruct': 48,
    'Qwen2.5-32B-Instruct': 64,
}

STANDARD_KV_QUANT_CONFIG = ['KV8', 'K8V4', 'KV4', 'K4V2', 'KV2']

global_args = {}
model = None
tokenizer = None
dataset = None

num_fewshots = None
limit = None
device = None

quant_scheme = None
max_per_layer_scale = None

current_layer_grouping = []
current_special_layers = {}
current_grouping_quant_template = []
current_tot_layers = -1


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model_name', type=str, default="meta-llama/Llama-2-7b-hf")
    # parser.add_argument('--model_name', type=str, default="Qwen/Qwen2.5-3B-Instruct-AWQ")
    # parser.add_argument('--model_name', type=str, default="Qwen/Qwen2.5-7B-Instruct")
    # parser.add_argument('--model_name', type=str, default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument('--model_name', type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    # parser.add_argument('--residual_length', type=int, default=0)
    # parser.add_argument('--group_size', type=int, default=-1)
    parser.add_argument('--quant_scheme', type=str, default="per-token-asym") # per-token-asym or per-channel-asym
    parser.add_argument('--asym', type=bool, default=True)
    # in Vanilla, 0 for per-token, 1 for per-channel, we have to use per-channel there as residual_length is 0
    parser.add_argument('--axis_key', type=int, default=0)
    parser.add_argument('--axis_value', type=int, default=0)
    parser.add_argument('--limit', type=int, default=20)
    parser.add_argument('--num_fewshots', type=int, default=4)
    parser.add_argument('--max_per_layer_scale', type=str, default='8')
    parser.add_argument('--n_trials', type=int, default=100)
    parser.add_argument('--device', type=str, default="cuda")
    return parser.parse_args(args)


def parse_quant_config(quant_config: str):
    if len(quant_config) == 3:
        precision = int(quant_config[2])
        return {'nbits_key': precision, 'nbits_value': precision}
    precision_key = int(quant_config[1])
    precision_value = int(quant_config[3])
    return {'nbits_key': precision_key, 'nbits_value': precision_value}

def prepare_layer_grouping_config(model_name: str, quant_scheme: str):
    model_name = model_name.split('/')[-1]
    model_name = model_name.replace('-AWQ', '') # Qwen2.5-3B-Instruct-AWQ -> Qwen2.5-3B-Instruct
    global current_layer_grouping, current_special_layers, current_grouping_quant_template, current_tot_layers
    current_layer_grouping = LAYER_GROUPING_CONFIG[model_name][quant_scheme]
    current_special_layers = SPECIAL_LAYERS[model_name][quant_scheme]
    current_tot_layers = TOT_LAYER[model_name]
    # check if current_special_layers breaks the current_layer_grouping
    for group in current_layer_grouping:
        group_quant_template = STANDARD_KV_QUANT_CONFIG
        for layer in group:
            for special_layer in current_special_layers.keys():
                if layer in special_layer:
                    group_quant_template = current_special_layers[special_layer]
                    for other_layer in group:
                        if not other_layer in special_layer:
                            raise ValueError("Special layer {} breaks the layer grouping for model {}, quant scheme {}".format(special_layer, model_name, quant_scheme))
        current_grouping_quant_template.append(group_quant_template)

def run_gsm8k(per_layer_config: dict, model_name: str, num_fewshots: int, limit: int, device: str):
    results = lm_eval.simple_evaluate(
        model='hf-quant',
        model_args={
            'pretrained': model_name,
            'nbits_key': -1,
            'nbits_value': -1,
            'residual_length': 32 if quant_scheme == 'per-channel-asym' else 0,
            'q_group_size': 32 if quant_scheme == 'per-channel-asym' else -1,
            'asym': True,
            'axis_key': 1 if quant_scheme == 'per-channel-asym' else 0,
            'axis_value': 0,
            'dtype': torch.bfloat16,
            'force_quant': False,
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


def build_per_layer_config(config_list: int):
    per_layer_config = {}
    tot_scale = 0
    for i, config in enumerate(config_list):
        layers = current_layer_grouping[i]
        quant_config = parse_quant_config(current_grouping_quant_template[i][config])
        for layer in layers:
            per_layer_config[layer] = quant_config
        tot_scale += (quant_config['nbits_key'] + quant_config['nbits_value']) * len(layers)
    tot_scale /= current_tot_layers * 2
    return per_layer_config, tot_scale


def objective(trial):    
    config_list = []
    for i in range(0, len(current_layer_grouping)):
        config_current_layer = trial.suggest_int('group_{}'.format(i), 0, len(current_grouping_quant_template[i]) - 1)
        config_list.append(config_current_layer)
    
    per_layer_config, tot_scale = build_per_layer_config(config_list)
    
    # Constraints which are considered feasible if less than or equal to zero.
    
    c = tot_scale - max_per_layer_scale
    print('c = ', c)
    
    trial.set_user_attr('constraints', (c, ))
    
    accuracy = run_gsm8k(per_layer_config,  model, num_fewshots, limit, device)
    
    return accuracy, tot_scale

def constraints(trial):
    return trial.user_attrs["constraints"]

if __name__ == "__main__":
    args = parse_args()
    model = args.model_name
    quant_scheme = args.quant_scheme
    max_per_layer_scale = float(args.max_per_layer_scale)
    num_fewshots = args.num_fewshots
    limit = args.limit
    device = args.device
    
    
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "OPTUNA_SEARCH_ADAPTIVE_{}_GSM8K_FIRST{}_{}SHOTS_MAXSCALE{}_SCHEME{}".format(model.replace("/", "_"), limit, num_fewshots, max_per_layer_scale, quant_scheme)
    storage_name = "sqlite:///{}.db".format(study_name)
    sampler = optuna.samplers.NSGAIISampler(constraints_func=constraints)
    study = optuna.create_study(directions=["maximize", "minimize"], study_name=study_name, storage=storage_name, sampler=sampler)
    
    print(args)
    print('Preparing layer grouping config...')
    prepare_layer_grouping_config(model, quant_scheme)
    print('Layer grouping: ', current_layer_grouping)
    print('Special layers: ', current_special_layers)
    print('Grouping quant template: ', current_grouping_quant_template)
    print('Total layers: ', current_tot_layers)
    
    study.optimize(objective, n_trials=args.n_trials)