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
import json
import os
from transformers import AutoConfig

# For reproducibility
random.seed(0)
torch.manual_seed(0)
CACHE_DIR = "./models_storage"

# Global variables for flexible evaluation
global_evaluation_mode = "single_task"
global_evaluation_tasks = ["gsm8k"]
global_custom_dataset_path = None

def get_task_suite(domain: str):
    """Get task suite for different domains"""
    task_suites = {
        'general': ['hellaswag', 'arc_easy', 'winogrande', 'boolq'],
        'reasoning': ['gsm8k', 'arc_challenge', 'piqa', 'hellaswag'],
        'knowledge': ['mmlu', 'truthfulqa_mc2', 'arc_challenge'],
        'code': ['humaneval', 'mbpp'],
        'reading': ['drop', 'boolq', 'piqa'],
        'math': ['gsm8k', 'math_qa'],
        'comprehensive': ['hellaswag', 'arc_challenge', 'gsm8k', 'winogrande', 'boolq', 'piqa']
    }
    return task_suites.get(domain, ['gsm8k'])

def get_metric_key(task_name: str):
    """Get the appropriate metric for different tasks"""
    task_metrics = {
        'gsm8k': 'exact_match,flexible-extract',
        'hellaswag': 'acc_norm',
        'arc_easy': 'acc',
        'arc_challenge': 'acc_norm', 
        'winogrande': 'acc',
        'boolq': 'acc',
        'piqa': 'acc_norm',
        'humaneval': 'pass@1',
        'mbpp': 'pass@1',
        'mmlu': 'acc',
        'truthfulqa_mc2': 'acc',
        'drop': 'f1',
        'math_qa': 'acc',
    }
    return task_metrics.get(task_name, 'acc')  # Default to 'acc'

def setup_local_model_config(model_path: str, num_layers: int = None):
    """Automatically configure a local model"""
    if num_layers is None:
        try:
            config = AutoConfig.from_pretrained(model_path)
            num_layers = config.num_hidden_layers
            print(f"Auto-detected {num_layers} layers for model: {model_path}")
        except Exception as e:
            print(f"Could not auto-detect layers: {e}")
            print("Please specify --local_model_layers manually")
            return None
    
    model_key = 'local-model'
    
    # Add to TOT_LAYER
    TOT_LAYER[model_key] = num_layers
    
    # Create simple layer grouping (each layer in its own group initially)
    LAYER_GROUPING_CONFIG[model_key] = {
        'per-token-asym': [[i] for i in range(num_layers)],
        'per-channel-asym': [[i] for i in range(num_layers)],
    }
    
    # Add conservative special layers (first and last layers get more conservative options)
    SPECIAL_LAYERS[model_key] = {
        'per-token-asym': {
            (0,): ['KV8', 'K8V4', 'K8V2', 'K4V2'],  # Conservative for first layer
            (num_layers-1,): ['KV8', 'K8V4', 'K8V2', 'K4V2'],  # Conservative for last layer
        },
        'per-channel-asym': {
            (0, num_layers-1): ['KV8', 'K4V8', 'KV4', 'K4V2'],  # Conservative for first and last
        },
    }
    
    print(f"Configured local model '{model_key}' with {num_layers} layers")
    return model_key



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
            (0,): ['KV8', 'K4V8', 'KV4', 'K4V2', 'KV2'],
        },
        'per-channel-asym': {
            (0,): ['KV8', 'K4V8', 'KV4', 'K2V4', 'KV2'],
            (1, 2, 3, 7, 29, 31): ['KV8', 'K4V8', 'KV4', 'K4V2', 'KV2'],
        },
    },
    'Mistral-7B-Instruct-v0.3': {
        'per-token-asym': {
            (0,): ['KV8', 'K4V8', 'KV4', 'K2V4', 'KV2'],
        },
        'per-channel-asym': {
            # (0,): ['KV8', 'K4V8', 'KV4', 'K2V4', 'KV2'],
            (0, 1, 31): ['KV8', 'K4V8', 'KV4', 'K2V4', 'KV2'], # fix: grouping
            (2, 3, 4, 6, 7, 8, 9, 10, 14, 18, 27, 29): ['KV8', 'K4V8', 'KV4', 'K4V2', 'KV2'],
        },
    },
    'Qwen2.5-3B-Instruct': {
        'per-token-asym': {
            (0,): ['KV8', 'K8V4', 'K8V2', 'K4V2', 'KV2'],
            (18, 27, 29): ['KV8', 'K8V4', 'K8V2', 'KV4', 'K4V2', 'KV2'],
        },
        'per-channel-asym': {
            (0, 1, 2, 4, 34, 35): ['KV8', 'K4V8', 'KV4', 'K2V4', 'KV2'],
            (3, 6, 11, 13, 23): ['KV8', 'K4V8', 'KV4', 'K4V2', 'KV2'],
        },
    },
    'Qwen2.5-7B-Instruct': {
        'per-token-asym': {
            (0,): ['KV8', 'K8V4', 'K8V2', 'K4V2', 'KV2'],
            (3, 13, 27): ['KV8', 'K8V4', 'K8V2', 'KV4', 'K4V2', 'KV2'],
        },
        'per-channel-asym': {
            (0, 1, 2, 3): ['KV8', 'K4V8', 'KV4', 'K2V4', 'KV2'],
            (6,): ['KV8', 'K4V8', 'KV4', 'K4V2', 'KV2'],
        },
    },
    'Qwen2.5-14B-Instruct': {
        'per-token-asym': {
            # no layers listed
        },
        'per-channel-asym': {
            (0, 1, 2, 3, 4): ['KV8', 'K4V8', 'KV4', 'K2V4', 'KV2'],
            (5, 6, 8, 9, 12): ['KV8', 'K4V8', 'KV4', 'K4V2', 'KV2'],
        },
    },
    'Qwen2.5-32B-Instruct': {
        'per-token-asym': {
            # no layers listed
        },
        'per-channel-asym': {
            (0, 1, 2, 3, 4, 11): ['KV8', 'K4V8', 'KV4', 'K2V4', 'KV2'],
            (5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 19, 20, 21, 23, 26, 27, 28, 32): ['KV8', 'K4V8', 'KV4', 'K4V2', 'KV2'],
            (63,): ['KV8', 'K8V4', 'KV4', 'K2V4', 'KV2'],
        },
    },
}


TOT_LAYER = {
    'Meta-Llama-3.1-8B-Instruct': 32,
    'Mistral-7B-Instruct-v0.3': 32,
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
debug_constraint = False

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
    parser.add_argument('--debug_constraint', default=False, action='store_true')
    
    # New evaluation options
    parser.add_argument('--evaluation_task', type=str, default="gsm8k",
                       help='Single evaluation task (default: gsm8k)')
    parser.add_argument('--evaluation_tasks', type=str, nargs='+',
                       help='Multiple evaluation tasks (overrides --evaluation_task)')
    parser.add_argument('--domain', type=str, choices=['general', 'reasoning', 'knowledge', 'code', 'reading', 'math', 'comprehensive'],
                       help='Pre-defined task domain (overrides other task options)')
    parser.add_argument('--custom_dataset_path', type=str,
                       help='Path to custom evaluation dataset JSON file')
    parser.add_argument('--local_model_layers', type=int,
                       help='Number of layers in local model (auto-detects if not provided)')
    
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
        if debug_constraint:
            group_quant_template = [i for i in group_quant_template if i != 'KV2'] # remove KV2
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
            'device_map': 'auto',
            'parallelize': True,
        },
        tasks=["gsm8k"],
        num_fewshot=num_fewshots,
        limit=limit,
        # device=device
    )
    print(results['results']['gsm8k']['exact_match,flexible-extract'])
    return float(results['results']['gsm8k']['exact_match,flexible-extract'])


def run_single_task_evaluation(per_layer_config: dict, model_name: str, task_name: str, num_fewshots: int, limit: int, device: str):
    """Run evaluation on a single task"""
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
            'device_map': 'auto',
            'parallelize': True,
        },
        tasks=[task_name],
        num_fewshot=num_fewshots,
        limit=limit,
    )
    
    metric_key = get_metric_key(task_name)
    score = float(results['results'][task_name][metric_key])
    print(f"{task_name} {metric_key}: {score}")
    return score


def run_multi_task_evaluation(per_layer_config: dict, model_name: str, tasks: list, num_fewshots: int, limit: int, device: str):
    """Run evaluation on multiple tasks and return average score"""
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
            'device_map': 'auto',
            'parallelize': True,
        },
        tasks=tasks,
        num_fewshot=num_fewshots,
        limit=limit,
    )
    
    # Compute average score across tasks
    total_score = 0
    task_scores = {}
    
    for task in tasks:
        metric_key = get_metric_key(task)
        score = float(results['results'][task][metric_key])
        task_scores[task] = score
        total_score += score
        print(f"{task} {metric_key}: {score}")
    
    average_score = total_score / len(tasks)
    print(f"Average score across {len(tasks)} tasks: {average_score}")
    return average_score


def run_custom_evaluation(per_layer_config: dict, model_name: str, dataset_path: str, limit: int):
    """Run evaluation on custom dataset (placeholder implementation)"""
    # This is a simplified implementation - would need full integration with quantization
    # For now, return a placeholder score
    print(f"Custom dataset evaluation not fully implemented yet. Using placeholder score.")
    print(f"Dataset path: {dataset_path}")
    print(f"Per-layer config: {len(per_layer_config)} layers configured")
    
    # Placeholder: return a score based on compression level
    # More compressed = lower score (simulating quality degradation)
    total_bits = sum(config.get('nbits_key', 8) + config.get('nbits_value', 8) 
                    for config in per_layer_config.values())
    avg_bits = total_bits / (len(per_layer_config) * 2)
    placeholder_score = max(0.3, min(0.95, 0.95 - (8 - avg_bits) * 0.1))
    
    print(f"Custom evaluation placeholder score: {placeholder_score}")
    return placeholder_score


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
    
    if not debug_constraint:
        trial.set_user_attr('constraints', (c, ))
    
    # Flexible evaluation based on configuration
    if global_evaluation_mode == "custom":
        accuracy = run_custom_evaluation(per_layer_config, model, global_custom_dataset_path, limit)
    elif global_evaluation_mode == "multi_task":
        accuracy = run_multi_task_evaluation(per_layer_config, model, global_evaluation_tasks, num_fewshots, limit, device)
    else:  # single_task
        if global_evaluation_tasks[0] == "gsm8k":
            accuracy = run_gsm8k(per_layer_config, model, num_fewshots, limit, device)  # Keep original for GSM8K
        else:
            accuracy = run_single_task_evaluation(per_layer_config, model, global_evaluation_tasks[0], num_fewshots, limit, device)
    
    c2 = 0.6 - accuracy
    
    if debug_constraint:
        print('c2 = ', c2)
        trial.set_user_attr('constraints', (c, c2))
    
    print(f"Trial result: accuracy={accuracy:.4f}, tot_scale={tot_scale:.2f}")
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
    debug_constraint = args.debug_constraint
    
    # Determine evaluation tasks
    evaluation_tasks = None
    evaluation_mode = "single_task"
    
    if args.custom_dataset_path:
        evaluation_mode = "custom"
        print(f"Using custom dataset: {args.custom_dataset_path}")
    elif args.domain:
        evaluation_tasks = get_task_suite(args.domain)
        evaluation_mode = "multi_task"
        print(f"Using domain '{args.domain}' tasks: {evaluation_tasks}")
    elif args.evaluation_tasks:
        evaluation_tasks = args.evaluation_tasks
        evaluation_mode = "multi_task" if len(evaluation_tasks) > 1 else "single_task"
        print(f"Using specified tasks: {evaluation_tasks}")
    else:
        evaluation_tasks = [args.evaluation_task]
        evaluation_mode = "single_task"
        print(f"Using single task: {args.evaluation_task}")
    
    # Setup local model if needed
    if args.local_model_layers or (not model.startswith(('meta-llama', 'Qwen', 'mistralai'))):
        print("Setting up local model configuration...")
        model_key = setup_local_model_config(model, args.local_model_layers)
        if model_key is None:
            sys.exit(1)
    
    # Create study name based on evaluation mode
    task_suffix = ""
    if evaluation_mode == "custom":
        task_suffix = "CUSTOM"
    elif evaluation_mode == "multi_task":
        task_suffix = "_".join(evaluation_tasks[:3])  # Use first 3 tasks in name
        if len(evaluation_tasks) > 3:
            task_suffix += f"_PLUS{len(evaluation_tasks)-3}"
    else:
        task_suffix = evaluation_tasks[0].upper()
    
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "OPTUNA_SEARCH_ADAPTIVE_{}_{}_FIRST{}_{}SHOTS_MAXSCALE{}_SCHEME{}".format(
        model.replace("/", "_"), task_suffix, limit, num_fewshots, max_per_layer_scale, quant_scheme)
    storage_name = "sqlite:///{}.db".format(study_name)
    sampler = optuna.samplers.NSGAIISampler(constraints_func=constraints)
    study = optuna.create_study(directions=["maximize", "minimize"], study_name=study_name, storage=storage_name, sampler=sampler)
    
    print("Configuration:")
    print(f"  Model: {model}")
    print(f"  Evaluation mode: {evaluation_mode}")
    print(f"  Tasks: {evaluation_tasks if evaluation_tasks else args.custom_dataset_path}")
    print(f"  Max scale: {max_per_layer_scale}")
    print(f"  Trials: {args.n_trials}")
    print(f"  Samples per trial: {limit}")
    
    print('\nPreparing layer grouping config...')
    prepare_layer_grouping_config(model, quant_scheme)
    print('Layer grouping: ', current_layer_grouping)
    print('Special layers: ', current_special_layers)
    print('Grouping quant template: ', current_grouping_quant_template)
    print('Total layers: ', current_tot_layers)
    
    # Store evaluation config in global variables for use in objective function
    global_evaluation_mode = evaluation_mode
    global_evaluation_tasks = evaluation_tasks
    global_custom_dataset_path = args.custom_dataset_path
    
    study.optimize(objective, n_trials=args.n_trials)