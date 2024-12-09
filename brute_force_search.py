import warnings
warnings.filterwarnings("ignore")
import torch
import random
import argparse
import torch
import lm_eval
from lm_eval.models.huggingface_quant import HFLM_Quant

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

LLAMA3_IMPORTANT_LAYERS = [0, 3, 5, 7, 12, 15, 22, 26, 30, 31]
LLAMA3_MEDIUM_LAYERS = [6, 8, 9, 10, 11, 13, 14, 25, 27, 28, 29]

QWEN_IMPORTANT_LAYERS = [0, 3, 13, 19, 27]

global_args = {}
model = None
tokenizer = None
dataset = None

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model_name', type=str, default="meta-llama/Llama-2-7b-hf")
    # parser.add_argument('--model_name', type=str, default="Qwen/Qwen2.5-3B-Instruct-AWQ")
    # parser.add_argument('--model_name', type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument('--model_name', type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument('--residual_length', type=int, default=0)
    parser.add_argument('--group_size', type=int, default=32)
    parser.add_argument('--asym', type=bool, default=True)
    # in Vanilla, 0 for per-token, 1 for per-channel, we have to use per-channel there as residual_length is 0
    parser.add_argument('--axis_key', type=int, default=0)
    parser.add_argument('--axis_value', type=int, default=0)
    parser.add_argument('--limit', type=int, default=200)
    parser.add_argument('--num_fewshots', type=int, default=0)
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

def build_per_layer_config(model: str, config_high: int, config_medium: int, config_low: int):
    important_layers = []
    if 'llama' in model.lower():
        important_layers = LLAMA3_IMPORTANT_LAYERS
        medium_layers = LLAMA3_MEDIUM_LAYERS
    if 'qwen' in model.lower():
        important_layers = QWEN_IMPORTANT_LAYERS
        medium_layers = []
    per_layer_config = {}
    tot_scale = 0
    tot_layers = 32 if 'llama' in model.lower() else 28
    for layer in range(0, tot_layers):
        if layer in important_layers:
            per_layer_config[layer] = TEMPLATE_KV_QUANT_CONFIG[config_high]
        elif layer in medium_layers:
            per_layer_config[layer] = TEMPLATE_KV_QUANT_CONFIG[config_medium]
        else:
            per_layer_config[layer] = TEMPLATE_KV_QUANT_CONFIG[config_low]
        tot_scale += per_layer_config[layer]['nbits_key'] + per_layer_config[layer]['nbits_value']
    tot_scale /= tot_layers * 2
    return per_layer_config, tot_scale


if __name__ == "__main__":
    args = parse_args()
    model_name = args.model_name
    
    global_args['model_name'] = model_name
    global_args['residual_length'] = args.residual_length
    global_args['group_size'] = args.group_size
    global_args['asym'] = args.asym
    global_args['axis_key'] = args.axis_key
    global_args['axis_value'] = args.axis_value
    global_args['limit'] = args.limit
    global_args['num_fewshots'] = args.num_fewshots
    global_args['device'] = args.device
    
    print(global_args)
    valid_params = []
    for profile_high in range(5):
        for profile_medium in range(profile_high, 5):
            for profile_low in range(profile_medium, 5):
                valid_params.append((profile_high, profile_medium, profile_low))
    
    for profile_high, profile_medium, profile_low in valid_params:
        per_layer_config, tot_scale = build_per_layer_config(args.model_name, profile_high, profile_medium, profile_low)
        accuracy = run_gsm8k(global_args['residual_length'], global_args['group_size'], global_args['asym'], global_args['axis_key'], global_args['axis_value'], per_layer_config, global_args['model_name'], global_args['num_fewshots'], global_args['limit'], global_args['device'])
        print(f"Profile: {profile_high}, {profile_medium}, {profile_low}, Accuracy: {accuracy}, Scale: {tot_scale}")
        print("")