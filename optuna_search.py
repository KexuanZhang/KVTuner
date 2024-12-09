# LLaMA model with KIVI
import warnings
warnings.filterwarnings("ignore")
import torch
import random
import argparse
import torch
from flexible_quant.flexible_quantized_cache import FlexibleQuantizedCacheConfig, FlexibleVanillaQuantizedCache
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from transformers import AutoTokenizer
from datasets import load_dataset
from evals.gsm8k_utils import *
import tqdm
import optuna
import lm_eval
from lm_eval.models.huggingface_quant import HFLM_Quant

# For reproducibility
random.seed(0)
torch.manual_seed(0)
CACHE_DIR = "./models_storage"
LLAMA3_IMPORTANT_LAYERS = [0, 3, 5, 7, 12, 15, 22, 26, 30, 31]
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
    # in Vanilla, 0 for per-token, 1 for per-channel
    parser.add_argument('--axis_key', type=int, default=1)
    parser.add_argument('--axis_value', type=int, default=0)
    return parser.parse_args(args)



def run_gsm8k(residual_length: int, group_size: int, asym: bool, axis_key: int, axis_value: int, per_layer_config: dict, model_name: str):
    model = HFLM_Quant(
        pretrained=model_name,
        nbits_key=-1,
        nbits_value=-1,
        residual_length=residual_length,
        q_group_size=group_size,
        asym=asym,
        axis_key=axis_key,
        axis_value=axis_value,
        # device=device,
        dtype=torch.bfloat16,
        force_quant=True,
        per_layer_quant=True,
        per_layer_config=per_layer_config,
        quantilizer='vanilla',
    )

    task_manager = lm_eval.tasks.TaskManager()
    results = lm_eval.simple_evaluate(
        model=model,
        tasks=["gsm8k"],
        # tasks=["ceval-valid"],
        num_fewshot=0,
        task_manager=task_manager,
        batch_size=8,
    )
    print(results)
    return results['gsm8k']['acc']

def build_per_layer_config(model: str, nbits_key_high: int, nbits_value_high: int, nbits_key_low: int, nbits_value_low: int):
    important_layers = []
    if 'llama' in model.lower():
        important_layers = LLAMA3_IMPORTANT_LAYERS
    if 'qwen' in model.lower():
        important_layers = QWEN_IMPORTANT_LAYERS
    per_layer_config = {}
    tot_scale = 0
    tot_layers = important_layers[-1] + 1
    for layer in range(0, tot_layers):
        nbits_key = nbits_key_high if layer in important_layers else nbits_key_low
        nbits_value = nbits_value_high if layer in important_layers else nbits_value_low
        per_layer_config[layer] = {'nbits_key': nbits_key, 'nbits_value': nbits_value}
        tot_scale += nbits_key + nbits_value
    tot_scale /= tot_layers * 2
    return per_layer_config, tot_scale


def objective(trial):
    nbits_key_high = trial.suggest_int('nbits_key_high', 1, 3)
    nbits_value_high = trial.suggest_int('nbits_value_high', 1, 3)
    nbits_key_low = trial.suggest_int('nbits_key_low', 1, 3)
    nbits_value_low = trial.suggest_int('nbits_value_low', 1, 3)
    
    per_layer_config, tot_scale = build_per_layer_config(args.model_name, 2 ** nbits_key_high, 2 ** nbits_value_high, 2 ** nbits_key_low, 2 ** nbits_value_low)
    
    accuracy = run_gsm8k(global_args['residual_length'], global_args['group_size'], global_args['asym'], global_args['axis_key'], global_args['axis_value'], per_layer_config, global_args['model_name'])
    
    return accuracy, tot_scale


if __name__ == "__main__":
    args = parse_args()
    model_name = args.model_name
    
    global_args['model_name'] = model_name
    global_args['residual_length'] = args.residual_length
    global_args['group_size'] = args.group_size
    global_args['asym'] = args.asym
    global_args['axis_key'] = args.axis_key
    global_args['axis_value'] = args.axis_value
    
    study = optuna.create_study(directions=["maximize", "minimize"])
    study.optimize(objective, n_trials=30)
