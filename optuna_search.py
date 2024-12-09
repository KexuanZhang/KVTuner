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



def run_gsm8k(residual_length: int, group_size: int, asym: bool, axis_key: int, axis_value: int, per_layer_config: dict, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, dataset: dict):
    cache_config = FlexibleQuantizedCacheConfig(residual_length=args.residual_length, q_group_size=args.group_size,
                                                asym=args.asym, axis_key=args.axis_key, axis_value=args.axis_value, device='cuda', compute_dtype=torch.bfloat16, per_layer_quant=True, per_layer_config=per_layer_config)
    answers = []
    
    for _question_answer in tqdm.tqdm(dataset['test']):
        past_key_values = FlexibleVanillaQuantizedCache(cache_config=cache_config)
        prompt = build_prompt_from_trainset(dataset['train'], _question_answer["question"], 4, True)
        inputs = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        output = model.generate(inputs, past_key_values=past_key_values, use_cache=True, max_new_tokens=256)
        model_completion = tokenizer.decode(output[0].tolist()[inputs.shape[1]:], skip_special_tokens=True)
        model_answer = clean_answer(model_completion)
        is_cor = is_correct(model_answer, _question_answer["answer"])
        answers.append(is_cor)
    
    accuracy = float(sum(answers))/len(answers)
    print(
            f"Num of total question: {len(answers)}, "
            f"Correct num: {sum(answers)}, "
            f"Accuracy: {accuracy}."
        )
    return accuracy

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
    
    accuracy = run_gsm8k(global_args['residual_length'], global_args['group_size'], global_args['asym'], global_args['axis_key'], global_args['axis_value'], per_layer_config, model, tokenizer, dataset)
    
    return accuracy, tot_scale


if __name__ == "__main__":
    args = parse_args()
    model_name = args.model_name
    
    global_args['residual_length'] = args.residual_length
    global_args['group_size'] = args.group_size
    global_args['asym'] = args.asym
    global_args['axis_key'] = args.axis_key
    global_args['axis_value'] = args.axis_value
        
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=CACHE_DIR, torch_dtype=torch.bfloat16).cuda()
    model = model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)

    dataset = load_dataset('gsm8k', 'main')
    
    study = optuna.create_study(directions=["maximize", "minimize"])
    study.optimize(objective, n_trials=30)
