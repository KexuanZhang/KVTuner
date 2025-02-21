import json
import os
import tqdm
import argparse
from flexible_quant.flexible_quantized_cache import FlexibleQuantizedCacheConfig, FlexibleHQQQuantizedCache, FlexibleVanillaQuantizedCache
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaConfig, AutoTokenizer, LlamaForCausalLM
import torch
import random

import importlib
bench_function = importlib.import_module("GAOKAO-Bench.Bench.bench_function")


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

QWEN_IMPORTANT_LAYERS = [0, 18, 20, 27, 29, 35]
QWEN_MEDIUM_LAYERS = [3, 4, 5]

global_args = {}

def get_dtype(str):
    if str == "bfloat16":
        return torch.bfloat16
    elif str == "float16":
        return torch.float16
    elif str == "float32":
        return torch.float32
    else:
        raise ValueError(f"Unsupported dtype {str}")

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model_name', type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen2.5-3B-Instruct-AWQ")
    parser.add_argument('--front_filename', type=str, default="front_profiles_qwen2.txt")
    # parser.add_argument('--model_name', type=str, default="Qwen/Qwen2.5-7B-Instruct")
    # parser.add_argument('--model_name', type=str, default="meta-llama/Meta-Llama-3-8B-Instruct") 
    parser.add_argument('--residual_length', type=int, default=32)
    parser.add_argument('--group_size', type=int, default=32)
    parser.add_argument('--asym', type=bool, default=True)
    # in Vanilla, 0 for per-token, 1 for per-channel, we have to use per-channel there as residual_length is 0
    parser.add_argument('--axis_key', type=int, default=1)
    parser.add_argument('--axis_value', type=int, default=0)
    # parser.add_argument('--limit', type=int, default=200)
    # parser.add_argument('--num_fewshots', type=int, default=0)
    parser.add_argument('--device', type=str, default="cuda:0")
    return parser.parse_args(args)

#         results = lm_eval.simple_evaluate(
#             model='hf-quant',
#             model_args={
#                 'pretrained': model_name,
#                 'nbits_key': -1,
#                 'nbits_value': -1,
#                 'residual_length': residual_length,
#                 'q_group_size': group_size,
#                 'asym': asym,
#                 'axis_key': axis_key,
#                 'axis_value': axis_value,
#                 'dtype': torch.bfloat16,
#                 'force_quant': True,
#                 'per_layer_quant': True,
#                 'per_layer_config': per_layer_config,
#                 'quantilizer': 'vanilla',
#             },
#             tasks=["gsm8k"],
#             num_fewshot=num_fewshots,
#             limit=limit,
#             device=device
#         )
#     else:
#         results = lm_eval.simple_evaluate(
#             model='hf-quant',
#             model_args={
#                 'pretrained': model_name,
#                 'nbits_key': -1,
#                 'nbits_value': -1,
#                 'residual_length': residual_length,
#                 'q_group_size': group_size,
#                 'asym': asym,
#                 'axis_key': axis_key,
#                 'axis_value': axis_value,
#                 'dtype': torch.bfloat16,
#                 'force_quant': True,
#                 'per_layer_quant': True,
#                 'per_layer_config': per_layer_config,
#                 'quantilizer': 'vanilla',
#             },
#             tasks=["gsm8k"],
#             num_fewshot=num_fewshots,
#             device=device
#         )
#     print(results['results']['gsm8k']['exact_match,flexible-extract'])
#     return float(results['results']['gsm8k']['exact_match,flexible-extract'])

def run_gaokaobench(residual_length: int, group_size: int, asym: bool, axis_key: int, axis_value: int, per_layer_config: dict, model_name: str, device: str, dtype: str):
    device = torch.device(args.device)
    
    model = AutoModelForCausalLM.from_pretrained(args.model_name, cache_dir=CACHE_DIR, torch_dtype=get_dtype(dtype)).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False, trust_remote_code=True)
    
    with open("GAOKAO-Bench/Bench/Obj_Prompt.json", "r") as f:
        examples = json.load(f)['examples']
    f.close()
    
    tests_all = []

    for example in examples:
        directory = "GAOKAO-Bench/Data/Objective_Questions"

        keyword = example['keyword']
        question_type = example['type']
        zero_shot_prompt_text = example['prefix_prompt']
        print('Building data for keyword:', keyword)
        
        filepath = os.path.join(directory, f"{keyword}.json")
        with open(filepath, "r") as f:
            data = json.load(f)
        f.close()
        
        data = data['example']
        example_num = len(data)
        
        
        for i in tqdm.tqdm(range(example_num)):
            if question_type in ["single_choice", "five_out_of_seven", "multi_question_choice", "multi_choice"]:
                index = data[i]['index']
                question = data[i]['question'].strip() + '\n'
                year = data[i]['year']
                category = data[i]['category']
                score = data[i]['score']
                standard_answer = data[i]['answer']
                answer_length = len(standard_answer)
                analysis = data[i]['analysis']
                prompt = zero_shot_prompt_text
                
                current_test_dict = {
                    'index': index, 
                    'type': question_type,
                    'year': year, 
                    'category': category,
                    'score': score,
                    'question': question, 
                    'standard_answer': standard_answer,
                    'analysis': analysis,
                    'prompt': prompt
                }
                tests_all.append(current_test_dict)
            elif question_type in ["subjective", "cloze"]:
                raise NotImplementedError('subjective and cloze question types are not supported')
            elif question_type == 'correction':
                raise NotImplementedError('correction question type is not supported')
    # now init LLM
    cache_config = FlexibleQuantizedCacheConfig(residual_length=residual_length, q_group_size=group_size,
                                                asym=asym, axis_key=axis_key, axis_value=axis_value, device=device, compute_dtype=get_dtype(dtype),
                                                per_layer_quant=True, per_layer_config=per_layer_config)
    
    # tests_all = tests_all[:3]
    print('Running tests')
    results = []
    # for test in tqdm.tqdm(tests):
    idx, correct = 0, 0
    for test in tqdm.tqdm(tests_all):
        past_key_values = FlexibleVanillaQuantizedCache(cache_config=cache_config)
        prompt = test['prompt'] + test['question']
        inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        output = model.generate(inputs, past_key_values=past_key_values, use_cache=True, max_new_tokens=256, pad_token_id=None, eos_token_id=None)
        model_completion = tokenizer.decode(output[0].tolist()[inputs.shape[1]:], skip_special_tokens=True)
        test['model_completion'] = model_completion
        model_answer = bench_function.extract_choice_answer(model_completion, test['type'], len(test['standard_answer']))
        test['model_answer'] = model_answer
        test['is_correct'] = model_answer == test['standard_answer']
        results.append(test)
        idx += 1
        if test['is_correct']:
            correct += 1
        # if idx % 50 == 0:
        #     print(f"Num of total question: {idx}, Correct num: {correct}, Accuracy: {float(correct)/idx}")
        # if idx % 50 == 1:
        #     print('promot:', prompt)
        #     print('===')
        #     print('model output:', model_answer)
        #     print('===')
        #     print('standard answer:', test['standard_answer'])
        #     print('===')
        #     print('is correct:', test['is_correct'])
        #     print('====================')
    
    print(f"Num of total question: {idx}, Correct num: {correct}, Accuracy: {float(correct)/idx}")
    
    return float(correct)/idx
    # filename_out = f"GAOKAO-Bench_{args.model_name.replace('/', '_')}_Q_{args.quantizer}_k{args.k_bits}_v{args.v_bits}_r{args.residual_length}_g{args.group_size}.json"
    # with open(filename_out, 'w') as f:
    #     json.dump(results, f)
    # f.close()

def build_per_layer_config(model: str, config_high: int, config_medium: int, config_low: int):
    important_layers = []
    if 'llama' in model.lower():
        important_layers = LLAMA3_IMPORTANT_LAYERS
        medium_layers = LLAMA3_MEDIUM_LAYERS
    if 'qwen' in model.lower():
        important_layers = QWEN_IMPORTANT_LAYERS
        medium_layers = QWEN_MEDIUM_LAYERS
    per_layer_config = {}
    tot_scale = 0
    tot_layers = 32 if 'llama' in model.lower() else 36
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
    # global_args['limit'] = args.limit
    # global_args['num_fewshots'] = args.num_fewshots
    global_args['device'] = args.device
    
    print(global_args)
    
    with open(args.front_filename, "r") as f:
        front_profiles = f.readlines()
    f.close()
    
    for front_profile in front_profiles:
        # 2.0 0.01061410159211524 4, 4, 4
        print('running profile:', front_profile)
        profile_high, profile_medium, profile_low = map(int, front_profile.replace(',', '').split(' ')[-3:])
        print('profile:', profile_high, profile_medium, profile_low)
        per_layer_config, tot_scale = build_per_layer_config(args.model_name, profile_high, profile_medium, profile_low)
        accuracy = run_gaokaobench(global_args['residual_length'], global_args['group_size'], global_args['asym'], global_args['axis_key'], global_args['axis_value'], per_layer_config, global_args['model_name'], global_args['device'], 'bfloat16')
        print(f"Profile: {profile_high}, {profile_medium}, {profile_low}, Accuracy: {accuracy}, Scale: {tot_scale}")
        print("")
    # valid_params = []
    # for profile_high in range(5):
    #     for profile_medium in range(profile_high, 5):
    #         for profile_low in range(profile_medium, 5):
    #             valid_params.append((profile_high, profile_medium, profile_low))
    
    # for profile_high, profile_medium, profile_low in valid_params:
    #     per_layer_config, tot_scale = build_per_layer_config(args.model_name, profile_high, profile_medium, profile_low)
    #     accuracy = run_gsm8k(global_args['residual_length'], global_args['group_size'], global_args['asym'], global_args['axis_key'], global_args['axis_value'], per_layer_config, global_args['model_name'], global_args['num_fewshots'], global_args['limit'], global_args['device'])
    #     print(f"Profile: {profile_high}, {profile_medium}, {profile_low}, Accuracy: {accuracy}, Scale: {tot_scale}")
    #     print("")