
import json
import os
import tqdm
import argparse
from flexible_quant.flexible_quantized_cache import FlexibleQuantizedCacheConfig, FlexibleHQQQuantizedCache, FlexibleVanillaQuantizedCache
from transformers import AutoTokenizer, AutoModelForCausalLM, QuantizedCacheConfig, HQQQuantizedCache, QuantoQuantizedCache
from datasets import load_dataset
from transformers import LlamaConfig, AutoTokenizer, LlamaForCausalLM
import torch
import random
# from accelerate import Accelerator

# accelerator = Accelerator()
# device = accelerator.device
device = None

import importlib
bench_function = importlib.import_module("GAOKAO-Bench.Bench.bench_function")

# For reproducibility
random.seed(0)
torch.manual_seed(0)
CACHE_DIR = "./models_storage"

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
    # parser.add_argument('--model_name', type=str, default="Qwen/Qwen2.5-3B-Instruct-AWQ")
    # parser.add_argument('--model_name', type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument('--model_name', type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    # parser.add_argument('--model_name', type=str, default="")
    # parser.add_argument('--nshots', type=int, default=5)
    parser.add_argument('--dtype', type=str, default="bfloat16")
    parser.add_argument('--k_bits', type=int, default=8)
    parser.add_argument('--v_bits', type=int, default=8)
    parser.add_argument('--residual_length', type=int, default=128)
    parser.add_argument('--group_size', type=int, default=64)
    parser.add_argument('--asym', type=bool, default=True)
    parser.add_argument('--quantizer', type=str, default="Vanilla")
    # in HQQ, 0 for per-channel, 1 for per-token
    # in Vanilla, 0 for per-token, 1 for per-channel
    parser.add_argument('--axis_key', type=int, default=0)
    parser.add_argument('--axis_value', type=int, default=0)
    parser.add_argument('--per_layer_quant', type=bool, default=False)
    parser.add_argument('--per_layer_config_path', type=str, default="")
    parser.add_argument('--limit', type=int, default=-1)
    parser.add_argument('--device', type=str, default="cuda")
    return parser.parse_args(args)

tests_all = []

if __name__ == "__main__":
    args = parse_args()
    print(args)
    
    device = torch.device(args.device)
    
    model = AutoModelForCausalLM.from_pretrained(args.model_name, cache_dir=CACHE_DIR, torch_dtype=get_dtype(args.dtype)).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False, trust_remote_code=True)
    
    with open("GAOKAO-Bench/Bench/Obj_Prompt.json", "r") as f:
        examples = json.load(f)['examples']
    f.close()

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
    cache_config = FlexibleQuantizedCacheConfig(nbits_key=args.k_bits, nbits_value=args.v_bits, residual_length=args.residual_length, q_group_size=args.group_size,
                                                asym=args.asym, axis_key=args.axis_key, axis_value=args.axis_value, device=device, compute_dtype=get_dtype(args.dtype),
                                                per_layer_quant=args.per_layer_quant, per_layer_config_path=args.per_layer_config_path)
    
    # tests_all = tests_all[:3]
    print('Running tests')
    results = []
    # for test in tqdm.tqdm(tests):
    if args.limit != -1:
        tests_all = tests_all[:args.limit]
    idx, correct = 0, 0
    for test in tqdm.tqdm(tests_all):
        past_key_values = FlexibleVanillaQuantizedCache(cache_config=cache_config) if args.quantizer == 'Vanilla' else FlexibleHQQQuantizedCache(cache_config=cache_config)
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
        if idx % 50 == 0:
            print(f"Num of total question: {idx}, Correct num: {correct}, Accuracy: {float(correct)/idx}")
        if idx % 50 == 1:
            print('promot:', prompt)
            print('===')
            print('model output:', model_answer)
            print('===')
            print('standard answer:', test['standard_answer'])
            print('===')
            print('is correct:', test['is_correct'])
            print('====================')
    
    print(f"Num of total question: {idx}, Correct num: {correct}, Accuracy: {float(correct)/idx}")
    filename_out = f"GAOKAO-Bench_{args.model_name.replace('/', '_')}_Q_{args.quantizer}_k{args.k_bits}_v{args.v_bits}_r{args.residual_length}_g{args.group_size}.json"
    with open(filename_out, 'w') as f:
        json.dump(results, f)
    f.close()