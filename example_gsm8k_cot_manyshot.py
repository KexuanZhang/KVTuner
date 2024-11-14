# LLaMA model with KIVI
import warnings
warnings.filterwarnings("ignore")
import torch
import random
import argparse
import torch
from src.flexible_quantized_cache import FlexibleQuantizedCacheConfig, FlexibleHQQQuantizedCache, FlexibleVanillaQuantizedCache
from transformers import AutoTokenizer, AutoModelForCausalLM, QuantizedCacheConfig, HQQQuantizedCache, QuantoQuantizedCache
from datasets import load_dataset
from transformers import LlamaConfig, AutoTokenizer, LlamaForCausalLM
from datasets import load_dataset
from evals.gsm8k_utils import *

# For reproducibility
random.seed(0)
torch.manual_seed(0)
CACHE_DIR = "./models_storage"

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model_name', type=str, default="meta-llama/Llama-2-7b-hf")
    # parser.add_argument('--model_name', type=str, default="Qwen/Qwen2.5-3B-Instruct-AWQ")
    # parser.add_argument('--model_name', type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument('--model_name', type=str, default="")
    parser.add_argument('--nshots', type=int, default=5)
    parser.add_argument('--k_bits', type=int, default=8)
    parser.add_argument('--v_bits', type=int, default=8)
    parser.add_argument('--residual_length', type=int, default=128)
    parser.add_argument('--group_size', type=int, default=64)
    parser.add_argument('--asym', type=bool, default=True)
    # in HQQ, 0 for per-channel, 1 for per-token
    parser.add_argument('--axis_key', type=int, default=0)
    parser.add_argument('--axis_value', type=int, default=1)
    return parser.parse_args(args)

def args_to_str(args):
    ret = ""
    for arg in vars(args):
        ret += f"{arg}: {getattr(args, arg)}\n"
    return ret


if __name__ == "__main__":
    args = parse_args()
    model_name = args.model_name
    num_cot = args.nshots
    
    # asym only works for VanillaQuantizedCache
    cache_config = FlexibleQuantizedCacheConfig(nbits_key=args.k_bits, nbits_value=args.v_bits, residual_length=args.residual_length, q_group_size=args.group_size,
                                                asym=args.asym, axis_key=args.axis_key, axis_value=args.axis_value, device='cuda')
    
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=CACHE_DIR, torch_dtype=torch.float16).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)

    dataset = load_dataset('gsm8k', 'main')

    answers = []
    num_testcases = len(dataset['test'])
    for idx, _question_answer in enumerate(dataset['test']):
        # past_key_values = FlexibleVanillaQuantizedCache(cache_config=cache_config)
        past_key_values = FlexibleHQQQuantizedCache(cache_config=cache_config)
        
        prompt = build_prompt_from_trainset(dataset['train'], _question_answer["question"], num_cot, COT_FLAG)

        inputs = tokenizer(prompt, return_tensors="pt").input_ids.cuda()\
        
        output = model.generate(inputs, past_key_values=past_key_values, use_cache=True, max_new_tokens=256)
        # config_str = f"# prompt tokens: {inputs.shape[1]}, K bit: {cache_config.nbits_key}, v_bits: {cache_config.nbits_value}, num_cot: {num_cot} group_size: {cache_config.q_group_size}, residual_length: {cache_config.residual_length}"
        config_str = args_to_str(args)
        model_completion = tokenizer.decode(output[0].tolist()[inputs.shape[1]:], skip_special_tokens=True)
        
        model_answer = clean_answer(model_completion)
        is_cor = is_correct(model_answer, _question_answer["answer"])
        answers.append(is_cor)
        
        print("\n\n" + "=" * 88 + "\n\t\t{} / {}-th testcase".format(idx, num_testcases))

        if idx % 50 == 0:
            print(prompt + "\n\n\n" +  "=" * 10 + f'\n{config_str}\n' + f"model_name : {args.model_name}\n" + "=" * 10  + "\nExample Output:")
        else:
            print(_question_answer["question"] + "\n\n\n" +  "=" * 10 + f'\n{config_str}\n' + f"model_name : {args.model_name}\n" + "=" * 10  + "\nExample Output:")
        print(model_completion)
        print("\nTarget answer: {}".format(_question_answer["answer"]))
        print("\n=== Is correct: {}".format(is_cor))

        print(
                f"Num of total question: {len(answers)}, "
                f"Correct num: {sum(answers)}, "
                f"Accuracy: {float(sum(answers))/len(answers)}."
            )

    print("Final result summary:\n")
    print(
            f"Num of total question: {len(answers)}, "
            f"Correct num: {sum(answers)}, "
            f"Accuracy: {float(sum(answers))/len(answers)}."
        )

    for i in range(5):
        str_ = 'Question: ' + dataset['test'][i]['question'] + '\nAnswer: ' + dataset['test'][i]['answer'] + "\n"
        print("TEST [{}]: \n{}".format(i, str_))