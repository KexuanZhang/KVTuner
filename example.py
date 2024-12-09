import torch
from flexible_quant.flexible_quantized_cache import FlexibleQuantizedCacheConfig, FlexibleHQQQuantizedCache, FlexibleVanillaQuantizedCache
from transformers import AutoTokenizer, AutoModelForCausalLM, QuantizedCacheConfig, HQQQuantizedCache, QuantoQuantizedCache
from datasets import load_dataset

CACHE_DIR = "./models_storage"
# model_name = 'Qwen/Qwen2.5-3B-Instruct-AWQ'
# model_name = 'Qwen/Qwen2.5-7B-Instruct'
model_name = 'meta-llama/Meta-Llama-3-8B'
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=CACHE_DIR, torch_dtype=torch.float16).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)

# Quanto from huggingface is not working at all
# ValueError("shift must be specified for qtypes lower than 8-bit")

cache_config = FlexibleQuantizedCacheConfig(nbits_key=4, nbits_value=4, asym=True, axis_key=1, axis_value=0, device='cuda', per_layer_config=True, per_layer_config_path='config/meta-llama_Meta-Llama-3-8B-Instruct_k8_v4_per_layer.yaml')
# past_key_values = FlexibleHQQQuantizedCache(cache_config=cache_config) # it seems in HQQ, 0 for per-token and 1 for per-channel
past_key_values = FlexibleVanillaQuantizedCache(cache_config=cache_config)

# cache_config = QuantizedCacheConfig(nbits=4, axis_key=0, axis_value=0, device='cuda')
# past_key_values = QuantoQuantizedCache(cache_config=cache_config)

dataset = load_dataset('gsm8k', 'main')

prompt = ''
for i in range(5):
    prompt += 'Question: ' + dataset['train'][i]['question'] + '\nAnswer: ' + dataset['train'][i]['answer'] + '\n'
prompt += "Question: John takes care of 10 dogs. Each dog takes .5 hours a day to walk and take care of their business. How many hours a week does he spend taking care of dogs?"
inputs = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
print('======')

outputs = model.generate(inputs, past_key_values=past_key_values, use_cache=True, max_new_tokens=256)

# config_str = f"# prompt tokens: {inputs.shape[1]}, K bit: {config.k_bits}, v_bits: {config.v_bits}, group_size: {config.group_size}, residual_length: {config.residual_length}"
config_str = f"# prompt tokens: {inputs.shape[1]}"

print(prompt + "\n" + "=" * 10 + f'\n{config_str}\n' + "=" * 10 + "\nExample Output:")
print(tokenizer.decode(outputs[0].tolist()[inputs.shape[1]:], skip_special_tokens=True))
