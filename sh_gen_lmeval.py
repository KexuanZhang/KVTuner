# tasks in total: ceval-valid,mmlu,triviaqa,race,truthfulqa,gsm8k
# models in total: 
# meta-llama/Llama-2-7b-chat-hf,Qwen/Qwen2.5-3B-Instruct-AWQ
# meta-llama/Meta-Llama-3-8B-Instruct
# mistralai/Mistral-7B-v0.3
# Qwen/Qwen2.5-7B-Instruct,Qwen/Qwen2.5-Math-7B-Instruct

model_args_template = "{},nbits_key={},nbits_value={},residual_length=32,q_group_size=32,axis_key=0,axis_value=1,trust_remote_code=True,dtype=bfloat16"

command_template = '''accelerate launch -m lm_eval --model hf-quant \\
    --model_args pretrained={} \\
    --tasks {} \\
    --batch_size {} \\
    --output_path lmeval_results/{} \\
    | tee {}.log'''

kv_config = [
    [8, 8],
    [8, 4],
    [8, 2],
    [4, 8],
    [4, 4],
    [4, 2],
    [2, 4],
    [2, 2],
]
nshots = [0, 4, 8, 16]
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--tasks', type=str, required=False, default="ceval-valid,mmlu,triviaqa,race,truthfulqa,gsm8k")
parser.add_argument('--models', type=str, required=True)
parser.add_argument('--filename', type=str, required=False, default='run.sh')

args = parser.parse_args()

tasks = args.tasks.split(',')
models = args.models.split(',')
out_filename = args.filename

with open(out_filename, 'w+') as f:
    f.write("export NCCL_IB_DISABLE=1\nexport NCCL_P2P_DISABLE=1\n\n")
    for model in models:
        batch_size = 16
        if '7B' in model or '8B' in model or '7b' in model or '8b' in model:
            batch_size = 4
        filename_model = model.replace('/', '_')
        for kv in kv_config:
            nbits_key, nbits_value = kv
            model_args = model_args_template.format(model, nbits_key, nbits_value)
            for task in tasks:
                if task == 'gsm8k':
                    for nshot in nshots:
                        filename = f'{filename_model}_{task}_k{nbits_key}_k{nbits_value}_n{nshot}'
                        command = command_template.format(model_args + (',n-shot={}'.format(nshot)), task, batch_size, filename, filename)
                        f.write(command)
                        f.write('\n')
                        f.write('\n')
                else:
                    filename = f'{filename_model}_{task}_k{nbits_key}_k{nbits_value}'
                    command = command_template.format(model_args, task, batch_size, filename, filename)
                    f.write(command)
                    f.write('\n')
                    f.write('\n')

import os
os.system('chmod +x {}'.format(out_filename))



