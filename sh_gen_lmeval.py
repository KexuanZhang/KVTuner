# tasks in total: ceval-valid,mmlu,triviaqa,race,truthfulqa,gsm8k
# models in total: 
# meta-llama/Llama-2-7b-chat-hf,Qwen/Qwen2.5-3B-Instruct-AWQ
# meta-llama/Meta-Llama-3-8B-Instruct,Qwen/Qwen2.5-7B-Instruct
# mistralai/Mistral-7B-v0.3,Qwen/Qwen2.5-Math-7B-Instruct
# Qwen/Qwen2.5-7B-Instruct,Qwen/Qwen2.5-Math-7B-Instruct

# FOR WS-9: meta-llama/Meta-Llama-3-8B-Instruct,mistralai/Mistral-7B-v0.3
# FOR WS-13: Qwen/Qwen2.5-7B-Instruct,Qwen/Qwen2.5-Math-7B-Instruct,Qwen/Qwen2.5-3B-Instruct-AWQ

# model_args_template = "{},nbits_key={},nbits_value={},residual_length=32,q_group_size=32,axis_key=0,axis_value=1,trust_remote_code=True,dtype=bfloat16,force_quant=True"

model_args_template = "{},nbits_key={},nbits_value={},residual_length=0,q_group_size=-1,axis_key=0,axis_value=0,trust_remote_code=True,dtype=bfloat16,force_quant=True,quantilizer=vanilla"

model_args_template_bf16 = "{},trust_remote_code=True,dtype=bfloat16"

command_template = '''accelerate launch -m lm_eval --model hf-quant \\
    --model_args pretrained={} \\
    --tasks {} \\
    --batch_size {} \\
    --output_path lmeval_results/{} \\
    | tee {}.log'''

command_fewshot_template = '''accelerate launch -m lm_eval --model hf-quant \\
    --model_args pretrained={} \\
    --tasks {} \\
    --batch_size {} \\
    --num_fewshot {} \\
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
parser.add_argument('--bf16', action='store_true', default=False)

args = parser.parse_args()
bf16 = args.bf16

tasks = args.tasks.split(',')
models = args.models.split(',')
out_filename = args.filename

with open(out_filename, 'w+') as f:
    f.write("export NCCL_IB_DISABLE=1\nexport NCCL_P2P_DISABLE=1\n\n")
    for model in models:
        # batch_size = 16
        # if '7B' in model or '8B' in model or '7b' in model or '8b' in model:
        #     batch_size = 4
        batch_size = 1
        filename_model = model.replace('/', '_')
        for kv in kv_config:
            nbits_key, nbits_value = kv
            model_args = model_args_template.format(model, nbits_key, nbits_value)
            if bf16:
                model_args = model_args_template_bf16.format(model)
            task_fewshots = [task for task in tasks if task == 'gsm8k']
            task_others = [task for task in tasks if task != 'gsm8k']
            if task_others:
                task_others_str = ','.join(task_others)
                filename = f'{filename_model}_others_k{nbits_key}_v{nbits_value}'
                if bf16:
                    filename = f'{filename_model}_others_bf16'
                command = command_template.format(model_args, task_others_str, batch_size, filename, filename)
                if bf16:
                    command = command.replace('--model hf-quant', '--model hf')
                f.write(command)
                f.write('\n')
                f.write('\n')
            for task in task_fewshots:
                for nshot in nshots:
                    filename = f'{filename_model}_{task}_k{nbits_key}_v{nbits_value}_n{nshot}'
                    if bf16:
                        filename = f'{filename_model}_{task}_bf16_n{nshot}'
                    # command = command_template.format(model_args + (',n-shot={}'.format(nshot)), task, batch_size, filename, filename)
                    command = command_fewshot_template.format(model_args, task, batch_size, nshot, filename, filename)
                    if bf16:
                        command = command.replace('--model hf-quant', '--model hf')
                    f.write(command)
                    f.write('\n')
                    f.write('\n')
            if bf16:
                break

import os
os.system('chmod +x {}'.format(out_filename))



