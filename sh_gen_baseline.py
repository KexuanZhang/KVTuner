model_args_template_pertoken = "pretrained={},nbits_key={},nbits_value={},residual_length=0,q_group_size=-1,axis_key=0,axis_value=0,trust_remote_code=True,dtype=bfloat16,force_quant=False,quantilizer=vanilla"
# per_layer_config_path is yaml file path

command_fewshot_template = '''accelerate launch -m lm_eval --model hf-quant \\
    --model_args {} \\
    --tasks {} \\
    --batch_size 1 \\
    --num_fewshot {} \\
    --limit 200 \\
    --output_path lmeval_results/{} \\
    | tee {}.log'''

TASKS = [
    {
        'filename': 'gsm8k',
        'tasks': ['gsm8k'],
        'nshots': [4],
    },
]

STANDARD_KV_CONFIG = ['kv8', 'k8v4', 'k4v8', 'k8v2', 'kv4', 'k4v2', 'k2v4', 'kv2']


def extract_kv_config(config_str: str):
    if len(config_str) == 3:
        return int(config_str[2]), int(config_str[2])
    return int(config_str[1]), int(config_str[3])

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--models', type=str, required=True)
parser.add_argument('--filename', type=str, required=False, default='run.sh')

args = parser.parse_args()

models = args.models.split(',')
out_filename = args.filename

# naming scheme: {model}_pertoken_{task_filename}_{nshot}_{kv_config or bf16 or id}.

def get_filename(model, task_filename, nshots, kvconfig_or_bf16_or_id):
    return f'{model}_pertoken_{task_filename}_{nshots}_{kvconfig_or_bf16_or_id}'

tot_commands = 0
tot_time = 0
with open(out_filename, 'w+') as f:
    f.write("export NCCL_IB_DISABLE=1\nexport NCCL_P2P_DISABLE=1\nexport HF_ALLOW_CODE_EVAL=1\nexport TRANSFORMERS_CACHE=./models_storage\n\n")
    for model in models:
        filename_model = model.replace('/', '_') + '_pertoken_baseline_limit200'
        f.write(f'# ======== {model} standard kv configs ========\n')
        for kv_config in STANDARD_KV_CONFIG:
            for task_preset in TASKS:
                for nshot in task_preset['nshots']:
                    nbits_key, nbits_value = extract_kv_config(kv_config)
                    filename = get_filename(filename_model, task_preset['filename'], nshot, kv_config)
                    model_arg = model_args_template_pertoken.format(model, nbits_key, nbits_value)
                    command = command_fewshot_template.format(model_arg, ','.join(task_preset['tasks']), nshot, filename, filename)
                    f.write(command)
                    tot_commands += 1
                    tot_time += 10
                    f.write('\n')
                    f.write('\n')

import os
os.system('chmod +x {}'.format(out_filename))

print(f'Generated {tot_commands} commands in {out_filename}.')
print(f'Estimated total running time (on dual RTX 4090): {tot_time} minutes. aka {tot_time/60} hours.')