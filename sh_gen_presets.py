model_args_template_pertoken = "pretrained={},nbits_key={},nbits_value={},residual_length=0,q_group_size=-1,axis_key=0,axis_value=0,trust_remote_code=True,dtype=bfloat16,force_quant=False,quantilizer=vanilla"
model_args_template_pertoken_perlayer = "pretrained={},nbits_key=-1,nbits_value=-1,residual_length=0,q_group_size=-1,axis_key=0,axis_value=0,trust_remote_code=True,dtype=bfloat16,force_quant=False,quantilizer=vanilla,per_layer_quant=True,per_layer_config_path={}"
# per_layer_config_path is yaml file path


model_args_template_bf16 = "pretrained={},trust_remote_code=True,dtype=bfloat16"

command_fewshot_template = '''accelerate launch -m lm_eval --model hf-quant \\
    --model_args {} \\
    --tasks {} \\
    --batch_size 1 \\
    --num_fewshot {} \\
    --output_path lmeval_results/{} \\
    | tee {}.log'''

command_fewshot_as_multiturn = '''accelerate launch -m lm_eval --model hf-quant \\
    --model_args {} \\
    --tasks {} \\
    --batch_size 1 \\
    --num_fewshot {} \\
    --fewshot_as_multiturn \\
    --apply_chat_template \\
    --output_path lmeval_results/{} \\
    | tee {}.log'''

TASKS = [
    {
        'filename': 'leaderboard_musr',
        'tasks': ['leaderboard_musr'],
        'nshots': [0],
    },
    # {
    #     'filename': 'gpqa_extended',
    #     'tasks': ['gpqa_extended_n_shot', 'gpqa_extended_generative_n_shot'],
    #     'nshots': [5, 10, 20],
    # },
    # gpqa_extended gets OOM on RTX 4090
    {
        'filename': 'gsm8k',
        'tasks': ['gsm8k'],
        'nshots': [4, 8, 16],
    },
    {
        'filename': 'gsm8k_multiturn',
        'tasks': ['gsm8k'],
        'nshots': [4, 8, 16],
        'fewshot_as_multiturn': True,
    },
]

STANDARD_KV_CONFIG = ['kv8', 'k8v4', 'k4v8', 'kv4', 'k4v2', 'kv2']

def get_calibration_filepath(model: str):
    model_name = model.split('/')[-1]
    path = './calibration_presets'
    import os
    if not os.path.exists(path):
        return []
    files = os.listdir(path)
    files = [f for f in files if model_name in f]
    # filename like: modelname_KVTuner{4/6}_{id}.yaml
    ret = []
    for f in files:
        full_path = os.path.join(path, f)
        fid = '_'.join(f.split('_')[-2:]).replace('.yaml', '')
        print(f'Found calibration file {f} with id {fid}')
        ret.append((full_path, fid))
    return ret

def extract_kv_config(config_str: str):
    if len(config_str) == 3:
        return int(config_str[2]), int(config_str[2])
    return int(config_str[1]), int(config_str[3])

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--models', type=str, required=True)
parser.add_argument('--filename', type=str, required=False, default='run.sh')
parser.add_argument('--baseline_only', type=bool, required=False, default=False)
parser.add_argument('--kvturner_only', type=bool, required=False, default=False)

args = parser.parse_args()

models = args.models.split(',')
out_filename = args.filename

# naming scheme: {model}_pertoken_{task_filename}_{nshot}_{kv_config or bf16 or id}.

def get_filename(model, task_filename, nshots, kvconfig_or_bf16_or_id):
    return f'{model}_pertoken_{task_filename}_{nshots}_{kvconfig_or_bf16_or_id}'

tot_commands = 0
tot_time = 0
with open(out_filename, 'w+') as f:
    f.write("export NCCL_IB_DISABLE=1\nexport NCCL_P2P_DISABLE=1\nexport TRANSFORMERS_CACHE=./models_storage\n\n")
    for model in models:
        calibration_files = get_calibration_filepath(model)
        filename_model = model.replace('/', '_') + '_pertoken'
        # first, run bf16
        if not args.kvturner_only:
            f.write(f'# ======== {model} bf16 ========\n')
            for task_preset in TASKS:
                for nshot in task_preset['nshots']:
                    filename = get_filename(filename_model, task_preset['filename'], nshot, 'bf16')
                    model_arg = model_args_template_bf16.format(model)
                    if task_preset.get('fewshot_as_multiturn', False):
                        command = command_fewshot_as_multiturn.format(model_arg, ','.join(task_preset['tasks']), nshot, filename, filename)
                    else:
                        command = command_fewshot_template.format(model_arg, ','.join(task_preset['tasks']), nshot, filename, filename)
                    command = command.replace('hf-quant', 'hf')
                    f.write(command)
                    tot_commands += 1
                    tot_time += 25 if 'gsm8k' in task_preset['filename'] else 10
                    f.write('\n')
                    f.write('\n')
            f.write('\n\n\n')
        if not args.baseline_only:
            # then, run kv configs
            f.write(f'# ======== {model} kv calibration ========\n')
            for (calibration_file, calibration_file_id) in calibration_files:
                for task_preset in TASKS:
                    for nshot in task_preset['nshots']:
                        filename = get_filename(filename_model, task_preset['filename'], nshot, calibration_file_id)
                        model_arg = model_args_template_pertoken_perlayer.format(model, calibration_file)
                        if task_preset.get('fewshot_as_multiturn', False):
                            command = command_fewshot_as_multiturn.format(model_arg, ','.join(task_preset['tasks']), nshot, filename, filename)
                        else:
                            command = command_fewshot_template.format(model_arg, ','.join(task_preset['tasks']), nshot, filename, filename)
                        f.write(command)
                        tot_commands += 1
                        tot_time += 25 if 'gsm8k' in task_preset['filename'] else 10
                        f.write('\n')
                        f.write('\n')
            f.write('\n\n\n')
        if not args.kvturner_only:
            # standard kv configs
            f.write(f'# ======== {model} standard kv configs ========\n')
            for kv_config in STANDARD_KV_CONFIG:
                for task_preset in TASKS:
                    for nshot in task_preset['nshots']:
                        nbits_key, nbits_value = extract_kv_config(kv_config)
                        filename = get_filename(filename_model, task_preset['filename'], nshot, kv_config)
                        model_arg = model_args_template_pertoken.format(model, nbits_key, nbits_value)
                        if task_preset.get('fewshot_as_multiturn', False):
                            command = command_fewshot_as_multiturn.format(model_arg, ','.join(task_preset['tasks']), nshot, filename, filename)
                        else:
                            command = command_fewshot_template.format(model_arg, ','.join(task_preset['tasks']), nshot, filename, filename)
                        f.write(command)
                        tot_commands += 1
                        tot_time += 25 if 'gsm8k' in task_preset['filename'] else 10
                        f.write('\n')
                        f.write('\n')

import os
os.system('chmod +x {}'.format(out_filename))

print(f'Generated {tot_commands} commands in {out_filename}.')
print(f'Estimated total running time (on dual RTX 4090): {tot_time} minutes. aka {tot_time/60} hours.')



