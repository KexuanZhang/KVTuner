# tasks in total: ceval-valid,mmlu,triviaqa,race,truthfulqa,gsm8k
# models in total: 
# Qwen/Qwen2.5-3B-Instruct-AWQ
# meta-llama/Meta-Llama-3-8B-Instruct,Qwen/Qwen2.5-7B-Instruct
# mistralai/Mistral-7B-v0.3,Qwen/Qwen2.5-Math-7B-Instruct


command_template_vanliia = 'python3 gaokao_bench_obj.py --device cuda:0 --model_name {0} --k_bits {1} --v_bits {2} --residual_length 0 --group_size 32 --quantizer Vanilla --axis_key 0 --axis_value 0'
command_template_hqq = 'python3 gaokao_bench_obj.py --device cuda:1 --model_name {0} --k_bits {1} --v_bits {2} --residual_length 32 --group_size 32 --quantizer HQQ --axis_key 0 --axis_value 1'

log_filename = "GAOKAO-Bench_{0}_Q_{1}_k{2}_v{3}.json"

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
parser.add_argument('--models', type=str, required=True)
parser.add_argument('--filename', type=str, required=True)
args = parser.parse_args()

models = args.models.split(',')
out_filename = args.filename

out_filename_gpu0 = out_filename.replace('.sh', '_gpu0.sh')
out_filename_gpu1 = out_filename.replace('.sh', '_gpu1.sh')

with open(out_filename_gpu0, 'w+') as f0, open(out_filename_gpu1, 'w+') as f1:
    # run vanilla on gpu0, hqq on gpu1
    f0.write("export NCCL_IB_DISABLE=1\nexport NCCL_P2P_DISABLE=1\n\n")
    f1.write("export NCCL_IB_DISABLE=1\nexport NCCL_P2P_DISABLE=1\n\n")
    for model in models:
        for kv in kv_config:
            nbits_key, nbits_value = kv
            command_vanilla = command_template_vanliia.format(model, nbits_key, nbits_value)
            command_hqq = command_template_hqq.format(model, nbits_key, nbits_value)
            logfile_vanilla = log_filename.format(model.replace('/', '_'), 'Vanilla', nbits_key, nbits_value)
            logfile_hqq = log_filename.format(model.replace('/', '_'), 'HQQ', nbits_key, nbits_value)
            f0.write(command_vanilla + ' | tee ' + logfile_vanilla)
            f0.write('\n')
            f0.write('\n')
            f1.write(command_hqq + ' | tee ' + logfile_hqq)
            f1.write('\n')
            f1.write('\n')
f0.close()
f1.close()

import os
os.system('chmod +x {}'.format(out_filename_gpu0))
os.system('chmod +x {}'.format(out_filename_gpu1))
