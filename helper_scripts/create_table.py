# Quant. method & Precision &  CEVAL-VALID & MMLU & TriviaQA & RACE & TruthfulQA & GSM8K & GSM8K 4-shot & GSM8K 8-shot & GSM8K 16-shot \\ \hline
# \multicolumn{11}{c}{\textbf{Mistral-7B-Instruct-v0.3}} \\ \hline
# \multirow{5}{*}{KIVI} 
# & KV8 & 0.4368  & 0.5904 & 0.3246 & 0.4622 & 0.5435 & - & - & - & -  \\
# & K8V4 & 0.4368  & 0.5904 & 0.3243 & 0.4622 & 0.5483 & - & - & - & -  \\ 
# & K8V2 & 0.4368  & 0.5904 & 0.3208 & 0.4622 & 0.5459 & - & - & - & -  \\ 
# & K4V8 & 0.4368  & 0.5904 & 0.3242 & 0.4622 & 0.5349 & - & - & - & -  \\ 
# & KV4 & 0.4368  & 0.5904 & 0.3245 & 0.4622 & 0.5373 & - & - & - & -  \\ 
# & K4V2 & 0.4368  & 0.5904 & 0.3199 & 0.4622 & 0.5398 & - & - & - & -  \\ 
# & K2V4 & 0.4368  & 0.5904 & 0.3231 & 0.4622 & 0.5471 & - & - & - & -  \\ 
# & KV2 & 0.4368  & 0.5904 & 0.3190 & 0.4622 & 0.5300 & - & - & - & - \\ \hline

# CEVAL-VALID & MMLU & TriviaQA & RACE & TruthfulQA & GSM8K & GSM8K 4-shot & GSM8K 8-shot & GSM8K 16-shot
datasets = ['CEVAL-VALID', 'MMLU', 'TriviaQA', 'RACE', 'TruthfulQA', 'GSM8K', 'GSM8K 4-shot', 'GSM8K 8-shot', 'GSM8K 16-shot']
# KV config
kv_configs = [[8, 8], [8, 4], [8, 2], [4, 8], [4, 4], [4, 2], [2, 4], [2, 2]]\

def KV_config_str(k_nbit, v_nbit):
    if k_nbit == v_nbit:
        return f'KV{k_nbit}'
    else:
        return f'K{k_nbit}V{v_nbit}'

# |ceval-valid       |      2|none  |      |acc     |↑  |0.2585|±  |0.0119|
def extrace_value(lines: list, start_str: str):
    for line in lines:
        if line.startswith(start_str):
            rem = line.split(start_str)[1]
            return rem.split('|')[0].strip()
    return '-'

import argparse
import os
# add param: model_name
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, required=True)
args = parser.parse_args()

model_name = args.model_name.replace('/', '_')
display_model_name = args.model_name.split('/')[-1]

# create empty table
table = {}
for kv_config in kv_configs:
    k_nbit, v_nbit = kv_config
    table[KV_config_str(k_nbit, v_nbit)] = {dataset: '-' for dataset in datasets}
    for dataset in datasets:
        if not dataset.startswith('GSM8K'):
            log_filename = f'{model_name}_others_k{k_nbit}_v{v_nbit}.log'
            # test if file exists
            if not os.path.exists(log_filename):
                continue
            with open(log_filename, 'r') as f:
                lines = f.readlines()
                if dataset == 'CEVAL-VALID':
                    table[KV_config_str(k_nbit, v_nbit)][dataset] = extrace_value(lines, '|ceval-valid                                            |      2|none             |      |acc        |↑  |')
                elif dataset == 'MMLU':
                    table[KV_config_str(k_nbit, v_nbit)][dataset] = extrace_value(lines, '|mmlu                                                   |      2|none             |      |acc        |↑  |')
                elif dataset == 'TriviaQA':
                    table[KV_config_str(k_nbit, v_nbit)][dataset] = extrace_value(lines, '|triviaqa                                               |      3|remove_whitespace|     0|exact_match|↑  |')
                elif dataset == 'RACE':
                    table[KV_config_str(k_nbit, v_nbit)][dataset] = extrace_value(lines, '|race                                                   |      2|none             |     0|acc        |↑  |')
                elif dataset == 'TruthfulQA':
                    table[KV_config_str(k_nbit, v_nbit)][dataset] = extrace_value(lines, '|truthfulqa_gen                                         |      3|none             |     0|bleu_acc   |↑  |')
        else:
            n_shot = 0
            if dataset == 'GSM8K 4-shot':
                n_shot = 4
            elif dataset == 'GSM8K 8-shot':
                n_shot = 8
            elif dataset == 'GSM8K 16-shot':
                n_shot = 16
            log_filename = f'{model_name}_gsm8k_k{k_nbit}_v{v_nbit}_n{n_shot}.log'
            if not os.path.exists(log_filename):
                continue
            with open(log_filename, 'r') as f:
                lines = f.readlines()
                if n_shot != 16:
                    table[KV_config_str(k_nbit, v_nbit)][dataset] = extrace_value(lines, f'|gsm8k|      3|flexible-extract|     {n_shot}|exact_match|↑  |')
                else:
                    table[KV_config_str(k_nbit, v_nbit)][dataset] = extrace_value(lines, f'|gsm8k|      3|flexible-extract|    {n_shot}|exact_match|↑  |')
# print(table)

print('\\multicolumn{11}{c}{\\textbf{' + display_model_name + '}} \\\\ \\hline')
print('\\multirow{5}{*}{KIVI}')
for kv_config in kv_configs:
    k_nbit, v_nbit = kv_config
    print('& ' + KV_config_str(k_nbit, v_nbit), end=' ')
    for dataset in datasets:
        print('&', table[KV_config_str(k_nbit, v_nbit)][dataset], end=' ')
    print('\\\\', end='')
    if kv_config == kv_configs[-1]:
        print('\\hline')
    else:
        print()
