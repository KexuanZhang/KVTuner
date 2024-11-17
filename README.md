### Install
```bash
cd flexible_quant
pip install -e .
cd lm-evaluation-harness-X
pip install -e .
```

### Run Example
```bash
python3 example.py
```

### Run GSM8K
```bash
python3 example_gsm8k_cot_manyshot.py --model_name="mistralai/Mistral-7B-Instruct-v0.2" --k_bits=8 --v_bits=8 | tee Mistral_7B_k8v8_r128_g64_hqq.log
```

### Run LongBench
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 pred_longbench.py
```

### Run LM Evaluation Harness
```bash
lm_eval --model hf-quant \
    --model_args pretrained=meta-llama/Llama-2-7b-hf,nbits_key=8,nbits_value=8,residual_length=128,q_group_size=64,axis_key=0,axis_value=1 \
    --tasks hellaswag \
    --device cuda:0 \
    --batch_size 8
```
