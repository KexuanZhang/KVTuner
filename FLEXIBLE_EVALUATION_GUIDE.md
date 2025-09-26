# KVTuner Flexible Evaluation System Usage Guide

This guide shows how to use the updated `search_optuna_adaptive.py` script with flexible evaluation options for finding optimal quantization configurations.

## **Quick Start Examples**

### **1. Single Task Evaluation**
```bash
# GSM8K (default - math reasoning)
python search_optuna_adaptive.py \
  --model_name "Qwen/Qwen2.5-3B-Instruct" \
  --evaluation_task "gsm8k" \
  --max_per_layer_scale 4.0 \
  --limit 50 \
  --n_trials 100

# HellaSwag (commonsense reasoning)
python search_optuna_adaptive.py \
  --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct" \
  --evaluation_task "hellaswag" \
  --max_per_layer_scale 6.0 \
  --limit 30 \
  --n_trials 50

# HumanEval (code generation)
python search_optuna_adaptive.py \
  --model_name "codellama/CodeLlama-7b-Python-hf" \
  --evaluation_task "humaneval" \
  --max_per_layer_scale 5.0 \
  --limit 40 \
  --n_trials 75
```

### **2. Multi-Task Evaluation**
```bash
# Multiple specific tasks
python search_optuna_adaptive.py \
  --model_name "Qwen/Qwen2.5-7B-Instruct" \
  --evaluation_tasks hellaswag arc_easy winogrande boolq \
  --max_per_layer_scale 4.5 \
  --limit 40 \
  --n_trials 100

# Math reasoning suite
python search_optuna_adaptive.py \
  --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct" \
  --evaluation_tasks gsm8k math_qa arc_challenge \
  --max_per_layer_scale 5.0 \
  --limit 50 \
  --n_trials 80
```

### **3. Domain-Specific Evaluation**
```bash
# General-purpose models
python search_optuna_adaptive.py \
  --model_name "Qwen/Qwen2.5-3B-Instruct" \
  --domain general \
  --max_per_layer_scale 4.0 \
  --limit 30 \
  --n_trials 100

# Code generation models  
python search_optuna_adaptive.py \
  --model_name "codellama/CodeLlama-7b-Instruct-hf" \
  --domain code \
  --max_per_layer_scale 5.0 \
  --limit 50 \
  --n_trials 75

# Reasoning-focused evaluation
python search_optuna_adaptive.py \
  --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct" \
  --domain reasoning \
  --max_per_layer_scale 6.0 \
  --limit 40 \
  --n_trials 120

# Comprehensive evaluation (multiple domains)
python search_optuna_adaptive.py \
  --model_name "Qwen/Qwen2.5-7B-Instruct" \
  --domain comprehensive \
  --max_per_layer_scale 5.5 \
  --limit 35 \
  --n_trials 150
```

### **4. Local Model Configuration**
```bash
# Auto-detect layer count
python search_optuna_adaptive.py \
  --model_name "/path/to/your/local/model" \
  --domain general \
  --max_per_layer_scale 4.0 \
  --limit 30 \
  --n_trials 75

# Manually specify layer count
python search_optuna_adaptive.py \
  --model_name "/path/to/your/local/Qwen2.5-3B" \
  --local_model_layers 36 \
  --evaluation_task "hellaswag" \
  --max_per_layer_scale 4.5 \
  --limit 40 \
  --n_trials 100
```

### **5. Custom Dataset Evaluation**
```bash
# Prepare your dataset (JSON format)
echo '[
  {"prompt": "What is 2+2?", "answer": "4"},
  {"prompt": "Capital of France?", "answer": "Paris"},
  {"prompt": "Who wrote Romeo and Juliet?", "answer": "Shakespeare"}
]' > my_custom_dataset.json

# Run with custom dataset
python search_optuna_adaptive.py \
  --model_name "/path/to/your/model" \
  --custom_dataset_path my_custom_dataset.json \
  --max_per_layer_scale 4.0 \
  --limit 50 \
  --n_trials 100
```

## **Available Options**

### **Evaluation Tasks** (Single Task Mode)
| Task | Domain | Metric | Description |
|------|--------|--------|-------------|
| `gsm8k` | Math | Exact Match | Grade school math problems |
| `hellaswag` | Reasoning | Accuracy (Norm) | Commonsense reasoning |
| `arc_easy` | Knowledge | Accuracy | Elementary science questions |
| `arc_challenge` | Knowledge | Accuracy (Norm) | Advanced science questions |
| `winogrande` | Reasoning | Accuracy | Pronoun resolution |
| `boolq` | Reading | Accuracy | Yes/no questions |
| `piqa` | Reasoning | Accuracy (Norm) | Physical reasoning |
| `humaneval` | Code | Pass@1 | Python programming |
| `mbpp` | Code | Pass@1 | Basic programming problems |
| `mmlu` | Knowledge | Accuracy | Academic subjects |
| `truthfulqa_mc2` | Knowledge | Accuracy | Truthfulness evaluation |
| `drop` | Reading | F1 Score | Reading comprehension |

### **Domain Task Suites**
| Domain | Tasks Included |
|--------|----------------|
| `general` | hellaswag, arc_easy, winogrande, boolq |
| `reasoning` | gsm8k, arc_challenge, piqa, hellaswag |
| `knowledge` | mmlu, truthfulqa_mc2, arc_challenge |
| `code` | humaneval, mbpp |
| `reading` | drop, boolq, piqa |
| `math` | gsm8k, math_qa |
| `comprehensive` | hellaswag, arc_challenge, gsm8k, winogrande, boolq, piqa |

### **Key Parameters**
- `--max_per_layer_scale`: Target average bits (2.0-8.0, lower = more compression)
- `--limit`: Number of evaluation samples per task (10-200, higher = more accurate)
- `--n_trials`: Optimization trials (50-500, higher = better optimization)
- `--num_fewshots`: Few-shot examples (0-8, task-dependent)
- `--quant_scheme`: Quantization scheme (`per-token-asym` or `per-channel-asym`)

## **Advanced Usage**

### **Parameter Tuning for Different Use Cases**

#### **Fast Prototyping**:
```bash
--limit 10 --n_trials 30 --max_per_layer_scale 6.0
```

#### **Balanced Search**:
```bash
--limit 50 --n_trials 100 --max_per_layer_scale 4.5
```

#### **Thorough Optimization**:
```bash
--limit 100 --n_trials 300 --max_per_layer_scale 4.0
```

#### **Aggressive Compression**:
```bash
--limit 75 --n_trials 200 --max_per_layer_scale 3.0
```

### **Model-Specific Recommendations**

#### **For Chat/General Models**:
```bash
--domain general --max_per_layer_scale 4.5 --limit 50
```

#### **For Code Models**:
```bash
--domain code --max_per_layer_scale 5.0 --limit 40
```

#### **For Math/Reasoning Models**:
```bash
--domain reasoning --max_per_layer_scale 5.5 --limit 60
```

#### **For Academic/Knowledge Models**:
```bash
--domain knowledge --max_per_layer_scale 6.0 --limit 80
```

## **Output Analysis**

### **Study Results**
The search creates an SQLite database with results:
```
OPTUNA_SEARCH_ADAPTIVE_[MODEL]_[TASKS]_FIRST[LIMIT]_[SHOTS]SHOTS_MAXSCALE[SCALE]_SCHEME[SCHEME].db
```

### **Expected Output**
```
Trial 0: Group configs [1,2,0,3,1,2,0,1] → 4.2 bits, 0.85 accuracy ✓
Trial 1: Group configs [0,0,0,0,0,0,0,0] → 8.0 bits, 0.95 accuracy (constraint violation)
Trial 2: Group configs [4,4,3,4,4,3,4,3] → 3.1 bits, 0.72 accuracy ✓
...
hellaswag acc_norm: 0.834
arc_easy acc: 0.762
winogrande acc: 0.698
boolq acc: 0.821
Average score across 4 tasks: 0.779
Trial result: accuracy=0.779, tot_scale=4.48
```

### **Pareto Front Analysis**
The optimization finds multiple optimal solutions:
- **High Quality**: 6.2 bits, 0.89 accuracy
- **Balanced**: 4.5 bits, 0.81 accuracy  
- **High Compression**: 3.2 bits, 0.73 accuracy

### **Next Steps**
1. **Extract best configuration** from the database
2. **Convert to YAML format** for vLLM integration
3. **Test with vLLM serving** using the optimal preset
4. **Deploy in production** with your chosen trade-off point

This flexible system allows you to optimize quantization for your specific use case, model type, and quality requirements while supporting both standard benchmarks and custom evaluation datasets.