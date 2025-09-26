# KVTuner YAML Configuration Extractor

Extract optimal quantization configurations from Optuna search results and convert them to YAML format for use with vLLM.

## Quick Usage

### 1. After Your Search Completes

Your search will create a database file like:
```
OPTUNA_SEARCH_ADAPTIVE__home_data_semantics_so5_models_Qwen2.5-3B-Instruct_hellaswag_arc_easy_winogrande_PLUS1_FIRST30_4SHOTS_MAXSCALE4.0_SCHEMEper-token-asym.db
```

### 2. Extract YAML Configurations

```bash
# Basic usage - extract top 5 configs
python extract_yaml_configs.py *.db

# Custom output directory and number of configs
python extract_yaml_configs.py search_results.db --output_dir ./my_configs --max_configs 3

# Get help
python extract_yaml_configs.py --help
```

### 3. View Generated Files

```bash
ls -la kvtuner_configs/
# Example output:
# Qwen2.5-3B-Instruct_acc0.851_bits4.2_trial42.yaml      # High accuracy config
# Qwen2.5-3B-Instruct_acc0.847_bits3.8_trial67.yaml      # High compression config  
# Qwen2.5-3B-Instruct_acc0.839_bits3.2_trial23.yaml      # Very compressed config
# Qwen2.5-3B-Instruct_balanced_optimal.yaml              # Recommended balanced config
```

### 4. Use with vLLM

```bash
# Use the balanced configuration (recommended)
vllm serve /home/data/semantics/so5/models/Qwen2.5-3B-Instruct \
  --quantization kvtuner \
  --kvtuner-preset-path kvtuner_configs/Qwen2.5-3B-Instruct_balanced_optimal.yaml \
  --kvtuner-method kivi \
  --host 0.0.0.0 \
  --port 8007

# Or use a specific accuracy/compression trade-off
vllm serve /home/data/semantics/so5/models/Qwen2.5-3B-Instruct \
  --quantization kvtuner \
  --kvtuner-preset-path kvtuner_configs/Qwen2.5-3B-Instruct_acc0.851_bits4.2_trial42.yaml \
  --kvtuner-method kivi
```

## Generated YAML Format

Example generated file:
```yaml
# KVTuner Configuration
# Model: Qwen2.5-3B-Instruct
# Accuracy: 0.8510
# Average bits: 4.2
# Trial: 42
# Scheme: per-token-asym

0:
  nbits_key: 8
  nbits_value: 8
1:
  nbits_key: 4
  nbits_value: 2
2:
  nbits_key: 4
  nbits_value: 4
# ... configuration for all layers
```

## Configuration Selection Guide

- **`*_balanced_optimal.yaml`**: **Recommended** - Best overall trade-off
- **`*_acc0.9XX_bits8.0_*.yaml`**: Highest accuracy, minimal compression
- **`*_acc0.8XX_bits3.X_*.yaml`**: Aggressive compression, moderate accuracy loss
- **`*_acc0.7XX_bits2.X_*.yaml`**: Maximum compression, significant accuracy loss

## Troubleshooting

### Database Not Found
```bash
# Check for .db files in current directory
ls -la *.db

# Search in subdirectories
find . -name "*.db" -type f
```

### No Results Found
- Make sure your Optuna search completed successfully
- Check that the database file isn't corrupted
- Verify the study name matches the filename

### Model Configuration Missing
- The script includes configs for common models
- For local/custom models, it will auto-generate a simple configuration
- Check the console output for warnings about missing configurations

## Advanced Usage

### Extract from Multiple Search Results
```bash
# Process all .db files
for db in *.db; do
    python extract_yaml_configs.py "$db" --output_dir "configs_$(basename $db .db)"
done
```

### Compare Configurations
```bash
# Extract fewer configs for comparison
python extract_yaml_configs.py search_math.db --output_dir configs_math --max_configs 2
python extract_yaml_configs.py search_general.db --output_dir configs_general --max_configs 2
```

### Custom Analysis
You can modify the script to:
- Change the balanced configuration selection criteria
- Add custom filename patterns
- Export additional metadata
- Generate different YAML formats