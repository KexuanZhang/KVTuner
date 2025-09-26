#!/usr/bin/env python3
"""
Extract optimal KVTuner configurations from Optuna search results and convert to YAML format.
Usage: python extract_yaml_configs.py <database_file.db> [--output_dir ./configs]
"""

import optuna
import yaml
import sys
import os
import argparse
from pathlib import Path

def parse_quant_config(quant_config: str):
    """Parse quantization config string like 'K4V2' or 'KV4'"""
    if len(quant_config) == 3:  # KV4
        precision = int(quant_config[2])
        return {'nbits_key': precision, 'nbits_value': precision}
    # K4V2 format
    precision_key = int(quant_config[1])
    precision_value = int(quant_config[3])
    return {'nbits_key': precision_key, 'nbits_value': precision_value}

def get_layer_grouping_config(model_name: str, quant_scheme: str):
    """Get the layer grouping configuration - copied from search script"""
    
    # Layer grouping configurations (copied from search_optuna_adaptive.py)
    LAYER_GROUPING_CONFIG = {
        'Meta-Llama-3.1-8B-Instruct': {
            'per-token-asym': [[0], [1, 2, 3, 4, 7, 13, 18, 25, 27, 31], [5, 6, 12, 21, 26, 28], [8, 9, 10, 11, 14, 15, 16, 17, 20, 30], [19, 22], [23, 24, 29]],
            'per-channel-asym': [[0], [1, 2, 3, 7, 29, 31], [4, 25, 27], [5, 21, 23, 24], [6, 8, 9, 10, 11, 12, 14, 15, 16, 18, 19, 20, 22, 26, 28, 30], [13, 17]],
        },
        'Mistral-7B-Instruct-v0.3': {
            'per-token-asym': [[0], [1, 2], [3, 4, 23, 31], [5, 6], [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 29, 30]],
            'per-channel-asym': [[0, 1, 31], [2, 3, 4], [6, 27, 29], [7, 8, 10, 18], [9, 14], [5, 21, 22, 23, 24, 25, 26, 28, 30], [11, 12, 13, 15, 17, 19, 20], [16]],
        },
        'Qwen2.5-3B-Instruct': {
            'per-token-asym': [[0], [1, 3, 4, 5, 6, 8, 9, 12, 13, 15, 20], [2, 14, 23, 35], [7, 11, 16, 25, 28, 32], [10, 19, 24, 26, 33], [17, 30, 31, 34], [21, 22], [18, 27, 29]],
            'per-channel-asym': [[0, 1], [2, 4], [34, 35], [3, 6, 11, 13, 23], [5, 7, 25, 32, 33], [8, 16, 18, 21, 22, 24, 26, 27, 30], [9, 10, 14, 15, 17, 19, 20, 29, 31], [12, 28]],
        },
        'Qwen2.5-7B-Instruct': {
            'per-token-asym': [[0], [1, 2, 4, 5, 25], [6, 19], [7, 10, 11, 15, 23], [8, 24], [9, 12, 16, 17, 18, 21, 22, 26], [14, 20], [3, 13, 27]],
            'per-channel-asym': [[0, 2], [1, 3], [4, 5, 12, 22, 23, 24, 25], [7, 9, 10, 13, 14, 16, 18, 19, 20, 21, 27], [8, 26], [11, 15, 17], [6]],
        },
        'Qwen2.5-14B-Instruct': {
            'per-token-asym': [[0, 1, 2, 6, 11, 12, 19, 23, 24, 25, 41], [3, 4, 5, 8], [7, 10, 15], [9, 13, 14, 31, 38, 39], [16, 17, 18, 20, 21, 27, 28, 30, 32, 33, 34, 35, 36, 37, 40, 42, 43, 44, 46, 47], [22, 26, 29, 45]],
            'per-channel-asym': [[0, 2], [1, 3, 4], [5, 6, 8, 9, 12], [7, 10, 13, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35, 36, 37, 38, 44, 45, 46, 47], [11, 25, 41, 42], [14, 39, 40, 43], [22, 34]],
        },
        'Qwen2.5-32B-Instruct': {
            'per-token-asym': [[0, 2, 11, 12, 15, 33, 54, 57], [1, 5, 7, 8, 9, 10, 13, 14, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 55, 56, 58, 59, 60, 61, 62, 63], [3, 4], [6, 16]],
            'per-channel-asym': [[0, 1, 2, 3, 4], [11], [5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 32], [13, 15, 17, 22, 24, 25, 29, 30, 31, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62], [63]]
        },
        # For local models detected during search
        'local-model': {}  # Will be populated dynamically
    }

    SPECIAL_LAYERS = {
        'Meta-Llama-3.1-8B-Instruct': {
            'per-token-asym': {
                (0,): ['KV8', 'K4V8', 'KV4', 'K4V2', 'KV2'],
            },
            'per-channel-asym': {
                (0,): ['KV8', 'K4V8', 'KV4', 'K2V4', 'KV2'],
                (1, 2, 3, 7, 29, 31): ['KV8', 'K4V8', 'KV4', 'K4V2', 'KV2'],
            },
        },
        'Mistral-7B-Instruct-v0.3': {
            'per-token-asym': {
                (0,): ['KV8', 'K4V8', 'KV4', 'K2V4', 'KV2'],
            },
            'per-channel-asym': {
                (0, 1, 31): ['KV8', 'K4V8', 'KV4', 'K2V4', 'KV2'],
                (2, 3, 4, 6, 7, 8, 9, 10, 14, 18, 27, 29): ['KV8', 'K4V8', 'KV4', 'K4V2', 'KV2'],
            },
        },
        'Qwen2.5-3B-Instruct': {
            'per-token-asym': {
                (0,): ['KV8', 'K8V4', 'K8V2', 'K4V2', 'KV2'],
                (18, 27, 29): ['KV8', 'K8V4', 'K8V2', 'KV4', 'K4V2', 'KV2'],
            },
            'per-channel-asym': {
                (0, 1, 2, 4, 34, 35): ['KV8', 'K4V8', 'KV4', 'K2V4', 'KV2'],
                (3, 6, 11, 13, 23): ['KV8', 'K4V8', 'KV4', 'K4V2', 'KV2'],
            },
        },
        'Qwen2.5-7B-Instruct': {
            'per-token-asym': {
                (0,): ['KV8', 'K8V4', 'K8V2', 'K4V2', 'KV2'],
                (3, 13, 27): ['KV8', 'K8V4', 'K8V2', 'KV4', 'K4V2', 'KV2'],
            },
            'per-channel-asym': {
                (0, 1, 2, 3): ['KV8', 'K4V8', 'KV4', 'K2V4', 'KV2'],
                (6,): ['KV8', 'K4V8', 'KV4', 'K4V2', 'KV2'],
            },
        },
        'Qwen2.5-14B-Instruct': {
            'per-token-asym': {},
            'per-channel-asym': {
                (0, 1, 2, 3, 4): ['KV8', 'K4V8', 'KV4', 'K2V4', 'KV2'],
                (5, 6, 8, 9, 12): ['KV8', 'K4V8', 'KV4', 'K4V2', 'KV2'],
            },
        },
        'Qwen2.5-32B-Instruct': {
            'per-token-asym': {},
            'per-channel-asym': {
                (0, 1, 2, 3, 4, 11): ['KV8', 'K4V8', 'KV4', 'K2V4', 'KV2'],
                (5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 19, 20, 21, 23, 26, 27, 28, 32): ['KV8', 'K4V8', 'KV4', 'K4V2', 'KV2'],
                (63,): ['KV8', 'K8V4', 'KV4', 'K2V4', 'KV2'],
            },
        },
        'local-model': {}  # Will be populated dynamically
    }

    TOT_LAYER = {
        'Meta-Llama-3.1-8B-Instruct': 32,
        'Mistral-7B-Instruct-v0.3': 32,
        'Qwen2.5-3B-Instruct': 36,
        'Qwen2.5-7B-Instruct': 28,
        'Qwen2.5-14B-Instruct': 48,
        'Qwen2.5-32B-Instruct': 64,
    }

    STANDARD_KV_QUANT_CONFIG = ['KV8', 'K8V4', 'KV4', 'K4V2', 'KV2']

    # Model detection - match the search script logic exactly
    if 'Qwen2.5-3B-Instruct' in model_name:
        model_key = 'Qwen2.5-3B-Instruct'
    elif 'Qwen2.5-7B-Instruct' in model_name:
        model_key = 'Qwen2.5-7B-Instruct'
    elif 'Qwen2.5-14B-Instruct' in model_name:
        model_key = 'Qwen2.5-14B-Instruct'
    elif 'Qwen2.5-32B-Instruct' in model_name:
        model_key = 'Qwen2.5-32B-Instruct'
    elif 'Meta-Llama-3.1-8B-Instruct' in model_name:
        model_key = 'Meta-Llama-3.1-8B-Instruct'
    elif 'Mistral-7B-Instruct' in model_name:
        model_key = 'Mistral-7B-Instruct-v0.3'
    elif model_name == 'local-model' or model_name not in LAYER_GROUPING_CONFIG:
        # Handle truly local/unknown models
        if 'local-model' not in TOT_LAYER:
            # Default to 36 layers for local Qwen2.5-3B-Instruct
            num_layers = 36
            TOT_LAYER['local-model'] = num_layers
            
            # Create simple per-layer grouping for local models
            LAYER_GROUPING_CONFIG['local-model'] = {
                'per-token-asym': [[i] for i in range(num_layers)],
                'per-channel-asym': [[i] for i in range(num_layers)],
            }
            
            # Conservative special layers for local models
            SPECIAL_LAYERS['local-model'] = {
                'per-token-asym': {
                    (0,): ['KV8', 'K8V4', 'K8V2', 'K4V2'],
                    (num_layers-1,): ['KV8', 'K8V4', 'K8V2', 'K4V2'],
                },
                'per-channel-asym': {
                    (0, num_layers-1): ['KV8', 'K4V8', 'KV4', 'K4V2'],
                },
            }
        model_key = 'local-model'
    else:
        model_key = model_name.split('/')[-1].replace('-AWQ', '')
    
    print(f"Model name: {model_name}")
    print(f"Detected model key: {model_key}")

    layer_grouping = LAYER_GROUPING_CONFIG[model_key][quant_scheme]
    special_layers = SPECIAL_LAYERS[model_key][quant_scheme]
    
    # Build grouping quant template
    grouping_quant_template = []
    for group in layer_grouping:
        group_quant_template = STANDARD_KV_QUANT_CONFIG
        for layer in group:
            for special_layer in special_layers.keys():
                if layer in special_layer:
                    group_quant_template = special_layers[special_layer]
                    break
        grouping_quant_template.append(group_quant_template)
    
    return layer_grouping, grouping_quant_template

def convert_trial_to_yaml(trial, layer_grouping, grouping_quant_template):
    """Convert Optuna trial to KVTuner YAML format"""
    layer_config = {}
    
    for i, group in enumerate(layer_grouping):
        param_name = f'group_{i}'
        if param_name not in trial.params:
            print(f"Warning: Parameter {param_name} not found in trial {trial.number}")
            continue
            
        config_idx = trial.params[param_name]
        if config_idx >= len(grouping_quant_template[i]):
            print(f"Warning: Config index {config_idx} out of range for group {i}")
            continue
            
        quant_string = grouping_quant_template[i][config_idx]
        quant_config = parse_quant_config(quant_string)
        
        # Apply to all layers in group
        for layer in group:
            layer_config[layer] = quant_config
    
    return layer_config

def parse_study_name(study_name: str):
    """Parse study name to extract model info and parameters"""
    # Format: OPTUNA_SEARCH_ADAPTIVE_{model}_{task}_FIRST{limit}_{shots}SHOTS_MAXSCALE{scale}_SCHEME{scheme}
    parts = study_name.split('_')
    
    model_name = None
    quant_scheme = None
    max_scale = None
    task_info = None
    
    # Find scheme
    for i, part in enumerate(parts):
        if part.startswith('SCHEME'):
            quant_scheme = part.replace('SCHEME', '')
        elif part.startswith('MAXSCALE'):
            max_scale = part.replace('MAXSCALE', '')
    
    # Find model name (between ADAPTIVE and task info)
    model_parts = []
    collecting = False
    for part in parts:
        if part == 'ADAPTIVE':
            collecting = True
            continue
        elif part in ['HELLASWAG', 'GSM8K', 'ARC', 'WINOGRANDE', 'BOOLQ', 'PIQA', 'CUSTOM'] or part.startswith('FIRST'):
            if collecting:
                task_info = part
            break
        elif collecting:
            model_parts.append(part)
    
    if model_parts:
        model_name = '_'.join(model_parts).replace('_', '/')
        # Clean up model name
        model_name = model_name.split('/')[-1].replace('-AWQ', '')
    
    return model_name, quant_scheme, max_scale, task_info

def extract_yaml_configs(database_path: str, output_dir: str = "./kvtuner_configs", max_configs: int = 5):
    """Extract optimal configurations and save as YAML files"""
    
    if not os.path.exists(database_path):
        print(f"❌ Database file not found: {database_path}")
        return
    
    # Extract study name from database filename
    db_filename = Path(database_path).stem
    study_name = db_filename
    
    print(f"📁 Loading study from: {database_path}")
    print(f"🔍 Study name: {study_name}")
    
    try:
        # Load the study
        storage_name = f"sqlite:///{database_path}"
        study = optuna.load_study(study_name=study_name, storage=storage_name)
        
        print(f"✅ Study loaded successfully!")
        print(f"📊 Total trials: {len(study.trials)}")
        print(f"🎯 Best trials (Pareto front): {len(study.best_trials)}")
        
        if len(study.best_trials) == 0:
            print("❌ No optimal trials found!")
            return
        
        # Parse study name to extract parameters
        model_name, quant_scheme, max_scale, task_info = parse_study_name(study_name)
        print(f"🤖 Detected model: {model_name}")
        print(f"⚙️  Detected scheme: {quant_scheme}")
        print(f"📏 Max scale: {max_scale}")
        print(f"📝 Task info: {task_info}")
        
        # Get layer grouping configuration
        try:
            layer_grouping, grouping_quant_template = get_layer_grouping_config(model_name, quant_scheme)
            print(f"📋 Layer groups: {len(layer_grouping)}")
        except KeyError as e:
            print(f"❌ Model configuration not found: {e}")
            print("💡 Trying local-model configuration...")
            try:
                layer_grouping, grouping_quant_template = get_layer_grouping_config('local-model', quant_scheme)
                print(f"✅ Using local-model configuration with {len(layer_grouping)} layer groups")
            except Exception as e2:
                print(f"❌ Failed to load local-model config: {e2}")
                return
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract and save top configurations
        num_configs = min(max_configs, len(study.best_trials))
        print(f"\n🎯 Extracting top {num_configs} configurations:")
        
        saved_configs = []
        
        for i, trial in enumerate(study.best_trials[:num_configs]):
            accuracy = trial.values[0]
            compression = trial.values[1] 
            
            print(f"\n📋 Config {i+1}:")
            print(f"   🎯 Accuracy: {accuracy:.4f}")
            print(f"   🗜️  Avg bits: {compression:.2f}")
            print(f"   🔢 Trial: {trial.number}")
            
            try:
                # Convert to YAML
                yaml_config = convert_trial_to_yaml(trial, layer_grouping, grouping_quant_template)
                
                if not yaml_config:
                    print(f"   ⚠️  Skipping empty configuration")
                    continue
                
                # Create filename
                model_simple = model_name if model_name else "unknown_model"
                output_filename = f"{model_simple}_acc{accuracy:.3f}_bits{compression:.1f}_trial{trial.number}.yaml"
                output_path = os.path.join(output_dir, output_filename)
                
                # Save YAML
                with open(output_path, 'w') as f:
                    f.write(f"# KVTuner Configuration\n")
                    f.write(f"# Generated from Optuna search results\n")
                    f.write(f"# Model: {model_name}\n")
                    f.write(f"# Accuracy: {accuracy:.4f}\n")
                    f.write(f"# Average bits: {compression:.2f}\n")
                    f.write(f"# Trial: {trial.number}\n")
                    f.write(f"# Scheme: {quant_scheme}\n")
                    f.write(f"# Max scale constraint: {max_scale}\n")
                    f.write(f"# Task: {task_info}\n\n")
                    yaml.dump(yaml_config, f, default_flow_style=False, sort_keys=True)
                
                saved_configs.append(output_path)
                print(f"   💾 Saved: {output_filename}")
                
            except Exception as e:
                print(f"   ❌ Error processing trial {trial.number}: {e}")
                continue
        
        # Save best balanced configuration
        if len(study.best_trials) > 1:
            print(f"\n🎯 Finding best balanced configuration...")
            # Find configuration with good accuracy/compression trade-off
            # Prefer higher accuracy but penalize very high compression
            best_balanced = max(study.best_trials, 
                              key=lambda t: t.values[0] - max(0, (t.values[1] - 4.0) * 0.05))
            
            try:
                yaml_config = convert_trial_to_yaml(best_balanced, layer_grouping, grouping_quant_template)
                model_simple = model_name if model_name else "unknown_model"
                balanced_filename = f"{model_simple}_balanced_optimal.yaml"
                balanced_path = os.path.join(output_dir, balanced_filename)
                
                with open(balanced_path, 'w') as f:
                    f.write(f"# KVTuner Balanced Configuration (Recommended)\n")
                    f.write(f"# Generated from Optuna search results\n")
                    f.write(f"# Model: {model_name}\n")
                    f.write(f"# Accuracy: {best_balanced.values[0]:.4f}\n")
                    f.write(f"# Average bits: {best_balanced.values[1]:.2f}\n")
                    f.write(f"# Trial: {best_balanced.number}\n")
                    f.write(f"# Scheme: {quant_scheme}\n")
                    f.write(f"# Max scale constraint: {max_scale}\n")
                    f.write(f"# Task: {task_info}\n")
                    f.write(f"# Selection: Best balance of accuracy vs compression\n\n")
                    yaml.dump(yaml_config, f, default_flow_style=False, sort_keys=True)
                
                saved_configs.append(balanced_path)
                print(f"⭐ Saved balanced config: {balanced_filename}")
            except Exception as e:
                print(f"❌ Error creating balanced config: {e}")
        
        print(f"\n🎉 Successfully extracted {len(saved_configs)} configurations to {output_dir}")
        print(f"\n📁 Generated files:")
        for config_path in saved_configs:
            filename = os.path.basename(config_path)
            print(f"   • {filename}")
        
        print(f"\n💡 Usage with vLLM:")
        if saved_configs:
            example_file = os.path.basename(saved_configs[-1]) if 'balanced' in saved_configs[-1] else os.path.basename(saved_configs[0])
            print(f"   vllm serve /path/to/your/model \\")
            print(f"     --quantization kvtuner \\")
            print(f"     --kvtuner-preset-path {output_dir}/{example_file} \\")
            print(f"     --kvtuner-method kivi")
        
    except Exception as e:
        print(f"❌ Error loading study: {e}")
        print("💡 Make sure the database file exists and contains valid Optuna results")

def main():
    parser = argparse.ArgumentParser(
        description="Extract YAML configs from KVTuner Optuna search results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_yaml_configs.py search_results.db
  python extract_yaml_configs.py *.db --output_dir ./my_configs --max_configs 3
  
The script will generate YAML files compatible with vLLM's KVTuner integration.
        """
    )
    
    parser.add_argument("database_path", 
                       help="Path to the SQLite database file (.db) from Optuna search")
    parser.add_argument("--output_dir", default="./kvtuner_configs",
                       help="Output directory for YAML files (default: ./kvtuner_configs)")
    parser.add_argument("--max_configs", type=int, default=5,
                       help="Maximum number of configurations to extract (default: 5)")
    
    args = parser.parse_args()
    
    # Handle glob patterns
    if '*' in args.database_path:
        import glob
        db_files = glob.glob(args.database_path)
        if not db_files:
            print(f"❌ No database files found matching: {args.database_path}")
            sys.exit(1)
        elif len(db_files) > 1:
            print(f"📁 Found {len(db_files)} database files:")
            for f in db_files:
                print(f"   • {f}")
            print("🔍 Using the first one. Specify exact filename to use a different one.")
        args.database_path = db_files[0]
    
    extract_yaml_configs(args.database_path, args.output_dir, args.max_configs)

if __name__ == "__main__":
    main()