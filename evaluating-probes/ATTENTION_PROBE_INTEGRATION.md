# Attention Probe Integration with find_best_fit

This document explains how attention probes are integrated with the main.py workflow to use the efficient `find_best_fit` function while maintaining proper hyperparameter naming.

## Overview

Attention probes use the `find_best_fit` function to efficiently train multiple hyperparameter combinations simultaneously, avoiding repeated GPU data transfers. The integration ensures that the final saved probe has the correct hyperparameters in its filename.

## How It Works

### 1. Job Creation (main.py)
- Only one attention probe job is created per dataset/layer/component combination
- The job uses the default probe config (lr=5e-4, weight_decay=0.0)
- No separate jobs are created for different hyperparameter combinations

### 2. Training (utils_training.py)
When an attention probe job is processed:

1. **Check if using find_best_fit**: If the probe config has default hyperparameters, use `find_best_fit`
2. **Run hyperparameter sweep**: Train multiple combinations simultaneously using efficient batch training
3. **Update probe config**: Set the best hyperparameters found by `find_best_fit`
4. **Regenerate filename**: Create a new filename that includes the best hyperparameters
5. **Save probe**: Save the probe with the correct hyperparameter-enhanced filename

### 3. Filename Generation
The final probe filename includes the best hyperparameters found:

```
# Example filenames:
attention_probe_lr5.00e-04_wd0.00e+00_state.npz  # Default config
attention_probe_lr1.00e-03_wd1.00e-05_state.npz  # Best hyperparameters found
```

### 4. Metadata
The probe metadata includes:
- Standard probe information (dataset, layer, component, etc.)
- Best hyperparameters found by `find_best_fit`
- Final loss achieved with the best hyperparameters

## Hyperparameter Sweep Values

The `find_best_fit` function uses these hyperparameter ranges:

```python
lr_values = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3]
weight_decay_values = [0.0, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]
```

This creates 81 hyperparameter combinations that are trained efficiently in parallel.

## Benefits

1. **Efficiency**: All hyperparameter combinations train simultaneously on each batch
2. **Reduced GPU transfers**: Data is moved to GPU once per batch, not per hyperparameter combination
3. **Proper naming**: Saved probes have correct hyperparameters in their filenames
4. **Clean organization**: All swept probes are saved directly in the trained/ directory
5. **Compatibility**: Works seamlessly with the existing main.py job system
6. **Flexibility**: Easy to adjust hyperparameter ranges by modifying the lists in utils_training.py

## File Structure

After training, you'll find:

```
results/
└── run_name/
    └── seed_42/
        └── experiment_name/
            └── trained/
                ├── attention_probe_lr1.00e-03_wd1.00e-05_state.npz  # Best probe (selected)
                ├── attention_probe_lr1.00e-03_wd1.00e-05_meta.json   # Metadata
                ├── attention_probe_lr1.00e-03_wd0.00e+00_state.npz   # Best for wd=0.0
                ├── attention_probe_lr5.00e-04_wd1.00e-06_state.npz   # Best for wd=1e-6
                ├── attention_probe_lr2.00e-04_wd5.00e-06_state.npz   # Best for wd=5e-6
                └── ... (all best probes per weight decay)
```

## Usage

Simply add an attention probe to your config:

```yaml
architectures:
  - name: "attention"
    config_name: "attention"
```

The system will automatically:
1. Use `find_best_fit` for hyperparameter optimization
2. Save the best probe with proper hyperparameter naming
3. Save all best probes per weight decay directly in the trained/ directory
