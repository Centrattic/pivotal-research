# Visualization System

This directory contains the visualization system for probe evaluation results.

## Files

### Core Files
- **`plot_generator.py`**: Original visualization system for default probes
- **`data_loader.py`**: Data loading and filtering utilities
- **`build_metrics_index.py`**: Builds metrics index from results files (now includes val_eval)
- **`generate_viz.py`**: Main script to generate all visualizations

### New Hyperparameter Analysis Files
- **`viz_util.py`**: Reusable visualization utilities with hyperparameter filtering
- **`hyperparameter_analysis.py`**: Cross-validation based hyperparameter selection and plotting
- **`hyp_sweep.py`**: Dedicated hyperparameter sweep plots

## Usage

### Generate Main Plots (Default Probes)
```bash
python src/visualize/generate_viz.py --plot-type main
```

### Generate Hyperparameter Sweep Plots
```bash
python src/visualize/generate_viz.py --plot-type hyp-sweep
```

### Generate Cross-Validation Based Plots
```bash
python src/visualize/generate_viz.py --plot-type cv-hyp
```

### Skip Existing Plots
```bash
python src/visualize/generate_viz.py --plot-type main --skip-existing
```

## Output Structure

- **`visualizations/main/`**: Main plots (default probes)
- **`visualizations/hyp/`**: Hyperparameter analysis plots (sweeps + cross-validation)

## System Architecture

1. **`build_metrics_index.py`**: Now includes val_eval results for cross-validation
2. **`data_loader.py`**: Filters out val_eval by default (backward compatibility)
3. **`viz_util.py`**: Reusable plotting functions with hyperparameter filtering
4. **`plot_generator.py`**: Uses default hyperparameters (original behavior)
5. **`hyperparameter_analysis.py`**: Uses cross-validation to select best hyperparameters
6. **`hyp_sweep.py`**: Creates dedicated hyperparameter sweep plots

## Key Features

- **Backward Compatibility**: Original plots work unchanged
- **Cross-Validation**: Uses val_eval to select optimal hyperparameters
- **Hyperparameter Sweeps**: Dedicated plots for C, topk, lr, weight_decay analysis
- **Modular Design**: Reusable components for different plot types
