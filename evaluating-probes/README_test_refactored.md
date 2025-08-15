# Test Refactored Config Execution Flow

This document provides a detailed step-by-step breakdown of how `test_refactored_config.yaml` is executed, including all intermediate steps, file outputs, and logic flow.

## Configuration Overview

**Config File**: `configs/test_refactored_config.yaml`

**Key Parameters**:
- Model: `google/gemma-2-9b-it`
- d_model: 3584
- Device: cuda:0
- Seeds: [42]
- Layers: [20]
- Components: ["resid_post"]
- Format Type: "r" (off-policy instruct)

**Experiments**:
1. `test-experiment` - Standard training
2. `test-llm-upsampling` - LLM-based data augmentation
3. `test-imbalanced` - Imbalanced dataset handling

**Architectures**:
1. `sklearn_linear` with `sklearn_linear_mean` config
2. `act_sim` with `act_sim_mean` config

## Step-by-Step Execution Flow

### Step 1: Main Entry Point
**File**: `main.py` → `main()` function

1. **Argument Parsing**: 
   - `config_yaml = "test_refactored_config.yaml"`
   - `rerun = False` (unless `--rerun` flag is used)

2. **Config Loading**:
   - Loads `configs/test_refactored_config.yaml`
   - Sets device to "cuda:0"
   - Model: "google/gemma-2-9b-it"
   - d_model: 3584

### Step 2: Model Loading
**File**: `main.py` → `main()` function

```python
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-9b-it", 
    device_map="cuda:0", 
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
```

### Step 3: Model Checks
**File**: `main.py` → `run_model_checks()`

1. **Check for on-policy experiments**:
   - All experiments have `on_policy: false`
   - So model checks WILL run

2. **Check existing runthrough directories**:
   - Looks for `results/test_refactored/runthrough_94_better_spam/`
   - Looks for `results/test_refactored/runthrough_87_is_spam/`
   - If they don't exist, runs model check

3. **Model Check Execution**:
   - Uses method: "it" (chat template)
   - Processes both datasets: "94_better_spam", "87_is_spam"
   - Creates filtered datasets based on model predictions
   - Saves results to `results/test_refactored/runthrough_*/`

### Step 4: LLM Upsampling Check
**File**: `main.py` → `run_llm_upsampling()`

1. **Check for LLM upsampling experiments**:
   - `test-llm-upsampling` experiment has `rebuild_config` with `llm_upsampling: True`
   - So LLM upsampling WILL run

2. **LLM Upsampling Execution**:
   - Calls `src.llm_upsampling.llm_upsampling_script.run_llm_upsampling()`
   - Generates synthetic samples for dataset "94_better_spam"
   - Saves to `results/test_refactored/seed_42/llm_samples_<dataset_name>/samples_*.csv`

### Step 5: Activation Extraction
**File**: `main.py` → `extract_all_activations()`

1. **Collect all datasets**:
   - Train datasets: ["94_better_spam"]
   - Eval datasets: ["94_better_spam", "87_is_spam"]
   - All datasets: ["94_better_spam", "87_is_spam"]

2. **For each dataset, layer, component**:
   - Dataset: "94_better_spam", Layer: 20, Component: "resid_post"
   - Dataset: "87_is_spam", Layer: 20, Component: "resid_post"

3. **Activation Extraction Logic**:
   - **Format**: "r" (off-policy instruct)
   - **On-policy**: false for all experiments
   - **Extraction**: Uses prompts only (not responses)

4. **For each dataset**:
   - Creates `Dataset` object
   - Calls `ds.act_manager.ensure_texts_cached()` with format_type="r"
   - Extracts activations using `_extract_activations()` method
   - Computes and saves aggregated activations (mean, max, last, softmax)

### Step 6: Model Unloading
**File**: `main.py` → `main()` function

1. **Free GPU Memory**:
   ```python
   del model
   torch.cuda.empty_cache()
   ```

### Step 7: Job Creation
**File**: `main.py` → `create_probe_jobs()`

1. **For each experiment**:
   - `test-experiment`
   - `test-llm-upsampling` 
   - `test-imbalanced`

2. **For each architecture**:
   - `sklearn_linear` with `config_name: "sklearn_linear_mean"`
   - `act_sim` with `config_name: "act_sim_mean"`

3. **Hyperparameter Sweep Generation**:
   - **sklearn_linear**: Creates 5 configs with C=[0.01, 0.1, 1.0, 10.0, 100.0]
   - **act_sim**: Creates 1 config (no hyperparameters to sweep)

4. **Job Creation**:
   - For each experiment × architecture × hyperparameter × dataset × layer × component × seed
   - Creates `ProbeJob` objects

**Expected Jobs**:
- `test-experiment` × `sklearn_linear` × 5 C values × 1 dataset × 1 layer × 1 component × 1 seed = 5 jobs
- `test-experiment` × `act_sim` × 1 config × 1 dataset × 1 layer × 1 component × 1 seed = 1 job
- `test-llm-upsampling` × `sklearn_linear` × 5 C values × 1 dataset × 1 layer × 1 component × 1 seed = 5 jobs
- `test-llm-upsampling` × `act_sim` × 1 config × 1 dataset × 1 layer × 1 component × 1 seed = 1 job
- `test-imbalanced` × `sklearn_linear` × 5 C values × 1 dataset × 1 layer × 1 component × 1 seed = 5 jobs
- `test-imbalanced` × `act_sim` × 1 config × 1 dataset × 1 layer × 1 component × 1 seed = 1 job
- **Total**: 18 jobs

### Step 8: Parallel Job Processing
**File**: `main.py` → `process_probe_job()` (called in parallel)

For each job, the following happens:

#### Step 8a: Directory Creation
```python
seed_dir = results_dir / f"seed_{job.seed}"  # results/test_refactored/seed_42/
experiment_dir = seed_dir / job.experiment_name  # results/test_refactored/seed_42/test-experiment/
trained_dir = experiment_dir / "trained"
val_eval_dir = experiment_dir / "val_eval" 
test_eval_dir = experiment_dir / "test_eval"
gen_eval_dir = experiment_dir / "gen_eval"
```

#### Step 8b: Training
**File**: `utils_training.py` → `train_single_probe()`

1. **Filename Generation**:
   - Base: `train_on_94_better_spam_sklearn_linear_mean_L20_resid_post`
   - With hyperparams: `train_on_94_better_spam_sklearn_linear_mean_L20_resid_post_C0.01` (for C=0.01)
   - Final path: `trained_dir/train_on_94_better_spam_sklearn_linear_mean_L20_resid_post_C0.01_state.npz`

2. **Caching Check**:
   - If file exists and `rerun=False`, skip training
   - Otherwise, proceed with training

3. **Dataset Preparation**:
   - **For `test-experiment`**: Simple dataset creation
   - **For `test-llm-upsampling`**: Uses `Dataset.build_llm_upsampled_dataset()`
   - **For `test-imbalanced`**: Uses `Dataset.build_imbalanced_train_balanced_eval()`

4. **Data Filtering** (for off-policy):
   - Filters data using model check results
   - Only keeps examples where model prediction matches ground truth

5. **Activation Loading**:
   - **sklearn_linear**: Uses `linear_mean` aggregated activations
   - **act_sim**: Uses `act_sim_mean` aggregated activations
   - Loads from cached files created in Step 5

6. **Probe Creation and Training**:
   - **sklearn_linear**: Creates `SklearnLinearProbe` with C parameter
   - **act_sim**: Creates `ActivationSimilarityProbe`
   - Fits probe on training data

7. **Probe Saving**:
   - Saves probe state to `.npz` file
   - Saves metadata to `.json` file

#### Step 8c: Evaluation
**File**: `utils_training.py` → `evaluate_single_probe()`

For each evaluation dataset:

1. **Dataset Selection**:
   - If `eval_dataset == train_dataset`: Evaluate on validation AND test sets
   - If `eval_dataset != train_dataset`: Evaluate on full dataset (only_test=True)

2. **Evaluation Logic**:
   - **val_eval**: Uses validation set from training data
   - **test_eval**: Uses test set from training data  
   - **gen_eval**: Uses entire evaluation dataset as test set

3. **Results Saving**:
   - Saves metrics to: `{eval_dir}/eval_on_{dataset}__{probe_filename}_seed{seed}_{agg}_results.json`

## Job Breakdown

### For each experiment, the 6 jobs are:

1. **sklearn_linear with C=0.01**
   - Architecture: `sklearn_linear`
   - Config: `sklearn_linear_mean` with C=0.01
   - Filename: `train_on_94_better_spam_sklearn_linear_mean_L20_resid_post_C0.01_state.npz`

2. **sklearn_linear with C=0.1**  
   - Architecture: `sklearn_linear`
   - Config: `sklearn_linear_mean` with C=0.1
   - Filename: `train_on_94_better_spam_sklearn_linear_mean_L20_resid_post_C0.1_state.npz`

3. **sklearn_linear with C=1.0**
   - Architecture: `sklearn_linear` 
   - Config: `sklearn_linear_mean` with C=1.0
   - Filename: `train_on_94_better_spam_sklearn_linear_mean_L20_resid_post_C1.0_state.npz`

4. **sklearn_linear with C=10.0**
   - Architecture: `sklearn_linear`
   - Config: `sklearn_linear_mean` with C=10.0  
   - Filename: `train_on_94_better_spam_sklearn_linear_mean_L20_resid_post_C10.0_state.npz`

5. **sklearn_linear with C=100.0**
   - Architecture: `sklearn_linear`
   - Config: `sklearn_linear_mean` with C=100.0
   - Filename: `train_on_94_better_spam_sklearn_linear_mean_L20_resid_post_C100.0_state.npz`

6. **act_sim (no hyperparameters)**
   - Architecture: `act_sim`
   - Config: `act_sim_mean` (no hyperparameter sweep)
   - Filename: `train_on_94_better_spam_act_sim_mean_L20_resid_post_state.npz`

## Expected Output Structure

```
results/test_refactored/
├── runthrough_94_better_spam/
├── runthrough_87_is_spam/
├── seed_42/
│   ├── llm_samples/
│   │   └── samples_*.csv
│   ├── test-experiment/
│   │   ├── trained/
│   │   │   ├── train_on_94_better_spam_sklearn_linear_mean_L20_resid_post_C0.01_state.npz
│   │   │   ├── train_on_94_better_spam_sklearn_linear_mean_L20_resid_post_C0.1_state.npz
│   │   │   ├── train_on_94_better_spam_sklearn_linear_mean_L20_resid_post_C1.0_state.npz
│   │   │   ├── train_on_94_better_spam_sklearn_linear_mean_L20_resid_post_C10.0_state.npz
│   │   │   ├── train_on_94_better_spam_sklearn_linear_mean_L20_resid_post_C100.0_state.npz
│   │   │   └── train_on_94_better_spam_act_sim_mean_L20_resid_post_state.npz
│   │   ├── val_eval/
│   │   │   └── eval_on_94_better_spam__*.json (6 files)
│   │   ├── test_eval/
│   │   │   └── eval_on_94_better_spam__*.json (6 files)
│   │   └── gen_eval/
│   │       └── eval_on_87_is_spam__*.json (6 files)
│   ├── test-llm-upsampling/
│   │   ├── trained/
│   │   │   └── (6 probe files with llm_upsampling suffix)
│   │   ├── val_eval/
│   │   ├── test_eval/
│   │   └── gen_eval/
│   └── test-imbalanced/
│       ├── trained/
│       │   └── (6 probe files with imbalanced suffix)
│       ├── val_eval/
│       ├── test_eval/
│       └── gen_eval/
```

## Running the Test

```bash
cd pivotal-research/evaluating-probes
python src/main.py -c test_refactored
```

To force rerun all probes:
```bash
python src/main.py -c test_refactored --rerun
```
