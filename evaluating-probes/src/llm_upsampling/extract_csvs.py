import argparse
import yaml
from pathlib import Path
import os
import sys
import pandas as pd
from src.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract_imbalanced_csvs(config_path):
    with open(
            config_path,
            'r',
    ) as f:
        config = yaml.safe_load(f)
    run_name = config['run_name']
    model_name = config['model_name']
    device = config.get(
        'device',
        'cpu',
    )
    seed = int(config.get(
        'seed',
        42,
    ))
    results_dir = Path('results') / run_name / 'imbalanced_csvs'
    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto" if "cuda" in (device or "") else None,
        torch_dtype=torch.float16 if (device and "cuda" in device) else torch.float32,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for experiment in config['experiments']:
        train_on = experiment['train_on']
        rebuild_config = experiment.get(
            'rebuild_config',
            {},
        )
        for group_name, configs in rebuild_config.items():
            for rc in configs:
                # Determine file suffix for saving
                if 'class_counts' in rc:
                    suffix = '_'.join(
                        [f"class{cls}_{rc['class_counts'][cls]}" for cls in sorted(rc['class_counts'])]
                    ) + f"_seed{rc.get('seed', seed,)}"
                elif 'class_percents' in rc:
                    suffix = '_'.join(
                        [f"class{cls}_{int(rc['class_percents'][cls]*100)}pct" for cls in sorted(rc['class_percents'])]
                    )
                    suffix += f"_total{rc['total_samples']}_seed{rc.get('seed', seed,)}"
                else:
                    continue
                # Load original dataset
                orig_ds = Dataset(
                    train_on,
                    model=model,
                    tokenizer=tokenizer,
                    model_name=model_name,
                    device=device,
                    seed=seed,
                )
                # Build imbalanced split
                ds = Dataset.rebuild_train_balanced_eval(
                    orig_ds,
                    train_class_counts=rc.get('class_counts'),
                    train_class_percents=rc.get('class_percents'),
                    train_total_samples=rc.get('total_samples'),
                    seed=rc.get(
                        'seed',
                        seed,
                    ),
                )
                # Save splits as CSV
                train_df = ds.df.iloc[ds.train_indices]
                val_df = ds.df.iloc[ds.val_indices]
                test_df = ds.df.iloc[ds.test_indices]
                train_df.to_csv(
                    results_dir / f"{train_on}_{suffix}_train.csv",
                    index=False,
                )
                val_df.to_csv(
                    results_dir / f"{train_on}_{suffix}_val.csv",
                    index=False,
                )
                test_df.to_csv(
                    results_dir / f"{train_on}_{suffix}_test.csv",
                    index=False,
                )
                print(f"Saved imbalanced CSVs for {train_on} ({suffix}) to {results_dir}")


# if __name__ == "__main__" or __name__ == "src.llm_upsampling.extract_imbalanced_csvs":
#     # Allow running as a module: python -m src.llm_upsampling.extract_imbalanced_csvs -c politician_exp
parser = argparse.ArgumentParser(
    description="Extract imbalanced CSVs from config file using rebuild_train_balanced_eval."
)
parser.add_argument(
    "-c",
    "--config",
    required=True,
    help="Config name (e.g. 'politician_exp') or path to config YAML file.",
)
args = parser.parse_args()
config_arg = args.config
# Expand short config name to full path if needed
if not config_arg.endswith('.yaml') and not config_arg.endswith('.yml'):
    config_path = Path('configs') / f"{config_arg}_config.yaml"
else:
    config_path = Path(config_arg)
if not config_path.exists():
    print(f"Config file not found: {config_path}")
    sys.exit(1)
extract_imbalanced_csvs(config_path)
