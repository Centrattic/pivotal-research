
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re

def plot_logit_diffs_by_class(
    probe,
    test_acts,
    test_labels,
    class_names: dict,
    save_path: str = "logit_diffs_by_class_hist.png",
    main_diff: tuple = None,
    bins: int = 50,
    x_range: tuple = (-10, 10),
):
    """
    Plots histogram of logit differences for each class or for a specified pair, recalculating scores on the fly.
    Args:
        probe: trained probe object with a .predict_logits method
        test_acts: test set activations (numpy array)
        test_labels: test set true labels (numpy array)
        class_names: dict mapping class idx to class name (e.g., {0: 'French', 1: 'English'})
        save_path: Path to save the histogram image.
        main_diff: tuple (idx1, idx2) to plot logit_class1 - logit_class2. If None and two classes, uses (0,1).
        bins: Number of bins for the histogram.
    """
    logits = probe.predict_logits(test_acts)  # shape: (n_samples, n_classes) or (n_samples, 1)
    class_keys = list(class_names.keys())
    if main_diff is None:
        if len(class_keys) == 2:
            main_diff = (class_keys[0], class_keys[1])
        else:
            raise ValueError("main_diff must be specified for more than 2 classes.")
    idx1, idx2 = main_diff
    name1 = class_names[idx1]
    name2 = class_names[idx2]
    # Handle binary and multiclass
    if logits.shape[1] == 1 or logits.ndim == 1:
        # Binary: logit for class 1 (positive class)
        # For logit diff, use logit (since only one output)
        logit_diff = logits[:, 0]
    else:
        # Multiclass: difference between two class logits
        logit_diff = logits[:, idx1] - logits[:, idx2]
    plt.figure(figsize=(8, 5))
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, (idx, name) in enumerate(class_names.items()):
        mask = test_labels == idx
        plt.hist(
            logit_diff[mask],
            bins=bins,
            range=x_range,
            alpha=0.7,
            label=f"{name} (N={mask.sum()})",
            color=color_cycle[i % len(color_cycle)],
            edgecolor="black"
        )
    plt.xlabel(f"Logit difference: {name1} - {name2}")
    plt.ylabel("Count")
    plt.title(f"Logit difference histogram: {name1} vs {name2}")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved histogram to {save_path}")
    plt.close()


def plot_class_logit_distributions(
    all_top_logits,          # list of dicts: each dict maps class idx to logit value
    all_labels,              # list of int class indices per sample
    class_names,             # dict, e.g. {0: "French", 1: "English"}
    bins=20,
    x_range=(-10, 10),
    run_name="run",
    save_path=None,
):
    """
    Plots the distribution of the logit for the correct class for each true label.
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4))
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, (idx, name) in enumerate(class_names.items()):
        # Collect logits for samples where the true label == idx
        class_logits = [
            logit_dict[idx]
            for logit_dict, label in zip(all_top_logits, all_labels)
            if idx in logit_dict and label == idx
        ]
        if class_logits:
            plt.hist(
                class_logits,
                bins=bins,
                range=x_range,
                alpha=0.7,
                label=f"{name} (N={len(class_logits)})",
                color=color_cycle[i % len(color_cycle)]
            )
    plt.xlabel("Logit Value for True Class")
    plt.ylabel("Frequency")
    plt.title(f"Logit Distributions for True Classes\n({run_name})")
    plt.legend()
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
    plt.show()


def plot_rebuild_experiment_results_grid(results_json_paths, probe_names, class_names, rebuild_configs, save_path=None, metrics=('acc', 'auc', 'precision', 'recall', 'fpr')):
    """
    Plots a Nx3 grid (rows=probes, cols=3 settings) of results from multiple probes' rebuild experiments.
    Uses rebuild_configs from the config to group results by experimental setting.
    Args:
        results_json_paths: list of paths to results JSONs (one per probe)
        probe_names: list of probe names (row labels)
        class_names: dict mapping class idx to class name
        rebuild_configs: list of dicts from config['rebuild_config']
        save_path: if provided, saves the plot to this path
        metrics: tuple/list of metric keys to plot (default: all supported)
    """
    import json
    import matplotlib.pyplot as plt
    import numpy as np
    n_probes = len(results_json_paths)
    ncols = 3
    nrows = n_probes
    fig, axs = plt.subplots(nrows, ncols, figsize=(6*ncols, 4*nrows), squeeze=False)
    setting_titles = [
        'Constant French, Increasing English',
        'Constant % French, Increasing Total',
        'Constant Total, Increasing % French'
    ]
    # Group rebuild_configs by setting
    constant_french = []
    constant_percent = []
    constant_total = []
    # Group by class_percents and total_samples
    perc_to_samples = {}
    samples_to_perc = {}
    for rc in rebuild_configs:
        if 'class_counts' in rc:
            constant_french.append(rc)
        elif 'class_percents' in rc:
            perc_tuple = tuple(sorted(rc['class_percents'].items()))
            perc_to_samples.setdefault(perc_tuple, []).append(rc['total_samples'])
            samples_to_perc.setdefault(rc['total_samples'], []).append(perc_tuple)
    # Now, assign each config to the correct group
    for rc in rebuild_configs:
        if 'class_percents' in rc:
            perc_tuple = tuple(sorted(rc['class_percents'].items()))
            total_samples = rc['total_samples']
            # If this percent is used with multiple total_samples, it's constant percent
            if len(set(perc_to_samples[perc_tuple])) > 1:
                constant_percent.append(rc)
            # If this total_samples is used with multiple percents, it's constant total
            elif len(set(samples_to_perc[total_samples])) > 1:
                constant_total.append(rc)
            # Fallback: if only one config, treat as constant percent
            else:
                constant_percent.append(rc)
    print(f"Grouping: {len(constant_french)} constant_french, {len(constant_percent)} constant_percent, {len(constant_total)} constant_total")
    for row, (results_json_path, probe_name) in enumerate(zip(results_json_paths, probe_names)):
        with open(results_json_path, 'r') as f:
            results = json.load(f)
        print(f"Loaded {results_json_path}, keys: {list(results.keys())}")
        # Map result keys to config entries
        key_to_config = {}
        for key, metrics_dict in results.items():
            # Try to match by class_counts/class_percents/total_samples/seed
            for rc in rebuild_configs:
                match = True
                if 'class_counts' in rc:
                    if not all(f"class{cls}_{rc['class_counts'][cls]}" in key for cls in rc['class_counts']):
                        match = False
                if 'class_percents' in rc:
                    for cls in rc['class_percents']:
                        pct = int(rc['class_percents'][cls]*100)
                        if f"class{cls}_{pct}pct" not in key:
                            match = False
                    if f"total{rc['total_samples']}" not in key:
                        match = False
                if 'seed' in rc and f"seed{rc['seed']}" not in key:
                    match = False
                if match:
                    key_to_config[key] = rc
                    break
        # Group results by setting
        setting_data = [[], [], []]  # 0: constant_french, 1: constant_percent, 2: constant_total
        for key, metrics_dict in results.items():
            rc = key_to_config.get(key)
            if rc is None:
                print(f"Warning: Could not match key {key} to a rebuild_config entry.")
                continue
            if rc in constant_french:
                n_french = rc['class_counts'][0] if 0 in rc['class_counts'] else list(rc['class_counts'].values())[0]
                n_english = rc['class_counts'][1] if 1 in rc['class_counts'] else list(rc['class_counts'].values())[1]
                total = n_french + n_english
                pct_french = 100 * n_french / total if total > 0 else 0
                val_dict = metrics_dict.get('all_examples', metrics_dict)
                setting_data[0].append((n_english, val_dict, n_french, pct_french))
            elif rc in constant_percent:
                n_french = int(rc['class_percents'][0] * rc['total_samples']) if 0 in rc['class_percents'] else 0
                pct_french = rc['class_percents'][0] * 100 if 0 in rc['class_percents'] else 0
                total = rc['total_samples']
                val_dict = metrics_dict.get('all_examples', metrics_dict)
                setting_data[1].append((total, val_dict, n_french, pct_french))
            elif rc in constant_total:
                n_french = int(rc['class_percents'][0] * rc['total_samples']) if 0 in rc['class_percents'] else 0
                pct_french = rc['class_percents'][0] * 100 if 0 in rc['class_percents'] else 0
                total = rc['total_samples']
                val_dict = metrics_dict.get('all_examples', metrics_dict)
                setting_data[2].append((pct_french, val_dict, n_french, pct_french))
        # Sort by x-axis
        for i in range(3):
            setting_data[i].sort(key=lambda x: x[0])
        # Plot each setting
        for col, (data, title) in enumerate(zip(setting_data, setting_titles)):
            ax = axs[row][col]
            if data:
                x = [d[0] for d in data]
                n_french = [d[2] for d in data]
                pct_french = [d[3] for d in data]
                for metric in metrics:
                    y = [d[1].get(metric, np.nan) for d in data]
                    print(f"Plotting {metric} for {probe_name}, {title}: x={x}, y={y}")
                    ax.plot(x, y, marker='o', label=metric)
                xticklabels = [f"{xi}\nN_fr={nf}\n%fr={pf:.1f}" for xi, nf, pf in zip(x, n_french, pct_french)]
                ax.set_xticks(x)
                ax.set_xticklabels(xticklabels, rotation=30, ha='right')
                ax.legend()
            else:
                print(f"No data for {probe_name}, {title}")
            ax.set_title(title)
            if col == 0:
                ax.set_ylabel(f"{probe_name}")
            else:
                ax.set_ylabel("")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved rebuild experiment results grid to {save_path}")
    else:
        plt.show()


def plot_probe_score_histogram_subplots(probe_results, class_names=None, save_path=None, bins=50):
    """
    Plots histograms of probe scores for all probes, with a subplot for each probe.
    Args:
        probe_results: dict mapping probe name to dict with 'scores' and 'labels' arrays
        class_names: dict mapping class index to name (optional)
        save_path: if provided, saves the plot to this path
        bins: number of histogram bins
    """
    import matplotlib.pyplot as plt
    import numpy as np
    n_probes = len(probe_results)
    if n_probes == 0:
        print("No probe results to plot.")
        return
    ncols = min(3, n_probes)
    nrows = int(np.ceil(n_probes / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(6*ncols, 4*nrows), squeeze=False)
    for idx, (probe_name, res) in enumerate(probe_results.items()):
        row, col = divmod(idx, ncols)
        ax = axs[row][col]
        scores = np.array(res['scores'])
        labels = np.array(res['labels'])
        if class_names is None:
            unique_classes = np.unique(labels)
            class_names_map = {c: f"Class {c}" for c in unique_classes}
        else:
            class_names_map = class_names
        for cls in np.unique(labels):
            mask = (labels == cls)
            ax.hist(
                scores[mask],
                bins=bins,
                alpha=0.7,
                label=f"{class_names_map.get(cls, str(cls))} (N={mask.sum()})",
                edgecolor="black"
            )
        ax.set_xlabel("Probe Score")
        ax.set_ylabel("Count")
        ax.set_title(f"Probe: {probe_name}")
        ax.legend()
    # Hide any unused subplots
    for idx in range(n_probes, nrows*ncols):
        row, col = divmod(idx, ncols)
        fig.delaxes(axs[row][col])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved probe score histogram subplots to {save_path}")
    else:
        plt.show()

# if __name__ == '__main__':
#     diff_file = "./results/gender_experiment_gemma/runthrough_4_hist_fig_ismale/logit_diff_gender-pred_model_check.csv"
#     plot_logit_diffs_by_gender(diff_file)