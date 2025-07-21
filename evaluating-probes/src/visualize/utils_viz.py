
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import torch
from src.probes import LinearProbe, AttentionProbe
import glob
import os
import seaborn as sns

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


def plot_rebuild_experiment_results_grid(dataclass_results_dir, probe_names, class_names, rebuild_configs, save_path=None, metrics=('acc', 'auc', 'precision', 'recall', 'fpr')):
    """
    Plots a grid (n_probes x 3) of results from multiple probes' rebuild experiments.
    Columns: [Constant French, Increasing English | Constant % French, Increasing Total | Constant Total, Increasing % French]
    Args:
        dataclass_results_dir: path to dataclass_exps_{dataset}
        probe_names: list of probe names (row labels)
        class_names: dict mapping class idx to class name
        rebuild_configs: list of dicts from config['rebuild_config']
        save_path: if provided, saves the plot to this path
        metrics: tuple/list of metric keys to plot (default: all supported)
    """
    import json
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import glob

    # Find all per-config *_results.json files in the dataclass results dir (exclude allres)
    results_json_paths = sorted(glob.glob(os.path.join(dataclass_results_dir, '*_results.json')))
    results_json_paths = [p for p in results_json_paths if '_allres' not in p]
    if not results_json_paths:
        print(f"No per-config *_results.json files found in {dataclass_results_dir}")
        return
    # Map probe_name to list of (config, result_path)
    probe_to_results = {name: [] for name in probe_names}
    for path in results_json_paths:
        for probe_name in probe_names:
            if probe_name in os.path.basename(path):
                probe_to_results[probe_name].append(path)
    n_probes = len(probe_names)
    ncols = 3
    nrows = n_probes
    fig, axs = plt.subplots(nrows, ncols, figsize=(6*ncols, 4*nrows), squeeze=False)
    setting_titles = [
        f'Constant {class_names[1]}, Increasing {class_names[0]}',
        f'Constant % {class_names[1]}, Increasing Total',
        f'Constant Total, Increasing % {class_names[1]}'
    ]
    # Group rebuild_configs by setting
    constant_french = []
    constant_percent = []
    constant_total = []
    for rc in rebuild_configs:
        if 'class_counts' in rc:
            constant_french.append(rc)
        elif 'class_percents' in rc:
            if 'total_samples' in rc:
                constant_percent.append(rc)
    percent_groups = {}
    total_groups = {}
    for rc in constant_percent:
        perc_tuple = tuple(sorted(rc['class_percents'].items()))
        total = rc['total_samples']
        percent_groups.setdefault(perc_tuple, []).append(rc)
        total_groups.setdefault(total, []).append(rc)
    constant_percent_final = []
    constant_total_final = []
    for perc_tuple, group in percent_groups.items():
        if len(group) > 1:
            constant_percent_final.extend(group)
    for total, group in total_groups.items():
        if len(group) > 1:
            constant_total_final.extend(group)
    constant_percent_final = list({id(rc): rc for rc in constant_percent_final}.values())
    constant_total_final = list({id(rc): rc for rc in constant_total_final}.values())
    # For each probe, plot the three settings
    for row, probe_name in enumerate(probe_names):
        # Build a mapping from config to result path
        config_to_path = {}
        for path in probe_to_results.get(probe_name, []):
            fname = os.path.basename(path)
            for rc in rebuild_configs:
                match = True
                if 'class_counts' in rc:
                    for cls in rc['class_counts']:
                        if f"class{cls}_{rc['class_counts'][cls]}" not in fname:
                            match = False
                if 'class_percents' in rc:
                    for cls in rc['class_percents']:
                        pct = int(rc['class_percents'][cls]*100)
                        if f"class{cls}_{pct}pct" not in fname:
                            match = False
                    if f"total{rc['total_samples']}" not in fname:
                        match = False
                if 'seed' in rc and f"seed{rc['seed']}" not in fname:
                    match = False
                if match:
                    config_to_path[str(rc)] = path
        # For each setting, collect data
        setting_data = [[], [], []]
        # 0: constant_french, 1: constant_percent, 2: constant_total
        for rc in constant_french:
            path = config_to_path.get(str(rc))
            if not path:
                continue
            with open(path, 'r') as f:
                d = json.load(f)
            val_dict = d.get('metrics', {}).get('all_examples', d.get('metrics', {}))
            n_english = rc['class_counts'][0]
            n_french = rc['class_counts'][1]
            total = n_french + n_english
            pct_french = 100 * n_french / total if total > 0 else 0
            setting_data[0].append((n_english, val_dict, n_french, pct_french))
        for rc in constant_percent_final:
            path = config_to_path.get(str(rc))
            if not path:
                continue
            with open(path, 'r') as f:
                d = json.load(f)
            val_dict = d.get('metrics', {}).get('all_examples', d.get('metrics', {}))
            n_french = int(rc['class_percents'][1] * rc['total_samples']) if 1 in rc['class_percents'] else 0
            pct_french = rc['class_percents'][1] * 100 if 1 in rc['class_percents'] else 0
            total = rc['total_samples']
            setting_data[1].append((total, val_dict, n_french, pct_french))
        for rc in constant_total_final:
            path = config_to_path.get(str(rc))
            if not path:
                continue
            with open(path, 'r') as f:
                d = json.load(f)
            val_dict = d.get('metrics', {}).get('all_examples', d.get('metrics', {}))
            n_french = int(rc['class_percents'][1] * rc['total_samples']) if 1 in rc['class_percents'] else 0
            pct_french = rc['class_percents'][1] * 100 if 1 in rc['class_percents'] else 0
            total = rc['total_samples']
            setting_data[2].append((pct_french, val_dict, n_french, pct_french))
        # Sort each setting by x-axis
        for i in range(3):
            setting_data[i].sort(key=lambda x: x[0])
        # Plot each setting in its own subplot
        for col, (data, title) in enumerate(zip(setting_data, setting_titles)):
            ax = axs[row][col]
            if data:
                x = [d[0] for d in data]
                n_french = [d[2] for d in data]
                pct_french = [d[3] for d in data]
                for metric in metrics:
                    y = [d[1].get(metric, np.nan) for d in data]
                    ax.plot(x, y, marker='o', label=metric)
                xticklabels = [f"{xi}\nN_fr={nf}\n%fr={pf:.1f}" for xi, nf, pf in zip(x, n_french, pct_french)]
                ax.set_xticks(x)
                ax.set_xticklabels(xticklabels, rotation=30, ha='right')
                ax.legend()
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


def plot_probe_score_violins_from_folder(folder, class_names=None, save_path=None, max_probes=9):
    """
    For each *_results.json in the folder, loads the 'scores' and 'labels', and creates a subplot with violin plots of probe scores for each class.
    Args:
        folder: Path to directory containing *_results.json files (from evaluate_probe)
        class_names: dict mapping class index to name (optional)
        save_path: if provided, saves the plot to this path
        max_probes: maximum number of probes to plot (for grid size)
    """

    result_files = sorted(glob.glob(os.path.join(folder, '*_results.json')))
    n_probes = min(len(result_files), max_probes)
    if n_probes == 0:
        print(f"No *_results.json files found in {folder}")
        return
    ncols = min(3, n_probes)
    nrows = int(np.ceil(n_probes / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(6*ncols, 4*nrows), squeeze=False)
    for idx, result_path in enumerate(result_files[:max_probes]):
        row, col = divmod(idx, ncols)
        ax = axs[row][col]
        with open(result_path, 'r') as f:
            d = json.load(f)
        scores = np.array(d['scores']['scores'])
        labels = np.array(d['scores']['labels'])
        if class_names is None:
            unique_classes = np.unique(labels)
            class_names_map = {c: f"Class {c}" for c in unique_classes}
        else:
            class_names_map = class_names
        data = []
        for cls in np.unique(labels):
            mask = (labels == cls)
            for s in scores[mask]:
                data.append({'score': s, 'class': class_names_map.get(cls, str(cls))})
        import pandas as pd
        df = pd.DataFrame(data)
        sns.violinplot(x='class', y='score', data=df, ax=ax, inner='box', cut=0)
        ax.set_title(os.path.basename(result_path).replace('_results.json', ''), fontsize=9)
        ax.set_xlabel('Class')
        ax.set_ylabel('Probe Score')
    # Hide any unused subplots
    for idx in range(n_probes, nrows*ncols):
        row, col = divmod(idx, ncols)
        fig.delaxes(axs[row][col])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved probe score violin subplots to {save_path}")
    else:
        plt.show()


def plot_all_probe_logit_weight_distributions(model, d_model, dataset_name, results_dir, viz_dir, ln_f=None, device="cpu"):
    """
    Plots logit weight distributions for all trained LinearProbes and AttentionProbes in train_{dataset} and dataclass_exps_{dataset}.
    For each probe, plot a single histogram for all logit weights, and overlay colored regions for negative (blue) and positive (red) values. Also save separate figures for all linear and all attention probes.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import glob

    train_dir = os.path.join(results_dir, f"train_{dataset_name}")
    dataclass_dir = os.path.join(results_dir, f"dataclass_exps_{dataset_name}")
    probe_files = glob.glob(os.path.join(train_dir, "*.npz"))
    if os.path.isdir(dataclass_dir):
        probe_files += glob.glob(os.path.join(dataclass_dir, "*.npz"))
    probe_files = [f for f in probe_files if ("linear" in os.path.basename(f) or "attention" in os.path.basename(f))]
    if not probe_files:
        print(f"No LinearProbe or AttentionProbe .npz files found for {dataset_name}.")
        return
    if hasattr(model, "lm_head"):
        W_U = model.lm_head.weight.detach().cpu().numpy()
    elif hasattr(model, "unembed") and hasattr(model.unembed, "W_U"):
        W_U = model.unembed.W_U.detach().cpu().numpy()
    else:
        raise AttributeError("Model does not have an lm_head or unembed.W_U attribute for the unembedding matrix.")
    # Split probes by type
    linear_probes = [f for f in probe_files if "linear" in os.path.basename(f)]
    attention_probes = [f for f in probe_files if "attention" in os.path.basename(f)]
    probe_groups = [("all", probe_files), ("linear", linear_probes), ("attention", attention_probes)]
    color_neg = '#377eb8'  # blue
    color_pos = '#e41a1c'  # red
    for group_name, group_files in probe_groups:
        if not group_files:
            continue
        n_probes = len(group_files)
        ncols = min(3, n_probes)
        nrows = int(np.ceil(n_probes / ncols))
        fig, axs = plt.subplots(nrows, ncols, figsize=(6*ncols, 4*nrows), squeeze=False)
        for idx, probe_path in enumerate(group_files):
            row, col = divmod(idx, ncols)
            ax = axs[row][col]
            if "linear" in os.path.basename(probe_path):
                probe = LinearProbe(d_model=d_model, device=device)
                probe.load_state(probe_path)
                v_probe = probe.model.linear.weight.detach().cpu().numpy()
            elif "attention" in os.path.basename(probe_path):
                probe = AttentionProbe(d_model=d_model, device=device)
                probe.load_state(probe_path)
                v_probe = probe.model.classifier.weight.detach().cpu().numpy()
            else:
                print(f"Skipping unknown probe type: {probe_path}")
                continue
            if v_probe.shape[0] == 1:
                v_proj = v_probe[0]
            else:
                v_proj = v_probe[0]
            if v_proj.shape != (d_model,):
                print(f"Skipping {probe_path}: probe direction shape {v_proj.shape} does not match d_model {d_model}")
                continue
            if ln_f is not None:
                v_proj = ln_f(torch.tensor(v_proj, dtype=torch.float32, device=device)).detach().cpu().numpy()
            logit_weights = W_U.T @ v_proj  # (vocab_size,)
            # Plot a single histogram for all weights
            bins = 100
            counts, bin_edges, patches = ax.hist(logit_weights, bins=bins, alpha=0.7, color='gray', label='all', edgecolor='black')
            # Overlay colored regions for negative and positive
            for i in range(len(bin_edges)-1):
                if bin_edges[i+1] <= 0:
                    patches[i].set_facecolor(color_neg)
                    patches[i].set_alpha(0.5)
                elif bin_edges[i] >= 0:
                    patches[i].set_facecolor(color_pos)
                    patches[i].set_alpha(0.5)
            ax.set_title(os.path.basename(probe_path).replace("_state.npz", ""), fontsize=8)
            ax.set_xlabel("logit weight")
            ax.set_ylabel("count")
            # Add legend manually
            from matplotlib.patches import Patch
            legend_patches = [Patch(facecolor=color_neg, alpha=0.5, label=f"neg (<0, N={(logit_weights < 0).sum()})"),
                              Patch(facecolor=color_pos, alpha=0.5, label=f"pos (>0, N={(logit_weights > 0).sum()})")]
            ax.legend(handles=legend_patches)
            # Boxplots above, split by sign
            box_data = []
            box_labels = []
            box_colors = []
            neg_weights = logit_weights[logit_weights < 0]
            pos_weights = logit_weights[logit_weights > 0]
            if len(neg_weights) > 0:
                box_data.append(neg_weights)
                box_labels.append("neg")
                box_colors.append(color_neg)
            if len(pos_weights) > 0:
                box_data.append(pos_weights)
                box_labels.append("pos")
                box_colors.append(color_pos)
            box_ax = ax.inset_axes([0, 1.05, 1, 0.18])
            bplots = box_ax.boxplot(box_data, vert=False, patch_artist=True, labels=box_labels)
            for patch, color in zip(bplots['boxes'], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.5)
            box_ax.set_xticks([])
            box_ax.set_yticks([])
            box_ax.set_frame_on(False)
        for idx in range(n_probes, nrows*ncols):
            row, col = divmod(idx, ncols)
            fig.delaxes(axs[row][col])
        plt.tight_layout()
        if group_name == "all":
            save_path = os.path.join(viz_dir, f"probe_logit_weight_distributions_{dataset_name}.png")
        else:
            save_path = os.path.join(viz_dir, f"probe_logit_weight_distributions_{dataset_name}_{group_name}.png")
        plt.savefig(save_path, dpi=150)
        print(f"Saved logit weight distributions to {save_path}")
        plt.close()

# if __name__ == '__main__':
#     diff_file = "./results/gender_experiment_gemma/runthrough_4_hist_fig_ismale/logit_diff_gender-pred_model_check.csv"
#     plot_logit_diffs_by_gender(diff_file)