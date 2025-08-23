"""
Cross-Validation Based Hyperparameter Analysis

This module creates plots using the best hyperparameters determined by cross-validation.
It uses val_eval results to select optimal hyperparameters and then plots test_eval results
with those optimal settings. Focuses on experiments 2- (imbalanced) and 4- (balanced).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import seaborn as sns
from scipy import stats
import re
try:
    from .data_loader import (
        get_data_for_visualization,
        get_probe_names,
        get_eval_datasets,
        get_run_names,
        load_metrics_data,
        extract_info_from_filename
    )
    from .viz_util import (
        plot_experiment_best_probes_generic,
        plot_probe_subplots_generic,
        plot_experiment_comparison_generic,
        get_default_hyperparameter_filters,
        filter_data_by_hyperparameters,
        format_run_name_for_display,
        format_dataset_name_for_display,
        get_probe_label
    )
except ImportError:
    from data_loader import (
        get_data_for_visualization,
        get_probe_names,
        get_eval_datasets,
        get_run_names,
        load_metrics_data,
        extract_info_from_filename
    )
    from viz_util import (
        plot_experiment_best_probes_generic,
        plot_probe_subplots_generic,
        get_default_hyperparameter_filters,
        filter_data_by_hyperparameters,
        format_run_name_for_display,
        format_dataset_name_for_display,
        get_probe_label
    )


class CrossValidationAnalyzer:
    """
    Analyzer for cross-validation based hyperparameter selection.
    
    This class provides methods to:
    1. Use val_eval results to determine optimal hyperparameters for experiments 2- and 4-
    2. Generate plots using test_eval results with optimal hyperparameters
    3. Focus on imbalanced vs balanced dataset comparison
    """
    
    def __init__(self):
        """
        Initialize the cross-validation analyzer.
        """
        self.df = self._load_extended_data()
    
    def _load_extended_data(self) -> pd.DataFrame:
        """
        Load data including val_eval results.
        
        Returns:
            DataFrame with extended data including validation results
        """
        # Load the base metrics data
        df = load_metrics_data()
        
        # Extract metadata from filenames
        metadata_list = []
        for filename in df['filename']:
            metadata = extract_info_from_filename(filename)
            metadata_list.append(metadata)
        
        # Add metadata columns
        for key in ['run_name', 'seed', 'experiment', 'eval_dataset', 'train_dataset', 'probe_name']:
            df[key] = [meta.get(key) for meta in metadata_list]
        
        # Add hyperparameter columns
        for key in ['C', 'topk', 'lr', 'weight_decay']:
            df[key] = [meta.get(key) for meta in metadata_list]
        
        # Add additional metadata columns
        for key in ['num_negative_samples', 'num_positive_samples', 'llm_upsampling_ratio', 'qwen_model_size']:
            df[key] = [meta.get(key) for meta in metadata_list]
        
        # Add evaluation type (gen_eval, test_eval, val_eval)
        df['eval_type'] = df['filename'].apply(self._extract_eval_type)
        
        return df.reset_index(drop=True)
    
    def _extract_eval_type(self, filename: str) -> str:
        """Extract evaluation type from filename."""
        if '/val_eval/' in filename:
            return 'val_eval'
        elif '/test_eval/' in filename:
            return 'test_eval'
        elif '/gen_eval/' in filename:
            return 'gen_eval'
        else:
            return 'unknown'
    
    def get_cross_validation_best_hyperparameters(
        self,
        eval_dataset: str,
        experiment: str,
        run_name: str,
        num_positive_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Determine best hyperparameters using cross-validation (val_eval performance).
        Chooses the best probe variant from each category based on val_eval AUC.
        
        Args:
            eval_dataset: Evaluation dataset name
            experiment: Experiment type ('2-' or '4-')
            run_name: Model run name
            num_positive_samples: Filter by number of positive samples
            
        Returns:
            Dictionary with best hyperparameters for each probe category
        """
        # Get val_eval data for cross-validation
        val_df = get_data_for_visualization(
            eval_dataset=eval_dataset,
            experiment=experiment,
            run_name=run_name,
            exclude_attention=False,
            include_val_eval=True
        )
        
        # Filter to val_eval results
        val_df = val_df[val_df['filename'].str.contains('/val_eval/')]
        
        if val_df.empty:
            print(f"No val_eval data found for cross-validation: {eval_dataset}, {run_name}")
            return {}
        
        # Filter by number of positive samples if specified
        if num_positive_samples is not None:
            val_df = val_df[val_df['num_positive_samples'] == num_positive_samples]
        
        if val_df.empty:
            return {}
        
        # Group probes by category and find the best variant
        best_hyperparams = {}
        
        # SAE probes - find best topk
        sae_probes = val_df[val_df['probe_name'].str.contains('sae', na=False)]
        if not sae_probes.empty:
            # Group by topk and calculate mean AUC across seeds
            topk_performance = sae_probes.groupby('topk')['auc'].mean()
            if not topk_performance.empty:
                best_topk = topk_performance.idxmax()
                best_hyperparams['sae'] = {'topk': best_topk}
                print(f"Best SAE topk: {best_topk} (val_eval AUC: {topk_performance[best_topk]:.3f})")
        
        # Linear probes - find best C
        linear_probes = val_df[val_df['probe_name'].str.contains('sklearn_linear', na=False)]
        if not linear_probes.empty:
            # Group by C and calculate mean AUC across seeds
            c_performance = linear_probes.groupby('C')['auc'].mean()
            if not c_performance.empty:
                best_c = c_performance.idxmax()
                best_hyperparams['sklearn_linear'] = {'C': best_c}
                print(f"Best linear C: {best_c} (val_eval AUC: {c_performance[best_c]:.3f})")
        
        # Attention probes - find best lr and weight_decay combination
        attention_probes = val_df[val_df['probe_name'].str.contains('attention', na=False)]
        if not attention_probes.empty:
            # Group by lr and weight_decay and calculate mean AUC across seeds
            lr_wd_performance = attention_probes.groupby(['lr', 'weight_decay'])['auc'].mean()
            if not lr_wd_performance.empty:
                best_lr_wd = lr_wd_performance.idxmax()
                best_hyperparams['attention'] = {
                    'lr': best_lr_wd[0],
                    'weight_decay': best_lr_wd[1]
                }
                print(f"Best attention lr: {best_lr_wd[0]:.1e}, wd: {best_lr_wd[1]:.1e} (val_eval AUC: {lr_wd_performance[best_lr_wd]:.3f})")
        
        return best_hyperparams

    def plot_experiment_with_cv_hyperparameters(
        self,
        eval_dataset: str,
        run_name: str,
        save_path: Path,
        experiment: str,
        metric: str = 'auc',
        num_positive_samples: Optional[int] = None
    ):
        """
        Plot experiment results using best hyperparameters determined by cross-validation.
        
        Args:
            eval_dataset: Evaluation dataset name
            run_name: Model run name
            save_path: Path to save the plot
            experiment: Experiment type ('2-' or '4-')
            metric: Metric to plot ('auc' or 'recall')
            num_positive_samples: Filter by number of positive samples
        """
        # Get best hyperparameters for each probe type using cross-validation
        best_hyperparams = self.get_cross_validation_best_hyperparameters(
            eval_dataset, experiment, run_name, num_positive_samples
        )
        
        if not best_hyperparams:
            print(f"No cross-validation hyperparameters found for {eval_dataset}, {run_name}")
            return
        
        # Generate title suffix indicating cross-validation was used
        title_suffix = ""
        
        # Use the generic plotting function with best hyperparameters
        plot_experiment_best_probes_generic(
            eval_dataset=eval_dataset,
            run_name=run_name,
            save_path=save_path,
            experiment=experiment,
            metric=metric,
            hyperparam_filters=best_hyperparams,
            title_suffix=title_suffix,
            output_dir=Path("visualizations/hyp")
        )

    def plot_subplots_with_cv_hyperparameters(
        self,
        eval_dataset: str,
        run_name: str,
        save_path: Path,
        experiment: str,
        metric: str = 'auc',
        num_positive_samples: Optional[int] = None
    ):
        """
        Plot subplots using best hyperparameters determined by cross-validation.
        
        Args:
            eval_dataset: Evaluation dataset name
            run_name: Model run name
            save_path: Path to save the plot
            experiment: Experiment type ('2-' or '4-')
            metric: Metric to plot ('auc' or 'recall')
            num_positive_samples: Filter by number of positive samples
        """
        # Get best hyperparameters for each probe type using cross-validation
        best_hyperparams = self.get_cross_validation_best_hyperparameters(
            eval_dataset, experiment, run_name, num_positive_samples
        )
        
        if not best_hyperparams:
            print(f"No cross-validation hyperparameters found for {eval_dataset}, {run_name}")
            return
        
        # Use the generic plotting function with best hyperparameters
        plot_probe_subplots_generic(
            experiment=experiment,
            eval_dataset=eval_dataset,
            run_name=run_name,
            save_path=save_path,
            metric=metric,
            hyperparam_filters=best_hyperparams,
            title_suffix="",
            output_dir=Path("visualizations/hyp")
        )

    def plot_experiment_comparison_with_cv_hyperparameters(
        self,
        eval_dataset: str,
        run_name: str,
        save_path: Path,
        metric: str = 'auc',
        num_positive_samples: Optional[int] = None
    ):
        """
        Plot joint comparison of experiment 2- vs 4- using best hyperparameters determined by cross-validation.
        
        Args:
            eval_dataset: Evaluation dataset name
            run_name: Model run name
            save_path: Path to save the plot
            metric: Metric to plot ('auc' or 'recall')
            num_positive_samples: Filter by number of positive samples
        """
        # Get best hyperparameters for both experiments using cross-validation
        best_hyperparams_2 = self.get_cross_validation_best_hyperparameters(
            eval_dataset, '2-', run_name, num_positive_samples
        )
        best_hyperparams_4 = self.get_cross_validation_best_hyperparameters(
            eval_dataset, '4-', run_name, num_positive_samples
        )
        
        if not best_hyperparams_2 and not best_hyperparams_4:
            print(f"No cross-validation hyperparameters found for {eval_dataset}, {run_name}")
            return
        
        # Use the generic plotting function to create joint comparison
        plot_experiment_comparison_generic(
            eval_dataset=eval_dataset,
            run_name=run_name,
            save_path=save_path,
            metric=metric,
            hyperparam_filters_2=best_hyperparams_2,
            hyperparam_filters_4=best_hyperparams_4,
            title_suffix="",
            output_dir=Path("visualizations/hyp")
        )


def generate_cross_validation_plots(skip_existing: bool = False):
    """
    Generate cross-validation based hyperparameter analysis plots.
    
    Args:
        skip_existing: If True, skip generating plots that already exist
    """
    # Create output directory
    output_dir = Path("visualizations/hyp")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize analyzer
    analyzer = CrossValidationAnalyzer()
    
    # Get available configurations
    eval_datasets = get_eval_datasets()
    run_names = get_run_names()
    
    # Filter datasets - exclude gen_eval datasets like 87_is_spam
    def _leading_id(name: str) -> int:
        try:
            return int(str(name).split('_', 1)[0])
        except Exception:
            return -1
    
    eval_datasets = [d for d in eval_datasets if _leading_id(d) == -1 or _leading_id(d) < 99]
    
    # Focus only on experiments 2- (imbalanced) and 4- (balanced)
    experiments = ['2-', '4-']
    
    # Generate cross-validation plots for each configuration
    for run_name in run_names:
        print(f"\nProcessing cross-validation analysis for: {run_name}")
        
        for eval_dataset in eval_datasets:
            for experiment in experiments:
                print(f"  Processing {experiment} for {eval_dataset}")
                
                # Generate best probes plots with CV-optimized hyperparameters
                for metric in ['auc', 'recall']:
                    # Best probes plot
                    save_path = output_dir / f"cv_best_probes_{metric}_{eval_dataset}_{run_name}_{experiment}.png"
                    if not skip_existing or not save_path.exists():
                        analyzer.plot_experiment_with_cv_hyperparameters(
                            eval_dataset, run_name, save_path, experiment, metric
                        )
                    
                    # Subplots
                    subplot_save_path = output_dir / f"cv_subplots_{metric}_{eval_dataset}_{run_name}_{experiment}.png"
                    if not skip_existing or not subplot_save_path.exists():
                        analyzer.plot_subplots_with_cv_hyperparameters(
                            eval_dataset, run_name, subplot_save_path, experiment, metric
                        )
    
    print(f"Cross-validation analysis complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    generate_cross_validation_plots()
