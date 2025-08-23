"""
Paper Statistics Calculator

This script calculates specific statistics for the paper:
1. Imbalanced vs balanced training performance improvements
2. LLM upsampling performance improvements
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

def find_equivalent_samples(df: pd.DataFrame, target_performance: float, metric: str, probe_type: str) -> float:
    """
    Find how many real samples would be equivalent to the target performance.
    
    Args:
        df: DataFrame with imbalanced training data
        target_performance: Target AUC or Recall value to match
        metric: 'auc' or 'recall'
        probe_type: 'linear_mean' or 'top5'
    
    Returns:
        Equivalent number of real samples (interpolated)
    """
    if probe_type == 'linear_mean':
        # Filter for linear-mean probes
        df_filtered = df[df['probe_name'].str.contains('sklearn_linear_mean', na=False)]
    elif probe_type == 'top5':
        # Get top 5 probes by AUC
        best_probes = get_best_probes_by_category(df)
        top5_data = []
        
        for probe in best_probes:
            probe_data = df[df['probe_name'] == probe]
            if not probe_data.empty:
                top5_data.append({
                    'probe': probe,
                    'auc': probe_data['auc'].mean(),
                    'recall': probe_data['recall'].mean()
                })
        
        if not top5_data:
            return None
        
        top5_data.sort(key=lambda x: x['auc'], reverse=True)
        top5_data = top5_data[:5]
        
        # Get the same top 5 probes for each sample count
        top5_probe_names = [p['probe'] for p in top5_data]
        df_filtered = df[df['probe_name'].isin(top5_probe_names)]
    else:
        return None
    
    if df_filtered.empty:
        return None
    
    # Group by sample count and calculate average performance
    sample_performance = df_filtered.groupby('num_positive_samples')[metric].mean().reset_index()
    sample_performance = sample_performance.sort_values('num_positive_samples')
    
    if sample_performance.empty:
        return None
    
    # Find the sample count that gives closest performance
    sample_counts = sample_performance['num_positive_samples'].values
    performances = sample_performance[metric].values
    
    # If target performance is higher than max available, extrapolate
    if target_performance > performances.max():
        # Use the last two points to extrapolate
        if len(performances) >= 2:
            slope = (performances[-1] - performances[-2]) / (sample_counts[-1] - sample_counts[-2])
            if slope > 0:  # Only extrapolate if performance is still improving
                additional_samples = (target_performance - performances[-1]) / slope
                return sample_counts[-1] + additional_samples
        return sample_counts[-1]  # Return max available if can't extrapolate
    
    # If target performance is lower than min available, return min
    if target_performance < performances.min():
        return sample_counts[0]
    
    # Interpolate to find equivalent sample count
    for i in range(len(performances) - 1):
        if performances[i] <= target_performance <= performances[i + 1]:
            # Linear interpolation
            ratio = (target_performance - performances[i]) / (performances[i + 1] - performances[i])
            equivalent_samples = sample_counts[i] + ratio * (sample_counts[i + 1] - sample_counts[i])
            return equivalent_samples
    
    # If we get here, return the closest value
    closest_idx = np.argmin(np.abs(performances - target_performance))
    return sample_counts[closest_idx]

try:
    from .data_loader import (
        get_data_for_visualization,
        get_eval_datasets,
        get_run_names
    )
    from .viz_util import (
        get_best_probes_by_category,
        apply_main_plot_filters
    )
except ImportError:
    from data_loader import (
        get_data_for_visualization,
        get_eval_datasets,
        get_run_names
    )
    from viz_util import (
        get_best_probes_by_category,
        apply_main_plot_filters
    )


def calculate_imbalanced_vs_balanced_performance():
    """
    Calculate performance improvements for imbalanced vs balanced training.
    
    Returns:
        Dictionary with performance improvements for each dataset and probe type
    """
    print("=== IMBALANCED VS BALANCED TRAINING ANALYSIS ===\n")
    
    # Datasets to analyze
    datasets = ['94_better_spam', '87_is_spam', '98_mask_all_honesty']
    models = ['spam_gemma_9b', 'mask_llama33_70b']  # Focus on main models
    sample_ranges = [(1, 10), (1, 20), (10, 20), (25, 50), (50,150)]  # Different sample ranges to analyze
    
    results = {}
    
    for dataset in datasets:
        print(f"Dataset: {dataset}")
        results[dataset] = {}
        
        for model in models:
            print(f"  Model: {model}")
            
            # Get data for both experiments
            try:
                # Imbalanced training (experiment 2-)
                df_imbalanced = get_data_for_visualization(
                    eval_dataset=dataset,
                    experiment='2-',
                    run_name=model,
                    exclude_attention=False,
                    include_val_eval=False
                )
                
                # Balanced training (experiment 4-)
                df_balanced = get_data_for_visualization(
                    eval_dataset=dataset,
                    experiment='4-',
                    run_name=model,
                    exclude_attention=False,
                    include_val_eval=False
                )
                
                if df_imbalanced.empty or df_balanced.empty:
                    print(f"    No data available for {model}")
                    continue
                
                # Apply filters to get default hyperparameters only
                df_imbalanced = apply_main_plot_filters(df_imbalanced)
                df_balanced = apply_main_plot_filters(df_balanced)
                
                results[dataset][model] = {}
                
                # Calculate for each sample range
                for min_samples, max_samples in sample_ranges:
                    print(f"    Sample range: {min_samples}-{max_samples}")
                    
                    # Filter to specific sample range
                    df_imbalanced_range = df_imbalanced[
                        (df_imbalanced['num_positive_samples'] >= min_samples) & 
                        (df_imbalanced['num_positive_samples'] <= max_samples)
                    ]
                    df_balanced_range = df_balanced[
                        (df_balanced['num_positive_samples'] >= min_samples) & 
                        (df_balanced['num_positive_samples'] <= max_samples)
                    ]
                    
                    if df_imbalanced_range.empty or df_balanced_range.empty:
                        print(f"      No data in {min_samples}-{max_samples} sample range for {model}")
                        continue
                    
                    # Calculate performance for linear-mean probes specifically
                    linear_mean_imbalanced = df_imbalanced_range[
                        df_imbalanced_range['probe_name'].str.contains('sklearn_linear_mean', na=False)
                    ]
                    linear_mean_balanced = df_balanced_range[
                        df_balanced_range['probe_name'].str.contains('sklearn_linear_mean', na=False)
                    ]
                    
                    if not linear_mean_imbalanced.empty and not linear_mean_balanced.empty:
                        # Calculate average performance
                        auc_imbalanced = linear_mean_imbalanced['auc'].mean()
                        auc_balanced = linear_mean_balanced['auc'].mean()
                        recall_imbalanced = linear_mean_imbalanced['recall'].mean()
                        recall_balanced = linear_mean_balanced['recall'].mean()
                        
                        # Calculate multiplicative improvements
                        auc_improvement = auc_imbalanced / auc_balanced
                        recall_improvement = recall_imbalanced / recall_balanced
                        
                        # Calculate AUC point improvements
                        auc_point_improvement = auc_imbalanced - auc_balanced
                        recall_point_improvement = recall_imbalanced - recall_balanced
                        
                        print(f"      Linear-mean probe improvements:")
                        print(f"        AUC: {auc_improvement:.3f}x ({auc_point_improvement:+.3f} points)")
                        print(f"        Recall: {recall_improvement:.3f}x ({recall_point_improvement:+.3f} points)")
                        
                        range_key = f"{min_samples}-{max_samples}"
                        if range_key not in results[dataset][model]:
                            results[dataset][model][range_key] = {}
                        results[dataset][model][range_key]['linear_mean'] = {
                            'auc_improvement': auc_improvement,
                            'recall_improvement': recall_improvement,
                            'auc_point_improvement': auc_point_improvement,
                            'recall_point_improvement': recall_point_improvement,
                            'auc_imbalanced': auc_imbalanced,
                            'auc_balanced': auc_balanced,
                            'recall_imbalanced': recall_imbalanced,
                            'recall_balanced': recall_balanced
                        }
                    
                    # Calculate for top 5 probes by AUC
                    best_probes_imbalanced = get_best_probes_by_category(df_imbalanced_range)
                    best_probes_balanced = get_best_probes_by_category(df_balanced_range)
                    
                    # Get top 5 probes by AUC performance
                    top5_imbalanced = []
                    top5_balanced = []
                    
                    for probe in best_probes_imbalanced:
                        probe_data = df_imbalanced_range[df_imbalanced_range['probe_name'] == probe]
                        if not probe_data.empty:
                            top5_imbalanced.append({
                                'probe': probe,
                                'auc': probe_data['auc'].mean(),
                                'recall': probe_data['recall'].mean()
                            })
                    
                    for probe in best_probes_balanced:
                        probe_data = df_balanced_range[df_balanced_range['probe_name'] == probe]
                        if not probe_data.empty:
                            top5_balanced.append({
                                'probe': probe,
                                'auc': probe_data['auc'].mean(),
                                'recall': probe_data['recall'].mean()
                            })
                    
                    # Sort by AUC and take top 5
                    top5_imbalanced.sort(key=lambda x: x['auc'], reverse=True)
                    top5_balanced.sort(key=lambda x: x['auc'], reverse=True)
                    top5_imbalanced = top5_imbalanced[:5]
                    top5_balanced = top5_balanced[:5]
                    
                    if top5_imbalanced and top5_balanced:
                        avg_auc_imbalanced = np.mean([p['auc'] for p in top5_imbalanced])
                        avg_auc_balanced = np.mean([p['auc'] for p in top5_balanced])
                        avg_recall_imbalanced = np.mean([p['recall'] for p in top5_imbalanced])
                        avg_recall_balanced = np.mean([p['recall'] for p in top5_balanced])
                        
                        top5_auc_improvement = avg_auc_imbalanced / avg_auc_balanced
                        top5_recall_improvement = avg_recall_imbalanced / avg_recall_balanced
                        top5_auc_point_improvement = avg_auc_imbalanced - avg_auc_balanced
                        top5_recall_point_improvement = avg_recall_imbalanced - avg_recall_balanced
                        
                        print(f"      Top 5 probes improvements:")
                        print(f"        AUC: {top5_auc_improvement:.3f}x ({top5_auc_point_improvement:+.3f} points)")
                        print(f"        Recall: {top5_recall_improvement:.3f}x ({top5_recall_point_improvement:+.3f} points)")
                        
                        range_key = f"{min_samples}-{max_samples}"
                        if range_key not in results[dataset][model]:
                            results[dataset][model][range_key] = {}
                        results[dataset][model][range_key]['top5'] = {
                            'auc_improvement': top5_auc_improvement,
                            'recall_improvement': top5_recall_improvement,
                            'auc_point_improvement': top5_auc_point_improvement,
                            'recall_point_improvement': top5_recall_point_improvement,
                            'auc_imbalanced': avg_auc_imbalanced,
                            'auc_balanced': avg_auc_balanced,
                            'recall_imbalanced': avg_recall_imbalanced,
                            'recall_balanced': avg_recall_balanced
                        }
                
            except Exception as e:
                print(f"    Error processing {model}: {e}")
        
        print()
    
    return results


def calculate_upsampling_performance():
    """
    Calculate performance improvements for LLM upsampling and find equivalent real sample count.
    
    Returns:
        Dictionary with upsampling performance improvements and equivalent real samples
    """
    print("=== LLM UPSAMPLING ANALYSIS ===\n")
    
    # Focus on experiment 3- (LLM upsampling) for 1-10 real samples
    datasets = ['94_better_spam']  # Focus on main dataset
    models = ['spam_gemma_9b']  # Focus on main model
    
    results = {}
    
    for dataset in datasets:
        print(f"Dataset: {dataset}")
        results[dataset] = {}
        
        for model in models:
            print(f"  Model: {model}")
            
            try:
                # Get LLM upsampling data (experiment 3-)
                df_upsampling = get_data_for_visualization(
                    eval_dataset=dataset,
                    experiment='3-',
                    run_name=model,
                    exclude_attention=False,
                    include_val_eval=False
                )
                
                if df_upsampling.empty:
                    print(f"    No upsampling data available for {model}")
                    continue
                
                # Apply filters to get default hyperparameters
                df_upsampling = apply_main_plot_filters(df_upsampling)
                
                # Filter to 1-10 real samples
                df_upsampling = df_upsampling[
                    (df_upsampling['num_positive_samples'] >= 1) & 
                    (df_upsampling['num_positive_samples'] <= 10)
                ]
                
                if df_upsampling.empty:
                    print(f"    No data in 1-10 sample range for {model}")
                    continue
                
                # Get upsampling ratios
                upsampling_ratios = sorted(df_upsampling['llm_upsampling_ratio'].dropna().unique())
                print(f"    Available upsampling ratios: {upsampling_ratios}")
                
                # Calculate performance for each upsampling ratio
                ratio_performance = {}
                
                for ratio in upsampling_ratios:
                    ratio_data = df_upsampling[df_upsampling['llm_upsampling_ratio'] == ratio]
                    if not ratio_data.empty:
                        # Get top 5 probes by AUC for this ratio
                        best_probes = get_best_probes_by_category(ratio_data)
                        top5_data = []
                        
                        for probe in best_probes:
                            probe_data = ratio_data[ratio_data['probe_name'] == probe]
                            if not probe_data.empty:
                                top5_data.append({
                                    'probe': probe,
                                    'auc': probe_data['auc'].mean(),
                                    'recall': probe_data['recall'].mean()
                                })
                        
                        if top5_data:
                            top5_data.sort(key=lambda x: x['auc'], reverse=True)
                            top5_data = top5_data[:5]
                            
                            avg_auc = np.mean([p['auc'] for p in top5_data])
                            avg_recall = np.mean([p['recall'] for p in top5_data])
                            
                            ratio_performance[ratio] = {
                                'auc': avg_auc,
                                'recall': avg_recall,
                                'top5_probes': [p['probe'] for p in top5_data]
                            }
                
                # Find best ratio
                if ratio_performance:
                    best_ratio = max(ratio_performance.keys(), key=lambda r: ratio_performance[r]['auc'])
                    baseline_ratio = 1  # 1x upsampling as baseline
                    
                    if baseline_ratio in ratio_performance and best_ratio != baseline_ratio:
                        baseline_auc = ratio_performance[baseline_ratio]['auc']
                        baseline_recall = ratio_performance[baseline_ratio]['recall']
                        best_auc = ratio_performance[best_ratio]['auc']
                        best_recall = ratio_performance[best_ratio]['recall']
                        
                        auc_improvement = best_auc / baseline_auc
                        recall_improvement = best_recall / baseline_recall
                        auc_point_improvement = best_auc - baseline_auc
                        recall_point_improvement = best_recall - baseline_recall
                        
                        print(f"    Best upsampling ratio: {best_ratio}x (Top 5 probes)")
                        print(f"    AUC improvement: {auc_improvement:.3f}x ({auc_point_improvement:+.3f} points)")
                        print(f"    Recall improvement: {recall_improvement:.3f}x ({recall_point_improvement:+.3f} points)")
                        
                        # Now calculate factor upgrade for each sample count
                        print(f"    Calculating factor upgrades for each sample count...")
                        
                        # Get imbalanced training data for comparison
                        df_imbalanced = get_data_for_visualization(
                            eval_dataset=dataset,
                            experiment='2-',
                            run_name=model,
                            exclude_attention=False,
                            include_val_eval=False
                        )
                        
                        if not df_imbalanced.empty:
                            df_imbalanced = apply_main_plot_filters(df_imbalanced)
                            
                            # Sample counts to analyze
                            sample_counts = [1, 2, 3, 4, 5, 10]
                            factor_upgrades_auc = []
                            factor_upgrades_recall = []
                            individual_ratios = {}
                            
                            for sample_count in sample_counts:
                                print(f"      Sample count {sample_count}:")
                                
                                # Get best upsampling performance for this sample count
                                sample_upsampling_data = df_upsampling[df_upsampling['num_positive_samples'] == sample_count]
                                if sample_upsampling_data.empty:
                                    print(f"        No upsampling data for {sample_count} samples")
                                    continue
                                
                                # Find best ratio for this sample count
                                best_ratio_for_sample = None
                                best_performance_auc = 0
                                best_performance_recall = 0
                                
                                for ratio in upsampling_ratios:
                                    ratio_data = sample_upsampling_data[sample_upsampling_data['llm_upsampling_ratio'] == ratio]
                                    if not ratio_data.empty:
                                        # Get top 5 probes for this ratio and sample count
                                        best_probes = get_best_probes_by_category(ratio_data)
                                        top5_data = []
                                        
                                        for probe in best_probes:
                                            probe_data = ratio_data[ratio_data['probe_name'] == probe]
                                            if not probe_data.empty:
                                                top5_data.append({
                                                    'probe': probe,
                                                    'auc': probe_data['auc'].mean(),
                                                    'recall': probe_data['recall'].mean()
                                                })
                                        
                                        if top5_data:
                                            top5_data.sort(key=lambda x: x['auc'], reverse=True)
                                            top5_data = top5_data[:5]
                                            
                                            avg_auc = np.mean([p['auc'] for p in top5_data])
                                            avg_recall = np.mean([p['recall'] for p in top5_data])
                                            
                                            if avg_auc > best_performance_auc:
                                                best_performance_auc = avg_auc
                                                best_performance_recall = avg_recall
                                                best_ratio_for_sample = ratio
                                
                                if best_ratio_for_sample is not None:
                                    # First, get the baseline performance (1x upsampling) for this sample count
                                    baseline_data = sample_upsampling_data[sample_upsampling_data['llm_upsampling_ratio'] == 1]
                                    baseline_auc = 0
                                    baseline_recall = 0
                                    
                                    if not baseline_data.empty:
                                        best_probes_baseline = get_best_probes_by_category(baseline_data)
                                        top5_baseline = []
                                        
                                        for probe in best_probes_baseline:
                                            probe_data = baseline_data[baseline_data['probe_name'] == probe]
                                            if not probe_data.empty:
                                                top5_baseline.append({
                                                    'probe': probe,
                                                    'auc': probe_data['auc'].mean(),
                                                    'recall': probe_data['recall'].mean()
                                                })
                                        
                                        if top5_baseline:
                                            top5_baseline.sort(key=lambda x: x['auc'], reverse=True)
                                            top5_baseline = top5_baseline[:5]
                                            baseline_auc = np.mean([p['auc'] for p in top5_baseline])
                                            baseline_recall = np.mean([p['recall'] for p in top5_baseline])
                                    
                                    print(f"        Baseline (1x): AUC={baseline_auc:.3f}, Recall={baseline_recall:.3f}")
                                    print(f"        Best upsampling ({best_ratio_for_sample}x): AUC={best_performance_auc:.3f}, Recall={best_performance_recall:.3f}")
                                    
                                    # Find what sample count in imbalanced training would give the upsampled performance
                                    equivalent_samples_auc = find_equivalent_samples(df_imbalanced, best_performance_auc, 'auc', 'top5')
                                    equivalent_samples_recall = find_equivalent_samples(df_imbalanced, best_performance_recall, 'recall', 'top5')
                                    
                                    if equivalent_samples_auc is not None and equivalent_samples_recall is not None:
                                        # Option 1: Compare to next sample count
                                        next_sample_count = sample_count + 1
                                        next_sample_data = df_imbalanced[df_imbalanced['num_positive_samples'] == next_sample_count]
                                        next_sample_auc = None
                                        next_sample_recall = None
                                        
                                        if not next_sample_data.empty:
                                            best_probes_next = get_best_probes_by_category(next_sample_data)
                                            top5_next = []
                                            for probe in best_probes_next:
                                                probe_data = next_sample_data[next_sample_data['probe_name'] == probe]
                                                if not probe_data.empty:
                                                    top5_next.append({
                                                        'probe': probe,
                                                        'auc': probe_data['auc'].mean(),
                                                        'recall': probe_data['recall'].mean()
                                                    })
                                            if top5_next:
                                                top5_next.sort(key=lambda x: x['auc'], reverse=True)
                                                top5_next = top5_next[:5]
                                                next_sample_auc = np.mean([p['auc'] for p in top5_next])
                                                next_sample_recall = np.mean([p['recall'] for p in top5_next])
                                        
                                        # Option 2: Performance improvement ratio (using Option 1 baseline)
                                        # Calculate how much upsampling improves performance
                                        upsampling_auc_improvement = best_performance_auc - baseline_auc
                                        upsampling_recall_improvement = best_performance_recall - baseline_recall
                                        
                                        # Calculate how much going to the next sample count would improve performance
                                        next_sample_auc_improvement = 0
                                        next_sample_recall_improvement = 0
                                        if next_sample_auc is not None:
                                            next_sample_auc_improvement = next_sample_auc - baseline_auc
                                            next_sample_recall_improvement = next_sample_recall - baseline_recall
                                        
                                        # Calculate improvement ratios
                                        if next_sample_auc_improvement > 0:
                                            auc_improvement_ratio = upsampling_auc_improvement / next_sample_auc_improvement
                                        else:
                                            auc_improvement_ratio = 0
                                            
                                        if next_sample_recall_improvement > 0:
                                            recall_improvement_ratio = upsampling_recall_improvement / next_sample_recall_improvement
                                        else:
                                            recall_improvement_ratio = 0
                                        
                                        # Original method (for comparison)
                                        factor_upgrade_auc = equivalent_samples_auc / sample_count
                                        factor_upgrade_recall = equivalent_samples_recall / sample_count
                                        
                                        # Option 1 method
                                        option1_factor_auc = next_sample_count / sample_count if next_sample_auc is not None else factor_upgrade_auc
                                        option1_factor_recall = next_sample_count / sample_count if next_sample_recall is not None else factor_upgrade_recall
                                        
                                        print(f"        Original method: {factor_upgrade_auc:.2f}x (AUC), {factor_upgrade_recall:.2f}x (Recall)")
                                        print(f"        Option 1 (vs next sample): {option1_factor_auc:.2f}x (AUC), {option1_factor_recall:.2f}x (Recall)")
                                        print(f"        Option 2 (improvement vs next sample): {auc_improvement_ratio:.2f}x (AUC), {recall_improvement_ratio:.2f}x (Recall)")
                                        
                                        if next_sample_auc is not None:
                                            print(f"        Next sample ({next_sample_count}) performance: AUC={next_sample_auc:.3f}, Recall={next_sample_recall:.3f}")
                                            print(f"        Upsampling vs next sample: AUC diff={best_performance_auc-next_sample_auc:+.3f}, Recall diff={best_performance_recall-next_sample_recall:+.3f}")
                                        
                                        # Use Option 1 for the main results (more reasonable)
                                        factor_upgrades_auc.append(option1_factor_auc)
                                        factor_upgrades_recall.append(option1_factor_recall)
                                        
                                        individual_ratios[sample_count] = {
                                            'best_ratio': best_ratio_for_sample,
                                            'original_factor_auc': factor_upgrade_auc,
                                            'original_factor_recall': factor_upgrade_recall,
                                            'option1_factor_auc': option1_factor_auc,
                                            'option1_factor_recall': option1_factor_recall,
                                            'option2_factor_auc': auc_improvement_ratio,
                                            'option2_factor_recall': recall_improvement_ratio,
                                            'equivalent_samples_auc': equivalent_samples_auc,
                                            'equivalent_samples_recall': equivalent_samples_recall,
                                            'best_performance_auc': best_performance_auc,
                                            'best_performance_recall': best_performance_recall,
                                            'baseline_auc': baseline_auc,
                                            'baseline_recall': baseline_recall,
                                            'next_sample_auc': next_sample_auc,
                                            'next_sample_recall': next_sample_recall
                                        }
                                        
                                        print(f"        Factor upgrade - AUC: {factor_upgrade_auc:.2f}x, Recall: {factor_upgrade_recall:.2f}x")
                                        print(f"        Equivalent to {equivalent_samples_auc:.1f} real samples (AUC), {equivalent_samples_recall:.1f} real samples (Recall)")
                            
                            # Calculate average factor upgrades
                            avg_factor_upgrade_auc = np.mean(factor_upgrades_auc) if factor_upgrades_auc else None
                            avg_factor_upgrade_recall = np.mean(factor_upgrades_recall) if factor_upgrades_recall else None
                            
                            print(f"    Average factor upgrades:")
                            print(f"      AUC: {avg_factor_upgrade_auc:.2f}x")
                            print(f"      Recall: {avg_factor_upgrade_recall:.2f}x")
                        
                        results[dataset][model] = {
                            'best_ratio': best_ratio,
                            'auc_improvement': auc_improvement,
                            'recall_improvement': recall_improvement,
                            'auc_point_improvement': auc_point_improvement,
                            'recall_point_improvement': recall_point_improvement,
                            'best_auc': best_auc,
                            'baseline_auc': baseline_auc,
                            'best_recall': best_recall,
                            'baseline_recall': baseline_recall,
                            'avg_factor_upgrade_auc': avg_factor_upgrade_auc if 'avg_factor_upgrade_auc' in locals() else None,
                            'avg_factor_upgrade_recall': avg_factor_upgrade_recall if 'avg_factor_upgrade_recall' in locals() else None,
                            'individual_ratios': individual_ratios if 'individual_ratios' in locals() else {},
                            'all_ratios': ratio_performance
                        }
                
            except Exception as e:
                print(f"    Error processing {model}: {e}")
        
        print()
    
    return results


def main():
    """Main function to calculate all paper statistics."""
    print("PAPER STATISTICS CALCULATOR")
    print("=" * 50)
    
    # Calculate imbalanced vs balanced performance
    imbalanced_results = calculate_imbalanced_vs_balanced_performance()
    
    # Calculate upsampling performance
    upsampling_results = calculate_upsampling_performance()
    
    # Print summary for paper
    print("\n" + "=" * 50)
    print("SUMMARY FOR PAPER")
    print("=" * 50)
    
    print("\nImbalanced vs Balanced Training:")
    for dataset in imbalanced_results:
        print(f"\n{dataset}:")
        for model in imbalanced_results[dataset]:
            if model == 'top5':
                continue
            if model in imbalanced_results[dataset]:
                print(f"  {model}:")
                for range_key in imbalanced_results[dataset][model]:
                    if 'linear_mean' in imbalanced_results[dataset][model][range_key]:
                        linear_data = imbalanced_results[dataset][model][range_key]['linear_mean']
                        print(f"    {range_key} - Linear-mean:")
                        print(f"      AUC: {linear_data['auc_improvement']:.3f}x ({linear_data['auc_point_improvement']:+.3f} points)")
                        print(f"      Recall: {linear_data['recall_improvement']:.3f}x ({linear_data['recall_point_improvement']:+.3f} points)")
                    
                    if 'top5' in imbalanced_results[dataset][model][range_key]:
                        top5_data = imbalanced_results[dataset][model][range_key]['top5']
                        print(f"    {range_key} - Top 5 probes:")
                        print(f"      AUC: {top5_data['auc_improvement']:.3f}x ({top5_data['auc_point_improvement']:+.3f} points)")
                        print(f"      Recall: {top5_data['recall_improvement']:.3f}x ({top5_data['recall_point_improvement']:+.3f} points)")
    
    print("\nLLM Upsampling (Top 5 probes):")
    for dataset in upsampling_results:
        print(f"\n{dataset}:")
        for model in upsampling_results[dataset]:
            data = upsampling_results[dataset][model]
            print(f"  {model}:")
            print(f"    Best ratio: {data['best_ratio']}x")
            print(f"    AUC: {data['auc_improvement']:.3f}x ({data['auc_point_improvement']:+.3f} points)")
            print(f"    Recall: {data['recall_improvement']:.3f}x ({data['recall_point_improvement']:+.3f} points)")
            
            if data.get('avg_factor_upgrade_auc') is not None:
                print(f"    Average factor upgrade (Option 1) - AUC: {data['avg_factor_upgrade_auc']:.2f}x, Recall: {data['avg_factor_upgrade_recall']:.2f}x")
                print(f"    Individual factor upgrades:")
                for sample_count, ratio_data in data['individual_ratios'].items():
                    print(f"      {sample_count} samples:")
                    print(f"        Original method: {ratio_data['original_factor_auc']:.2f}x (AUC), {ratio_data['original_factor_recall']:.2f}x (Recall)")
                    print(f"        Option 1 (vs next sample): {ratio_data['option1_factor_auc']:.2f}x (AUC), {ratio_data['option1_factor_recall']:.2f}x (Recall)")
                    print(f"        Option 2 (improvement vs next sample): {ratio_data['option2_factor_auc']:.2f}x (AUC), {ratio_data['option2_factor_recall']:.2f}x (Recall)")
                    print(f"        Best upsampling ratio: {ratio_data['best_ratio']}x")


if __name__ == "__main__":
    main()
