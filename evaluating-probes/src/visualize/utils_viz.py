"""
Thin shim that re-exports compact plotting utilities after refactor.
New code should import from `viz_core` and `exp_plots` directly.
"""

from .exp_plots import (
    plot_experiment_unified,
    plot_probe_group_comparison,
    plot_scaling_law_across_runs,
    get_best_default_probes_by_type,
    plot_scaling_law_all_probes_aggregated,
    plot_llm_upsampling_per_probe,
)

from .viz_core import (
    default_probe_patterns,
    collect_eval_result_files_for_pattern,
    find_experiment_folders,
    find_inner_results_folder,
    collect_result_files_for_pattern,
    _get_scores_and_labels_from_result_file,
    is_default_probe_file,
    parse_llm_upsampling_from_filename,
)

__all__ = [
    'plot_experiment_unified',
    'plot_probe_group_comparison',
    'plot_scaling_law_across_runs',
    'plot_scaling_law_all_probes_aggregated',
    'get_best_default_probes_by_type',
    'default_probe_patterns',
    'collect_eval_result_files_for_pattern',
    'plot_llm_upsampling_per_probe',
    'find_experiment_folders',
    'find_inner_results_folder',
    'collect_result_files_for_pattern',
    '_get_scores_and_labels_from_result_file',
    'is_default_probe_file',
    'parse_llm_upsampling_from_filename',
]


