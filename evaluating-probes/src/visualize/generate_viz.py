#!/usr/bin/env python3
"""
Generate all visualizations from the metrics_index.csv file.
"""

import argparse
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from visualize.plot_generator import generate_all_visualizations
from visualize.hyperparameter_analysis import generate_cross_validation_plots
from visualize.hyp_sweep import generate_all_hyperparameter_sweeps


# The generate_cross_validation_plots function is now imported from hyperparameter_analysis


def main():
    """Main function to generate visualizations."""
    parser = argparse.ArgumentParser(description='Generate visualizations from metrics_index.csv')
    parser.add_argument('--plot-type', choices=['main', 'hyp-sweep', 'cv-hyp'], 
                       default='main', help='Type of plots to generate')
    parser.add_argument('--skip-existing', action='store_true', 
                       help='Skip generating plots that already exist')
    
    args = parser.parse_args()
    
    if args.plot_type == 'main':
        print("Starting main visualization generation...")
        generate_all_visualizations(skip_existing=args.skip_existing)
        print("Main visualization generation complete!")
    
    elif args.plot_type == 'hyp-sweep':
        print("Starting hyperparameter sweep plot generation...")
        generate_all_hyperparameter_sweeps(skip_existing=args.skip_existing)
        print("Hyperparameter sweep plot generation complete!")
    
    elif args.plot_type == 'cv-hyp':
        generate_cross_validation_plots(skip_existing=args.skip_existing)


if __name__ == "__main__":
    main()


