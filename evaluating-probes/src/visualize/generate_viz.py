#!/usr/bin/env python3
"""
Generate all visualizations from the metrics_index.csv file.
"""

import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from visualize.plot_generator import generate_all_visualizations


if __name__ == "__main__":
    print("Starting visualization generation...")
    generate_all_visualizations()
    print("Visualization generation complete!")


