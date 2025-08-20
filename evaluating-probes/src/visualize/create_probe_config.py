#!/usr/bin/env python3
"""
Create the probe configuration file for detailed labeling.
"""

from plot_generator import create_probe_config_file

if __name__ == "__main__":
    create_probe_config_file()
    print("Probe configuration file created successfully!")
    print("You can now edit src/visualize/probe_config.json to customize labels and configurations.")
