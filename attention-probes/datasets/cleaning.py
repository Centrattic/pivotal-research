import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import importlib

MAIN_CSV = 'main.csv'

def run_extraction(source_folder, target_folder, main_csv=MAIN_CSV):
    os.makedirs(target_folder, exist_ok=True)
    index = pd.read_csv(main_csv)

    for _, row in tqdm(index.iterrows()): # for each dataset
        handler_name = row.get('handler', 'simple')
        handler_name = handler_name.strip()
        # Dynamically import the handler function
        try:
            handler_mod = importlib.import_module(f'handlers.{handler_name}')
        except Exception as e:
            print(f"Could not import handler '{handler_name}': {e}")
            continue
        if not hasattr(handler_mod, 'process'):
            print(f"Handler '{handler_name}' does not have a 'process' function!")
            continue

        source_file = os.path.join(source_folder, str(row['source']))
        out_file = os.path.join(target_folder, f"{row['number']}_{row['save_name']}")

        if os.path.exists(out_file):
            print(f"File already exists: {out_file}")
            continue

        try:
            # handler is responsible for reading the file
            out_df = handler_mod.process(row, source_file)
            out_df.to_csv(out_file, index=False)
            print(f"Saved: {out_file}")
        except Exception as e:
            print(f"Error processing {source_file} with handler '{handler_name}': {e}")
            continue

if __name__ == "__main__":
    source_folder = "./original"
    target_folder = "./cleaned"
    run_extraction(source_folder, target_folder)
