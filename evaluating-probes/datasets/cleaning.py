import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import importlib

MAIN_CSV = 'main.csv'

def run_extraction(source_folder, target_folder, main_csv=MAIN_CSV):
    os.makedirs(target_folder, exist_ok=True)
    index = pd.read_csv(main_csv)

    for _, row in tqdm(index.iterrows()):
        handler_name = row.get('handler', 'simple').strip() # defualt to simple

        try:
            # NEW: Intercept bracket-style meta-handlers
            if handler_name.startswith('hf_handler[') and handler_name.endswith(']'):
                downstream_handler = handler_name[len('hf_handler['):-1]
                handler_mod = importlib.import_module(f'handlers.hf_handler')
                handler_to_pass = downstream_handler
            else:
                handler_mod = importlib.import_module(f'handlers.{handler_name}')
                handler_to_pass = None

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
            if handler_to_pass is not None:
                out_df = handler_mod.process(row, source_file, handler_to_pass)
            elif handler_name == "phys_reasoning_balancer":
                out_df = handler_mod.process(row, source_folder)
            elif handler_name == "science_commonsense_merge":
                out_df = handler_mod.process(row, source_folder)
            elif handler_name == "llama_combined_handler":
                out_df = handler_mod.process(row, source_folder)
            else:
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
