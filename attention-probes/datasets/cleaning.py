import os
import pandas as pd

# Path to your cleaning CSV
MAIN_CSV = 'main.csv'

def run_extraction(source_folder, target_folder, main_csv=MAIN_CSV):
    os.makedirs(target_folder, exist_ok=True)
    index = pd.read_csv(main_csv)

    for _, row in index.iterrows():
        source_file = os.path.join(source_folder, row['source'])
        out_file = os.path.join(target_folder, f"{row['number']}_{row['save_name']}")

        try:
            df = pd.read_csv(source_file)
        except Exception as e:
            print(f"Error loading {source_file}: {e}")
            continue

        # Extract probe from
        probe_from = row['Probe from']
        probe_to = row['Probe to']
        probe_from_extract = row['probe from extraction']
        probe_to_extract = row['probe to extraction']

        from_series = eval(probe_from_extract.replace("col", f"df[probe_from]"))
        to_series = eval(probe_to_extract.replace("col", f"df[probe_to]"))

        # Combine into new DataFrame and save
        out_df = pd.DataFrame({probe_from: from_series, probe_to: to_series})
        out_df.to_csv(out_file, index=False)
        print(f"Saved: {out_file}")

if __name__ == "__main__":
    source_folder = "./original"
    target_folder = "./cleaned"
    run_extraction(source_folder, target_folder)
