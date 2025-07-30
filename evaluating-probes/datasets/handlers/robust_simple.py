import csv
import os
import pandas as pd

def robust_csv_reader(filepath, delimiter=',', quotechar='"', encoding='utf-8'):
    """
    Generator that yields rows from a possibly broken CSV file.
    Tries to recover from unclosed quotes and embedded newlines.
    Skips broken/incomplete rows and logs them.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    with open(filepath, 'r', encoding=encoding, errors='replace') as f:
        row = ''
        in_quote = False
        line_num = 0
        for line in f:
            line_num += 1
            quote_count = line.count(quotechar)
            # If not in a quoted field, start a new row
            if not in_quote:
                row = line
            else:
                row += line
            # Toggle in_quote status for each quotechar found
            if quote_count % 2 != 0:
                in_quote = not in_quote
            # If not in a quoted field, yield the row
            if not in_quote:
                try:
                    for parsed_row in csv.reader([row], delimiter=delimiter, quotechar=quotechar):
                        yield parsed_row
                except Exception as e:
                    print(f"[robust_csv_reader] Skipping broken row at line {line_num}: {row[:100]}... Error: {e}")
                row = ''
        # If file ends while still in a quote, skip the last row
        if in_quote:
            print(f"[robust_csv_reader] Skipping incomplete row at EOF: {row[:100]}...")

def process(row, source_file_or_df):
    # Accept either a DataFrame or a file path, DataFrame from other handlers calling it
    if isinstance(source_file_or_df, pd.DataFrame):
        df = source_file_or_df
    else:
        _, ext = os.path.splitext(source_file_or_df)
        ext = ext.lower()
        if ext == ".csv":
            # Use robust_csv_reader to read the file
            rows = list(robust_csv_reader(source_file_or_df))
            # Try to get header from the first row
            if not rows:
                raise ValueError(f"No rows found in {source_file_or_df}")
            header = rows[0]
            data = rows[1:]
            df = pd.DataFrame(data, columns=header)
        elif ext == ".tsv":
            df = pd.read_csv(source_file_or_df, sep='\t', header=None)
        elif ext == ".parquet":
            df = pd.read_parquet(source_file_or_df)
        elif ext == ".json" or ext == '.jsonl':
            df = pd.read_json(source_file_or_df, lines=True)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    df.columns = [str(c) for c in df.columns]

    probe_from = list(str(row['Probe from']).split(',')) if pd.notnull(row['Probe from']) else []
    probe_to = list(str(row['Probe to']).split(',')) if pd.notnull(row['Probe to']) else []
    probe_from_extract = row['probe from extraction']
    probe_to_extract = row['probe to extraction']

    for i in probe_from:
        probe_from_extract = probe_from_extract.replace(f"{i}", f"df['{i}'].astype(str)")

    for i in probe_to:
        probe_to_extract = probe_to_extract.replace(f"{i}", f"df['{i}'].astype(str)")

    from_series = eval(probe_from_extract)
    to_series = eval(probe_to_extract)

    out_df = pd.DataFrame({
        "prompt": from_series,
        "prompt_len": from_series.str.len(),
        "target": to_series
    })

    # Keep one of the complete duplicates.
    out_df = out_df.drop_duplicates(keep='first').reset_index(drop=True)
    # Drop both of the opposite duplicates, unclear which is correct.
    out_df = out_df.drop_duplicates(subset=["prompt"], keep=False).reset_index(drop=True)
    return out_df

# Example usage as a script
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python robust_simple.py <csv_file>")
        sys.exit(1)
    csv_file = sys.argv[1]
    for row in robust_csv_reader(csv_file):
        print(row)