import csv
import os
import pandas as pd
import numpy as np


def robust_csv_reader(
    filepath,
    delimiter=',',
    quotechar='"',
    encoding='utf-8',
):
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


def process(
    row,
    source_file_or_df,
    max_samples_per_class=4000,
    min_samples_per_class=4000,
    sampling_strategy='random',
):
    """
    Process the data with configurable sampling per class.
    
    Args:
        row: Configuration row containing probe information
        source_file_or_df: Source data file or DataFrame
        max_samples_per_class: Maximum number of samples to take per class (None for all)
        min_samples_per_class: Minimum number of samples required per class (None for no minimum)
        sampling_strategy: Strategy for sampling ('random', 'first', 'last', 'balanced')
    """
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

    out_df = pd.DataFrame({"prompt": from_series, "prompt_len": from_series.str.len(), "target": to_series})

    # Keep one of the complete duplicates.
    out_df = out_df.drop_duplicates(keep='first').reset_index(drop=True)
    # Drop both of the opposite duplicates, unclear which is correct.
    out_df = out_df.drop_duplicates(subset=["prompt"], keep=False).reset_index(drop=True)

    # Filter out top 20% longest samples for emails before class sampling
    out_df = filter_long_samples(out_df)

    # Apply class-based sampling to ensure balanced dataset
    out_df = sample_by_class(out_df, max_samples_per_class, min_samples_per_class, sampling_strategy)

    return out_df


def filter_long_samples(
    df,
    percentile=80,
):
    """
    Filter out the top 20% longest samples (keep bottom 80% by default).
    This is particularly useful for email datasets to remove very long emails.
    
    Args:
        df: DataFrame with 'prompt_len' column
        percentile: Percentile threshold (default 80 means keep bottom 80%)
    
    Returns:
        Filtered DataFrame
    """
    if df.empty:
        return df

    original_count = len(df)

    # Calculate the length threshold at the specified percentile
    length_threshold = df['prompt_len'].quantile(percentile / 100)

    # Filter out samples longer than the threshold
    filtered_df = df[df['prompt_len'] <= length_threshold].copy()

    filtered_count = len(filtered_df)
    removed_count = original_count - filtered_count

    print(
        f"[filter_long_samples] Removed {removed_count} samples ({removed_count/original_count*100:.1f}%) "
        f"with length > {length_threshold:.0f} characters (top {100-percentile}%)"
    )
    print(f"[filter_long_samples] Kept {filtered_count} samples with length <= {length_threshold:.0f} characters")

    # Show length statistics
    print(
        f"[filter_long_samples] Length stats - Min: {filtered_df['prompt_len'].min():.0f}, "
        f"Max: {filtered_df['prompt_len'].max():.0f}, "
        f"Mean: {filtered_df['prompt_len'].mean():.1f}, "
        f"Median: {filtered_df['prompt_len'].median():.1f}"
    )

    return filtered_df.reset_index(drop=True)


def sample_by_class(
    df,
    max_samples_per_class=None,
    min_samples_per_class=None,
    sampling_strategy='random',
):
    """
    Sample data based on target classes with specified constraints.
    This ensures you get the specified number of samples from EACH class.
    
    Args:
        df: DataFrame with 'target' column
        max_samples_per_class: Maximum samples per class (applied to EACH class)
        min_samples_per_class: Minimum samples per class
        sampling_strategy: 'random', 'first', 'last', or 'balanced'
    
    Returns:
        Sampled DataFrame with balanced classes
    """
    if df.empty:
        return df

    # Get unique classes and their counts
    class_counts = df['target'].value_counts()
    print(f"[sample_by_class] Original class distribution: {dict(class_counts)}")

    sampled_dfs = []

    for class_name in class_counts.index:
        class_df = df[df['target'] == class_name].copy()
        class_size = len(class_df)

        print(f"[sample_by_class] Processing class '{class_name}' with {class_size} samples")

        # Check minimum requirement
        if min_samples_per_class is not None and class_size < min_samples_per_class:
            print(
                f"[sample_by_class] Warning: Class '{class_name}' has {class_size} samples, "
                f"but minimum required is {min_samples_per_class}"
            )
            if min_samples_per_class > class_size:
                print(f"[sample_by_class] Skipping class '{class_name}' - insufficient samples")
                continue  # Skip this class if we can't meet minimum

        # Determine how many samples to take from THIS class
        samples_to_take = class_size
        if max_samples_per_class is not None:
            samples_to_take = min(samples_to_take, max_samples_per_class)

        print(f"[sample_by_class] Taking {samples_to_take} samples from class '{class_name}'")

        # Apply sampling strategy
        if sampling_strategy == 'random':
            if samples_to_take < class_size:
                class_df = class_df.sample(n=samples_to_take, random_state=42)
        elif sampling_strategy == 'first':
            class_df = class_df.head(samples_to_take)
        elif sampling_strategy == 'last':
            class_df = class_df.tail(samples_to_take)
        elif sampling_strategy == 'balanced':
            # For balanced strategy, we'll take equal samples from each class
            # This will be handled by max_samples_per_class
            if samples_to_take < class_size:
                class_df = class_df.sample(n=samples_to_take, random_state=42)
        else:
            raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")

        sampled_dfs.append(class_df)

    if not sampled_dfs:
        print("[sample_by_class] Warning: No classes met the sampling criteria")
        return pd.DataFrame(columns=df.columns)

    result_df = pd.concat(sampled_dfs, ignore_index=True)

    # Show final distribution
    final_counts = result_df['target'].value_counts()
    print(f"[sample_by_class] Final class distribution: {dict(final_counts)}")
    print(f"[sample_by_class] Total samples: {len(result_df)}")

    return result_df


# Example usage as a script
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python robust_simple.py <csv_file>")
        sys.exit(1)
    csv_file = sys.argv[1]
    for row in robust_csv_reader(csv_file):
        print(row)
