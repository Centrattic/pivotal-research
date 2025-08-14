import importlib
from datasets import Dataset
import os


def process(
    row,
    source_file,
    downstream_handler,
):
    ds = Dataset.from_file(source_file)
    df = ds.to_pandas()
    downstream_mod = importlib.import_module(f'handlers.{downstream_handler}')
    row = row.copy()
    row['__merged_df__'] = True
    return downstream_mod.process(row, df)
