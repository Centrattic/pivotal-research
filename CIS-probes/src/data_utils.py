import pandas as pd
from typing import List

def load_countries_csv(path: str = 'data/countries.csv') -> pd.DataFrame:
    return pd.read_csv(path)

def get_columns(df: pd.DataFrame) -> List[str]:
    return list(df.columns) 