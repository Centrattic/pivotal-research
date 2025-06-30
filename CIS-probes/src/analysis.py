import os
import numpy as np
import pandas as pd
from embeddings import GPT2CountryEmbeddings
from data_utils import load_countries_csv
from projection_methods import project_column, encode_labels
import argparse
import warnings

warnings.filterwarnings('ignore')

# abspath = os.path.abspath(__file__)
# dname = os.path.dirname(abspath)
# print(dname)
# os.chdir(dname)

COUNTRIES = [
    'France', 'Germany', 'China', 'Australia', 'Brazil', 'Russia', 'Ukraine', 'United States', 'Europe', 'Japan',
    'Belgium', 'Denmark', 'Greece', 'Iraq', 'Israel', 'Turkey', 'Cuba', 'Libya', 'Panama', 'Qatar', 'Bosnia', 'Kenya',
    'Korea', 'Liberia', 'Macedonia', 'Nicaragua', 'Switzerland', 'Cambodia', 'Finland', 'Iceland', 'Norway', 'Poland',
    'Sweden', 'Taiwan', 'Venezuela', 'Vietnam', 'Bolivia', 'Chile', 'Ghana', 'India', 'Jordan', 'Mexico', 'Nepal',
    'Nigeria', 'Pakistan', 'Serbia', 'Slovenia', 'Sudan', 'Uganda', 'Yemen', 'Zambia'
]

def normalize(name):
    return name.strip().lower()

def make_country_mapping(countries_list, csv_countries):
    mapping = {}
    csv_norm_to_orig = {normalize(name): name for name in csv_countries}
    for c in countries_list:
        norm = normalize(c)
        if norm in csv_norm_to_orig:
            mapping[c] = csv_norm_to_orig[norm]
        else:
            mapping[c] = None
    return mapping

def save_results(df, y_true, y_pred, method, col_name):
    os.makedirs('results', exist_ok=True)
    out_df = pd.DataFrame({
        'Country': df['Country'].values,
        'GroundTruth': y_true,
        'Prediction': y_pred
    })
    out_df.to_csv(f'results/{col_name}-{method}.csv', index=False)

def main(column: str, task_type: str = 'auto', data_path: str = 'data/countries.csv'):
    # Load data
    df = load_countries_csv(data_path)
    print("Unique country names in CSV:", df['Country'].unique())
    # Normalize country names
    df['Country_norm'] = df['Country'].apply(normalize)
    COUNTRIES_NORM = [normalize(c) for c in COUNTRIES]
    # Filter to countries in our list (normalized)
    df = df[df['Country_norm'].isin(COUNTRIES_NORM)]
    print("Countries after normalization and filtering:", df['Country'].tolist())
    # Print mapping from COUNTRIES to CSV names
    mapping = make_country_mapping(COUNTRIES, df['Country'].tolist())
    print("Country mapping (COUNTRIES list -> CSV):")
    for k, v in mapping.items():
        if v is None:
            print(f"{k} -> NO MATCH")
        elif normalize(k) != normalize(v):
            print(f"{k} -> {v}")
    if len(df) == 0:
        print("No matching countries found after normalization. Check country names in CSV and COUNTRIES list.")
        return
    # Get embeddings
    embedder = GPT2CountryEmbeddings()
    embeddings = embedder.get_embeddings(df['Country'].tolist())
    X = np.stack([embeddings[c] for c in df['Country']])
    y = df[column].values
    print("Embeddings and labels created")
    # Encode labels if needed
    if y.dtype == object or y.dtype.kind in {'O', 'U', 'S'}:
        y = encode_labels(y)
    # Map short task type to full
    if task_type == 'c':
        task_type_full = 'classification'
    elif task_type == 'r':
        task_type_full = 'regression'
    else:
        task_type_full = 'auto'
    results = project_column(X, y, task_type=task_type_full)
    print(f'Projection results for column: {column}')
    for method, result in results.items():
        print(f'{method}: {result["score"]:.4f}')
        save_results(df, y, result['preds'], method.replace('mlp_classifier','-mlp').replace('mlp_regressor','-mlp').replace('linear_classifier','linear').replace('linear_regression','linear'), column)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Probe country info from GPT-2 embeddings.")
    parser.add_argument('--col', type=str, required=True, help='Column name to probe')
    parser.add_argument('--type', type=str, default='auto', choices=['c', 'r', 'auto'], help='Task type: c=classification, r=regression, auto=auto-detect')
    args = parser.parse_args()
    main(args.col, args.type) 