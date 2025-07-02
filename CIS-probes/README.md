# CIS-probes: Probing Country Information in GPT-2 Embeddings

## Overview
This mini-experiment explores what information about countries (e.g., language, capital, region, etc.) can be projected out of GPT-2 embeddings for country names. It uses a list of single-token country names and a CSV file of country properties.

## Structure
- `src/embeddings.py`: Extracts GPT-2 embeddings for country names.
- `src/data_utils.py`: Loads and preprocesses country data from CSV.
- `src/projection_methods.py`: Implements linear and MLP classifiers/regressors for probing.
- `src/analysis.py`: Runs the full pipeline for a given property/column.
- `data/countries.csv`: Country data (move your CSV here).

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Ensure `countries.csv` is in the `data/` directory.

## Usage
To run a probe for a specific column (e.g., `Region`):
```bash
python src/analysis.py Region
```

## Notes
- Only single-token country names are used (see list in `analysis.py`).
- No fine-tuning is performed; only probing via classifiers/regressors. 