import zipfile
import os

with zipfile.ZipFile('results/resolts.zip', 'r') as zip_ref:
    zip_ref.extractall('results/french_probing')
