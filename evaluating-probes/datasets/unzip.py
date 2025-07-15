import zipfile
import os

with zipfile.ZipFile('./original/eng-french.zip', 'r') as zip_ref:
    zip_ref.extractall('./original/eng-french')
