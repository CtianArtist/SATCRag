"""Fetch the SATC dataset from Kaggle into data/raw/ (the corpus itself is not part of this repo).

Needs `pip install kagglehub`. Pinned to dataset version 3; the file's SHA-256 is checked
against src/config.py so a changed upstream file is noticed before anything is parsed.
"""
import hashlib
import shutil

import kagglehub

from src import config

# Download the pinned version (goes to kagglehub's cache folder)
path = kagglehub.dataset_download(config.KAGGLE_DATASET)
print("Downloaded to cache:", path)

# Copy into the project's data/raw folder
dest = config.RAW_CSV.parent
dest.mkdir(parents=True, exist_ok=True)
shutil.copytree(path, dest, dirs_exist_ok=True)

print("Copied files to:", dest.resolve())
for f in sorted(dest.rglob("*")):
    if f.is_file():
        print(" -", f.relative_to(dest))

digest = hashlib.sha256(config.RAW_CSV.read_bytes()).hexdigest()
if digest == config.RAW_CSV_SHA256:
    print("SHA-256 matches the file this project was built from.")
else:
    print(f"WARNING: {config.RAW_CSV.name} has SHA-256 {digest}, expected {config.RAW_CSV_SHA256}. "
          "Source rows, chunk ids and eval targets may not line up with this data.")
