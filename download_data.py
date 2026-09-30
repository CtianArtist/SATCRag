import shutil
from pathlib import Path
import kagglehub

# Download latest version (goes to kagglehub's cache folder)
path = kagglehub.dataset_download("snapcrack/every-sex-and-the-city-script")
print("Downloaded to cache:", path)

# Copy into the project's data/raw folder
dest = Path("data/raw")
dest.mkdir(parents=True, exist_ok=True)
shutil.copytree(path, dest, dirs_exist_ok=True)

print("Copied files to:", dest.resolve())
for f in sorted(dest.rglob("*")):
    if f.is_file():
        print(" -", f.relative_to(dest))