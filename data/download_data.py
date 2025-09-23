import gdown
import os

# Where to save the file locally
output_path = "data/hdb_resale_full.csv"

# File ID from Google Drive link
file_id = "1c1khwawDT-r3fa4mm5RvbSkXyCyyMShG"
url = f"https://drive.google.com/uc?id={file_id}"

# Only download if file not already present
if not os.path.exists(output_path):
    print("Downloading HDB resale data...")
    gdown.download(url, output_path, quiet=False)
else:
    print("File already exists locally. Skipping download.")
