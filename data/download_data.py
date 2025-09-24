import gdown
import os
import duckdb

# Where to save the file locally
csv_path = "data/hdb_resale_full.csv"
duckdb_path = "data/hdb_resale_full.duckdb"

# File ID from Google Drive link
file_id = "1c1khwawDT-r3fa4mm5RvbSkXyCyyMShG"
url = f"https://drive.google.com/uc?id={file_id}"

# Only download if file not already present
if not os.path.exists(csv_path):
    print("Downloading HDB resale data...")
    gdown.download(url, csv_path, quiet=False)
else:
    print("File already exists locally. Skipping download.")

# Create the DuckDB database if it doesn't exist
if not os.path.exists(duckdb_path):
    print("Creating DuckDB database from csv...")
    con = duckdb.connect(duckdb_path)
    con.execute(f"""
        CREATE TABLE resale AS
        SELECT * FROM read_csv_auto('{csv_path}', header=True)
    """)
    con.close()
else:
    print("DuckDB file already exists. Skipping.")
