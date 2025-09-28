import gdown
import os
import duckdb

# Where to save the file locally
CSV_PATH = "data/hdb_df_geocoded_condensed.csv"
DUCKDB_PATH = "data/hdb_df_geocoded_condensed.duckdb"

# File ID from Google Drive link
FILE_ID = "1pe2CsZCELJX6yE5uh2chLaryzwfGjWjx"
URL = f"https://drive.google.com/uc?id={FILE_ID}"

# Only download if file not already present
if not os.path.exists(CSV_PATH):
    print("Downloading HDB resale data...")
    gdown.download(URL, CSV_PATH, quiet=False)
else:
    print("File already exists locally. Skipping download.")

# Create the DuckDB database if it doesn't exist
if not os.path.exists(DUCKDB_PATH):
    print("Creating DuckDB database from csv...")
    con = duckdb.connect(DUCKDB_PATH)
    con.execute(f"""
        CREATE TABLE resale AS
        SELECT * FROM read_csv_auto('{CSV_PATH}', header=True)
    """)
    con.close()
else:
    print("DuckDB file already exists. Skipping.")
