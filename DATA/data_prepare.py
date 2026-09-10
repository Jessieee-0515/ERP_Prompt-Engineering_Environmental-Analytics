import os
import gzip
import shutil
import requests
import pandas as pd
import numpy as np

# Configuration
STATION = "45004"
START_YEAR = 1990
END_YEAR = 1999

DOWNLOAD_DIR = "download"
OUTPUT_DIR = "output"

os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE_URL = "https://www.ndbc.noaa.gov/data/historical/stdmet/"

# Download

def download_year(year):

    filename = f"{STATION}h{year}.txt.gz"

    url = BASE_URL + filename

    gz_path = os.path.join(DOWNLOAD_DIR, filename)

    if os.path.exists(gz_path):
        print(f"{year} already downloaded.")
        return gz_path

    print(f"Downloading {year}...")

    r = requests.get(url, timeout=60)

    if r.status_code != 200:
        print(f"{year} download failed.")
        return None

    with open(gz_path, "wb") as f:
        f.write(r.content)

    return gz_path

# Unzip

def unzip_file(gz_path):

    txt_path = gz_path[:-3]

    if os.path.exists(txt_path):
        return txt_path

    with gzip.open(gz_path, "rb") as f_in:
        with open(txt_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    return txt_path

# Read NOAA File

def read_file(txt_path):

    df = pd.read_csv(
        txt_path,
        sep=r"\s+",
        engine="python"
    )

    #years
    if "YY" in df.columns:

        df["YEAR"] = np.where(
            df["YY"] < 70,
            df["YY"] + 2000,
            df["YY"] + 1900
        )

    elif "#YY" in df.columns:

        df["YEAR"] = np.where(
            df["#YY"] < 70,
            df["#YY"] + 2000,
            df["#YY"] + 1900
        )

    elif "YYYY" in df.columns:

        df["YEAR"] = df["YYYY"]

    else:
        raise ValueError("Cannot find year column.")

    #minutes
    if "mm" in df.columns:

        minute = df["mm"]

    else:

        minute = 0

    #Datetime

    df["Datetime"] = pd.to_datetime(
        dict(
            year=df["YEAR"],
            month=df["MM"],
            day=df["DD"],
            hour=df["hh"],
            minute=minute
        ),
        errors="coerce"
    )

    # Missing Values

    missing_values = [
        99,
        99.0,
        999,
        999.0,
        9999,
        9999.0,
        99999,
        99999.0
    ]

    df.replace(missing_values, np.nan, inplace=True)

    return df


# Main

all_df = []

for year in range(START_YEAR, END_YEAR + 1):

    print("=" * 60)
    print(year)

    try:

        gz = download_year(year)

        if gz is None:
            continue

        txt = unzip_file(gz)

        df = read_file(txt)

        df["SourceYear"] = year

        all_df.append(df)

        print(f"Rows: {len(df)}")

    except Exception as e:

        print(f"Error in {year}: {e}")

        continue


# Merge

if len(all_df) == 0:

    raise RuntimeError("No data downloaded.")

df = pd.concat(all_df, ignore_index=True)
df = df.sort_values("Datetime")
cols = ["Datetime"] + [c for c in df.columns if c != "Datetime"]
df = df[cols]

# Save

outfile = os.path.join(
    OUTPUT_DIR,
    f"{STATION}_{START_YEAR}_{END_YEAR}.csv"
)

df.to_csv(outfile, index=False)

print("\n")
print("=" * 60)
print("Finished")
print("Rows :", len(df))
print("Columns :", len(df.columns))
print("Saved :", outfile)
print("=" * 60)

print("\nFirst five rows:\n")
print(df.head())

print("\nMissing values:\n")
print(df.isna().sum())