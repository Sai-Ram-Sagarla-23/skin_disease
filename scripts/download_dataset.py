"""
Download the HAM10000 dataset — free, no paid API, no hard-coded credentials.

Order of attempts:
  1. kagglehub (free, open-source Python package). Kaggle requires a free
     account and a `kaggle.json` token to use their API/kagglehub — this is
     NOT a paid key, but it IS an authentication step, so if it's not
     configured this script does not fail silently: it falls through to
     option 2.
  2. Direct download from the Harvard Dataverse, which hosts the official
     HAM10000 release and does NOT require authentication for the public
     files. This is the true zero-credential path.
  3. If both fail (e.g. no internet in a sandboxed environment), print
     clear manual-download instructions and exit gracefully rather than
     crashing the rest of the pipeline.

Usage:
    python scripts/download_dataset.py --output_dir data/raw
"""
import argparse
import os
import shutil
import sys
import zipfile

import requests

DATAVERSE_FILES = {
    # Harvard Dataverse DOI: 10.7910/DVN/DBW86T (Tschandl et al., 2018)
    # These are the persistent, public, no-auth-required file IDs for the
    # HAM10000 dataset release.
    "HAM10000_metadata.csv": "https://dataverse.harvard.edu/api/access/datafile/3172288",
    "HAM10000_images_part_1.zip": "https://dataverse.harvard.edu/api/access/datafile/3172585",
    "HAM10000_images_part_2.zip": "https://dataverse.harvard.edu/api/access/datafile/3172584",
}


def try_kagglehub(output_dir: str) -> bool:
    try:
        import kagglehub
        print("Attempting download via kagglehub (requires a free Kaggle "
              "account token at ~/.kaggle/kaggle.json)...")
        path = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
        print(f"kagglehub download succeeded: {path}")
        for item in os.listdir(path):
            shutil.move(os.path.join(path, item), os.path.join(output_dir, item))
        return True
    except Exception as e:
        print(f"kagglehub download not available/failed ({e}). Falling back...")
        return False


def try_dataverse(output_dir: str) -> bool:
    try:
        print("Attempting direct, no-auth download from Harvard Dataverse...")
        for filename, url in DATAVERSE_FILES.items():
            dest = os.path.join(output_dir, filename)
            print(f"  Downloading {filename} ...")
            resp = requests.get(url, stream=True, timeout=30)
            resp.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            if dest.endswith(".zip"):
                with zipfile.ZipFile(dest, "r") as zf:
                    zf.extractall(output_dir)
                os.remove(dest)
        print("Dataverse download succeeded.")
        return True
    except Exception as e:
        print(f"Dataverse download failed ({e}).")
        return False


def print_manual_instructions(output_dir: str) -> None:
    print(f"""
==================================================================
AUTOMATIC DOWNLOAD FAILED — MANUAL DOWNLOAD REQUIRED (still free)
==================================================================
This is likely because the current environment has no internet access,
or Kaggle/Dataverse rate-limited the request. No payment or subscription
is required for HAM10000 — please use ONE of the following free sources:

Option A — Harvard Dataverse (no account needed):
  https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T
  Download:
    - HAM10000_metadata.csv
    - HAM10000_images_part_1.zip
    - HAM10000_images_part_2.zip
  Extract the two zip files.

Option B — Kaggle (free account required, no payment):
  https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

Place the files so the structure looks like:

  {output_dir}/
      HAM10000_metadata.csv
      HAM10000_images_part_1/
          ISIC_0024306.jpg ...
      HAM10000_images_part_2/
          ISIC_0029306.jpg ...

Once placed, re-run this script (it will detect the existing files) or
proceed directly to scripts/prepare_dataset.py.
==================================================================
""")


def dataset_already_present(output_dir: str) -> bool:
    meta = os.path.join(output_dir, "HAM10000_metadata.csv")
    part1 = os.path.join(output_dir, "HAM10000_images_part_1")
    part2 = os.path.join(output_dir, "HAM10000_images_part_2")
    return os.path.exists(meta) and os.path.isdir(part1) and os.path.isdir(part2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/raw")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if dataset_already_present(args.output_dir):
        print(f"Dataset already present at {args.output_dir} — skipping download.")
        return

    if try_kagglehub(args.output_dir) and dataset_already_present(args.output_dir):
        return
    if try_dataverse(args.output_dir) and dataset_already_present(args.output_dir):
        return

    print_manual_instructions(args.output_dir)
    sys.exit(0)  # graceful exit, not a crash — README explains next steps


if __name__ == "__main__":
    main()
