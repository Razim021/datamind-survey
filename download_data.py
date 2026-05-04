#!/usr/bin/env python3
"""
One-shot data downloader -- no huggingface-cli needed.
Run from inside datamind-survey/:

    python download_data.py --dataset qrdata

Default behavior downloads QRData only, because the main Colab workflow is
designed to finish within a 3-hour A100 session. DiscoveryBench remains
available as an optional appendix dataset.
"""

import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path

DATA_DIR = Path("data")


def download_qrdata() -> None:
    """Clone QRData and extract the CSV archive when needed."""
    qrdata_dir = DATA_DIR / "QRData"
    qrdata_meta = qrdata_dir / "benchmark" / "QRData.json"
    if not qrdata_meta.exists():
        print("\n[QRData] Cloning from GitHub ...")
        if qrdata_dir.exists():
            import shutil

            shutil.rmtree(qrdata_dir)
        result = subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/xxxiaol/QRData", str(qrdata_dir)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print("git clone failed:", result.stderr)
            sys.exit(1)
        print(f"    QRData saved to {qrdata_dir}/")
    else:
        print(f"[QRData] Already exists at {qrdata_dir}/ -- skipping.")

    qrdata_zip = qrdata_dir / "benchmark" / "data.zip"
    qrdata_tables = qrdata_dir / "benchmark" / "data"
    if qrdata_zip.exists() and not qrdata_tables.exists():
        print("    Extracting QRData CSV tables ...")
        with zipfile.ZipFile(qrdata_zip) as zf:
            zf.extractall(qrdata_zip.parent)


def download_discoverybench() -> None:
    """Download DiscoveryBench for optional appendix experiments."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Installing huggingface_hub ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        from huggingface_hub import snapshot_download

    db_dir = DATA_DIR / "DiscoveryBench"
    db_answer_key = db_dir / "answer_key" / "answer_key_real.csv"
    if not db_answer_key.exists():
        print("\n[DiscoveryBench] Downloading from HuggingFace ...")
        snapshot_download(
            repo_id="allenai/discoverybench",
            repo_type="dataset",
            local_dir=str(db_dir),
            ignore_patterns=["*.parquet"],
        )
        print(f"    DiscoveryBench saved to {db_dir}/")
    else:
        print(f"[DiscoveryBench] Already exists at {db_dir}/ -- skipping.")


def print_layout() -> None:
    print("\nDone. Data layout:")
    for root, dirs, files in os.walk(DATA_DIR):
        dirs[:] = sorted(dirs)
        depth = Path(root).relative_to(DATA_DIR).parts
        if len(depth) > 2:
            continue
        indent = "  " * len(depth)
        print(f"{indent}{os.path.basename(root)}/")
        if len(depth) == 2 and files:
            print(f"{indent}  ({len(files)} files)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download benchmark data for the DataMind survey.")
    parser.add_argument(
        "--dataset",
        choices=["qrdata", "discoverybench", "all"],
        default="qrdata",
        help="Dataset to download. Default is qrdata for the 3-hour Colab workflow.",
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)
    if args.dataset in {"qrdata", "all"}:
        download_qrdata()
    if args.dataset in {"discoverybench", "all"}:
        download_discoverybench()
    print_layout()


if __name__ == "__main__":
    main()
