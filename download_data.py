#!/usr/bin/env python3
"""
One-shot data downloader — no huggingface-cli needed.
Run from inside datamind-survey/:

    python download_data.py

Downloads:
  - QRData      → data/QRData/
  - DiscoveryBench → data/DiscoveryBench/
"""

import os
import subprocess
import sys
import zipfile
from pathlib import Path

# ── 1. Install huggingface_hub if missing ─────────────────────────────────────
try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("Installing huggingface_hub ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
    from huggingface_hub import snapshot_download

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# ── 2. QRData (GitHub clone) ──────────────────────────────────────────────────
qrdata_dir = "data/QRData"
qrdata_meta = Path(qrdata_dir) / "benchmark" / "QRData.json"
if not qrdata_meta.exists():
    print("\n[1/2] Cloning QRData from GitHub ...")
    # Remove leftover empty/partial directory from a previous failed clone
    if Path(qrdata_dir).exists():
        import shutil
        shutil.rmtree(qrdata_dir)
    result = subprocess.run(
        ["git", "clone", "https://github.com/xxxiaol/QRData", qrdata_dir],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("git clone failed:", result.stderr)
        sys.exit(1)
    print(f"    QRData saved to {qrdata_dir}/")
else:
    print(f"[1/2] QRData already exists at {qrdata_dir}/ — skipping.")

qrdata_zip = Path(qrdata_dir) / "benchmark" / "data.zip"
qrdata_tables = Path(qrdata_dir) / "benchmark" / "data"
if qrdata_zip.exists() and not qrdata_tables.exists():
    print("    Extracting QRData CSV tables ...")
    with zipfile.ZipFile(qrdata_zip) as zf:
        zf.extractall(qrdata_zip.parent)

# ── 3. DiscoveryBench (HuggingFace) ──────────────────────────────────────────
db_dir = "data/DiscoveryBench"
db_answer_key = Path(db_dir) / "answer_key" / "answer_key_real.csv"
if not db_answer_key.exists():
    print("\n[2/2] Downloading DiscoveryBench from HuggingFace ...")
    snapshot_download(
        repo_id="allenai/discoverybench",
        repo_type="dataset",
        local_dir=db_dir,
        ignore_patterns=["*.parquet"],   # skip large parquet files, we only need CSVs/JSONs
    )
    print(f"    DiscoveryBench saved to {db_dir}/")
else:
    print(f"[2/2] DiscoveryBench already exists at {db_dir}/ — skipping.")

print("\n✓ Done. Data layout:")
for root, dirs, files in os.walk("data"):
    dirs[:] = sorted(dirs)
    depth = root.count(os.sep) - "data".count(os.sep)
    if depth > 2:
        continue
    indent = "  " * depth
    print(f"{indent}{os.path.basename(root)}/")
    if depth == 2:
        fcount = len(files)
        if fcount:
            print(f"{indent}  ({fcount} files)")
