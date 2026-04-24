"""
Dataset loaders for QRData and DiscoveryBench.
Matches the exact file structure expected by DataMind's data_process.py.

Directory layout (set via DATA_DIR env var or --data_dir flag):
  data/
  ├── QRData/
  │   ├── QRData.json          ← question + answer metadata
  │   └── data/                ← CSV files referenced by each sample
  └── DiscoveryBench/
      ├── answer_key_real.csv  ← ground-truth answers
      ├── {domain}/            ← one folder per domain (sociology, etc.)
      │   └── {dataset}/       ← per-dataset folder with CSV + metadata.json
      └── ...
"""

import json
import os
import zipfile
from pathlib import Path

import pandas as pd


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def load_qrdata(data_dir: str) -> list[dict]:
    """
    Load QRData samples.

    Expects:  data_dir/QRData/QRData.json
              data_dir/QRData/data/*.csv

    Each entry in QRData.json has:
        'question', 'answer', 'file_name' (or 'file_names')

    Returns list of dicts normalised to:
        question (str), answer (str), file_paths (list[str])
    """
    root = Path(data_dir)
    meta_path = _first_existing(
        [
            root / "QRData" / "QRData.json",
            root / "QRData" / "benchmark" / "QRData.json",
        ]
    )
    if meta_path is None:
        raise FileNotFoundError(
            f"QRData.json not found under {root / 'QRData'}.\n"
            "Download with:\n"
            "  git clone https://github.com/xxxiaol/QRData data/QRData"
        )

    csv_dir = _first_existing(
        [
            root / "QRData" / "data",
            root / "QRData" / "benchmark" / "data",
        ]
    )
    zip_path = root / "QRData" / "benchmark" / "data.zip"
    if csv_dir is None and zip_path.exists():
        print(f"Extracting QRData tables from {zip_path} ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(zip_path.parent)
        csv_dir = root / "QRData" / "benchmark" / "data"

    if csv_dir is None:
        raise FileNotFoundError(
            "QRData CSV folder not found. Expected data/QRData/data/ or "
            "data/QRData/benchmark/data/. If you have data.zip, extract it with:\n"
            "  unzip data/QRData/benchmark/data.zip -d data/QRData/benchmark"
        )

    with open(meta_path) as f:
        raw = json.load(f)

    samples = []
    for entry in raw:
        # Normalise file reference field name
        fnames = (
            entry.get("data_files")
            or entry.get("file_names")
            or entry.get("file_name")
            or []
        )
        if isinstance(fnames, str):
            fnames = [fnames]
        file_paths = [str(csv_dir / fn) for fn in fnames]
        samples.append({
            "question":   entry.get("question", ""),
            "answer":     str(entry.get("answer", "")),
            "file_paths": file_paths,
            "data_description": entry.get("data_description", ""),
        })
    print(f"Loaded {len(samples)} QRData samples from {meta_path}")
    return samples


def load_discoverybench(data_dir: str) -> list[dict]:
    """
    Load DiscoveryBench real-world samples.

    Expects:  data_dir/DiscoveryBench/answer_key_real.csv
              data_dir/DiscoveryBench/{domain}/{dataset}/metadata.json
              data_dir/DiscoveryBench/{domain}/{dataset}/*.csv

    Returns list of dicts:
        question (str), answer (str), file_paths (list[str]), domain (str)
    """
    db_dir = Path(data_dir) / "DiscoveryBench"
    key_csv = _first_existing(
        [
            db_dir / "answer_key_real.csv",
            db_dir / "answer_key" / "answer_key_real.csv",
            db_dir / "eval" / "answer_key_real.csv",
        ]
    )
    if key_csv is None:
        raise FileNotFoundError(
            f"DiscoveryBench answer_key_real.csv not found under {db_dir}.\n"
            "Download with:\n"
            "  huggingface-cli download allenai/discoverybench \\\n"
            "    --repo-type dataset --local-dir data/DiscoveryBench"
        )

    real_root = _first_existing(
        [
            db_dir / "discoverybench" / "real" / "test",
            db_dir / "real" / "test",
            db_dir,
        ]
    )
    if real_root is None:
        raise FileNotFoundError(f"DiscoveryBench real/test folder not found under {db_dir}")

    answer_df = pd.read_csv(key_csv)
    samples = []

    for _, row in answer_df.iterrows():
        dataset = str(row.get("dataset", ""))
        meta_id = int(row.get("metadataid", row.get("metadata_id", 0)))
        query_id = int(row.get("query_id", 0))
        answer = str(row.get("gold_hypo", row.get("answer", "")))

        dataset_dir = real_root / dataset
        meta_path = dataset_dir / f"metadata_{meta_id}.json"
        if not meta_path.exists():
            continue
        with open(meta_path, encoding="utf-8", errors="replace") as f:
            meta = json.load(f)

        queries = meta.get("queries", [])
        flat_queries = []
        for group in queries:
            if isinstance(group, list):
                flat_queries.extend(group)
            elif isinstance(group, dict):
                flat_queries.append(group)
        matching_query = next(
            (q for q in flat_queries if int(q.get("qid", -1)) == query_id),
            flat_queries[0] if flat_queries else {},
        )
        question = matching_query.get("question", meta.get("question", ""))

        # Collect CSV data files for this dataset
        file_paths = []
        for table in meta.get("datasets", []):
            path = dataset_dir / table.get("name", "")
            if path.exists():
                file_paths.append(str(path))
        if not file_paths:
            file_paths = [str(p) for p in dataset_dir.glob("*.csv") if " 2.csv" not in p.name]

        samples.append({
            "question":   question,
            "answer":     answer,
            "file_paths": file_paths,
            "domain":     meta.get("domain", ""),
            "dataset":    dataset,
            "metadata_id": meta_id,
            "query_id":   query_id,
        })

    print(f"Loaded {len(samples)} DiscoveryBench real-world samples from {db_dir}")
    return samples


def build_prompt_with_info(sample: dict) -> str:
    """
    Build an input prompt that includes explicit table metadata (column names,
    data types, and sample rows) -- the 'w/ Info' condition in Exp. 1.
    """
    import pandas as pd
    info_parts = []
    for fp in sample.get("file_paths", []):
        try:
            df = pd.read_csv(fp, nrows=3)
            col_info = ", ".join(
                f"{col} ({dtype})"
                for col, dtype in zip(df.columns, df.dtypes)
            )
            sample_rows = df.head(3).to_string(index=False)
            info_parts.append(
                f"File: {os.path.basename(fp)}\n"
                f"Columns: {col_info}\n"
                f"Sample rows:\n{sample_rows}"
            )
        except Exception:
            info_parts.append(f"File: {os.path.basename(fp)} (could not read)")

    schema_block = "\n\n".join(info_parts)
    return (
        f"Question: {sample['question']}\n\n"
        f"Background: {sample.get('data_description', '')}\n\n"
        f"Available data files:\n{schema_block}"
    )


def build_prompt_without_info(sample: dict) -> str:
    """
    Build an input prompt with only filenames -- the 'w/o Info' condition.
    """
    filenames = [os.path.basename(fp) for fp in sample.get("file_paths", [])]
    file_list = "\n".join(f"  - {fn}" for fn in filenames)
    return (
        f"Question: {sample['question']}\n\n"
        f"Background: {sample.get('data_description', '')}\n\n"
        f"Available data files:\n{file_list}"
    )


def add_extra_files(sample: dict, extra_files: list[str]) -> dict:
    """
    Return a copy of the sample with additional irrelevant file paths injected.
    Used for the 'w/ Extra' data complexity experiment.
    """
    import copy
    s = copy.deepcopy(sample)
    s["file_paths"] = s.get("file_paths", []) + extra_files
    return s


def get_turn_length_category(n_turns: int) -> str:
    """Classify a trajectory by number of assistant turns."""
    if n_turns <= 3:
        return "short"
    elif n_turns <= 5:
        return "medium"
    else:
        return "long"
