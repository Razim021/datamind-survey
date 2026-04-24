"""
Compatibility helpers for running the survey experiments against the bundled
DataMind codebase.

The upstream DataMind repository changed names/functions across revisions, so
the experiment scripts import from this module instead of importing DataMind
symbols directly.
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import traceback
from pathlib import Path

from openai import OpenAI


REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_datamind_root() -> Path | None:
    """Find DataMind either from DATAMIND_ROOT or from common local locations."""
    candidates = []
    if os.environ.get("DATAMIND_ROOT"):
        candidates.append(Path(os.environ["DATAMIND_ROOT"]).expanduser())
    candidates.extend(
        [
            REPO_ROOT / "Datamind-main",
            REPO_ROOT.parent / "Datamind-main",
            REPO_ROOT.parent / "DataMind-main",
        ]
    )

    for candidate in candidates:
        eval_dir = candidate / "eval" / "DataMind-Analysis"
        if eval_dir.exists():
            return candidate

    return None


DATAMIND_ROOT = resolve_datamind_root()
DATAMIND_EVAL = (
    DATAMIND_ROOT / "eval" / "DataMind-Analysis"
    if DATAMIND_ROOT is not None
    else None
)
if DATAMIND_EVAL is not None and str(DATAMIND_EVAL) not in sys.path:
    sys.path.insert(0, str(DATAMIND_EVAL))


try:
    from prompt import system_prompt_template as SYSTEM_PROMPT
except Exception:
    SYSTEM_PROMPT = (
        "You are an experienced data analyst. Use clear ## Thought steps, "
        "write Python code when needed, wait for ## Observation messages, "
        "and end with ## Final Answer:."
    )


def chat_with_model(
    messages: list[dict],
    model: str,
    port: int | None = 8000,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    top_p: float = 1.0,
    api_key: str | None = None,
) -> str:
    """
    Call either a local vLLM OpenAI-compatible server or an API model.

    Non-GPT/non-DeepSeek model names are treated as local vLLM models. For local
    runs, start vLLM with --served-model-name matching the --model_name value.
    """
    model_lower = model.lower()
    if "deepseek" in model_lower and port is None:
        client = OpenAI(
            api_key=api_key or os.environ.get("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )
    elif model_lower.startswith("gpt") or "gpt-" in model_lower:
        client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url="https://api.openai.com/v1",
        )
    else:
        client = OpenAI(
            api_key=api_key or "EMPTY",
            base_url=f"http://localhost:{port}/v1",
        )

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=False,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
    )
    return response.choices[0].message.content or ""


if os.environ.get("USE_DATAMIND_CODERUNNER") == "1":
    try:
        from python_executor import CodeRunner as _DataMindCodeRunner
    except Exception:
        _DataMindCodeRunner = None
else:
    _DataMindCodeRunner = None


class _FallbackCodeRunner:
    """Small persistent Python runner used when optional DataMind deps are absent."""

    def __init__(self, timeout: int = 120):
        self.timeout = timeout
        self.globals: dict = {}
        self._initialize()

    def _initialize(self) -> None:
        imports = [
            "import math",
            "import statistics",
            "import numpy as np",
            "import pandas as pd",
            "import scipy.stats as stats",
            "import statsmodels.api as sm",
            "from statsmodels.formula.api import ols",
            "from sklearn.linear_model import LinearRegression, LogisticRegression",
            "from sklearn.model_selection import train_test_split",
            "from sklearn.preprocessing import StandardScaler",
        ]
        for stmt in imports:
            try:
                exec(stmt, self.globals)
            except Exception:
                pass

    def run_code(self, python_code: str, base_path: str | None = None):
        cwd = os.getcwd()
        try:
            if base_path:
                os.chdir(base_path)
            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exec(python_code, self.globals)
            err = stderr.getvalue().strip()
            return stdout.getvalue().strip(), err, bool(err)
        except Exception:
            return "", traceback.format_exc(limit=5).strip(), True
        finally:
            os.chdir(cwd)


CodeRunner = _DataMindCodeRunner or _FallbackCodeRunner


def get_sample_base_path(sample: dict) -> str:
    """Return the directory where sample files should be read from."""
    file_paths = [Path(p) for p in sample.get("file_paths", []) if p]
    existing_parents = [p.parent for p in file_paths if p.exists()]
    if existing_parents:
        return str(existing_parents[0])
    if file_paths:
        return str(file_paths[0].parent)
    return str(REPO_ROOT)


def run_python_code(runner, python_code: str, sample: dict):
    """Execute model code in the sample's data directory and normalize outputs."""
    cwd = os.getcwd()
    try:
        result = runner.run_code(python_code, base_path=get_sample_base_path(sample))
    finally:
        os.chdir(cwd)

    if isinstance(result, tuple) and len(result) == 3:
        out, err, has_error = result
    elif isinstance(result, tuple) and len(result) == 2:
        out, err = result
        has_error = bool(err)
    else:
        out, err, has_error = str(result), "", False
    return out, err, has_error
