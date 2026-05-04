"""
Evaluation utilities shared across experiments.
Uses GPT-4o-mini as agreement judge, matching the DataMind evaluation protocol.
"""

import json
import os
import re
from openai import OpenAI

# ── Judge client ──────────────────────────────────────────────────────────────

def get_judge_client(backend: str = "openai"):
    if backend == "local":
        return None
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set.")
    return OpenAI(api_key=api_key)


JUDGE_SYSTEM_PROMPT = (
    "You are a strict answer judge. Compare the predicted answer and the ground-truth "
    "answer. Reply with exactly 'correct' if they are equivalent (within tolerance if "
    "numerical), or 'incorrect' otherwise. Do not output anything else."
)

QRDATA_JUDGE_PROMPT = (
    "Ground truth: {gt}\n"
    "Prediction:   {pred}\n"
    "Rule: for numerical answers, allow 3% relative tolerance. "
    "Is the prediction correct? Reply 'correct' or 'incorrect'."
)

DISCOVERYBENCH_JUDGE_PROMPT = (
    "Ground truth: {gt}\n"
    "Prediction:   {pred}\n"
    "Rule: for numerical answers, allow 1% relative tolerance. "
    "Is the prediction correct? Reply 'correct' or 'incorrect'."
)


def _normalize_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9.+\-% ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _extract_numbers(text: str) -> list[float]:
    values = []
    for match in re.finditer(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?%?", str(text)):
        raw = match.group(0)
        try:
            if raw.endswith("%"):
                values.append(float(raw[:-1]) / 100.0)
            else:
                values.append(float(raw))
        except ValueError:
            continue
    return values


def _check_answer_equiv_local(pred: str, gt: str, dataset: str) -> bool:
    """Deterministic fallback judge for API-free pilot experiments."""
    pred_norm = _normalize_text(pred)
    gt_norm = _normalize_text(gt)
    if not pred_norm or not gt_norm:
        return False

    gt_nums = _extract_numbers(gt)
    pred_nums = _extract_numbers(pred)
    if gt_nums and pred_nums:
        rel_tol = 0.03 if dataset.lower() == "qrdata" else 0.01
        for gt_num in gt_nums:
            matched = False
            for pred_num in pred_nums:
                scale = max(abs(gt_num), 1.0)
                if abs(pred_num - gt_num) <= rel_tol * scale:
                    matched = True
                    break
            if not matched:
                return False
        return True

    return gt_norm == pred_norm or gt_norm in pred_norm


def check_answer_equiv(pred: str, gt: str, dataset: str, client: OpenAI | None) -> bool:
    """Call GPT-4o-mini to judge if pred matches gt."""
    if client is None:
        return _check_answer_equiv_local(pred, gt, dataset)

    template = (
        QRDATA_JUDGE_PROMPT if dataset.lower() == "qrdata"
        else DISCOVERYBENCH_JUDGE_PROMPT
    )
    user_msg = template.format(gt=gt, pred=pred)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0,
        max_tokens=10,
    )
    verdict = resp.choices[0].message.content.strip().lower()
    return verdict == "correct"


# ── Result I/O ────────────────────────────────────────────────────────────────

def load_results(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def save_results(results: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} results to {path}")


# ── Accuracy computation ──────────────────────────────────────────────────────

def compute_accuracy(results: list[dict]) -> float:
    """Compute accuracy from a list of result dicts with a 'correct' bool field."""
    if not results:
        return 0.0
    return sum(r["correct"] for r in results) / len(results) * 100


def compute_code_error_rate(results: list[dict]) -> float:
    """Fraction of results that contain at least one code execution error."""
    if not results:
        return 0.0
    errors = sum(1 for r in results if r.get("has_code_error", False))
    return errors / len(results) * 100


# ── Text extraction helpers ───────────────────────────────────────────────────

def extract_final_answer(response: str) -> str:
    """Extract the content after '## Final Answer:' marker."""
    marker = "## Final Answer:"
    idx = response.rfind(marker)
    if idx == -1:
        return response.strip()
    return response[idx + len(marker):].strip()


def extract_code_blocks(response: str) -> list[str]:
    """Extract all Python code blocks from a markdown-formatted response."""
    return [
        block.strip()
        for block in re.findall(r"```(?:python|py)?\s*\n(.*?)```", response, re.DOTALL | re.IGNORECASE)
        if block.strip()
    ]


def count_turns(messages: list[dict]) -> int:
    """Count the number of assistant turns in a conversation."""
    return sum(1 for m in messages if m.get("role") == "assistant")


def has_code_error(messages: list[dict]) -> bool:
    """Return True if any observation message contains an execution error."""
    for m in messages:
        content = m.get("content", "")
        if m.get("role") == "tool" or "Observation" in content:
            if any(kw in content for kw in ["Error", "Traceback", "Exception"]):
                return True
    return False


# ── Summary printing ──────────────────────────────────────────────────────────

def print_summary(label: str, results: list[dict]) -> None:
    acc = compute_accuracy(results)
    err = compute_code_error_rate(results)
    print(f"[{label}]  Accuracy: {acc:.2f}%  |  Code Error Rate: {err:.2f}%  |  N={len(results)}")
