#!/usr/bin/env python3
"""
Experiment 2 – Code Generation Analysis
========================================
Evaluates multi-turn accuracy and code error rates across multiple models,
then categorizes a sampled set of errors into:
  - Planning / Reasoning errors
  - Data Understanding errors
  - Code (syntax/semantic) errors

Usage:
    python run_exp2_code_analysis.py \
        --models Qwen2.5-7B-Instruct,Qwen2.5-14B-Instruct \
        --data_dir /path/to/data \
        --api_port 8000 \
        --output_dir results/exp2

    # After collecting results, run error categorization:
    python run_exp2_code_analysis.py \
        --mode categorize \
        --results_dir results/exp2 \
        --n_samples 354 \
        --output_dir results/exp2
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils.evaluate import (
    check_answer_equiv, compute_accuracy, compute_code_error_rate,
    save_results, extract_final_answer, has_code_error,
    print_summary, get_judge_client,
)
from utils.data_loader import load_qrdata, load_discoverybench, build_prompt_without_info
from utils.datamind_compat import (
    CodeRunner,
    SYSTEM_PROMPT,
    chat_with_model,
    run_python_code,
)


# ── Error categorization ───────────────────────────────────────────────────────

ERROR_CATEGORIZATION_PROMPT = """
You are an expert at analyzing LLM data analysis failures.

Below is an incorrect model response to a data analysis question.
Categorize the primary cause of failure into exactly ONE of these categories:

1. "planning_reasoning" – incorrect hypothesis formulation, flawed step sequencing,
   premature termination, wrong analytical approach, or failure to adapt to
   intermediate observations.
2. "data_understanding" – wrong table/column interpretation, misidentified relevant
   data, misread data types or values.
3. "code_error" – pure syntax error, runtime exception, incorrect use of pandas/numpy
   API, or code that does not produce any meaningful output.

Question: {question}
Model response: {response}
Ground truth answer: {ground_truth}

Reply with ONLY one of: planning_reasoning, data_understanding, code_error
"""

# Error category display names
CATEGORY_LABELS = {
    "planning_reasoning": "Planning & Reasoning Error",
    "data_understanding": "Data Understanding Error",
    "code_error":         "Code Error",
}


def categorize_errors(error_samples: list[dict], judge_client) -> dict:
    """
    Use GPT-4o-mini to categorize each error sample.
    Returns a dict mapping category -> count.
    """
    from openai import OpenAI
    counts = {"planning_reasoning": 0, "data_understanding": 0, "code_error": 0}

    for i, sample in enumerate(error_samples):
        last_assistant = ""
        for m in reversed(sample.get("messages", [])):
            if m.get("role") == "assistant":
                last_assistant = m["content"]
                break

        if judge_client is None:
            text = last_assistant.lower()
            if sample.get("has_code_error"):
                category = "code_error"
            elif any(word in text for word in ["column", "file", "csv", "dataframe", "keyerror"]):
                category = "data_understanding"
            else:
                category = "planning_reasoning"
        else:
            prompt = ERROR_CATEGORIZATION_PROMPT.format(
                question=sample.get("question", ""),
                response=last_assistant[:3000],  # truncate to save tokens
                ground_truth=sample.get("ground_truth", ""),
            )
            resp = judge_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=20,
            )
            category = resp.choices[0].message.content.strip().lower()
        if category in counts:
            counts[category] += 1
        else:
            counts["planning_reasoning"] += 1  # default to planning if unclear

        if (i + 1) % 50 == 0:
            print(f"  Categorized {i+1}/{len(error_samples)} errors ...")

    return counts


# ── Multi-turn evaluation (shared with Exp 1 runner) ──────────────────────────

def run_single(sample: dict, prompt: str, model_name: str,
               api_port: int, max_rounds: int = 10) -> dict:
    import re
    runner = CodeRunner()
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt}]
    final_answer = ""
    code_error_detected = False

    for _ in range(max_rounds):
        response = chat_with_model(messages=messages, model=model_name,
                                   port=api_port, temperature=0)
        messages.append({"role": "assistant", "content": response})

        if "## Final Answer:" in response:
            final_answer = extract_final_answer(response)
            break

        code_blocks = re.findall(r"```python\n(.*?)```", response, re.DOTALL)
        for code in code_blocks:
            stdout, stderr, has_error = run_python_code(runner, code, sample)
            obs = stdout or stderr or "[Executed Successfully with No Output]"
            if has_error:
                code_error_detected = True
            messages.append({"role": "user", "content": f"## Observation:\n{obs}"})

    return {
        "question":       sample["question"],
        "ground_truth":   sample["answer"],
        "prediction":     final_answer,
        "messages":       messages,
        "has_code_error": code_error_detected,
        "correct":        False,
    }


def evaluate_model(model_name: str, samples: list[dict], dataset_name: str,
                   api_port: int, judge_client, max_rounds: int = 10) -> list[dict]:
    results = []
    for i, sample in enumerate(samples):
        prompt = build_prompt_without_info(sample)
        result = run_single(sample, prompt, model_name, api_port, max_rounds)
        result["correct"] = check_answer_equiv(
            result["prediction"], result["ground_truth"],
            dataset_name, judge_client
        )
        results.append(result)
        if (i + 1) % 25 == 0:
            print(f"  [{model_name}] [{i+1}/{len(samples)}] "
                  f"Acc={compute_accuracy(results):.1f}%")
    return results


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Exp 2: Code Generation Analysis")
    parser.add_argument("--mode", choices=["evaluate", "categorize"], default="evaluate")
    parser.add_argument("--models",
                        default="Qwen2.5-7B-Instruct,Qwen2.5-14B-Instruct",
                        help="Comma-separated model names to evaluate")
    parser.add_argument("--data_dir",    default="data")
    parser.add_argument("--api_port",    type=int, default=8000)
    parser.add_argument("--output_dir",  default="results/exp2")
    parser.add_argument("--results_dir", default="results/exp2",
                        help="Used in categorize mode to load existing results")
    parser.add_argument("--dataset",
                        choices=["qrdata", "discoverybench", "both"],
                        default="both")
    parser.add_argument("--n_samples",   type=int, default=None)
    parser.add_argument("--n_errors",    type=int, default=354,
                        help="Number of error samples to categorize")
    parser.add_argument("--max_rounds",  type=int, default=10)
    parser.add_argument("--judge_backend", choices=["openai", "local"],
                        default=os.environ.get("JUDGE_BACKEND", "openai"),
                        help="Use 'local' to avoid OpenAI API judging.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    judge = get_judge_client(args.judge_backend)

    if args.mode == "evaluate":
        # ── Evaluation mode ────────────────────────────────────────────────
        models = [m.strip() for m in args.models.split(",")]
        datasets = {}
        if args.dataset in ("qrdata", "both"):
            datasets["qrdata"] = load_qrdata(args.data_dir)
        if args.dataset in ("discoverybench", "both"):
            datasets["discoverybench"] = load_discoverybench(args.data_dir)
        if args.n_samples:
            datasets = {k: v[:args.n_samples] for k, v in datasets.items()}

        all_summary = {}
        for model in models:
            for ds_name, samples in datasets.items():
                print(f"\n[EVAL] Model={model}  Dataset={ds_name.upper()}")
                results = evaluate_model(
                    model, samples, ds_name, args.api_port, judge, args.max_rounds
                )
                out_path = os.path.join(
                    args.output_dir, f"{ds_name}_{model.replace('/', '_')}.json"
                )
                save_results(results, out_path)
                acc = compute_accuracy(results)
                err = compute_code_error_rate(results)
                all_summary[f"{model}_{ds_name}"] = {
                    "accuracy":        round(acc, 2),
                    "code_error_rate": round(err, 2),
                    "n":               len(results),
                }
                print_summary(f"{model} | {ds_name}", results)

        print("\n\n=== EXPERIMENT 2 SUMMARY ===")
        for k, v in all_summary.items():
            print(f"  {k}: Acc={v['accuracy']}%  CodeErr={v['code_error_rate']}%")
        with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
            json.dump(all_summary, f, indent=2)

    elif args.mode == "categorize":
        # ── Error categorization mode ──────────────────────────────────────
        # Collect all incorrect results from existing result files
        error_samples = []
        for fp in Path(args.results_dir).glob("*.json"):
            if "summary" in fp.name:
                continue
            with open(fp) as f:
                data = json.load(f)
            errors = [r for r in data if not r.get("correct", True)]
            error_samples.extend(errors)

        if not error_samples:
            print("No error samples found in results_dir. Run --mode evaluate first.")
            return

        # Sample without replacement if we have more than needed
        if len(error_samples) > args.n_errors:
            random.seed(42)
            error_samples = random.sample(error_samples, args.n_errors)

        print(f"\nCategorizing {len(error_samples)} error samples ...")
        counts = categorize_errors(error_samples, judge)
        total = sum(counts.values())

        print("\n=== ERROR CATEGORIZATION RESULTS ===")
        for cat, cnt in counts.items():
            pct = cnt / total * 100 if total > 0 else 0
            print(f"  {CATEGORY_LABELS[cat]}: {cnt}/{total} ({pct:.1f}%)")

        out = {
            "n_total": total,
            "categories": {
                k: {"count": v, "percent": round(v / total * 100, 1) if total else 0}
                for k, v in counts.items()
            },
        }
        out_path = os.path.join(args.output_dir, "error_categories.json")
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
