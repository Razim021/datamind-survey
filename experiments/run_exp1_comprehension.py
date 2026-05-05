#!/usr/bin/env python3
"""
Experiment 1 – Data Comprehension Ablation
==========================================
Tests whether providing explicit table metadata (column names, data types, sample
rows) or introducing irrelevant distractor files affects model accuracy.

Two sub-experiments:
  A) w/o Info vs w/ Info  (tabular schema visibility)
  B) w/o Extra vs w/ Extra (irrelevant distractor files)

Usage:
    # Start vLLM server first (see README.md for instructions), then:
    python run_exp1_comprehension.py \
        --model_name Qwen2.5-7B-Instruct \
        --data_dir /path/to/data \
        --api_port 8000 \
        --output_dir results/exp1

Results are saved as JSON files and a summary table is printed
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ── project imports ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from utils.evaluate import (
    check_answer_equiv, compute_accuracy, save_results,
    extract_final_answer, print_summary, get_judge_client,
    compute_code_error_rate, extract_code_blocks,
)
from utils.data_loader import (
    load_qrdata, load_discoverybench,
    build_prompt_with_info, build_prompt_without_info, add_extra_files,
)
from utils.datamind_compat import (
    CodeRunner,
    SYSTEM_PROMPT,
    chat_with_model,
    run_python_code,
)


# ── Core evaluation loop ───────────────────────────────────────────────────────

def run_analysis(sample: dict, prompt: str, model_name: str,
                 api_port: int, max_rounds: int = 10) -> dict:
    """
    Run a single multi-turn ReAct data-analysis interaction.
    Returns a result dict with question, prediction, correctness, messages, etc.
    """
    runner = CodeRunner()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.append({"role": "user", "content": prompt})

    final_answer = ""
    code_error_detected = False

    for turn_idx in range(max_rounds):
        response = chat_with_model(
            messages=messages,
            model=model_name,
            port=api_port,
            temperature=0,
        )
        messages.append({"role": "assistant", "content": response})

        if "## Final Answer:" in response:
            final_answer = extract_final_answer(response)
            break

        code_blocks = extract_code_blocks(response)
        if not code_blocks and turn_idx == max_rounds - 1:
            final_answer = extract_final_answer(response)

        for code in code_blocks:
            stdout, stderr, has_error = run_python_code(runner, code, sample)
            obs_content = stdout or stderr or "[Executed Successfully with No Output]"
            if has_error:
                code_error_detected = True
            messages.append({"role": "user", "content": f"## Observation:\n{obs_content}"})

    return {
        "question":        sample["question"],
        "ground_truth":    sample["answer"],
        "prediction":      final_answer,
        "messages":        messages,
        "has_code_error":  code_error_detected,
        "correct":         False,  # filled in after judge call
    }


def evaluate_condition(samples: list[dict], prompt_fn, model_name: str,
                       dataset_name: str, api_port: int,
                       judge_client, max_rounds: int = 4,
                       time_budget_s: float | None = None,
                       checkpoint_path: str | None = None) -> list[dict]:
    """Evaluate a list of samples using a given prompt-building function.

    If ``time_budget_s`` is set, stop early when budget is exhausted and
    return what we have. ``checkpoint_path`` (optional) saves results after
    each sample so a Colab disconnect doesn't lose data.
    """
    results = []
    start = time.time()
    for i, sample in enumerate(samples):
        try:
            prompt = prompt_fn(sample)
            result = run_analysis(sample, prompt, model_name, api_port, max_rounds)
            result["correct"] = check_answer_equiv(
                result["prediction"], result["ground_truth"],
                dataset_name, judge_client
            )
        except Exception as exc:  # one bad sample shouldn't kill the whole run
            print(f"  [warn] sample {i} failed: {exc!r}")
            result = {
                "question": sample.get("question", ""),
                "ground_truth": sample.get("answer", ""),
                "prediction": "",
                "messages": [],
                "has_code_error": True,
                "correct": False,
                "error": repr(exc),
            }
        results.append(result)
        if checkpoint_path:
            try:
                save_results(results, checkpoint_path)
            except Exception:
                pass
        if (i + 1) % 10 == 0 or (i + 1) == len(samples):
            acc = compute_accuracy(results)
            elapsed = time.time() - start
            print(f"  [{i+1}/{len(samples)}] acc={acc:.1f}%  elapsed={elapsed:.0f}s")
        if time_budget_s and (time.time() - start) > time_budget_s:
            print(f"  [time] budget {time_budget_s:.0f}s exceeded after {i+1} samples - stopping early")
            break
    return results


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Exp 1: Data Comprehension Ablation")
    parser.add_argument("--model_name", default="Qwen2.5-7B-Instruct")
    parser.add_argument("--data_dir",   default="data",
                        help="Root directory containing QRData/ and DiscoveryBench/")
    parser.add_argument("--api_port",   type=int, default=8000,
                        help="vLLM server port")
    parser.add_argument("--output_dir", default="results/exp1")
    parser.add_argument("--dataset",    choices=["qrdata", "discoverybench", "both"],
                        default="qrdata")
    parser.add_argument("--max_rounds", type=int, default=4)
    parser.add_argument("--sub_experiment", choices=["info", "extra", "both"],
                        default="info",
                        help="'info' = table metadata; 'extra' = distractor files")
    parser.add_argument("--n_samples",  type=int, default=None,
                        help="Limit samples for quick testing (None = all)")
    parser.add_argument("--extra_files_dir", default=None,
                        help="Directory of distractor CSV files for sub-exp B")
    parser.add_argument("--judge_backend", choices=["openai", "local"],
                        default=os.environ.get("JUDGE_BACKEND", "local"),
                        help="Use 'local' to avoid OpenAI API judging.")
    parser.add_argument("--time_budget_s", type=int, default=None,
                        help="Soft per-condition wall-clock budget in seconds.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    judge = get_judge_client(args.judge_backend)

    datasets = {}
    if args.dataset in ("qrdata", "both"):
        datasets["qrdata"] = load_qrdata(args.data_dir)
    if args.dataset in ("discoverybench", "both"):
        datasets["discoverybench"] = load_discoverybench(args.data_dir)

    # Optionally limit sample count for quick runs
    if args.n_samples:
        datasets = {k: v[:args.n_samples] for k, v in datasets.items()}

    # Collect distractor files if needed
    extra_files = []
    if args.sub_experiment in ("extra", "both") and args.extra_files_dir:
        extra_files = [
            str(p) for p in Path(args.extra_files_dir).glob("*.csv")
        ][:3]  # use at most 3 distractors
        print(f"Using {len(extra_files)} distractor files from {args.extra_files_dir}")

    summary = {}
    for ds_name, samples in datasets.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name.upper()}  |  Model: {args.model_name}")
        print(f"{'='*60}")

        if args.sub_experiment in ("info", "both"):
            # ── Sub-experiment A: w/o Info ──────────────────────────────────
            print("\n[Sub-exp A] Condition: w/o Info")
            ckpt_wo = os.path.join(args.output_dir, f"{ds_name}_wo_info_{args.model_name}.json")
            wo_info = evaluate_condition(
                samples, build_prompt_without_info,
                args.model_name, ds_name, args.api_port, judge, args.max_rounds,
                time_budget_s=args.time_budget_s, checkpoint_path=ckpt_wo,
            )
            save_results(wo_info, os.path.join(
                args.output_dir, f"{ds_name}_wo_info_{args.model_name}.json"))
            print_summary(f"{ds_name} | w/o Info", wo_info)
            summary[f"{ds_name}_without_metadata"] = {
                "accuracy": round(compute_accuracy(wo_info), 2),
                "code_error_rate": round(compute_code_error_rate(wo_info), 2),
                "n": len(wo_info),
            }

            # ── Sub-experiment A: w/ Info ───────────────────────────────────
            print("\n[Sub-exp A] Condition: w/ Info")
            ckpt_w = os.path.join(args.output_dir, f"{ds_name}_w_info_{args.model_name}.json")
            w_info = evaluate_condition(
                samples, build_prompt_with_info,
                args.model_name, ds_name, args.api_port, judge, args.max_rounds,
                time_budget_s=args.time_budget_s, checkpoint_path=ckpt_w,
            )
            save_results(w_info, os.path.join(
                args.output_dir, f"{ds_name}_w_info_{args.model_name}.json"))
            print_summary(f"{ds_name} | w/ Info", w_info)
            summary[f"{ds_name}_with_metadata"] = {
                "accuracy": round(compute_accuracy(w_info), 2),
                "code_error_rate": round(compute_code_error_rate(w_info), 2),
                "n": len(w_info),
            }
            summary[f"{ds_name}_info_delta"] = (
                compute_accuracy(w_info) - compute_accuracy(wo_info)
            )

        if args.sub_experiment in ("extra", "both") and extra_files:
            # ── Sub-experiment B: w/o Extra ────────────────────────────────
            print("\n[Sub-exp B] Condition: w/o Extra files")
            wo_extra = evaluate_condition(
                samples, build_prompt_without_info,
                args.model_name, ds_name, args.api_port, judge, args.max_rounds,
                time_budget_s=args.time_budget_s,
                checkpoint_path=os.path.join(args.output_dir, f"{ds_name}_wo_extra_{args.model_name}.json"),
            )
            save_results(wo_extra, os.path.join(
                args.output_dir, f"{ds_name}_wo_extra_{args.model_name}.json"))
            print_summary(f"{ds_name} | w/o Extra", wo_extra)
            summary[f"{ds_name}_without_extra_files"] = {
                "accuracy": round(compute_accuracy(wo_extra), 2),
                "code_error_rate": round(compute_code_error_rate(wo_extra), 2),
                "n": len(wo_extra),
            }

            # ── Sub-experiment B: w/ Extra ─────────────────────────────────
            print("\n[Sub-exp B] Condition: w/ Extra files")
            samples_with_extra = [
                add_extra_files(s, extra_files) for s in samples
            ]
            w_extra = evaluate_condition(
                samples_with_extra, build_prompt_without_info,
                args.model_name, ds_name, args.api_port, judge, args.max_rounds,
                time_budget_s=args.time_budget_s,
                checkpoint_path=os.path.join(args.output_dir, f"{ds_name}_w_extra_{args.model_name}.json"),
            )
            save_results(w_extra, os.path.join(
                args.output_dir, f"{ds_name}_w_extra_{args.model_name}.json"))
            print_summary(f"{ds_name} | w/ Extra", w_extra)
            summary[f"{ds_name}_with_extra_files"] = {
                "accuracy": round(compute_accuracy(w_extra), 2),
                "code_error_rate": round(compute_code_error_rate(w_extra), 2),
                "n": len(w_extra),
            }
            summary[f"{ds_name}_extra_delta"] = (
                compute_accuracy(w_extra) - compute_accuracy(wo_extra)
            )

    print("\n\n=== EXPERIMENT 1 SUMMARY ===")
    for k, v in summary.items():
        if isinstance(v, dict):
            print(f"  {k}: Acc={v['accuracy']}%  CodeErr={v['code_error_rate']}%  N={v['n']}")
        else:
            print(f"  {k}: delta = {v:+.2f} pp")

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {args.output_dir}/summary.json")


if __name__ == "__main__":
    main()
