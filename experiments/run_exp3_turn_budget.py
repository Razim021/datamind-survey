#!/usr/bin/env python3
"""
Experiment 3 - Turn-Budget Study
================================
Tests whether giving the same model more ReAct analysis turns improves or hurts
data-analysis accuracy. This keeps the project resource-aware: no fine-tuning,
no multi-GPU job, and no generated training corpus is required.

Example:
    python run_exp3_turn_budget.py \
        --model_name Qwen2.5-7B-Instruct \
        --data_dir ../data \
        --dataset qrdata \
        --turn_budgets 2,4,6 \
        --n_samples 50 \
        --api_port 8000 \
        --judge_backend local \
        --output_dir results/exp3_colab
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils.data_loader import load_discoverybench, load_qrdata, build_prompt_without_info
from utils.datamind_compat import (
    CodeRunner,
    SYSTEM_PROMPT,
    chat_with_model,
    run_python_code,
)
from utils.evaluate import (
    check_answer_equiv,
    compute_accuracy,
    compute_code_error_rate,
    extract_code_blocks,
    extract_final_answer,
    get_judge_client,
    print_summary,
    save_results,
)


def run_single(sample: dict, model_name: str, api_port: int, max_rounds: int) -> dict:
    runner = CodeRunner()
    prompt = build_prompt_without_info(sample)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
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
            observation = stdout or stderr or "[Executed Successfully with No Output]"
            if has_error:
                code_error_detected = True
            messages.append({"role": "user", "content": f"## Observation:\n{observation}"})

    return {
        "question": sample["question"],
        "ground_truth": sample["answer"],
        "prediction": final_answer,
        "turn_budget": max_rounds,
        "actual_assistant_turns": sum(1 for m in messages if m.get("role") == "assistant"),
        "messages": messages,
        "has_code_error": code_error_detected,
        "correct": False,
    }


def evaluate_budget(
    samples: list[dict],
    dataset_name: str,
    model_name: str,
    api_port: int,
    judge_client,
    max_rounds: int,
    time_budget_s: float | None = None,
    checkpoint_path: str | None = None,
) -> list[dict]:
    results = []
    start = time.time()
    for i, sample in enumerate(samples):
        try:
            result = run_single(sample, model_name, api_port, max_rounds)
            result["correct"] = check_answer_equiv(
                result["prediction"],
                result["ground_truth"],
                dataset_name,
                judge_client,
            )
        except Exception as exc:
            print(f"  [warn] sample {i} failed: {exc!r}")
            result = {
                "question": sample.get("question", ""),
                "ground_truth": sample.get("answer", ""),
                "prediction": "",
                "turn_budget": max_rounds,
                "actual_assistant_turns": 0,
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
        if (i + 1) % 5 == 0 or (i + 1) == len(samples):
            elapsed = time.time() - start
            print(
                f"  [turns={max_rounds}] [{i + 1}/{len(samples)}] "
                f"Acc={compute_accuracy(results):.1f}% elapsed={elapsed:.0f}s"
            )
        if time_budget_s and (time.time() - start) > time_budget_s:
            print(f"  [time] budget {time_budget_s:.0f}s exceeded after {i + 1} samples - stopping early")
            break
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp 3: turn-budget study")
    parser.add_argument("--model_name", default="Qwen2.5-7B-Instruct")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--api_port", type=int, default=8000)
    parser.add_argument("--output_dir", default="results/exp3")
    parser.add_argument("--dataset", choices=["qrdata", "discoverybench", "both"], default="qrdata")
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--turn_budgets", default="2,4,6")
    parser.add_argument(
        "--judge_backend",
        choices=["openai", "local"],
        default=os.environ.get("JUDGE_BACKEND", "local"),
        help="Use 'local' to avoid OpenAI API judging.",
    )
    parser.add_argument(
        "--time_budget_s",
        type=int,
        default=None,
        help="Soft per-turn-budget wall-clock limit (seconds).",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    judge = get_judge_client(args.judge_backend)

    datasets = {}
    if args.dataset in ("qrdata", "both"):
        datasets["qrdata"] = load_qrdata(args.data_dir)
    if args.dataset in ("discoverybench", "both"):
        datasets["discoverybench"] = load_discoverybench(args.data_dir)
    if args.n_samples:
        datasets = {name: rows[: args.n_samples] for name, rows in datasets.items()}

    budgets = [int(x.strip()) for x in args.turn_budgets.split(",") if x.strip()]
    summary = {}

    for dataset_name, samples in datasets.items():
        for budget in budgets:
            print(
                f"\n[EXP3] Model={args.model_name} Dataset={dataset_name.upper()} "
                f"Turn budget={budget}"
            )
            safe_model = args.model_name.replace("/", "_")
            out_path = os.path.join(
                args.output_dir,
                f"{dataset_name}_turns{budget}_{safe_model}.json",
            )
            results = evaluate_budget(
                samples=samples,
                dataset_name=dataset_name,
                model_name=args.model_name,
                api_port=args.api_port,
                judge_client=judge,
                max_rounds=budget,
                time_budget_s=args.time_budget_s,
                checkpoint_path=out_path,
            )
            save_results(results, out_path)
            print_summary(f"{dataset_name} | turns={budget}", results)
            summary[f"{dataset_name}_turns_{budget}"] = {
                "accuracy": round(compute_accuracy(results), 2),
                "code_error_rate": round(compute_code_error_rate(results), 2),
                "n": len(results),
            }

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== EXPERIMENT 3 SUMMARY ===")
    for key, value in summary.items():
        print(f"  {key}: Acc={value['accuracy']}% CodeErr={value['code_error_rate']}%")
    print(f"\nSummary saved to {args.output_dir}/summary.json")


if __name__ == "__main__":
    main()
