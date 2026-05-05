# Survey: A Resource-Aware Reproduction of DataMind

Runnable code, Colab notebook, and final experiment outputs for our survey of:

> Zhu et al. (2025). *Why Do Open-Source LLMs Struggle with Data Analysis?
> A Systematic Empirical Study* (arXiv:2506.19794).
> Upstream code: <https://github.com/zjunlp/DataMind>.

This repo is not a fork of DataMind. It uses the upstream `DataMind/eval/`
codebase as a local dependency for prompt/executor compatibility, while our
own scripts in `experiments/` define the runnable course experiments.

## Revised Scope

The proposal feedback was right: the original plan was too broad for a final
run tonight. It referred to many original-paper tables without spelling out
the actual measurements, assumed multi-A100 access, and included fine-tuning.

This version is intentionally smaller and safer:

| # | Paper connection | What this repo runs | Default Colab cost |
|---|---|---|---|
| Exp 1 | Zhu et al. Table 1, table-info ablation | QRData prompts with filenames only vs. filenames plus columns, dtypes, and 3 sample rows | about 50 min |
| Exp 2 | Zhu et al. Table 4 and Figure 2, code/error analysis | Categorize wrong baseline Exp-1 trajectories without extra model runs | under 1 min |
| Exp 3 | Lightweight inference analogue of Zhu et al. Section 4.3/Table 5 turn-length finding | Same model and samples, max ReAct turns in `{2, 4, 6}` | about 75 min |
| Optional | Zhu et al. Table 1 model-scale comparison | Rerun Exp 1 only with Qwen2.5-14B-Instruct | about 30-60 min |

Dropped from the final runnable scope:

- No LoRA, SFT, RL, or DataMind-12K training.
- No DiscoveryBench on the default Colab path.
- No full multi-model sweep.
- No paid API judge by default.

The notebook is designed for one Colab Pro A100 and has hard wall-clock caps
plus per-sample checkpointing, so partial results survive a disconnect.
If time remains, the notebook includes an optional 14B Exp 1 rerun. This adds
a focused 7B-vs-14B comparison without changing the resource story into a
multi-model benchmark sweep.

## Repo Layout

```text
datamind-survey/
+-- experiments/
|   +-- run_exp1_comprehension.py
|   +-- run_exp2_code_analysis.py
|   +-- run_exp3_turn_budget.py
|   +-- utils/
|       +-- data_loader.py
|       +-- datamind_compat.py
|       +-- evaluate.py
+-- notebooks/
|   +-- run_first3_colab.ipynb
+-- results 2/
|   +-- exp1_colab/
|   +-- exp1_14b_colab/
|   +-- exp2_colab/
|   +-- exp3_colab/
+-- download_data.py
+-- requirements.txt
+-- README.md
```

Recreated locally and not committed:

- `Datamind-main/`
- `data/`
- `experiments/results/`
- `.venv/`, `*.zip`, model checkpoints, and vLLM logs

## Quick Start on Colab A100

1. Make sure this GitHub repo is public.
2. Open the notebook:
   <https://colab.research.google.com/github/Razim021/datamind-survey/blob/main/notebooks/run_first3_colab.ipynb>
3. In Colab, choose `Runtime -> Change runtime type -> GPU -> A100`.
4. Run once with `SMOKE_TEST = True`.
5. If the smoke test works, set `SMOKE_TEST = False` and run all cells again.
6. Optional stronger run: execute the `Optional Upgrade: 14B Exp 1 Only` cells.
7. Download `/content/datamind_first3_results.zip` from the final cell.

Colab writes fresh outputs under `experiments/results/`. The completed outputs
from the final run are committed in `results 2/`:

```text
results 2/exp1_colab/summary.json
results 2/exp1_14b_colab/summary.json
results 2/exp2_colab/error_categories.json
results 2/exp3_colab/summary.json
```

Use those JSON summaries when writing the report.

## What Each Experiment Does

### Exp 1: Table-Metadata Ablation

Script: `experiments/run_exp1_comprehension.py`

Same QRData samples, same model, same ReAct loop. Only the prompt changes:

- `without_metadata`: question, background, and filenames only.
- `with_metadata`: filenames plus column names, pandas dtypes, and the first
  three rows for each CSV.

Metrics: answer accuracy and code-error rate.

### Exp 2: Baseline Error Categorization

Script: `experiments/run_exp2_code_analysis.py`

The default notebook does not run extra inference for Exp 2. It reads only the
baseline Exp-1 file `qrdata_wo_info_*.json`, keeps incorrect answers, and
assigns each failure to:

- `code_error`
- `data_understanding`
- `planning_reasoning`

This keeps the experiment cheap and directly tied to the baseline condition.

### Exp 3: Turn-Budget Sweep

Script: `experiments/run_exp3_turn_budget.py`

Same model and QRData setting as the baseline. The only variable is maximum
assistant turns: `2`, `4`, or `6`. This is not the full fine-tuned turn-length
experiment from the paper; it is a lightweight test of whether extra inference
turns alone help under the course-project budget.

### Optional Upgrade: 14B Exp 1

Script: `experiments/run_exp1_comprehension.py`

The notebook can stop the 7B vLLM server, load `Qwen2.5-14B-Instruct`, and
rerun only Exp 1 into `results/exp1_14b_colab`. This is the recommended
upgrade if there is spare Colab time because it directly strengthens the
table-information ablation with a model-scale comparison.

## Manual Commands

From a fresh Colab runtime:

```bash
git clone https://github.com/Razim021/datamind-survey.git
cd datamind-survey
git clone --depth 1 https://github.com/zjunlp/DataMind Datamind-main
pip install -r requirements.txt
pip install -U vllm
python download_data.py --dataset qrdata
```

Start vLLM:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --served-model-name Qwen2.5-7B-Instruct \
  --port 8000 \
  --dtype float16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85
```

Run the experiments from another shell:

```bash
export DATAMIND_ROOT="$(pwd)/Datamind-main"
export DATA_DIR="$(pwd)/data"
export JUDGE_BACKEND=local
cd experiments

python run_exp1_comprehension.py \
  --model_name Qwen2.5-7B-Instruct \
  --data_dir ../data --dataset qrdata --sub_experiment info \
  --n_samples 40 --max_rounds 4 --api_port 8000 \
  --time_budget_s 1800 --judge_backend local \
  --output_dir results/exp1_colab

python run_exp2_code_analysis.py \
  --mode categorize \
  --results_dir results/exp1_colab \
  --file_glob "qrdata_wo_info_*.json" \
  --n_errors 40 --judge_backend local \
  --output_dir results/exp2_colab

python run_exp3_turn_budget.py \
  --model_name Qwen2.5-7B-Instruct \
  --data_dir ../data --dataset qrdata \
  --turn_budgets 2,4,6 --n_samples 25 --api_port 8000 \
  --time_budget_s 1800 --judge_backend local \
  --output_dir results/exp3_colab
```

Optional 14B Exp 1 command after restarting vLLM with
`Qwen/Qwen2.5-14B-Instruct` served as `Qwen2.5-14B-Instruct`:

```bash
python run_exp1_comprehension.py \
  --model_name Qwen2.5-14B-Instruct \
  --data_dir ../data --dataset qrdata --sub_experiment info \
  --n_samples 40 --max_rounds 4 --api_port 8000 \
  --time_budget_s 1800 --judge_backend local \
  --output_dir results/exp1_14b_colab
```

Optional appendix data:

```bash
python download_data.py --dataset discoverybench
```

## Resources

Default run: one Colab Pro A100, Qwen2.5-7B-Instruct served locally with vLLM,
QRData only, and local deterministic judging. Optional upgrade: rerun Exp 1
only with Qwen2.5-14B-Instruct on the same A100. No paid API calls are required.
