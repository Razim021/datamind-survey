# Survey: DataMind for Data Analysis Agents

This repo is the runnable code and report draft for:

**Survey: A Resource-Aware Reproduction of DataMind for Data Analysis Agents**

It uses the public DataMind project as a local dependency, but this repo is not a
copy of DataMind. The class work is in `experiments/`, `notebooks/`, and
`report/`.

Original paper/code:

- Paper: *Why Do Open-Source LLMs Struggle with Data Analysis? A Systematic Empirical Study*
- Upstream code: <https://github.com/zjunlp/DataMind>

## Project Scope

The professor said the proposal was too ambitious. So this version keeps the
project strong but doable on **one Colab Pro A100**.

You will run exactly three experiments:

1. **Exp. 1: Table metadata ablation**
   - Same model, same questions.
   - Compare prompt with only file names vs prompt with column names and sample rows.
   - Script: `experiments/run_exp1_comprehension.py`

2. **Exp. 2: Code behavior and error categories**
   - Measure accuracy and code error rate.
   - Categorize wrong answers as planning, data-understanding, or code errors.
   - Script: `experiments/run_exp2_code_analysis.py`

3. **Exp. 3: Turn-budget study**
   - Same model, same questions.
   - Compare 2 vs 4 vs 6 allowed analysis turns.
   - Script: `experiments/run_exp3_turn_budget.py`

Not included anymore:

- No LoRA fine-tuning.
- No 4xA100 training job.
- No full reproduction of every table in the paper.

That is the point: smaller scope, clearer execution.

## Folder Layout

```text
datamind-survey/
├── experiments/
│   ├── run_exp1_comprehension.py
│   ├── run_exp2_code_analysis.py
│   ├── run_exp3_turn_budget.py
│   └── utils/
├── notebooks/
│   └── run_first3_colab.ipynb
├── report/
│   ├── main.tex
│   └── references.bib
├── download_data.py
├── requirements.txt
├── .gitignore
└── README.md
```

Downloaded folders are intentionally not committed:

- `data/`
- `Datamind-main/`
- `.venv/`
- `results/`

They are recreated when you run the setup.

## Easiest Way: Run on Colab Pro A100

Open the notebook:

[Open in Colab](https://colab.research.google.com/github/Razim021/datamind-survey/blob/main/notebooks/run_first3_colab.ipynb)

Button-by-button:

1. Open the link above.
2. Top menu: click **Runtime**.
3. Click **Change runtime type**.
4. Set **Hardware accelerator** to **GPU**.
5. If Colab shows **GPU type**, choose **A100**.
6. Click **Save**.
7. Run the first cell.
8. Run the GPU check cell. You should see `A100` in `nvidia-smi`.
9. Run the setup cell. It clones this repo, clones DataMind, installs packages,
   and downloads data.
10. Run the vLLM server cell. Wait until it prints `vLLM is ready`.
11. Run Exp. 1.
12. Run Exp. 2.
13. Run Exp. 2b error categorization.
14. Run Exp. 3.
15. Run the summary cell.
16. Run the zip cell.
17. In the left Colab sidebar, click the **folder** icon.
18. Find `datamind_first3_results.zip`.
19. Click the three dots beside it.
20. Click **Download**.

Important: you do not really run the notebook "on your Mac and connect it to
Colab A100." You open the notebook in Colab. Your Mac is for editing and GitHub;
Colab is the A100 machine.

## Recommended Run Size

In the first notebook cell:

```python
N_SAMPLES = 50
MAX_ROUNDS = 6
```

For a quick smoke test:

```python
N_SAMPLES = 5
```

For stronger final numbers:

```python
N_SAMPLES = 100
```

Do not start with all samples at midnight. Run 5 first, then 50.

## Local Mac Setup

Only use this for checking scripts and editing the report. Do not run the model
experiments on your Mac.

```bash
cd ~/Desktop/datamind-survey
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Download local data if needed:

```bash
python download_data.py
```

Clone the upstream DataMind dependency if needed:

```bash
git clone https://github.com/zjunlp/DataMind Datamind-main
```

Set paths:

```bash
export DATAMIND_ROOT="$(pwd)/Datamind-main"
export DATA_DIR="$(pwd)/data"
```

## Manual GPU Commands

If you are not using the notebook, start vLLM first:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --served-model-name Qwen2.5-7B-Instruct \
  --port 8000 \
  --dtype float16 \
  --max-model-len 4096
```

Then open a second terminal:

```bash
cd experiments
```

Exp. 1:

```bash
python run_exp1_comprehension.py \
  --model_name Qwen2.5-7B-Instruct \
  --data_dir ../data \
  --dataset qrdata \
  --sub_experiment info \
  --n_samples 50 \
  --api_port 8000 \
  --max_rounds 6 \
  --judge_backend local \
  --output_dir results/exp1_colab
```

Exp. 2:

```bash
python run_exp2_code_analysis.py \
  --mode evaluate \
  --models Qwen2.5-7B-Instruct \
  --data_dir ../data \
  --dataset qrdata \
  --n_samples 50 \
  --api_port 8000 \
  --max_rounds 6 \
  --judge_backend local \
  --output_dir results/exp2_colab
```

Exp. 2b:

```bash
python run_exp2_code_analysis.py \
  --mode categorize \
  --results_dir results/exp2_colab \
  --n_errors 50 \
  --judge_backend local \
  --output_dir results/exp2_colab
```

Exp. 3:

```bash
python run_exp3_turn_budget.py \
  --model_name Qwen2.5-7B-Instruct \
  --data_dir ../data \
  --dataset qrdata \
  --turn_budgets 2,4,6 \
  --n_samples 50 \
  --api_port 8000 \
  --judge_backend local \
  --output_dir results/exp3_colab
```

## Where Results Appear

After Colab finishes, look at:

```text
experiments/results/exp1_colab/summary.json
experiments/results/exp2_colab/summary.json
experiments/results/exp2_colab/error_categories.json
experiments/results/exp3_colab/summary.json
```

Copy those numbers into `report/main.tex` where it says `TODO`.

## Make the Report PDF in Overleaf

Button-by-button:

1. Go to the ACL template:
   <https://www.overleaf.com/latex/templates/association-for-computational-linguistics-acl-conference/jvxskxpnznfj>
2. Click **Open as Template**.
3. In the left file panel, click **Upload**.
4. Upload `report/main.tex`.
5. Upload `report/references.bib`.
6. If Overleaf asks about replacing `main.tex`, click **Replace**.
7. Top-left: click **Menu**.
8. Set **Compiler** to **pdfLaTeX**.
9. Make sure **Main document** is `main.tex`.
10. Click outside the menu.
11. Click **Recompile**.
12. If citations show as `?`, click **Recompile** one more time.
13. Check the PDF page count. Main content must be 8 pages or less, not counting
    references.
14. Top right beside Recompile: click the **Download PDF** icon.

Before final submission:

1. Replace every `TODO` result in `report/main.tex`.
2. Keep the GitHub link as <https://github.com/Razim021/datamind-survey>.
3. Make sure the report says you used one Colab Pro A100.
4. Submit the PDF on Canvas.
5. Your groupmate submits separately too.

## Final Order

1. Push this repo to GitHub.
2. Open the notebook in Colab.
3. Choose A100.
4. Run `N_SAMPLES = 5`.
5. If that works, run `N_SAMPLES = 50`.
6. Download `datamind_first3_results.zip`.
7. Copy summary numbers into `report/main.tex`.
8. Build PDF in Overleaf.
9. Submit PDF on Canvas.
