# To share with students

[01_baseline_fast_model.ipynb](https://colab.research.google.com/github/alxefremov/esmt-workshop/blob/main/notebooks/01_baseline_fast_model.ipynb)

[02_prompt_tuning.ipynb](https://colab.research.google.com/github/alxefremov/esmt-workshop/blob/main/notebooks/02_prompt_tuning.ipynb)

[03_advanced_model.ipynb](https://colab.research.google.com/github/alxefremov/esmt-workshop/blob/main/notebooks/03_advanced_model.ipynb)

[04_two_stage_with_kb.ipynb](https://colab.research.google.com/github/alxefremov/esmt-workshop/blob/main/notebooks/04_two_stage_with_kb.ipynb)

[05_input_guardrails.ipynb](https://colab.research.google.com/github/alxefremov/esmt-workshop/blob/main/notebooks/05_input_guardrails.ipynb)

[06_final_submission_and_validation.ipynb](https://colab.research.google.com/github/alxefremov/esmt-workshop/blob/main/notebooks/06_final_submission_and_validation.ipynb)


# ESMT Workshop: LLM for AML Address Structuring

## Business Context
Starting in **November 2026**, SWIFT AML workflows no longer accept unstructured postal addresses. Address data must be mapped into structured fields.

This workshop demonstrates how teams iteratively improve LLM performance on a real business problem where a single basic prompt is not enough.

## Learning Path
1. Baseline with a fast, low-cost model.
2. Prompt tuning on the same model.
3. Advanced model comparison.
4. Two-stage pipeline (country detection + knowledge base retrieval).
5. Input guardrails.
6. Final submission and validation.

## Repository Structure

### Core Docs and Config
- `README.md`: full workshop documentation.
- `requirements.txt`: dependencies for local and Colab runs.
- `.env.example`: environment variables for proxy endpoint and student email.

### Input Data
- `data/input/address_formats.csv`: country-specific address format knowledge base.
- `data/input/reference_address_cropped_Unstructured_col_100.xlsx`: labeled workshop source dataset.
- `data/input/reference_address_cropped_Unstructured_col.xlsx`: small labeled subset.
- `data/input/all_address.xlsx`: larger source dataset.

### Generated Workshop Data (`dev/test` only)
- `data/workshop/reference_100.csv`: normalized CSV source used by the workshop.
- `data/workshop/dev_labeled.csv`: development split for iterative experiments.
- `data/workshop/test_labeled.csv`: labeled test split (typically instructor-side).
- `data/workshop/test_unlabeled.csv`: unlabeled test split for team submission.

### Notebooks (Main Workshop Flow)
- `notebooks/00_admin_student_registration.ipynb` (organizer/admin notebook for `/register`, `/user/{email}`, optional delete; includes health + admin precheck and one-call register+verify pipeline)
- `notebooks/00_setup_and_data.ipynb`
- `notebooks/01_baseline_fast_model.ipynb`
- `notebooks/02_prompt_tuning.ipynb`
- `notebooks/03_advanced_model.ipynb`
- `notebooks/04_two_stage_with_kb.ipynb`
- `notebooks/05_input_guardrails.ipynb`
- `notebooks/06_final_submission_and_validation.ipynb`

`00_setup_and_data.ipynb` includes a parameter playground for:
- generation controls (`temperature`, `top_p`, `top_k`),
- runtime comparison,
- repeatability checks across repeated runs.

All workshop notebooks are documented for reuse:
- Markdown explanations before execution blocks.
- Inline code comments for reusable logic.
- Direct prompt editing inside notebook cells (`PROMPT_TEMPLATE`).
- Run-by-run logging with a history summary table at the end of each notebook.

### Prompt Templates in Code
- `src/esmt_workshop/prompts.py` contains editable string variables:
- `BASELINE_PROMPT_TEMPLATE`
- `TUNED_PROMPT_TEMPLATE`
- `TWO_STAGE_KB_PROMPT_TEMPLATE`
- Each notebook also exposes editable `PROMPT_TEMPLATE` cells for direct prompt tuning.

### Reusable Python Module (`src/esmt_workshop`)
- `src/esmt_workshop/api_client.py`: proxy API client (email-based access, no API key usage).
- `src/esmt_workshop/student_api.py`: student-facing LLM-call functions (`call_llm`, `call_llm_batch`) and stage wrappers.
- `src/esmt_workshop/student_utils.py`: service helpers (JSON parser, structured-address parser, country parser).
- `src/esmt_workshop/pipeline.py`: stage pipelines (`baseline`, `prompt_tuned`, `advanced`, `two_stage`).
- `src/esmt_workshop/guardrails.py`: input filters.
- `src/esmt_workshop/kb.py`: KB lookup utilities.
- `src/esmt_workshop/prompts.py`: prompt builders and template rendering.
- `src/esmt_workshop/parsing.py`: robust response parsing.
- `src/esmt_workshop/evaluation.py`: metrics and report generation.
- `src/esmt_workshop/constants.py`, `src/esmt_workshop/utils.py`, `src/esmt_workshop/__init__.py`.

### Scripts
- `scripts/prepare_workshop_data.py`: create deterministic `dev/test` artifacts.
- `scripts/run_stage.py`: run one stage end-to-end from CLI.
- `scripts/validate_submission.py`: compare predictions with labeled references.

### Outputs
- `outputs/`: generated predictions and validation reports.
- `outputs/admin/`: student registration run logs from admin notebook.

## Setup
```bash
pip install -r requirements.txt
python scripts/prepare_workshop_data.py --output-dir data/workshop --dev-size 0.7
```

## Google Colab Quick Start
```python
!git clone <YOUR_REPOSITORY_URL>
%cd ESMT_Workshop
%pip install -r requirements.txt
```

Set environment variables:
```python
import os
os.environ['WORKSHOP_API_BASE_URL'] = 'https://gemini-workshop-gateway-395622257429.europe-west4.run.app'
os.environ['WORKSHOP_API_ENDPOINT'] = '/chat'
os.environ['WORKSHOP_EMAIL'] = 'student@esmt.edu'
os.environ['WORKSHOP_USE_TOKEN_AUTH'] = '0'  # set to '1' if gateway requires /token + Bearer auth
os.environ['WORKSHOP_BASELINE_MODEL'] = 'gemini-2.5-flash'
os.environ['WORKSHOP_ADVANCED_MODEL'] = 'gemini-2.5-pro'
os.environ['WORKSHOP_MODEL_CATALOG'] = 'gemini-2.5-flash-lite,gemini-2.5-flash,gemini-2.5-pro,gemini-3-flash-preview,gemini-3-pro-preview,gemini-2.5-flash-preview-09-2025,gemini-2.5-flash-lite-preview-09-2025,gemini-2.0-flash-001,gemini-2.0-flash-lite-001'
os.environ['WORKSHOP_ADMIN_EMAIL'] = 'btc.esmt.workshop@gmail.com'
```

## Proxy API Contract (Email-Based Access)
`WorkshopApiClient` sends `POST` to:
- `WORKSHOP_API_BASE_URL + WORKSHOP_API_ENDPOINT`

Request payload fields:
- `email`
- `model`
- `messages`
- `stream` (set to `false` in this project)
- `temperature`
- `top_p`
- `top_k`

No API key is used in this codebase. Access control is expected to be enforced by the proxy using student email.

Direct model IDs are used in notebooks and scripts (no model aliases).
Stage defaults:
- baseline/prompt_tuned: `WORKSHOP_BASELINE_MODEL` (default `gemini-2.5-flash`)
- advanced/two_stage: `WORKSHOP_ADVANCED_MODEL` (default `gemini-2.5-pro`)

Student model picker list:
- From `WORKSHOP_MODEL_CATALOG` if set.
- Otherwise defaults to: `gemini-2.5-flash-lite`, `gemini-2.5-flash`, `gemini-2.5-pro`, `gemini-3-flash-preview`, `gemini-3-pro-preview`, `gemini-2.5-flash-preview-09-2025`, `gemini-2.5-flash-lite-preview-09-2025`, `gemini-2.0-flash-001`, `gemini-2.0-flash-lite-001`.

To list Gemini models visible in your GCP project:
```bash
gcloud ai model-garden models list \
  --model-filter=gemini \
  --billing-project=<YOUR_PROJECT_ID> \
  --project=<YOUR_PROJECT_ID> \
  --format='table(name,version_id)'
```

## Admin API Authorization (Registration Endpoints)
For `/register`, `GET /user/{email}`, and `DELETE /user/{email}`, the service requires an admin actor provided via one of:
- `Authorization: Bearer <token>`
- `X-Admin-Email: <admin_email>`

Current default admin email in service config:
- `btc.esmt.workshop@gmail.com`

Admin notebook default behavior:
- Reads `WORKSHOP_ADMIN_EMAIL` (default `btc.esmt.workshop@gmail.com`) and always sends it as `X-Admin-Email`.
- Optionally adds `Authorization: Bearer ...` if `WORKSHOP_ADMIN_BEARER_TOKEN` is set.
- Runs a non-destructive admin probe (`POST /register` with `{"users":[]}`) before any real registration call.

If you get `admin_email_required`:
- Ensure Step 3 and Step 4 were executed in the current kernel session.
- Confirm `WORKSHOP_ADMIN_EMAIL` is not empty.
- Re-run Step 5 and verify admin probe returns status `200`.

## Student Inference Functions (No Client Setup in Notebooks)
Students can call the LLM directly and then apply parsing utilities, or use ready stage wrappers.

Direct LLM call:
```python
from esmt_workshop.student_api import call_llm
from esmt_workshop.student_utils import parse_llm_json

raw = call_llm(
    "Extract structured fields and return JSON for: 1600 Pennsylvania Ave NW, Washington, DC 20500, USA",
    email="student@esmt.edu",
    model="gemini-2.5-flash",
)
parsed = parse_llm_json(raw)
```

Batch LLM call:
```python
from esmt_workshop.student_api import call_llm_batch

batch = call_llm_batch(
    [
        "Extract JSON for: 1600 Pennsylvania Ave NW, Washington, DC 20500, USA",
        "Extract JSON for: Royal Opera House, Bow St, Covent Garden, London WC2E 9DD, United Kingdom",
    ],
    email="student@esmt.edu",
    model="gemini-2.5-flash",
)
```

Stage wrappers:
```python
from esmt_workshop.student_api import process_single_address, process_batch_addresses
```

Single address:
```python
from esmt_workshop.student_api import process_single_address

result = process_single_address(
    "1600 Pennsylvania Ave NW, Washington, DC 20500, USA",
    email="student@esmt.edu",
    stage="baseline",
    model="gemini-2.5-flash",
)
```

Batch addresses:
```python
from esmt_workshop.student_api import process_batch_addresses

pred_df = process_batch_addresses(
    [
        "1600 Pennsylvania Ave NW, Washington, DC 20500, USA",
        "Royal Opera House, Bow St, Covent Garden, London WC2E 9DD, United Kingdom",
    ],
    email="student@esmt.edu",
    stage="two_stage",
    model="gemini-2.5-pro",
    country_model="gemini-2.5-flash",
)
```

Unified helper (string -> dict, list/DataFrame -> DataFrame):
```python
from esmt_workshop.student_api import process_addresses
```

Common generation controls are available in all wrappers and all stages:
- `temperature`
- `top_p`
- `top_k`
- `max_tokens`

Prompt can be passed directly from notebook cells:
- `custom_prompt_template` (string from `PROMPT_TEMPLATE` cell).

## Stage Execution Example (CLI)

Baseline:
```bash
python scripts/run_stage.py \
  --input-csv data/workshop/dev_labeled.csv \
  --output-csv outputs/dev_baseline.csv \
  --stage baseline \
  --model gemini-2.5-flash \
  --temperature 0.0 --top-p 1.0 --top-k 40
```

Two-stage + KB + guardrails:
```bash
python scripts/run_stage.py \
  --input-csv data/workshop/dev_labeled.csv \
  --output-csv outputs/dev_two_stage.csv \
  --stage two_stage \
  --model gemini-2.5-pro \
  --country-model gemini-2.5-flash \
  --kb-csv data/input/address_formats.csv \
  --use-guardrails \
  --temperature 0.0 --top-p 0.9 --top-k 40
```

Offline smoke test:
```bash
python scripts/run_stage.py \
  --input-csv data/workshop/dev_labeled.csv \
  --output-csv outputs/dev_mock.csv \
  --stage baseline \
  --mock-mode
```

## Validation and Metrics
Scored fields:
- `Town Name`
- `Postal Code`
- `Country Code (2 characters)`

Reported metrics:
- `micro_accuracy`
- `row_exact_match`
- mismatch table per field

CLI validation:
```bash
python scripts/validate_submission.py \
  --predictions outputs/final_submission.csv \
  --ground-truth data/workshop/test_labeled.csv \
  --report-dir outputs/final_validation_report
```

Generated report artifacts:
- `summary.json`
- `field_metrics.csv`
- `mismatches.csv`
- `joined_predictions_vs_truth.csv`

## Prompt Run Logging and Comparison
Every notebook stage writes one run log after each execution using `log_experiment_run`.

History artifacts:
- `outputs/history/prompt_runs.csv`: append-only run table for comparisons.
- `outputs/history/prompts/*.txt`: saved prompt snapshots by run.
- `outputs/history/runs/*.json`: per-run metadata dump.

Notebook summary cells load history with `load_experiment_history` so teams can compare:
- prompt variants,
- generation parameters,
- stage/model settings,
- runtime and quality metrics.

Each notebook shows:
- raw run history (latest runs),
- aggregated comparison table grouped by prompt + pipeline parameters.

## Final Team Deliverables
1. `outputs/final_submission.csv`
2. Best prompt text (copied from notebook `PROMPT_TEMPLATE` cell used for the winning run)
3. Short report with architecture, parameters, residual errors, and tradeoffs
