# CMTI Freight

CMTI Freight is an exploratory freight-shipment damage-risk simulation. It generates synthetic shipment sensor readings, derives roughness and fatigue indicators, and demonstrates a threshold-based inspection decision. It does not connect to freight systems or use real shipment records.

## What is present

- `cmti_live.py` generates a reproducible in-memory batch of 50 synthetic shipments with trip duration, simulated G-force variability, and peak counts.
- `cmti_proceed.py` derives two features, creates heuristic damage labels and scores, flags shipments above a configurable threshold, and prints classification metrics and per-shipment actions.
- `jounal program.py` generates six analytical figures from placeholder false-positive/false-negative and cost simulations. Its outputs are reproducible and ignored by Git.
- `untitled17.py` is a standalone toy hybrid recommender using hard-coded ratings and item features. It is not connected to the shipment simulation; its relationship to the freight work is unknown.

The code comments mention an LSTM architecture, but no trained or implemented LSTM is present. The CMTI score is a hand-written heuristic with random noise, not a validated predictive model.

## Technology

- Python
- NumPy and pandas for synthetic data and feature calculations
- scikit-learn for classification metrics and cosine similarity
- Matplotlib for simulation figures

Dependencies are listed in `requirements.txt`. No environment variables, credentials, external services, or database are required by the inspected scripts.

## Setup and execution

From the project directory, create and activate a virtual environment, then install the dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run the shipment simulation:

```powershell
python cmti_proceed.py
```

Generate the analytical figures:

```powershell
python "jounal program.py"
```

Run the separate toy recommender example:

```powershell
python untitled17.py
```

`cmti_live.py` is imported by `cmti_proceed.py` and does not run the pipeline by itself.

## Limitations

- Shipment inputs and analytical figures are simulated; no real freight, customer, or sensor data is included.
- Damage labels, risk scores, costs, and thresholds are illustrative heuristics/placeholders. They have not been trained, calibrated, or validated against operational data.
- The code is a small set of scripts rather than a packaged application or deployed service. There are no automated tests or production integrations in this directory.
- `cmti_proceed.py` calculates metrics for the generated batch; its results should not be used for operational decisions.
- The provenance and intended role of the standalone recommender example are unknown.
- Dependency installation was not verified in the available Python 3.14.6 environment: its package index did not provide a binary NumPy wheel, and building from source failed because Ninja/make was unavailable. Use an environment/package index with compatible prebuilt wheels or install the required native build tools.

## Privacy and configuration

Do not add credentials, private datasets, real shipment records, customer/contact details, or local configuration to this repository. Local environment files, datasets, logs, database files, Python caches, and the known generated plots are excluded by `.gitignore`. Python bytecode is excluded because it can retain local filesystem metadata. No `.env` file is needed.
