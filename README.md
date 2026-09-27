# Credit Card Fraud Detection

Compare Random Forest and CatBoost with validation-only model selection and decision-threshold tuning.

## Experiment

The pipeline removes exact duplicate rows, creates stratified 60/20/20 train/validation/test splits, then compares Random Forest with CatBoost. CatBoost early stopping, model selection by Average Precision, and the F1 decision threshold all use validation data. Only the selected model is evaluated on test data. A constant-probability baseline has Average Precision equal to test prevalence.

Outputs include validation scores, test probabilities and predictions, a confusion matrix, and a precision–recall plot. The CSV must contain numeric features plus `Class`, `Time` and `Amount`, with binary labels and no missing values.

Random stratification is an educational evaluation protocol, not a deployment simulation. Future-transaction evaluation needs a temporal split and drift analysis. Maximizing F1 does not account for the monetary cost of different mistakes. This code is an experiment, not an operational fraud service.

## Quick start

Python 3.12 was used for local validation. Run commands from the repository root.

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m src.benchmark --data data/creditcard.csv
```

The recorded dependency versions were tested on Linux CPU.

```bash
python -m src.benchmark --help
```

The cleaned Colab/Jupyter experiment is in [`notebooks/experiment.ipynb`](notebooks/experiment.ipynb). To open it locally, install Jupyter separately (`python -m pip install jupyterlab`) and run `jupyter lab`. Command-line runs save figures instead of requiring an interactive window.

## Data

Supply a local `creditcard.csv`. Without `--data`, the script uses the public TensorFlow teaching mirror specified in the source. The underlying benchmark is Credit Card Fraud Detection (ULB/Worldline). No transaction rows or personal data are distributed in this repository. Obtain the data separately and comply with its terms.

## Validation and results

Full validation/test performance on the original benchmark is pending; no synthetic-data metrics are presented as fraud-detection quality. See [VALIDATION.md](VALIDATION.md) for exactly what was checked. No historical notebook output is used as evidence for the corrected implementation.

```bash
python -m unittest discover -s tests -v
```

## Repository layout

- `src/`: importable implementation and command-line entry points.
- `notebooks/`: cleaned experiment notebook; original explanatory notes are in Russian.
- `tests/`: focused regression checks.
- `requirements.txt`: direct dependency versions used during validation.
- `DATA.md`: data access and redistribution notes.

This project was developed from a university Colab experiment and subsequently cleaned up for reproducibility. Generated data, trained weights and local paths are excluded from version control.
