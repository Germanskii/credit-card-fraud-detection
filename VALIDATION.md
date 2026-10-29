# Validation

## Isolated recheck: 29 October 2026

A fresh per-project virtual environment was created on Windows with Python
3.12.10. The pinned runtime and development dependencies installed from wheels;
`pip check` passed. The environment's pip was updated to 26.2.2. The final
`pip-audit` scan reported no known vulnerabilities in that installed package
set at the time of the check. This is a dated advisory check, not a security guarantee.

- Six input/threshold regression tests passed; CLI help passed.
- The full CLI ran on the real public teaching-mirror dataset, then reran on a
  local CSV with recorded SHA256. Random Forest selection, threshold
  0.6193435223573353 and test AP 0.7674643890375847 agreed between runs.
- The local rerun produced `summary.json`, validation CSV, test predictions and
  both plots. Per-row data/predictions stay in ignored local directories.
- Notebook schema/syntax passed; all code cells also executed sequentially on
  real data with their original settings using a Python harness, not a Jupyter UI.
- Source/test formatting and focused Ruff syntax/import checks passed.

Aggregate metrics, split sizes and provenance are in [docs/results.json](docs/results.json).
A repeated seeded split checks reproducibility, not generalization to a new
population. No temporal validation, cost-sensitive threshold, drift study,
production deployment or new external test set was evaluated. Remote CI has not
run on these prepared changes.

## Earlier preparation record: 27 September 2026

Validated on 2026-09-27 with Python 3.12 on Linux CPU.

Full validation/test performance on the original benchmark is pending; no synthetic-data metrics are presented as fraud-detection quality.

The source notebook passed schema checks and sequential execution during cleanup. For ML notebooks, external data were replaced only inside the test harness by small explicitly synthetic fixtures, and training was shortened. These checks verify code paths, not scientific performance. Published notebooks retain the full experiment settings and contain no synthetic scores.

The extracted command-line modules are additionally checked for importability, help output and the regression cases included in `tests/`. No training or download occurs on module import.


## Input validation regression coverage

The schema check now rejects fractional labels before converting them to integers, nonnumeric features, missing/infinite values, missing columns and a single-class target. Four additional tests cover these cases and valid binary input. The same validation is used in the notebook and command-line implementation.
