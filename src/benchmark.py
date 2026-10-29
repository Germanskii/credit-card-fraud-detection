"""Compare Random Forest and CatBoost with validation-only model selection and decision-threshold tuning."""

import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    classification_report,
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
)
from catboost import CatBoostClassifier

SEED = 42
DATA_URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
OUTPUT = Path("results/fraud")


def validate_data(data):
    """Reject malformed features and labels before any conversion or splitting."""
    required = {"Class", "Time", "Amount"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if data.empty or not data.columns.is_unique:
        raise ValueError("Data must be nonempty with unique column names.")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in data.dtypes):
        raise ValueError("All features and Class must be numeric.")
    if not np.isfinite(data.to_numpy(dtype=float)).all():
        raise ValueError("Data must not contain missing or infinite values.")
    if set(data["Class"].unique()) != {0, 1}:
        raise ValueError("Class must contain both binary labels 0 and 1 only.")
    return data.drop(columns="Class"), data["Class"].astype(int)


def choose_threshold(y, prob):
    precision, recall, thresholds = precision_recall_curve(y, prob)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.argmax(f1))])


def save_figure():
    figure = plt.gcf()
    count = len(list(OUTPUT.glob("figure_*.png"))) + 1
    figure.savefig(OUTPUT / f"figure_{count:02d}.png", dpi=150, bbox_inches="tight")
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare Random Forest and CatBoost with validation-only model selection and decision-threshold tuning."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Local creditcard.csv; otherwise download the public teaching dataset",
    )
    parser.add_argument("--output", type=Path, default=Path("results/fraud"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    global SEED, OUTPUT
    SEED = args.seed
    OUTPUT = args.output
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.data if args.data is not None else DATA_URL).drop_duplicates()
    X, y = validate_data(data)
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=0.25, stratify=y_dev, random_state=SEED
    )
    print("train/validation/test:", len(y_train), len(y_val), len(y_test))
    print("Fraud rate:", y_train.mean())

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=80,
            max_depth=10,
            min_samples_split=30,
            min_samples_leaf=15,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=500,
            depth=8,
            learning_rate=0.1,
            loss_function="Logloss",
            random_seed=SEED,
            verbose=False,
            auto_class_weights="Balanced",
            allow_writing_files=False,
        ),
    }
    validation = []
    thresholds = {}
    for name, model in models.items():
        started = time.perf_counter()
        if name == "CatBoost":
            model.fit(
                X_train,
                y_train,
                eval_set=(X_val, y_val),
                use_best_model=True,
                early_stopping_rounds=40,
            )
        else:
            model.fit(X_train, y_train)
        probs = model.predict_proba(X_val)[:, 1]
        thresholds[name] = choose_threshold(y_val, probs)
        validation.append(
            {
                "model": name,
                "validation_AP": average_precision_score(y_val, probs),
                "threshold": thresholds[name],
                "train_seconds": time.perf_counter() - started,
            }
        )
    validation = pd.DataFrame(validation).set_index("model")
    print(validation.to_string())
    selected = validation["validation_AP"].idxmax()
    print("Selected by validation AP:", selected)

    prob = models[selected].predict_proba(X_test)[:, 1]
    pred = (prob >= thresholds[selected]).astype(int)
    print(classification_report(y_test, pred, digits=4, zero_division=0))
    test_ap = float(average_precision_score(y_test, prob))
    print("Test Average Precision:", test_ap)
    print("Baseline AP (constant probability):", y_test.mean())
    ConfusionMatrixDisplay.from_predictions(y_test, pred)
    save_figure()
    PrecisionRecallDisplay.from_predictions(y_test, prob)
    save_figure()
    validation.to_csv(OUTPUT / "validation.csv")
    pd.DataFrame({"y_true": y_test.to_numpy(), "probability": prob, "prediction": pred}).to_csv(
        OUTPUT / "test_predictions.csv", index=False
    )
    summary = {
        "seed": SEED,
        "source": "local CSV" if args.data is not None else DATA_URL,
        "rows_after_deduplication": len(data),
        "split_rows": {"train": len(y_train), "validation": len(y_val), "test": len(y_test)},
        "selected_model": selected,
        "selection_metric": "validation average precision",
        "threshold": float(thresholds[selected]),
        "test_average_precision": test_ap,
        "test_prevalence": float(y_test.mean()),
        "test_classification_report": classification_report(
            y_test, pred, output_dict=True, zero_division=0
        ),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
