"""Train and assess an engine-grouped XGBoost RUL pipeline."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

from src.data_pipeline import (
    add_rul_labels, dataset_path, engineer_features, feature_columns, load_cycles,
    nasa_score,
)


PROJECT_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="FD001", choices=["FD001", "FD002", "FD003", "FD004"])
    args = parser.parse_args()
    dataset = args.dataset.upper()
    data = add_rul_labels(load_cycles(dataset_path(PROJECT_DIR, f"train_{dataset}.txt")))
    data = engineer_features(data)
    columns = feature_columns(data)
    X, y, groups = data[columns], data["RUL"], data["engine_id"]

    # Hold out whole engines so no overlapping cycles from one trajectory appear in both sets.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(X, y, groups))
    params = dict(
        objective="reg:squarederror", n_estimators=700, max_depth=4,
        learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=5, reg_lambda=5.0, random_state=42, n_jobs=-1,
    )
    validation_model = xgb.XGBRegressor(**params)
    validation_model.fit(X.iloc[train_idx], y.iloc[train_idx])
    actual = y.iloc[val_idx].to_numpy()
    predicted = np.clip(validation_model.predict(X.iloc[val_idx]), 0, 125)
    metrics = {
        "dataset": dataset,
        "validation": "20% held-out engines; all cycles; capped RUL labels",
        "validation_engines": int(groups.iloc[val_idx].nunique()),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
        "nasa_score": nasa_score(actual, predicted),
    }

    # Refit the deployable model on all run-to-failure training trajectories.
    model = xgb.XGBRegressor(**params)
    model.fit(X, y)
    model_dir = PROJECT_DIR / "models"
    report_dir = PROJECT_DIR / "reports"
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_columns": columns, "dataset": dataset},
                model_dir / "xgboost_model.pkl")
    pd.DataFrame({"actual_rul": actual, "predicted_rul": predicted,
                  "engine_id": groups.iloc[val_idx].to_numpy(),
                  "cycle": data.iloc[val_idx]["cycle"].to_numpy()}).to_csv(
                      report_dir / f"{dataset.lower()}_validation_predictions.csv", index=False)
    (report_dir / f"{dataset.lower()}_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved model: {model_dir / 'xgboost_model.pkl'}")
    print(f"Saved validation report: {report_dir / f'{dataset.lower()}_metrics.json'}")


if __name__ == "__main__":
    main()
