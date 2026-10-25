"""Score each test engine at its last observed cycle and save benchmark results."""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.data_pipeline import dataset_path, engineer_features, load_cycles, nasa_score


PROJECT_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="FD001", choices=["FD001", "FD002", "FD003", "FD004"])
    args = parser.parse_args()
    dataset = args.dataset.upper()
    test = engineer_features(load_cycles(dataset_path(PROJECT_DIR, f"test_{dataset}.txt")))
    rul_path = dataset_path(PROJECT_DIR, f"RUL_{dataset}.txt")
    actual = pd.read_csv(rul_path, sep=r"\s+", header=None).iloc[:, 0].to_numpy()
    last = test.groupby("engine_id", as_index=False, sort=True).tail(1).sort_values("engine_id")
    if len(actual) != len(last):
        raise ValueError(f"{len(last)} test engines but {len(actual)} RUL values in {rul_path}")

    saved = joblib.load(PROJECT_DIR / "models" / "xgboost_model.pkl")
    model = saved["model"] if isinstance(saved, dict) else saved
    columns = saved.get("feature_columns") if isinstance(saved, dict) else None
    if columns is None:
        columns = [c for c in last.columns if c not in {"engine_id", "cycle", "RUL"}]
    predicted = np.clip(model.predict(last[columns]), 0, 125)
    output = pd.DataFrame({
        "engine_id": last["engine_id"].to_numpy(), "cycle": last["cycle"].to_numpy(),
        "predicted_rul": predicted, "actual_rul": actual,
    })
    out_dir = PROJECT_DIR / "Data" / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fleet_predictions.csv"
    output.to_csv(out_path, index=False)
    metrics = {
        "dataset": dataset,
        "test_engines": len(output),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "nasa_score": nasa_score(actual, predicted),
    }
    print(pd.Series(metrics).to_string())
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
