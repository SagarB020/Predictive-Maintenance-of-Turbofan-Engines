# Predictive maintenance: NASA C-MAPSS RUL

This project trains an XGBoost regressor to estimate remaining useful life (RUL) for C-MAPSS turbofan engines, evaluates by held-out engines, scores the official truncated test trajectories, and serves results through FastAPI and Streamlit.

## Important fixes

- Data files are resolved from `data/raw/`, `Data/raw/`, or the bundled `Data/` directory. This fixes the original scripts' lowercase path assumption and supports all four bundled subsets.
- Validation splits by engine. Cycles from the same engine cannot leak across train and validation. Training-row metrics are not used as generalization results.
- Training, test scoring, API inference, and dashboard use the same causal rolling and difference features.
- Test predictions are made only at each engine's last observed cycle and aligned with the official `RUL_FDxxx.txt` order. The export includes the cycle, predicted RUL, and actual RUL.
- The dashboard reads from the same `Data/predictions/` location the pipeline writes and has predicted/actual and error charts.

## Files

```text
train_xgboost.py             grouped validation and final model training
precompute_predictions.py   official truncated-test scoring and CSV export
src/data_pipeline.py        shared loading, RUL labels, feature engineering
api/main.py                  FastAPI single-snapshot endpoint
dashboard/app.py             Streamlit fleet dashboard
Data/                        bundled C-MAPSS data, FD001 through FD004
models/                      trained model artifact (created by pipeline)
reports/                     validation metrics and predictions (created by pipeline)
```

`train_lstm.py` is an older experimental implementation and is not in the validated production pipeline. Do not compare its previous scores with XGBoost; it randomly split overlapping sequences and saved the last epoch rather than the best checkpoint.

## Run in VS Code (Windows PowerShell)

1. Extract/open the `predictive_maintenance` folder in VS Code. Install the Python and Jupyter VS Code extensions if prompted.
2. Open **Terminal → New Terminal** and create an environment:

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

   If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process Bypass` in that terminal, then activate again. Choose `.venv` with **Python: Select Interpreter** from the command palette.

3. Train, validate, and build the test predictions:

   ```powershell
   python train_xgboost.py --dataset FD001
   python precompute_predictions.py --dataset FD001
   ```

   The scripts print MAE, RMSE, and NASA's asymmetric score. Grouped validation metrics are written to `reports/fd001_metrics.json`; official test-set results are printed and `Data/predictions/fleet_predictions.csv` is created. For FD002–FD004, repeat both commands with the same dataset argument.

4. Inspect results in VS Code: open `reports/fd001_metrics.json` and `Data/predictions/fleet_predictions.csv`. Use the CSV preview to sort engines by predicted or actual RUL. Open `01_predictive_maintenance.ipynb`, select the `.venv` kernel, and run cells top to bottom to display sensor, validation, and official test plots inline.
5. Launch the dashboard from the project terminal:

   ```powershell
   streamlit run dashboard/app.py
   ```

   VS Code's terminal displays a local URL; Ctrl+click it. Adjust the high-risk threshold and review the fleet bars, actual-versus-predicted plot, and per-engine errors. Stop the app with Ctrl+C.

6. Launch the API in another terminal:

   ```powershell
   uvicorn api.main:app --reload
   ```

   Open `http://127.0.0.1:8000/docs`, try `/health`, then submit a valid single-engine reading to `/predict`. This endpoint receives one snapshot; because it has no sensor history, rolling difference and standard-deviation features use single-row defaults. Fleet benchmarking should use the offline pipeline, which has observed history.

## Dataset selection and scores

The bundled `Data/` directory includes train, test, and RUL files for FD001–FD004. FD001 is the default. FD002 and FD004 contain multiple operating regimes; FD003 and FD004 contain multiple fault modes. Run each subset separately; one trained artifact is kept in `models/xgboost_model.pkl` at a time.

RUL training targets and predictions are capped at 125 cycles. MAE/RMSE are measured in cycles. NASA score penalizes overestimating RUL more strongly than underestimating it. Test labels are used only for reporting, never for fitting.

## Docker (optional)

```powershell
docker compose up --build
```

Open `http://localhost:8501` for the dashboard and `http://localhost:8000/docs` for the API. Docker Compose runs training first and gates the API/dashboard on successful artifact creation.

## Limitations

- The single-snapshot API cannot reproduce the full temporal context used for offline engine trajectories. A history-aware endpoint would be needed for equivalent online accuracy.
- The current model is a solid reproducible baseline, not a claim of best possible benchmark accuracy. Compare grouped validation and the official test metrics before changing features or tuning parameters.
- `Data/raw/` in the archive contains only FD001; the path resolver also checks `Data/` for the other three sets.

Dataset citation: A. Saxena and K. Goebel, *Turbofan Engine Degradation Simulation Data Set*, NASA Prognostics Data Repository, 2008.
