#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-FD001}"
python train_xgboost.py --dataset "$DATASET"
python precompute_predictions.py --dataset "$DATASET"
echo "Pipeline completed for $DATASET. Start dashboard with: streamlit run dashboard/app.py"
