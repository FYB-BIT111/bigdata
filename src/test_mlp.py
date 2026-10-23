"""Run: python src/test_mlp.py. Evaluate fixed weights on all 365 days of 2019.

Only observed history inputs refresh each week. No refitting, augmentation,
dropout, output rounding or clipping is performed during evaluation.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mlp_common import (
    DATA_PATH, FEATURE_NAMES, RESULT_DIR, TEST_START, TEST_END, TRAIN_END,
    ResidualMLP, build_features, input_tensor, load_daily, plot_comparison,
    predict, prepare_output_dir, regression_metrics,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--model", type=Path, default=RESULT_DIR / "train/mlp_model.pt")
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR / "test")
    args = parser.parse_args()
    torch.set_num_threads(1)
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    metadata = checkpoint["metadata"]
    if metadata["feature_names"] != FEATURE_NAMES:
        raise ValueError("The model uses a different feature definition.")
    if pd.Timestamp(metadata["training_sample_end"]) > TRAIN_END:
        raise ValueError("The model was fitted on test-period targets.")
    if hashlib.sha256(args.data.read_bytes()).hexdigest() != metadata["source_sha256"]:
        raise ValueError("Source data differs from the file used for training.")
    sessions = load_daily(args.data, TEST_END)
    dates = pd.date_range(TEST_START, TEST_END)
    features = build_features(sessions, dates)
    model = ResidualMLP(**metadata["architecture"])
    model.load_state_dict(checkpoint["state_dict"])
    inputs = input_tensor(features, np.array(metadata["scaler_mean"]), np.array(metadata["scaler_scale"]))
    predicted = predict(model, inputs)
    actual = sessions.loc[dates].to_numpy()
    if not np.isfinite(predicted).all() or (predicted < 0).any():
        raise RuntimeError("The model produced invalid predictions.")
    week_starts = dates - pd.to_timedelta(dates.dayofweek, unit="D")
    origins = pd.DatetimeIndex([max(date, TEST_START) for date in week_starts])
    if not (pd.Timestamp(metadata["training_sample_end"]) < origins.min()):
        raise ValueError("The model was fitted after the first forecast origin.")
    # Mean(y7, y14, y21, y28) = y7 - (delta14 + delta21 + delta28) / 4.
    four_week_mean = (features["previous_same_weekday_sessions"]
                      - features[FEATURE_NAMES[-3:]].sum(axis=1) / 4)
    results = pd.DataFrame({
        "date": dates, "weekday": dates.day_name(), "forecast_origin": origins,
        "actual_sessions": actual.astype(int), "predicted_sessions": predicted,
        "error_prediction_minus_actual": predicted - actual,
        "previous_week_baseline": features["previous_same_weekday_sessions"].to_numpy(),
        "four_week_mean_baseline": four_week_mean.to_numpy(),
    })
    for name in FEATURE_NAMES[-4:]:
        results[name] = features[name].to_numpy()
    report = {
        "test_start": str(TEST_START.date()), "test_end": str(TEST_END.date()),
        "test_samples": len(dates), "forecast_batches": int(len(origins.unique())),
        "first_batch_days": int((origins == origins.min()).sum()),
        "last_batch_days": int((origins == origins.max()).sum()),
        "model_updated_during_test": False, "dropout_enabled": False, "data_augmentation": False,
        "negative_prediction_count": int((predicted < 0).sum()),
        "training_start": metadata["training_period_start"],
        "training_samples": metadata["training_samples"],
        "mlp": regression_metrics(actual, predicted),
        "previous_week_baseline": regression_metrics(actual, results.previous_week_baseline),
        "four_week_mean_baseline": regression_metrics(actual, results.four_week_mean_baseline),
        "model_path": str(args.model.resolve()),
        "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
    }
    output = prepare_output_dir(args.output_dir)
    results.to_csv(output / "predictions.csv", index=False, date_format="%Y-%m-%d")
    (output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    plot_comparison(dates, actual, predicted, "Daily charging sessions: Hamilton, 2019 (fixed residual MLP)",
                    "Predicted sessions", output / "test_predictions.png")
    print(json.dumps(report, indent=2, allow_nan=False))
    print("Test outputs:", output.resolve())


if __name__ == "__main__":
    main()
