"""Select MLP settings exclusively within 2016-2018 and refit all training data.

Run: python src/tune_mlp.py
Then: python src/test_mlp.py
All candidates use the same 20 features and chronological validation dates.
2019 targets are not loaded by this script. No data augmentation is used.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

from mlp_common import (
    DATA_PATH, FEATURE_NAMES, RESULT_DIR, TRAIN_START, TRAIN_END, TRAIN_SAMPLE_START,
    build_features, fit_scaler, input_tensor, load_daily, predict,
    prepare_output_dir, regression_metrics,
)
from train_mlp import fit_model


FOLDS = (
    ("2017_H1", "2017-01-02", "2017-07-02"),
    ("2017_H2", "2017-07-03", "2017-12-31"),
    ("2018_H1", "2018-01-01", "2018-07-01"),
    ("2018_H2", "2018-07-02", "2018-12-31"),
)
SEEDS = (42, 2024)
CONFIGURATIONS = (
    {"id": "baseline16", "width": 16, "blocks": 1, "dropout": 0.1, "lr": 0.001, "weight_decay": 0.0},
    {"id": "small8", "width": 8, "blocks": 1, "dropout": 0.1, "lr": 0.001, "weight_decay": 0.01},
    {"id": "regularized16", "width": 16, "blocks": 1, "dropout": 0.2, "lr": 0.001, "weight_decay": 0.01},
    {"id": "faster16", "width": 16, "blocks": 1, "dropout": 0.1, "lr": 0.003, "weight_decay": 0.001},
    {"id": "wide32", "width": 32, "blocks": 1, "dropout": 0.2, "lr": 0.001, "weight_decay": 0.01},
    {"id": "deep16", "width": 16, "blocks": 2, "dropout": 0.2, "lr": 0.001, "weight_decay": 0.01},
    {"id": "faster8", "width": 8, "blocks": 1, "dropout": 0.2, "lr": 0.003, "weight_decay": 0.01},
    {"id": "deep32", "width": 32, "blocks": 2, "dropout": 0.2, "lr": 0.001, "weight_decay": 0.001},
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--max-epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=30)
    args = parser.parse_args()
    if min(args.max_epochs, args.patience) <= 0:
        parser.error("Epochs and patience must be positive.")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    output = prepare_output_dir(args.output_dir)
    sessions = load_daily(args.data, TRAIN_END)
    dates = pd.date_range(TRAIN_SAMPLE_START, TRAIN_END)
    features = build_features(sessions, dates)
    target = sessions.loc[dates]
    fold_scores, predictions, histories, summaries = [], [], [], []

    for config in CONFIGURATIONS:
        config_predictions, best_epochs = [], []
        architecture = {"input_size": len(FEATURE_NAMES), "width": config["width"],
                        "blocks": config["blocks"], "dropout": config["dropout"]}
        for seed in SEEDS:
            for fold_name, start, end in FOLDS:
                start, end = pd.Timestamp(start), pd.Timestamp(end)
                fitting_x = features.loc[features.index < start]
                validation_x = features.loc[start:end]
                if fitting_x.index.max() >= start or start.dayofweek != 0:
                    raise ValueError("A fold overlaps training or starts inside a forecast week.")
                mean, scale = fit_scaler(fitting_x)
                fit_inputs = input_tensor(fitting_x, mean, scale)
                validation_inputs = input_tensor(validation_x, mean, scale)
                fitting_y = torch.tensor(target.loc[fitting_x.index].to_numpy(), dtype=torch.float32)
                validation_y = torch.tensor(target.loc[validation_x.index].to_numpy(), dtype=torch.float32)
                model, history = fit_model(
                    fit_inputs, fitting_y, args.max_epochs, seed, config["lr"], 32,
                    model_config=architecture, weight_decay=config["weight_decay"],
                    validation=(validation_inputs, validation_y), patience=args.patience, verbose=False,
                )
                best_epoch = int(history.loc[history.selected_checkpoint, "epoch"].iloc[0])
                best_epochs.append(best_epoch)
                forecast = predict(model, validation_inputs)
                actual = target.loc[validation_x.index].to_numpy()
                scores = regression_metrics(actual, forecast)
                selected_mse = float(history.loc[history.selected_checkpoint, "validation_mse"].iloc[0])
                if not np.isclose(scores["rmse"] ** 2, selected_mse, atol=2e-6):
                    raise RuntimeError("The restored checkpoint differs from the selected validation epoch.")
                fold_scores.append({
                    "config_id": config["id"], "seed": seed, "fold": fold_name,
                    "fit_start": str(fitting_x.index.min().date()),
                    "fit_end": str(fitting_x.index.max().date()), "fit_samples": len(fitting_x),
                    "validation_start": str(start.date()), "validation_end": str(end.date()),
                    "validation_samples": len(validation_x), "best_epoch": best_epoch,
                    "epochs_run": len(history), "scaler_mean": json.dumps(mean.tolist()),
                    "scaler_scale": json.dumps(scale.tolist()), **scores,
                })
                frame = pd.DataFrame({"config_id": config["id"], "seed": seed, "fold": fold_name,
                                      "date": validation_x.index, "actual_sessions": actual,
                                      "predicted_sessions": forecast})
                predictions.append(frame)
                config_predictions.append(frame)
                histories.append(history.assign(config_id=config["id"], seed=seed, fold=fold_name))
                print("{} seed {} {}: epoch {}, RMSE {:.3f}, R2 {:.3f}".format(
                    config["id"], seed, fold_name, best_epoch, scores["rmse"], scores["r2"]), flush=True)
        pooled = pd.concat(config_predictions, ignore_index=True)
        scores = regression_metrics(pooled.actual_sessions, pooled.predicted_sessions)
        summaries.append({**config, "cv_samples": len(pooled), "final_epochs": int(np.ceil(np.median(best_epochs))),
                          "cv_mae": scores["mae"], "cv_rmse": scores["rmse"], "cv_r2": scores["r2"]})
        # Save completed candidates so an interrupted search leaves reviewable progress.
        pd.DataFrame(summaries).to_csv(output / "cv_summary.csv", index=False)
        pd.DataFrame(fold_scores).to_csv(output / "cv_fold_scores.csv", index=False)
        pd.concat(predictions, ignore_index=True).to_csv(output / "cv_predictions.csv", index=False)
        pd.concat(histories, ignore_index=True).to_csv(output / "cv_training_history.csv", index=False)
        print("Completed {}: pooled CV RMSE {:.3f}, R2 {:.3f}".format(
            config["id"], scores["rmse"], scores["r2"]), flush=True)

    summary = pd.DataFrame(summaries)
    best = summary.sort_values(["cv_rmse", "cv_mae", "id"]).iloc[0]
    summary["selected"] = summary.id == best.id
    summary.to_csv(output / "cv_summary.csv", index=False)
    selection = {
        "method": "Expanding chronological folds; pooled RMSE over both seeds; no 2019 targets",
        "training_period_start": str(TRAIN_START.date()), "training_period_end": str(TRAIN_END.date()),
        "training_sample_start": str(TRAIN_SAMPLE_START.date()), "training_samples": len(dates),
        "feature_names": FEATURE_NAMES, "folds": [list(fold) for fold in FOLDS], "seeds": list(SEEDS),
        "configurations": list(CONFIGURATIONS), "selected_config_id": str(best.id),
        "selected_architecture": {"input_size": len(FEATURE_NAMES), "width": int(best.width),
                                  "blocks": int(best.blocks), "dropout": float(best.dropout)},
        "selected_learning_rate": float(best.lr), "selected_weight_decay": float(best.weight_decay),
        "selected_cv_metrics": {"mae": float(best.cv_mae), "rmse": float(best.cv_rmse), "r2": float(best.cv_r2)},
        "final_epochs": int(best.final_epochs), "final_seed": 42,
        "epoch_rule": "Ceiling of median best epoch over the four folds and two seeds",
        "max_epochs": args.max_epochs, "patience": args.patience,
        "data_augmentation": False, "source_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
    }
    selection_path = output / "selection_report.json"
    selection_path.write_text(json.dumps(selection, indent=2, allow_nan=False), encoding="utf-8")
    print("Selected configuration:", json.dumps(selection, allow_nan=False), flush=True)
    command = [sys.executable, str(Path(__file__).with_name("train_mlp.py")),
               "--data", str(args.data), "--output-dir", str(output / "train"),
               "--epochs", str(int(best.final_epochs)), "--seed", "42",
               "--width", str(int(best.width)), "--blocks", str(int(best.blocks)),
               "--dropout", str(float(best.dropout)), "--learning-rate", str(float(best.lr)),
               "--weight-decay", str(float(best.weight_decay)), "--selection-report", str(selection_path)]
    subprocess.run(command, check=True)
    print("Tuning and final fitting complete. Evaluate the frozen winner with test_mlp.py.", flush=True)


if __name__ == "__main__":
    main()
