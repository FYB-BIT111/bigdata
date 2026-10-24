"""Run: python src/train_mlp.py --epochs 40.

Fit all available 2016-2018 samples for a fixed epoch count. tune_mlp.py can
choose settings using training-period validation before this final fit.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mlp_common import (
    DATA_PATH, FEATURE_NAMES, LAG_DAYS, MODEL_CONFIG, RESULT_DIR, TRAIN_END,
    TRAIN_START, TRAIN_SAMPLE_START, ResidualMLP, build_features, fit_scaler, input_tensor,
    load_daily, plot_comparison, predict, prepare_output_dir, regression_metrics,
)
import matplotlib.pyplot as plt


def fit_model(inputs, target, epochs, seed, lr, batch_size, *, model_config=None,
              weight_decay=0.0, validation=None, patience=30, verbose=True):
    torch.manual_seed(seed)
    model = ResidualMLP(**(MODEL_CONFIG if model_config is None else model_config))
    # Begin near the observed training level.
    with torch.no_grad():
        model.output_layer.bias.fill_(float(np.log(np.expm1(float(target.mean())))))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_function = nn.MSELoss()
    loader = DataLoader(TensorDataset(inputs, target), batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    history = []
    best_loss, best_epoch, stale, best_state = float("inf"), 0, 0, None
    for epoch in range(1, epochs + 1):
        model.train()
        error_sum = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = loss_function(model(batch_x), batch_y)
            if not torch.isfinite(loss):
                raise RuntimeError("Training generated a nonfinite loss.")
            loss.backward()
            optimizer.step()
            error_sum += float(loss.detach()) * len(batch_y)
        fitted = predict(model, inputs)
        row = {"epoch": epoch, "optimization_mse_with_dropout": error_sum / len(target),
               "training_mse_eval": float(np.mean((fitted - target.numpy()) ** 2))}
        if validation is not None:
            validation_x, validation_y = validation
            residual = predict(model, validation_x) - validation_y.numpy()
            validation_mse = float(np.mean(residual ** 2))
            if not np.isfinite(validation_mse):
                raise RuntimeError("Validation generated a nonfinite loss.")
            row["validation_mse"] = validation_mse
            row["validation_mae"] = float(np.mean(np.abs(residual)))
            if validation_mse < best_loss - 1e-4:
                best_loss, best_epoch, stale = validation_mse, epoch, 0
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            else:
                stale += 1
        history.append(row)
        if verbose and (epoch == 1 or epoch % 50 == 0 or epoch == epochs):
            print("epoch {}: train MSE {:.4f}".format(epoch, row["training_mse_eval"]), flush=True)
        if validation is not None and stale >= patience:
            break
    history = pd.DataFrame(history)
    if validation is not None:
        model.load_state_dict(best_state)
        history["selected_checkpoint"] = history.epoch == best_epoch
    return model, history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR / "train")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=MODEL_CONFIG["width"])
    parser.add_argument("--blocks", type=int, default=MODEL_CONFIG["blocks"])
    parser.add_argument("--dropout", type=float, default=MODEL_CONFIG["dropout"])
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--selection-report", type=Path,
                        help="Training-period CV report associated with the chosen settings.")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.width, args.blocks) <= 0:
        parser.error("Epochs, batch size, width and block count must be positive.")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("Learning rate must be finite and positive.")
    if not 0 <= args.dropout < 1 or not np.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("Dropout must be in [0, 1); weight decay must be finite and nonnegative.")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    output = prepare_output_dir(args.output_dir)
    sessions = load_daily(args.data, TRAIN_END)
    dates = pd.date_range(TRAIN_SAMPLE_START, TRAIN_END)
    features = build_features(sessions, dates)
    target = torch.tensor(sessions.loc[dates].to_numpy(), dtype=torch.float32)
    print("Training all {} samples from 2016-2018 for {} epochs (no validation split).".format(
        len(dates), args.epochs), flush=True)
    final_mean, final_scale = fit_scaler(features)
    final_inputs = input_tensor(features, final_mean, final_scale)
    model_config = {"input_size": len(FEATURE_NAMES), "width": args.width,
                    "blocks": args.blocks, "dropout": args.dropout}
    model, final_history = fit_model(
        final_inputs, target, args.epochs, args.seed, args.learning_rate, args.batch_size,
        model_config=model_config, weight_decay=args.weight_decay)
    fitted = predict(model, final_inputs)
    year2018 = dates.year == 2018
    report = {
        "architecture": model_config, "feature_names": FEATURE_NAMES,
        "feature_definition": "Year-day sin/cos (2), weekday one-hot (7), holiday indicators (5), lag7 count, same-weekday mean/std over lags 7/14/21/28 (population std), previous complete week mean, difference from preceding complete week mean, difference from preceding four complete weeks mean",
        "holiday_definition": "Major: actual and observed New Year, Independence Day, Thanksgiving, Christmas plus Easter Sunday (not federal); other federal holidays; major-holiday eve and following day; December 24-January 1 inclusive",
        "data_augmentation": False, "optimizer": "Adam", "loss": "MSE",
        "learning_rate": args.learning_rate, "batch_size": args.batch_size, "seed": args.seed,
        "weight_decay": args.weight_decay,
        "torch_version": str(torch.__version__), "device": "cpu",
        "training_sample_start": str(dates.min().date()), "training_sample_end": str(TRAIN_END.date()),
        "training_period_start": str(TRAIN_START.date()),
        "training_range_fixed": True, "validation_used": False, "early_stopping": False,
        "epochs": args.epochs,
        "epoch_selection_rule": "Fixed before final fitting; never selected on 2019",
        "selection_report_path": str(args.selection_report.resolve()) if args.selection_report else None,
        "training_samples": len(dates),
        "history_warmup_days": int((TRAIN_SAMPLE_START - TRAIN_START).days),
        "history_lookback_days": max(LAG_DAYS),
        "scaler_mean": final_mean.tolist(), "scaler_scale": final_scale.tolist(),
        "training_metrics": regression_metrics(target.numpy(), fitted),
        "fit_2018_metrics": regression_metrics(target.numpy()[year2018], fitted[year2018]),
        "source_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "data_path": str(args.data.resolve()),
    }
    torch.save({"state_dict": model.state_dict(), "metadata": report}, output / "mlp_model.pt")
    (output / "training_report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    final_history.to_csv(output / "training_history.csv", index=False)
    pd.DataFrame({"date": dates, "actual_sessions": target.numpy().astype(int),
                  "fitted_sessions": fitted}).to_csv(output / "training_fitted.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    ax.plot(final_history.epoch, final_history.training_mse_eval, color="firebrick", label="Full 2016-2018 fit")
    ax.set(title="Training loss: fixed 2016-2018 data", xlabel="Epoch", ylabel="MSE (sessions squared)")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.savefig(output / "training_curves.png", dpi=180)
    plt.close(fig)
    plot_comparison(dates[year2018], target.numpy()[year2018], fitted[year2018],
                    "Training-set fit: Hamilton, 2018 (residual MLP)",
                    "Fitted sessions (in-sample)", output / "fit_2018.png")
    print(json.dumps(report, indent=2, allow_nan=False))
    print("Training outputs:", output.resolve())


if __name__ == "__main__":
    main()
