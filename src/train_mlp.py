"""Run: python src/train_mlp.py --epochs 40.

Fit all available 2016-2018 samples for a fixed epoch count. No validation split,
early stopping, range selection, data augmentation or 2019 model selection.
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
    TRAIN_START, ResidualMLP, build_features, fit_scaler, input_tensor,
    load_daily, plot_comparison, predict, prepare_output_dir, regression_metrics,
)
import matplotlib.pyplot as plt


def fit_model(inputs, target, epochs, seed, lr, batch_size):
    torch.manual_seed(seed)
    model = ResidualMLP(**MODEL_CONFIG)
    # Begin near the observed training level.
    with torch.no_grad():
        model.output_layer.bias.fill_(float(np.log(np.expm1(float(target.mean())))))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.MSELoss()
    loader = DataLoader(TensorDataset(inputs, target), batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    history = []
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
        history.append(row)
        if epoch == 1 or epoch % 50 == 0 or epoch == epochs:
            print("epoch {}: train MSE {:.4f}".format(epoch, row["training_mse_eval"]), flush=True)
    return model, pd.DataFrame(history)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR / "train")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size) <= 0:
        parser.error("Epochs and batch size must be positive.")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("Learning rate must be finite and positive.")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    output = prepare_output_dir(args.output_dir)
    sessions = load_daily(args.data, TRAIN_END)
    dates = pd.date_range(TRAIN_START + pd.Timedelta(days=max(LAG_DAYS)), TRAIN_END)
    features = build_features(sessions, dates)
    target = torch.tensor(sessions.loc[dates].to_numpy(), dtype=torch.float32)
    print("Training all {} samples from 2016-2018 for {} epochs (no validation split).".format(
        len(dates), args.epochs), flush=True)
    final_mean, final_scale = fit_scaler(features)
    final_inputs = input_tensor(features, final_mean, final_scale)
    model, final_history = fit_model(
        final_inputs, target, args.epochs, args.seed, args.learning_rate, args.batch_size)
    fitted = predict(model, final_inputs)
    year2018 = dates.year == 2018
    report = {
        "architecture": MODEL_CONFIG, "feature_names": FEATURE_NAMES,
        "feature_definition": "Year-day sin/cos, weekday one-hot, lag 7 count, lag7-lag14, lag7-lag21, lag7-lag28 (signed differences)",
        "data_augmentation": False, "optimizer": "Adam", "loss": "MSE",
        "learning_rate": args.learning_rate, "batch_size": args.batch_size, "seed": args.seed,
        "torch_version": str(torch.__version__), "device": "cpu",
        "training_sample_start": str(dates.min().date()), "training_sample_end": str(TRAIN_END.date()),
        "training_period_start": str(TRAIN_START.date()),
        "training_range_fixed": True, "validation_used": False, "early_stopping": False,
        "epochs": args.epochs,
        "epoch_selection_rule": "Fixed before this run; default 40 carried over from previous configuration",
        "training_samples": len(dates), "history_warmup_days": 28,
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
