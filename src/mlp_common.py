"""Hamilton daily features, residual MLP, and shared evaluation utilities."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data/clean/hamilton_daily_summary.csv"
RESULT_DIR = ROOT / "outputs/mlp_fixed_2016_2018"
TRAIN_START = pd.Timestamp("2016-01-01")
TRAIN_END = pd.Timestamp("2018-12-31")
TEST_START = pd.Timestamp("2019-01-01")
TEST_END = pd.Timestamp("2019-12-31")
LAG_DAYS = (7, 14, 21, 28)
WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
FEATURE_NAMES = ["day_of_year_sin", "day_of_year_cos"] + [
    "weekday_" + name for name in WEEKDAYS
] + ["previous_same_weekday_sessions", "lag7_minus_lag14",
     "lag7_minus_lag21", "lag7_minus_lag28"]
MODEL_CONFIG = {"input_size": 13, "width": 16, "blocks": 1, "dropout": 0.1}


def load_daily(path, end):
    frame = pd.read_csv(path, usecols=["date", "daily_sessions"], parse_dates=["date"])
    if frame.date.isna().any():
        raise ValueError("Input contains an invalid date.")
    frame = frame.loc[frame.date.between(TRAIN_START, end)].copy()
    if frame.date.duplicated().any():
        raise ValueError("Input contains duplicate dates.")
    frame = frame.set_index("date").sort_index()
    if not frame.index.equals(pd.date_range(TRAIN_START, end)):
        raise ValueError("Every date in the requested period must be present.")
    sessions = pd.to_numeric(frame.daily_sessions, errors="raise").astype(float)
    values = sessions.to_numpy()
    if not np.isfinite(values).all() or (values < 0).any() or (values % 1 != 0).any():
        raise ValueError("Daily counts must be finite nonnegative integers without missing values.")
    return sessions


def build_features(sessions, dates):
    dates = pd.DatetimeIndex(dates)
    origins = dates - pd.to_timedelta(dates.dayofweek, unit="D")
    lags = []
    for lag in LAG_DAYS:
        history_dates = dates - pd.Timedelta(days=lag)
        if not (history_dates < origins).all():
            raise ValueError("A historical feature is inside the forecast week.")
        lags.append(sessions.reindex(history_dates).to_numpy(dtype=float))
    lags = np.column_stack(lags)
    if not np.isfinite(lags).all():
        raise ValueError("A required historical count is missing.")
    angle = 2 * np.pi * (dates.dayofyear.to_numpy() - 1) / np.where(dates.is_leap_year, 366, 365)
    matrix = np.column_stack([
        np.sin(angle), np.cos(angle), np.eye(7)[dates.dayofweek],
        lags[:, 0], lags[:, 0] - lags[:, 1],
        lags[:, 0] - lags[:, 2], lags[:, 0] - lags[:, 3],
    ])
    return pd.DataFrame(matrix, index=dates, columns=FEATURE_NAMES)


def fit_scaler(features):
    """Standardize the four history features using only the fitting subset."""
    values = np.asarray(features, dtype=float)
    mean = np.zeros(len(FEATURE_NAMES))
    scale = np.ones(len(FEATURE_NAMES))
    mean[-4:] = values[:, -4:].mean(axis=0)
    scale[-4:] = values[:, -4:].std(axis=0)
    scale[scale == 0] = 1.0
    return mean, scale


def input_tensor(features, mean, scale):
    return torch.tensor((np.asarray(features, dtype=float) - mean) / scale, dtype=torch.float32)


class ResidualBlock(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Linear(width, width), nn.ReLU(), nn.Dropout(dropout), nn.Linear(width, width),
        )

    def forward(self, inputs):
        return torch.relu(inputs + self.branch(inputs))


class ResidualMLP(nn.Module):
    def __init__(self, input_size=13, width=16, blocks=1, dropout=0.1):
        super().__init__()
        self.input_layer = nn.Sequential(nn.Linear(input_size, width), nn.ReLU())
        self.blocks = nn.Sequential(*[ResidualBlock(width, dropout) for _ in range(blocks)])
        self.output_layer = nn.Linear(width, 1)
        self.nonnegative = nn.Softplus()

    def forward(self, inputs):
        hidden = self.blocks(self.input_layer(inputs))
        return self.nonnegative(self.output_layer(hidden)).squeeze(-1)


def predict(model, inputs):
    model.eval()
    with torch.no_grad():
        return model(inputs).cpu().numpy()


def regression_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = predicted - actual
    variance = np.sum((actual - actual.mean()) ** 2)
    return {"mae": float(np.abs(error).mean()), "rmse": float(np.sqrt(np.mean(error ** 2))),
            "r2": float(1 - np.sum(error ** 2) / variance) if variance else None}


def prepare_output_dir(directory):
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError("Outputs already exist; choose a new --output-dir: " + str(directory))
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def plot_comparison(dates, actual, predicted, title, prediction_label, output):
    scores = regression_metrics(actual, predicted)
    fig, ax = plt.subplots(figsize=(16, 5.5), constrained_layout=True)
    ax.plot(dates, actual, color="royalblue", label="Actual sessions", marker="o",
            markersize=2.5, linewidth=1, alpha=0.85)
    ax.plot(dates, predicted, color="firebrick", label=prediction_label, marker="o",
            markersize=2, linewidth=1.2, alpha=0.9)
    ax.set(title=title, xlabel="Date", ylabel="Sessions per day", xlim=(dates.min(), dates.max()))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", ncol=2)
    ax.text(0.01, 0.97, "MAE = {mae:.3f}   RMSE = {rmse:.3f}   R-squared = {r2:.3f}".format(**scores),
            transform=ax.transAxes, va="top", bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none"})
    fig.savefig(output, dpi=180)
    plt.close(fig)
