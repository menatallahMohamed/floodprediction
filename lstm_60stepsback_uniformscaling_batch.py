from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import stats
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential


SEED = 420
np.random.seed(SEED)
tf.random.set_seed(SEED)


MODEL_CONFIG = {
    "lstm_units": [64, 32],
    "dropout_rate": 0.2,
    "dense_units": 16,
    "dense_activation": "relu",
    "optimizer": "adam",
    "loss_name": "huber",
    "metrics": ["mae"],
}

TRAINING_CONFIG = {
    "seq_length": 60,
    "max_gap_min": 5,
    "epochs": 5,
    "batch_size": 128,
    "test_fraction": 0.20,
}

OUTPUT_CONFIG = {
    "output_dir": Path("analysis_outputs") / "lstm_60stepsback_uniformscaling_batch",
}

PROGRESS_FILES = {
    "completed": "completed_runs_latest.csv",
    "skipped": "skipped_runs_latest.csv",
}


def resolve_main_dataset_path() -> Path:
    candidate_paths = [
        Path("..")
        / "floodprediction"
        / "Flood-Prevention-Using-River-Level-Prediction"
        / "datasets"
        / "processed"
        / "weather_rainfall_riverlevel_20212025"
        / "v20260309_02"
        / "weatherrainfallrivermaindataset.parquet",
        Path("datasets")
        / "processed"
        / "weather_rainfall_riverlevel_20212025"
        / "v20260309_02"
        / "weatherrainfallrivermaindataset.parquet",
    ]

    for candidate_path in candidate_paths:
        if candidate_path.exists():
            return candidate_path

    searched_paths = "\n".join(str(path.resolve()) for path in candidate_paths)
    raise FileNotFoundError(
        "Could not find weatherrainfallrivermaindataset.parquet. Searched:\n"
        f"{searched_paths}"
    )


def load_and_preprocess_data(
    file_path: Path,
    selected_columns: list[str],
    locations: list[str] | None = None,
) -> pd.DataFrame:
    rename_map = {
        "timestamp": "Timestamp",
        "location": "river_location",
        "windspeed": "Windspeed",
        "humidity": "Humidity",
        "temperature": "Temperature",
        "dewpoint": "Dewpoint",
        "pressure": "Pressure",
        "rainfall_reading": "Rainfall",
        "wind_direction": "Wind direction",
        "river_level": "Level",
    }

    filters = [("location", "in", locations)] if locations else None
    data = pd.read_parquet(file_path, filters=filters).rename(columns=rename_map)

    wind_direction_radians = np.deg2rad(data["Wind direction"])
    data["Wind direction sin"] = np.sin(wind_direction_radians)
    data["Wind direction cos"] = np.cos(wind_direction_radians)
    data.loc[data["Wind direction"].isna(), ["Wind direction sin", "Wind direction cos"]] = np.nan

    missing_columns = [column for column in selected_columns if column not in data.columns]
    if missing_columns:
        raise KeyError(f"Missing expected columns: {missing_columns}")

    data = data.loc[:, selected_columns].copy()
    data["Timestamp"] = pd.to_datetime(data["Timestamp"])
    data = data.dropna(subset=["Timestamp", "river_location", "Level"])
    data["Year"] = data["Timestamp"].dt.year
    data = data.sort_values(["river_location", "Timestamp"]).reset_index(drop=True)
    return data


def split_features_target(
    data: pd.DataFrame,
    input_feature_columns: list[str],
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    X = data.loc[:, input_feature_columns].copy()
    y = data[target_column].copy()
    return X, y


def split_train_test_chronological(
    X: pd.DataFrame,
    y: pd.Series,
    timestamps: np.ndarray,
    test_frac: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, np.ndarray, np.ndarray]:
    n_rows = len(X)
    test_start = int(n_rows * (1 - test_frac))

    X_train, y_train = X.iloc[:test_start].copy(), y.iloc[:test_start].copy()
    X_test, y_test = X.iloc[test_start:].copy(), y.iloc[test_start:].copy()

    ts_train = timestamps[:test_start]
    ts_test = timestamps[test_start:]
    return X_train, X_test, y_train, y_test, ts_train, ts_test


def impute_feature_values(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    imputation_values: dict[str, float] = {}

    for column in feature_columns:
        train_series = X_train[column].copy()
        if column == "Pressure":
            train_series = train_series.replace(0, np.nan)

        fill_value = train_series.mean()
        if pd.isna(fill_value):
            fill_value = 0.0

        for frame in (X_train, X_test):
            if column == "Pressure":
                frame.loc[frame[column] == 0, column] = np.nan
            frame.loc[:, column] = frame[column].fillna(fill_value)

        imputation_values[column] = float(fill_value)

    return X_train, X_test, imputation_values


def scale_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, StandardScaler]:
    feature_scaler = StandardScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = feature_scaler.transform(X_test).astype(np.float32)

    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.to_numpy().reshape(-1, 1)).astype(np.float32).flatten()
    y_test_scaled = target_scaler.transform(y_test.to_numpy().reshape(-1, 1)).astype(np.float32).flatten()

    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, feature_scaler, target_scaler


def create_sequences_gap_aware(
    X: np.ndarray,
    y: np.ndarray,
    timestamps: np.ndarray,
    seq_length: int,
    max_gap_minutes: int,
) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X)
    y = np.asarray(y)
    timestamps = np.asarray(timestamps, dtype="datetime64[ns]")

    if len(X) <= seq_length:
        return (
            np.empty((0, seq_length, X.shape[1]), dtype=X.dtype),
            np.empty((0,), dtype=y.dtype),
        )

    time_diffs = np.diff(timestamps).astype("timedelta64[m]").astype(np.float64)

    is_gap = time_diffs > max_gap_minutes
    gap_indices = np.where(is_gap)[0] + 1
    seg_starts = np.concatenate([[0], gap_indices])
    seg_ends = np.concatenate([gap_indices, [len(X)]])

    X_sequences = []
    y_sequences = []
    for start, end in zip(seg_starts, seg_ends):
        X_segment = X[start:end]
        y_segment = y[start:end]
        segment_length = end - start
        if segment_length <= seq_length:
            continue

        start_indices = np.arange(segment_length - seq_length)[:, None]
        window_indices = start_indices + np.arange(seq_length)[None, :]
        X_sequences.append(X_segment[window_indices])
        y_sequences.append(y_segment[seq_length:])

    if not X_sequences:
        return (
            np.empty((0, seq_length, X.shape[1]), dtype=X.dtype),
            np.empty((0,), dtype=y.dtype),
        )

    return np.concatenate(X_sequences, axis=0), np.concatenate(y_sequences, axis=0)


def resolve_loss(loss_name: str):
    if loss_name == "huber":
        return tf.keras.losses.Huber()
    if loss_name == "mse":
        return "mse"
    raise ValueError(f"Unsupported loss_name: {loss_name}")


def evaluate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {
        "mse": float(mse),
        "rmse": rmse,
        "mae": float(mae),
        "r2": float(r2),
    }


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = SEED,
) -> tuple[float, float, float]:
    """Return (point_estimate, ci_lower, ci_upper) for a metric via bootstrap."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    point = metric_fn(y_true, y_pred)
    scores = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        scores[i] = metric_fn(y_true[idx], y_pred[idx])
    alpha = 1 - confidence_level
    ci_lower = float(np.percentile(scores, 100 * alpha / 2))
    ci_upper = float(np.percentile(scores, 100 * (1 - alpha / 2)))
    return float(point), ci_lower, ci_upper


def compute_statistical_tests(
    y_true: np.ndarray,
    y_pred_lstm: np.ndarray,
    y_pred_naive: np.ndarray,
    y_pred_linear: np.ndarray,
    n_bootstrap: int = 1000,
) -> dict[str, object]:
    """Compute Wilcoxon signed-rank tests and bootstrap 95% CIs."""
    ae_lstm = np.abs(y_true - y_pred_lstm)
    ae_naive = np.abs(y_true - y_pred_naive)
    ae_linear = np.abs(y_true - y_pred_linear)

    # Wilcoxon signed-rank tests on absolute errors (two-sided)
    # Subsample if very large to keep runtime reasonable
    max_samples = 50_000
    if len(ae_lstm) > max_samples:
        rng = np.random.RandomState(SEED)
        idx = rng.choice(len(ae_lstm), size=max_samples, replace=False)
        ae_lstm_sub, ae_naive_sub, ae_linear_sub = ae_lstm[idx], ae_naive[idx], ae_linear[idx]
    else:
        ae_lstm_sub, ae_naive_sub, ae_linear_sub = ae_lstm, ae_naive, ae_linear

    try:
        stat_naive, p_naive = stats.wilcoxon(ae_lstm_sub, ae_naive_sub, alternative="two-sided")
    except ValueError:
        stat_naive, p_naive = np.nan, np.nan

    try:
        stat_linear, p_linear = stats.wilcoxon(ae_lstm_sub, ae_linear_sub, alternative="two-sided")
    except ValueError:
        stat_linear, p_linear = np.nan, np.nan

    # Bootstrap 95% CIs for LSTM metrics
    def _mae(yt, yp):
        return float(np.mean(np.abs(yt - yp)))

    def _rmse(yt, yp):
        return float(np.sqrt(np.mean((yt - yp) ** 2)))

    def _r2(yt, yp):
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - np.mean(yt)) ** 2)
        return float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    _, mae_ci_lo, mae_ci_hi = bootstrap_metric_ci(y_true, y_pred_lstm, _mae, n_bootstrap)
    _, rmse_ci_lo, rmse_ci_hi = bootstrap_metric_ci(y_true, y_pred_lstm, _rmse, n_bootstrap)
    _, r2_ci_lo, r2_ci_hi = bootstrap_metric_ci(y_true, y_pred_lstm, _r2, n_bootstrap)

    return {
        "wilcoxon_stat_vs_naive": float(stat_naive),
        "wilcoxon_p_vs_naive": float(p_naive),
        "wilcoxon_stat_vs_linear": float(stat_linear),
        "wilcoxon_p_vs_linear": float(p_linear),
        "lstm_mae_ci_lower": mae_ci_lo,
        "lstm_mae_ci_upper": mae_ci_hi,
        "lstm_rmse_ci_lower": rmse_ci_lo,
        "lstm_rmse_ci_upper": rmse_ci_hi,
        "lstm_r2_ci_lower": r2_ci_lo,
        "lstm_r2_ci_upper": r2_ci_hi,
    }


def build_lstm_model(input_shape: tuple[int, int], model_config: dict | None = None) -> Sequential:
    config = MODEL_CONFIG.copy()
    if model_config is not None:
        config.update(model_config)

    layers = [Input(shape=input_shape)]
    lstm_units = config["lstm_units"]
    dropout_rate = config["dropout_rate"]

    for layer_index, units in enumerate(lstm_units):
        layers.append(LSTM(units, return_sequences=layer_index < len(lstm_units) - 1))
        layers.append(Dropout(dropout_rate))

    layers.append(Dense(config["dense_units"], activation=config["dense_activation"]))
    layers.append(Dense(1))

    model = Sequential(layers)
    model.compile(
        optimizer=config["optimizer"],
        loss=resolve_loss(config["loss_name"]),
        metrics=config["metrics"],
    )
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the LSTM-60stepsback notebook logic as a batch script.")
    parser.add_argument("--location", nargs="+", help="Optional subset of locations to run.")
    parser.add_argument("--year", nargs="+", type=int, help="Optional subset of years to run.")
    parser.add_argument("--resume", action="store_true", help="Resume from the latest progress files and skip completed location-year runs.")
    parser.add_argument("--start-at-location", type=str, help="Optional location at which to start the ordered run list.")
    parser.add_argument("--start-at-year", type=int, help="Optional year paired with --start-at-location for the first combination to run.")
    parser.add_argument("--epochs", type=int, default=TRAINING_CONFIG["epochs"], help="Training epochs per run.")
    parser.add_argument("--batch-size", type=int, default=TRAINING_CONFIG["batch_size"], help="Batch size per run.")
    parser.add_argument("--seq-length", type=int, default=TRAINING_CONFIG["seq_length"], help="Sequence length.")
    parser.add_argument("--max-gap-min", type=int, default=TRAINING_CONFIG["max_gap_min"], help="Maximum allowed time gap in minutes inside a sequence.")
    parser.add_argument("--test-fraction", type=float, default=TRAINING_CONFIG["test_fraction"], help="Chronological test fraction.")
    parser.add_argument("--skip-plots", action="store_true", help="Skip per-run and summary plot generation to maximize throughput.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_CONFIG["output_dir"], help="Directory for summaries and plots.")
    return parser.parse_args()


def select_experiment_keys(
    available_keys: list[tuple[str, int]],
    requested_locations: list[str] | None,
    requested_years: list[int] | None,
) -> list[tuple[str, int]]:
    experiment_keys = list(available_keys)

    if requested_locations is not None:
        requested_location_set = {str(location) for location in requested_locations}
        experiment_keys = [
            (location, year)
            for location, year in experiment_keys
            if location in requested_location_set
        ]

    if requested_years is not None:
        requested_year_set = {int(year) for year in requested_years}
        experiment_keys = [
            (location, year)
            for location, year in experiment_keys
            if int(year) in requested_year_set
        ]

    return [(str(location), int(year)) for location, year in experiment_keys]


def build_location_year_groups(full_df: pd.DataFrame) -> tuple[dict[tuple[str, int], pd.DataFrame], pd.Series]:
    grouped_frames: dict[tuple[str, int], pd.DataFrame] = {}
    location_year_counts = (
        full_df.groupby(["river_location", "Year"])
        .size()
        .rename("row_count")
        .sort_index()
    )

    for (location, year), location_df in full_df.groupby(["river_location", "Year"], sort=True):
        grouped_frames[(str(location), int(year))] = location_df.sort_values("Timestamp").reset_index(drop=True)

    return grouped_frames, location_year_counts


def apply_start_at_key(
    experiment_keys: list[tuple[str, int]],
    start_location: str | None,
    start_year: int | None,
) -> list[tuple[str, int]]:
    if start_location is None and start_year is None:
        return experiment_keys

    if not start_location or start_year is None:
        raise ValueError("--start-at-location and --start-at-year must be provided together.")

    start_key = (str(start_location), int(start_year))
    if start_key not in experiment_keys:
        raise ValueError(f"Start key {start_key} was not found in the requested experiment set.")

    start_index = experiment_keys.index(start_key)
    return experiment_keys[start_index:]


def save_location_run_plots(
    output_dir: Path,
    location: str,
    run_plot_data: list[dict[str, object]],
) -> dict[str, Path]:
    plot_dir = output_dir / "location_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    safe_location = "_".join(location.split()).lower()
    ordered_runs = sorted(run_plot_data, key=lambda item: int(item["Year"]))
    years = [int(item["Year"]) for item in ordered_runs]

    loss_plot_path = plot_dir / f"{safe_location}_loss_by_year.png"
    fig, ax = plt.subplots(figsize=(10, 5))
    for item in ordered_runs:
        ax.plot(item["history"].history["loss"], label=str(item["Year"]))
    ax.set_title(f"{location} - Training Loss by Year")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(title="Year", ncol=min(len(years), 5))
    fig.tight_layout()
    fig.savefig(loss_plot_path, dpi=150)
    plt.close(fig)

    prediction_plot_path = plot_dir / f"{safe_location}_prediction_by_year.png"
    fig, axes = plt.subplots(len(ordered_runs), 1, figsize=(12, max(4, 3 * len(ordered_runs))), squeeze=False)
    for axis, item in zip(axes.flatten(), ordered_runs):
        axis.plot(item["y_true"], label="Actual", alpha=0.7)
        axis.plot(item["y_pred"], label="LSTM Predicted", alpha=0.7)
        axis.set_title(f"{location} {item['Year']} - Actual vs Predicted (Test)")
        axis.set_xlabel("Sample")
        axis.set_ylabel("River Level")
        axis.legend()
    fig.tight_layout()
    fig.savefig(prediction_plot_path, dpi=150)
    plt.close(fig)

    return {
        "loss_plot": loss_plot_path,
        "prediction_plot": prediction_plot_path,
    }


def save_location_metric_bars(output_dir: Path, summary_df: pd.DataFrame, metric_name: str, ylabel: str) -> list[Path]:
    summary_dir = output_dir / "summary_plots"
    summary_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    bar_width = 0.25

    for location in summary_df["Location"].unique():
        location_df = summary_df[summary_df["Location"] == location].sort_values("Year").reset_index(drop=True)
        years = location_df["Year"].to_numpy()
        x = np.arange(len(years))

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(x - bar_width, location_df[f"lstm_{metric_name}"], bar_width, label="LSTM")
        ax.bar(x, location_df[f"naive_{metric_name}"], bar_width, label="Naive")
        ax.bar(x + bar_width, location_df[f"linear_{metric_name}"], bar_width, label="Linear")
        ax.set_xticks(x)
        ax.set_xticklabels(years)
        ax.set_xlabel("Year")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{location} - {ylabel} by Year (LSTM vs Baselines)")
        ax.legend()
        fig.tight_layout()

        plot_path = summary_dir / f"{'_'.join(location.split()).lower()}_{metric_name}.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        saved_paths.append(plot_path)

    return saved_paths


def load_progress_df(progress_path: Path) -> pd.DataFrame:
    if not progress_path.exists():
        return pd.DataFrame()
    return pd.read_csv(progress_path)


def load_completed_keys(output_dir: Path) -> set[tuple[str, int]]:
    progress_path = output_dir / PROGRESS_FILES["completed"]
    completed_df = load_progress_df(progress_path)
    if completed_df.empty:
        return set()
    return {
        (str(row.Location), int(row.Year))
        for row in completed_df.itertuples(index=False)
    }


def safe_write_csv(dataframe: pd.DataFrame, output_path: Path, description: str) -> Path:
    try:
        dataframe.to_csv(output_path, index=False)
        return output_path
    except PermissionError:
        fallback_path = output_path.with_name(f"{output_path.stem}_pending{output_path.suffix}")
        dataframe.to_csv(fallback_path, index=False)
        print(
            f"Warning: could not write {description} to {output_path.resolve()} because the file is locked. "
            f"Wrote a fallback copy to {fallback_path.resolve()} instead."
        )
        return fallback_path


def persist_progress(output_dir: Path, summary_rows: list[dict[str, object]], skipped_rows: list[dict[str, object]]) -> None:
    completed_path = output_dir / PROGRESS_FILES["completed"]
    skipped_path = output_dir / PROGRESS_FILES["skipped"]

    completed_df = pd.DataFrame(summary_rows)
    skipped_df = pd.DataFrame(skipped_rows)

    if not completed_df.empty:
        safe_write_csv(
            completed_df.sort_values(["Location", "Year"]).reset_index(drop=True),
            completed_path,
            "completed progress",
        )

    if not skipped_df.empty:
        safe_write_csv(
            skipped_df.sort_values(["Location", "Year"]).reset_index(drop=True),
            skipped_path,
            "skipped progress",
        )


def main() -> None:
    args = parse_args()

    raw_feature_columns = ["Windspeed", "Humidity", "Temperature", "Dewpoint", "Pressure", "Rainfall", "Wind direction"]
    feature_columns = ["Windspeed", "Humidity", "Temperature", "Dewpoint", "Pressure", "Rainfall", "Wind direction sin", "Wind direction cos"]
    target_column = "Level"
    input_feature_columns = [*feature_columns, target_column]
    selected_columns = ["Timestamp", "river_location", *raw_feature_columns, "Wind direction sin", "Wind direction cos", target_column]

    dataset_path = resolve_main_dataset_path()
    full_df = load_and_preprocess_data(dataset_path, selected_columns)

    print("Building location-year groups from the loaded dataset...")
    location_year_groups, location_year_counts = build_location_year_groups(full_df)
    available_keys = list(location_year_counts.index)

    experiment_keys = select_experiment_keys(available_keys, args.location, args.year)
    experiment_keys = apply_start_at_key(experiment_keys, args.start_at_location, args.start_at_year)
    if not experiment_keys:
        raise ValueError("No location-year combinations matched the requested filters.")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    completed_keys = set()
    if args.resume:
        completed_keys = load_completed_keys(output_dir)
        if completed_keys:
            experiment_keys = [key for key in experiment_keys if key not in completed_keys]
            print(f"Resume mode: skipping {len(completed_keys)} completed location-year runs already recorded in {PROGRESS_FILES['completed']}")

    if not experiment_keys:
        print("No remaining location-year combinations to run after applying resume state.")
        return

    seq_length = int(args.seq_length)
    max_gap_min = int(args.max_gap_min)
    epochs = int(args.epochs)
    batch_size = int(args.batch_size)
    test_fraction = float(args.test_fraction)
    min_rows_for_test_sequence = int(np.ceil((seq_length + 1) / test_fraction))

    print(f"Dataset path: {dataset_path.resolve()}")
    print(f"Loaded rows: {len(full_df):,}")
    print(f"Observed location-year groups prepared in memory: {len(location_year_groups)}")
    print(f"Requested experiments: {len(experiment_keys)}")
    for location, year in experiment_keys:
        print(f"  - {location} | {year}")

    summary_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, object]] = []

    # When resuming, seed summary_rows with previously completed results so
    # persist_progress keeps the full history in completed_runs_latest.csv.
    if args.resume and completed_keys:
        prev_completed = load_progress_df(output_dir / PROGRESS_FILES["completed"])
        if not prev_completed.empty:
            summary_rows.extend(prev_completed.to_dict("records"))
        prev_skipped = load_progress_df(output_dir / PROGRESS_FILES["skipped"])
        if not prev_skipped.empty:
            skipped_rows.extend(prev_skipped.to_dict("records"))

    location_plot_data: dict[str, list[dict[str, object]]] = {}
    total_experiments = len(experiment_keys)
    run_count = 0

    for experiment_number, (location, year) in enumerate(experiment_keys, start=1):
        run_label = f"{location} | {year}"
        print(f"\n{'=' * 80}")
        print(f"[{experiment_number}/{total_experiments}] Processing: {run_label}")
        print(f"{'=' * 80}")
        started_at = time.perf_counter()

        location_df = location_year_groups[(location, year)]

        total_rows = len(location_df)
        print(f"Total rows: {total_rows:,}")

        if total_rows < min_rows_for_test_sequence:
            reason = (
                f"not enough rows for a chronological 80/20 split with {seq_length}-step test sequences "
                f"(need at least {min_rows_for_test_sequence})"
            )
            print(f"  SKIPPING - {reason}")
            skipped_rows.append({
                "Location": location,
                "Year": year,
                "rows": total_rows,
                "reason": reason,
            })
            persist_progress(output_dir, summary_rows, skipped_rows)
            continue

        modeling_df = location_df.drop(["Timestamp", "river_location", "Year"], axis=1)
        timestamps = location_df["Timestamp"].to_numpy()

        X_all, y_all = split_features_target(modeling_df, input_feature_columns, target_column)
        X_train, X_test, y_train, y_test, ts_train, ts_test = split_train_test_chronological(
            X_all,
            y_all,
            timestamps,
            test_frac=test_fraction,
        )

        X_train, X_test, _ = impute_feature_values(X_train, X_test, feature_columns)
        X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, _, target_scaler = scale_data(
            X_train,
            X_test,
            y_train,
            y_test,
        )

        print("Creating gap-aware sequences...")
        X_train_seq, y_train_seq = create_sequences_gap_aware(
            X_train_scaled,
            y_train_scaled,
            ts_train,
            seq_length,
            max_gap_min,
        )
        X_test_seq, y_test_seq = create_sequences_gap_aware(
            X_test_scaled,
            y_test_scaled,
            ts_test,
            seq_length,
            max_gap_min,
        )

        train_level_history, y_train_raw_seq = create_sequences_gap_aware(
            X_train[[target_column]].to_numpy(),
            y_train.to_numpy(),
            ts_train,
            seq_length,
            max_gap_min,
        )
        test_level_history, y_test_raw_seq = create_sequences_gap_aware(
            X_test[[target_column]].to_numpy(),
            y_test.to_numpy(),
            ts_test,
            seq_length,
            max_gap_min,
        )

        print(f"Sequence shapes - Train: {X_train_seq.shape}, Test: {X_test_seq.shape}")

        if min(len(X_train_seq), len(X_test_seq)) == 0:
            reason = "gap-aware sequence creation produced an empty train or test split"
            print(f"  SKIPPING - {reason}.")
            skipped_rows.append({
                "Location": location,
                "Year": year,
                "rows": total_rows,
                "reason": reason,
            })
            persist_progress(output_dir, summary_rows, skipped_rows)
            continue

        tf.keras.backend.clear_session()
        model = build_lstm_model((seq_length, X_train_seq.shape[2]))

        reduce_lr = ReduceLROnPlateau(monitor="loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1)
        early_stop = EarlyStopping(monitor="loss", patience=15, restore_best_weights=True, verbose=1)

        history = model.fit(
            X_train_seq,
            y_train_seq,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[reduce_lr, early_stop],
            verbose=1,
            shuffle=False,
        )

        y_pred_scaled = model.predict(X_test_seq, verbose=0).flatten()
        y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_true = target_scaler.inverse_transform(y_test_seq.reshape(-1, 1)).flatten()

        lstm_metrics = evaluate_regression_metrics(y_true, y_pred)
        naive_pred = test_level_history[:, -1, 0]
        naive_metrics = evaluate_regression_metrics(y_test_raw_seq, naive_pred)

        linear_baseline = LinearRegression()
        linear_baseline.fit(train_level_history[:, :, 0], y_train_raw_seq)
        linear_pred = linear_baseline.predict(test_level_history[:, :, 0])
        linear_metrics = evaluate_regression_metrics(y_test_raw_seq, linear_pred)

        print("  Computing statistical significance tests and bootstrap CIs...")
        stat_results = compute_statistical_tests(
            y_test_raw_seq, y_pred, naive_pred, linear_pred, n_bootstrap=1000,
        )

        elapsed_seconds = time.perf_counter() - started_at
        run_count += 1

        print(
            f"  Done [{run_count}] LSTM RMSE={lstm_metrics['rmse']:.4f}, "
            f"R²={lstm_metrics['r2']:.4f}, elapsed={elapsed_seconds / 60:.2f} min"
        )
        print(
            f"  MAE={lstm_metrics['mae']:.4f} "
            f"[{stat_results['lstm_mae_ci_lower']:.4f}, {stat_results['lstm_mae_ci_upper']:.4f}] | "
            f"Wilcoxon p vs Naive={stat_results['wilcoxon_p_vs_naive']:.2e}, "
            f"vs Linear={stat_results['wilcoxon_p_vs_linear']:.2e}"
        )

        saved_plots = {"loss_plot": None, "prediction_plot": None}
        if not args.skip_plots:
            location_plot_data.setdefault(location, []).append({
                "Year": year,
                "history": history,
                "y_true": y_true.copy(),
                "y_pred": y_pred.copy(),
            })

        summary_rows.append({
            "Location": location,
            "Year": year,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "train_fraction": len(X_train) / total_rows,
            "test_fraction": len(X_test) / total_rows,
            "train_sequences": len(X_train_seq),
            "test_sequences": len(X_test_seq),
            "lstm_rmse": lstm_metrics["rmse"],
            "naive_rmse": naive_metrics["rmse"],
            "linear_rmse": linear_metrics["rmse"],
            "lstm_mae": lstm_metrics["mae"],
            "naive_mae": naive_metrics["mae"],
            "linear_mae": linear_metrics["mae"],
            "lstm_r2": lstm_metrics["r2"],
            "naive_r2": naive_metrics["r2"],
            "linear_r2": linear_metrics["r2"],
            **stat_results,
            "elapsed_seconds": elapsed_seconds,
            "loss_plot": "" if saved_plots["loss_plot"] is None else str(saved_plots["loss_plot"].resolve()),
            "prediction_plot": "" if saved_plots["prediction_plot"] is None else str(saved_plots["prediction_plot"].resolve()),
        })

        is_last_experiment = experiment_number == total_experiments
        next_location = None if is_last_experiment else experiment_keys[experiment_number][0]
        if not args.skip_plots and (is_last_experiment or next_location != location):
            saved_plots = save_location_run_plots(output_dir, location, location_plot_data.get(location, []))
            for row in summary_rows:
                if row["Location"] == location:
                    row["loss_plot"] = str(saved_plots["loss_plot"].resolve())
                    row["prediction_plot"] = str(saved_plots["prediction_plot"].resolve())
            location_plot_data.pop(location, None)

        persist_progress(output_dir, summary_rows, skipped_rows)

        del model, history, linear_baseline
        del X_train_seq, y_train_seq, X_test_seq, y_test_seq
        del train_level_history, y_train_raw_seq, test_level_history, y_test_raw_seq
        del y_pred_scaled, y_pred, y_true, naive_pred, linear_pred
        tf.keras.backend.clear_session()
        gc.collect()

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["Location", "Year"]).reset_index(drop=True)

    skipped_df = pd.DataFrame(skipped_rows)
    if not skipped_df.empty:
        skipped_df = skipped_df.sort_values(["Location", "Year"]).reset_index(drop=True)

    timestamp_label = time.strftime("%Y%m%d_%H%M%S")
    summary_csv = output_dir / f"lstm_60stepsback_summary_{timestamp_label}.csv"
    skipped_csv = output_dir / f"lstm_60stepsback_skipped_{timestamp_label}.csv"

    if not summary_df.empty:
        summary_df.to_csv(summary_csv, index=False)
        if not args.skip_plots:
            save_location_metric_bars(output_dir, summary_df, "rmse", "RMSE")
            save_location_metric_bars(output_dir, summary_df, "r2", "R²")

    if not skipped_df.empty:
        skipped_df.to_csv(skipped_csv, index=False)

    print(f"\nAll training complete. {run_count} experiments finished out of {total_experiments} requested combinations.")
    if not summary_df.empty:
        print("\nCompleted split fractions:")
        print(summary_df[["Location", "Year", "train_fraction", "test_fraction"]].to_string(index=False))

        print("\n--- Statistical Significance Summary ---")
        sig_cols = ["Location", "Year", "lstm_mae", "lstm_mae_ci_lower", "lstm_mae_ci_upper",
                    "wilcoxon_p_vs_naive", "wilcoxon_p_vs_linear"]
        avail_cols = [c for c in sig_cols if c in summary_df.columns]
        print(summary_df[avail_cols].to_string(index=False))

        if "wilcoxon_p_vs_naive" in summary_df.columns:
            n_sig_naive = (summary_df["wilcoxon_p_vs_naive"] < 0.05).sum()
            n_sig_linear = (summary_df["wilcoxon_p_vs_linear"] < 0.05).sum()
            print(f"\nLSTM vs Naive:  {n_sig_naive}/{len(summary_df)} runs significantly different (p < 0.05)")
            print(f"LSTM vs Linear: {n_sig_linear}/{len(summary_df)} runs significantly different (p < 0.05)")

        print(f"\nSummary CSV: {summary_csv.resolve()}")
    if not skipped_df.empty:
        print("\nSkipped combinations:")
        print(skipped_df.to_string(index=False))
        print(f"\nSkipped CSV: {skipped_csv.resolve()}")


if __name__ == "__main__":
    main()