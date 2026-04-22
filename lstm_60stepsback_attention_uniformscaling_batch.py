from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, Input, Lambda, LSTM, Softmax


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
    "output_dir": Path("analysis_outputs") / "lstm_60stepsback_attention_uniformscaling_batch",
}

PROGRESS_FILES = {
    "completed": "completed_runs_latest.csv",
    "skipped": "skipped_runs_latest.csv",
}

EXPLAINABILITY_CONFIG = {
    "max_test_sequences": 5000,
    "random_seed": SEED,
    "local_explanations_per_run": 3,
    "integrated_gradient_steps": 24,
    "heatmap_max_samples": 100,
}

# ---------------------------------------------------------------------------
# Dataset resolution
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Data loading & preprocessing
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Feature / target splits
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Gap-aware sequence creation
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Loss helper
# ---------------------------------------------------------------------------

def resolve_loss(loss_name: str):
    if loss_name == "huber":
        return tf.keras.losses.Huber()
    if loss_name == "mse":
        return "mse"
    raise ValueError(f"Unsupported loss_name: {loss_name}")

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Attention LSTM model (functional API)
# ---------------------------------------------------------------------------

def build_attention_lstm_model(input_shape: tuple[int, int], model_config: dict | None = None) -> Model:
    config = MODEL_CONFIG.copy()
    if model_config is not None:
        config.update(model_config)

    inputs = Input(shape=input_shape, name="sequence_input")
    x = inputs

    for units in config["lstm_units"]:
        x = LSTM(units, return_sequences=True)(x)
        x = Dropout(config["dropout_rate"])(x)

    attention_scores = Dense(1, activation="tanh", name="attention_scores")(x)
    attention_weights = Softmax(axis=1, name="attention_weights")(attention_scores)
    context_vector = Lambda(
        lambda tensors: tf.reduce_sum(tensors[0] * tensors[1], axis=1),
        name="context_vector",
    )([x, attention_weights])

    dense_features = Dense(
        config["dense_units"],
        activation=config["dense_activation"],
        name="dense_projection",
    )(context_vector)
    output = Dense(1, name="river_level_prediction")(dense_features)

    model = Model(inputs=inputs, outputs=output, name="attention_lstm_regressor")
    model.compile(
        optimizer=config["optimizer"],
        loss=resolve_loss(config["loss_name"]),
        metrics=config["metrics"],
    )
    return model

# ---------------------------------------------------------------------------
# Attention weight extraction
# ---------------------------------------------------------------------------

def extract_attention_weights(model: Model, X_sequences: np.ndarray) -> np.ndarray:
    attention_extractor = Model(
        inputs=model.input,
        outputs=model.get_layer("attention_weights").output,
        name="attention_weight_extractor",
    )
    return attention_extractor.predict(X_sequences, verbose=0).squeeze(-1)

# ---------------------------------------------------------------------------
# Explainability: permutation feature importance
# ---------------------------------------------------------------------------

def downsample_sequences(
    X_sequences: np.ndarray,
    y_sequences: np.ndarray,
    max_sequences: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(X_sequences) <= max_sequences:
        return X_sequences, y_sequences
    selected_indices = np.linspace(0, len(X_sequences) - 1, num=max_sequences, dtype=int)
    return X_sequences[selected_indices], y_sequences[selected_indices]


def compute_permutation_feature_importance(
    model: Model,
    X_test_seq: np.ndarray,
    y_test_seq: np.ndarray,
    target_scaler: StandardScaler,
    feature_names: list[str],
) -> pd.DataFrame:
    explain_X, explain_y = downsample_sequences(
        X_test_seq,
        y_test_seq,
        EXPLAINABILITY_CONFIG["max_test_sequences"],
    )

    baseline_pred_scaled = model.predict(explain_X, verbose=0).flatten()
    baseline_pred = target_scaler.inverse_transform(baseline_pred_scaled.reshape(-1, 1)).flatten()
    baseline_true = target_scaler.inverse_transform(explain_y.reshape(-1, 1)).flatten()
    baseline_metrics = evaluate_regression_metrics(baseline_true, baseline_pred)

    rng = np.random.default_rng(EXPLAINABILITY_CONFIG["random_seed"])
    importance_rows: list[dict[str, object]] = []

    last_timestep_frame = pd.DataFrame(explain_X[:, -1, :], columns=feature_names)

    for feature_index, feature_name in enumerate(feature_names):
        permuted_X = explain_X.copy()
        shuffled_feature = permuted_X[:, :, feature_index].copy()
        rng.shuffle(shuffled_feature, axis=0)
        permuted_X[:, :, feature_index] = shuffled_feature

        permuted_pred_scaled = model.predict(permuted_X, verbose=0).flatten()
        permuted_pred = target_scaler.inverse_transform(permuted_pred_scaled.reshape(-1, 1)).flatten()
        permuted_metrics = evaluate_regression_metrics(baseline_true, permuted_pred)

        last_step_correlation = np.corrcoef(last_timestep_frame[feature_name], baseline_true)[0, 1]
        if np.isnan(last_step_correlation):
            last_step_correlation = 0.0

        importance_rows.append({
            "feature": feature_name,
            "baseline_rmse": baseline_metrics["rmse"],
            "permuted_rmse": permuted_metrics["rmse"],
            "rmse_increase": permuted_metrics["rmse"] - baseline_metrics["rmse"],
            "baseline_mae": baseline_metrics["mae"],
            "permuted_mae": permuted_metrics["mae"],
            "mae_increase": permuted_metrics["mae"] - baseline_metrics["mae"],
            "last_timestep_corr_with_level": float(last_step_correlation),
            "explanation_hint": (
                "positive last-step correlation"
                if last_step_correlation > 0
                else "negative last-step correlation"
            ),
        })

    importance_df = pd.DataFrame(importance_rows).sort_values(
        "rmse_increase", ascending=False,
    ).reset_index(drop=True)
    return importance_df

# ---------------------------------------------------------------------------
# Explainability: integrated gradients (per-sample attribution)
# ---------------------------------------------------------------------------

def compute_integrated_gradients(
    model: Model,
    input_sequence: np.ndarray,
    baseline_sequence: np.ndarray | None = None,
    steps: int = 24,
) -> np.ndarray:
    if baseline_sequence is None:
        baseline_sequence = np.zeros_like(input_sequence, dtype=np.float32)

    input_tensor = tf.convert_to_tensor(input_sequence[np.newaxis, ...], dtype=tf.float32)
    baseline_tensor = tf.convert_to_tensor(baseline_sequence[np.newaxis, ...], dtype=tf.float32)
    alpha_values = tf.linspace(0.0, 1.0, steps + 1)

    gradient_list = []
    for alpha in alpha_values:
        interpolated = baseline_tensor + alpha * (input_tensor - baseline_tensor)
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            predictions = model(interpolated, training=False)
        gradients = tape.gradient(predictions, interpolated)
        gradient_list.append(gradients)

    stacked_gradients = tf.stack(gradient_list, axis=0)
    average_gradients = tf.reduce_mean(stacked_gradients[:-1] + stacked_gradients[1:], axis=0) / 2.0
    integrated_gradients = (input_tensor - baseline_tensor) * average_gradients
    return integrated_gradients.numpy()[0]


def select_local_explanation_indices(n_sequences: int, n_samples: int) -> list[int]:
    if n_sequences == 0:
        return []
    sample_count = min(n_sequences, n_samples)
    return np.linspace(0, n_sequences - 1, num=sample_count, dtype=int).tolist()


def compute_local_explanations(
    model: Model,
    X_test_seq: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    feature_names: list[str],
) -> list[dict[str, object]]:
    sample_indices = select_local_explanation_indices(
        len(X_test_seq),
        EXPLAINABILITY_CONFIG["local_explanations_per_run"],
    )
    local_explanations: list[dict[str, object]] = []

    for sample_index in sample_indices:
        input_sequence = X_test_seq[sample_index]
        attributions = compute_integrated_gradients(
            model,
            input_sequence,
            steps=EXPLAINABILITY_CONFIG["integrated_gradient_steps"],
        )
        feature_contributions = attributions.sum(axis=0)
        timestep_contributions = attributions.sum(axis=1)
        ranked_feature_indices = np.argsort(np.abs(feature_contributions))[::-1]

        top_features = [
            {
                "feature": feature_names[feature_index],
                "contribution": float(feature_contributions[feature_index]),
                "abs_contribution": float(abs(feature_contributions[feature_index])),
            }
            for feature_index in ranked_feature_indices[:5]
        ]

        local_explanations.append({
            "sample_index": int(sample_index),
            "actual_level": float(y_true[sample_index]),
            "predicted_level": float(y_pred[sample_index]),
            "attributions": attributions,
            "feature_contributions": feature_contributions,
            "timestep_contributions": timestep_contributions,
            "top_features": top_features,
        })

    return local_explanations


def format_local_explanation_summary(local_explanations: list[dict[str, object]]) -> str:
    summary_lines = []
    for explanation in local_explanations:
        top_feature_text = ", ".join(
            f"{feature['feature']} ({feature['contribution']:+.4f})"
            for feature in explanation["top_features"][:3]
        )
        summary_lines.append(
            " | ".join([
                f"sample={explanation['sample_index']}",
                f"actual={explanation['actual_level']:.4f}",
                f"predicted={explanation['predicted_level']:.4f}",
                f"top={top_feature_text}",
            ])
        )
    return "\n".join(summary_lines)

# ---------------------------------------------------------------------------
# Plot helpers: attention-specific
# ---------------------------------------------------------------------------

def save_attention_plots(
    output_dir: Path,
    location: str,
    year: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    attention_weights: np.ndarray,
) -> dict[str, Path]:
    plot_dir = output_dir / "attention_plots" / str(year)
    plot_dir.mkdir(parents=True, exist_ok=True)
    safe_location = "_".join(location.split()).lower()

    mean_attention = attention_weights.mean(axis=0)
    lag_steps = np.arange(-len(mean_attention), 0)

    # Prediction vs actual
    prediction_plot_path = plot_dir / f"{safe_location}_{year}_prediction_vs_actual.png"
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(y_true, label="Actual", linewidth=1.2)
    ax.plot(y_pred, label="Attention LSTM Predicted", linewidth=1.2, alpha=0.85)
    ax.set_title(f"Attention LSTM Predictions: {location} {year}")
    ax.set_xlabel("Test Sequence Index")
    ax.set_ylabel("River Level")
    ax.legend()
    fig.tight_layout()
    fig.savefig(prediction_plot_path, dpi=150)
    plt.close(fig)

    # Attention weight profile + heatmap
    attention_plot_path = plot_dir / f"{safe_location}_{year}_attention_weights.png"
    heatmap_samples = min(EXPLAINABILITY_CONFIG["heatmap_max_samples"], attention_weights.shape[0])
    heatmap_data = attention_weights[:heatmap_samples]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    axes[0].plot(lag_steps, mean_attention, color="tab:blue", linewidth=2)
    axes[0].set_title(f"Mean Attention Weight by Lag: {location} {year}")
    axes[0].set_xlabel("Lag Step")
    axes[0].set_ylabel("Average Weight")
    axes[0].grid(True, alpha=0.3)

    image = axes[1].imshow(heatmap_data, aspect="auto", cmap="viridis", origin="lower")
    axes[1].set_title(f"Attention Heatmap for First {heatmap_samples} Test Sequences: {location} {year}")
    axes[1].set_xlabel("Sequence Timestep")
    axes[1].set_ylabel("Test Sequence Index")
    tick_positions = np.linspace(0, len(lag_steps) - 1, num=6, dtype=int)
    axes[1].set_xticks(tick_positions)
    axes[1].set_xticklabels(lag_steps[tick_positions])
    fig.colorbar(image, ax=axes[1], label="Attention Weight")
    fig.savefig(attention_plot_path, dpi=150)
    plt.close(fig)

    return {
        "prediction_plot": prediction_plot_path,
        "attention_plot": attention_plot_path,
    }


def save_explainability_artifacts(
    output_dir: Path,
    importance_df: pd.DataFrame,
    location: str,
    year: int,
) -> dict[str, Path]:
    explainability_dir = output_dir / "explainability" / str(year)
    explainability_dir.mkdir(parents=True, exist_ok=True)
    safe_location = "_".join(location.split()).lower()

    csv_path = explainability_dir / f"{safe_location}_{year}_feature_importance.csv"
    plot_path = explainability_dir / f"{safe_location}_{year}_feature_importance.png"

    importance_df.to_csv(csv_path, index=False)

    ranked_df = importance_df.sort_values("rmse_increase", ascending=True)
    colors = ["tab:blue" if value >= 0 else "tab:red" for value in ranked_df["rmse_increase"]]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(ranked_df["feature"], ranked_df["rmse_increase"], color=colors)
    ax.set_title(f"Permutation Feature Importance: {location} {year}")
    ax.set_xlabel("RMSE increase after feature permutation")
    ax.set_ylabel("Feature")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    return {
        "importance_csv": csv_path,
        "importance_plot": plot_path,
    }


def save_local_explanation_artifacts(
    output_dir: Path,
    local_explanations: list[dict[str, object]],
    location: str,
    year: int,
    feature_names: list[str],
) -> list[dict[str, object]]:
    explainability_dir = output_dir / "explainability" / str(year)
    explainability_dir.mkdir(parents=True, exist_ok=True)
    safe_location = "_".join(location.split()).lower()
    saved_artifacts: list[dict[str, object]] = []

    for explanation in local_explanations:
        sample_index = int(explanation["sample_index"])
        attributions = np.asarray(explanation["attributions"])
        feature_contributions = np.asarray(explanation["feature_contributions"])
        timestep_contributions = np.asarray(explanation["timestep_contributions"])

        attribution_csv = explainability_dir / f"{safe_location}_{year}_sample_{sample_index:05d}_local_attributions.csv"
        heatmap_plot = explainability_dir / f"{safe_location}_{year}_sample_{sample_index:05d}_local_heatmap.png"
        feature_plot = explainability_dir / f"{safe_location}_{year}_sample_{sample_index:05d}_local_features.png"

        attribution_frame = pd.DataFrame(attributions, columns=feature_names)
        attribution_frame.insert(0, "timestep", np.arange(-len(attribution_frame), 0))
        attribution_frame.to_csv(attribution_csv, index=False)

        # Heatmap + timestep contribution
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
        image = axes[0].imshow(attributions.T, aspect="auto", cmap="coolwarm", origin="lower")
        axes[0].set_title(f"Local Attribution Heatmap: {location} {year} sample {sample_index}")
        axes[0].set_xlabel("Sequence Timestep")
        axes[0].set_ylabel("Feature")
        axes[0].set_yticks(np.arange(len(feature_names)))
        axes[0].set_yticklabels(feature_names)
        fig.colorbar(image, ax=axes[0], label="Integrated gradient attribution")

        axes[1].plot(np.arange(-len(timestep_contributions), 0), timestep_contributions, color="tab:orange")
        axes[1].set_title("Net contribution by timestep")
        axes[1].set_xlabel("Lag Step")
        axes[1].set_ylabel("Contribution")
        axes[1].grid(True, alpha=0.3)
        fig.savefig(heatmap_plot, dpi=150)
        plt.close(fig)

        # Per-feature signed contribution bar
        ranked_indices = np.argsort(feature_contributions)
        fig, ax = plt.subplots(figsize=(10, 5))
        bar_colors = ["tab:blue" if value >= 0 else "tab:red" for value in feature_contributions[ranked_indices]]
        ax.barh(np.array(feature_names)[ranked_indices], feature_contributions[ranked_indices], color=bar_colors)
        ax.set_title(f"Local Feature Contributions: {location} {year} sample {sample_index}")
        ax.set_xlabel("Signed contribution to prediction")
        ax.set_ylabel("Feature")
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(feature_plot, dpi=150)
        plt.close(fig)

        saved_artifacts.append({
            "sample_index": sample_index,
            "actual_level": explanation["actual_level"],
            "predicted_level": explanation["predicted_level"],
            "top_features": explanation["top_features"],
            "attribution_csv": attribution_csv,
            "heatmap_plot": heatmap_plot,
            "feature_plot": feature_plot,
        })

    return saved_artifacts

# ---------------------------------------------------------------------------
# Plot helpers: loss-by-year and metric bars (carried over from plain LSTM)
# ---------------------------------------------------------------------------

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
    ax.set_title(f"{location} - Attention LSTM Training Loss by Year")
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
        axis.plot(item["y_pred"], label="Attention LSTM Predicted", alpha=0.7)
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
        ax.bar(x - bar_width, location_df[f"lstm_{metric_name}"], bar_width, label="Attention LSTM")
        ax.bar(x, location_df[f"naive_{metric_name}"], bar_width, label="Naive")
        ax.bar(x + bar_width, location_df[f"linear_{metric_name}"], bar_width, label="Linear")
        ax.set_xticks(x)
        ax.set_xticklabels(years)
        ax.set_xlabel("Year")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{location} - {ylabel} by Year (Attention LSTM vs Baselines)")
        ax.legend()
        fig.tight_layout()

        plot_path = summary_dir / f"{'_'.join(location.split()).lower()}_{metric_name}.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        saved_paths.append(plot_path)

    return saved_paths

# ---------------------------------------------------------------------------
# Progress persistence & resume
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Attention LSTM with explainability for river-level prediction (batch mode).",
    )
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
    parser.add_argument("--skip-explainability", action="store_true", help="Skip permutation importance and integrated-gradient explanations.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_CONFIG["output_dir"], help="Directory for summaries and plots.")
    return parser.parse_args()

# ---------------------------------------------------------------------------
# Experiment key helpers
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
    print(f"Model: Attention LSTM {MODEL_CONFIG['lstm_units']} + temporal softmax attention")
    print(f"Explainability: {'ENABLED (permutation importance + integrated gradients)' if not args.skip_explainability else 'DISABLED'}")
    print(f"Requested experiments: {len(experiment_keys)}")
    for location, year in experiment_keys:
        print(f"  - {location} | {year}")

    summary_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, object]] = []
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
            X_train_scaled, y_train_scaled, ts_train, seq_length, max_gap_min,
        )
        X_test_seq, y_test_seq = create_sequences_gap_aware(
            X_test_scaled, y_test_scaled, ts_test, seq_length, max_gap_min,
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

        # ----- Build & train the attention LSTM -----
        tf.keras.backend.clear_session()
        model = build_attention_lstm_model((seq_length, X_train_seq.shape[2]))

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

        # ----- Baselines -----
        naive_pred = test_level_history[:, -1, 0]
        naive_metrics = evaluate_regression_metrics(y_test_raw_seq, naive_pred)

        linear_baseline = LinearRegression()
        linear_baseline.fit(train_level_history[:, :, 0], y_train_raw_seq)
        linear_pred = linear_baseline.predict(test_level_history[:, :, 0])
        linear_metrics = evaluate_regression_metrics(y_test_raw_seq, linear_pred)

        # ----- Attention weights -----
        attention_weights = extract_attention_weights(model, X_test_seq)

        # ----- Explainability -----
        importance_csv_path = ""
        importance_plot_path = ""
        local_explanation_count = 0
        local_explanation_dir = ""
        local_explanation_summary = ""

        if not args.skip_explainability:
            print("Computing permutation feature importance...")
            importance_df = compute_permutation_feature_importance(
                model, X_test_seq, y_test_seq, target_scaler, input_feature_columns,
            )
            explainability_paths = save_explainability_artifacts(
                output_dir, importance_df, location, year,
            )
            importance_csv_path = str(explainability_paths["importance_csv"].resolve())
            importance_plot_path = str(explainability_paths["importance_plot"].resolve())

            print("Computing integrated gradient local explanations...")
            local_explanations = compute_local_explanations(
                model, X_test_seq, y_true, y_pred, input_feature_columns,
            )
            local_artifacts = save_local_explanation_artifacts(
                output_dir, local_explanations, location, year, input_feature_columns,
            )
            local_explanation_count = len(local_artifacts)
            local_explanation_dir = str((output_dir / "explainability" / str(year)).resolve())
            local_explanation_summary = format_local_explanation_summary(local_explanations)

            print(f"  Feature importance: top feature = {importance_df.iloc[0]['feature']} "
                  f"(RMSE increase = {importance_df.iloc[0]['rmse_increase']:.4f})")

        elapsed_seconds = time.perf_counter() - started_at
        run_count += 1

        print(
            f"  Done [{run_count}] Attention LSTM RMSE={lstm_metrics['rmse']:.4f}, "
            f"R²={lstm_metrics['r2']:.4f}, elapsed={elapsed_seconds / 60:.2f} min"
        )

        # ----- Plots -----
        attention_prediction_plot = ""
        attention_weight_plot = ""

        if not args.skip_plots:
            attention_plot_paths = save_attention_plots(
                output_dir, location, year, y_true, y_pred, attention_weights,
            )
            attention_prediction_plot = str(attention_plot_paths["prediction_plot"].resolve())
            attention_weight_plot = str(attention_plot_paths["attention_plot"].resolve())

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
            "elapsed_seconds": elapsed_seconds,
            "attention_prediction_plot": attention_prediction_plot,
            "attention_weight_plot": attention_weight_plot,
            "importance_csv": importance_csv_path,
            "importance_plot": importance_plot_path,
            "local_explanation_count": local_explanation_count,
            "local_explanation_dir": local_explanation_dir,
            "local_explanation_summary": local_explanation_summary,
        })

        is_last_experiment = experiment_number == total_experiments
        next_location = None if is_last_experiment else experiment_keys[experiment_number][0]
        if not args.skip_plots and (is_last_experiment or next_location != location):
            saved_plots = save_location_run_plots(output_dir, location, location_plot_data.get(location, []))
            location_plot_data.pop(location, None)

        persist_progress(output_dir, summary_rows, skipped_rows)

        del model, history, linear_baseline, attention_weights
        del X_train_seq, y_train_seq, X_test_seq, y_test_seq
        del train_level_history, y_train_raw_seq, test_level_history, y_test_raw_seq
        del y_pred_scaled, y_pred, y_true, naive_pred, linear_pred
        tf.keras.backend.clear_session()
        gc.collect()

    # ----- Final summaries -----
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["Location", "Year"]).reset_index(drop=True)

    skipped_df = pd.DataFrame(skipped_rows)
    if not skipped_df.empty:
        skipped_df = skipped_df.sort_values(["Location", "Year"]).reset_index(drop=True)

    timestamp_label = time.strftime("%Y%m%d_%H%M%S")
    summary_csv = output_dir / f"attention_lstm_60stepsback_summary_{timestamp_label}.csv"
    skipped_csv = output_dir / f"attention_lstm_60stepsback_skipped_{timestamp_label}.csv"

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
        print(f"\nSummary CSV: {summary_csv.resolve()}")
    if not skipped_df.empty:
        print("\nSkipped combinations:")
        print(skipped_df.to_string(index=False))
        print(f"\nSkipped CSV: {skipped_csv.resolve()}")


if __name__ == "__main__":
    main()
