from __future__ import annotations

import argparse
from pathlib import Path

import lstm_60stepsback_attention_uniformscaling_batch as base_batch


MODEL_CONFIG_OVERRIDE = {
    "lstm_units": [64, 32, 16],
}

OUTPUT_DIR = Path("analysis_outputs") / "lstm_60stepsback_attention_3layer_uniformscaling_batch"
PROGRESS_FILES = {
    "completed": "completed_runs_latest.csv",
    "skipped": "skipped_runs_latest.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Attention LSTM-60stepsback batch pipeline with a 3-layer LSTM model."
    )
    parser.add_argument("--location", nargs="+", help="Optional subset of locations to run.")
    parser.add_argument("--year", nargs="+", type=int, help="Optional subset of years to run.")
    parser.add_argument("--resume", action="store_true", help="Resume from the latest progress files and skip completed location-year runs.")
    parser.add_argument("--start-at-location", type=str, help="Optional location at which to start the ordered run list.")
    parser.add_argument("--start-at-year", type=int, help="Optional year paired with --start-at-location for the first combination to run.")
    parser.add_argument("--epochs", type=int, default=base_batch.TRAINING_CONFIG["epochs"], help="Training epochs per run.")
    parser.add_argument("--batch-size", type=int, default=base_batch.TRAINING_CONFIG["batch_size"], help="Batch size per run.")
    parser.add_argument("--seq-length", type=int, default=base_batch.TRAINING_CONFIG["seq_length"], help="Sequence length.")
    parser.add_argument("--max-gap-min", type=int, default=base_batch.TRAINING_CONFIG["max_gap_min"], help="Maximum allowed time gap in minutes inside a sequence.")
    parser.add_argument("--test-fraction", type=float, default=base_batch.TRAINING_CONFIG["test_fraction"], help="Chronological test fraction.")
    parser.add_argument("--skip-plots", action="store_true", help="Skip per-run and summary plot generation to maximize throughput.")
    parser.add_argument("--skip-explainability", action="store_true", help="Skip permutation importance and integrated-gradient explanations.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Directory for summaries and plots.")
    return parser.parse_args()


def configure_three_layer_run() -> None:
    base_batch.MODEL_CONFIG.update(MODEL_CONFIG_OVERRIDE)
    base_batch.OUTPUT_CONFIG["output_dir"] = OUTPUT_DIR
    base_batch.PROGRESS_FILES.update(PROGRESS_FILES)
    base_batch.parse_args = parse_args


def main() -> None:
    configure_three_layer_run()
    print(f"Configured 3-layer Attention LSTM units: {base_batch.MODEL_CONFIG['lstm_units']}")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")
    base_batch.main()


if __name__ == "__main__":
    main()
