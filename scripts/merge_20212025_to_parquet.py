from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


DATASETS_ROOT = Path(
    r"c:\PhD\floodprediction\Flood-Prevention-Using-River-Level-Prediction\datasets"
)
YEARS = range(2021, 2026)
CHUNK_SIZE = 250_000
COMPRESSION_CANDIDATES = ("zstd", "snappy", "gzip")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge dataset CSV files from 2021 through 2025 into one parquet file."
    )
    parser.add_argument(
        "dataset",
        choices=("rainfall", "riverlevel", "weather"),
        help="Dataset folder name under the datasets directory.",
    )
    return parser.parse_args()


def choose_compression() -> str | None:
    for codec in COMPRESSION_CANDIDATES:
        if pa.Codec.is_available(codec):
            return codec

    return None


def iter_source_files(source_root: Path) -> list[Path]:
    files: list[Path] = []

    for year in YEARS:
        year_dir = source_root / str(year)
        files.extend(sorted(year_dir.glob("*.csv")))

    return files


def build_output_paths(dataset: str) -> tuple[Path, Path]:
    source_root = DATASETS_ROOT / dataset
    output_file = source_root / f"{dataset}20212025_data.parquet"
    temp_file = source_root / f"{dataset}20212025_data.tmp.parquet"
    return output_file, temp_file


def main() -> None:
    args = parse_args()
    dataset = args.dataset
    source_root = DATASETS_ROOT / dataset
    output_file, temp_file = build_output_paths(dataset)
    files = iter_source_files(source_root)
    writer: pq.ParquetWriter | None = None
    written_rows = 0
    skipped_files = 0
    compression = choose_compression()

    if temp_file.exists():
        temp_file.unlink()

    try:
        for csv_file in files:
            if csv_file.stat().st_size == 0:
                skipped_files += 1
                continue

            for chunk in pd.read_csv(
                csv_file,
                chunksize=CHUNK_SIZE,
                parse_dates=["Timestamp"],
            ):
                if chunk.empty:
                    continue

                table = pa.Table.from_pandas(chunk, preserve_index=False)

                if writer is None:
                    writer = pq.ParquetWriter(
                        temp_file,
                        table.schema,
                        compression=compression,
                    )

                writer.write_table(table)
                written_rows += len(chunk)

            print(f"processed {csv_file.name}")

        if writer is None:
            raise RuntimeError(f"No non-empty CSV files were found for {dataset}.")
    finally:
        if writer is not None:
            writer.close()

    temp_file.replace(output_file)
    print(f"dataset={dataset}")
    print(f"compression={compression or 'none'}")
    print(f"written_rows={written_rows}")
    print(f"skipped_files={skipped_files}")
    print(f"output={output_file}")


if __name__ == "__main__":
    main()
