
"""Run financial fragility predictions on a CSV dataset."""

import argparse
from pathlib import Path

import pandas as pd

from src.model import load_model, predict


def main():
    parser = argparse.ArgumentParser(
        description="Predict household financial fragility."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to an input CSV containing model features.",
    )

    parser.add_argument(
        "--output",
        default="outputs/predictions.csv",
        help="Path to save the prediction results.",
    )

    parser.add_argument(
        "--model",
        default="logistic_regression_with_ef",
        help="Name of the trained model to load.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.is_file():
        parser.error(f"Input file not found: {input_path}")

    data = pd.read_csv(input_path, low_memory=False)

    model, metadata = load_model(args.model)

    results = predict(model, metadata, data)

    if "shedid" in data.columns:
        results.insert(0, "shedid", data["shedid"])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results.to_csv(output_path, index=False)

    print(f"Predictions completed for {len(results):,} rows.")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
