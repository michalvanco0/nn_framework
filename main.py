"""CLI entrypoint for NN Reliability Framework baseline runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.baseline import BaselineConfig, run_baseline


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for baseline configuration."""
    parser = argparse.ArgumentParser(description="Run CIFAR-10 ResNet-18 baseline experiment")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Training/evaluation batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader worker count")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Dataset directory")
    parser.add_argument("--results-dir", type=Path, default=Path("results"), help="Output directory")
    return parser.parse_args()


def main() -> None:
    """Run configured baseline experiment and print summary metrics."""
    args = parse_args()
    config = BaselineConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        seed=args.seed,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
    )
    result = run_baseline(config)
    print(
        "Evaluation complete:",
        {
            "accuracy": result["accuracy"],
            "precision": result["precision"],
            "recall": result["recall"],
            "f1": result["f1"],
        },
    )


if __name__ == "__main__":
    main()
