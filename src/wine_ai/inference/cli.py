"""CLI for sampling wines from the dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .generators import GenerationRequest, generate_wines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wine-type", type=str, default=None)
    parser.add_argument("--price-low", type=float, default=None)
    parser.add_argument("--price-high", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--dataset", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    price_range = None
    if args.price_low is not None or args.price_high is not None:
        price_range = (
            args.price_low if args.price_low is not None else 0.0,
            args.price_high if args.price_high is not None else float("inf"),
        )
    request = GenerationRequest(
        wine_type=args.wine_type,
        price_range=price_range,
        seed=args.seed,
        count=args.count,
        dataset_path=args.dataset,
    )
    print(json.dumps(generate_wines(request), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
