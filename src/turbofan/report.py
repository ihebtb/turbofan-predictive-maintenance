"""CLI: investigation report for one engine of the official test set.

    python -m turbofan.report --subset FD001 --unit 24
"""

import argparse

from .data import DEFAULT_ROOT, load_subset
from .service import EngineAssessor, investigation_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", default="FD001")
    parser.add_argument("--unit", type=int, default=1)
    parser.add_argument("--models", default="models")
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    args = parser.parse_args()

    _, test, test_rul = load_subset(args.subset, args.root)
    history = test[test["unit"] == args.unit]
    if history.empty:
        raise SystemExit(f"unit {args.unit} not in {args.subset} test set")
    result = EngineAssessor(args.models, args.subset).assess(history)
    print(investigation_report(result))
    print(f"- True remaining life (official label): {int(test_rul[args.unit - 1])} cycles")


if __name__ == "__main__":
    main()
