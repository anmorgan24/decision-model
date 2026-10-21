"""Is one run better than another, or is the difference noise? A paired bootstrap over cases.

    python -m decision_model.compare outputs/logit_readout outputs/decision_head --split val
    python -m decision_model.compare outputs/logit_readout_zero_shot outputs/logit_readout --split val

Both runs must have been scored on the same split with decision_model.evaluate,
which saves predictions_<split>.pt. Each bootstrap sample redraws cases (not
individual decisions, since one case's five answers are correlated) and
recomputes every metric for both runs on the same cases. The interval is the
middle 95% of the differences.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .metrics import METRICS, TYPE_NAMES

LOWER_IS_BETTER = {"log_loss", "kl", "brier", "brier_label", "ece", "ece_label"}
DEFAULT_METRICS = ["accuracy", "kl", "log_loss", "brier", "ece"]


def load_predictions(run_dir: str | Path, split: str, calibrated: bool) -> dict:
    preds = torch.load(Path(run_dir) / f"predictions_{split}.pt", weights_only=False)
    if calibrated:
        if preds["calibrated_logits"] is None:
            raise ValueError(f"{run_dir} has no calibrated predictions (zero-shot runs don't fit temperatures)")
        preds["logits"] = preds["calibrated_logits"]
    return preds


def check_aligned(a: dict, b: dict) -> None:
    if a["example_ids"] != b["example_ids"] or a["question_ids"] != b["question_ids"]:
        raise ValueError("The two runs were scored on different examples or in a different order")
    if not torch.equal(a["labels"], b["labels"]):
        raise ValueError("The two runs have different gold labels")


def bootstrap(a: dict, b: dict, metrics: list[str], rows: torch.Tensor | None = None,
              samples: int = 2000, seed: int = 0) -> dict[str, dict]:
    """For each metric: run A, run B, B minus A, and a 95% interval for B minus A."""
    keep = a["labels"] != -100
    if rows is not None:
        keep &= rows
    cases = a["case_index"][keep]
    data = {name: (p["logits"][keep], p["targets"][keep], p["labels"][keep]) for name, p in (("a", a), ("b", b))}

    unique_cases = cases.unique()
    rows_by_case = [torch.nonzero(cases == c).squeeze(1) for c in unique_cases]
    generator = torch.Generator().manual_seed(seed)

    diffs = {m: [] for m in metrics}
    for _ in range(samples):
        drawn = torch.randint(len(unique_cases), (len(unique_cases),), generator=generator)
        idx = torch.cat([rows_by_case[i] for i in drawn.tolist()])
        for m in metrics:
            fn = METRICS[m]
            diffs[m].append(fn(*(t[idx] for t in data["b"])) - fn(*(t[idx] for t in data["a"])))

    result = {}
    for m in metrics:
        d = torch.tensor(diffs[m])
        value_a, value_b = METRICS[m](*data["a"]), METRICS[m](*data["b"])
        low, high = d.quantile(0.025).item(), d.quantile(0.975).item()
        better = "B" if (high < 0 if m in LOWER_IS_BETTER else low > 0) else "A" if (low > 0 if m in LOWER_IS_BETTER else high < 0) else "neither"
        result[m] = {"a": value_a, "b": value_b, "diff": value_b - value_a, "low": low, "high": high, "better": better}
    return result


def print_table(title: str, result: dict) -> None:
    print(f"\n{title}")
    print(f"{'metric':10}{'A':>9}{'B':>9}{'B - A':>9}   95% interval      clearly better")
    for m, r in result.items():
        print(f"{m:10}{r['a']:9.4f}{r['b']:9.4f}{r['diff']:+9.4f}   [{r['low']:+.4f}, {r['high']:+.4f}]   {r['better']}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_a")
    parser.add_argument("run_b")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--calibrated", action="store_true", help="compare after temperature scaling")
    parser.add_argument("--by-type", action="store_true", help="also compare per question type")
    parser.add_argument("--metrics", nargs="*", default=DEFAULT_METRICS, choices=sorted(METRICS))
    parser.add_argument("--samples", type=int, default=2000)
    args = parser.parse_args(argv)

    a = load_predictions(args.run_a, args.split, args.calibrated)
    b = load_predictions(args.run_b, args.split, args.calibrated)
    check_aligned(a, b)
    print(f"A = {args.run_a}\nB = {args.run_b}\nsplit = {args.split}, {'calibrated' if args.calibrated else 'raw'}, "
          f"{len(set(a['case_index'].tolist()))} cases, {args.samples} bootstrap samples")

    print_table("all questions", bootstrap(a, b, args.metrics, samples=args.samples))
    if args.by_type:
        for code, name in TYPE_NAMES.items():
            rows = a["question_type"] == code
            if rows.any():
                print_table(name, bootstrap(a, b, args.metrics, rows=rows, samples=args.samples))


if __name__ == "__main__":
    main()
