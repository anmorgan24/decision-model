"""Score a trained run, or an untrained baseline, on a split.

    python -m decision_model.evaluate outputs/decision_head --split test
    python -m decision_model.evaluate --config configs/logit_readout.yaml --zero-shot --split val

Results are written next to the run and, when Comet is on, logged to the run's
own Comet experiment. Score the test split once, after every decision about the
models has been made on validation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .data import load_typed_decisions
from .inference import run_model
from .metrics import TemperatureScaler, reliability_bins, summarize
from .schema import load_jsonl
from .tracking import Tracker
from .train import TrainConfig, build_model, load_data, load_tokenizer, resolve_device, resolve_dtype


def load_split(cfg: TrainConfig, split: str, data_path: str | None):
    if data_path:
        return load_jsonl(data_path)
    if split == "val":
        return load_data(cfg)[1]  # the same seeded split used in training
    return load_typed_decisions(split, cfg.workflows)


def load_trained(run_dir: Path, cfg: TrainConfig, model) -> TemperatureScaler:
    state = torch.load(run_dir / "trainable.pt", map_location="cpu")
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(f"Checkpoint has keys this model doesn't: {result.unexpected_keys[:5]}")
    return TemperatureScaler(json.loads((run_dir / "temperatures.json").read_text()))


def print_overall(name: str, summary: dict) -> None:
    o = summary["overall"]
    print(f"{name:12} accuracy {o['accuracy']:.4f}  kl {o['kl']:.4f}  log_loss {o['log_loss']:.4f}  "
          f"brier {o['brier']:.4f}  ece {o['ece']:.4f}  (n={o['n']})")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", nargs="?", help="a directory written by decision_model.train")
    parser.add_argument("--config", help="config for --zero-shot runs")
    parser.add_argument("--zero-shot", action="store_true", help="evaluate the untrained model")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--data", help="score a local JSONL file instead")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)

    if args.zero_shot:
        cfg = TrainConfig.load(args.config, args.set + ["lora_r=0"])
        run_dir = cfg.run_dir.with_name(cfg.run_name + "_zero_shot")
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = Path(args.run_dir)
        cfg = TrainConfig.load(run_dir / "config.yaml", args.set)

    device = resolve_device(cfg.device)
    tokenizer, pad_id = load_tokenizer(cfg)
    model = build_model(cfg, tokenizer, dtype=resolve_dtype(cfg.dtype, device)).to(device)
    temperatures = None if args.zero_shot else load_trained(run_dir, cfg, model)

    examples = load_split(cfg, args.split, args.data)
    if args.split == "test":
        print("Scoring the test split. Report these numbers; don't tune anything against them.")
    outputs = run_model(model, examples, tokenizer, pad_id, cfg.batch_size, cfg.max_options, cfg.max_state_tokens, device)

    results = {"raw": summarize(outputs)}
    if temperatures:
        results["calibrated"] = summarize(outputs, temperatures)
    (run_dir / f"results_{args.split}.json").write_text(json.dumps(results, indent=2))
    for name, summary in results.items():
        print_overall(name, summary)

    if cfg.comet:
        key = None if args.zero_shot else json.loads((run_dir / "run.json").read_text()).get("comet_experiment_key")
        tracker = Tracker.start(True, cfg.comet_project, run_dir.name, cfg.tags + [cfg.model_kind], cfg.comet_offline_dir, key)
        for name, summary in results.items():
            prefix = f"{args.split}" if name == "raw" else f"{args.split}_calibrated"
            tracker.log_summary(summary, prefix)
            logits = outputs.logits if name == "raw" else temperatures.apply(outputs.logits, outputs.question_type)
            tracker.log_reliability_diagram(f"{prefix}/reliability", reliability_bins(logits, outputs.labels))
        tracker.end()


if __name__ == "__main__":
    main()
