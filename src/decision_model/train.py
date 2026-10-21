"""Train a decision model, or the logit-readout baseline, and track the run in Comet.

    python -m decision_model.train configs/decision_head.yaml
    python -m decision_model.train configs/logit_readout.yaml --set epochs=1 comet=false

The test split is never touched here. Score it once, at the end, with
`python -m decision_model.evaluate <run dir> --split test`.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import torch
import yaml

from .data import hard_labels, load_typed_decisions, shuffle_choice_options, split_train_val
from .inference import run_model
from .metrics import TemperatureScaler, reliability_bins, summarize
from .model import DecisionModel, add_lora, decision_loss, freeze
from .packing import collate, pack_example, to_device
from .readout import LogitReadoutModel, answer_token_ids
from .schema import Example, load_jsonl
from .tracking import Tracker


@dataclass
class TrainConfig:
    run_name: str = "decision_head"
    model_kind: str = "decision_head"  # "decision_head" or "logit_readout"
    backbone: str = "Qwen/Qwen3-0.6B-Base"
    lora_r: int = 16  # 0 freezes the backbone (a linear probe)
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    readout_loss: str = "restricted"  # logit_readout only: "restricted" or "full_vocab"
    label_mode: str = "soft"  # "soft" trains on gold distributions, "hard" on one-hot labels
    shuffle_options: bool = True
    epochs: int = 3
    batch_size: int = 4
    grad_accum: int = 4
    lr_backbone: float = 2e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.0
    warmup_ratio: float = 0.06
    max_grad_norm: float = 1.0
    patience: int = 2  # epochs without validation improvement before stopping
    max_options: int = 16
    max_state_tokens: int = 512
    val_fraction: float = 0.15
    workflows: list[str] | None = None
    data_path: str | None = None  # a local JSONL file instead of the benchmark
    max_train_examples: int | None = None  # cap for quick runs
    seed: int = 0
    device: str = "auto"
    dtype: str = "auto"
    attn_implementation: str = "sdpa"
    gradient_checkpointing: bool = False
    output_dir: str = "outputs"
    comet: bool = True
    comet_project: str = "decision-model"
    comet_offline_dir: str | None = None
    tags: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path, overrides: list[str] | None = None) -> "TrainConfig":
        values = yaml.safe_load(Path(path).read_text()) or {}
        for item in overrides or []:
            key, _, raw = item.partition("=")
            values[key] = yaml.safe_load(raw)
        return cls(**values)

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.run_name


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name != "auto":
        return getattr(torch, name)
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def load_tokenizer(cfg: TrainConfig):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.backbone)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    return tokenizer, pad_id


def build_model(cfg: TrainConfig, tokenizer, pretrained=None, dtype=torch.float32) -> torch.nn.Module:
    """pretrained lets tests pass in a tiny model; otherwise the backbone is downloaded."""
    if cfg.model_kind == "decision_head":
        from transformers import AutoModel

        backbone = pretrained or AutoModel.from_pretrained(cfg.backbone, dtype=dtype, attn_implementation=cfg.attn_implementation)
    elif cfg.model_kind == "logit_readout":
        from transformers import AutoModelForCausalLM

        backbone = pretrained or AutoModelForCausalLM.from_pretrained(cfg.backbone, dtype=dtype, attn_implementation=cfg.attn_implementation)
    else:
        raise ValueError(f"Unknown model_kind {cfg.model_kind!r}")

    if cfg.gradient_checkpointing:
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    backbone = add_lora(backbone, cfg.lora_r, cfg.lora_alpha, cfg.lora_dropout) if cfg.lora_r > 0 else freeze(backbone)

    if cfg.model_kind == "decision_head":
        return DecisionModel(backbone, cfg.max_options)
    return LogitReadoutModel(backbone, answer_token_ids(tokenizer, cfg.max_options), cfg.readout_loss)


def load_data(cfg: TrainConfig) -> tuple[list[Example], list[Example]]:
    examples = load_jsonl(cfg.data_path) if cfg.data_path else load_typed_decisions("train", cfg.workflows)
    train, val = split_train_val(examples, cfg.val_fraction, cfg.seed)
    if cfg.max_train_examples:
        train = train[: cfg.max_train_examples]
    if cfg.label_mode == "hard":
        train = [hard_labels(e) for e in train]
    return train, val


def trainable_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if k in names}


def make_optimizer(model: torch.nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    head = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("head.")]
    rest = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("head.")]
    groups = [g for g in ({"params": head, "lr": cfg.lr_head}, {"params": rest, "lr": cfg.lr_backbone}) if g["params"]]
    return torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)


def warmup_cosine(total_steps: int, warmup_ratio: float):
    warmup = max(1, int(total_steps * warmup_ratio))

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    return factor


def option_names(examples: list[Example]) -> dict[str, list[str]]:
    return {f"{e.workflow}/{q.qid}".lstrip("/"): list(q.options) for e in examples for q in e.questions}


def log_evaluation(tracker: Tracker, outputs, examples: list[Example], prefix: str, step: int,
                   temperatures: TemperatureScaler | None = None, confusion: bool = False) -> dict:
    summary = summarize(outputs, temperatures)
    tracker.log_summary(summary, prefix, step=step)
    logits = outputs.logits if temperatures is None else temperatures.apply(outputs.logits, outputs.question_type)
    tracker.log_reliability_diagram(f"{prefix}/reliability", reliability_bins(logits, outputs.targets), step=step)
    if confusion:
        names = option_names(examples)
        predicted = logits.argmax(dim=1).tolist()
        for qid in sorted(set(outputs.question_ids)):
            rows = [i for i, q in enumerate(outputs.question_ids) if q == qid and outputs.labels[i] != -100]
            tracker.log_confusion_matrix(f"{prefix}/confusion/{qid}", names[qid],
                                         [int(outputs.labels[i]) for i in rows], [predicted[i] for i in rows], step=step)
    return summary


def train(cfg: TrainConfig, model, tokenizer, pad_id: int, train_set: list[Example], val_set: list[Example],
          tracker: Tracker, device: torch.device) -> tuple[dict, int, int]:
    """Returns the best trainable weights, the epoch they came from, and the total optimizer steps."""
    rng = random.Random(cfg.seed)
    optimizer = make_optimizer(model, cfg)
    micro_batches = math.ceil(len(train_set) / cfg.batch_size)
    total_steps = cfg.epochs * math.ceil(micro_batches / cfg.grad_accum)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine(total_steps, cfg.warmup_ratio))

    best_loss, best_state, best_epoch, stale, step = float("inf"), trainable_state(model), 0, 0, 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        order = list(range(len(train_set)))
        rng.shuffle(order)
        for i in range(micro_batches):
            chunk = [train_set[j] for j in order[i * cfg.batch_size:(i + 1) * cfg.batch_size]]
            if cfg.shuffle_options:
                chunk = [shuffle_choice_options(e, rng) for e in chunk]
            batch = to_device(collate(chunk, [pack_example(e, tokenizer, cfg.max_state_tokens) for e in chunk],
                                      pad_id, cfg.max_options), device)
            loss = decision_loss(model(batch), batch["targets"], len(chunk), model.loss_normalizes)
            (loss / cfg.grad_accum).backward()

            if (i + 1) % cfg.grad_accum == 0 or i + 1 == micro_batches:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1
                tracker.log_metrics({"loss": loss.item(), "lr": scheduler.get_last_lr()[-1]}, step=step, epoch=epoch, prefix="train")

        outputs = run_model(model, val_set, tokenizer, pad_id, cfg.batch_size, cfg.max_options, cfg.max_state_tokens, device)
        summary = log_evaluation(tracker, outputs, val_set, "val", step)
        val_loss = summary["overall"]["log_loss"]
        print(f"epoch {epoch}: val log_loss {val_loss:.4f}  accuracy {summary['overall']['accuracy']:.4f}  "
              f"kl {summary['overall']['kl']:.4f}  ece {summary['overall']['ece']:.4f}")

        if val_loss < best_loss:
            best_loss, best_state, best_epoch, stale = val_loss, trainable_state(model), epoch, 0
        else:
            stale += 1
            if stale >= cfg.patience:
                print(f"stopping early: no improvement for {stale} epochs")
                break
    return best_state, best_epoch, step


def save_run(cfg: TrainConfig, state: dict, temperatures: TemperatureScaler, val_summary: dict,
             splits: dict[str, list[str]], run_info: dict) -> Path:
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(dataclasses.asdict(cfg), sort_keys=False))
    torch.save(state, run_dir / "trainable.pt")
    (run_dir / "temperatures.json").write_text(json.dumps(temperatures.temperatures, indent=2))
    (run_dir / "val_summary.json").write_text(json.dumps(val_summary, indent=2))
    (run_dir / "split_ids.json").write_text(json.dumps(splits, indent=2))
    (run_dir / "run.json").write_text(json.dumps(run_info, indent=2))
    return run_dir


def run(cfg: TrainConfig, tokenizer=None, pad_id: int | None = None, pretrained=None,
        data: tuple[list[Example], list[Example]] | None = None) -> Path:
    """The whole pipeline. The optional arguments let tests inject a tiny model and data."""
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    dtype = resolve_dtype(cfg.dtype, device)
    if tokenizer is None:
        tokenizer, pad_id = load_tokenizer(cfg)

    tracker = Tracker.start(cfg.comet, cfg.comet_project, cfg.run_name, cfg.tags + [cfg.model_kind], cfg.comet_offline_dir)
    tracker.log_params(dataclasses.asdict(cfg))

    train_set, val_set = data or load_data(cfg)
    splits = {"train": [e.id for e in train_set], "val": [e.id for e in val_set]}
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    (cfg.run_dir / "split_ids.json").write_text(json.dumps(splits, indent=2))
    tracker.log_dataset_artifact("typed-decisions-splits", [cfg.run_dir / "split_ids.json"],
                                 {"source": cfg.data_path or "LocalLLaMA/typed-decisions", "seed": cfg.seed,
                                  "val_fraction": cfg.val_fraction, "train": len(train_set), "val": len(val_set)})

    model = build_model(cfg, tokenizer, pretrained, dtype).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tracker.log_other("trainable_parameters", n_trainable)
    print(f"{cfg.model_kind}: {n_trainable:,} trainable parameters, {len(train_set)} train / {len(val_set)} val examples")

    best_state, best_epoch, steps = train(cfg, model, tokenizer, pad_id, train_set, val_set, tracker, device)
    model.load_state_dict(best_state, strict=False)

    outputs = run_model(model, val_set, tokenizer, pad_id, cfg.batch_size, cfg.max_options, cfg.max_state_tokens, device)
    temperatures = TemperatureScaler.fit(outputs)
    tracker.log_metrics({f"temperature_{k}": v for k, v in temperatures.temperatures.items()})
    raw = log_evaluation(tracker, outputs, val_set, "val_best", steps, confusion=True)
    calibrated = log_evaluation(tracker, outputs, val_set, "val_best_calibrated", steps, temperatures)

    run_dir = save_run(cfg, best_state, temperatures, {"raw": raw, "calibrated": calibrated}, splits,
                       {"comet_experiment_key": tracker.key, "best_epoch": best_epoch, "optimizer_steps": steps})
    tracker.log_model(cfg.run_name, run_dir, {"model_kind": cfg.model_kind, "backbone": cfg.backbone, "best_epoch": best_epoch})
    tracker.end()
    print(f"saved {run_dir}  (temperatures: {temperatures.temperatures})")
    return run_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="YAML config file")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config values")
    args = parser.parse_args(argv)
    run(TrainConfig.load(args.config, args.set))


if __name__ == "__main__":
    main()
