"""How much does packing save? Packed questions versus one sequence per question.

Packing gives every question the same hidden states it would get alone (see
tests/test_isolation.py), so this measures efficiency only: tokens processed
and wall-clock time for the same answers.

    python -m decision_model.bench_latency configs/decision_head.yaml --examples 32
"""

from __future__ import annotations

import argparse
import dataclasses
import statistics
import time

import torch

from .packing import collate, pack_example, to_device
from .schema import Example


def unpack(example: Example) -> list[Example]:
    """One example per question, each repeating the full state."""
    return [dataclasses.replace(example, questions=(q,)) for q in example.questions]


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def time_forward(model, examples: list[Example], tokenizer, pad_id: int, max_options: int,
                 device: torch.device, repeats: int = 5) -> tuple[float, int]:
    """Median seconds for one forward pass over these examples as a single batch, and tokens processed."""
    packed = [pack_example(e, tokenizer) for e in examples]
    batch = to_device(collate(examples, packed, pad_id, max_options), device)
    model(batch)  # warm up
    times = []
    for _ in range(repeats):
        _sync(device)
        start = time.perf_counter()
        model(batch)
        _sync(device)
        times.append(time.perf_counter() - start)
    return statistics.median(times), sum(len(p.input_ids) for p in packed)


def compare(model, examples: list[Example], tokenizer, pad_id: int, max_options: int, device) -> dict:
    """Per example: packed (one sequence, all questions) versus unpacked (one sequence per question)."""
    rows = []
    for example in examples:
        packed_s, packed_tokens = time_forward(model, [example], tokenizer, pad_id, max_options, device)
        unpacked_s, unpacked_tokens = time_forward(model, unpack(example), tokenizer, pad_id, max_options, device)
        rows.append((packed_s, unpacked_s, packed_tokens, unpacked_tokens))
    packed_s, unpacked_s, packed_tokens, unpacked_tokens = (statistics.median(col) for col in zip(*rows))
    return {
        "questions_per_example": statistics.median(len(e.questions) for e in examples),
        "packed_ms": packed_s * 1000,
        "unpacked_ms": unpacked_s * 1000,
        "packed_tokens": packed_tokens,
        "unpacked_tokens": unpacked_tokens,
        "token_ratio": unpacked_tokens / packed_tokens,
    }


def main(argv: list[str] | None = None) -> None:
    from .train import TrainConfig, build_model, load_data, load_tokenizer, resolve_device, resolve_dtype

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config")
    parser.add_argument("--examples", type=int, default=32)
    args = parser.parse_args(argv)

    cfg = TrainConfig.load(args.config, ["lora_r=0"])  # untrained weights time the same as trained ones
    device = resolve_device(cfg.device)
    tokenizer, pad_id = load_tokenizer(cfg)
    model = build_model(cfg, tokenizer, dtype=resolve_dtype(cfg.dtype, device)).to(device).eval()
    examples = load_data(cfg)[1][: args.examples]
    result = compare(model, examples, tokenizer, pad_id, cfg.max_options, device)
    for key, value in result.items():
        print(f"{key:24}{value:10.2f}")


if __name__ == "__main__":
    main()
