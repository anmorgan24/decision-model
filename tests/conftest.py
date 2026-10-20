"""Fixtures that run offline: tiny randomly initialized Qwen3 models and a toy tokenizer.

The tests check structure (masking, positions, isolation, the training loop),
not quality, so random models are enough. The same code runs unchanged on
Qwen/Qwen3-0.6B-Base.
"""

import json
import re
import zlib
from pathlib import Path

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM, Qwen3Model

from decision_model import load_jsonl, parse_typed_decisions_row

VOCAB_SIZE = 512
PAD_ID = 0
ROOT = Path(__file__).parent.parent


class ToyTokenizer:
    """Splits on words and punctuation and hashes each piece to a stable id."""

    pad_token_id = PAD_ID

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        pieces = re.findall(r"\w+|[^\w\s]|\n", text)
        return [1 + zlib.crc32(p.encode()) % (VOCAB_SIZE - 1) for p in pieces]


def tiny_config(attn_implementation: str = "sdpa") -> Qwen3Config:
    config = Qwen3Config(
        vocab_size=VOCAB_SIZE, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=2048,
    )
    config._attn_implementation = attn_implementation
    return config


def tiny_qwen3(attn_implementation: str = "sdpa") -> Qwen3Model:
    torch.manual_seed(0)
    return Qwen3Model(tiny_config(attn_implementation)).eval()


def tiny_qwen3_lm(attn_implementation: str = "sdpa") -> Qwen3ForCausalLM:
    torch.manual_seed(0)
    return Qwen3ForCausalLM(tiny_config(attn_implementation)).eval()


@pytest.fixture
def tokenizer():
    return ToyTokenizer()


@pytest.fixture
def examples():
    """Three hand-written tickets with hard labels."""
    return load_jsonl(ROOT / "data" / "sample.jsonl")


@pytest.fixture
def typed_examples():
    """Two rows shaped like LocalLLaMA/typed-decisions, with soft gold distributions."""
    with open(ROOT / "tests" / "fixtures" / "typed_decisions_rows.jsonl") as f:
        return [parse_typed_decisions_row(json.loads(line)) for line in f]


@pytest.fixture(params=["eager", "sdpa"])
def backbone(request):
    return tiny_qwen3(request.param)
