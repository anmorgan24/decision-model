"""Pack one state and several questions into a single sequence.

Layout:  [state][question 1 ... Answer:][question 2 ... Answer:] ...

Every question branch can attend to the state and to itself, never to another
branch. Each branch's position ids restart right after the state, so a branch
sees exactly what it would see in a standalone `[state][question]` sequence.
The answer is read from the hidden state at the last token of each branch
(the "decision token").

Packing is about cost, not accuracy: each question's hidden states are
identical to asking it alone (tests/test_isolation.py checks this), but the
state is encoded once however many questions there are.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .schema import Example, Question

STATE_TEMPLATE = "Input:\n{state}\n"
QUESTION_TYPE_CODES = {"choice": 0, "score": 1, "noul": 2}
OPTION_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def format_question(question: Question) -> str:
    """Render a question as text. Every branch ends with 'Answer:'."""
    if question.type == "noul":
        lines = [f"\nYes or no: {question.instructions}"]
        no_desc, yes_desc = question.descriptions
        if yes_desc or no_desc:
            lines += [f"Yes means: {yes_desc}", f"No means: {no_desc}"]
    else:
        lines = [f"\nQuestion: {question.instructions}"]
        for letter, label, desc in zip(OPTION_LETTERS, question.options, question.descriptions):
            lines.append(f"{letter}) {label}: {desc}" if desc else f"{letter}) {label}")
    lines.append("Answer:")
    return "\n".join(lines)


@dataclass
class Packed:
    input_ids: list[int]
    position_ids: list[int]
    segment_ids: list[int]  # 0 = state, i + 1 = question i
    decide_index: list[int]  # position of each question's decision token


def pack_example(example: Example, tokenizer, max_state_tokens: int = 512) -> Packed:
    state_ids = tokenizer.encode(STATE_TEMPLATE.format(state=example.state), add_special_tokens=False)
    state_ids = state_ids[:max_state_tokens]
    state_len = len(state_ids)

    input_ids, position_ids = list(state_ids), list(range(state_len))
    segment_ids, decide_index = [0] * state_len, []

    for i, question in enumerate(example.questions, start=1):
        branch = tokenizer.encode(format_question(question), add_special_tokens=False)
        input_ids += branch
        position_ids += range(state_len, state_len + len(branch))  # restart after the state
        segment_ids += [i] * len(branch)
        decide_index.append(len(input_ids) - 1)  # last token of the branch

    return Packed(input_ids, position_ids, segment_ids, decide_index)


def build_attention_mask(segment_ids: list[int]) -> torch.Tensor:
    """Boolean (T, T) matrix: True where token t may attend to token u."""
    seg = torch.tensor(segment_ids)
    causal = torch.ones(len(seg), len(seg), dtype=torch.bool).tril()
    same_branch = seg[:, None] == seg[None, :]
    is_state = (seg == 0)[None, :]
    return causal & (same_branch | is_state)


def additive_mask(allowed: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Turn a boolean mask into the additive float mask the backbone expects.

    In transformers 5, eager attention misreads a boolean 4D mask, while an
    additive float mask works with every attention implementation. The blocked
    value is the most negative number of the backbone's own dtype.
    """
    return torch.zeros(allowed.shape, dtype=dtype, device=allowed.device).masked_fill(~allowed, torch.finfo(dtype).min)


def collate(examples: list[Example], packed: list[Packed], pad_id: int, max_options: int = 16) -> dict:
    """Batch packed examples, with one row per question for the head and the loss."""
    batch_size, max_len = len(packed), max(len(p.input_ids) for p in packed)

    input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    position_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
    allowed = torch.zeros((batch_size, 1, max_len, max_len), dtype=torch.bool)

    rows = []
    for b, (example, p) in enumerate(zip(examples, packed)):
        n = len(p.input_ids)
        input_ids[b, :n] = torch.tensor(p.input_ids)
        position_ids[b, :n] = torch.tensor(p.position_ids)
        allowed[b, 0, :n, :n] = build_attention_mask(p.segment_ids)
        pad = torch.arange(n, max_len)
        allowed[b, 0, pad, pad] = True  # padding attends to itself, so no row is fully blocked

        for question, pos in zip(example.questions, p.decide_index):
            if question.num_options > max_options:
                raise ValueError(f"{question.qid} has {question.num_options} options; max_options is {max_options}")
            target = list(example.gold.get(question.qid, ())) + [0.0] * (max_options - len(example.gold.get(question.qid, ())))
            rows.append((b, pos, QUESTION_TYPE_CODES[question.type], question.num_options,
                         target, example.label.get(question.qid, -100), f"{example.workflow}/{question.qid}".lstrip("/")))

    b_idx, pos, qtype, n_opts, targets, labels, qids = zip(*rows)
    return {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "attention_allowed": allowed,
        "question_batch_index": torch.tensor(b_idx),
        "question_position": torch.tensor(pos),
        "question_type": torch.tensor(qtype),
        "num_options": torch.tensor(n_opts),
        "targets": torch.tensor(targets, dtype=torch.float32),  # all zeros when unlabeled
        "labels": torch.tensor(labels),  # -100 when unlabeled
        "question_ids": list(qids),
    }


def to_device(batch: dict, device) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
