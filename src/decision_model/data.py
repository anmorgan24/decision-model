"""Loading the benchmark, splitting it, and augmenting it.

The default dataset is LocalLLaMA/typed-decisions (Apache-2.0): 1,600 synthetic
cases across four workflows, five typed questions each, with soft gold
distributions from a teacher model. Its `test` split is held out; the
validation set is carved out of `train`.
"""

from __future__ import annotations

import dataclasses
import random
from collections import defaultdict

from .schema import Example, parse_typed_decisions_row

DATASET = "LocalLLaMA/typed-decisions"
WORKFLOWS = ("agent_trace_observability", "customer_service", "invoice_processing", "security_incidents")


def load_typed_decisions(split: str, workflows: list[str] | None = None) -> list[Example]:
    from datasets import load_dataset

    rows = load_dataset(DATASET, "all", split=split)
    examples = [parse_typed_decisions_row(row) for row in rows]
    if workflows:
        examples = [e for e in examples if e.workflow in workflows]
    return examples


def split_train_val(examples: list[Example], val_fraction: float, seed: int) -> tuple[list[Example], list[Example]]:
    """Hold out the same fraction of every workflow, deterministically."""
    by_workflow = defaultdict(list)
    for example in examples:
        by_workflow[example.workflow].append(example)

    rng = random.Random(seed)
    train, val = [], []
    for workflow in sorted(by_workflow):
        group = sorted(by_workflow[workflow], key=lambda e: e.id)
        rng.shuffle(group)
        n_val = round(len(group) * val_fraction)
        val += group[:n_val]
        train += group[n_val:]
    return train, val


def shuffle_choice_options(example: Example, rng: random.Random) -> Example:
    """Randomly reorder the options of every choice question, keeping gold aligned.

    The head scores option slots by position, so without this it can learn that
    "billing" is always slot A instead of reading the options. Score levels are
    ordered and yes/no is fixed, so only choice questions are shuffled.
    """
    questions, gold, label = [], dict(example.gold), dict(example.label)
    for q in example.questions:
        if q.type != "choice":
            questions.append(q)
            continue
        order = list(range(q.num_options))
        rng.shuffle(order)
        questions.append(dataclasses.replace(
            q,
            options=tuple(q.options[i] for i in order),
            descriptions=tuple(q.descriptions[i] for i in order),
        ))
        if q.qid in gold:
            gold[q.qid] = tuple(example.gold[q.qid][i] for i in order)
        if q.qid in label:
            label[q.qid] = order.index(example.label[q.qid])
    return dataclasses.replace(example, questions=tuple(questions), gold=gold, label=label)


def hard_labels(example: Example) -> Example:
    """Replace soft gold distributions with one-hot vectors on the gold label."""
    sizes = {q.qid: q.num_options for q in example.questions}
    gold = {qid: tuple(1.0 if i == idx else 0.0 for i in range(sizes[qid])) for qid, idx in example.label.items()}
    return dataclasses.replace(example, gold=gold)
