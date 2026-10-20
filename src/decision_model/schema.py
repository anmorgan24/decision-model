"""The data format: one example is a state, a set of typed questions, and gold answers.

Rows use the Jev / System One request shape (`state`, `questions`) plus gold
answers, so a row can be replayed against any API with that shape.

Question types follow Jev's names:
  choice: pick one of several named options
  score:  pick a level on an ordered scale
  noul:   yes/no, answered as the probability that a statement is true

Gold answers are stored two ways:
  gold:  a probability for every option. Datasets with soft labels (such as
         typed-decisions, where gold is the mean of three teacher samples) fill
         this in directly; hard labels become one-hot vectors.
  label: the index of the single gold answer, used for accuracy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

QuestionType = Literal["choice", "score", "noul"]


@dataclass(frozen=True)
class Question:
    qid: str
    type: QuestionType
    instructions: str
    options: tuple[str, ...]  # choice: labels; score: level descriptions; noul: ("no", "yes")
    descriptions: tuple[str, ...]  # one per option, "" when there is none

    @property
    def num_options(self) -> int:
        return len(self.options)


@dataclass(frozen=True)
class Example:
    state: str
    questions: tuple[Question, ...]
    gold: dict[str, tuple[float, ...]] = field(default_factory=dict)  # qid -> probability per option
    label: dict[str, int] = field(default_factory=dict)  # qid -> index of the gold answer
    id: str = ""
    workflow: str = ""


def render_state(state) -> str:
    """States can be text or JSON. JSON states are rendered compactly, keys in their given order."""
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def parse_question(qid: str, spec: dict) -> Question:
    qtype = spec["type"]
    criteria = spec.get("criteria")
    if qtype == "choice":
        labels, descriptions = tuple(criteria.keys()), tuple(criteria.values())
    elif qtype == "score":
        # Levels come as a list, or as a dict keyed by level ("0", "1", ...).
        levels = list(criteria.values()) if isinstance(criteria, dict) else list(criteria)
        labels, descriptions = tuple(levels), ("",) * len(levels)
    elif qtype == "noul":
        criteria = criteria or {}
        labels = ("no", "yes")
        descriptions = (criteria.get("false", ""), criteria.get("true", ""))
    else:
        raise ValueError(f"Unknown question type {qtype!r} for {qid!r}")
    return Question(qid, qtype, spec["instructions"], labels, descriptions)


def answer_index(question: Question, answer) -> int:
    """Convert a single gold answer (label, level, or bool) into an option index."""
    if question.type == "choice":
        return question.options.index(answer)
    if question.type == "score":
        return int(answer)
    if isinstance(answer, str):
        return int(answer.lower() == "true")
    return int(bool(answer))


def one_hot(index: int, size: int) -> tuple[float, ...]:
    return tuple(1.0 if i == index else 0.0 for i in range(size))


def parse_row(row: dict) -> Example:
    """A row with hard gold answers: {"state", "questions", "gold": {qid: label | level | bool}}."""
    questions = tuple(parse_question(qid, spec) for qid, spec in row["questions"].items())
    by_id = {q.qid: q for q in questions}
    label = {qid: answer_index(by_id[qid], answer) for qid, answer in row.get("gold", {}).items()}
    gold = {qid: one_hot(i, by_id[qid].num_options) for qid, i in label.items()}
    return Example(render_state(row["state"]), questions, gold, label, row.get("id", ""), row.get("workflow", ""))


def _loads(value):
    return json.loads(value) if isinstance(value, str) else value


def parse_typed_decisions_row(row: dict) -> Example:
    """A row from the LocalLLaMA/typed-decisions benchmark, with soft gold distributions.

    `state`, `questions` and `gold` arrive as JSON strings. Each gold entry holds
    `probabilities` keyed by option (choice: label, score: level, noul: "false"/"true")
    and the gold `label`.
    """
    questions = tuple(parse_question(qid, spec) for qid, spec in _loads(row["questions"]).items())
    gold_specs = _loads(row["gold"])
    gold, label = {}, {}
    for q in questions:
        spec = gold_specs[q.qid]
        probs = spec["probabilities"]
        if q.type == "choice":
            keys = q.options
        elif q.type == "score":
            keys = tuple(str(i) for i in range(q.num_options))
        else:
            keys = ("false", "true")
        vector = [float(probs.get(k, 0.0)) for k in keys]
        total = sum(vector)
        gold[q.qid] = tuple(p / total for p in vector)
        label[q.qid] = answer_index(q, spec["label"])
    return Example(render_state(_loads(row["state"])), questions, gold, label, row.get("id", ""), row.get("workflow", ""))


def load_jsonl(path: str | Path) -> list[Example]:
    with open(path) as f:
        return [parse_row(json.loads(line)) for line in f if line.strip()]
