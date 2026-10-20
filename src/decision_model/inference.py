"""Running a model over examples, and turning its outputs into Jev-shaped answers.

`predict(state, questions)` takes and returns the System One request and
response shapes, so the evaluation harness can call this model, the baseline,
or a hosted model like Jev through the same interface.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .packing import QUESTION_TYPE_CODES, collate, pack_example, to_device
from .schema import Example, parse_question, render_state


@dataclass
class Outputs:
    """One row per question, across a whole set of examples."""

    logits: torch.Tensor  # (N, K), -inf in unused slots
    targets: torch.Tensor  # (N, K) gold distribution, zeros when unlabeled
    labels: torch.Tensor  # (N,) gold index, -100 when unlabeled
    question_type: torch.Tensor  # (N,)
    num_options: torch.Tensor  # (N,)
    question_ids: list[str]  # "workflow/qid"
    case_index: torch.Tensor | None = None  # (N,) position of each row's example in the input list


def gold_outputs(examples: list[Example], max_options: int = 16) -> Outputs:
    """Targets and metadata for a set of examples, with no model and empty logits."""
    rows = []
    for i, e in enumerate(examples):
        for q in e.questions:
            target = list(e.gold.get(q.qid, ())) + [0.0] * (max_options - len(e.gold.get(q.qid, ())))
            rows.append((target, e.label.get(q.qid, -100), QUESTION_TYPE_CODES[q.type], q.num_options,
                         f"{e.workflow}/{q.qid}".lstrip("/"), i))
    targets, labels, qtype, n_opts, qids, cases = zip(*rows)
    n_opts = torch.tensor(n_opts)
    logits = torch.zeros(len(rows), max_options).masked_fill(torch.arange(max_options)[None, :] >= n_opts[:, None], float("-inf"))
    return Outputs(logits, torch.tensor(targets), torch.tensor(labels), torch.tensor(qtype), n_opts, list(qids),
                   torch.tensor(cases))


@torch.no_grad()
def run_model(model, examples: list[Example], tokenizer, pad_id: int, batch_size: int = 8,
              max_options: int = 16, max_state_tokens: int = 512, device="cpu") -> Outputs:
    model.eval()
    parts = []
    for start in range(0, len(examples), batch_size):
        chunk = examples[start:start + batch_size]
        batch = collate(chunk, [pack_example(e, tokenizer, max_state_tokens) for e in chunk], pad_id, max_options)
        logits = model(to_device(batch, device)).float().cpu()
        batch["case_index"] = batch["question_batch_index"] + start
        parts.append((logits, batch))
    return Outputs(
        logits=torch.cat([p[0] for p in parts]),
        targets=torch.cat([p[1]["targets"] for p in parts]),
        labels=torch.cat([p[1]["labels"] for p in parts]),
        question_type=torch.cat([p[1]["question_type"] for p in parts]),
        num_options=torch.cat([p[1]["num_options"] for p in parts]),
        question_ids=[q for p in parts for q in p[1]["question_ids"]],
        case_index=torch.cat([p[1]["case_index"] for p in parts]),
    )


def jev_confidence(probs: list[float]) -> float:
    """Jev's confidence summary: (K * p_max - 1) / (K - 1). 0 when uniform, 1 when certain.

    It's arithmetic on the distribution, not a separate estimate.
    """
    k = len(probs)
    return (k * max(probs) - 1) / (k - 1)


def to_answers(example: Example, probs: torch.Tensor) -> dict:
    """Format one example's per-question probabilities (rows of probs) as System One answers."""
    answers = {}
    for q, row in zip(example.questions, probs):
        p = [round(x, 6) for x in row[: q.num_options].tolist()]
        if q.type == "noul":
            answers[q.qid] = {"type": "noul", "noul": p[1]}
        elif q.type == "choice":
            best = max(range(q.num_options), key=p.__getitem__)
            answers[q.qid] = {"type": "choice", "choice": q.options[best],
                              "probabilities": dict(zip(q.options, p)), "confidence": round(jev_confidence(p), 6)}
        else:
            answers[q.qid] = {"type": "score", "score": round(sum(i * x for i, x in enumerate(p)), 6),
                              "probabilities": {str(i): x for i, x in enumerate(p)}, "confidence": round(jev_confidence(p), 6)}
    return answers


def predict(model, tokenizer, state, questions: dict, pad_id: int, temperatures=None,
            max_options: int = 16, device="cpu") -> dict:
    """System One in, System One out: {"answers": {qid: answer}}."""
    example = Example(render_state(state), tuple(parse_question(qid, spec) for qid, spec in questions.items()))
    outputs = run_model(model, [example], tokenizer, pad_id, batch_size=1, max_options=max_options, device=device)
    logits = outputs.logits if temperatures is None else temperatures.apply(outputs.logits, outputs.question_type)
    return {"answers": to_answers(example, logits.softmax(dim=1))}
