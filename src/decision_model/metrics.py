"""Metrics for decision models, and temperature scaling.

Accuracy asks how often the top answer is right. The rest ask whether the
probabilities can be trusted:

  log_loss:  cross-entropy against the gold distribution (a proper scoring rule)
  kl:        KL divergence from the gold distribution to the prediction
  brier:     squared error against the gold distribution, summed over options
             (matches the typed-decisions card's per-decision Brier)
  brier_label: squared error against the one-hot gold label, summed over options
  ece:       expected calibration error: group predictions by top-answer
             confidence (10 equal-width bins) and compare each group's average
             confidence with its accuracy. Definitions vary between benchmarks,
             so compare ECE only between models scored by this code.

ECE alone can't tell a useful model from one that ignores its input: a model
that always predicts the label frequencies is well calibrated and useless. Read
it next to accuracy, KL, or Brier.

All functions take logits (N, K) with -inf in unused option slots, targets
(N, K) holding the gold distribution, and labels (N,) holding the gold index.
"""

from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn.functional as F

from .packing import QUESTION_TYPE_CODES

TYPE_NAMES = {code: name for name, code in QUESTION_TYPE_CODES.items()}
KL_EPS = 1e-6  # predicted probabilities are clamped here before taking logs


def accuracy(logits, targets, labels) -> float:
    """How often the top answer matches the gold label.

    When several options tie for the top probability, the prediction counts as
    a random pick among them, so a uniform prediction scores 1/K rather than
    whatever the first option happens to be. Without ties this is plain accuracy.
    """
    probs = logits.softmax(dim=1)
    top = probs >= probs.max(dim=1, keepdim=True).values - 1e-9
    hit = top.gather(1, labels.clamp_min(0)[:, None]).squeeze(1).float()
    return (hit / top.sum(dim=1)).mean().item()


def accuracy_ties(logits, targets, labels) -> float:
    """Counts a prediction as correct if it lands on any option tied for the highest gold probability."""
    top_gold = targets >= targets.max(dim=1, keepdim=True).values - 1e-6
    return top_gold.gather(1, logits.argmax(dim=1, keepdim=True)).float().mean().item()


def log_loss(logits, targets, labels) -> float:
    log_probs = logits.log_softmax(dim=1).masked_fill(targets == 0, 0.0)
    return -(targets * log_probs).sum(dim=1).mean().item()


def kl(logits, targets, labels) -> float:
    log_p = logits.softmax(dim=1).clamp_min(KL_EPS).log()
    log_q = targets.clamp_min(1e-12).log()
    return (targets * (log_q - log_p)).masked_fill(targets == 0, 0.0).sum(dim=1).mean().item()


def brier(logits, targets, labels) -> float:
    return ((logits.softmax(dim=1) - targets) ** 2).sum(dim=1).mean().item()


def brier_label(logits, targets, labels) -> float:
    onehot = F.one_hot(labels, num_classes=logits.shape[1]).float()
    return ((logits.softmax(dim=1) - onehot) ** 2).sum(dim=1).mean().item()


def reliability_bins(logits, labels, n_bins: int = 10) -> list[dict]:
    """Per confidence bin: how many predictions, their average confidence, and their accuracy."""
    confidence, predicted = logits.softmax(dim=1).max(dim=1)
    correct = (predicted == labels).float()
    edges = torch.linspace(0, 1, n_bins + 1)
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > lo) & (confidence <= hi)
        count = int(in_bin.sum())
        bins.append({
            "lower": round(lo.item(), 2),
            "upper": round(hi.item(), 2),
            "count": count,
            "confidence": confidence[in_bin].mean().item() if count else 0.0,
            "accuracy": correct[in_bin].mean().item() if count else 0.0,
        })
    return bins


def ece(logits, targets, labels, n_bins: int = 10) -> float:
    total = len(labels)
    return sum(b["count"] / total * abs(b["confidence"] - b["accuracy"]) for b in reliability_bins(logits, labels, n_bins))


def score_mae(logits, targets, labels) -> float:
    """Mean absolute error between predicted and gold expected level (score questions)."""
    levels = torch.arange(logits.shape[1], dtype=torch.float32)
    predicted = (logits.softmax(dim=1) * levels).sum(dim=1)
    gold = (targets * levels).sum(dim=1)
    return (predicted - gold).abs().mean().item()


METRICS = {
    "accuracy": accuracy,
    "accuracy_ties": accuracy_ties,
    "log_loss": log_loss,
    "kl": kl,
    "brier": brier,
    "brier_label": brier_label,
    "ece": ece,
}


def compute(logits, targets, labels) -> dict[str, float]:
    return {name: fn(logits, targets, labels) for name, fn in METRICS.items()} | {"n": len(labels)}


def summarize(outputs, temperatures: "TemperatureScaler | None" = None) -> dict:
    """Metrics overall, per question type, and per question. outputs: an inference.Outputs."""
    logits = outputs.logits if temperatures is None else temperatures.apply(outputs.logits, outputs.question_type)
    labeled = outputs.labels != -100
    logits, targets, labels = logits[labeled], outputs.targets[labeled], outputs.labels[labeled]
    qtype = outputs.question_type[labeled]
    qids = [q for q, keep in zip(outputs.question_ids, labeled.tolist()) if keep]

    summary = {"overall": compute(logits, targets, labels), "by_type": {}, "by_question": {}}
    for code, name in TYPE_NAMES.items():
        rows = qtype == code
        if rows.any():
            summary["by_type"][name] = compute(logits[rows], targets[rows], labels[rows])
    if (qtype == QUESTION_TYPE_CODES["score"]).any():
        rows = qtype == QUESTION_TYPE_CODES["score"]
        summary["by_type"]["score"]["score_mae"] = score_mae(logits[rows], targets[rows], labels[rows])

    groups = defaultdict(list)
    for i, qid in enumerate(qids):
        groups[qid].append(i)
    for qid, idx in sorted(groups.items()):
        idx = torch.tensor(idx)
        summary["by_question"][qid] = compute(logits[idx], targets[idx], labels[idx])
    return summary


def fit_temperature(logits: torch.Tensor, targets: torch.Tensor, steps: int = 100) -> float:
    """The single temperature T that minimizes log loss of logits / T on held-out data.

    Dividing by T rescales confidence without changing which answer ranks first,
    so accuracy is unchanged and only calibration moves.
    """
    # Unused option slots hold -inf, and d(-inf / T)/dT is NaN even where the
    # probability is zero, so give them a large finite value for the fit.
    logits = logits.masked_fill(~torch.isfinite(logits), -1e9)
    log_t = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=steps)

    def closure():
        optimizer.zero_grad()
        log_probs = (logits / log_t.exp()).log_softmax(dim=1).masked_fill(targets == 0, 0.0)
        loss = -(targets * log_probs).sum(dim=1).mean()
        loss.backward()
        return loss

    optimizer.step(closure)
    return log_t.exp().item()


class TemperatureScaler:
    """One temperature per question type, fitted on validation data."""

    def __init__(self, temperatures: dict[str, float] | None = None):
        self.temperatures = temperatures or {}

    @classmethod
    def fit(cls, outputs) -> "TemperatureScaler":
        temperatures = {}
        for code, name in TYPE_NAMES.items():
            rows = (outputs.question_type == code) & (outputs.labels != -100)
            if rows.any():
                temperatures[name] = fit_temperature(outputs.logits[rows], outputs.targets[rows])
        return cls(temperatures)

    def apply(self, logits: torch.Tensor, question_type: torch.Tensor) -> torch.Tensor:
        t = torch.ones(len(logits))
        for name, value in self.temperatures.items():
            t[question_type == QUESTION_TYPE_CODES[name]] = value
        return logits / t[:, None].to(logits.device)
