"""Reference baselines that read nothing, and a check of our scorer against the benchmark.

Prior:   for every case, predict each question's label frequencies from train
Uniform: the same probability on every option

The typed-decisions dataset card reports both on `test`. Reproducing those
numbers checks that our metrics are computed the same way before we compare
any trained model against published results.

    python -m decision_model.references
"""

from __future__ import annotations

from collections import defaultdict

import torch

from .inference import Outputs, gold_outputs
from .schema import Example

# From the LocalLLaMA/typed-decisions dataset card, test split.
CARD = {
    "prior": {"accuracy": 0.470, "kl": 0.347, "brier": 0.189, "ece": 0.088},
    "uniform": {"accuracy": 0.308, "kl": 0.444, "brier": 0.238, "ece": 0.169},
}


def prior_outputs(train: list[Example], evaluate: list[Example], max_options: int = 16,
                  from_labels: bool = True, eps: float = 1e-6) -> Outputs:
    """from_labels=True counts gold labels, as the card does; False averages the soft gold distributions."""
    sums, counts = defaultdict(lambda: torch.zeros(max_options)), defaultdict(int)
    for e in train:
        for q in e.questions:
            if q.qid in e.gold:
                key = f"{e.workflow}/{q.qid}".lstrip("/")
                if from_labels:
                    sums[key][e.label[q.qid]] += 1
                else:
                    sums[key][: q.num_options] += torch.tensor(e.gold[q.qid])
                counts[key] += 1

    outputs = gold_outputs(evaluate, max_options)
    for i, (key, n) in enumerate(zip(outputs.question_ids, outputs.num_options.tolist())):
        prior = sums[key][:n] / max(counts[key], 1) if counts[key] else torch.full((n,), 1.0 / n)
        outputs.logits[i, :n] = prior.clamp_min(eps).log()
    return outputs


def uniform_outputs(evaluate: list[Example], max_options: int = 16) -> Outputs:
    return gold_outputs(evaluate, max_options)  # zero logits on every real option


def main() -> None:
    from .data import load_typed_decisions
    from .metrics import compute

    train, test = load_typed_decisions("train"), load_typed_decisions("test")
    rows = {"prior": prior_outputs(train, test), "uniform": uniform_outputs(test)}
    print(f"{'':10}{'metric':12}{'ours':>9}{'card':>9}")
    for name, outputs in rows.items():
        ours = compute(outputs.logits, outputs.targets, outputs.labels)
        for metric, card_value in CARD[name].items():
            print(f"{name:10}{metric:12}{ours[metric]:9.3f}{card_value:9.3f}")
    print("\nECE is defined differently on the card, so it isn't expected to match.")


if __name__ == "__main__":
    main()
