import math

import torch
import torch.nn.functional as F

from decision_model import DecisionHead, TemperatureScaler, decision_loss
from decision_model.metrics import accuracy, accuracy_ties, brier, brier_label, compute, ece, ece_label, fit_temperature, kl, log_loss
from decision_model.packing import QUESTION_TYPE_CODES

CHOICE, SCORE, NOUL = (QUESTION_TYPE_CODES[t] for t in ("choice", "score", "noul"))
INF = float("-inf")


def test_head_starts_uniform_and_masks_unused_slots():
    head = DecisionHead(hidden_size=8, max_options=5)
    logits = head(torch.randn(3, 8), torch.tensor([CHOICE, SCORE, NOUL]), torch.tensor([3, 4, 2]))
    probs = logits.softmax(dim=1)
    assert torch.allclose(probs[0, :3], torch.full((3,), 1 / 3))
    assert torch.allclose(probs[2, :2], torch.full((2,), 0.5))
    assert torch.isinf(logits[0, 3:]).all() and torch.isinf(logits[1, 4:]).all()


def test_noul_is_a_sigmoid():
    torch.manual_seed(0)
    head = DecisionHead(hidden_size=8, max_options=5)
    torch.nn.init.normal_(head.noul.weight)
    hidden = torch.randn(1, 8)
    p_yes = head(hidden, torch.tensor([NOUL]), torch.tensor([2]))[0].softmax(dim=0)[1]
    assert torch.isclose(p_yes, torch.sigmoid(head.noul(hidden)).squeeze())


def test_soft_loss_matches_cross_entropy_on_one_hot_targets():
    logits = torch.tensor([[2.0, 0.5, INF], [0.1, 1.0, 0.3]])
    targets = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    expected = F.cross_entropy(logits, torch.tensor([0, 2]), reduction="sum")
    assert torch.isclose(decision_loss(logits, targets, batch_size=1), expected)


def test_soft_loss_is_lowest_at_the_gold_distribution():
    gold = torch.tensor([[0.6, 0.3, 0.1]])
    at_gold = decision_loss(gold.log(), gold, batch_size=1)
    for other in ([0.8, 0.1, 0.1], [0.34, 0.33, 0.33], [0.5, 0.4, 0.1]):
        assert at_gold < decision_loss(torch.tensor([other]).log(), gold, batch_size=1)


def test_unlabeled_rows_contribute_nothing():
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    labeled = decision_loss(logits[:1], torch.tensor([[1.0, 0.0]]), batch_size=1)
    mixed = decision_loss(logits, torch.tensor([[1.0, 0.0], [0.0, 0.0]]), batch_size=1)
    assert torch.isclose(labeled, mixed)


def test_kl_is_zero_at_the_gold_distribution_and_ties_count_as_correct():
    targets = torch.tensor([[0.5, 0.5, 0.0], [0.2, 0.8, 0.0]])
    labels = torch.tensor([0, 1])
    logits = torch.tensor([[0.5, 0.5, 1e-9], [0.2, 0.8, 1e-9]]).log().masked_fill(torch.tensor([[0, 0, 1], [0, 0, 1]]).bool(), INF)
    assert kl(logits, targets, labels) < 1e-5
    tied_other = torch.tensor([[0.0, 1.0, INF], [0.0, 1.0, INF]])  # picks option 1 on the tied first row
    assert accuracy(tied_other, targets, labels) == 0.5
    assert accuracy_ties(tied_other, targets, labels) == 1.0


def test_base_rate_predictor_is_calibrated_but_only_as_accurate_as_the_base_rate():
    # The article's example: 60% of tickets are urgent, and the model says 0.6 on every one.
    labels = torch.tensor([1] * 60 + [0] * 40)
    targets = F.one_hot(labels, 2).float()
    logits = torch.log(torch.tensor([[0.4, 0.6]])).repeat(100, 1)
    assert ece(logits, targets, labels) < 1e-6
    assert math.isclose(accuracy(logits, targets, labels), 0.6, abs_tol=1e-6)


def test_temperature_scaling_fixes_overconfidence_without_changing_accuracy():
    torch.manual_seed(0)
    true_logits = torch.randn(2000, 4) * 1.5
    labels = torch.distributions.Categorical(logits=true_logits).sample()
    targets = F.one_hot(labels, 4).float()
    overconfident = 3.0 * true_logits
    temperature = fit_temperature(overconfident, targets)
    scaled = overconfident / temperature
    assert 2.5 < temperature < 3.5
    assert ece(scaled, targets, labels) < ece(overconfident, targets, labels)
    assert log_loss(scaled, targets, labels) < log_loss(overconfident, targets, labels)
    assert accuracy(scaled, targets, labels) == accuracy(overconfident, targets, labels)


def test_temperature_scaler_fits_one_temperature_per_type():
    from decision_model.inference import Outputs

    torch.manual_seed(0)
    n = 1500
    true_logits = torch.randn(n, 3)
    labels = torch.distributions.Categorical(logits=true_logits).sample()
    qtype = torch.tensor([CHOICE] * (n // 2) + [SCORE] * (n - n // 2))
    scale = torch.where(qtype == CHOICE, 4.0, 1.0)[:, None]  # choice is 4x overconfident, score is fine
    outputs = Outputs(true_logits * scale, F.one_hot(labels, 3).float(), labels, qtype, torch.full((n,), 3), ["q"] * n)
    scaler = TemperatureScaler.fit(outputs)
    assert 3.0 < scaler.temperatures["choice"] < 5.0
    assert 0.7 < scaler.temperatures["score"] < 1.4


def test_brier_variants():
    logits = torch.tensor([[50.0, 0.0, INF]])
    labels = torch.tensor([0])
    assert brier_label(logits, torch.tensor([[1.0, 0.0, 0.0]]), labels) < 1e-6
    assert math.isclose(brier(logits, torch.tensor([[0.5, 0.5, 0.0]]), labels), 0.5, abs_tol=1e-6)
    assert set(compute(logits, torch.tensor([[1.0, 0.0, 0.0]]), labels)) >= {"accuracy", "kl", "brier", "ece", "log_loss"}


def test_temperature_fit_handles_masked_option_slots():
    torch.manual_seed(0)
    logits = torch.randn(200, 4) * 3
    logits[:, 3] = INF  # a three-option question in a four-slot layout
    labels = logits[:, :3].softmax(dim=1).multinomial(1).squeeze(1)
    t = fit_temperature(logits, F.one_hot(labels, 4).float())
    assert math.isfinite(t) and t > 0


def test_tied_predictions_count_as_a_random_pick():
    uniform = torch.zeros(3, 4).masked_fill(torch.tensor([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1]]).bool(), INF)
    labels = torch.tensor([0, 2, 1])
    gold = F.one_hot(labels, 4).float()
    expected = (1 / 4 + 1 / 3 + 1 / 2) / 3
    assert math.isclose(accuracy(uniform, gold, labels), expected, abs_tol=1e-6)
    confident = torch.tensor([[3.0, 0.0], [0.0, 3.0]])
    assert accuracy(confident, torch.eye(2), torch.tensor([0, 0])) == 0.5  # no ties: plain accuracy


def test_matching_a_soft_gold_distribution_is_calibrated_under_ece_but_not_ece_label():
    # Gold is 0.7 / 0.3 on every case, and the label is always the 0.7 answer.
    targets = torch.tensor([[0.7, 0.3]]).repeat(200, 1)
    labels = torch.zeros(200, dtype=torch.long)
    matching = targets.log()
    assert ece(matching, targets, labels) < 1e-6  # a labeler's sample agrees 70% of the time, as predicted
    assert abs(ece_label(matching, targets, labels) - 0.3) < 1e-6  # looks underconfident against the 0.7 label


def test_ece_and_ece_label_agree_on_hard_labels():
    torch.manual_seed(0)
    logits = torch.randn(300, 4)
    labels = torch.randint(0, 4, (300,))
    targets = F.one_hot(labels, 4).float()
    assert abs(ece(logits, targets, labels) - ece_label(logits, targets, labels)) < 1e-6
