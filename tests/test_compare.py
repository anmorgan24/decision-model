import torch
import torch.nn.functional as F

from decision_model.compare import bootstrap, check_aligned
from decision_model.inference import run_model
from decision_model import DecisionModel

from conftest import PAD_ID, tiny_qwen3


def fake_predictions(logits, labels, cases):
    return {"logits": logits, "targets": F.one_hot(labels, logits.shape[1]).float(), "labels": labels,
            "case_index": cases, "question_type": torch.zeros_like(labels), "question_ids": ["q"] * len(labels),
            "example_ids": [str(c) for c in sorted(set(cases.tolist()))], "calibrated_logits": None}


def test_identical_runs_show_no_difference():
    torch.manual_seed(0)
    labels = torch.randint(0, 3, (300,))
    preds = fake_predictions(torch.randn(300, 3), labels, torch.arange(300) // 5)
    result = bootstrap(preds, preds, ["accuracy", "kl"], samples=200)
    assert all(r["diff"] == 0 and r["better"] == "neither" for r in result.values())


def test_a_clearly_better_run_is_detected():
    torch.manual_seed(0)
    labels = torch.randint(0, 3, (500,))
    cases = torch.arange(500) // 5
    weak = fake_predictions(torch.randn(500, 3), labels, cases)
    strong = fake_predictions(torch.randn(500, 3) + 4 * F.one_hot(labels, 3), labels, cases)
    result = bootstrap(weak, strong, ["accuracy", "kl", "brier"], samples=200)
    assert result["accuracy"]["better"] == "B" and result["accuracy"]["low"] > 0
    assert result["kl"]["better"] == "B" and result["kl"]["high"] < 0


def test_misaligned_runs_are_rejected():
    labels = torch.tensor([0, 1])
    a = fake_predictions(torch.zeros(2, 2), labels, torch.tensor([0, 1]))
    b = dict(a, example_ids=["x", "y"])
    try:
        check_aligned(a, b)
    except ValueError:
        return
    raise AssertionError("expected a ValueError")


def test_run_model_records_which_case_each_row_came_from(typed_examples, examples, tokenizer):
    out = run_model(DecisionModel(tiny_qwen3(), 8), typed_examples + examples, tokenizer, PAD_ID, batch_size=2, max_options=8)
    assert out.case_index.tolist() == [0] * 5 + [1] * 5 + [2] * 3 + [3] * 3 + [4] * 3


def test_saved_predictions_load_for_comparison(tmp_path, typed_examples, tokenizer):
    from decision_model.compare import load_predictions
    from decision_model.evaluate import save_predictions
    from decision_model.metrics import TemperatureScaler

    out = run_model(DecisionModel(tiny_qwen3(), 8), typed_examples, tokenizer, PAD_ID, max_options=8)
    save_predictions(tmp_path / "predictions_val.pt", out, typed_examples, TemperatureScaler({"choice": 2.0}))
    raw = load_predictions(tmp_path, "val", calibrated=False)
    calibrated = load_predictions(tmp_path, "val", calibrated=True)
    assert raw["example_ids"] == [e.id for e in typed_examples]
    check_aligned(raw, calibrated)
    result = bootstrap(raw, calibrated, ["accuracy"], samples=50)
    assert result["accuracy"]["diff"] == 0  # temperature scaling never changes the top answer
