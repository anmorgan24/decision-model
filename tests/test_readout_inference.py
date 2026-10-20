import dataclasses

import torch

from decision_model import LogitReadoutModel, collate, pack_example, predict
from decision_model.inference import gold_outputs, jev_confidence, run_model
from decision_model.readout import answer_token_ids
from decision_model.references import prior_outputs, uniform_outputs
from decision_model.metrics import compute

from conftest import PAD_ID, ToyTokenizer, tiny_qwen3_lm


def readout(loss_space="restricted"):
    return LogitReadoutModel(tiny_qwen3_lm(), answer_token_ids(ToyTokenizer(), 8), loss_space).eval()


def test_answer_tokens_are_single_tokens(tokenizer):
    table = answer_token_ids(tokenizer, 4)
    assert table.shape == (3, 4)
    assert table[2, 0] == tokenizer.encode(" No")[0] and table[2, 1] == tokenizer.encode(" Yes")[0]


def test_readout_keeps_questions_isolated(examples, tokenizer):
    model = readout()
    original = examples[0]
    q1, q2, q3 = original.questions
    edited = dataclasses.replace(original, questions=(q1, dataclasses.replace(q2, instructions="Ignore the ticket."), q3))
    with torch.no_grad():
        before = model(collate([original], [pack_example(original, tokenizer)], PAD_ID, 8))
        after = model(collate([edited], [pack_example(edited, tokenizer)], PAD_ID, 8))
    assert torch.allclose(before[0], after[0], atol=1e-5) and torch.allclose(before[2], after[2], atol=1e-5)


def test_full_vocab_and_restricted_report_the_same_distribution(examples, tokenizer):
    batch = collate(examples, [pack_example(e, tokenizer) for e in examples], PAD_ID, 8)
    with torch.no_grad():
        restricted = readout("restricted")(batch).softmax(dim=1)
        full = readout("full_vocab")(batch).softmax(dim=1)
    assert torch.allclose(restricted, full, atol=1e-5)


def test_predict_returns_system_one_answers(examples, tokenizer):
    model = readout()
    spec = {
        "route": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "refunds", "tech": "bugs"}},
        "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "medium", "high"]},
        "churn": {"type": "noul", "instructions": "The customer threatens to leave."},
    }
    answers = predict(model, tokenizer, {"body": "Refund me or I cancel."}, spec, PAD_ID, max_options=8)["answers"]
    assert answers["route"]["choice"] in ("billing", "tech")
    assert abs(sum(answers["route"]["probabilities"].values()) - 1) < 1e-4
    assert 0 <= answers["urgency"]["score"] <= 2 and set(answers["urgency"]["probabilities"]) == {"0", "1", "2"}
    assert 0 <= answers["churn"]["noul"] <= 1 and set(answers["churn"]) == {"type", "noul"}


def test_jev_confidence():
    assert jev_confidence([1 / 3, 1 / 3, 1 / 3]) == 0
    assert jev_confidence([1.0, 0.0, 0.0]) == 1
    assert abs(jev_confidence([0.6, 0.2, 0.2]) - 0.4) < 1e-9


def test_run_model_matches_gold_metadata(typed_examples, tokenizer):
    out = run_model(readout(), typed_examples, tokenizer, PAD_ID, batch_size=1, max_options=8)
    gold = gold_outputs(typed_examples, max_options=8)
    assert out.question_ids == gold.question_ids
    assert torch.equal(out.labels, gold.labels) and torch.allclose(out.targets, gold.targets)


def test_prior_is_the_average_training_distribution(typed_examples):
    prior = prior_outputs(typed_examples, typed_examples[:1], max_options=8, from_labels=False)
    action = prior.logits[0, :4].softmax(dim=0)
    expected = (torch.tensor(typed_examples[0].gold["action"]) + torch.tensor(typed_examples[1].gold["action"])) / 2
    assert torch.allclose(action, expected, atol=1e-4)
    uniform = uniform_outputs(typed_examples, max_options=8)
    assert compute(uniform.logits, uniform.targets, uniform.labels)["n"] == 10


def test_latency_benchmark_counts_the_state_once_when_packed(typed_examples, tokenizer):
    from decision_model import DecisionModel
    from decision_model.bench_latency import compare
    from conftest import tiny_qwen3

    result = compare(DecisionModel(tiny_qwen3(), 8).eval(), typed_examples, tokenizer, PAD_ID, 8, torch.device("cpu"))
    assert result["questions_per_example"] == 5
    assert result["unpacked_tokens"] > result["packed_tokens"]  # unpacked repeats the state for every question
    assert result["packed_ms"] > 0 and result["unpacked_ms"] > 0


def test_label_prior_counts_gold_labels(typed_examples):
    prior = prior_outputs(typed_examples, typed_examples[:1], max_options=8)
    outcome = prior.question_ids.index("agent_trace_observability/outcome")
    probs = prior.logits[outcome, :4].softmax(dim=0)
    # The two training cases are labeled "partial" (index 2) and "harmful" (index 1).
    assert torch.allclose(probs, torch.tensor([0.0, 0.5, 0.5, 0.0]), atol=1e-4)
