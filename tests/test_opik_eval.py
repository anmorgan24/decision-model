"""The Opik evaluation pieces, tested without an Opik server: items, metrics, tasks, and the Jev client."""

import json
import threading
import zlib
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
import torch

from decision_model import DecisionModel
from decision_model.opik_eval import (DecisionMetrics, IrrelevantOption, Sensitivity, example_to_item,
                                      experiment_summary, make_task, spec_keys)
from decision_model.predictors import JevPredictor, LocalPredictor

from conftest import PAD_ID, ToyTokenizer, tiny_qwen3


def gold_as_answers(item):
    answers = {}
    for qid, g in item["gold"].items():
        if g["type"] == "noul":
            answers[qid] = {"type": "noul", "noul": g["probabilities"]["true"]}
        else:
            answers[qid] = {"type": g["type"], "probabilities": dict(g["probabilities"])}
    return answers


class ScriptedPredictor:
    """Answers from a rule, to test the property metrics against known behavior."""

    def __init__(self, rule):
        self.rule = rule

    def predict(self, state, questions):
        answers = {}
        for qid, spec in questions.items():
            if spec["type"] == "noul":
                answers[qid] = {"type": "noul", "noul": self.rule("noul", [], len(questions))}
            else:
                listed = list(spec["criteria"]) if spec["type"] == "choice" else [str(i) for i in range(len(spec["criteria"]))]
                probs = self.rule(spec["type"], listed, len(questions))
                answers[qid] = {"type": spec["type"], "probabilities": dict(zip(listed, probs))}
        return {"answers": answers}


def first_listed(kind, listed, n):  # puts most of its weight on whatever is listed first
    return 0.6 if kind == "noul" else [0.7] + [0.3 / (len(listed) - 1)] * (len(listed) - 1)


def by_label(kind, listed, n):  # depends on the labels, not their order
    if kind == "noul":
        return 0.6
    w = [1 + zlib.crc32(label.encode()) % 7 for label in listed]
    return [x / sum(w) for x in w]


def uniform(kind, listed, n):
    return 0.5 if kind == "noul" else [1 / len(listed)] * len(listed)


def leaky(kind, listed, n):  # answers depend on how many questions were asked together
    return (0.9 if n > 1 else 0.6) if kind == "noul" else [1 / len(listed)] * len(listed)


@pytest.fixture
def items(typed_examples):
    return [example_to_item(e) for e in typed_examples]


def test_items_keep_the_request_and_label_keyed_gold(items, typed_examples):
    item = items[0]
    assert item["case_id"] == typed_examples[0].id and "id" not in item
    assert item["state"]["agent"]["autonomy"] == "checkpointed"
    assert item["questions"]["urgency"]["criteria"][0] == "no rush"
    assert item["gold"]["action"]["label"] == "human_review"
    assert item["gold"]["needs_review"] == {"type": "noul", "label": "true", "probabilities": {"false": 0.45, "true": 0.55}}
    assert item["gold"]["risk"]["label"] == "0"
    assert spec_keys(item["questions"]["risk"]) == ["0", "1", "2", "3"]


def test_answering_with_the_gold_scores_zero_kl(items):
    results = {r.name: r.value for r in DecisionMetrics().score(**items[0], answers=gold_as_answers(items[0]))}
    assert results["kl"] < 1e-5 and results["accuracy"] == 1.0


def test_scores_dont_depend_on_option_order(items):
    item = items[1]
    answers = gold_as_answers(item)
    shuffled = json.loads(json.dumps(item))
    shuffled["questions"]["action"]["criteria"] = dict(reversed(list(item["questions"]["action"]["criteria"].items())))
    a = {r.name: r.value for r in DecisionMetrics().score(**item, answers=answers)}
    b = {r.name: r.value for r in DecisionMetrics().score(**shuffled, answers=answers)}
    assert a == b


def run_test(rule, test, item):
    output = make_task(ScriptedPredictor(rule).predict, test)(item)
    metric = IrrelevantOption() if test == "irrelevant" else Sensitivity(test, ("choice",) if test == "order" else ("choice", "score", "noul"))
    return {r.name: r.value for r in metric.score(**item, **output)}


def test_order_test_catches_position_bias_and_passes_label_based_answers(items):
    biased = run_test(first_listed, "order", items[0])
    assert biased["order_flip"] == 1.0 and biased["order_shift"] > 0.3
    assert run_test(by_label, "order", items[0]) == {"order_shift": 0.0, "order_flip": 0.0}


def test_irrelevant_option_measures_mass_and_shift(items):
    result = run_test(uniform, "irrelevant", items[0])
    assert abs(result["irrelevant_mass"] - 0.2) < 1e-9  # 1 / (4 real options + 1)
    assert result["irrelevant_shift"] < 1e-9  # renormalized, the real options are unchanged


def test_isolation_catches_leaks_and_passes_our_model(items, tokenizer):
    leak = run_test(leaky, "isolation", items[0])
    assert leak["isolation_shift"] > 0

    model = DecisionModel(tiny_qwen3(), 8).eval()
    for p in model.head.parameters():
        torch.nn.init.normal_(p, std=0.5)
    ours = LocalPredictor("tiny", model, ToyTokenizer(), PAD_ID, max_options=8)
    output = make_task(ours.predict, "isolation")(items[0])
    result = {r.name: r.value for r in Sensitivity("isolation", ("choice", "score", "noul")).score(**items[0], **output)}
    assert result["isolation_shift"] < 1e-4 and result["isolation_flip"] == 0.0


def test_experiment_summary_reports_ece_and_per_type_results(items):
    from opik.evaluation.test_case import TestCase
    from opik.evaluation.test_result import TestResult

    results = [TestResult(test_case=TestCase(trace_id="t", dataset_item_id=str(i), task_output={"answers": gold_as_answers(item)},
                                             dataset_item_content=item, mapped_scoring_inputs=None, dataset_item=None),
                          score_results=[], trial_id=0, task_execution_time=0.0, scoring_time=0.0)
               for i, item in enumerate(items)]
    names = {r.name: r.value for r in experiment_summary(results)}
    assert {"ece", "choice_accuracy", "choice_kl", "score_kl", "noul_accuracy"} <= set(names)
    assert names["choice_kl"] < 1e-5


def test_jev_client_sends_system_one_requests_and_retries(monkeypatch, items):
    seen, statuses = [], [503, 200]

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((self.path, self.headers["Authorization"], body))
            status = statuses.pop(0)
            self.send_response(status)
            self.end_headers()
            if status == 200:
                self.wfile.write(json.dumps({"answers": gold_as_answers(items[0]), "model": "jev-1.13.0"}).encode())

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr("decision_model.predictors.time.sleep", lambda s: None)
    try:
        jev = JevPredictor(base_url=f"http://127.0.0.1:{server.server_port}", api_key="test-key")
        answers = jev.predict(items[0]["state"], items[0]["questions"])["answers"]
    finally:
        server.shutdown()

    assert len(seen) == 2  # one 503, then a retry
    path, auth, body = seen[-1]
    assert path == "/v1/systemone" and auth == "Bearer test-key"
    assert body["model"] == "jev-latest" and set(body) == {"model", "state", "questions"}
    assert answers["action"]["probabilities"]["human_review"] == pytest.approx(0.433333, abs=1e-5)
    assert jev.reported_models == {"jev-1.13.0"}


def test_experiment_summary_skips_failed_cases(items):
    from opik.evaluation.test_case import TestCase
    from opik.evaluation.test_result import TestResult

    def result(output):
        return TestResult(test_case=TestCase(trace_id="t", dataset_item_id="d", task_output=output, dataset_item_content=items[0],
                                             mapped_scoring_inputs=None, dataset_item=None),
                          score_results=[], trial_id=0, task_execution_time=0.0, scoring_time=0.0)

    names = {r.name for r in experiment_summary([result({"answers": gold_as_answers(items[0])}), result({})])}
    assert "ece" in names


def test_opik_scores_match_the_repo_metrics_on_the_same_model(items, typed_examples, tokenizer):
    """Opik's numbers and evaluate.py's numbers must agree, or the article's two sections disagree."""
    from opik.evaluation.test_case import TestCase
    from opik.evaluation.test_result import TestResult

    from decision_model.inference import run_model
    from decision_model.metrics import compute

    model = DecisionModel(tiny_qwen3(), 8).eval()
    for p in model.head.parameters():
        torch.nn.init.normal_(p, std=0.5)
    ours = LocalPredictor("tiny", model, ToyTokenizer(), PAD_ID, max_options=8)

    task = make_task(ours.predict, "standard")
    results = [TestResult(test_case=TestCase(trace_id="t", dataset_item_id=str(i), task_output=task(item),
                                             dataset_item_content=item, mapped_scoring_inputs=None, dataset_item=None),
                          score_results=[], trial_id=0, task_execution_time=0.0, scoring_time=0.0)
               for i, item in enumerate(items)]
    opik_ece = {r.name: r.value for r in experiment_summary(results)}["ece"]
    per_case = [{r.name: r.value for r in DecisionMetrics().score(**item, answers=res.test_case.task_output["answers"])}
                for item, res in zip(items, results)]

    out = run_model(model, typed_examples, tokenizer, PAD_ID, max_options=8)
    direct = compute(out.logits, out.targets, out.labels)
    assert abs(opik_ece - direct["ece"]) < 1e-3
    for metric in ("accuracy", "kl", "brier"):
        # Every case has five questions, so the mean of per-case means is the overall mean.
        assert abs(sum(c[metric] for c in per_case) / len(per_case) - direct[metric]) < 1e-3
