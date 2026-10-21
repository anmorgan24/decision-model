"""Evaluate decision models in Opik: standard metrics, plus three property tests.

    python -m decision_model.opik_eval --runs outputs/decision_head outputs/logit_readout --split val
    python -m decision_model.opik_eval --runs outputs/decision_head --calibrated --split val
    python -m decision_model.opik_eval --runs outputs/decision_head --jev --split test
    python -m decision_model.opik_eval --zero-shot configs/logit_readout.yaml --tests standard --split val

Trained models are scored on their raw probabilities by default, matching the
`raw` results from decision_model.evaluate; --calibrated applies each run's
fitted temperatures instead. Calibrated experiments get "-calibrated" in
their names.

Each (model, test) pair becomes one Opik experiment on the same dataset:

  standard:   accuracy, log loss, KL, and Brier per case; ECE (measured
              against the chance a teacher sample agrees with the top answer)
              and per-type results for the whole experiment; latency per case
  order:      reverse the order of every choice question's options. How much
              do the answers move?
  irrelevant: add an option unrelated to the decision. How much probability
              does it take, and how much do the other options shift?
  isolation:  ask each question on its own and compare with asking all five
              together. Do the answers depend on what else was asked?

Shifts are total variation distance: half the summed absolute difference
between two distributions, from 0 (identical) to 1 (no overlap).

Opik reads OPIK_API_KEY and OPIK_WORKSPACE from the environment (or run
`opik configure`). Jev needs TYPESAFE_API_KEY.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch
from opik.evaluation.metrics import base_metric, score_result

from .metrics import TYPE_NAMES, compute
from .packing import QUESTION_TYPE_CODES
from .predictors import JevPredictor, LocalPredictor, answer_distribution
from .schema import Example

PROJECT = "decision-model"
IRRELEVANT_OPTION = ("banana", "A yellow fruit. Unrelated to this decision.")


# ---- Dataset items -------------------------------------------------------------------------------

def question_spec(question) -> dict:
    """A parsed Question back in System One request form."""
    spec = {"type": question.type, "instructions": question.instructions}
    if question.type == "choice":
        spec["criteria"] = dict(zip(question.options, question.descriptions))
    elif question.type == "score":
        spec["criteria"] = list(question.options)
    elif any(question.descriptions):
        spec["criteria"] = {"false": question.descriptions[0], "true": question.descriptions[1]}
    return spec


def spec_keys(spec: dict) -> list[str]:
    """How answers to a question are keyed. Choice labels are sorted, because stored JSON may not keep key order."""
    if spec["type"] == "choice":
        return sorted(spec["criteria"])
    if spec["type"] == "score":
        return [str(i) for i in range(len(spec["criteria"]))]
    return ["false", "true"]


def example_to_item(example: Example) -> dict:
    """One Opik dataset item per case: the System One request plus gold answers keyed by label."""
    questions = {q.qid: question_spec(q) for q in example.questions}
    gold = {}
    for q in example.questions:
        if q.qid not in example.gold:
            continue
        native = (list(q.options) if q.type == "choice" else
                  [str(i) for i in range(q.num_options)] if q.type == "score" else ["false", "true"])
        gold[q.qid] = {"type": q.type, "label": native[example.label[q.qid]],
                       "probabilities": dict(zip(native, example.gold[q.qid]))}
    try:
        state = json.loads(example.state)
    except json.JSONDecodeError:
        state = example.state
    # "case_id", not "id": Opik reserves "id" for its own item ids.
    return {"case_id": example.id, "workflow": example.workflow, "state": state, "questions": questions, "gold": gold}


# ---- Scoring -------------------------------------------------------------------------------------

def decision_rows(questions: dict, gold: dict, answers: dict, max_options: int = 16) -> dict[str, torch.Tensor]:
    """Turn one or more cases' answers and gold into the tensors metrics.compute expects."""
    logits, targets, labels, qtypes = [], [], [], []
    for qid, entry in gold.items():
        keys = spec_keys(questions[qid])
        pred = torch.tensor(answer_distribution(questions[qid], answers[qid], keys))
        row = torch.full((max_options,), float("-inf"))
        row[: len(keys)] = pred.clamp_min(1e-12).log()
        target = torch.zeros(max_options)
        target[: len(keys)] = torch.tensor([float(entry["probabilities"].get(k, 0.0)) for k in keys])
        logits.append(row)
        targets.append(target / target.sum())
        labels.append(keys.index(str(entry["label"]).lower() if entry["type"] == "noul" else str(entry["label"])))
        qtypes.append(QUESTION_TYPE_CODES[entry["type"]])
    return {"logits": torch.stack(logits), "targets": torch.stack(targets),
            "labels": torch.tensor(labels), "question_type": torch.tensor(qtypes)}


class DecisionMetrics(base_metric.BaseMetric):
    """Accuracy, log loss, KL, and Brier for one case, averaged over its questions."""

    def __init__(self):
        super().__init__(name="decision_metrics", track=False)

    def score(self, questions: dict, gold: dict, answers: dict, **ignored) -> list[score_result.ScoreResult]:
        rows = decision_rows(questions, gold, answers)
        values = compute(rows["logits"], rows["targets"], rows["labels"])
        return [score_result.ScoreResult(name=m, value=values[m]) for m in ("accuracy", "log_loss", "kl", "brier")]


class Latency(base_metric.BaseMetric):
    def __init__(self):
        super().__init__(name="latency_ms", track=False)

    def score(self, latency_ms: float, **ignored) -> score_result.ScoreResult:
        return score_result.ScoreResult(name="latency_ms", value=latency_ms)


def experiment_summary(test_results) -> list[score_result.ScoreResult]:
    """Set-level results: ECE needs every prediction at once, so it can't be averaged from per-case scores."""
    finished = [r for r in test_results if r.test_case.task_output and "answers" in r.test_case.task_output]
    if not finished:
        return [score_result.ScoreResult(name="ece", value=0.0, scoring_failed=True)]
    parts = [decision_rows(r.test_case.dataset_item_content["questions"], r.test_case.dataset_item_content["gold"],
                           r.test_case.task_output["answers"]) for r in finished]  # cases whose model call failed are skipped
    rows = {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
    results = [score_result.ScoreResult(name="ece", value=compute(rows["logits"], rows["targets"], rows["labels"])["ece"])]
    for code, name in TYPE_NAMES.items():
        keep = rows["question_type"] == code
        if keep.any():
            values = compute(rows["logits"][keep], rows["targets"][keep], rows["labels"][keep])
            results += [score_result.ScoreResult(name=f"{name}_{m}", value=values[m]) for m in ("accuracy", "kl")]
    return results


def total_variation(p: list[float], q: list[float]) -> float:
    return 0.5 * sum(abs(a - b) for a, b in zip(p, q))


def top(keys: list[str], probs: list[float]) -> str:
    return keys[max(range(len(probs)), key=probs.__getitem__)]


class Sensitivity(base_metric.BaseMetric):
    """Compares the original answers with a variant's answers, question by question."""

    def __init__(self, prefix: str, question_types: tuple[str, ...]):
        super().__init__(name=f"{prefix}_sensitivity", track=False)
        self.prefix, self.question_types = prefix, question_types

    def score(self, questions: dict, answers: dict, variant_answers: dict, **ignored) -> list[score_result.ScoreResult]:
        shifts, flips = [], []
        for qid, spec in questions.items():
            if spec["type"] not in self.question_types:
                continue
            keys = spec_keys(spec)
            p = answer_distribution(spec, answers[qid], keys)
            q = answer_distribution(spec, variant_answers[qid], keys)
            shifts.append(total_variation(p, q))
            flips.append(float(top(keys, p) != top(keys, q)))
        if not shifts:
            return [score_result.ScoreResult(name=f"{self.prefix}_shift", value=0.0, scoring_failed=True)]
        return [score_result.ScoreResult(name=f"{self.prefix}_shift", value=sum(shifts) / len(shifts)),
                score_result.ScoreResult(name=f"{self.prefix}_flip", value=sum(flips) / len(flips))]


class IrrelevantOption(base_metric.BaseMetric):
    """How much probability the unrelated option takes, and how much the real options shift without it."""

    def __init__(self):
        super().__init__(name="irrelevant_option", track=False)

    def score(self, questions: dict, answers: dict, variant_answers: dict, **ignored) -> list[score_result.ScoreResult]:
        masses, shifts = [], []
        for qid, spec in questions.items():
            if spec["type"] != "choice":
                continue
            keys = spec_keys(spec)
            p = answer_distribution(spec, answers[qid], keys)
            with_extra = answer_distribution(spec, variant_answers[qid], keys + [IRRELEVANT_OPTION[0]])
            masses.append(with_extra[-1])
            kept = sum(with_extra[:-1]) or 1.0
            shifts.append(total_variation(p, [x / kept for x in with_extra[:-1]]))
        return [score_result.ScoreResult(name="irrelevant_mass", value=sum(masses) / len(masses)),
                score_result.ScoreResult(name="irrelevant_shift", value=sum(shifts) / len(shifts))]


# ---- Tasks ---------------------------------------------------------------------------------------

def reversed_choices(questions: dict) -> dict:
    variant = copy.deepcopy(questions)
    for spec in variant.values():
        if spec["type"] == "choice":
            spec["criteria"] = dict(reversed(list(spec["criteria"].items())))
    return variant


def with_irrelevant_option(questions: dict) -> dict:
    variant = copy.deepcopy(questions)
    for spec in variant.values():
        if spec["type"] == "choice" and IRRELEVANT_OPTION[0] not in spec["criteria"]:
            spec["criteria"][IRRELEVANT_OPTION[0]] = IRRELEVANT_OPTION[1]
    return variant


def make_task(predict, test: str):
    """A function from an Opik dataset item to the outputs that test's metrics read."""

    def task(item: dict) -> dict:
        state, questions = item["state"], item["questions"]
        start = time.perf_counter()
        answers = predict(state, questions)["answers"]
        output = {"answers": answers, "latency_ms": (time.perf_counter() - start) * 1000}
        if test == "order":
            output["variant_answers"] = predict(state, reversed_choices(questions))["answers"]
        elif test == "irrelevant":
            output["variant_answers"] = predict(state, with_irrelevant_option(questions))["answers"]
        elif test == "isolation":
            output["variant_answers"] = {qid: predict(state, {qid: spec})["answers"][qid] for qid, spec in questions.items()}
        return output

    return task


TESTS = {
    "standard": (lambda: [DecisionMetrics(), Latency()], [experiment_summary]),
    "order": (lambda: [Sensitivity("order", ("choice",))], None),
    "irrelevant": (lambda: [IrrelevantOption()], None),
    "isolation": (lambda: [Sensitivity("isolation", ("choice", "score", "noul"))], None),
}


# ---- Running -------------------------------------------------------------------------------------

def load_local(run_dir: Path | None = None, zero_shot_config: str | None = None, calibrated: bool = False) -> LocalPredictor:
    from .evaluate import load_trained
    from .train import TrainConfig, build_model, load_tokenizer, resolve_device, resolve_dtype

    if zero_shot_config:
        cfg = TrainConfig.load(zero_shot_config, ["lora_r=0"])
        name = f"{cfg.run_name}_zero_shot"
    else:
        cfg = TrainConfig.load(run_dir / "config.yaml")
        name = run_dir.name
    device = resolve_device(cfg.device)
    tokenizer, pad_id = load_tokenizer(cfg)
    model = build_model(cfg, tokenizer, dtype=resolve_dtype(cfg.dtype, device)).to(device).eval()
    temperatures = load_trained(run_dir, cfg, model) if run_dir else None  # always loads the trained weights
    if not calibrated:
        temperatures = None
    elif temperatures:
        name += "-calibrated"
    predictor = LocalPredictor(name, model, tokenizer, pad_id, temperatures, cfg.max_options, device)
    predictor.config.update({
        "model_kind": cfg.model_kind, "backbone": cfg.backbone, "readout_loss": cfg.readout_loss,
        "label_mode": cfg.label_mode, "trained": run_dir is not None, "temperature_scaled": bool(temperatures),
    })
    return predictor


def load_examples(split: str, config_path: Path | None):
    from .data import load_typed_decisions
    from .train import TrainConfig, load_data

    if split == "test":
        return load_typed_decisions("test")
    cfg = TrainConfig.load(config_path) if config_path else TrainConfig()
    return load_data(cfg)[1]  # the same seeded validation split used in training


def run_experiments(predictor, dataset, tests: list[str], split: str, limit: int | None, threads: int) -> None:
    import opik
    from opik.evaluation import evaluate

    predict = opik.track(name=f"{predictor.name}.predict")(predictor.predict)
    for test in tests:
        metrics, experiment_functions = TESTS[test]
        evaluate(
            dataset=dataset,
            task=make_task(predict, test),
            scoring_metrics=metrics(),
            experiment_scoring_functions=experiment_functions,
            experiment_name=f"{predictor.name}-{test}-{split}",
            project_name=PROJECT,
            experiment_config={"model": predictor.name, "test": test, "split": split, **predictor.config},
            experiment_tags=[test, split, predictor.config["kind"]],
            nb_samples=limit,
            task_threads=threads,
        )


def main(argv: list[str] | None = None) -> None:
    import opik

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="*", default=[], type=Path, help="run directories written by train.py")
    parser.add_argument("--zero-shot", nargs="*", default=[], help="configs to evaluate untrained")
    parser.add_argument("--calibrated", action="store_true", help="apply each trained run's fitted temperatures")
    parser.add_argument("--jev", action="store_true", help="also evaluate TypeSafe's Jev (needs TYPESAFE_API_KEY)")
    parser.add_argument("--jev-model", default="jev-latest")
    parser.add_argument("--tests", nargs="*", default=list(TESTS), choices=list(TESTS))
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--limit", type=int, help="evaluate only the first N cases")
    args = parser.parse_args(argv)

    config_path = args.runs[0] / "config.yaml" if args.runs else None
    items = [example_to_item(e) for e in load_examples(args.split, config_path)]
    dataset = opik.Opik(project_name=PROJECT).get_or_create_dataset(
        f"typed-decisions-{args.split}", description=f"LocalLLaMA/typed-decisions, {args.split} split")
    dataset.insert(items)

    loaders = [lambda d=d: load_local(run_dir=d, calibrated=args.calibrated) for d in args.runs]
    loaders += [lambda c=c: load_local(zero_shot_config=c) for c in args.zero_shot]
    for load in loaders:
        predictor = load()  # one model on the GPU at a time
        run_experiments(predictor, dataset, args.tests, args.split, args.limit, threads=1)
        del predictor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.jev:
        jev = JevPredictor(args.jev_model)
        run_experiments(jev, dataset, args.tests, args.split, args.limit, threads=8)
        print(f"Jev versions reported by the API: {sorted(jev.reported_models) or 'none reported'}")


if __name__ == "__main__":
    main()
