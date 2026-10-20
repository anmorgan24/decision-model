"""End-to-end: the training pipeline runs and the evaluation reloads what it saved."""

import json

import pytest
import torch

from decision_model.evaluate import load_trained
from decision_model.inference import run_model
from decision_model.metrics import log_loss
from decision_model.train import TrainConfig, build_model, run

from conftest import PAD_ID, ToyTokenizer, tiny_qwen3, tiny_qwen3_lm


def config(tmp_path, kind, **overrides):
    values = dict(run_name=kind, model_kind=kind, lora_r=4, epochs=2, batch_size=2, grad_accum=1, max_options=8,
                  lr_backbone=5e-3, lr_head=5e-2, val_fraction=0.0, output_dir=str(tmp_path), comet=False, device="cpu")
    values.update(overrides)
    return TrainConfig(**values)


@pytest.mark.parametrize("kind,factory", [("decision_head", tiny_qwen3), ("logit_readout", tiny_qwen3_lm)])
def test_pipeline_trains_saves_and_reloads(tmp_path, typed_examples, kind, factory):
    cfg = config(tmp_path, kind, epochs=8, patience=8)
    tokenizer = ToyTokenizer()
    run_dir = run(cfg, tokenizer, PAD_ID, pretrained=factory(), data=(typed_examples, typed_examples))

    summary = json.loads((run_dir / "val_summary.json").read_text())
    assert summary["raw"]["overall"]["n"] == 10
    assert set(json.loads((run_dir / "temperatures.json").read_text())) == {"choice", "noul", "score"}

    # A fresh model with the saved weights gives the same predictions as the trained one.
    fresh = build_model(cfg, tokenizer, pretrained=factory())
    load_trained(run_dir, cfg, fresh)
    out = run_model(fresh, typed_examples, tokenizer, PAD_ID, max_options=8)
    assert abs(summary["raw"]["overall"]["log_loss"] - log_loss(out.logits, out.targets, out.labels)) < 1e-4


def test_training_moves_predictions_toward_the_gold_distributions(tmp_path, typed_examples):
    from decision_model.inference import gold_outputs
    from decision_model.metrics import kl

    cfg = config(tmp_path, "decision_head", epochs=10, patience=10, lr_head=1e-2)
    run_dir = run(cfg, ToyTokenizer(), PAD_ID, pretrained=tiny_qwen3(), data=(typed_examples, typed_examples))
    summary = json.loads((run_dir / "val_summary.json").read_text())
    untrained = gold_outputs(typed_examples, 8)  # the zero-initialized head predicts uniform
    # With soft gold, log loss can't drop below the gold's own entropy; KL can reach zero.
    assert summary["raw"]["overall"]["kl"] < 0.8 * kl(untrained.logits, untrained.targets, untrained.labels)
    assert all(torch.isfinite(torch.tensor(t)) for t in json.loads((run_dir / "temperatures.json").read_text()).values())


def test_linear_probe_trains_only_the_head(typed_examples):
    cfg = TrainConfig(model_kind="decision_head", lora_r=0, max_options=8, comet=False)
    model = build_model(cfg, ToyTokenizer(), pretrained=tiny_qwen3())
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and all(n.startswith("head.") for n in trainable)


def test_comet_offline_logging(tmp_path, typed_examples):
    pytest.importorskip("comet_ml")
    offline = tmp_path / "comet"
    offline.mkdir()
    cfg = config(tmp_path, "decision_head", comet=True, comet_offline_dir=str(offline), epochs=1)
    run(cfg, ToyTokenizer(), PAD_ID, pretrained=tiny_qwen3(), data=(typed_examples, typed_examples))
    assert list(offline.glob("*.zip")), "Comet should write an offline experiment archive"
