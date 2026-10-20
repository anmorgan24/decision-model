"""A decision model: a pretrained transformer backbone plus a decision head.

One forward pass reads the packed sequence, gathers the hidden state at each
question's decision token, and returns logits over that question's options.
Nothing is generated.
"""

from __future__ import annotations

import torch
from torch import nn

from .head import DecisionHead
from .packing import additive_mask

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class DecisionModel(nn.Module):
    loss_normalizes = True  # the loss applies softmax over option slots

    def __init__(self, backbone: nn.Module, max_options: int = 16):
        super().__init__()
        self.backbone = backbone  # a base model without its LM head, e.g. Qwen3Model
        self.head = DecisionHead(backbone.config.hidden_size, max_options)

    def forward(self, batch: dict) -> torch.Tensor:
        hidden = self.backbone(
            input_ids=batch["input_ids"],
            attention_mask=additive_mask(batch["attention_allowed"], self.backbone.dtype),
            position_ids=batch["position_ids"],
        ).last_hidden_state
        decision_states = hidden[batch["question_batch_index"], batch["question_position"]]
        return self.head(decision_states, batch["question_type"], batch["num_options"])


def decision_loss(logits: torch.Tensor, targets: torch.Tensor, batch_size: int, normalize: bool = True) -> torch.Tensor:
    """Cross-entropy against the gold distribution, summed over questions, averaged over examples.

    targets holds a probability per option slot: soft gold, or one-hot for hard
    labels (then this is ordinary cross-entropy). Cross-entropy is a proper
    scoring rule: its expected value is lowest when the predicted distribution
    equals the true one, which is what pushes training toward calibration.

    normalize=False treats logits as log-probabilities already (used by the
    logit-readout baseline when it trains against the full vocabulary).
    """
    log_probs = logits.log_softmax(dim=-1) if normalize else logits
    log_probs = log_probs.masked_fill(targets == 0, 0.0)  # avoid 0 * -inf in masked slots
    return -(targets * log_probs).sum() / batch_size


def add_lora(model: nn.Module, r: int = 16, alpha: int = 32, dropout: float = 0.05) -> nn.Module:
    from peft import LoraConfig, get_peft_model

    return get_peft_model(model, LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=LORA_TARGETS))


def freeze(model: nn.Module) -> nn.Module:
    for p in model.parameters():
        p.requires_grad_(False)
    return model
