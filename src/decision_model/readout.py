"""The baseline: read answer probabilities from the language-model head.

This is how most open-source "Jev alternatives" work. After "Answer:", read the
next-token logits for the option letters (" A", " B", ...) or for " No"/" Yes",
and apply softmax over those tokens only. No text is generated.

It takes the same packed batch as DecisionModel, so the two differ only in the
readout: rows of a pretrained LM head versus a decision head trained from
scratch. Because packed and standalone questions get identical hidden states,
this is equivalent to asking each question in its own sequence.

loss_space controls what fine-tuning optimizes:
  "restricted": cross-entropy over the answer tokens only (a proper scoring rule
                over the answer space)
  "full_vocab": cross-entropy of the answer token against the whole vocabulary,
                like ordinary supervised fine-tuning
Both report the same renormalized distribution over the answer tokens.
"""

from __future__ import annotations

import torch
from torch import nn

from .packing import OPTION_LETTERS, QUESTION_TYPE_CODES, additive_mask

ANSWER_TEXT = {
    "choice": [f" {letter}" for letter in OPTION_LETTERS],
    "score": [f" {letter}" for letter in OPTION_LETTERS],
    "noul": [" No", " Yes"],
}


def answer_token_ids(tokenizer, max_options: int) -> torch.Tensor:
    """(3, max_options) table: the token id that answers each option slot, per question type."""
    table = torch.zeros(len(QUESTION_TYPE_CODES), max_options, dtype=torch.long)
    for qtype, code in QUESTION_TYPE_CODES.items():
        for slot, text in enumerate(ANSWER_TEXT[qtype][:max_options]):
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"{text!r} is {len(ids)} tokens for this tokenizer; the readout needs one")
            table[code, slot] = ids[0]
    return table


class LogitReadoutModel(nn.Module):
    def __init__(self, causal_lm: nn.Module, answer_ids: torch.Tensor, loss_space: str = "restricted"):
        super().__init__()
        if loss_space not in ("restricted", "full_vocab"):
            raise ValueError(f"loss_space must be 'restricted' or 'full_vocab', not {loss_space!r}")
        self.lm = causal_lm
        self.register_buffer("answer_ids", answer_ids)
        self.loss_normalizes = loss_space == "restricted"
        self.max_options = answer_ids.shape[1]

    def forward(self, batch: dict) -> torch.Tensor:
        decoder = self.lm.get_decoder()
        hidden = decoder(
            input_ids=batch["input_ids"],
            attention_mask=additive_mask(batch["attention_allowed"], decoder.dtype),
            position_ids=batch["position_ids"],
        ).last_hidden_state
        decision_states = hidden[batch["question_batch_index"], batch["question_position"]]
        vocab_logits = self.lm.get_output_embeddings()(decision_states).float()  # (N, vocab)
        if not self.loss_normalizes:
            vocab_logits = vocab_logits.log_softmax(dim=-1)

        slot_logits = vocab_logits.gather(1, self.answer_ids[batch["question_type"]])
        slots = torch.arange(self.max_options, device=slot_logits.device)
        return slot_logits.masked_fill(slots[None, :] >= batch["num_options"][:, None], float("-inf"))
