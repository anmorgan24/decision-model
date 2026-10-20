"""The decision head: turns each question's decision-token hidden state into logits.

Every question type returns logits over option slots, so one loss and one set
of metrics cover all three:

  choice: one logit per listed option (slot k = the k-th option shown)
  score:  one logit per level, in order
  noul:   logits [0, z], so softmax gives P(yes) = sigmoid(z)

Slots beyond a question's real number of options are masked out.

The layers start at zero, so before training every question gets a uniform
distribution: the model starts out knowing nothing and saying so.
"""

from __future__ import annotations

import torch
from torch import nn

from .packing import QUESTION_TYPE_CODES

CHOICE, SCORE, NOUL = (QUESTION_TYPE_CODES[t] for t in ("choice", "score", "noul"))


class DecisionHead(nn.Module):
    def __init__(self, hidden_size: int, max_options: int = 16):
        super().__init__()
        self.max_options = max_options
        self.choice = nn.Linear(hidden_size, max_options)
        self.score = nn.Linear(hidden_size, max_options)
        self.noul = nn.Linear(hidden_size, 1)
        for layer in (self.choice, self.score, self.noul):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, hidden: torch.Tensor, question_type: torch.Tensor, num_options: torch.Tensor) -> torch.Tensor:
        """hidden: (N, d), one row per question. Returns logits of shape (N, max_options)."""
        hidden = hidden.float()
        z = self.noul(hidden)
        noul = torch.cat([torch.zeros_like(z), z, torch.zeros(len(z), self.max_options - 2, device=z.device)], dim=1)

        qtype = question_type[:, None]
        logits = torch.where(qtype == CHOICE, self.choice(hidden), torch.where(qtype == SCORE, self.score(hidden), noul))

        slots = torch.arange(self.max_options, device=logits.device)
        return logits.masked_fill(slots[None, :] >= num_options[:, None], float("-inf"))
