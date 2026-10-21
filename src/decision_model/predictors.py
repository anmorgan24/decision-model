"""One interface for every model we evaluate: System One in, System One out.

    predictor.predict(state, questions) -> {"answers": {qid: answer}}

`state` and `questions` are the body of a System One request, and the answers
follow the System One response shape. Our trained models and hosted models like
Jev implement the same method, so the evaluation code never needs to know which
one it's calling.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

import torch

from .inference import predict as local_predict


class LocalPredictor:
    """A model trained with this repo (decision head or logit readout)."""

    def __init__(self, name: str, model, tokenizer, pad_id: int, temperatures=None,
                 max_options: int = 16, device="cpu"):
        self.name = name
        self.model, self.tokenizer, self.pad_id = model, tokenizer, pad_id
        self.temperatures, self.max_options, self.device = temperatures, max_options, device
        self.config = {"kind": "local"}

    def predict(self, state, questions: dict) -> dict:
        with torch.no_grad():
            return local_predict(self.model, self.tokenizer, state, questions, self.pad_id,
                                 self.temperatures, self.max_options, self.device)


class JevPredictor:
    """TypeSafe's hosted Jev, called through its System One endpoint.

    Reads TYPESAFE_API_KEY from the environment. Any server that speaks the same
    protocol works by changing base_url (TYPESAFE_BASE_URL).
    """

    def __init__(self, model: str = "jev-latest", base_url: str | None = None, api_key: str | None = None,
                 timeout: float = 60.0, retries: int = 4):
        self.name = model
        self.model = model
        self.base_url = (base_url or os.environ.get("TYPESAFE_BASE_URL", "https://api.typesafe.ai")).rstrip("/")
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY")
        if not self.api_key:
            raise ValueError("Set TYPESAFE_API_KEY to evaluate Jev")
        self.timeout, self.retries = timeout, retries
        self.config = {"kind": "api", "model": model, "base_url": self.base_url}
        self.reported_models: set[str] = set()  # the model versions the API says answered

    def predict(self, state, questions: dict) -> dict:
        body = json.dumps({"model": self.model, "state": state, "questions": questions}).encode()
        request = urllib.request.Request(
            f"{self.base_url}/v1/systemone", data=body, method="POST",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    result = json.loads(response.read())
                if "model" in result:
                    self.reported_models.add(str(result["model"]))
                return {"answers": result["answers"]}
            except urllib.error.HTTPError as error:
                if error.code not in (429, 500, 502, 503, 504, 529) or attempt == self.retries:
                    raise
            except urllib.error.URLError:
                if attempt == self.retries:
                    raise
            time.sleep(2 ** attempt)  # back off before retrying a rate limit or server error
        raise RuntimeError("unreachable")


def answer_distribution(question: dict, answer: dict, keys: list[str]) -> list[float]:
    """A System One answer as probabilities over `keys` (labels, levels as "0".."n", or "false"/"true")."""
    if question["type"] == "noul":
        p_yes = float(answer["noul"])
        return [1.0 - p_yes, p_yes]
    probs = answer.get("probabilities", {})
    vector = [float(probs.get(k, 0.0)) for k in keys]
    total = sum(vector)
    return [p / total for p in vector] if total > 0 else [1.0 / len(keys)] * len(keys)
