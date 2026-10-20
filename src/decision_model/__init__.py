"""A minimal decision model: a transformer backbone, a packed input, and a decision head."""

from .head import DecisionHead
from .inference import Outputs, predict, run_model
from .metrics import TemperatureScaler, summarize
from .model import DecisionModel, decision_loss
from .packing import build_attention_mask, collate, format_question, pack_example
from .readout import LogitReadoutModel
from .schema import Example, Question, load_jsonl, parse_row, parse_typed_decisions_row

__all__ = [
    "DecisionHead",
    "DecisionModel",
    "Example",
    "LogitReadoutModel",
    "Outputs",
    "Question",
    "TemperatureScaler",
    "build_attention_mask",
    "collate",
    "decision_loss",
    "format_question",
    "load_jsonl",
    "pack_example",
    "parse_row",
    "parse_typed_decisions_row",
    "predict",
    "run_model",
    "summarize",
]
