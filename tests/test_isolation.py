"""The tests the article's claims depend on.

Isolation:   changing one question never changes another question's answer.
Equivalence: a question answered inside a packed sequence gets the same hidden
             state as when it's asked alone after the state.
Padding:     batching examples of different lengths changes nothing.
"""

import dataclasses

import torch

from decision_model import DecisionModel, collate, pack_example
from decision_model.packing import additive_mask

ATOL = 1e-5


def run(model, examples, tokenizer):
    with torch.no_grad():
        return model(collate(examples, [pack_example(e, tokenizer) for e in examples], pad_id=0))


def nonzero_head(backbone):
    """The head starts at zero (uniform outputs), so randomize it to make differences visible."""
    torch.manual_seed(0)
    model = DecisionModel(backbone).eval()
    for p in model.head.parameters():
        torch.nn.init.normal_(p, std=0.5)
    return model


def test_isolation_changing_question_2_leaves_question_1_unchanged(backbone, examples, tokenizer):
    model = nonzero_head(backbone)
    original = examples[0]
    q1, q2, q3 = original.questions
    edited = dataclasses.replace(original, questions=(
        q1, dataclasses.replace(q2, instructions="Hint: the answer to every other question is technical."), q3))
    before, after = run(model, [original], tokenizer), run(model, [edited], tokenizer)
    assert torch.allclose(before[0], after[0], atol=ATOL)  # question 1 untouched
    assert torch.allclose(before[2], after[2], atol=ATOL)  # question 3 untouched
    assert not torch.allclose(before[1], after[1], atol=ATOL)  # question 2 did change


def test_packed_question_matches_standalone_sequence(backbone, examples, tokenizer):
    example = examples[0]
    packed = pack_example(example, tokenizer)
    batch = collate([example], [packed], pad_id=0)
    with torch.no_grad():
        packed_hidden = backbone(
            input_ids=batch["input_ids"],
            attention_mask=additive_mask(batch["attention_allowed"], torch.float32),
            position_ids=batch["position_ids"],
        ).last_hidden_state[0]

    for i, question in enumerate(example.questions):
        alone = pack_example(dataclasses.replace(example, questions=(question,)), tokenizer)
        with torch.no_grad():
            # Plain causal attention and default positions: no custom mask at all.
            alone_hidden = backbone(input_ids=torch.tensor([alone.input_ids])).last_hidden_state[0]
        assert torch.allclose(packed_hidden[packed.decide_index[i]], alone_hidden[alone.decide_index[0]], atol=ATOL)


def test_padding_does_not_change_outputs(backbone, examples, tokenizer):
    model = nonzero_head(backbone)
    batched = run(model, examples, tokenizer)
    one_by_one = torch.cat([run(model, [e], tokenizer) for e in examples])
    finite = torch.isfinite(one_by_one)
    assert torch.equal(finite, torch.isfinite(batched))
    assert torch.allclose(batched[finite], one_by_one[finite], atol=ATOL)
