import torch

from decision_model import build_attention_mask, collate, pack_example


def test_positions_restart_after_state(examples, tokenizer):
    packed = pack_example(examples[0], tokenizer)
    state_len = packed.segment_ids.count(0)
    for branch in (1, 2, 3):
        positions = [p for p, s in zip(packed.position_ids, packed.segment_ids) if s == branch]
        assert positions == list(range(state_len, state_len + len(positions)))


def test_decide_index_is_last_token_of_each_branch(examples, tokenizer):
    packed = pack_example(examples[0], tokenizer)
    for branch, idx in enumerate(packed.decide_index, start=1):
        assert packed.segment_ids[idx] == branch
        assert idx + 1 == len(packed.segment_ids) or packed.segment_ids[idx + 1] != branch


def test_mask_blocks_other_branches_and_lets_branches_see_state():
    allowed = build_attention_mask([0, 0, 0, 1, 1, 2, 2])
    state, q1, q2 = [0, 1, 2], [3, 4], [5, 6]
    assert not allowed[0, 1]  # causal within the state
    assert all(allowed[t, u] for t in q1 + q2 for u in state)  # every branch sees the state
    assert not any(allowed[t, u] for t in q2 for u in q1)  # branches never see each other
    assert not any(allowed[u, t] for t in q1 + q2 for u in state)  # the state never sees a question


def test_collate_shapes_targets_and_labels(examples, tokenizer):
    packed = [pack_example(e, tokenizer) for e in examples]
    batch = collate(examples, packed, pad_id=0, max_options=8)
    max_len = max(len(p.input_ids) for p in packed)
    assert batch["attention_allowed"].shape == (3, 1, max_len, max_len)
    assert batch["attention_allowed"].dtype == torch.bool
    assert batch["attention_allowed"].any(dim=-1).all()  # no row fully blocked, padding included
    assert batch["labels"].tolist()[:3] == [0, 1, 1]  # billing, yes, level 1
    assert torch.equal(batch["targets"][0], torch.tensor([1.0, 0, 0, 0, 0, 0, 0, 0]))
    assert torch.equal(batch["num_options"][:3], torch.tensor([3, 2, 3]))
    assert batch["question_ids"][:3] == ["route", "escalate", "priority"]


def test_collate_carries_soft_targets(typed_examples, tokenizer):
    batch = collate(typed_examples, [pack_example(e, tokenizer) for e in typed_examples], pad_id=0, max_options=8)
    first = batch["targets"][0, :4]
    assert torch.allclose(first.sum(), torch.tensor(1.0))
    assert 0 < first.max() < 1  # a real distribution, not one-hot
    assert batch["question_ids"][0] == "agent_trace_observability/action"
