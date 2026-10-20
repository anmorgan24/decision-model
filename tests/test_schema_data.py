import random

from decision_model.data import hard_labels, shuffle_choice_options, split_train_val


def test_typed_decisions_row_parses_soft_gold_and_labels(typed_examples):
    e = typed_examples[0]
    assert e.workflow == "agent_trace_observability"
    assert e.state.startswith('{"agent"')
    types = {q.qid: q.type for q in e.questions}
    assert types == {"action": "choice", "needs_review": "noul", "outcome": "choice", "risk": "score", "urgency": "score"}
    action = next(q for q in e.questions if q.qid == "action")
    assert action.options == ("continue", "human_review", "observe", "stop")
    assert e.label["action"] == 1  # human_review
    assert abs(sum(e.gold["action"]) - 1) < 1e-9
    assert e.gold["needs_review"] == (0.45, 0.55)  # (no, yes)
    assert e.label["needs_review"] == 1
    assert e.label["risk"] == 0 and len(e.gold["risk"]) == 4


def test_hard_labels_are_one_hot(typed_examples):
    hard = hard_labels(typed_examples[0])
    assert hard.gold["action"] == (0.0, 1.0, 0.0, 0.0)
    assert hard.gold["needs_review"] == (0.0, 1.0)


def test_shuffle_keeps_gold_aligned_and_leaves_scores_alone(typed_examples):
    original = typed_examples[1]
    shuffled = shuffle_choice_options(original, random.Random(3))
    for q_old, q_new in zip(original.questions, shuffled.questions):
        if q_new.type == "choice":
            assert sorted(q_new.options) == sorted(q_old.options)
            for label, p in zip(q_old.options, original.gold[q_old.qid]):
                assert shuffled.gold[q_new.qid][q_new.options.index(label)] == p
            assert q_new.options[shuffled.label[q_new.qid]] == q_old.options[original.label[q_old.qid]]
        else:
            assert q_new == q_old and shuffled.gold[q_new.qid] == original.gold[q_old.qid]


def test_split_is_deterministic_and_stratified(typed_examples):
    import dataclasses

    examples = [dataclasses.replace(typed_examples[i % 2], id=f"{w}_{i}", workflow=w)
                for w in ("a", "b") for i in range(20)]
    train1, val1 = split_train_val(examples, 0.25, seed=0)
    train2, val2 = split_train_val(examples, 0.25, seed=0)
    assert [e.id for e in val1] == [e.id for e in val2]
    assert sum(e.workflow == "a" for e in val1) == sum(e.workflow == "b" for e in val1) == 5
    assert not {e.id for e in train1} & {e.id for e in val1}
