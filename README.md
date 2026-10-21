# Decision model tutorial

A minimal decision model built on a pretrained transformer backbone. It reads a
state and several typed questions in one forward pass and returns a probability
for every allowed answer. Nothing is generated.

Companion code for an article on what a decision model is and what training one
looks like, with training tracked in Comet and evaluation in Opik.

## What gets compared

Two models share everything except the readout:

| | Decision head | Logit-readout baseline |
|---|---|---|
| Backbone | Qwen3-0.6B-Base + LoRA | the same |
| Input | packed: state once, questions isolated | the same |
| Readout | a head trained from scratch on the decision-token hidden state | LM-head logits for " A", " B", ... or " No"/" Yes" |
| Loss | cross-entropy against the gold distribution | the same, over answer tokens (or the full vocabulary) |

Packing changes cost, not predictions: each question gets the same hidden
states it would get alone (tested). So accuracy and calibration differences
come from the readout and how it's trained, and latency differences come from
packing. `bench_latency.py` measures the second.

## Data

[LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions)
(Apache-2.0): 1,600 synthetic cases across four workflows, including agent-trace
observability, with five typed questions per case. Gold answers are soft: the
mean of three samples from a roughly 4B-class teacher. Validation is carved out
of `train` (15% of each workflow, seeded); `test` is scored once, at the end.

Read results with the card's caveats in mind:

- Scores measure agreement with the teacher. Teacher self-agreement is 0.735
  accuracy, so scores well above it mean learning the teacher's quirks.
- A model fine-tuned on `train` is a *specialist*. Compare it with the card's
  fitted/fine-tuned table, not with zero-shot models like Jev.
- The card's Prior row has the best ECE while reading nothing, which is why
  KL and Brier matter more than ECE alone.

## Run order

```bash
pip install -e ".[tracking,dev]"
export COMET_API_KEY=...            # optional; set comet: false in a config to skip

pytest                              # offline, about 5 seconds

# 1. Check our scorer reproduces the card's Prior and Uniform rows
python -m decision_model.references

# 2. Zero-shot baseline: the logit readout with no fine-tuning
python -m decision_model.evaluate --config configs/logit_readout.yaml --zero-shot --split val

# 3. Train (each run is one Comet experiment). Each step changes one thing:
python -m decision_model.train configs/logit_readout_sft.yaml        # the common recipe: full vocabulary, hard labels
python -m decision_model.train configs/logit_readout.yaml            # objective: answer tokens only, soft labels
python -m decision_model.train configs/decision_head.yaml            # readout: a fresh decision head
python -m decision_model.train configs/logit_readout_full_vocab.yaml # optional: full vocabulary, soft labels
python -m decision_model.train configs/linear_probe.yaml             # optional: frozen backbone

# 4. Score each run on validation, then compare pairs with confidence intervals
python -m decision_model.evaluate outputs/logit_readout_sft --split val
python -m decision_model.evaluate outputs/logit_readout --split val
python -m decision_model.evaluate outputs/decision_head --split val
python -m decision_model.compare outputs/logit_readout_zero_shot outputs/logit_readout_sft --split val
python -m decision_model.compare outputs/logit_readout_sft outputs/logit_readout --split val --by-type
python -m decision_model.compare outputs/logit_readout outputs/decision_head --split val --by-type

# 5. What packing saves
python -m decision_model.bench_latency configs/decision_head.yaml

# 6. Once every decision is made on validation, score test
python -m decision_model.evaluate outputs/decision_head --split test
python -m decision_model.evaluate outputs/logit_readout --split test
python -m decision_model.evaluate outputs/logit_readout_sft --split test
```

`compare.py` runs a paired bootstrap over cases (a case's five answers are
correlated, so cases are resampled, not decisions) and prints a 95% interval
for each difference. "neither" means the interval includes zero.

Override any config value from the command line, for example a quick run:
`python -m decision_model.train configs/decision_head.yaml --set epochs=1 max_train_examples=64`.

## What each run logs to Comet

- Hyperparameters and the trainable parameter count
- Training loss and learning rate per optimizer step
- Validation metrics per epoch: overall, per question type, and per question
- A reliability diagram per epoch
- For the best epoch: confusion matrices per question, and metrics before and
  after temperature scaling
- The train/val split ids as a dataset artifact
- The checkpoint folder as a model

## Layout

```
src/decision_model/
  schema.py         data format: state, typed questions, soft gold, labels
  data.py           benchmark loading, seeded split, option shuffling
  packing.py        one sequence per example, shared-state attention mask
  head.py           decision head (zero-initialized: starts uniform)
  model.py          backbone + head, soft-target loss, LoRA
  readout.py        logit-readout baseline on the same batch format
  metrics.py        accuracy, KL, log loss, Brier, ECE, per-type temperatures
  inference.py      batched inference and System One-shaped answers
  references.py     Prior and Uniform baselines, scorer check
  tracking.py       Comet wrapper (no-op when off)
  train.py          config-driven training
  evaluate.py       scoring a run or a zero-shot baseline
  bench_latency.py  packed versus one-sequence-per-question
  compare.py        paired bootstrap comparison of two runs
configs/            one YAML per run
scripts/            article figures (plot_attention_mask.py)
tests/              offline tests on tiny random Qwen3 models
```

## Design notes

**Input layout.** `[state][question 1 ... Answer:][question 2 ... Answer:]`.
Each branch attends to the state and to itself, and its position ids restart
after the state.

**One loss for every question type.** Choice and score questions get one logit
per option. Yes/no questions get logits `[0, z]`, so softmax gives
`P(yes) = sigmoid(z)`. Cross-entropy against the gold distribution covers all
three, and it's a proper scoring rule.

**Option shuffling.** The head scores option slots by position, so training
shuffles the order of choice options every epoch. Without it, the head can
learn "billing is always slot A" instead of reading the options.

**Zero-initialized head.** Before training, every answer is uniform.

**Float attention mask.** In transformers 5, eager attention misreads a
boolean 4D mask. The mask is passed as an additive float mask in the
backbone's own dtype, which works under every attention implementation.

**Temperature fitting with masked slots.** Unused option slots hold `-inf`,
and the gradient of `-inf / T` is NaN even where the probability is zero, so
the fit replaces them with a large finite value.

**Comet offline mode.** Offline experiments can't log artifacts, so the
tracker falls back to logging the split file as an asset.

## Scorer check against the benchmark card

`python -m decision_model.references` scores two baselines that read nothing
and compares them with the typed-decisions card:

- **KL and Brier** use the card's definitions. Brier here is squared error
  against the gold distribution, summed over options (`brier_label` is the
  version against the one-hot label).
- **Accuracy** counts a tied top prediction as a random pick among the tied
  options, so a uniform prediction scores 1/K.
- **ECE** is standard top-label ECE with 10 bins. The card computes ECE
  differently, so compare ECE only between models scored by this code.
- `KL_EPS` in `metrics.py` clamps predicted probabilities before taking logs.
  This mostly affects models that put near-zero probability on options.
