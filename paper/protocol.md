# RAGTruth evaluation protocol

Fixed on 11-10-2026.
Author: Marco Pérez Padilla

## Purpose

HaluEval QA is synthetic and its responses are mostly short answer spans.
RAGTruth has organic, full-sentence responses written by real models. This
protocol fixes, in advance, how the detectors are run on RAGTruth and what
each possible outcome means, so that no choice is made after seeing the
test results.

## Data

Facts read from the raw files with `scripts/inspect_ragtruth.py`:

- Only the QA task is used: 989 sources and 5934 responses, always 6
  responses per source (one per generating model).
- Official partition (given in `response.jsonl`): train has 5034 responses
  (839 sources, 31.1% positive); test has 900 responses (150 sources,
  17.8% positive).
- Quality flags in QA: good, incorrect_refusal (143), truncated (24).

## Inputs, labels and splits

- Context: `source_info["passages"]` verbatim, including the "passage N:"
  markers, with leading and trailing whitespace stripped. The question is
  not part of the context. HaluEval follows the same rule (knowledge
  passage only).
- Response: used exactly as provided.
- Label: a response is positive if it has at least one annotated
  hallucination span. The span flags `implicit_true` and `due_to_null` are
  ignored.
- No filtering on `quality`: every response is kept in every split. A
  deployed guardrail also sees refusals and truncated outputs, and the
  affected slice is 2.8% of QA responses.
- Test: the official test partition, untouched.
- Validation: 15% of the official train, drawn by source id (all responses
  to one source stay on the same side) with seed 42. The rest of the train
  is the training set for the in-domain model.
- Thresholds are calibrated on validation for every detector, including
  the ones that were not trained on RAGTruth. Metrics are reported on test.

## Detectors

- lexical: unigram overlap, identical to HaluEval. No changes.
- distilbert-transfer: the model trained on HaluEval train, with no further
  training.
- distilbert-indomain: fine-tuned on the RAGTruth training set with the
  same hyperparameters as on HaluEval (3 epochs, learning rate 2e-5, batch
  size 16, seed 42). The checkpoint with the lowest validation loss is
  kept.
- Both distilbert arms: maximum 512 tokens. The passages are truncated and
  the response is never truncated (fallback to longest_first only if the
  response alone exceeds the limit).
- nli (`nli_chunked`, a single fixed configuration):
  cross-encoder/nli-deberta-v3-base in fp32. The response is split into
  sentences with a regex on terminal punctuation and newlines. The
  passages are split on sentence boundaries into windows of at most 350
  tokens. The support of a sentence is the maximum P(entailment) over the
  windows. The response score is the maximum over its sentences of
  (1 - support). A response with no sentences, or a context with no text,
  scores 1.0.

Every arm listed here is reported, whatever its result.

## Metrics

- Primary: AUC-ROC with a 95% percentile bootstrap interval (1000
  resamples, resampling whole sources, seed 42). AUC is primary because
  the positive rate differs between train (31%) and test (18%), which
  shifts F1, precision and recall at a validation-calibrated threshold for
  every detector.
- Secondary: precision, recall and F1 of the positive class at the
  calibrated threshold. Accuracy is not reported.
- Latency: p50 and p95 per response, timed one response at a time after 3
  warm-up calls, on the first 200 test responses in file order. For
  nli_chunked it includes sentence splitting and windowing. The GPU, driver
  and library versions are recorded with the results.

## Frontier rule

Detector A dominates detector B if A's p50 latency is less than or equal
to B's and A's AUC interval lies entirely above B's AUC interval. A
detector is on the frontier if no other detector dominates it. If nli is
dominated it is reported as a negative result; otherwise it stays on the
frontier.

## Not allowed after seeing test results

- Changing the nli model, window size, aggregation or any threshold.
- Replacing a primary result with a variant. Variants may be run on
  validation only and must be labelled exploratory.
- A genuine bug (a crash or an empty output) may be fixed, but the fix is
  logged at the bottom of this file with its date.

## Limitations declared up front

- Many RAGTruth inputs are long. The distilbert arms lose evidence to
  truncation while nli_chunked does not (it windows the passages). This
  asymmetry is intentional and is reported as such, together with the
  share of test pairs that exceed 512 tokens.
- Class balance differs from HaluEval, and between RAGTruth train and test.
- Words with missing spaces occur in both responses and passages. Texts
  are used as provided.
- With 160 positive responses in test, the intervals will be wide.
- Conclusions apply only to these detectors, this dataset and these
  settings. They do not support claims about hallucination detectors in
  general.

## Changes to this protocol

Any change made after the first detector has been run on RAGTruth is
appended below as a dated amendment. The text above is not edited.

## Amendments

### 14-10-2026

Seen before this amendment: the test results of lexical, distilbert_transfer
and distilbert_indomain on RAGTruth, and train-only diagnostics
(response length and per-model lexical breakdown). Not seen yet: any
nli_chunked result.

Findings that motivate it (train only): response length alone separates the
classes (AUC 0.972 on HaluEval, 0.707 on RAGTruth QA train), and the pooled
lexical AUC on RAGTruth train (0.728) is higher than the AUC inside every
single generating model (0.605 to 0.699).

Additions. They do not change any arm, threshold, primary metric or the
frontier rule above:

- A reference baseline `length`: the word count of the response used as the
  score, with no parameters. It is reported on every dataset and excluded
  from the frontier rule. The AUC difference between each detector and this
  baseline is reported with its bootstrap interval.
- A secondary metric: AUC within each generating model (RAGTruth only, on
  the official test), with the pooled AUC remaining the primary metric. A
  deployed guardrail protects one model, so the within-model AUC is the
  deployment-relevant number.


### 17-10-2026 

Seen before this amendment: the per-model AUC table for lexical,
distilbert_transfer, distilbert_indomain and length on the RAGTruth test,
and the pooled results listed above. Not seen yet: any nli_chunked result.

Why: the per-model AUCs are too noisy to read one by one (a model with a
single positive response in test has an AUC decided by one comparison).

Additions. They do not change any arm, threshold, primary metric or the
frontier rule:

- Within-model AUC, summarised in one number: the probability that a
  hallucinated response outscores a correct response written by the same
  model (pairs from different models are not compared), with each model
  weighted by its number of (hallucinated, correct) pairs. It gets a 95%
  bootstrap interval resampling sources (1000 resamples, seed 42). This is
  the definition of the secondary metric of the previous amendment; no
  alternative was tried.
- Per-model rows are listed with their own bootstrap interval. A model with
  fewer than 20 positive responses in test is flagged as not interpretable
  and is never used in a conclusion.
- Pooled AUC: 95% bootstrap interval, resampling sources.
- For every detector, the AUC difference with the `length` baseline, with a
  paired 95% bootstrap interval (same resamples for both).
- The frontier table, computed with the rule above on the pooled AUC and
  p50 latency. `length` is listed as "reference" and takes no part in the
  rule.