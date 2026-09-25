# Collecting data from live traffic

A UCCI router is fitted on data from the workload it will route. Section 6.1 of the paper
needs three disjoint splits of labelled queries: calibration (to fit \(g\)), validation (to
select \(\theta^*\); both models must have answered these), and test (to evaluate end to end).
This page shows how to collect them from a running system with `ucci.integrations.cascade`.

## 1. Log a shadow period

Run the small model on every query, as in production, and also run the large model on the
queries you will label (`shadow_large=True`). The served answer is still the routed one; the
extra large-model call is what threshold selection needs, because Eq. 7 compares the actual
outputs of both models. `JsonlLogger` writes one validated record per query in the format the
`ucci` tools read.

The model functions below are stand-ins so the page runs anywhere; in practice they call your
serving stack (for example `ucci.integrations.openai.make_chat_fn`).

```python
import json
import math
import random

from ucci import UCCIRouter
from ucci.integrations.cascade import Cascade, JsonlLogger, attach_labels
from ucci.integrations.openai import signals_from_chat_completion

random.seed(0)
gold = {f"q{i}": random.choice(["Canon", "Nikon", "Sony"]) for i in range(1000)}

def chat_response(text, p1, p2):
    """The shape of a Chat Completions response with top_logprobs=2."""
    return {"choices": [{"index": 0, "finish_reason": "stop", "message": {"content": text},
            "logprobs": {"content": [{"token": text, "logprob": math.log(p1), "top_logprobs": [
                {"token": text, "logprob": math.log(p1)},
                {"token": "?", "logprob": math.log(p2)}]}]}}]}

def small_fn(qid):            # stand-in for the small model: (answer, raw response)
    p1 = random.uniform(0.4, 1.0)
    answer = gold[qid] if random.random() < p1 else "Unknown"
    return answer, chat_response(answer, p1, (1 - p1) / 2)

def large_fn(qid):            # stand-in for the large model: answer only
    return gold[qid] if random.random() < 0.95 else "Unknown"

# Before the first fit any router will do; this placeholder escalates when u > 0.5.
router = UCCIRouter.from_dict({
    "format": "ucci-router", "version": 1,
    "calibrator": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
    "theta": 0.5, "c_small": 1.0, "c_large": 3.02, "cost_model": "routing",
})
cascade = Cascade(router, small_fn, large_fn, signals_from_chat_completion, shadow_large=True)

with JsonlLogger("traffic_log.jsonl", mode="w") as log:
    for qid in gold:
        log.log(cascade(qid), id=qid, include_answers=True)
```

Each record carries `id`, `u`, the `entropy` and `max_prob` signals (when the signal function
returns `TokenSignals`), both latencies, `escalated`, `p_hat`, and with `include_answers=True`
both answers, so they can be labelled later.

## 2. Label

Label both answers of every logged query: 1 when right, 0 when wrong, or a per-query score in
\([0, 1]\). `attach_labels` fills in `small_correct` and `large_correct` by id and validates the
records:

```python
records = [json.loads(line) for line in open("traffic_log.jsonl")]
labels = {
    r["id"]: (int(r["small_answer"] == gold[r["id"]]), int(r["large_answer"] == gold[r["id"]]))
    for r in records
}
with open("labelled.jsonl", "w") as fh:
    for r in attach_labels(records, labels):
        fh.write(json.dumps(r) + "\n")
```

The calibration event is \(e(x) = 1 - \) `small_correct` (Section 4.2). If labels come from an
automatic judge, its mistakes become the calibration target; a small hand-checked sample shows
how far to trust it.

## 3. Fit, evaluate, deploy

<!-- snippet: run -->
```bash
ucci fit --data labelled.jsonl --tau 0.85 --out router.json
ucci evaluate --router router.json --data labelled.jsonl --split test --bootstrap 200
```

`ucci fit` splits the records 30 / 20 / 50 by a hash of their ids (or by a `split` field) and
writes the router. With real model calls, `--cost-from-latency` sets \(c_\ell / c_s\) from the
logged latencies, the way the paper measured its cost ratio (Section 6.1); the stand-ins above
have no meaningful latency, so this example keeps the default 3.02. Load the file into the cascade
(`UCCIRouter.load("router.json")`) and turn off `shadow_large` for normal serving.

## What to watch

- **Same system.** A router transfers only to the prompt, models and decoding it was fitted on.
  Refit after changing any of them.
- **Representative labels.** Label a random sample of all traffic, not only escalated or only
  kept queries; a biased sample covers only part of the \(u(x)\) range.
- **Enough data.** Proposition 2 puts the calibration error at \(O(n^{-1/3})\); the paper used
  22,500 calibration and 15,000 validation queries.
- **Keep monitoring.** Keep labelling a small random sample after deployment and feed it to
  `ucci.online.CalibrationMonitor` ([monitoring](monitoring.md)).
