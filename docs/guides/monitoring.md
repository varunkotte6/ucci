# Monitoring and recalibration

!!! note "Extension, not in the paper"
    UCCI as published fits the calibration map once, on a held-out batch (Section 4.2). The
    paper's Section 7 ("Static calibration") names the gap: in streaming deployments with
    distribution shift, online or continual recalibration is needed, within the same calibrated
    threshold framework. `ucci.online` provides that. It is not used by the paper-faithful core
    and is never on by default.

A calibrated threshold means "escalate when the small answer is wrong with probability above
\(\theta\)". That meaning holds only while \(\hat p\) stays calibrated, and calibration can be tested
on fresh labels. `ucci.online` has two tools for this:

- `CalibrationMonitor` scores the forecasts on a sliding window of labelled queries and raises
  a drift flag from a two-sided calibration test;
- `RecalibratingRouter` keeps \(\theta\) fixed as an error-probability threshold and refits the
  isotonic map on a sliding window of recent labels.

## Detect drift

```python
import numpy as np
from ucci import IsotonicCalibrator
from ucci.online import CalibrationMonitor

rng = np.random.default_rng(0)

def traffic(n, shift=0.0):
    """Simulated labelled traffic; `shift` raises the true error rate at every u."""
    u = rng.beta(2, 5, n)
    p_wrong = 1 / (1 + np.exp(-10 * (u - 0.45 + shift)))
    return u, (rng.random(n) < p_wrong).astype(float)

u_cal, e_cal = traffic(5000)
calibrator = IsotonicCalibrator().fit(u_cal, e_cal)

monitor = CalibrationMonitor(calibrator, window=2000, alpha=0.01, test="binomial")
monitor.update(*traffic(2000))             # same distribution as the calibration data
print(monitor.stats().drift)                # expected: False

monitor.update(*traffic(2000, shift=0.1))  # the small model got worse
s = monitor.stats()
print(s.drift, round(s.mean_predicted, 3), round(s.observed_error_rate, 3), round(s.z, 1))
```

`update(u, e)` computes \(\hat p = g(u)\) with the forecaster as it is at that moment (prequential
scoring) and adds the pair to the window; `record(p_hat, e)` adds logged forecasts directly.
`stats()` returns the window size, mean forecast against observed error rate, expected and
observed error counts, windowed ECE, the test statistic, its p-value and the drift flag.

Two tests are available, both two-sided with a normal approximation:

- `test="binomial"` (calibration in the large): \(z = (O - E)/\sqrt{V}\) with \(O = \sum e_i\),
  \(E = \sum \hat p_i\), \(V = \sum \hat p_i (1 - \hat p_i)\). It detects a shift of the overall error rate.
- `test="spiegelhalter"`: Spiegelhalter's (1986) z statistic, which tests the Brier score
  against its expectation under calibration. It also reacts to forecasts that become too
  extreme or too flat while the mean error rate is unchanged.

The flag is raised only when the window holds at least `min_count` labels and the p-value is
below `alpha`. The false-alarm rate `alpha` holds per check: checking after every label over
overlapping windows makes many dependent tests, so use `alpha / K` for \(K\) checks or check once
per window. Both tests assume the labels are a representative sample of the routed traffic.

## Recalibrate on a sliding window

```python
from ucci import UCCIRouter
from ucci.online import RecalibratingRouter

router = UCCIRouter().calibrate(u_cal, e_cal)
router.theta = 0.3            # or router.choose_threshold(...) on validation data

online = RecalibratingRouter.from_router(router, window=5000, refit_every=500, min_labels=1000)
n_refits = online.update(*traffic(4000, shift=0.1))
print(n_refits, online.labels_in_window, online.escalate(0.25))

snapshot = online.to_router()   # a UCCIRouter in the shared router format
print(snapshot)
```

Because \(\theta\) lives on the probability scale, the implied cut on \(u(x)\) moves as \(g\) is
refit. Feeding labels one at a time or in batches gives identical states. Re-selecting \(\theta\)
itself (Eq. 7) needs large-model outcomes on a validation set; call `ucci.select_threshold`
on fresh validation data when the accuracy target must be re-certified. A snapshot from
`to_router()` has no `tau` for that reason.

## Reference

D. J. Spiegelhalter. Probabilistic prediction in patient management and clinical trials.
*Statistics in Medicine* 5(5):421-433, 1986.
doi:[10.1002/sim.4780050506](https://doi.org/10.1002/sim.4780050506).

API reference: [`ucci.online`](../api/online.md).
