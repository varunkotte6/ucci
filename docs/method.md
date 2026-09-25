# The method

This page states UCCI as the paper does (arXiv v1 numbering throughout: Sections 3 to 6,
Eqs. 1 to 11, Appendix A) and links each piece to the code. [Paper to code](paper_mapping.md)
has the complete mapping and every implementation choice.

## The problem (Section 3)

Inputs \(x \sim \mathcal{D}\) have ground truth \(y\). A small model \(f_s\) and a large model
\(f_\ell\) produce \(\hat y_s = f_s(x)\) and \(\hat y_\ell = f_\ell(x)\). \(\mathrm{Acc}(\hat y, y) \in [0, 1]\)
is the accuracy metric (the paper uses micro-averaged F1) and \(c_s < c_\ell\) are the per-query
costs. A routing policy \(\pi : \mathcal{X} \to \{s, \ell\}\) returns \(\hat y_\pi(x) = f_{\pi(x)}(x)\)
(Eq. 1) at cost

\[
C_\pi(x) = c_s \, \mathbb{1}\{\pi(x) = s\} + c_\ell \, \mathbb{1}\{\pi(x) = \ell\},
\]

with accuracy \(\mathrm{Acc}_\pi(x) = \mathrm{Acc}(\hat y_\pi(x), y)\). With
\(\alpha_s = \mathbb{E}[\mathrm{Acc}(f_s(x), y)]\), \(\alpha_\ell = \mathbb{E}[\mathrm{Acc}(f_\ell(x), y)]\)
and a target \(\tau \in [\alpha_s, \alpha_\ell]\), cascade design is

\[
\min_\pi \ \mathbb{E}[C_\pi(x)] \quad \text{subject to} \quad \mathbb{E}[\mathrm{Acc}_\pi(x)] \ge \tau .
\qquad \text{(Eqs. 2 and 3)}
\]

UCCI solves this in three steps.

## Step 1: token-margin uncertainty (Section 4.1)

The small model decodes greedily. At each generated token \(t = 1, \dots, T\) let \(p_{t,1}\) and
\(p_{t,2}\) be the top-1 and top-2 next-token probabilities. The margin is
\(m_t = p_{t,1} - p_{t,2} \in [0, 1]\) and

\[
u(x) = 1 - \frac{1}{T} \sum_{t=1}^{T} m_t . \qquad \text{(Eq. 4)}
\]

Larger \(u(x)\) means the small model was less decisive: \(u = 0\) when every token had
probability 1, \(u = 1\) when every position was an exact tie. The two probabilities come with
the generation (the serving stacks in the [guides](guides/index.md) return top-\(k\)
log-probabilities on request), so the signal costs no extra model call.

Code: `ucci.token_margin_uncertainty`, `ucci.uncertainty_from_logprobs`,
`ucci.batch_uncertainty`, and the serving adapters in `ucci.integrations`. The code counts
every generated content token and excludes padding and the terminating EOS token; see
[token convention](paper_mapping.md#implementation-choices).

## Step 2: calibration to an error probability (Section 4.2)

The calibration event is \(e(x) = 1\) when the small model's output is wrong (the paper uses
exact match of the JSON output across all entity fields) and \(e(x) = 0\) otherwise. UCCI learns
a non-decreasing map \(g : [0, 1] \to [0, 1]\) with

\[
g(u) \approx P\big(e(x) = 1 \mid u(x) = u\big) \qquad \text{(Eq. 5)}
\]

by isotonic regression on a calibration set \(\mathcal{C} = \{(u_i, e_i)\}_{i=1}^n\):

\[
\hat g = \arg\min_{g \ \text{non-decreasing}} \ \sum_{i=1}^{n} w_i \big(e_i - g(u_i)\big)^2 .
\]

The pool-adjacent-violators algorithm (PAV) solves this exactly in \(O(n)\) after sorting
(Barlow, Bartholomew, Bremner and Brunk, 1972). The fit is a non-decreasing step function
on the calibration points; between knots the code interpolates linearly and outside the
calibration range it clips to the end values, as scikit-learn's
`IsotonicRegression(out_of_bounds="clip")` does. The calibrated forecast is
\(\hat p(x) = \hat g(u(x))\).

Code: `ucci.IsotonicCalibrator`, `ucci.pav`, `UCCIRouter.calibrate`. The tests check the
fit against scikit-learn's `IsotonicRegression` knot for knot.

## Step 3: threshold policy and threshold selection (Section 4.3)

The policy keeps the small answer when the calibrated error probability is at most \(\theta\):

\[
\pi_\theta(x) =
\begin{cases}
s & \text{if } \hat p(x) \le \theta, \\
\ell & \text{if } \hat p(x) > \theta.
\end{cases}
\qquad \text{(Eq. 6)}
\]

On a validation set \(\mathcal{V}\) where both models have been run,

\[
\theta^* = \arg\min_\theta \ \widehat{\mathrm{Cost}}(\pi_\theta)
\quad \text{subject to} \quad \widehat{\mathrm{Acc}}(\pi_\theta) \ge \tau, \qquad \text{(Eq. 7)}
\]

where \(\widehat{\mathrm{Cost}}(\pi_\theta) = |\mathcal{V}|^{-1} \sum_{x \in \mathcal{V}} C_{\pi_\theta}(x)\)
uses actual costs and \(\widehat{\mathrm{Acc}}(\pi_\theta) = |\mathcal{V}|^{-1} \sum_{x \in \mathcal{V}} \mathrm{Acc}_{\pi_\theta}(x)\)
uses actual model outputs. The code searches \(\theta \in \{0, 0.005, 0.010, \dots, 1\}\).

If a fraction \(r\) of queries is escalated, the mean cost is

\[
\text{routing (the paper's model):}\ \ c_s (1 - r) + c_\ell \, r,
\qquad
\text{sequential:}\ \ c_s + c_\ell \, r .
\]

Both increase with \(r\), and \(r\) falls as \(\theta\) rises, so the argmin is the largest feasible
\(\theta\): the fewest escalations that still meet \(\tau\). Ties in cost go to the higher
accuracy, then to the larger \(\theta\). The chosen \(\theta\) does not depend on the cost values
(as long as escalation costs more than keeping); only the reported cost does, which is why the
paper's Table 3 can re-cost one routing at several cost ratios.

The **budget form** used for the bottom block of Table 2 swaps objective and constraint:
the most accurate \(\theta\) with \(\widehat{\mathrm{Cost}}(\pi_\theta) \le B\).

For a corpus-level metric such as micro-F1 over entities, \(\widehat{\mathrm{Acc}}\) is the
metric of the routed answers as a whole rather than a mean of per-query scores; pass
`metric=ucci.routed_micro_f1(small_counts, large_counts)`.

Code: `ucci.escalate`, `ucci.select_threshold`, `ucci.select_threshold_for_budget`,
`ucci.policy_cost`, `ucci.pareto_frontier`, `UCCIRouter.choose_threshold`,
`UCCIRouter.choose_threshold_for_budget`, `UCCIRouter.route`.

## Evaluation protocol (Section 6.1)

The paper evaluates every method end to end, never by simulating routing from global
accuracies:

1. fit the calibration map \(g\) on the calibration set;
2. select \(\theta^*\) on the validation set;
3. for each test query, compute \(\hat p(x) = g(u(x))\), apply \(\pi_{\theta^*}\), take the actual
   output of the chosen model and accumulate its actual cost and accuracy.

The splits are disjoint: 30% calibration, 20% validation, 50% test. Costs are normalized to
\(c_s = 1.0\) and \(c_\ell = 3.02\), the measured latency ratio (Section 6.1). Code:
`UCCIRouter.calibrate`, `choose_threshold`, `evaluate`; `ucci.baselines.compare_routers` for
several methods at once; `ucci fit` and `ucci evaluate` on the command line.

## Theorem 1: threshold policies are cost-optimal (Section 5, Appendix A.1)

**Theorem 1** (Optimality of threshold policies). Assume

1. \(c_\ell > c_s\);
2. \(f_\ell\) achieves a fixed accuracy \(\alpha_\ell\) on the test distribution that is invariant to
   which queries are escalated;
3. we have access to the calibrated error probability \(\hat p(x) = P(e(x) = 1 \mid u(x))\).

Then among policies that depend only on \(u(x)\), there exists a cost-optimal threshold policy
\(\pi_\theta\) for some \(\theta \in [0, 1]\) that satisfies the accuracy constraint (3), and every
cost-optimal policy agrees with such a \(\pi_\theta\) up to tie-breaking on level sets of \(\hat p\).

**Proof sketch** (Appendix A.1). Keeping \(x\) gives expected accuracy
\(\mathbb{E}[\mathrm{Acc}(f_s(x), y) \mid u(x)] = 1 - \hat p(x)\) at cost \(c_s\); escalating gives
\(\alpha_\ell\) at cost \(c_\ell\). The marginal accuracy gain of escalating is

\[
\Delta \mathrm{Acc}(x) = \alpha_\ell - 1 + \hat p(x) \qquad \text{(Eq. 8)}
\]

at the constant marginal cost \(\Delta c = c_\ell - c_s > 0\). Meeting the constraint at minimum
cost is then a knapsack problem with equal item costs, solved by escalating in decreasing
order of \(\Delta \mathrm{Acc}(x)\), that is in decreasing order of \(\hat p(x)\), until the
constraint holds. Since \(\hat p = g(u)\) with \(g\) non-decreasing, that rule is a threshold on
\(\hat p\). An isotonic \(\hat g\) is piecewise constant, so \(\hat p\) has atoms; escalating the same
fraction of a level set \(\{x : \hat p(x) = \theta\}\) gives the same expected cost and accuracy
whichever queries are picked, which is the "up to tie-breaking" clause.

Three remarks connect the theorem to the code:

- **Largest feasible \(\theta\).** The greedy allocation escalates as few queries as the
  constraint allows, which is the largest \(\theta\) with \(\widehat{\mathrm{Acc}}(\pi_\theta) \ge \tau\).
- **Grid thresholds.** Hitting \(\tau\) exactly may require escalating part of one level set. A
  deterministic threshold escalates whole level sets, so `select_threshold` returns the
  cheapest feasible threshold on the grid.
- **Checked by brute force.** `tests/test_core_policy.py` enumerates all \(2^n\) routing
  subsets of small random problems under assumptions (ii) and (iii) and checks that none
  meets \(\tau\) more cheaply than the selected threshold (both cost models), and that no subset
  within a budget is more accurate than the threshold the budget form selects.

The same argument holds under the sequential cost model, where the marginal cost of an
escalation is \(c_\ell > 0\).

## Proposition 2: sample complexity of calibration (Section 5, Appendix A.2)

**Proposition 2.** Under standard regularity conditions on the conditional error probability
and bounded \(u(x)\), isotonic regression on \(n\) calibration examples satisfies

\[
\mathbb{E}\big[\mathrm{ECE}(\hat g)\big] = O\big(n^{-1/3}\big).
\]

**Proof sketch** (Appendix A.2). Isotonic regression estimates a monotone conditional
probability \(g^*\) with mean squared error \(\mathbb{E}\|\hat g - g^*\|_2^2 = O(n^{-2/3})\) (Eq. 10;
Barlow et al., 1972). ECE is an \(L_1\) distance, so by Cauchy-Schwarz
\(\mathrm{ECE} \le \sqrt{\mathbb{E}\|\hat g - g^*\|_2^2} = O(n^{-1/3})\) (Eq. 11). A concentration
argument turns the expectation into a high-probability statement.

At this rate, quadrupling the calibration set shrinks the bound by a factor of
\(4^{1/3} \approx 1.6\). `ucci.ece` and `ucci.reliability_table` measure the calibration error on
held-out data.

## Why calibrate if the policy is a threshold anyway

Because \(g\) is non-decreasing, \(\{x : \hat g(u(x)) > \theta\}\) is an upper set of \(u\): a threshold
on \(\hat p\) is a threshold on \(u\). For a fixed target on fixed validation data, calibration
therefore does not add policies; it changes what \(\theta\) means. That meaning is what makes the
threshold usable:

- **\(\theta\) is an error probability.** Escalate when the small answer is wrong with probability
  above \(\theta\). With a price \(\lambda\) (cost units per unit of accuracy), minimizing
  \(\mathbb{E}[C_\pi] - \lambda\, \mathbb{E}[\mathrm{Acc}_\pi]\) escalates exactly when
  \(\hat p(x) > 1 - \alpha_\ell + (c_\ell - c_s)/\lambda\), a direct consequence of Eq. 8. A price for
  accuracy sets \(\theta\) directly, without a search over raw scores; a cost budget fixes the
  escalation rate, and \(\hat p\) says what that rate buys.
- **The accuracy of kept answers can be predicted without labels.** If \(\hat p\) is calibrated,
  the error rate of the answers the small model keeps is
  \(\mathbb{E}[\hat p(x) \mid \hat p(x) \le \theta]\), computable from unlabelled traffic.
- **Thresholds compare across models, prompts and time.** \(\theta = 0.2\) means the same thing
  for any small model; a cut on raw \(u\) does not.
- **Calibration can be monitored.** Fresh labels test whether \(\hat p\) is still calibrated
  (ECE, calibration in the large, Spiegelhalter's z); when it is not, the guarantee behind
  \(\theta\) has lapsed and \(g\) needs a refit (`ucci.online`, an extension addressing the paper's
  Section 7 "Static calibration" limitation). A raw score offers no such check.

The paper's empirical summary makes the same point: most of the cascade engineering value
comes from making the routing score probabilistic, not from hand-tuning the final threshold
(Section 1 and Section 8).

## References

- R. E. Barlow, D. J. Bartholomew, J. M. Bremner and H. D. Brunk. *Statistical Inference Under
  Order Restrictions: The Theory and Application of Isotonic Regression.* John Wiley & Sons,
  1972.
- B. Zadrozny and C. Elkan. Transforming classifier scores into accurate multiclass probability
  estimates. *KDD*, 2002.
- D. J. Spiegelhalter. Probabilistic prediction in patient management and clinical trials.
  *Statistics in Medicine* 5(5):421-433, 1986. doi:[10.1002/sim.4780050506](https://doi.org/10.1002/sim.4780050506).
