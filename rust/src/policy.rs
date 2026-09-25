//! Threshold policy and threshold selection (paper Section 4.3 and Section 5).
//!
//! The policy (Eq. 6) keeps the small model's answer when the calibrated error
//! probability is at most `theta` and escalates otherwise:
//!
//! ```text
//! pi_theta(x) = small   if p_hat(x) <= theta
//!               large   if p_hat(x) >  theta
//! ```
//!
//! Threshold selection (Eq. 7) runs on a validation set `V` where both models
//! have been run:
//!
//! ```text
//! theta* = argmin_theta Cost(pi_theta)   subject to   Acc(pi_theta) >= tau
//! ```
//!
//! where `Cost` and `Acc` average the actual per-query costs and the actual
//! scores of the answers the policy returns (no simulated routing).
//! [`select_threshold`] solves Eq. 7 over a finite grid of thresholds
//! ([`default_grid`]: `theta` in `[0, 1]` at resolution 0.005). With per-query
//! scores the cost falls as `theta` rises, so the argmin is the largest
//! feasible `theta` (fewest escalations).
//!
//! [`select_threshold_for_budget`] solves the budget form used for the bottom
//! block of Table 2 (methods compared at a matched cost budget of 2.00): the
//! highest accuracy with `Cost(pi_theta) <= budget`. [`pareto_frontier`]
//! returns cost and accuracy for every grid threshold (Figure 2).
//!
//! # Cost model
//!
//! A query answered by the small model costs `c_small` and an escalated one
//! costs `c_large`, so a policy that escalates a fraction `r` of queries has
//! mean cost
//!
//! ```text
//! routing:     c_small * (1 - r) + c_large * r      (the paper's model)
//! sequential:  c_small + c_large * r                (small model always runs)
//! ```
//!
//! The paper reports normalized costs with `c_small = 1.0` and
//! `c_large = 3.02`, the measured H100 latency ratio (Section 6.1); see
//! [`Costs::paper`]. For example, escalating 53.45% of queries costs
//! `1 + 2.02 * 0.5345 = 2.08` at `c_large = 3.02` and `1 + 4.0 * 0.5345 = 3.14`
//! at `c_large = 5.00`, the first two rows of Table 3. Every escalation adds the
//! same marginal cost, so the chosen `theta` does not depend on the costs (as
//! long as escalating costs more than keeping); only the reported cost does.
//!
//! # Theorem 1
//!
//! Assume (i) `c_large > c_small`, (ii) the large model's accuracy does not
//! depend on which queries are escalated and (iii) `p_hat` is calibrated.
//! Escalating `x` then gains `alpha_large - 1 + p_hat(x)` expected accuracy
//! (Eq. 8) at a fixed marginal cost, so the cheapest way to reach `tau`
//! escalates queries in decreasing order of `p_hat`: among policies that
//! depend only on `u(x)`, a threshold policy on `p_hat` is cost-optimal
//! (Section 5, Appendix A.1).
//!
//! # Numerics
//!
//! All sweeps use the same operations as the Python package (per-bin sums in
//! input order, cumulative sums, one division by `n`), so thresholds, costs
//! and accuracies agree bit for bit. With 0/1 scores every accuracy is an
//! exact integer count divided by `n`.
//!
//! # Example
//!
//! ```
//! use ucci::policy::{default_grid, select_threshold, Costs};
//!
//! let p_hat = [0.1, 0.2, 0.6, 0.9];
//! let small = [1.0, 1.0, 0.0, 0.0];
//! let large = [1.0, 1.0, 1.0, 1.0];
//! let choice = select_threshold(&p_hat, &small, &large, 1.0, &Costs::paper(), &default_grid())?;
//! assert_eq!(choice.theta, 0.595); // largest theta that still escalates 0.6 and 0.9
//! assert_eq!(choice.accuracy, 1.0);
//! assert_eq!(choice.escalation_rate, 0.5);
//! assert_eq!(choice.cost, 1.0 * 0.5 + 3.02 * 0.5);
//! # Ok::<(), ucci::UcciError>(())
//! ```

use std::collections::HashMap;
use std::fmt;
use std::str::FromStr;

use crate::error::{Result, UcciError};
use crate::num::{check_finite, check_finite_scalar, check_same_len, pairwise_sum};

/// Normalized small-model cost (Section 6.1).
pub const DEFAULT_COST_SMALL: f64 = 1.0;

/// Normalized large-model cost: the measured latency ratio
/// 142.3 ms / 47.2 ms = 3.02 (Section 6.1, Appendix B.3).
pub const DEFAULT_COST_LARGE: f64 = 3.02;

/// Resolution of the default threshold grid.
pub const DEFAULT_GRID_STEP: f64 = 0.005;

/// Number of thresholds in [`default_grid`].
pub const DEFAULT_GRID_LEN: usize = 201;

/// How the mean per-query cost of a routing policy is computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "json", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "json", serde(rename_all = "lowercase"))]
pub enum CostModel {
    /// The paper's model (Section 3): a kept query costs `c_small`, an
    /// escalated one costs `c_large`.
    #[default]
    Routing,
    /// The small model always runs first (it produces `u(x)`), so an
    /// escalated query costs `c_small + c_large`.
    Sequential,
}

impl CostModel {
    /// The name used in router files and by the Python package
    /// (`"routing"` or `"sequential"`).
    pub fn as_str(self) -> &'static str {
        match self {
            CostModel::Routing => "routing",
            CostModel::Sequential => "sequential",
        }
    }
}

impl fmt::Display for CostModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for CostModel {
    type Err = UcciError;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "routing" => Ok(CostModel::Routing),
            "sequential" => Ok(CostModel::Sequential),
            other => Err(UcciError::InvalidParameter {
                name: "cost_model",
                reason: format!("must be 'routing' or 'sequential', got {other:?}"),
            }),
        }
    }
}

/// Per-query costs of the two models and the [`CostModel`] that combines them.
///
/// Costs are in any consistent unit (latency, dollars, energy); the paper uses
/// latency normalized so that `c_small = 1` (Section 6.1).
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "json", derive(serde::Serialize, serde::Deserialize))]
pub struct Costs {
    /// Cost of answering a query with the small model.
    pub c_small: f64,
    /// Cost of answering a query with the large model.
    pub c_large: f64,
    /// How the two combine for an escalated query.
    pub model: CostModel,
}

impl Costs {
    /// Validated costs.
    ///
    /// # Errors
    ///
    /// [`UcciError::InvalidParameter`] unless both costs are finite and
    /// strictly positive.
    pub fn new(c_small: f64, c_large: f64, model: CostModel) -> Result<Self> {
        let costs = Costs {
            c_small,
            c_large,
            model,
        };
        costs.validate()?;
        Ok(costs)
    }

    /// The paper's normalized costs: `c_small = 1.0`, `c_large = 3.02`, routing
    /// cost model (Section 6.1).
    pub fn paper() -> Self {
        Costs {
            c_small: DEFAULT_COST_SMALL,
            c_large: DEFAULT_COST_LARGE,
            model: CostModel::Routing,
        }
    }

    /// Checks that both costs are finite and strictly positive.
    ///
    /// # Errors
    ///
    /// [`UcciError::InvalidParameter`] naming the offending cost.
    pub fn validate(&self) -> Result<()> {
        check_finite_scalar(self.c_small, "c_small")?;
        check_finite_scalar(self.c_large, "c_large")?;
        if self.c_small <= 0.0 {
            return Err(UcciError::InvalidParameter {
                name: "c_small",
                reason: format!("must be positive, got {}", self.c_small),
            });
        }
        if self.c_large <= 0.0 {
            return Err(UcciError::InvalidParameter {
                name: "c_large",
                reason: format!("must be positive, got {}", self.c_large),
            });
        }
        Ok(())
    }

    /// Validation plus Theorem 1 assumption (i): under the routing model,
    /// escalating must cost more than keeping (`c_large > c_small`).
    fn validate_ordered(&self) -> Result<()> {
        self.validate()?;
        if self.model == CostModel::Routing && self.c_large <= self.c_small {
            return Err(UcciError::InvalidParameter {
                name: "c_large",
                reason: format!(
                    "Theorem 1 assumption (i) needs c_large > c_small under the routing cost \
                     model; got c_small={}, c_large={}",
                    self.c_small, self.c_large
                ),
            });
        }
        Ok(())
    }

    /// Mean per-query cost when a fraction `rate` of queries is escalated:
    /// `c_small * (1 - rate) + c_large * rate` (routing) or
    /// `c_small + c_large * rate` (sequential).
    ///
    /// Every function in this crate uses this exact expression, so equal
    /// escalation rates give bit-identical costs.
    pub fn cost_at_rate(&self, rate: f64) -> f64 {
        match self.model {
            CostModel::Routing => self.c_small * (1.0 - rate) + self.c_large * rate,
            CostModel::Sequential => self.c_small + self.c_large * rate,
        }
    }
}

impl Default for Costs {
    /// [`Costs::paper`].
    fn default() -> Self {
        Costs::paper()
    }
}

/// A threshold with the cost and accuracy it achieves on some split.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "json", derive(serde::Serialize, serde::Deserialize))]
pub struct ThresholdChoice {
    /// The threshold of the policy `pi_theta` (Eq. 6).
    pub theta: f64,
    /// Mean per-query cost under the chosen cost model.
    pub cost: f64,
    /// Mean per-query score of the returned answers, or the value of the
    /// custom metric.
    pub accuracy: f64,
    /// Fraction of queries sent to the large model.
    pub escalation_rate: f64,
}

/// Cost and accuracy of `pi_theta` for every threshold of a grid (Figure 2).
///
/// All vectors have one entry per grid value, in increasing `theta`.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "json", derive(serde::Serialize, serde::Deserialize))]
pub struct ParetoFrontier {
    /// Grid thresholds, strictly increasing.
    pub theta: Vec<f64>,
    /// Mean per-query cost at each threshold.
    pub cost: Vec<f64>,
    /// Accuracy at each threshold.
    pub accuracy: Vec<f64>,
    /// Escalation rate at each threshold.
    pub escalation_rate: Vec<f64>,
    /// True where no other grid point is at most as costly and at least as
    /// accurate with one of the two strictly better. Plot the efficient
    /// points' cost against accuracy for the frontier.
    pub efficient: Vec<bool>,
}

impl ParetoFrontier {
    /// Number of grid thresholds.
    pub fn len(&self) -> usize {
        self.theta.len()
    }

    /// True when the frontier has no points (never the case for a frontier
    /// returned by [`pareto_frontier`]).
    pub fn is_empty(&self) -> bool {
        self.theta.is_empty()
    }

    /// Every grid point as a [`ThresholdChoice`], in increasing `theta`.
    pub fn points(&self) -> impl Iterator<Item = ThresholdChoice> + '_ {
        (0..self.len()).map(move |j| self.point(j))
    }

    /// The efficient (non-dominated) points, in increasing `theta`.
    pub fn efficient_points(&self) -> impl Iterator<Item = ThresholdChoice> + '_ {
        (0..self.len())
            .filter(move |&j| self.efficient[j])
            .map(move |j| self.point(j))
    }

    fn point(&self, j: usize) -> ThresholdChoice {
        ThresholdChoice {
            theta: self.theta[j],
            cost: self.cost[j],
            accuracy: self.accuracy[j],
            escalation_rate: self.escalation_rate[j],
        }
    }
}

/// `numpy.linspace(0, 1, n + 1)`: `i * (1 / n)` with the last value exactly 1.
pub(crate) fn linspace01(n: usize) -> Vec<f64> {
    let step = 1.0 / n as f64;
    let mut v: Vec<f64> = (0..=n).map(|i| i as f64 * step + 0.0).collect();
    if n > 0 {
        v[n] = 1.0;
    }
    v
}

/// Round half to even (C `rint` in the default rounding mode).
fn rint(x: f64) -> f64 {
    let r = x.round();
    if (r - x).abs() == 0.5 {
        // Halfway case: round() went away from zero; step back to the even one.
        2.0 * (x / 2.0).round()
    } else {
        r
    }
}

/// `numpy.round(v, decimals)` for small non-negative `decimals`.
fn np_round(v: f64, decimals: i32) -> f64 {
    let f = 10f64.powi(decimals);
    rint(v * f) / f
}

/// The paper's threshold grid: `theta` in `[0, 1]` at resolution 0.005, 201
/// values (Section 4.3). Identical, bit for bit, to the Python package's
/// `DEFAULT_GRID = numpy.round(numpy.linspace(0, 1, 201), 3)`.
///
/// # Example
///
/// ```
/// let g = ucci::policy::default_grid();
/// assert_eq!(g.len(), 201);
/// assert_eq!((g[0], g[1], g[200]), (0.0, 0.005, 1.0));
/// assert_eq!(g[57], 0.285);
/// ```
pub fn default_grid() -> Vec<f64> {
    linspace01(DEFAULT_GRID_LEN - 1)
        .into_iter()
        .map(|v| np_round(v, 3))
        .collect()
}

/// Evenly spaced thresholds on `[0, 1]`, both ends included (the Python
/// package's `make_grid`). `make_grid(0.005)` equals [`default_grid`].
///
/// # Errors
///
/// [`UcciError::InvalidParameter`] unless `step` lies in `(0, 1]` and `1 / step`
/// is a whole number (0.005, 0.01, 0.1, ...).
///
/// # Example
///
/// ```
/// assert_eq!(ucci::policy::make_grid(0.25)?, vec![0.0, 0.25, 0.5, 0.75, 1.0]);
/// assert_eq!(ucci::policy::make_grid(0.005)?, ucci::policy::default_grid());
/// assert!(ucci::policy::make_grid(0.3).is_err());
/// # Ok::<(), ucci::UcciError>(())
/// ```
pub fn make_grid(step: f64) -> Result<Vec<f64>> {
    check_finite_scalar(step, "step")?;
    if !(step > 0.0 && step <= 1.0) {
        return Err(UcciError::InvalidParameter {
            name: "step",
            reason: format!("must lie in (0, 1], got {step}"),
        });
    }
    // Python's round(): half to even.
    let n = rint(1.0 / step);
    if (n * step - 1.0).abs() > 1e-9 {
        return Err(UcciError::InvalidParameter {
            name: "step",
            reason: format!("must divide 1 exactly (e.g. 0.005 or 0.01), got {step}"),
        });
    }
    Ok(linspace01(n as usize)
        .into_iter()
        .map(|v| np_round(v, 12))
        .collect())
}

/// Validates a grid and returns it sorted with duplicates removed.
fn check_grid(grid: &[f64]) -> Result<Vec<f64>> {
    if grid.is_empty() {
        return Err(UcciError::Empty { what: "grid" });
    }
    check_finite(grid, "grid")?;
    if let Some(index) = grid.iter().position(|g| !(0.0..=1.0).contains(g)) {
        return Err(UcciError::OutOfRange {
            what: "grid",
            index,
            value: grid[index],
            expected: "in [0, 1]",
        });
    }
    let mut g = grid.to_vec();
    g.sort_by(f64::total_cmp);
    g.dedup_by(|a, b| a == b);
    Ok(g)
}

fn check_scores(p_len: usize, small: &[f64], large: &[f64]) -> Result<()> {
    if small.is_empty() {
        return Err(UcciError::Empty {
            what: "small_score",
        });
    }
    check_finite(small, "small_score")?;
    if large.is_empty() {
        return Err(UcciError::Empty {
            what: "large_score",
        });
    }
    check_finite(large, "large_score")?;
    check_same_len(p_len, small.len(), "p_hat and small_score")?;
    check_same_len(p_len, large.len(), "p_hat and large_score")
}

fn check_p_hat(p_hat: &[f64]) -> Result<()> {
    if p_hat.is_empty() {
        return Err(UcciError::Empty { what: "p_hat" });
    }
    check_finite(p_hat, "p_hat")
}

/// The routing decision of `pi_theta` (Eq. 6): `true` means escalate.
///
/// A query is escalated when `p_hat > theta` (strictly), so a query with
/// `p_hat == theta` stays with the small model.
///
/// # Errors
///
/// [`UcciError::NonFinite`] or [`UcciError::InvalidParameter`] if `p_hat` or
/// `theta` is NaN or infinite.
///
/// # Example
///
/// ```
/// assert!(!ucci::escalate(0.3, 0.3)?);
/// assert!(ucci::escalate(0.30001, 0.3)?);
/// # Ok::<(), ucci::UcciError>(())
/// ```
pub fn escalate(p_hat: f64, theta: f64) -> Result<bool> {
    check_finite_scalar(theta, "theta")?;
    if !p_hat.is_finite() {
        return Err(UcciError::NonFinite {
            what: "p_hat",
            index: 0,
        });
    }
    Ok(p_hat > theta)
}

/// [`escalate`] for every element of `p_hat`.
///
/// # Errors
///
/// As [`escalate`]; the error names the first non-finite element.
pub fn escalate_many(p_hat: &[f64], theta: f64) -> Result<Vec<bool>> {
    check_finite_scalar(theta, "theta")?;
    check_finite(p_hat, "p_hat")?;
    Ok(p_hat.iter().map(|&p| p > theta).collect())
}

fn escalation_rate(esc: &[bool]) -> f64 {
    esc.iter().filter(|&&e| e).count() as f64 / esc.len() as f64
}

/// Mean per-query cost of a routing mask (Section 3, Table 3):
/// [`Costs::cost_at_rate`] at the mask's escalation rate.
///
/// # Errors
///
/// [`UcciError::Empty`] for an empty mask and [`UcciError::InvalidParameter`]
/// for non-positive or non-finite costs.
///
/// # Example
///
/// ```
/// use ucci::policy::{policy_cost, CostModel, Costs};
///
/// let esc = [true, false, false, true];
/// assert_eq!(policy_cost(&esc, &Costs::new(1.0, 3.0, CostModel::Routing)?)?, 2.0);
/// assert_eq!(policy_cost(&esc, &Costs::new(1.0, 3.0, CostModel::Sequential)?)?, 2.5);
/// # Ok::<(), ucci::UcciError>(())
/// ```
pub fn policy_cost(esc: &[bool], costs: &Costs) -> Result<f64> {
    if esc.is_empty() {
        return Err(UcciError::Empty { what: "esc" });
    }
    costs.validate()?;
    Ok(costs.cost_at_rate(escalation_rate(esc)))
}

/// Mean score of the answers a routing mask returns (`Acc` in Eq. 7):
/// `(sum of small_score over kept + sum of large_score over escalated) / n`.
///
/// Scores are per-query scores of each model's actual output: 0/1 correctness
/// or any per-query metric such as per-query F1. For a corpus-level metric
/// (micro-F1 over entities), use [`select_threshold_with_metric`].
///
/// # Errors
///
/// [`UcciError::Empty`], [`UcciError::LengthMismatch`] or
/// [`UcciError::NonFinite`] for empty, misaligned or non-finite inputs.
pub fn policy_accuracy(esc: &[bool], small_score: &[f64], large_score: &[f64]) -> Result<f64> {
    if esc.is_empty() {
        return Err(UcciError::Empty { what: "esc" });
    }
    check_scores(esc.len(), small_score, large_score)?;
    let kept: Vec<f64> = esc
        .iter()
        .zip(small_score)
        .filter(|(&e, _)| !e)
        .map(|(_, &s)| s)
        .collect();
    let escalated: Vec<f64> = esc
        .iter()
        .zip(large_score)
        .filter(|(&e, _)| e)
        .map(|(_, &s)| s)
        .collect();
    Ok((pairwise_sum(&kept) + pairwise_sum(&escalated)) / esc.len() as f64)
}

/// Escalation counts and accuracies of `pi_theta` for every grid threshold.
struct Sweep {
    theta: Vec<f64>,
    rate: Vec<f64>,
    accuracy: Vec<f64>,
}

impl Sweep {
    /// `bins[i]` = number of grid values strictly below `p_hat[i]`, so that
    /// `p_hat[i] > grid[j]` exactly when `j < bins[i]`.
    fn bins(p_hat: &[f64], grid: &[f64]) -> Vec<usize> {
        p_hat
            .iter()
            .map(|&p| grid.partition_point(|&g| g < p))
            .collect()
    }

    /// Escalation counts and rates per grid threshold.
    fn counts(bins: &[usize], g_len: usize) -> (Vec<usize>, Vec<f64>) {
        let mut counts = vec![0usize; g_len + 1];
        for &b in bins {
            counts[b] += 1;
        }
        // k[j] = #{i : bins[i] > j}
        let mut k = vec![0usize; g_len];
        let mut acc = 0usize;
        for j in (0..g_len).rev() {
            acc += counts[j + 1];
            k[j] = acc;
        }
        let n = bins.len() as f64;
        let rate = k.iter().map(|&kj| kj as f64 / n).collect();
        (k, rate)
    }

    fn with_scores(p_hat: &[f64], small: &[f64], large: &[f64], grid: &[f64]) -> Result<Self> {
        check_p_hat(p_hat)?;
        let theta = check_grid(grid)?;
        check_scores(p_hat.len(), small, large)?;
        let g_len = theta.len();
        let bins = Self::bins(p_hat, &theta);
        let (_, rate) = Self::counts(&bins, g_len);
        // Per-bin score sums in input order (numpy.bincount with weights).
        let mut s_bin = vec![0.0f64; g_len + 1];
        let mut l_bin = vec![0.0f64; g_len + 1];
        for (i, &b) in bins.iter().enumerate() {
            s_bin[b] += small[i];
            l_bin[b] += large[i];
        }
        // kept_small[j] = cumsum(s_bin)[j]: bins 0..=j are kept.
        // esc_large[j] = reverse cumsum of l_bin from the top down to j + 1.
        let mut esc_large = vec![0.0f64; g_len];
        let mut acc = l_bin[g_len];
        for j in (0..g_len).rev() {
            esc_large[j] = acc;
            acc += l_bin[j];
        }
        let n = p_hat.len() as f64;
        let mut accuracy = Vec::with_capacity(g_len);
        let mut kept_small = 0.0f64;
        for (j, &s) in s_bin.iter().take(g_len).enumerate() {
            kept_small = if j == 0 { s } else { kept_small + s };
            accuracy.push((kept_small + esc_large[j]) / n);
        }
        Ok(Sweep {
            theta,
            rate,
            accuracy,
        })
    }

    fn with_metric<F>(p_hat: &[f64], grid: &[f64], mut metric: F) -> Result<Self>
    where
        F: FnMut(&[bool]) -> f64,
    {
        check_p_hat(p_hat)?;
        let theta = check_grid(grid)?;
        let bins = Self::bins(p_hat, &theta);
        let (k, rate) = Self::counts(&bins, theta.len());
        // The metric is called once per distinct escalation mask, in grid order.
        let mut cache: HashMap<usize, f64> = HashMap::new();
        let mut accuracy = Vec::with_capacity(theta.len());
        for (j, &t) in theta.iter().enumerate() {
            let value = match cache.get(&k[j]) {
                Some(&v) => v,
                None => {
                    let mask: Vec<bool> = p_hat.iter().map(|&p| p > t).collect();
                    let v = metric_value(&mut metric, &mask, t)?;
                    cache.insert(k[j], v);
                    v
                }
            };
            accuracy.push(value);
        }
        Ok(Sweep {
            theta,
            rate,
            accuracy,
        })
    }

    fn costs(&self, costs: &Costs) -> Vec<f64> {
        self.rate.iter().map(|&r| costs.cost_at_rate(r)).collect()
    }

    fn choice(&self, j: usize, cost: &[f64]) -> ThresholdChoice {
        ThresholdChoice {
            theta: self.theta[j],
            cost: cost[j],
            accuracy: self.accuracy[j],
            escalation_rate: self.rate[j],
        }
    }

    /// Eq. 7: lowest cost with accuracy >= tau; ties to the higher accuracy,
    /// then to the larger theta.
    fn select(&self, tau: f64, costs: &Costs) -> Result<ThresholdChoice> {
        let cost = self.costs(costs);
        let feasible: Vec<usize> = (0..self.theta.len())
            .filter(|&j| self.accuracy[j] >= tau)
            .collect();
        if feasible.is_empty() {
            let j = argmax_first(&self.accuracy);
            return Err(UcciError::Infeasible {
                tau,
                best_accuracy: self.accuracy[j],
                best_theta: self.theta[j],
            });
        }
        let min_cost = feasible
            .iter()
            .map(|&j| cost[j])
            .fold(f64::INFINITY, f64::min);
        let cheapest: Vec<usize> = feasible
            .into_iter()
            .filter(|&j| cost[j] == min_cost)
            .collect();
        let best_acc = cheapest
            .iter()
            .map(|&j| self.accuracy[j])
            .fold(f64::NEG_INFINITY, f64::max);
        let j = *cheapest
            .iter()
            .rev()
            .find(|&&j| self.accuracy[j] == best_acc)
            .expect("non-empty");
        Ok(self.choice(j, &cost))
    }

    /// Budget form: highest accuracy with cost <= budget; ties to the lower
    /// cost, then to the larger theta.
    fn select_for_budget(&self, budget: f64, costs: &Costs) -> Result<ThresholdChoice> {
        let cost = self.costs(costs);
        let feasible: Vec<usize> = (0..self.theta.len())
            .filter(|&j| cost[j] <= budget)
            .collect();
        if feasible.is_empty() {
            let j = argmin_first(&cost);
            return Err(UcciError::OverBudget {
                budget,
                min_cost: cost[j],
                cheapest_theta: self.theta[j],
            });
        }
        let best_acc = feasible
            .iter()
            .map(|&j| self.accuracy[j])
            .fold(f64::NEG_INFINITY, f64::max);
        let most_accurate: Vec<usize> = feasible
            .into_iter()
            .filter(|&j| self.accuracy[j] == best_acc)
            .collect();
        let min_cost = most_accurate
            .iter()
            .map(|&j| cost[j])
            .fold(f64::INFINITY, f64::min);
        let j = *most_accurate
            .iter()
            .rev()
            .find(|&&j| cost[j] == min_cost)
            .expect("non-empty");
        Ok(self.choice(j, &cost))
    }

    fn frontier(self, costs: &Costs) -> ParetoFrontier {
        let cost = self.costs(costs);
        let acc = &self.accuracy;
        // Group grid points by exact cost value, cheapest group first.
        let mut unique_cost = cost.clone();
        unique_cost.sort_by(f64::total_cmp);
        unique_cost.dedup_by(|a, b| a == b);
        let group: Vec<usize> = cost
            .iter()
            .map(|c| unique_cost.partition_point(|u| u < c))
            .collect();
        let mut best_in_group = vec![f64::NEG_INFINITY; unique_cost.len()];
        for (j, &g) in group.iter().enumerate() {
            best_in_group[g] = best_in_group[g].max(acc[j]);
        }
        // best_cheaper[g] = best accuracy over strictly cheaper groups.
        let mut best_cheaper = vec![f64::NEG_INFINITY; unique_cost.len()];
        let mut running = f64::NEG_INFINITY;
        for g in 0..unique_cost.len() {
            best_cheaper[g] = running;
            running = running.max(best_in_group[g]);
        }
        let efficient = (0..cost.len())
            .map(|j| acc[j] == best_in_group[group[j]] && acc[j] > best_cheaper[group[j]])
            .collect();
        ParetoFrontier {
            theta: self.theta,
            cost,
            accuracy: self.accuracy,
            escalation_rate: self.rate,
            efficient,
        }
    }
}

fn argmax_first(v: &[f64]) -> usize {
    let mut best = 0;
    for (j, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = j;
        }
    }
    best
}

fn argmin_first(v: &[f64]) -> usize {
    let mut best = 0;
    for (j, &x) in v.iter().enumerate() {
        if x < v[best] {
            best = j;
        }
    }
    best
}

fn metric_value<F: FnMut(&[bool]) -> f64>(
    metric: &mut F,
    mask: &[bool],
    theta: f64,
) -> Result<f64> {
    let v = metric(mask);
    if v.is_finite() {
        Ok(v)
    } else {
        Err(UcciError::InvalidParameter {
            name: "metric",
            reason: format!("returned {v} at theta={theta}"),
        })
    }
}

/// Cheapest grid threshold whose validation accuracy is at least `tau`
/// (Section 4.3, Eq. 7).
///
/// `p_hat` holds the calibrated error probabilities of the validation queries
/// and `small_score`, `large_score` the per-query scores of each model's
/// actual output on the same queries (both models run on the validation
/// split). `grid` is sorted and de-duplicated internally; pass
/// [`default_grid`] for the paper's grid.
///
/// Returns the threshold with the lowest cost among those with accuracy
/// `>= tau`. Ties in cost go to the higher accuracy, then to the larger
/// `theta`. Comparisons are exact.
///
/// # Errors
///
/// * [`UcciError::Infeasible`] if no threshold on the grid reaches `tau`; it
///   carries the best accuracy on the grid.
/// * [`UcciError::InvalidParameter`] for a non-finite `tau`, invalid costs, or
///   `c_large <= c_small` under the routing cost model (Theorem 1,
///   assumption (i)).
/// * [`UcciError::Empty`], [`UcciError::NonFinite`],
///   [`UcciError::LengthMismatch`] or [`UcciError::OutOfRange`] for invalid
///   inputs or grid values outside `[0, 1]`.
pub fn select_threshold(
    p_hat: &[f64],
    small_score: &[f64],
    large_score: &[f64],
    tau: f64,
    costs: &Costs,
    grid: &[f64],
) -> Result<ThresholdChoice> {
    check_finite_scalar(tau, "tau")?;
    costs.validate_ordered()?;
    Sweep::with_scores(p_hat, small_score, large_score, grid)?.select(tau, costs)
}

/// [`select_threshold`] with a custom accuracy functional.
///
/// `metric(esc_mask)` replaces the mean per-query score, for a corpus-level
/// metric such as micro-F1 of the routed answers. It is called once per
/// distinct escalation mask on the grid, in increasing `theta`, and must
/// return a finite number.
///
/// # Errors
///
/// As [`select_threshold`], plus [`UcciError::InvalidParameter`] if the metric
/// returns NaN or an infinity.
///
/// # Example
///
/// ```
/// use ucci::policy::{default_grid, select_threshold_with_metric, Costs};
///
/// // Accuracy = fraction of the two hardest queries that are escalated.
/// let p_hat = [0.1, 0.2, 0.6, 0.9];
/// let metric = |esc: &[bool]| (esc[2] as u8 + esc[3] as u8) as f64 / 2.0;
/// let c = select_threshold_with_metric(&p_hat, 1.0, &Costs::paper(), &default_grid(), metric)?;
/// assert_eq!(c.theta, 0.595);
/// # Ok::<(), ucci::UcciError>(())
/// ```
pub fn select_threshold_with_metric<F>(
    p_hat: &[f64],
    tau: f64,
    costs: &Costs,
    grid: &[f64],
    metric: F,
) -> Result<ThresholdChoice>
where
    F: FnMut(&[bool]) -> f64,
{
    check_finite_scalar(tau, "tau")?;
    costs.validate_ordered()?;
    Sweep::with_metric(p_hat, grid, metric)?.select(tau, costs)
}

/// Most accurate grid threshold whose validation cost is within `budget`:
/// the budget form of Eq. 7, `argmax Acc(pi_theta)` subject to
/// `Cost(pi_theta) <= budget`, as used for the matched-cost comparison in the
/// bottom block of Table 2 (budget 2.00).
///
/// Ties in accuracy go to the lower cost, then to the larger `theta`.
///
/// # Errors
///
/// [`UcciError::OverBudget`] if every threshold costs more than `budget` (it
/// carries the cheapest cost on the grid), plus the input errors of
/// [`select_threshold`].
pub fn select_threshold_for_budget(
    p_hat: &[f64],
    small_score: &[f64],
    large_score: &[f64],
    budget: f64,
    costs: &Costs,
    grid: &[f64],
) -> Result<ThresholdChoice> {
    check_finite_scalar(budget, "budget")?;
    costs.validate_ordered()?;
    Sweep::with_scores(p_hat, small_score, large_score, grid)?.select_for_budget(budget, costs)
}

/// [`select_threshold_for_budget`] with a custom accuracy functional (see
/// [`select_threshold_with_metric`]).
///
/// # Errors
///
/// As [`select_threshold_for_budget`], plus [`UcciError::InvalidParameter`] if
/// the metric returns NaN or an infinity.
pub fn select_threshold_for_budget_with_metric<F>(
    p_hat: &[f64],
    budget: f64,
    costs: &Costs,
    grid: &[f64],
    metric: F,
) -> Result<ThresholdChoice>
where
    F: FnMut(&[bool]) -> f64,
{
    check_finite_scalar(budget, "budget")?;
    costs.validate_ordered()?;
    Sweep::with_metric(p_hat, grid, metric)?.select_for_budget(budget, costs)
}

/// Cost and accuracy of `pi_theta` at every grid threshold, with the mask of
/// efficient (non-dominated) points (Figure 2).
///
/// Any positive costs are accepted (the sweep is descriptive).
///
/// # Errors
///
/// The input errors of [`select_threshold`].
pub fn pareto_frontier(
    p_hat: &[f64],
    small_score: &[f64],
    large_score: &[f64],
    costs: &Costs,
    grid: &[f64],
) -> Result<ParetoFrontier> {
    costs.validate()?;
    Ok(Sweep::with_scores(p_hat, small_score, large_score, grid)?.frontier(costs))
}

/// [`pareto_frontier`] with a custom accuracy functional (see
/// [`select_threshold_with_metric`]).
///
/// # Errors
///
/// As [`pareto_frontier`], plus [`UcciError::InvalidParameter`] if the metric
/// returns NaN or an infinity.
pub fn pareto_frontier_with_metric<F>(
    p_hat: &[f64],
    costs: &Costs,
    grid: &[f64],
    metric: F,
) -> Result<ParetoFrontier>
where
    F: FnMut(&[bool]) -> f64,
{
    costs.validate()?;
    Ok(Sweep::with_metric(p_hat, grid, metric)?.frontier(costs))
}

/// Routes every query with `pi_theta` and reports the actual cost and
/// accuracy: step 3 of the evaluation protocol (Section 6.1), applied to the
/// test split with the threshold chosen on validation.
///
/// Any positive costs are accepted.
///
/// # Errors
///
/// Invalid costs, a non-finite `theta`, or empty, misaligned or non-finite
/// inputs.
pub fn evaluate(
    p_hat: &[f64],
    small_score: &[f64],
    large_score: &[f64],
    theta: f64,
    costs: &Costs,
) -> Result<ThresholdChoice> {
    costs.validate()?;
    check_p_hat(p_hat)?;
    let mask = escalate_many(p_hat, theta)?;
    let accuracy = policy_accuracy(&mask, small_score, large_score)?;
    let rate = escalation_rate(&mask);
    Ok(ThresholdChoice {
        theta,
        cost: costs.cost_at_rate(rate),
        accuracy,
        escalation_rate: rate,
    })
}

/// [`evaluate`] with a custom accuracy functional (see
/// [`select_threshold_with_metric`]).
///
/// # Errors
///
/// As [`evaluate`], plus [`UcciError::InvalidParameter`] if the metric returns
/// NaN or an infinity.
pub fn evaluate_with_metric<F>(
    p_hat: &[f64],
    theta: f64,
    costs: &Costs,
    mut metric: F,
) -> Result<ThresholdChoice>
where
    F: FnMut(&[bool]) -> f64,
{
    costs.validate()?;
    check_p_hat(p_hat)?;
    let mask = escalate_many(p_hat, theta)?;
    let accuracy = metric_value(&mut metric, &mask, theta)?;
    let rate = escalation_rate(&mask);
    Ok(ThresholdChoice {
        theta,
        cost: costs.cost_at_rate(rate),
        accuracy,
        escalation_rate: rate,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn routing(c_small: f64, c_large: f64) -> Costs {
        Costs::new(c_small, c_large, CostModel::Routing).unwrap()
    }

    #[test]
    fn grid_matches_numpy() {
        let g = default_grid();
        assert_eq!(g.len(), 201);
        for (i, &v) in g.iter().enumerate() {
            // numpy.round(numpy.linspace(0, 1, 201), 3)[i] is the double
            // nearest to i / 200.
            assert_eq!(v, i as f64 / 200.0, "i = {i}");
        }
        assert_eq!(make_grid(DEFAULT_GRID_STEP).unwrap(), g);
        assert_eq!(make_grid(1.0).unwrap(), vec![0.0, 1.0]);
        assert_eq!(make_grid(0.1).unwrap()[3], 0.3);
        for bad in [0.0, -0.1, 1.5, 0.3, f64::NAN] {
            assert!(make_grid(bad).is_err(), "step {bad}");
        }
    }

    #[test]
    fn rint_is_half_to_even() {
        assert_eq!(rint(0.5), 0.0);
        assert_eq!(rint(1.5), 2.0);
        assert_eq!(rint(2.5), 2.0);
        assert_eq!(rint(-0.5), -0.0);
        assert_eq!(rint(-1.5), -2.0);
        assert_eq!(rint(2.4), 2.0);
        assert_eq!(rint(2.6), 3.0);
    }

    #[test]
    fn escalate_is_strict() {
        assert!(!escalate(0.3, 0.3).unwrap());
        assert!(escalate(0.31, 0.3).unwrap());
        assert!(escalate(f64::NAN, 0.3).is_err());
        assert!(escalate(0.3, f64::INFINITY).is_err());
        assert_eq!(
            escalate_many(&[0.1, 0.5, 0.9], 0.5).unwrap(),
            vec![false, false, true]
        );
        assert!(escalate_many(&[0.1, f64::NAN], 0.5).is_err());
    }

    #[test]
    fn cost_models() {
        let esc = [true, false, false, true];
        assert_eq!(policy_cost(&esc, &routing(1.0, 3.0)).unwrap(), 2.0);
        let seq = Costs::new(1.0, 3.0, CostModel::Sequential).unwrap();
        assert_eq!(policy_cost(&esc, &seq).unwrap(), 2.5);
        assert!(policy_cost(&[], &seq).is_err());
        // Table 3 arithmetic: the same routing at two cost ratios, which the
        // paper reports rounded to 2.08 and 3.14.
        let paper = Costs::paper();
        assert!((paper.cost_at_rate(0.5345) - (1.0 + 2.02 * 0.5345)).abs() < 1e-12);
        assert!((routing(1.0, 5.0).cost_at_rate(0.5345) - (1.0 + 4.0 * 0.5345)).abs() < 1e-12);
        assert_eq!((paper.cost_at_rate(0.5345) * 100.0).round(), 208.0);
        assert_eq!(
            (routing(1.0, 5.0).cost_at_rate(0.5345) * 100.0).round(),
            314.0
        );
        assert_eq!(Costs::default(), paper);
    }

    #[test]
    fn costs_validation() {
        assert!(Costs::new(0.0, 3.0, CostModel::Routing).is_err());
        assert!(Costs::new(1.0, -3.0, CostModel::Routing).is_err());
        assert!(Costs::new(f64::NAN, 3.0, CostModel::Routing).is_err());
        assert!(Costs::new(1.0, f64::INFINITY, CostModel::Routing).is_err());
        // c_large <= c_small is allowed for describing, not for selecting.
        let flat = Costs::new(2.0, 1.0, CostModel::Routing).unwrap();
        let err = select_threshold(&[0.5], &[1.0], &[1.0], 0.5, &flat, &default_grid());
        assert!(matches!(
            err,
            Err(UcciError::InvalidParameter {
                name: "c_large",
                ..
            })
        ));
        let seq = Costs::new(2.0, 1.0, CostModel::Sequential).unwrap();
        assert!(select_threshold(&[0.5], &[1.0], &[1.0], 0.5, &seq, &default_grid()).is_ok());
        assert!(pareto_frontier(&[0.5], &[1.0], &[1.0], &flat, &default_grid()).is_ok());
    }

    #[test]
    fn cost_model_names() {
        assert_eq!("routing".parse::<CostModel>().unwrap(), CostModel::Routing);
        assert_eq!(
            "sequential".parse::<CostModel>().unwrap(),
            CostModel::Sequential
        );
        assert!("Routing".parse::<CostModel>().is_err());
        assert_eq!(CostModel::Sequential.to_string(), "sequential");
        assert_eq!(CostModel::default(), CostModel::Routing);
    }

    #[test]
    fn accuracy_of_mask() {
        let esc = [true, false, true];
        let acc = policy_accuracy(&esc, &[0.0, 1.0, 0.0], &[1.0, 0.0, 0.5]).unwrap();
        assert_eq!(acc, (1.0 + 1.0 + 0.5) / 3.0);
        assert!(policy_accuracy(&esc, &[0.0, 1.0], &[1.0, 0.0, 0.5]).is_err());
        assert!(policy_accuracy(&esc, &[0.0, 1.0, f64::NAN], &[1.0, 0.0, 0.5]).is_err());
    }

    #[test]
    fn python_docstring_example() {
        let c = select_threshold(
            &[0.1, 0.2, 0.6, 0.9],
            &[1.0, 1.0, 0.0, 0.0],
            &[1.0; 4],
            1.0,
            &Costs::paper(),
            &default_grid(),
        )
        .unwrap();
        assert_eq!((c.theta, c.accuracy, c.escalation_rate), (0.595, 1.0, 0.5));
    }

    #[test]
    fn tau_zero_keeps_everything_and_picks_largest_theta() {
        let c = select_threshold(
            &[0.1, 0.9],
            &[0.0, 0.0],
            &[1.0, 1.0],
            0.0,
            &Costs::paper(),
            &default_grid(),
        )
        .unwrap();
        assert_eq!(c.escalation_rate, 0.0);
        assert_eq!(c.theta, 1.0);
        assert_eq!(c.cost, 1.0);
    }

    #[test]
    fn infeasible_reports_best_accuracy() {
        let err = select_threshold(
            &[0.1, 0.9],
            &[1.0, 0.0],
            &[0.0, 0.0],
            0.9,
            &Costs::paper(),
            &default_grid(),
        )
        .unwrap_err();
        match err {
            UcciError::Infeasible {
                best_accuracy,
                best_theta,
                ..
            } => {
                assert_eq!(best_accuracy, 0.5);
                assert_eq!(best_theta, 0.1);
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn select_matches_brute_force() {
        // Deterministic pseudo-random data; brute force over the grid with
        // evaluate() must agree with the sweep.
        let mut state = 12345u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64) / ((1u64 << 53) as f64)
        };
        let n = 300;
        let p: Vec<f64> = (0..n).map(|_| (next() * 40.0).floor() / 40.0).collect();
        let small: Vec<f64> = p
            .iter()
            .map(|&pi| if next() < pi { 0.0 } else { 1.0 })
            .collect();
        let large: Vec<f64> = (0..n)
            .map(|_| if next() < 0.1 { 0.0 } else { 1.0 })
            .collect();
        let grid = default_grid();
        let costs = Costs::paper();
        for tau in [0.5, 0.7, 0.8, 0.85, 0.9] {
            let brute: Vec<ThresholdChoice> = grid
                .iter()
                .map(|&t| evaluate(&p, &small, &large, t, &costs).unwrap())
                .filter(|c| c.accuracy >= tau)
                .collect();
            let got = select_threshold(&p, &small, &large, tau, &costs, &grid);
            if brute.is_empty() {
                assert!(got.is_err());
                continue;
            }
            let min_cost = brute.iter().map(|c| c.cost).fold(f64::INFINITY, f64::min);
            let best = brute
                .iter()
                .filter(|c| c.cost == min_cost)
                .max_by(|a, b| a.theta.total_cmp(&b.theta))
                .unwrap();
            let got = got.unwrap();
            assert_eq!(got.theta, best.theta, "tau {tau}");
            assert_eq!(got.cost, best.cost);
            assert_eq!(got.escalation_rate, best.escalation_rate);
            // 0/1 scores: both accuracies are exact counts / n.
            assert_eq!(got.accuracy, best.accuracy);
        }
    }

    #[test]
    fn metric_matches_scores_for_mean_accuracy() {
        let p = [0.05, 0.3, 0.3, 0.7, 0.95];
        let small = [1.0, 1.0, 0.0, 0.0, 0.0];
        let large = [1.0, 1.0, 1.0, 1.0, 0.0];
        let grid = default_grid();
        let costs = Costs::paper();
        let mut calls = 0;
        let metric = |esc: &[bool]| {
            calls += 1;
            policy_accuracy(esc, &small, &large).unwrap()
        };
        let a = select_threshold_with_metric(&p, 0.8, &costs, &grid, metric).unwrap();
        // One call per distinct mask: 0, 1, 2, 4, 5 escalations -> 5 masks.
        assert_eq!(calls, 5);
        let b = select_threshold(&p, &small, &large, 0.8, &costs, &grid).unwrap();
        assert_eq!(a, b);
        let bad = select_threshold_with_metric(&p, 0.8, &costs, &grid, |_| f64::NAN);
        assert!(matches!(
            bad,
            Err(UcciError::InvalidParameter { name: "metric", .. })
        ));
    }

    #[test]
    fn budget_form() {
        let p = [0.1, 0.2, 0.6, 0.9];
        let small = [1.0, 1.0, 0.0, 0.0];
        let large = [1.0; 4];
        let costs = routing(1.0, 3.0);
        let grid = default_grid();
        // Budget 2.0 allows escalating half the queries: the two hard ones.
        let c = select_threshold_for_budget(&p, &small, &large, 2.0, &costs, &grid).unwrap();
        assert_eq!((c.accuracy, c.cost, c.theta), (1.0, 2.0, 0.595));
        // Budget 1.5 allows one escalation: the hardest query.
        let c = select_threshold_for_budget(&p, &small, &large, 1.5, &costs, &grid).unwrap();
        assert_eq!((c.accuracy, c.cost, c.theta), (0.75, 1.5, 0.895));
        match select_threshold_for_budget(&p, &small, &large, 0.5, &costs, &grid) {
            Err(UcciError::OverBudget { min_cost, .. }) => assert_eq!(min_cost, 1.0),
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn frontier_marks_efficient_points() {
        let p = [0.1, 0.2, 0.6, 0.9];
        let small = [1.0, 1.0, 0.0, 0.0];
        let large = [1.0; 4];
        let f = pareto_frontier(
            &p,
            &small,
            &large,
            &routing(1.0, 3.0),
            &[0.0, 0.15, 0.5, 0.8, 1.0],
        )
        .unwrap();
        assert_eq!(f.len(), 5);
        assert_eq!(f.escalation_rate, vec![1.0, 0.75, 0.5, 0.25, 0.0]);
        assert_eq!(f.accuracy, vec![1.0, 1.0, 1.0, 0.75, 0.5]);
        assert_eq!(f.efficient, vec![false, false, true, true, true]);
        assert_eq!(f.efficient_points().count(), 3);
        assert_eq!(f.points().count(), 5);
        assert!(!f.is_empty());
    }

    #[test]
    fn grid_is_sorted_and_deduplicated() {
        let p = [0.1, 0.9];
        let s = [1.0, 0.0];
        let l = [1.0, 1.0];
        let f = pareto_frontier(&p, &s, &l, &Costs::paper(), &[0.5, 0.2, 0.5, 0.0]).unwrap();
        assert_eq!(f.theta, vec![0.0, 0.2, 0.5]);
        assert!(pareto_frontier(&p, &s, &l, &Costs::paper(), &[]).is_err());
        assert!(pareto_frontier(&p, &s, &l, &Costs::paper(), &[1.5]).is_err());
        assert!(pareto_frontier(&p, &s, &l, &Costs::paper(), &[f64::NAN]).is_err());
    }

    #[test]
    fn evaluate_reports_actual_cost_and_accuracy() {
        let c = evaluate(&[0.1, 0.6], &[1.0, 0.0], &[1.0, 1.0], 0.5, &Costs::paper()).unwrap();
        assert_eq!(c.escalation_rate, 0.5);
        assert_eq!(c.accuracy, 1.0);
        assert_eq!(c.cost, 1.0 * 0.5 + 3.02 * 0.5);
        let m = evaluate_with_metric(&[0.1, 0.6], 0.5, &Costs::paper(), |esc| {
            esc.iter().filter(|&&e| e).count() as f64
        })
        .unwrap();
        assert_eq!(m.accuracy, 1.0);
        assert!(evaluate(&[], &[], &[], 0.5, &Costs::paper()).is_err());
        assert!(evaluate(&[0.1], &[1.0], &[1.0], f64::NAN, &Costs::paper()).is_err());
    }

    #[test]
    fn input_errors() {
        let g = default_grid();
        let c = Costs::paper();
        assert!(select_threshold(&[], &[], &[], 0.5, &c, &g).is_err());
        assert!(select_threshold(&[0.1], &[1.0, 1.0], &[1.0], 0.5, &c, &g).is_err());
        assert!(select_threshold(&[f64::NAN], &[1.0], &[1.0], 0.5, &c, &g).is_err());
        assert!(select_threshold(&[0.1], &[1.0], &[1.0], f64::NAN, &c, &g).is_err());
        assert!(select_threshold_for_budget(&[0.1], &[1.0], &[1.0], f64::NAN, &c, &g).is_err());
    }
}
