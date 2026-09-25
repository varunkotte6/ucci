//! Forecast-quality and evaluation metrics (Section 6.2, Figure 1).
//!
//! * [`ece`] and [`reliability_table`]: expected calibration error and the
//!   reliability diagram of a probability forecast. The paper reports ECE 0.12
//!   for the raw token margin and 0.03 after isotonic calibration on its
//!   private workload (Section 6.2).
//! * [`brier_score`]: mean squared error of a probability forecast.
//! * [`micro_f1`] and [`RoutedMicroF1`]: the paper's evaluation metric,
//!   micro-averaged F1 over entities (Sections 3 and 6.1), for a single model
//!   and for the answers a routing policy returns.
//!
//! For UCCI the forecast `p` is `p_hat(x)`, the calibrated probability that the
//! small model is wrong, and the event `y` is `e(x)` (1 when it is wrong). The
//! paper does not state its ECE binning; the default in the Python package is
//! 10 equal-width bins, and [`BinStrategy::Quantile`] gives equal-count bins
//! (deciles with 10 bins).
//!
//! Bootstrap confidence intervals depend on numpy's random generator and are
//! only provided by the Python package (`ucci.bootstrap_ci`).
//!
//! # Example
//!
//! ```
//! use ucci::metrics::{ece, reliability_table, BinStrategy};
//!
//! let p = [0.05, 0.15, 0.15, 0.95];
//! let y = [0.0, 0.0, 1.0, 1.0];
//! let rows = reliability_table(&p, &y, 10, BinStrategy::Uniform, None)?;
//! let summary: Vec<(usize, f64)> = rows.iter().map(|r| (r.count, r.observed_frequency)).collect();
//! assert_eq!(summary, vec![(1, 0.0), (2, 0.5), (1, 1.0)]);
//! assert_eq!(ece(&[0.25; 4], &[1.0, 0.0, 0.0, 0.0], 10, BinStrategy::Uniform, None)?, 0.0);
//! # Ok::<(), ucci::UcciError>(())
//! ```

use std::fmt;
use std::str::FromStr;

use crate::error::{Result, UcciError};
use crate::num::{check_finite, check_finite_scalar, check_same_len, pairwise_sum};
use crate::policy::linspace01;

/// How [`reliability_table`] and [`ece`] place their bins.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "json", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "json", serde(rename_all = "lowercase"))]
pub enum BinStrategy {
    /// Equal-width bins on `[0, 1]`.
    #[default]
    Uniform,
    /// Equal-count bins at the quantiles of the forecasts (numpy's default
    /// linear interpolation); duplicate edges from repeated forecasts are
    /// merged, so there can be fewer bins than requested. Quantile edges
    /// ignore the weights.
    Quantile,
}

impl BinStrategy {
    /// The name used by the Python package (`"uniform"` or `"quantile"`).
    pub fn as_str(self) -> &'static str {
        match self {
            BinStrategy::Uniform => "uniform",
            BinStrategy::Quantile => "quantile",
        }
    }
}

impl fmt::Display for BinStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for BinStrategy {
    type Err = UcciError;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "uniform" => Ok(BinStrategy::Uniform),
            "quantile" => Ok(BinStrategy::Quantile),
            other => Err(UcciError::InvalidParameter {
                name: "strategy",
                reason: format!("must be 'uniform' or 'quantile', got {other:?}"),
            }),
        }
    }
}

/// One non-empty bin of a reliability diagram (Figure 1).
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "json", derive(serde::Serialize, serde::Deserialize))]
pub struct ReliabilityRow {
    /// Lower bin edge. Bins are right-closed, `(bin_lower, bin_upper]`, except
    /// the first, which also contains `bin_lower`.
    pub bin_lower: f64,
    /// Upper bin edge.
    pub bin_upper: f64,
    /// Number of forecasts in the bin (zero-weight points excluded).
    pub count: usize,
    /// (Weighted) mean forecast probability in the bin.
    pub mean_forecast: f64,
    /// (Weighted) frequency of the event in the bin.
    pub observed_frequency: f64,
}

/// Validated forecasts, outcomes and positive weights (zero weights dropped).
fn forecast_inputs(
    p: &[f64],
    y: &[f64],
    sample_weight: Option<&[f64]>,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    if p.is_empty() {
        return Err(UcciError::Empty { what: "p" });
    }
    check_finite(p, "p")?;
    check_unit_interval(p, "p")?;
    if y.is_empty() {
        return Err(UcciError::Empty { what: "y" });
    }
    check_finite(y, "y")?;
    check_unit_interval(y, "y")?;
    check_same_len(p.len(), y.len(), "p and y")?;
    match sample_weight {
        None => Ok((p.to_vec(), y.to_vec(), vec![1.0; p.len()])),
        Some(w) => {
            check_same_len(p.len(), w.len(), "p and sample_weight")?;
            check_finite(w, "sample_weight")?;
            if let Some(index) = w.iter().position(|&v| v < 0.0) {
                return Err(UcciError::OutOfRange {
                    what: "sample_weight",
                    index,
                    value: w[index],
                    expected: "non-negative",
                });
            }
            if !w.iter().any(|&v| v > 0.0) {
                return Err(UcciError::InvalidParameter {
                    name: "sample_weight",
                    reason: "sums to zero; at least one weight must be positive".to_string(),
                });
            }
            let keep: Vec<usize> = (0..w.len()).filter(|&i| w[i] > 0.0).collect();
            Ok((
                keep.iter().map(|&i| p[i]).collect(),
                keep.iter().map(|&i| y[i]).collect(),
                keep.iter().map(|&i| w[i]).collect(),
            ))
        }
    }
}

fn check_unit_interval(v: &[f64], what: &'static str) -> Result<()> {
    match v.iter().position(|x| !(0.0..=1.0).contains(x)) {
        Some(index) => Err(UcciError::OutOfRange {
            what,
            index,
            value: v[index],
            expected: "in [0, 1]",
        }),
        None => Ok(()),
    }
}

/// `numpy.quantile(sorted, q)` with the default linear method, for sorted,
/// non-empty input.
fn np_quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    let last = (n - 1) as f64;
    let v = last * q;
    if v >= last {
        return sorted[n - 1];
    }
    if v < 0.0 {
        return sorted[0];
    }
    let prev = v.floor();
    let gamma = v - prev;
    let i = prev as usize;
    let (a, b) = (sorted[i], sorted[i + 1]);
    let diff = b - a;
    if gamma >= 0.5 {
        b - diff * (1.0 - gamma)
    } else {
        a + diff * gamma
    }
}

fn bin_edges(p: &[f64], n_bins: usize, strategy: BinStrategy) -> Vec<f64> {
    let q = linspace01(n_bins);
    match strategy {
        BinStrategy::Uniform => q,
        BinStrategy::Quantile => {
            let mut sorted = p.to_vec();
            sorted.sort_by(f64::total_cmp);
            let mut edges: Vec<f64> = q
                .iter()
                .map(|&qi| np_quantile_sorted(&sorted, qi))
                .collect();
            edges.sort_by(f64::total_cmp);
            edges.dedup_by(|a, b| a == b);
            edges
        }
    }
}

/// Reliability rows plus per-row total weight and the overall total weight.
fn binned(
    p: &[f64],
    y: &[f64],
    n_bins: usize,
    strategy: BinStrategy,
    sample_weight: Option<&[f64]>,
) -> Result<(Vec<ReliabilityRow>, Vec<f64>, f64)> {
    if n_bins < 1 {
        return Err(UcciError::InvalidParameter {
            name: "n_bins",
            reason: format!("must be at least 1, got {n_bins}"),
        });
    }
    let (pp, yy, w) = forecast_inputs(p, y, sample_weight)?;
    let edges = bin_edges(&pp, n_bins, strategy);
    let n_edges = edges.len();
    let (ids, lowers, uppers): (Vec<usize>, Vec<f64>, Vec<f64>) = if n_edges == 1 {
        // Quantile bins of a constant forecast: one degenerate bin.
        (vec![0; pp.len()], edges.clone(), edges.clone())
    } else {
        // Right-closed bins; a forecast equal to edges[0] lands in the first bin.
        let ids = pp
            .iter()
            .map(|&x| {
                let pos = edges.partition_point(|&e| e < x);
                pos.saturating_sub(1).min(n_edges - 2)
            })
            .collect();
        (ids, edges[..n_edges - 1].to_vec(), edges[1..].to_vec())
    };
    let n_groups = lowers.len();
    let mut counts = vec![0usize; n_groups];
    let mut w_bin = vec![0.0f64; n_groups];
    let mut wp_bin = vec![0.0f64; n_groups];
    let mut wy_bin = vec![0.0f64; n_groups];
    for (i, &b) in ids.iter().enumerate() {
        counts[b] += 1;
        w_bin[b] += w[i];
        wp_bin[b] += w[i] * pp[i];
        wy_bin[b] += w[i] * yy[i];
    }
    let mut rows = Vec::new();
    let mut weights = Vec::new();
    for b in 0..n_groups {
        if counts[b] == 0 {
            continue;
        }
        rows.push(ReliabilityRow {
            bin_lower: lowers[b],
            bin_upper: uppers[b],
            count: counts[b],
            mean_forecast: wp_bin[b] / w_bin[b],
            observed_frequency: wy_bin[b] / w_bin[b],
        });
        weights.push(w_bin[b]);
    }
    Ok((rows, weights, pairwise_sum(&w)))
}

/// Reliability diagram rows (Figure 1): per non-empty bin, the mean forecast
/// against the observed frequency of the event.
///
/// `p` holds forecast probabilities of the event `y = 1` (for UCCI: `p_hat`,
/// the probability that the small model is wrong) and `y` the outcomes in
/// `[0, 1]` (for UCCI: `e(x)`). `sample_weight` holds optional non-negative
/// weights; zero-weight points are dropped. Rows come in increasing bin order.
///
/// # Errors
///
/// [`UcciError::Empty`], [`UcciError::NonFinite`], [`UcciError::OutOfRange`] or
/// [`UcciError::LengthMismatch`] for invalid inputs and
/// [`UcciError::InvalidParameter`] for `n_bins == 0` or weights that sum to
/// zero.
pub fn reliability_table(
    p: &[f64],
    y: &[f64],
    n_bins: usize,
    strategy: BinStrategy,
    sample_weight: Option<&[f64]>,
) -> Result<Vec<ReliabilityRow>> {
    Ok(binned(p, y, n_bins, strategy, sample_weight)?.0)
}

/// Expected calibration error (Naeini et al., 2015; Guo et al., 2017), the
/// calibration measure of Section 6.2:
///
/// ```text
/// ECE = sum_b (W_b / W) * |mean_forecast_b - observed_frequency_b|
/// ```
///
/// where `W_b` is the total weight in bin `b` (the count when unweighted).
/// Proposition 2 bounds the expected ECE of the isotonic calibrator by
/// `O(n^{-1/3})` in the calibration set size.
///
/// # Errors
///
/// As [`reliability_table`].
pub fn ece(
    p: &[f64],
    y: &[f64],
    n_bins: usize,
    strategy: BinStrategy,
    sample_weight: Option<&[f64]>,
) -> Result<f64> {
    let (rows, weights, total) = binned(p, y, n_bins, strategy, sample_weight)?;
    let terms: Vec<f64> = rows
        .iter()
        .zip(&weights)
        .map(|(r, &wb)| wb / total * (r.mean_forecast - r.observed_frequency).abs())
        .collect();
    Ok(pairwise_sum(&terms))
}

/// Brier score: the (weighted) mean of `(p - y)^2`. Lower is better.
///
/// # Errors
///
/// [`UcciError::Empty`], [`UcciError::NonFinite`], [`UcciError::OutOfRange`] or
/// [`UcciError::LengthMismatch`] for invalid inputs and
/// [`UcciError::InvalidParameter`] for weights that sum to zero.
///
/// # Example
///
/// ```
/// let b = ucci::metrics::brier_score(&[0.0, 1.0, 0.5], &[0.0, 0.0, 1.0], None)?;
/// assert_eq!(b, (0.0 + 1.0 + 0.25) / 3.0);
/// # Ok::<(), ucci::UcciError>(())
/// ```
pub fn brier_score(p: &[f64], y: &[f64], sample_weight: Option<&[f64]>) -> Result<f64> {
    let (pp, yy, w) = forecast_inputs(p, y, sample_weight)?;
    let terms: Vec<f64> = (0..pp.len())
        .map(|i| {
            let d = pp[i] - yy[i];
            w[i] * (d * d)
        })
        .collect();
    Ok(pairwise_sum(&terms) / pairwise_sum(&w))
}

/// Micro-averaged F1 from summed counts: `F1 = 2 TP / (2 TP + FP + FN)`,
/// the paper's evaluation metric (Section 6.1).
///
/// `zero_division` is returned when `2 TP + FP + FN = 0` (no gold and no
/// predicted entities); 0.0 is scikit-learn's default for that case.
///
/// # Errors
///
/// [`UcciError::InvalidParameter`] for negative or non-finite counts or a
/// non-finite `zero_division`.
///
/// # Example
///
/// ```
/// assert_eq!(ucci::metrics::micro_f1(3.0, 1.0, 1.0, 0.0)?, 0.75);
/// assert_eq!(ucci::metrics::micro_f1(0.0, 0.0, 0.0, 1.0)?, 1.0);
/// # Ok::<(), ucci::UcciError>(())
/// ```
pub fn micro_f1(tp: f64, fp: f64, fn_: f64, zero_division: f64) -> Result<f64> {
    for (name, v) in [("tp", tp), ("fp", fp), ("fn", fn_)] {
        check_finite_scalar(v, name)?;
        if v < 0.0 {
            return Err(UcciError::InvalidParameter {
                name,
                reason: format!("counts must be non-negative, got {v}"),
            });
        }
    }
    let denom = 2.0 * tp + fp + fn_;
    if denom == 0.0 {
        check_finite_scalar(zero_division, "zero_division")?;
        return Ok(zero_division);
    }
    Ok(2.0 * tp / denom)
}

/// Micro-F1 of the answers a routing mask returns: the corpus-level accuracy
/// functional for threshold selection on micro-F1 (Section 6.1).
///
/// Holds per-query `(tp, fp, fn)` entity counts of each model's output against
/// the gold labels. [`RoutedMicroF1::score`] sums the small model's counts over
/// kept queries and the large model's over escalated ones and returns
/// [`micro_f1`] of the totals.
///
/// # Example
///
/// ```
/// use ucci::metrics::RoutedMicroF1;
/// use ucci::policy::{default_grid, select_threshold_with_metric, Costs};
///
/// let small = [[1.0, 0.0, 1.0], [2.0, 0.0, 0.0]];
/// let large = [[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
/// let f1 = RoutedMicroF1::new(&small, &large, 0.0)?;
/// assert!((f1.score(&[false, false])? - 6.0 / 7.0).abs() < 1e-15);
/// assert_eq!(f1.score(&[true, false])?, 1.0);
///
/// // As the accuracy functional of Eq. 7:
/// let p_hat = [0.8, 0.1];
/// let choice = select_threshold_with_metric(&p_hat, 1.0, &Costs::paper(), &default_grid(), f1.metric())?;
/// assert_eq!(choice.escalation_rate, 0.5);
/// # Ok::<(), ucci::UcciError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct RoutedMicroF1 {
    small: Vec<[f64; 3]>,
    large: Vec<[f64; 3]>,
    zero_division: f64,
}

impl RoutedMicroF1 {
    /// Per-query `(tp, fp, fn)` counts of the small and the large model.
    ///
    /// # Errors
    ///
    /// [`UcciError::Empty`] for no queries, [`UcciError::LengthMismatch`] if
    /// the two differ in length, and [`UcciError::InvalidParameter`] for
    /// negative or non-finite counts or a non-finite `zero_division`.
    pub fn new(
        small_counts: &[[f64; 3]],
        large_counts: &[[f64; 3]],
        zero_division: f64,
    ) -> Result<Self> {
        for (name, counts) in [
            ("small_counts", small_counts),
            ("large_counts", large_counts),
        ] {
            if counts.is_empty() {
                return Err(UcciError::Empty { what: name });
            }
            if counts.iter().flatten().any(|v| !v.is_finite() || *v < 0.0) {
                return Err(UcciError::InvalidParameter {
                    name,
                    reason: "counts must be finite and non-negative".to_string(),
                });
            }
        }
        check_same_len(
            small_counts.len(),
            large_counts.len(),
            "small_counts and large_counts",
        )?;
        check_finite_scalar(zero_division, "zero_division")?;
        Ok(Self {
            small: small_counts.to_vec(),
            large: large_counts.to_vec(),
            zero_division,
        })
    }

    /// Number of queries.
    pub fn len(&self) -> usize {
        self.small.len()
    }

    /// True when there are no queries (never the case after [`RoutedMicroF1::new`]).
    pub fn is_empty(&self) -> bool {
        self.small.is_empty()
    }

    /// Micro-F1 of the routed answers for an escalation mask (`true` means the
    /// large model answers).
    ///
    /// # Errors
    ///
    /// [`UcciError::LengthMismatch`] if the mask length differs from the
    /// number of queries.
    pub fn score(&self, esc: &[bool]) -> Result<f64> {
        check_same_len(esc.len(), self.small.len(), "esc and counts")?;
        let mut small = [0.0f64; 3];
        let mut large = [0.0f64; 3];
        for (i, &e) in esc.iter().enumerate() {
            let (acc, row) = if e {
                (&mut large, &self.large[i])
            } else {
                (&mut small, &self.small[i])
            };
            for k in 0..3 {
                acc[k] += row[k];
            }
        }
        micro_f1(
            small[0] + large[0],
            small[1] + large[1],
            small[2] + large[2],
            self.zero_division,
        )
    }

    /// This metric as a closure for [`crate::policy::select_threshold_with_metric`]
    /// and the other `*_with_metric` functions. A mask of the wrong length
    /// yields NaN, which those functions report as an error.
    pub fn metric(&self) -> impl FnMut(&[bool]) -> f64 + '_ {
        move |esc: &[bool]| self.score(esc).unwrap_or(f64::NAN)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_docstring_examples() {
        let rows = reliability_table(
            &[0.05, 0.15, 0.15, 0.95],
            &[0.0, 0.0, 1.0, 1.0],
            10,
            BinStrategy::Uniform,
            None,
        )
        .unwrap();
        let got: Vec<(usize, f64)> = rows
            .iter()
            .map(|r| (r.count, r.observed_frequency))
            .collect();
        assert_eq!(got, vec![(1, 0.0), (2, 0.5), (1, 1.0)]);
        assert_eq!(rows[1].bin_lower, 0.1);
        assert_eq!(rows[1].bin_upper, 0.2);
        assert_eq!(
            ece(
                &[0.25; 4],
                &[1.0, 0.0, 0.0, 0.0],
                10,
                BinStrategy::Uniform,
                None
            )
            .unwrap(),
            0.0
        );
        assert_eq!(micro_f1(3.0, 1.0, 1.0, 0.0).unwrap(), 0.75);
    }

    #[test]
    fn bins_are_right_closed() {
        // 0.1 is an edge: it belongs to the first bin (0, 0.1].
        let rows = reliability_table(
            &[0.0, 0.1, 0.1000001],
            &[0.0; 3],
            10,
            BinStrategy::Uniform,
            None,
        )
        .unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].count, 2);
        assert_eq!(rows[1].count, 1);
        // 1.0 lands in the last bin.
        let rows = reliability_table(&[1.0], &[1.0], 10, BinStrategy::Uniform, None).unwrap();
        assert_eq!((rows[0].bin_lower, rows[0].bin_upper), (0.9, 1.0));
    }

    #[test]
    fn quantile_bins() {
        // numpy.quantile([0.1, 0.2, 0.3, 0.4], [0, 0.5, 1]) = [0.1, 0.25, 0.4]
        let rows = reliability_table(
            &[0.4, 0.1, 0.3, 0.2],
            &[1.0, 0.0, 1.0, 0.0],
            2,
            BinStrategy::Quantile,
            None,
        )
        .unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].count, 2);
        assert!((rows[0].bin_upper - 0.25).abs() < 1e-15);
        // A constant forecast collapses to one degenerate bin.
        let rows = reliability_table(
            &[0.3; 5],
            &[1.0, 0.0, 0.0, 0.0, 0.0],
            10,
            BinStrategy::Quantile,
            None,
        )
        .unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!((rows[0].bin_lower, rows[0].bin_upper), (0.3, 0.3));
        let e = ece(
            &[0.3; 5],
            &[1.0, 0.0, 0.0, 0.0, 0.0],
            10,
            BinStrategy::Quantile,
            None,
        )
        .unwrap();
        assert!((e - 0.1).abs() < 1e-15);
    }

    #[test]
    fn np_quantile_linear() {
        let s = [1.0, 2.0, 4.0, 8.0];
        assert_eq!(np_quantile_sorted(&s, 0.0), 1.0);
        assert_eq!(np_quantile_sorted(&s, 1.0), 8.0);
        assert_eq!(np_quantile_sorted(&s, 0.5), 3.0);
        assert!((np_quantile_sorted(&s, 0.25) - 1.75).abs() < 1e-15);
        assert_eq!(np_quantile_sorted(&[5.0], 0.3), 5.0);
    }

    #[test]
    fn weights() {
        let p = [0.2, 0.2, 0.8];
        let y = [0.0, 1.0, 1.0];
        // Zero weight drops the second point entirely.
        let a = ece(&p, &y, 10, BinStrategy::Uniform, Some(&[1.0, 0.0, 1.0])).unwrap();
        let b = ece(&[0.2, 0.8], &[0.0, 1.0], 10, BinStrategy::Uniform, None).unwrap();
        assert_eq!(a, b);
        // Integer weights equal repetition.
        let c = brier_score(&p, &y, Some(&[2.0, 1.0, 1.0])).unwrap();
        let d = brier_score(&[0.2, 0.2, 0.2, 0.8], &[0.0, 0.0, 1.0, 1.0], None).unwrap();
        assert!((c - d).abs() < 1e-15);
        assert!(ece(&p, &y, 10, BinStrategy::Uniform, Some(&[0.0; 3])).is_err());
        assert!(ece(&p, &y, 10, BinStrategy::Uniform, Some(&[1.0, -1.0, 1.0])).is_err());
        assert!(ece(&p, &y, 10, BinStrategy::Uniform, Some(&[1.0])).is_err());
    }

    #[test]
    fn input_errors() {
        let u = BinStrategy::Uniform;
        assert!(ece(&[], &[], 10, u, None).is_err());
        assert!(ece(&[0.5], &[1.0, 0.0], 10, u, None).is_err());
        assert!(ece(&[1.5], &[1.0], 10, u, None).is_err());
        assert!(ece(&[0.5], &[2.0], 10, u, None).is_err());
        assert!(ece(&[f64::NAN], &[1.0], 10, u, None).is_err());
        assert!(ece(&[0.5], &[1.0], 0, u, None).is_err());
        assert!(brier_score(&[0.5], &[f64::INFINITY], None).is_err());
        assert!("deciles".parse::<BinStrategy>().is_err());
        assert_eq!(
            "quantile".parse::<BinStrategy>().unwrap(),
            BinStrategy::Quantile
        );
        assert_eq!(BinStrategy::Uniform.to_string(), "uniform");
    }

    #[test]
    fn micro_f1_and_routed() {
        assert_eq!(micro_f1(0.0, 0.0, 0.0, 0.0).unwrap(), 0.0);
        assert!(micro_f1(-1.0, 0.0, 0.0, 0.0).is_err());
        assert!(micro_f1(f64::NAN, 0.0, 0.0, 0.0).is_err());
        let small = [[1.0, 0.0, 1.0], [2.0, 0.0, 0.0]];
        let large = [[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        let f1 = RoutedMicroF1::new(&small, &large, 0.0).unwrap();
        assert_eq!(f1.len(), 2);
        assert!(!f1.is_empty());
        assert_eq!(f1.score(&[true, true]).unwrap(), 1.0);
        assert!(f1.score(&[true]).is_err());
        let mut m = f1.metric();
        assert!(m(&[true]).is_nan());
        assert!(RoutedMicroF1::new(&[], &[], 0.0).is_err());
        assert!(RoutedMicroF1::new(&small, &large[..1], 0.0).is_err());
        assert!(RoutedMicroF1::new(&[[1.0, -1.0, 0.0]], &[[1.0, 0.0, 0.0]], 0.0).is_err());
    }
}
