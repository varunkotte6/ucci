//! Isotonic calibration of u(x) into an error probability (paper Section 4.2).
//!
//! The calibrator learns a non-decreasing map `g` with
//!
//! ```text
//! g(u) ~= P(e(x) = 1 | u(x) = u)        (Eq. 5)
//! ```
//!
//! on a held-out calibration set of `(u_i, e_i)` pairs, where `e_i = 1` when
//! the small model's answer is wrong. The calibrated forecast is
//! `p_hat(x) = g(u(x))`.
//!
//! The paper fits `g` by isotonic regression with "standard open-source
//! libraries (default settings)" (Appendix B.2), that is scikit-learn's
//! `IsotonicRegression`. [`IsotonicCalibrator::fit_weighted`] reproduces that
//! estimator step by step (zero-weight points dropped, tied `u` values pooled,
//! weighted pool-adjacent-violators, flat interior knots trimmed) with the
//! same floating-point operations as the Python package, so knots fitted in
//! Rust and in Python agree bit for bit.
//!
//! Predictions interpolate linearly between the fitted knots and are clipped
//! at both ends, like `IsotonicRegression(out_of_bounds="clip")`. The clip is
//! the one deliberate difference from scikit-learn's default
//! (`out_of_bounds="nan"`), which returns NaN outside the calibration range.
//!
//! # Example
//!
//! ```
//! use ucci::calibration::IsotonicCalibrator;
//!
//! let u = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
//! let e = [0.0, 1.0, 0.0, 0.0, 1.0, 1.0];
//! let g = IsotonicCalibrator::fit(&u, &e)?;
//! // The violators (0.2 -> 1, 0.3 -> 0, 0.4 -> 0) are pooled to 1/3; the
//! // flat interior knot at 0.3 is trimmed.
//! assert_eq!(g.x(), &[0.1, 0.2, 0.4, 0.5, 0.6]);
//! assert_eq!(g.y(), &[0.0, 1.0 / 3.0, 1.0 / 3.0, 1.0, 1.0]);
//! assert_eq!(g.predict(0.0)?, 0.0); // clipped below the calibration range
//! assert_eq!(g.predict(0.9)?, 1.0); // clipped above it
//! assert!((g.predict(0.15)? - 1.0 / 6.0).abs() < 1e-15); // linear interpolation
//! # Ok::<(), ucci::UcciError>(())
//! ```

use crate::error::{Result, UcciError};
use crate::num::{check_finite, check_same_len, reduceat_segment};

/// `u` values closer than this to the first value of their group are pooled
/// into one knot, as in scikit-learn (`numpy.finfo(numpy.float64).resolution`).
pub const TIE_TOLERANCE: f64 = 1e-15;

/// Weighted least-squares non-decreasing fit of `y` (already ordered by the
/// covariate) by the pool-adjacent-violators algorithm (Section 4.2).
///
/// Solves `min_f sum_i w_i (y_i - f_i)^2` subject to `f_1 <= ... <= f_n`
/// (Barlow et al., 1972). `w = None` means unit weights. Each maximal run of
/// equal fitted values is the weighted mean of the targets it covers. Blocks
/// are kept as (weight sum, weighted target sum) and merged while the block
/// below has a strictly larger mean, with the same floating-point operations
/// as the Python package, so the two agree bit for bit. O(n).
///
/// # Errors
///
/// [`UcciError::Empty`] for empty `y`, [`UcciError::LengthMismatch`] if `w`
/// has the wrong length, [`UcciError::NonFinite`] for NaN or infinite values
/// and [`UcciError::OutOfRange`] for a weight that is not strictly positive
/// (drop zero-weight points first, as [`IsotonicCalibrator::fit_weighted`]
/// does).
///
/// # Example
///
/// ```
/// let fit = ucci::calibration::pav(&[1.0, 3.0, 2.0, 4.0], None)?;
/// assert_eq!(fit, vec![1.0, 2.5, 2.5, 4.0]);
/// let fit = ucci::calibration::pav(&[3.0, 1.0], Some(&[3.0, 1.0]))?;
/// assert_eq!(fit, vec![2.5, 2.5]);
/// # Ok::<(), ucci::UcciError>(())
/// ```
pub fn pav(y: &[f64], w: Option<&[f64]>) -> Result<Vec<f64>> {
    if y.is_empty() {
        return Err(UcciError::Empty { what: "y" });
    }
    check_finite(y, "y")?;
    let ones;
    let w = match w {
        Some(w) => {
            check_same_len(y.len(), w.len(), "y and w")?;
            check_finite(w, "w")?;
            if let Some(index) = w.iter().position(|&wi| wi <= 0.0) {
                return Err(UcciError::OutOfRange {
                    what: "w",
                    index,
                    value: w[index],
                    expected: "strictly positive (drop zero-weight points before calling pav)",
                });
            }
            w
        }
        None => {
            ones = vec![1.0; y.len()];
            &ones[..]
        }
    };
    Ok(expand(&pav_blocks(y, w)))
}

/// One PAV block: weight sum, weighted target sum, mean and number of points.
#[derive(Debug, Clone, Copy)]
struct Block {
    sum_w: f64,
    sum_wy: f64,
    mean: f64,
    len: usize,
}

/// Weighted PAV on validated input, returning the final blocks.
///
/// A new point starts as its own block (mean = its target) and is pooled with
/// the block below while that block's mean is strictly larger, so monotone
/// input passes through unchanged. Mirrors `ucci.calibration._pav_blocks`.
fn pav_blocks(y: &[f64], w: &[f64]) -> Vec<Block> {
    let mut blocks: Vec<Block> = Vec::with_capacity(y.len());
    for (&yi, &wi) in y.iter().zip(w) {
        let mut cur = Block {
            sum_w: wi,
            sum_wy: wi * yi,
            mean: yi,
            len: 1,
        };
        while let Some(top) = blocks.last() {
            if top.mean <= cur.mean {
                break;
            }
            let top = blocks.pop().expect("checked by last()");
            cur.sum_w += top.sum_w;
            cur.sum_wy += top.sum_wy;
            cur.len += top.len;
            cur.mean = cur.sum_wy / cur.sum_w;
        }
        blocks.push(cur);
    }
    blocks
}

fn expand(blocks: &[Block]) -> Vec<f64> {
    let n = blocks.iter().map(|b| b.len).sum();
    let mut out = Vec::with_capacity(n);
    for b in blocks {
        out.extend(std::iter::repeat(b.mean).take(b.len));
    }
    out
}

/// Monotone map from uncertainty `u` to `P(small model wrong)`: the
/// calibration map `g` of Section 4.2, Eq. 5.
///
/// Stored as knots `(x, y)` with `x` strictly increasing and `y`
/// non-decreasing in `[0, 1]` (scikit-learn's `X_thresholds_` and
/// `y_thresholds_`). [`IsotonicCalibrator::predict`] interpolates linearly
/// between knots and clips at both ends, exactly like `numpy.interp`.
#[derive(Debug, Clone, PartialEq)]
pub struct IsotonicCalibrator {
    x: Vec<f64>,
    y: Vec<f64>,
    n_samples: Option<usize>,
}

impl IsotonicCalibrator {
    /// Fits `g` on calibration pairs `(u_i, e_i)` with unit weights
    /// (Section 4.2, Eq. 5).
    ///
    /// `u[i]` is the uncertainty of calibration query `i` (Eq. 4) and `e[i]`
    /// its error label: 1 if the small model's output was wrong, 0 if right.
    /// Soft labels in `[0, 1]` are accepted.
    ///
    /// # Errors
    ///
    /// See [`IsotonicCalibrator::fit_weighted`].
    pub fn fit(u: &[f64], e: &[f64]) -> Result<Self> {
        Self::fit_impl(u, e, None)
    }

    /// Fits `g` with per-sample weights (Section 4.2), reproducing
    /// scikit-learn's `IsotonicRegression().fit(u, e, sample_weight)`:
    ///
    /// 1. zero-weight points are dropped;
    /// 2. points are sorted by `u` (ties by `e`), and `u` values closer than
    ///    [`TIE_TOLERANCE`] to the first value of their group are pooled into
    ///    one point with the weighted mean label and the summed weight;
    /// 3. the pooled labels are fitted by weighted PAV ([`pav`]);
    /// 4. interior knots whose value equals both neighbours are dropped, which
    ///    does not change the fitted function.
    ///
    /// # Errors
    ///
    /// [`UcciError::Empty`] for an empty calibration set,
    /// [`UcciError::LengthMismatch`] for misaligned inputs,
    /// [`UcciError::NonFinite`] for NaN or infinite values,
    /// [`UcciError::OutOfRange`] for a label outside `[0, 1]` or a negative
    /// weight, and [`UcciError::InvalidParameter`] if the weights sum to zero.
    pub fn fit_weighted(u: &[f64], e: &[f64], sample_weight: &[f64]) -> Result<Self> {
        Self::fit_impl(u, e, Some(sample_weight))
    }

    fn fit_impl(u: &[f64], e: &[f64], sample_weight: Option<&[f64]>) -> Result<Self> {
        if u.is_empty() {
            return Err(UcciError::Empty { what: "u" });
        }
        check_finite(u, "u")?;
        if e.is_empty() {
            return Err(UcciError::Empty { what: "e" });
        }
        check_finite(e, "e")?;
        if let Some(index) = e.iter().position(|v| !(0.0..=1.0).contains(v)) {
            return Err(UcciError::OutOfRange {
                what: "e",
                index,
                value: e[index],
                expected: "in [0, 1]",
            });
        }
        check_same_len(u.len(), e.len(), "u and e")?;
        if let Some(w) = sample_weight {
            check_same_len(u.len(), w.len(), "u and sample_weight")?;
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
        }
        let weight = |i: usize| sample_weight.map_or(1.0, |w| w[i]);

        // Drop zero-weight points, then sort by (u, e) with a stable sort,
        // which reproduces numpy.lexsort((e, u)). Adding 0.0 maps -0.0 to
        // 0.0 so that the two zeros compare equal, as they do in numpy.
        let mut order: Vec<usize> = (0..u.len()).filter(|&i| weight(i) > 0.0).collect();
        order.sort_by(|&i, &j| {
            (u[i] + 0.0)
                .total_cmp(&(u[j] + 0.0))
                .then((e[i] + 0.0).total_cmp(&(e[j] + 0.0)))
        });
        let n_samples = order.len();
        let su: Vec<f64> = order.iter().map(|&i| u[i]).collect();
        let sw: Vec<f64> = order.iter().map(|&i| weight(i)).collect();
        let swe: Vec<f64> = order.iter().map(|&i| e[i] * weight(i)).collect();

        // Group starts: exact ties first, then scikit-learn's greedy rule
        // when two distinct values are closer than TIE_TOLERANCE.
        let mut starts: Vec<usize> = (0..su.len())
            .filter(|&i| i == 0 || su[i] != su[i - 1])
            .collect();
        let near_tie = starts
            .windows(2)
            .any(|p| su[p[1]] - su[p[0]] < TIE_TOLERANCE);
        if near_tie {
            let mut keep = vec![starts[0]];
            let mut group_start = su[starts[0]];
            for &s in &starts[1..] {
                if su[s] - group_start >= TIE_TOLERANCE {
                    keep.push(s);
                    group_start = su[s];
                }
            }
            starts = keep;
        }

        // Pool each group: weighted mean label and summed weight, with the
        // summation order of numpy.add.reduceat.
        let mut xs = Vec::with_capacity(starts.len());
        let mut targets = Vec::with_capacity(starts.len());
        let mut weights = Vec::with_capacity(starts.len());
        for (k, &s) in starts.iter().enumerate() {
            let end = starts.get(k + 1).copied().unwrap_or(su.len());
            let w_sum = reduceat_segment(&sw[s..end]);
            xs.push(su[s]);
            targets.push(reduceat_segment(&swe[s..end]) / w_sum);
            weights.push(w_sum);
        }

        let fitted = expand(&pav_blocks(&targets, &weights));

        // Drop interior knots equal to both neighbours (scikit-learn's
        // trim_duplicates); the interpolated function is unchanged.
        let m = fitted.len();
        let keep = |i: usize| {
            i == 0 || i + 1 == m || fitted[i] != fitted[i - 1] || fitted[i] != fitted[i + 1]
        };
        let x: Vec<f64> = (0..m).filter(|&i| keep(i)).map(|i| xs[i]).collect();
        let y: Vec<f64> = (0..m).filter(|&i| keep(i)).map(|i| fitted[i]).collect();
        Ok(Self {
            x,
            y,
            n_samples: Some(n_samples),
        })
    }

    /// Builds a calibrator from existing knots, for example ones fitted by the
    /// Python package and stored in a router file.
    ///
    /// # Errors
    ///
    /// [`UcciError::InvalidCalibrator`] unless `x` and `y` are non-empty, of
    /// equal length and finite, with `x` strictly increasing and `y`
    /// non-decreasing within `[0, 1]`.
    ///
    /// # Example
    ///
    /// ```
    /// use ucci::calibration::IsotonicCalibrator;
    ///
    /// let g = IsotonicCalibrator::from_knots(vec![0.2, 0.6], vec![0.1, 0.5])?;
    /// assert!((g.predict(0.4)? - 0.3).abs() < 1e-15);
    /// assert_eq!(g.predict(0.0)?, 0.1); // clipped
    /// assert!(IsotonicCalibrator::from_knots(vec![0.2, 0.6], vec![0.5, 0.1]).is_err());
    /// # Ok::<(), ucci::UcciError>(())
    /// ```
    pub fn from_knots(x: Vec<f64>, y: Vec<f64>) -> Result<Self> {
        validate_knots(&x, &y)?;
        Ok(Self {
            x,
            y,
            n_samples: None,
        })
    }

    /// Knot positions (u values), strictly increasing. `x()[0]` and the last
    /// element bound the calibration range.
    pub fn x(&self) -> &[f64] {
        &self.x
    }

    /// Fitted error probabilities at the knots, non-decreasing in `[0, 1]`.
    pub fn y(&self) -> &[f64] {
        &self.y
    }

    /// Number of knots (at least one).
    pub fn n_knots(&self) -> usize {
        self.x.len()
    }

    /// Number of positive-weight calibration points used by the fit, or
    /// `None` when the knots came from [`IsotonicCalibrator::from_knots`].
    pub fn n_samples(&self) -> Option<usize> {
        self.n_samples
    }

    /// Consumes the calibrator and returns its knots `(x, y)`.
    pub fn into_knots(self) -> (Vec<f64>, Vec<f64>) {
        (self.x, self.y)
    }

    /// Calibrated error probability `p_hat = g(u)` (Sections 4.2 and 4.3).
    ///
    /// Linear interpolation between knots, clipped to the first and last knot
    /// values outside `[x_min, x_max]` (like scikit-learn with
    /// `out_of_bounds="clip"`). The result lies in `[0, 1]`.
    ///
    /// # Errors
    ///
    /// [`UcciError::NonFinite`] if `u` is NaN or infinite, as in Python.
    pub fn predict(&self, u: f64) -> Result<f64> {
        if !u.is_finite() {
            return Err(UcciError::NonFinite {
                what: "u",
                index: 0,
            });
        }
        Ok(interp(u, &self.x, &self.y))
    }

    /// [`IsotonicCalibrator::predict`] applied to every element of `u`.
    ///
    /// # Errors
    ///
    /// [`UcciError::NonFinite`] naming the first NaN or infinite element.
    pub fn predict_many(&self, u: &[f64]) -> Result<Vec<f64>> {
        check_finite(u, "u")?;
        Ok(u.iter().map(|&v| interp(v, &self.x, &self.y)).collect())
    }
}

/// Checks the knot invariants shared by fitted and deserialized calibrators.
pub(crate) fn validate_knots(x: &[f64], y: &[f64]) -> Result<()> {
    let bad = |reason: String| Err(UcciError::InvalidCalibrator { reason });
    if x.is_empty() {
        return bad("x and y must hold at least one knot".to_string());
    }
    if x.len() != y.len() {
        return bad(format!(
            "x and y must have the same length, got {} and {}",
            x.len(),
            y.len()
        ));
    }
    if let Some(i) = x.iter().position(|v| !v.is_finite()) {
        return bad(format!("x[{i}] is not finite"));
    }
    if let Some(i) = y.iter().position(|v| !v.is_finite()) {
        return bad(format!("y[{i}] is not finite"));
    }
    if let Some(i) = (1..x.len()).find(|&i| x[i] <= x[i - 1]) {
        return bad(format!(
            "x must be strictly increasing, but x[{}] = {} <= x[{}] = {}",
            i,
            x[i],
            i - 1,
            x[i - 1]
        ));
    }
    if let Some(i) = (1..y.len()).find(|&i| y[i] < y[i - 1]) {
        return bad(format!(
            "y must be non-decreasing, but y[{}] = {} < y[{}] = {}",
            i,
            y[i],
            i - 1,
            y[i - 1]
        ));
    }
    if let Some(i) = y.iter().position(|v| !(0.0..=1.0).contains(v)) {
        return bad(format!("y must lie in [0, 1], but y[{}] = {}", i, y[i]));
    }
    Ok(())
}

/// `numpy.interp(v, xp, fp)` for strictly increasing `xp`, with the default
/// `left = fp[0]` and `right = fp[-1]`.
fn interp(v: f64, xp: &[f64], fp: &[f64]) -> f64 {
    let n = xp.len();
    if n == 1 {
        return fp[0];
    }
    if v.is_nan() {
        return v;
    }
    if v < xp[0] {
        return fp[0];
    }
    if v >= xp[n - 1] {
        return fp[n - 1];
    }
    // Largest j with xp[j] <= v; here 0 <= j <= n - 2.
    let j = xp.partition_point(|&x| x <= v) - 1;
    if xp[j] == v {
        return fp[j];
    }
    let slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
    let r = slope * (v - xp[j]) + fp[j];
    if r.is_nan() {
        // numpy retries from the right end of the interval.
        let r2 = slope * (v - xp[j + 1]) + fp[j + 1];
        if r2.is_nan() && fp[j] == fp[j + 1] {
            return fp[j];
        }
        return r2;
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pav_known_cases() {
        assert_eq!(
            pav(&[1.0, 3.0, 2.0, 4.0], None).unwrap(),
            vec![1.0, 2.5, 2.5, 4.0]
        );
        assert_eq!(pav(&[3.0, 1.0], Some(&[3.0, 1.0])).unwrap(), vec![2.5, 2.5]);
        assert_eq!(pav(&[4.0, 3.0, 2.0, 1.0], None).unwrap(), vec![2.5; 4]);
        assert_eq!(pav(&[1.0, 1.0, 1.0], None).unwrap(), vec![1.0; 3]);
        assert_eq!(pav(&[7.0], None).unwrap(), vec![7.0]);
        // A late small value pulls several earlier blocks down: 3 and 0 pool
        // to 1.5, which then pools with 2 to 5/3; 1 stays below.
        assert_eq!(
            pav(&[1.0, 2.0, 3.0, 0.0], None).unwrap(),
            vec![1.0, 5.0 / 3.0, 5.0 / 3.0, 5.0 / 3.0]
        );
    }

    #[test]
    fn pav_errors() {
        assert!(matches!(pav(&[], None), Err(UcciError::Empty { .. })));
        assert!(matches!(
            pav(&[1.0], Some(&[1.0, 2.0])),
            Err(UcciError::LengthMismatch { .. })
        ));
        assert!(matches!(
            pav(&[1.0, 2.0], Some(&[1.0, 0.0])),
            Err(UcciError::OutOfRange { index: 1, .. })
        ));
        assert!(matches!(
            pav(&[1.0, 2.0], Some(&[1.0, -1.0])),
            Err(UcciError::OutOfRange { index: 1, .. })
        ));
        assert!(matches!(
            pav(&[f64::NAN], None),
            Err(UcciError::NonFinite { .. })
        ));
        assert!(matches!(
            pav(&[1.0], Some(&[f64::INFINITY])),
            Err(UcciError::NonFinite { .. })
        ));
    }

    #[test]
    fn fit_matches_python_docstring_example() {
        let g = IsotonicCalibrator::fit(&[0.1, 0.2, 0.3, 0.4], &[0.0, 1.0, 0.0, 1.0]).unwrap();
        assert_eq!(g.x(), &[0.1, 0.2, 0.3, 0.4]);
        assert_eq!(g.y(), &[0.0, 0.5, 0.5, 1.0]);
        assert_eq!(g.predict(0.25).unwrap(), 0.5);
        assert_eq!(g.predict_many(&[0.0, 1.0]).unwrap(), vec![0.0, 1.0]);
        assert_eq!(g.n_samples(), Some(4));
    }

    #[test]
    fn fit_pools_ties_like_sklearn() {
        // u = 0.5 appears twice with e = 1 and e = 0: pooled target 0.5.
        let g = IsotonicCalibrator::fit(&[0.5, 0.1, 0.5, 0.9], &[1.0, 0.0, 0.0, 1.0]).unwrap();
        assert_eq!(g.x(), &[0.1, 0.5, 0.9]);
        assert_eq!(g.y(), &[0.0, 0.5, 1.0]);
    }

    #[test]
    fn fit_pools_near_ties() {
        // 0.5 and 0.5 + 1 ulp are closer than TIE_TOLERANCE: one knot.
        let next = f64::from_bits(0.5f64.to_bits() + 1);
        let g = IsotonicCalibrator::fit(&[0.1, 0.5, next, 0.9], &[0.0, 1.0, 0.0, 1.0]).unwrap();
        assert_eq!(g.x(), &[0.1, 0.5, 0.9]);
        assert_eq!(g.y(), &[0.0, 0.5, 1.0]);
    }

    #[test]
    fn fit_trims_flat_interior_knots() {
        let g = IsotonicCalibrator::fit(&[0.1, 0.2, 0.3, 0.4, 0.5], &[0.0, 0.0, 0.0, 0.0, 1.0])
            .unwrap();
        assert_eq!(g.x(), &[0.1, 0.4, 0.5]);
        assert_eq!(g.y(), &[0.0, 0.0, 1.0]);
        assert_eq!(g.predict(0.3).unwrap(), 0.0);
    }

    #[test]
    fn fit_drops_zero_weights() {
        let g =
            IsotonicCalibrator::fit_weighted(&[0.0, 0.5, 1.0], &[1.0, 0.0, 1.0], &[0.0, 1.0, 1.0])
                .unwrap();
        assert_eq!(g.x(), &[0.5, 1.0]);
        assert_eq!(g.y(), &[0.0, 1.0]);
        assert_eq!(g.n_samples(), Some(2));
        assert_eq!(g.predict(-1.0).unwrap(), 0.0);
        assert_eq!(g.predict(0.75).unwrap(), 0.5);
        assert!(matches!(
            IsotonicCalibrator::fit_weighted(&[0.1], &[1.0], &[0.0]),
            Err(UcciError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn fit_errors() {
        let err = |u: &[f64], e: &[f64]| IsotonicCalibrator::fit(u, e).unwrap_err();
        assert!(matches!(err(&[], &[]), UcciError::Empty { .. }));
        assert!(matches!(err(&[0.1], &[]), UcciError::Empty { .. }));
        assert!(matches!(
            err(&[0.1, 0.2], &[1.0]),
            UcciError::LengthMismatch { .. }
        ));
        assert!(matches!(
            err(&[0.1, f64::NAN], &[1.0, 0.0]),
            UcciError::NonFinite { .. }
        ));
        assert!(matches!(
            err(&[0.1, 0.2], &[1.0, f64::INFINITY]),
            UcciError::NonFinite { .. }
        ));
        assert!(matches!(
            err(&[0.1, 0.2], &[1.0, 2.0]),
            UcciError::OutOfRange { index: 1, .. }
        ));
        assert!(matches!(
            IsotonicCalibrator::fit_weighted(&[0.1, 0.2], &[1.0, 0.0], &[1.0, -1.0]),
            Err(UcciError::OutOfRange { .. })
        ));
        assert!(matches!(
            IsotonicCalibrator::fit_weighted(&[0.1, 0.2], &[1.0, 0.0], &[1.0]),
            Err(UcciError::LengthMismatch { .. })
        ));
    }

    #[test]
    fn single_knot_is_constant() {
        let g = IsotonicCalibrator::fit(&[0.3, 0.3, 0.3], &[1.0, 0.0, 0.0]).unwrap();
        assert_eq!(g.n_knots(), 1);
        for v in [-5.0, 0.3, 7.0] {
            assert_eq!(g.predict(v).unwrap(), 1.0 / 3.0);
        }
    }

    #[test]
    fn predict_matches_numpy_interp_rules() {
        let g = IsotonicCalibrator::from_knots(vec![0.0, 0.5, 1.0], vec![0.0, 0.2, 1.0]).unwrap();
        assert_eq!(g.predict(-0.1).unwrap(), 0.0);
        assert_eq!(g.predict(0.0).unwrap(), 0.0);
        assert_eq!(g.predict(0.5).unwrap(), 0.2);
        assert_eq!(g.predict(1.0).unwrap(), 1.0);
        assert_eq!(g.predict(1.5).unwrap(), 1.0);
        assert!((g.predict(0.25).unwrap() - 0.1).abs() < 1e-15);
        assert!((g.predict(0.75).unwrap() - 0.6).abs() < 1e-15);
        assert!(g.predict(f64::NAN).is_err());
        assert!(g.predict(f64::INFINITY).is_err());
        assert!(matches!(
            g.predict_many(&[0.1, f64::NEG_INFINITY]),
            Err(UcciError::NonFinite { index: 1, .. })
        ));
    }

    #[test]
    fn from_knots_validation() {
        let bad = |x: Vec<f64>, y: Vec<f64>| IsotonicCalibrator::from_knots(x, y).is_err();
        assert!(bad(vec![], vec![]));
        assert!(bad(vec![0.1], vec![0.1, 0.2]));
        assert!(bad(vec![0.1, 0.1], vec![0.1, 0.2]));
        assert!(bad(vec![0.2, 0.1], vec![0.1, 0.2]));
        assert!(bad(vec![0.1, 0.2], vec![0.3, 0.2]));
        assert!(bad(vec![0.1, f64::NAN], vec![0.1, 0.2]));
        assert!(bad(vec![0.1, 0.2], vec![0.1, f64::INFINITY]));
        assert!(bad(vec![0.1, 0.2], vec![0.1, 1.2]));
        assert!(bad(vec![0.1, 0.2], vec![-0.1, 0.2]));
        assert!(!bad(vec![0.1, 0.2], vec![0.2, 0.2]));
        assert!(!bad(vec![-3.0], vec![1.0]));
        let g = IsotonicCalibrator::from_knots(vec![0.1, 0.2], vec![0.2, 0.3]).unwrap();
        assert_eq!(g.n_samples(), None);
        assert_eq!(g.into_knots(), (vec![0.1, 0.2], vec![0.2, 0.3]));
    }
}
