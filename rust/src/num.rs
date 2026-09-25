//! Small numeric helpers shared by the public modules (crate private).

use crate::error::{Result, UcciError};

/// numpy's pairwise summation of `n` values produced by `value(i)`, in the
/// same order and with the same blocking as `numpy.add.reduce` on a
/// contiguous float64 array (8 interleaved accumulators on blocks of at most
/// 128 values, recursive halving above that).
///
/// Following numpy's summation order makes the means in this crate agree
/// with the Python reference bit for bit, so tie-breaking and feasibility
/// decisions match between the two implementations.
pub(crate) fn pairwise_sum_by<F: Fn(usize) -> f64>(start: usize, n: usize, value: &F) -> f64 {
    const BLOCK: usize = 128;
    if n < 8 {
        let mut res = 0.0;
        for i in start..start + n {
            res += value(i);
        }
        res
    } else if n <= BLOCK {
        let mut r = [0.0f64; 8];
        for (k, rk) in r.iter_mut().enumerate() {
            *rk = value(start + k);
        }
        let mut i = 8;
        while i < n - (n % 8) {
            for (k, rk) in r.iter_mut().enumerate() {
                *rk += value(start + i + k);
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += value(start + i);
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise_sum_by(start, n2, value) + pairwise_sum_by(start + n2, n - n2, value)
    }
}

/// numpy's pairwise sum of a slice (see [`pairwise_sum_by`]).
pub(crate) fn pairwise_sum(values: &[f64]) -> f64 {
    pairwise_sum_by(0, values.len(), &|i| values[i])
}

/// `numpy.mean` of a non-empty slice.
pub(crate) fn np_mean(values: &[f64]) -> f64 {
    pairwise_sum(values) / values.len() as f64
}

/// One segment of `numpy.add.reduceat`: the first element plus the pairwise
/// sum of the rest. `values` must be non-empty.
pub(crate) fn reduceat_segment(values: &[f64]) -> f64 {
    values[0] + pairwise_sum(&values[1..])
}

/// Error unless every element of `values` is finite.
pub(crate) fn check_finite(values: &[f64], what: &'static str) -> Result<()> {
    match values.iter().position(|v| !v.is_finite()) {
        Some(index) => Err(UcciError::NonFinite { what, index }),
        None => Ok(()),
    }
}

/// Error unless the two slices have the same length.
pub(crate) fn check_same_len(left: usize, right: usize, what: &'static str) -> Result<()> {
    if left == right {
        Ok(())
    } else {
        Err(UcciError::LengthMismatch { what, left, right })
    }
}

/// Error unless the scalar is finite.
pub(crate) fn check_finite_scalar(value: f64, name: &'static str) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(UcciError::InvalidParameter {
            name,
            reason: format!("must be finite, got {value}"),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference values: `repr(np.sum(v))` and `repr(np.add.reduceat(v, [0])[0])`
    /// with `v = np.arange(1, n + 1, dtype=float) / 7`, numpy 2.5.3.
    #[test]
    fn pairwise_sum_follows_numpy_blocking() {
        let v = |n: usize| -> Vec<f64> { (1..=n).map(|i| i as f64 / 7.0).collect() };
        assert_eq!(pairwise_sum(&v(9)), 6.428571428571429);
        assert_eq!(reduceat_segment(&v(9)), 6.42857142857143);
        assert_eq!(pairwise_sum(&v(17)), 21.857142857142858);
        assert_eq!(pairwise_sum(&v(129)), 1197.8571428571427);
        assert_eq!(reduceat_segment(&v(129)), 1197.857142857143);
        assert_eq!(pairwise_sum(&v(300)), 6450.0);
        assert_eq!(pairwise_sum(&v(1000)), 71500.0);
        assert_eq!(pairwise_sum(&[]), 0.0);
        // Association differs from a left-to-right loop: numpy gives 0.6 here.
        assert_eq!(reduceat_segment(&[0.1, 0.2, 0.3]), 0.1 + (0.2 + 0.3));
        assert_eq!(np_mean(&[0.1, 0.2, 0.3]), ((0.0 + 0.1) + 0.2 + 0.3) / 3.0);
    }

    #[test]
    fn checks() {
        assert!(check_finite(&[0.0, 1.0], "x").is_ok());
        assert_eq!(
            check_finite(&[0.0, f64::NAN], "x"),
            Err(UcciError::NonFinite {
                what: "x",
                index: 1
            })
        );
        assert!(check_same_len(2, 2, "a and b").is_ok());
        assert!(check_same_len(2, 3, "a and b").is_err());
        assert!(check_finite_scalar(f64::INFINITY, "tau").is_err());
    }
}
