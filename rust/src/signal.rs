//! Token-margin uncertainty u(x) (paper Section 4.1, Eq. 4).
//!
//! For a greedy generation of `T` tokens with top-1 and top-2 next-token
//! probabilities `p_{t,1} >= p_{t,2}`, the per-token margin is
//!
//! ```text
//! m_t = p_{t,1} - p_{t,2}    in [0, 1]
//! ```
//!
//! and the query-level uncertainty is
//!
//! ```text
//! u(x) = 1 - (1/T) * sum_{t=1..T} m_t .        (Eq. 4)
//! ```
//!
//! Larger `u(x)` means the small model was less decisive. The paper uses
//! greedy decoding so that `p_{t,1}` and `p_{t,2}` are well defined at every
//! position. Serving stacks already return the top-k log-probabilities of
//! every generated token, so the signal costs nothing beyond the small
//! model's own call.
//!
//! # Token convention
//!
//! Count every generated content token and exclude padding and the
//! terminating EOS or stop token (API servers do not report a log-probability
//! for it). This is an implementation choice shared with the Python package;
//! the paper does not state it. Every function here treats its input as the
//! content tokens only.
//!
//! # Inputs
//!
//! * `(p1, p2)` probability pairs: [`token_margin_uncertainty`];
//! * aligned arrays of top-1 and top-2 probabilities: [`uncertainty_from_probs`];
//! * aligned arrays of top-1 and top-2 log-probabilities: [`uncertainty_from_logprobs`];
//! * the top-k candidate log-probabilities of each position (what an
//!   OpenAI-compatible server returns for `top_logprobs`):
//!   [`top2_from_logprobs`] and [`uncertainty_from_top_logprobs`];
//! * one token at a time while generating: [`MarginAccumulator`].
//!
//! Probabilities may exceed 1 by at most [`PROBABILITY_TOLERANCE`] (`exp`
//! round-off); such values are capped at 1 before the margin is taken. The two
//! values of a pair may be given in either order: the larger is the top-1.
//! Means use numpy's pairwise summation, so every result agrees with the
//! Python package bit for bit.
//!
//! # Example
//!
//! ```
//! use ucci::signal::{token_margin_uncertainty, uncertainty_from_top_logprobs};
//!
//! // Margins 0.85, 0.30 and 0.00 give u = 1 - 1.15 / 3.
//! let u = token_margin_uncertainty(&[(0.9, 0.05), (0.6, 0.3), (0.5, 0.5)])?;
//! assert!((u - (1.0 - 1.15 / 3.0)).abs() < 1e-12);
//!
//! // The same kind of signal from per-token candidate log-probabilities (any
//! // order, at least two candidates per token).
//! let per_token = vec![
//!     vec![(0.05f64).ln(), (0.9f64).ln()],
//!     vec![(0.6f64).ln(), (0.3f64).ln(), (0.01f64).ln()],
//! ];
//! let u2 = uncertainty_from_top_logprobs(&per_token)?;
//! assert!((u2 - (1.0 - (0.85 + 0.30) / 2.0)).abs() < 1e-12);
//! # Ok::<(), ucci::UcciError>(())
//! ```

use crate::error::{Result, UcciError};
use crate::num::{check_same_len, np_mean};

/// Slack allowed above 1 for a top-1 probability, to absorb rounding in
/// `exp(logprob)` for a log-probability of (almost) exactly 0.
///
/// Probabilities in `(1, 1 + PROBABILITY_TOLERANCE]` are accepted and capped
/// at 1 when forming the margin, as in the Python package (`PROB_ATOL`).
pub const PROBABILITY_TOLERANCE: f64 = 1e-9;

/// Largest accepted log-probability, `ln(1 + PROBABILITY_TOLERANCE)`.
///
/// Written out as the exact value of Python's `math.log1p(1e-9)` so that the
/// boundary is the same on every platform.
pub const MAX_LOGPROB: f64 = 9.999999995e-10;

const EMPTY_GENERATION: &str =
    "generation (u(x) needs at least one generated content token; Eq. 4 averages over T >= 1)";

/// Per-token margin `m_t = p_{t,1} - p_{t,2}` (Section 4.1).
///
/// The two probabilities may be given in either order; the larger one is
/// taken as the top-1 probability and capped at 1. The result lies in
/// `[0, 1]`.
///
/// # Errors
///
/// [`UcciError::OutOfRange`] unless `0 <= min(a, b)` and
/// `max(a, b) <= 1 + PROBABILITY_TOLERANCE`; NaN is rejected.
///
/// # Example
///
/// ```
/// use ucci::signal::token_margin;
///
/// assert!((token_margin(0.6, 0.3)? - 0.3).abs() < 1e-15);
/// assert_eq!(token_margin(0.3, 0.6)?, token_margin(0.6, 0.3)?);
/// assert!(token_margin(1.2, 0.1).is_err());
/// # Ok::<(), ucci::UcciError>(())
/// ```
pub fn token_margin(a: f64, b: f64) -> Result<f64> {
    margin_at(a, b, 0)
}

fn in_prob_range(p: f64) -> bool {
    (0.0..=1.0 + PROBABILITY_TOLERANCE).contains(&p)
}

fn margin_at(a: f64, b: f64, index: usize) -> Result<f64> {
    let (p1, p2) = if a >= b { (a, b) } else { (b, a) };
    // Every comparison is false for NaN, so NaN is rejected here.
    let valid = 0.0 <= p2 && p2 <= p1 && p1 <= 1.0 + PROBABILITY_TOLERANCE;
    if !valid {
        let value = if in_prob_range(a) { b } else { a };
        return Err(UcciError::OutOfRange {
            what: "top-2 probability",
            index,
            value,
            expected: "in [0, 1]",
        });
    }
    Ok(p1.min(1.0) - p2)
}

/// Per-token margins `m_t = p_{t,1} - p_{t,2}` from `(p1, p2)` probability
/// pairs (Section 4.1).
///
/// # Errors
///
/// [`UcciError::OutOfRange`] if any probability lies outside `[0, 1]` (see
/// [`token_margin`]); the error's `index` is the token position.
pub fn margins_from_top2(pairs: &[(f64, f64)]) -> Result<Vec<f64>> {
    pairs
        .iter()
        .enumerate()
        .map(|(t, &(a, b))| margin_at(a, b, t))
        .collect()
}

/// `u(x) = 1 - mean(m_t)` (Section 4.1, Eq. 4).
///
/// # Errors
///
/// [`UcciError::Empty`] for an empty generation, [`UcciError::NonFinite`] for
/// a NaN or infinite margin and [`UcciError::OutOfRange`] for a margin outside
/// `[0, 1]`.
pub fn uncertainty_from_margins(margins: &[f64]) -> Result<f64> {
    if margins.is_empty() {
        return Err(UcciError::Empty {
            what: EMPTY_GENERATION,
        });
    }
    crate::num::check_finite(margins, "margins")?;
    if let Some(index) = margins.iter().position(|m| !(0.0..=1.0).contains(m)) {
        return Err(UcciError::OutOfRange {
            what: "margins",
            index,
            value: margins[index],
            expected: "in [0, 1]",
        });
    }
    Ok(1.0 - np_mean(margins))
}

/// `u(x)` from per-token `(p1, p2)` probability pairs (Section 4.1, Eq. 4).
///
/// # Errors
///
/// [`UcciError::Empty`] for an empty generation and
/// [`UcciError::OutOfRange`] for a probability outside `[0, 1]`.
pub fn token_margin_uncertainty(pairs: &[(f64, f64)]) -> Result<f64> {
    uncertainty_from_margins(&margins_from_top2(pairs)?)
}

/// `u(x)` for one sequence from aligned arrays of top-1 and top-2
/// probabilities (Section 4.1, Eq. 4).
///
/// Swapped entries are allowed: the larger value at each position is the
/// top-1.
///
/// # Errors
///
/// [`UcciError::OutOfRange`] for a value outside `[0, 1]` (NaN included),
/// [`UcciError::LengthMismatch`] if the arrays differ in length and
/// [`UcciError::Empty`] if they are empty.
pub fn uncertainty_from_probs(p1: &[f64], p2: &[f64]) -> Result<f64> {
    check_probabilities(p1, "p1")?;
    check_probabilities(p2, "p2")?;
    check_same_len(p1.len(), p2.len(), "p1 and p2")?;
    if p1.is_empty() {
        return Err(UcciError::Empty {
            what: EMPTY_GENERATION,
        });
    }
    let margins: Vec<f64> = p1
        .iter()
        .zip(p2)
        .map(|(&a, &b)| a.max(b).min(1.0) - a.min(b))
        .collect();
    Ok(1.0 - np_mean(&margins))
}

/// `u(x)` for one sequence from aligned arrays of top-1 and top-2
/// log-probabilities (natural log; Section 4.1, Eq. 4).
///
/// `-inf` (probability 0) is allowed.
///
/// # Errors
///
/// [`UcciError::LengthMismatch`] if the arrays differ in length,
/// [`UcciError::Empty`] if they are empty, [`UcciError::NonFinite`] for NaN and
/// [`UcciError::OutOfRange`] for a value above [`MAX_LOGPROB`] (`+inf`
/// included).
pub fn uncertainty_from_logprobs(lp1: &[f64], lp2: &[f64]) -> Result<f64> {
    check_same_len(lp1.len(), lp2.len(), "lp1 and lp2")?;
    if lp1.is_empty() {
        return Err(UcciError::Empty {
            what: EMPTY_GENERATION,
        });
    }
    check_logprobs(lp1, "lp1")?;
    check_logprobs(lp2, "lp2")?;
    let margins: Vec<f64> = lp1
        .iter()
        .zip(lp2)
        .map(|(&x, &y)| {
            let (a, b) = (x.exp(), y.exp());
            a.max(b).min(1.0) - a.min(b)
        })
        .collect();
    Ok(1.0 - np_mean(&margins))
}

fn check_probabilities(p: &[f64], what: &'static str) -> Result<()> {
    match p.iter().position(|&v| !in_prob_range(v)) {
        Some(index) => Err(UcciError::OutOfRange {
            what,
            index,
            value: p[index],
            expected: "in [0, 1]",
        }),
        None => Ok(()),
    }
}

fn check_logprobs(lp: &[f64], what: &'static str) -> Result<()> {
    for (index, &v) in lp.iter().enumerate() {
        if v.is_nan() {
            return Err(UcciError::NonFinite { what, index });
        }
        if v > MAX_LOGPROB {
            return Err(UcciError::OutOfRange {
                what,
                index,
                value: v,
                expected: "a log-probability (<= 0)",
            });
        }
    }
    Ok(())
}

/// Top-2 probabilities `(p1, p2)`, with `p1 >= p2`, at one token position from
/// that position's candidate log-probabilities (any order, at least two).
///
/// Extra candidates are fine (vLLM adds the sampled token to the top-k).
/// `-inf` (probability 0) is allowed.
///
/// # Errors
///
/// [`UcciError::TooFewCandidates`] for fewer than two candidates,
/// [`UcciError::NonFinite`] for a NaN log-probability and
/// [`UcciError::OutOfRange`] if the largest is above [`MAX_LOGPROB`].
pub fn top2_from_candidates(candidates: &[f64]) -> Result<(f64, f64)> {
    top2_at(candidates, 0)
}

fn top2_at(candidates: &[f64], token: usize) -> Result<(f64, f64)> {
    if candidates.len() < 2 {
        return Err(UcciError::TooFewCandidates {
            token,
            found: candidates.len(),
        });
    }
    let mut first = f64::NEG_INFINITY;
    let mut second = f64::NEG_INFINITY;
    for &lp in candidates {
        if lp.is_nan() {
            return Err(UcciError::NonFinite {
                what: "log-probabilities",
                index: token,
            });
        }
        if lp > first {
            second = first;
            first = lp;
        } else if lp > second {
            second = lp;
        }
    }
    if first > MAX_LOGPROB {
        return Err(UcciError::OutOfRange {
            what: "log-probabilities",
            index: token,
            value: first,
            expected: "a log-probability (<= 0)",
        });
    }
    Ok((first.exp(), second.exp()))
}

/// Top-2 probabilities per token from each position's candidate
/// log-probabilities (Section 4.1).
///
/// Each element of `per_token` holds the natural-log probabilities of the
/// top-k candidates at one content position, `k >= 2`, in any order. This is
/// what an OpenAI-compatible server returns in `top_logprobs` when asked for
/// `logprobs=true, top_logprobs=2`, and what vLLM returns for
/// `SamplingParams(temperature=0, logprobs=2)`.
///
/// # Errors
///
/// As [`top2_from_candidates`]; the error's `token` or `index` names the
/// position.
pub fn top2_from_logprobs<T: AsRef<[f64]>>(per_token: &[T]) -> Result<Vec<(f64, f64)>> {
    per_token
        .iter()
        .enumerate()
        .map(|(t, cands)| top2_at(cands.as_ref(), t))
        .collect()
}

/// `u(x)` from per-token candidate log-probabilities (Section 4.1, Eq. 4).
///
/// Equivalent to `token_margin_uncertainty(&top2_from_logprobs(per_token)?)`,
/// which is what the Python package's `from_openai_logprobs` and
/// `from_vllm_logprobs` compute.
///
/// # Errors
///
/// Everything [`top2_from_logprobs`] and [`token_margin_uncertainty`] can
/// return.
pub fn uncertainty_from_top_logprobs<T: AsRef<[f64]>>(per_token: &[T]) -> Result<f64> {
    token_margin_uncertainty(&top2_from_logprobs(per_token)?)
}

/// `u(x)` from the `choice.logprobs.content` array of an OpenAI-compatible
/// chat completion (Section 4.1, Eq. 4).
///
/// Each element must carry a `top_logprobs` array of objects with a numeric
/// `logprob` field; request it with `logprobs: true, top_logprobs: 2` (or
/// more) and temperature 0. vLLM's OpenAI-compatible server accepts the same
/// arguments. Pass content tokens only (see the module documentation).
///
/// # Errors
///
/// [`UcciError::InvalidParameter`] if `content` is not an array or a token
/// lacks `top_logprobs` or a numeric `logprob`, plus everything
/// [`uncertainty_from_top_logprobs`] can return.
///
/// # Example
///
/// ```
/// let content = serde_json::json!([
///     {"token": "{", "logprob": -0.01,
///      "top_logprobs": [{"token": "{", "logprob": -0.01}, {"token": " {", "logprob": -4.7}]},
///     {"token": "\"", "logprob": -0.2,
///      "top_logprobs": [{"token": "\"", "logprob": -0.2}, {"token": "}", "logprob": -1.8}]}
/// ]);
/// let u = ucci::signal::from_openai_logprobs(&content)?;
/// assert!(u > 0.0 && u < 1.0);
/// # Ok::<(), ucci::UcciError>(())
/// ```
#[cfg(feature = "json")]
#[cfg_attr(docsrs, doc(cfg(feature = "json")))]
pub fn from_openai_logprobs(content: &serde_json::Value) -> Result<f64> {
    let invalid = |reason: String| UcciError::InvalidParameter {
        name: "content",
        reason,
    };
    let tokens = content.as_array().ok_or_else(|| {
        invalid("expected the array choice.logprobs.content of a chat completion".to_string())
    })?;
    let mut per_token: Vec<Vec<f64>> = Vec::with_capacity(tokens.len());
    for (t, tok) in tokens.iter().enumerate() {
        let top = match tok.get("top_logprobs") {
            Some(serde_json::Value::Array(top)) => top,
            _ => {
                return Err(invalid(format!(
                    "token {t}: top_logprobs is missing; request top_logprobs=2"
                )))
            }
        };
        let mut cands = Vec::with_capacity(top.len());
        for c in top {
            let lp = c
                .get("logprob")
                .and_then(serde_json::Value::as_f64)
                .ok_or_else(|| invalid(format!("token {t}: a candidate has no numeric logprob")))?;
            cands.push(lp);
        }
        per_token.push(cands);
    }
    uncertainty_from_top_logprobs(&per_token)
}

/// Streaming computation of `u(x)` while tokens are generated.
///
/// Push one content token at a time (as probabilities or as candidate
/// log-probabilities) and read [`MarginAccumulator::uncertainty`] when the
/// generation ends. The result is bit-identical to [`token_margin_uncertainty`]
/// on the same tokens. A failed push leaves the accumulator unchanged.
///
/// # Example
///
/// ```
/// use ucci::signal::MarginAccumulator;
///
/// let mut acc = MarginAccumulator::new();
/// acc.push_top2(0.9, 0.05)?;
/// acc.push_logprobs(&[(0.3f64).ln(), (0.6f64).ln(), (0.05f64).ln()])?;
/// assert_eq!(acc.len(), 2);
/// assert!((acc.uncertainty()? - (1.0 - (0.85 + 0.30) / 2.0)).abs() < 1e-12);
/// # Ok::<(), ucci::UcciError>(())
/// ```
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MarginAccumulator {
    margins: Vec<f64>,
}

impl MarginAccumulator {
    /// An accumulator with no tokens.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds one token from its top-1 and top-2 probabilities (either order)
    /// and returns that token's margin.
    ///
    /// # Errors
    ///
    /// [`UcciError::OutOfRange`] for a probability outside `[0, 1]`.
    pub fn push_top2(&mut self, a: f64, b: f64) -> Result<f64> {
        let m = margin_at(a, b, self.margins.len())?;
        self.margins.push(m);
        Ok(m)
    }

    /// Adds one token from its candidate log-probabilities (any order, at
    /// least two) and returns that token's margin.
    ///
    /// # Errors
    ///
    /// As [`top2_from_candidates`] and [`token_margin`].
    pub fn push_logprobs(&mut self, candidates: &[f64]) -> Result<f64> {
        let (p1, p2) = top2_at(candidates, self.margins.len())?;
        self.push_top2(p1, p2)
    }

    /// Number of tokens pushed so far (`T` in Eq. 4).
    pub fn len(&self) -> usize {
        self.margins.len()
    }

    /// True when no token has been pushed.
    pub fn is_empty(&self) -> bool {
        self.margins.is_empty()
    }

    /// Per-token margins pushed so far.
    pub fn margins(&self) -> &[f64] {
        &self.margins
    }

    /// Forgets every token, keeping the allocation for the next generation.
    pub fn clear(&mut self) {
        self.margins.clear();
    }

    /// `u(x) = 1 - (1/T) sum_t m_t` over the tokens pushed so far.
    ///
    /// # Errors
    ///
    /// [`UcciError::Empty`] when no token has been pushed.
    pub fn uncertainty(&self) -> Result<f64> {
        uncertainty_from_margins(&self.margins)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-12
    }

    #[test]
    fn matches_eq4() {
        let u = token_margin_uncertainty(&[(0.9, 0.05), (0.6, 0.3), (0.5, 0.5)]).unwrap();
        assert!(close(u, 1.0 - (0.85 + 0.3 + 0.0) / 3.0));
    }

    #[test]
    fn max_logprob_constant_is_log1p_of_tolerance() {
        assert!((PROBABILITY_TOLERANCE.ln_1p() - MAX_LOGPROB).abs() <= f64::EPSILON * 1e-9);
        assert!(MAX_LOGPROB.exp() <= 1.0 + PROBABILITY_TOLERANCE + f64::EPSILON);
    }

    #[test]
    fn pair_order_does_not_matter() {
        assert!(close(token_margin_uncertainty(&[(0.1, 0.8)]).unwrap(), 0.3));
        assert_eq!(
            margins_from_top2(&[(0.1, 0.8), (0.8, 0.1)]).unwrap(),
            vec![0.8 - 0.1, 0.8 - 0.1]
        );
    }

    #[test]
    fn extremes() {
        assert_eq!(token_margin_uncertainty(&[(1.0, 0.0); 4]).unwrap(), 0.0);
        assert_eq!(token_margin_uncertainty(&[(0.4, 0.4); 4]).unwrap(), 1.0);
    }

    #[test]
    fn tolerance_above_one_is_capped() {
        assert_eq!(token_margin(1.0 + 5e-10, 0.0).unwrap(), 1.0);
        assert!(token_margin(1.0 + 1e-8, 0.0).is_err());
        assert_eq!(uncertainty_from_probs(&[1.0 + 5e-10], &[0.0]).unwrap(), 0.0);
    }

    #[test]
    fn rejects_bad_probabilities_with_position() {
        assert!(token_margin(-0.1, 0.5).is_err());
        assert!(token_margin(f64::NAN, 0.5).is_err());
        assert!(token_margin(0.5, f64::NAN).is_err());
        assert!(token_margin(f64::INFINITY, 0.5).is_err());
        match margins_from_top2(&[(0.5, 0.4), (1.5, 0.1)]) {
            Err(UcciError::OutOfRange { index, value, .. }) => {
                assert_eq!(index, 1);
                assert_eq!(value, 1.5);
            }
            other => panic!("unexpected {other:?}"),
        }
        match margins_from_top2(&[(0.5, -0.25)]) {
            Err(UcciError::OutOfRange { value, .. }) => assert_eq!(value, -0.25),
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn empty_generation() {
        assert!(matches!(
            token_margin_uncertainty(&[]),
            Err(UcciError::Empty { .. })
        ));
        assert!(matches!(
            uncertainty_from_margins(&[]),
            Err(UcciError::Empty { .. })
        ));
        assert!(uncertainty_from_probs(&[], &[]).is_err());
        assert!(uncertainty_from_logprobs(&[], &[]).is_err());
        let empty: Vec<Vec<f64>> = vec![];
        assert!(uncertainty_from_top_logprobs(&empty).is_err());
        assert!(MarginAccumulator::new().uncertainty().is_err());
        assert!(MarginAccumulator::new().is_empty());
    }

    #[test]
    fn margins_must_be_valid() {
        assert!(uncertainty_from_margins(&[0.5, f64::NAN]).is_err());
        assert!(uncertainty_from_margins(&[0.5, 1.5]).is_err());
        assert!(uncertainty_from_margins(&[-0.1]).is_err());
        assert_eq!(uncertainty_from_margins(&[0.0, 1.0]).unwrap(), 0.5);
    }

    #[test]
    fn array_forms_agree_with_pairs() {
        let pairs = [(0.8, 0.1), (0.2, 0.55), (0.99, 0.005), (0.5, 0.5)];
        let p1: Vec<f64> = pairs.iter().map(|p| p.0).collect();
        let p2: Vec<f64> = pairs.iter().map(|p| p.1).collect();
        let a = token_margin_uncertainty(&pairs).unwrap();
        assert_eq!(uncertainty_from_probs(&p1, &p2).unwrap(), a);
        let l1: Vec<f64> = p1.iter().map(|p| p.ln()).collect();
        let l2: Vec<f64> = p2.iter().map(|p| p.ln()).collect();
        assert!(close(uncertainty_from_logprobs(&l1, &l2).unwrap(), a));
    }

    #[test]
    fn array_form_errors() {
        assert!(matches!(
            uncertainty_from_probs(&[0.5], &[0.2, 0.1]),
            Err(UcciError::LengthMismatch { .. })
        ));
        assert!(matches!(
            uncertainty_from_probs(&[0.5, f64::NAN], &[0.2, 0.1]),
            Err(UcciError::OutOfRange { index: 1, .. })
        ));
        assert!(matches!(
            uncertainty_from_logprobs(&[-0.1, f64::NAN], &[-2.0, -3.0]),
            Err(UcciError::NonFinite { index: 1, .. })
        ));
        assert!(matches!(
            uncertainty_from_logprobs(&[0.1], &[-2.0]),
            Err(UcciError::OutOfRange { .. })
        ));
        assert!(matches!(
            uncertainty_from_logprobs(&[f64::INFINITY], &[-2.0]),
            Err(UcciError::OutOfRange { .. })
        ));
        assert_eq!(
            uncertainty_from_logprobs(&[0.0], &[f64::NEG_INFINITY]).unwrap(),
            0.0
        );
    }

    #[test]
    fn top2_from_logprobs_sorts_and_needs_two() {
        let out = top2_from_logprobs(&[vec![0.2f64.ln(), 0.7f64.ln(), 0.05f64.ln()]]).unwrap();
        assert!(close(out[0].0, 0.7) && close(out[0].1, 0.2));
        match top2_from_logprobs(&[vec![0.0, -1.0], vec![0.9f64.ln()]]) {
            Err(UcciError::TooFewCandidates { token, found }) => {
                assert_eq!((token, found), (1, 1));
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn candidate_edge_cases() {
        let out = top2_from_candidates(&[-0.5, -0.5, -3.0]).unwrap();
        assert_eq!(out.0, out.1);
        assert_eq!(
            top2_from_candidates(&[f64::NEG_INFINITY, 0.0]).unwrap(),
            (1.0, 0.0)
        );
        assert_eq!(
            top2_from_candidates(&[f64::NEG_INFINITY, f64::NEG_INFINITY]).unwrap(),
            (0.0, 0.0)
        );
        assert!(top2_from_candidates(&[f64::NAN, 0.0]).is_err());
        assert!(top2_from_candidates(&[-0.1, -0.2, f64::NAN]).is_err());
        assert!(top2_from_candidates(&[f64::INFINITY, 0.0]).is_err());
        assert!(top2_from_candidates(&[0.5, -1.0]).is_err());
        assert!(top2_from_candidates(&[MAX_LOGPROB, -1.0]).is_ok());
        assert!(top2_from_candidates(&[]).is_err());
    }

    #[test]
    fn accumulator_matches_batch_and_is_unchanged_by_errors() {
        let pairs = [(0.9, 0.1), (0.35, 0.6), (0.7, 0.2)];
        let mut acc = MarginAccumulator::new();
        for &(a, b) in &pairs {
            acc.push_top2(a, b).unwrap();
        }
        assert!(acc.push_top2(2.0, 0.1).is_err());
        assert!(acc.push_logprobs(&[0.0]).is_err());
        assert_eq!(acc.len(), 3);
        assert_eq!(
            acc.uncertainty().unwrap(),
            token_margin_uncertainty(&pairs).unwrap()
        );
        assert_eq!(acc.margins().len(), 3);
        acc.clear();
        assert!(acc.is_empty());
    }

    #[test]
    fn logprob_paths_agree() {
        let probs: [(f64, f64); 3] = [(0.8, 0.1), (0.55, 0.4), (0.99, 0.005)];
        let lps: Vec<Vec<f64>> = probs.iter().map(|&(a, b)| vec![b.ln(), a.ln()]).collect();
        let a = token_margin_uncertainty(&probs).unwrap();
        let b = uncertainty_from_top_logprobs(&lps).unwrap();
        let c = token_margin_uncertainty(&top2_from_logprobs(&lps).unwrap()).unwrap();
        assert!(close(a, b));
        assert_eq!(b, c);
    }

    #[cfg(feature = "json")]
    #[test]
    fn openai_content_parsing() {
        let content = serde_json::json!([
            {"top_logprobs": [{"logprob": 0.8f64.ln()}, {"logprob": 0.1f64.ln()}]},
            {"top_logprobs": [{"logprob": 0.55f64.ln()}, {"logprob": 0.4f64.ln()}]}
        ]);
        let u = from_openai_logprobs(&content).unwrap();
        assert!(close(u, 1.0 - (0.7 + 0.15) / 2.0));
        assert!(from_openai_logprobs(&serde_json::json!({"a": 1})).is_err());
        assert!(from_openai_logprobs(&serde_json::json!([{"top_logprobs": null}])).is_err());
        assert!(from_openai_logprobs(&serde_json::json!([{"logprob": -0.1}])).is_err());
        assert!(from_openai_logprobs(&serde_json::json!([
            {"top_logprobs": [{"logprob": "x"}, {"logprob": -1.0}]}
        ]))
        .is_err());
        assert!(from_openai_logprobs(&serde_json::json!([])).is_err());
    }
}
