//! The error type shared by every module of the crate.

use std::fmt;

/// Result alias used by every fallible function in this crate.
pub type Result<T> = std::result::Result<T, UcciError>;

/// Everything that can go wrong when computing, calibrating or applying a
/// UCCI routing policy.
///
/// The variants mirror the `ValueError`s raised by the Python reference
/// package, so a call that fails in Python fails here too. The golden tests in
/// `tests/golden.rs` check this correspondence case by case.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum UcciError {
    /// An input that needs at least one element was empty (for example a
    /// generation with no tokens, Section 4.1, or an empty calibration set,
    /// Section 4.2).
    Empty {
        /// Which input was empty.
        what: &'static str,
    },
    /// Two inputs that must be aligned element by element had different
    /// lengths.
    LengthMismatch {
        /// Which pair of inputs disagreed.
        what: &'static str,
        /// Length of the first input.
        left: usize,
        /// Length of the second input.
        right: usize,
    },
    /// A value that must be finite was NaN or infinite.
    NonFinite {
        /// Which input held the value.
        what: &'static str,
        /// Position of the offending element (0 for scalar arguments).
        index: usize,
    },
    /// A value lay outside its admissible range.
    OutOfRange {
        /// Which input held the value.
        what: &'static str,
        /// Position of the offending element (0 for scalar arguments).
        index: usize,
        /// The offending value.
        value: f64,
        /// Human-readable description of the admissible range.
        expected: &'static str,
    },
    /// A token position reported fewer than two candidate log-probabilities,
    /// so the top-1 minus top-2 margin of Section 4.1 is undefined.
    TooFewCandidates {
        /// Zero-based position of the token in the generation.
        token: usize,
        /// Number of candidates that were supplied.
        found: usize,
    },
    /// A scalar parameter (costs, target accuracy, budget, grid, number of
    /// bins) was invalid.
    InvalidParameter {
        /// Name of the parameter.
        name: &'static str,
        /// Why it was rejected.
        reason: String,
    },
    /// No threshold on the grid reaches the accuracy target `tau` on the
    /// validation set (Section 4.3, Eq. 7 has no feasible point).
    Infeasible {
        /// The requested accuracy target.
        tau: f64,
        /// The best validation accuracy reached by any threshold on the grid.
        best_accuracy: f64,
        /// The (smallest) grid threshold that reaches `best_accuracy`.
        best_theta: f64,
    },
    /// No threshold on the grid meets the cost budget (the matched-budget
    /// form used for the bottom block of Table 2 has no feasible point).
    OverBudget {
        /// The requested mean cost budget per query.
        budget: f64,
        /// The lowest mean cost reached by any threshold on the grid.
        min_cost: f64,
        /// The (smallest) grid threshold that reaches `min_cost`.
        cheapest_theta: f64,
    },
    /// Calibration knots do not describe a valid non-decreasing map
    /// (Section 4.2).
    InvalidCalibrator {
        /// Why the knots were rejected.
        reason: String,
    },
    /// A serialized router could not be parsed or failed validation.
    InvalidRouter {
        /// Why the router was rejected.
        reason: String,
    },
}

impl fmt::Display for UcciError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UcciError::Empty { what } => write!(f, "{what} must not be empty"),
            UcciError::LengthMismatch { what, left, right } => {
                write!(
                    f,
                    "{what} must have the same length, got {left} and {right}"
                )
            }
            UcciError::NonFinite { what, index } => {
                write!(f, "{what} contains NaN or inf (at index {index})")
            }
            UcciError::OutOfRange {
                what,
                index,
                value,
                expected,
            } => write!(
                f,
                "{what} must be {expected}, got {value} (at index {index})"
            ),
            UcciError::TooFewCandidates { token, found } => write!(
                f,
                "token {token}: need the top-2 log-probs, got {found}; \
                 request top_logprobs >= 2 (OpenAI API) or logprobs >= 2 (vLLM)"
            ),
            UcciError::InvalidParameter { name, reason } => write!(f, "invalid {name}: {reason}"),
            UcciError::Infeasible {
                tau,
                best_accuracy,
                best_theta,
            } => write!(
                f,
                "no threshold on the grid reaches tau={tau}; the best accuracy on the grid is \
                 {best_accuracy} (theta={best_theta}). Lower tau or check that the large model \
                 beats the small one on this split."
            ),
            UcciError::OverBudget {
                budget,
                min_cost,
                cheapest_theta,
            } => write!(
                f,
                "no threshold on the grid has cost <= budget={budget}; the cheapest costs \
                 {min_cost} (theta={cheapest_theta})"
            ),
            UcciError::InvalidCalibrator { reason } => write!(f, "invalid calibrator: {reason}"),
            UcciError::InvalidRouter { reason } => write!(f, "invalid router: {reason}"),
        }
    }
}

impl std::error::Error for UcciError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_messages_are_informative() {
        let e = UcciError::Infeasible {
            tau: 0.95,
            best_accuracy: 0.9312,
            best_theta: 0.1,
        };
        assert!(e
            .to_string()
            .starts_with("no threshold on the grid reaches tau=0.95; the best accuracy on the grid is 0.9312 (theta=0.1)"));
        let e = UcciError::OverBudget {
            budget: 1.5,
            min_cost: 2.0,
            cheapest_theta: 1.0,
        };
        assert_eq!(
            e.to_string(),
            "no threshold on the grid has cost <= budget=1.5; the cheapest costs 2 (theta=1)"
        );
        let e = UcciError::TooFewCandidates { token: 3, found: 1 };
        assert!(e.to_string().contains("token 3"));
        let e = UcciError::LengthMismatch {
            what: "u and e",
            left: 2,
            right: 3,
        };
        assert_eq!(
            e.to_string(),
            "u and e must have the same length, got 2 and 3"
        );
        let e = UcciError::OutOfRange {
            what: "probability",
            index: 1,
            value: 1.5,
            expected: "in [0, 1]",
        };
        assert_eq!(
            e.to_string(),
            "probability must be in [0, 1], got 1.5 (at index 1)"
        );
    }

    #[test]
    fn is_std_error() {
        fn takes_error(_: &dyn std::error::Error) {}
        takes_error(&UcciError::Empty { what: "u" });
    }
}
