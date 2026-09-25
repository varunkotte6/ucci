//! # UCCI: calibrated uncertainty for cost-optimal LLM cascade routing
//!
//! Rust implementation of the method in Varun Kotte, "UCCI: Calibrated
//! Uncertainty for Cost-Optimal LLM Cascade Routing", arXiv:2605.18796 (2026),
//! <https://arxiv.org/abs/2605.18796>. It is a port of the Python reference
//! package in the same repository and is tested against golden vectors that
//! the Python package generates.
//!
//! A two-model cascade answers every query with a small model and escalates
//! to a large model only when the small model is likely wrong. UCCI builds the
//! routing policy in three steps (Section 4):
//!
//! 1. [`signal`]: a scalar uncertainty `u(x) = 1 - mean_t (p_{t,1} - p_{t,2})`
//!    from the small model's top-2 token probabilities (Section 4.1, Eq. 4);
//! 2. [`calibration`]: an isotonic map `g` from `u(x)` to the probability that
//!    the small model is wrong, `p_hat(x) = g(u(x))` (Section 4.2, Eq. 5);
//! 3. [`policy`]: escalate when `p_hat(x) > theta`, with `theta` chosen on a
//!    validation set as the cheapest threshold whose accuracy reaches a target
//!    `tau` (Section 4.3, Eqs. 6 and 7).
//!
//! Section 5 (Theorem 1) shows that, under three explicit assumptions, such a
//! threshold policy on the calibrated probability is cost-optimal among
//! policies that depend only on `u(x)`. [`metrics`] provides the diagnostics
//! of Section 6 (ECE, reliability table, Brier score, micro-F1 of routed
//! answers), and [`router`] bundles a fitted calibrator and threshold into a
//! [`Router`]. A router fitted by the Python package loads from its JSON file
//! (feature `json`, on by default), and [`RouterBuilder`] runs the whole
//! protocol of Section 6.1 in Rust: fit `g` on the calibration split, choose
//! `theta` on the validation split, evaluate on the test split.
//!
//! Every fallible function returns [`Result`] with a [`UcciError`] that names
//! the offending input. Inputs are validated like in the Python package:
//! NaN and infinities are rejected, probabilities must lie in `[0, 1]`, costs
//! must be positive.
//!
//! ## Quickstart
//!
//! ```
//! use ucci::{default_grid, select_threshold, Costs, IsotonicCalibrator, Router};
//!
//! // Calibration split: uncertainty u(x) and whether the small model was wrong.
//! let u_cal = [0.05, 0.10, 0.20, 0.30, 0.40, 0.55, 0.70, 0.90];
//! let e_cal = [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0];
//! let g = IsotonicCalibrator::fit(&u_cal, &e_cal)?;
//!
//! // Validation split: both models were run, so each query has a score for
//! // the small and for the large model's answer.
//! let u_val = [0.05, 0.25, 0.35, 0.60, 0.80, 0.95];
//! let small = [1.0, 1.0, 0.0, 0.0, 1.0, 0.0];
//! let large = [1.0, 1.0, 1.0, 1.0, 1.0, 0.0];
//! let p_val = g.predict_many(&u_val)?;
//! let costs = Costs::paper(); // c_small = 1.0, c_large = 3.02 (Section 6.1)
//! let choice = select_threshold(&p_val, &small, &large, 0.8, &costs, &default_grid())?;
//! assert!(choice.accuracy >= 0.8);
//!
//! // Route new queries.
//! let router = Router::new(g, choice.theta, costs)?;
//! assert!(!router.escalate(0.05)?); // confident: keep the small model's answer
//! assert!(router.escalate(0.95)?); // uncertain: escalate to the large model
//! # Ok::<(), ucci::UcciError>(())
//! ```
//!
//! ## Parity with the Python package
//!
//! Every computation follows the Python reference implementation operation by
//! operation, including numpy's summation order and `numpy.interp`, and router
//! files are parsed with exact float round-tripping. The golden test
//! (`tests/golden.rs`) replays every vector that `tools/make_golden.py`
//! produces with the Python package, requires agreement to `1e-12`, and
//! reports how many values match bit for bit
//! (`cargo test --test golden -- --nocapture`); with the current vectors all
//! of them do. Routers written by this crate are read back by the Python test
//! suite.
//!
//! What stays in the Python package: bootstrap confidence intervals (they
//! depend on numpy's random generator), the baselines of Section 6.1, the
//! command-line tool and the benchmark scripts.
//!
//! ## Features
//!
//! * `json` (default): `Router::from_json`, `Router::to_json`,
//!   `Router::load` and `Router::save` for the router file format shared with
//!   the Python package, serde support for [`Router`] and the result types,
//!   and `signal::from_openai_logprobs`. It pulls in `serde` and
//!   `serde_json`; without it the crate has no dependencies.
//!
//! The minimum supported Rust version is 1.70.

#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

pub mod calibration;
pub mod error;
pub mod metrics;
mod num;
pub mod policy;
pub mod router;
pub mod signal;

// Compile and run the code blocks of README.md as doctests.
#[cfg(all(doctest, feature = "json"))]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;

pub use calibration::{pav, IsotonicCalibrator};
pub use error::{Result, UcciError};
pub use metrics::{
    brier_score, ece, micro_f1, reliability_table, BinStrategy, ReliabilityRow, RoutedMicroF1,
};
pub use policy::{
    default_grid, escalate, escalate_many, evaluate, make_grid, pareto_frontier, policy_accuracy,
    policy_cost, select_threshold, select_threshold_for_budget, CostModel, Costs, ParetoFrontier,
    ThresholdChoice,
};
pub use router::{CalibratedRouter, Route, Router, RouterBuilder};
pub use signal::{
    margins_from_top2, token_margin_uncertainty, top2_from_logprobs, uncertainty_from_logprobs,
    uncertainty_from_margins, uncertainty_from_probs, uncertainty_from_top_logprobs,
    MarginAccumulator,
};
