//! A fitted UCCI router: calibrator plus threshold (Sections 4 and 6.1), and
//! the `ucci-router` JSON format shared with the Python package.
//!
//! A [`Router`] holds the calibration map `g` (Section 4.2), the threshold
//! `theta*` (Section 4.3) and the cost settings. It answers, for a new query
//! with uncertainty `u(x)`, whether to escalate (Eq. 6):
//! `p_hat = g(u) > theta*`.
//!
//! Routers come from three places:
//!
//! * a file written by the Python package (`UCCIRouter.save`), read with
//!   `Router::from_json` or `Router::load` (feature `json`);
//! * the Rust pipeline [`RouterBuilder`] -> [`CalibratedRouter`] -> [`Router`],
//!   which mirrors the evaluation protocol of Section 6.1: fit `g` on the
//!   calibration split, choose `theta*` on the validation split;
//! * [`Router::new`] from a calibrator and a threshold chosen elsewhere.
//!
//! # File format (version 1)
//!
//! ```text
//! {
//!   "format": "ucci-router",
//!   "version": 1,
//!   "calibrator": {"x": [float, ...], "y": [float, ...]},
//!   "theta": float,
//!   "c_small": float,
//!   "c_large": float,
//!   "cost_model": "routing" | "sequential",
//!   "tau": float | null,
//!   "grid_step": 0.005,
//!   "created_by": "ucci-python <version>"
//! }
//! ```
//!
//! Rules checked on read, the same as in Python: `format` and `version` are
//! known (`version` must be the JSON integer 1); `x` is strictly increasing;
//! `y` is non-decreasing within `[0, 1]`; `x` and `y` have the same length, at
//! least 1; every number is finite; costs are positive; `grid_step`, when
//! present, lies in `(0, 1]` and divides 1. `tau`, `grid_step` and
//! `created_by` may be missing or null. Unknown keys are ignored, so later
//! versions can add fields without breaking older readers. Prediction is
//! linear interpolation of `(x, y)` clipped at both ends, and the policy
//! escalates when the prediction is strictly greater than `theta`.
//!
//! # Example
//!
//! ```
//! # #[cfg(feature = "json")] {
//! use ucci::Router;
//!
//! let doc = r#"{
//!   "format": "ucci-router", "version": 1,
//!   "calibrator": {"x": [0.1, 0.4, 0.8], "y": [0.05, 0.2, 0.7]},
//!   "theta": 0.3, "c_small": 1.0, "c_large": 3.02, "cost_model": "routing",
//!   "tau": 0.91, "grid_step": 0.005, "created_by": "ucci-python 0.1.0"
//! }"#;
//! let router = Router::from_json(doc)?;
//! assert!(!router.escalate(0.2)?); // p_hat = 0.1 <= 0.3: keep the small model
//! assert!(router.escalate(0.7)?); // p_hat = 0.575 > 0.3: escalate
//! let again = Router::from_json(&router.to_json())?;
//! assert_eq!(again.theta(), router.theta());
//! # }
//! # Ok::<(), ucci::UcciError>(())
//! ```

use crate::calibration::IsotonicCalibrator;
use crate::error::{Result, UcciError};
use crate::num::check_finite_scalar;
use crate::policy::{
    escalate_many, evaluate, make_grid, select_threshold, select_threshold_for_budget, Costs,
    ThresholdChoice, DEFAULT_GRID_STEP,
};

/// Value of the `format` field of a router file.
pub const FORMAT_NAME: &str = "ucci-router";

/// The router file format version this crate reads and writes.
pub const FORMAT_VERSION: u64 = 1;

/// The `created_by` string this crate writes: `"ucci-rust <crate version>"`.
pub const CREATED_BY: &str = concat!("ucci-rust ", env!("CARGO_PKG_VERSION"));

/// The routing decision for one query (Eq. 6).
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "json", derive(serde::Serialize, serde::Deserialize))]
pub struct Route {
    /// True when the query goes to the large model (`p_hat > theta`).
    pub escalate: bool,
    /// Calibrated error probability `p_hat = g(u)` of the query.
    pub p_hat: f64,
}

/// A calibrated small-to-large cascade router with a chosen threshold.
///
/// See the [module documentation](self) for how to obtain one.
#[derive(Debug, Clone, PartialEq)]
pub struct Router {
    calibrator: IsotonicCalibrator,
    theta: f64,
    costs: Costs,
    grid_step: f64,
    tau: Option<f64>,
    budget: Option<f64>,
    choice: Option<ThresholdChoice>,
    created_by: Option<String>,
}

fn check_grid_step(step: f64) -> Result<f64> {
    make_grid(step).map_err(|_| UcciError::InvalidParameter {
        name: "grid_step",
        reason: format!("must lie in (0, 1] and divide 1 exactly (e.g. 0.005), got {step}"),
    })?;
    Ok(step)
}

impl Router {
    /// A router from a calibrator, a threshold and costs, with the default
    /// grid step (0.005) and no accuracy target.
    ///
    /// # Errors
    ///
    /// [`UcciError::InvalidParameter`] for a non-finite `theta` or invalid
    /// costs.
    pub fn new(calibrator: IsotonicCalibrator, theta: f64, costs: Costs) -> Result<Self> {
        check_finite_scalar(theta, "theta")?;
        costs.validate()?;
        Ok(Router {
            calibrator,
            theta,
            costs,
            grid_step: DEFAULT_GRID_STEP,
            tau: None,
            budget: None,
            choice: None,
            created_by: None,
        })
    }

    /// The calibration map `g` (Section 4.2).
    pub fn calibrator(&self) -> &IsotonicCalibrator {
        &self.calibrator
    }

    /// The escalation threshold `theta*` (Eq. 6).
    pub fn theta(&self) -> f64 {
        self.theta
    }

    /// Per-query costs and cost model.
    pub fn costs(&self) -> &Costs {
        &self.costs
    }

    /// Resolution of the threshold grid used to choose `theta*`.
    pub fn grid_step(&self) -> f64 {
        self.grid_step
    }

    /// Accuracy target `tau` the threshold was chosen for, if known.
    pub fn tau(&self) -> Option<f64> {
        self.tau
    }

    /// Cost budget the threshold was chosen for (budget form of Eq. 7), if
    /// any. Not stored in router files.
    pub fn budget(&self) -> Option<f64> {
        self.budget
    }

    /// Validation cost and accuracy of the chosen threshold, when the router
    /// was built with [`CalibratedRouter::choose_threshold`] or
    /// [`CalibratedRouter::choose_threshold_for_budget`]. Not stored in router
    /// files.
    pub fn choice(&self) -> Option<&ThresholdChoice> {
        self.choice.as_ref()
    }

    /// The `created_by` field of the file this router was read from, if any.
    /// Files written by this crate carry [`CREATED_BY`].
    pub fn created_by(&self) -> Option<&str> {
        self.created_by.as_deref()
    }

    /// A copy with a different grid step (recorded in router files).
    ///
    /// # Errors
    ///
    /// [`UcciError::InvalidParameter`] unless `step` lies in `(0, 1]` and
    /// divides 1.
    pub fn with_grid_step(mut self, step: f64) -> Result<Self> {
        self.grid_step = check_grid_step(step)?;
        Ok(self)
    }

    /// Calibrated forecast `p_hat(x) = g(u(x))` (Section 4.3).
    ///
    /// # Errors
    ///
    /// [`UcciError::NonFinite`] if `u` is NaN or infinite.
    pub fn error_probability(&self, u: f64) -> Result<f64> {
        self.calibrator.predict(u)
    }

    /// True when a query with uncertainty `u` should go to the large model
    /// (Eq. 6: `g(u) > theta`).
    ///
    /// # Errors
    ///
    /// [`UcciError::NonFinite`] if `u` is NaN or infinite.
    pub fn escalate(&self, u: f64) -> Result<bool> {
        Ok(self.calibrator.predict(u)? > self.theta)
    }

    /// Decision and calibrated probability for one query.
    ///
    /// # Errors
    ///
    /// [`UcciError::NonFinite`] if `u` is NaN or infinite.
    pub fn route(&self, u: f64) -> Result<Route> {
        let p_hat = self.calibrator.predict(u)?;
        Ok(Route {
            escalate: p_hat > self.theta,
            p_hat,
        })
    }

    /// [`Router::route`] for a batch of queries.
    ///
    /// # Errors
    ///
    /// [`UcciError::NonFinite`] naming the first NaN or infinite element.
    pub fn route_many(&self, u: &[f64]) -> Result<Vec<Route>> {
        let p_hat = self.calibrator.predict_many(u)?;
        Ok(p_hat
            .into_iter()
            .map(|p| Route {
                escalate: p > self.theta,
                p_hat: p,
            })
            .collect())
    }

    /// Routes a labelled test split end to end and reports the actual cost,
    /// accuracy and escalation rate (step 3 of the protocol in Section 6.1).
    ///
    /// # Errors
    ///
    /// As [`crate::policy::evaluate`].
    pub fn evaluate(
        &self,
        u_test: &[f64],
        small_score: &[f64],
        large_score: &[f64],
    ) -> Result<ThresholdChoice> {
        let p_hat = self.calibrator.predict_many(u_test)?;
        evaluate(&p_hat, small_score, large_score, self.theta, &self.costs)
    }

    /// Escalation mask for a batch of queries.
    ///
    /// # Errors
    ///
    /// [`UcciError::NonFinite`] naming the first NaN or infinite element.
    pub fn escalate_many(&self, u: &[f64]) -> Result<Vec<bool>> {
        escalate_many(&self.calibrator.predict_many(u)?, self.theta)
    }
}

/// First step of the Rust pipeline: costs and grid, before calibration.
///
/// # Example
///
/// ```
/// use ucci::router::RouterBuilder;
/// use ucci::Costs;
///
/// let u_cal = [0.05, 0.10, 0.20, 0.30, 0.40, 0.55, 0.70, 0.90];
/// let e_cal = [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0];
/// let u_val = [0.05, 0.25, 0.35, 0.60, 0.80, 0.95];
/// let small = [1.0, 1.0, 0.0, 0.0, 1.0, 0.0];
/// let large = [1.0, 1.0, 1.0, 1.0, 1.0, 0.0];
///
/// let calibrated = RouterBuilder::new(Costs::paper())?.calibrate(&u_cal, &e_cal)?;
/// let router = calibrated.choose_threshold(&u_val, &small, &large, 0.8)?;
/// assert_eq!(router.tau(), Some(0.8));
/// assert!(router.choice().unwrap().accuracy >= 0.8);
/// # Ok::<(), ucci::UcciError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct RouterBuilder {
    costs: Costs,
    grid_step: f64,
}

impl RouterBuilder {
    /// A builder with the given costs and the default grid step (0.005).
    ///
    /// # Errors
    ///
    /// [`UcciError::InvalidParameter`] for invalid costs.
    pub fn new(costs: Costs) -> Result<Self> {
        costs.validate()?;
        Ok(RouterBuilder {
            costs,
            grid_step: DEFAULT_GRID_STEP,
        })
    }

    /// Sets the resolution of the threshold grid.
    ///
    /// # Errors
    ///
    /// [`UcciError::InvalidParameter`] unless `step` lies in `(0, 1]` and
    /// divides 1.
    pub fn grid_step(mut self, step: f64) -> Result<Self> {
        self.grid_step = check_grid_step(step)?;
        Ok(self)
    }

    /// Fits `g` on the calibration split (Section 4.2; step 1 of
    /// Section 6.1). `e_cal[i]` is 1 where the small model was wrong.
    ///
    /// # Errors
    ///
    /// As [`IsotonicCalibrator::fit`].
    pub fn calibrate(self, u_cal: &[f64], e_cal: &[f64]) -> Result<CalibratedRouter> {
        Ok(self.with_calibrator(IsotonicCalibrator::fit(u_cal, e_cal)?))
    }

    /// [`RouterBuilder::calibrate`] with sample weights.
    ///
    /// # Errors
    ///
    /// As [`IsotonicCalibrator::fit_weighted`].
    pub fn calibrate_weighted(
        self,
        u_cal: &[f64],
        e_cal: &[f64],
        sample_weight: &[f64],
    ) -> Result<CalibratedRouter> {
        Ok(self.with_calibrator(IsotonicCalibrator::fit_weighted(
            u_cal,
            e_cal,
            sample_weight,
        )?))
    }

    /// Uses an already fitted calibrator.
    pub fn with_calibrator(self, calibrator: IsotonicCalibrator) -> CalibratedRouter {
        CalibratedRouter {
            calibrator,
            costs: self.costs,
            grid_step: self.grid_step,
        }
    }
}

/// Second step of the Rust pipeline: a fitted calibrator, before the
/// threshold is chosen.
#[derive(Debug, Clone, PartialEq)]
pub struct CalibratedRouter {
    calibrator: IsotonicCalibrator,
    costs: Costs,
    grid_step: f64,
}

impl CalibratedRouter {
    /// The fitted calibration map `g`.
    pub fn calibrator(&self) -> &IsotonicCalibrator {
        &self.calibrator
    }

    /// Calibrated forecast `p_hat = g(u)`.
    ///
    /// # Errors
    ///
    /// [`UcciError::NonFinite`] if `u` is NaN or infinite.
    pub fn error_probability(&self, u: f64) -> Result<f64> {
        self.calibrator.predict(u)
    }

    fn router(&self, choice: ThresholdChoice) -> Router {
        Router {
            calibrator: self.calibrator.clone(),
            theta: choice.theta,
            costs: self.costs,
            grid_step: self.grid_step,
            tau: None,
            budget: None,
            choice: Some(choice),
            created_by: None,
        }
    }

    /// Chooses `theta*` on the validation split: the cheapest grid threshold
    /// whose accuracy reaches `tau` (Eq. 7; step 2 of Section 6.1).
    ///
    /// # Errors
    ///
    /// As [`select_threshold`], including [`UcciError::Infeasible`].
    pub fn choose_threshold(
        &self,
        u_val: &[f64],
        small_score: &[f64],
        large_score: &[f64],
        tau: f64,
    ) -> Result<Router> {
        let p_hat = self.calibrator.predict_many(u_val)?;
        let grid = make_grid(self.grid_step)?;
        let choice = select_threshold(&p_hat, small_score, large_score, tau, &self.costs, &grid)?;
        let mut router = self.router(choice);
        router.tau = Some(tau);
        Ok(router)
    }

    /// Chooses the most accurate grid threshold within a mean cost `budget`
    /// (budget form of Eq. 7, Table 2 bottom block).
    ///
    /// # Errors
    ///
    /// As [`select_threshold_for_budget`], including
    /// [`UcciError::OverBudget`].
    pub fn choose_threshold_for_budget(
        &self,
        u_val: &[f64],
        small_score: &[f64],
        large_score: &[f64],
        budget: f64,
    ) -> Result<Router> {
        let p_hat = self.calibrator.predict_many(u_val)?;
        let grid = make_grid(self.grid_step)?;
        let choice = select_threshold_for_budget(
            &p_hat,
            small_score,
            large_score,
            budget,
            &self.costs,
            &grid,
        )?;
        let mut router = self.router(choice);
        router.budget = Some(budget);
        Ok(router)
    }

    /// A router with a threshold set by hand.
    ///
    /// # Errors
    ///
    /// [`UcciError::InvalidParameter`] for a non-finite `theta`.
    pub fn with_theta(&self, theta: f64) -> Result<Router> {
        let mut router = Router::new(self.calibrator.clone(), theta, self.costs)?;
        router.grid_step = self.grid_step;
        Ok(router)
    }
}

#[cfg(feature = "json")]
mod json {
    use std::path::Path;

    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use serde_json::{Map, Value};

    use super::{check_grid_step, Router, CREATED_BY, FORMAT_NAME, FORMAT_VERSION};
    use crate::calibration::IsotonicCalibrator;
    use crate::error::{Result, UcciError};
    use crate::policy::{CostModel, Costs, DEFAULT_GRID_STEP};

    /// Field order of a version 1 document, as written by the Python package.
    #[derive(Serialize)]
    struct Wire<'a> {
        format: &'static str,
        version: u64,
        calibrator: WireCalibrator<'a>,
        theta: f64,
        c_small: f64,
        c_large: f64,
        cost_model: &'static str,
        tau: Option<f64>,
        grid_step: f64,
        created_by: &'static str,
    }

    #[derive(Serialize)]
    struct WireCalibrator<'a> {
        x: &'a [f64],
        y: &'a [f64],
    }

    /// Compact JSON with `", "` and `": "` separators, the layout of Python's
    /// `json.dumps(value, separators=(", ", ": "))`.
    struct Spaced;

    impl serde_json::ser::Formatter for Spaced {
        fn begin_array_value<W: ?Sized + std::io::Write>(
            &mut self,
            writer: &mut W,
            first: bool,
        ) -> std::io::Result<()> {
            if first {
                Ok(())
            } else {
                writer.write_all(b", ")
            }
        }

        fn begin_object_key<W: ?Sized + std::io::Write>(
            &mut self,
            writer: &mut W,
            first: bool,
        ) -> std::io::Result<()> {
            if first {
                Ok(())
            } else {
                writer.write_all(b", ")
            }
        }

        fn begin_object_value<W: ?Sized + std::io::Write>(
            &mut self,
            writer: &mut W,
        ) -> std::io::Result<()> {
            writer.write_all(b": ")
        }
    }

    fn spaced<T: Serialize + ?Sized>(value: &T) -> String {
        let mut out = Vec::new();
        let mut ser = serde_json::Serializer::with_formatter(&mut out, Spaced);
        value
            .serialize(&mut ser)
            .expect("finite numbers and strings always serialize");
        String::from_utf8(out).expect("serde_json writes UTF-8")
    }

    fn invalid(reason: String) -> UcciError {
        UcciError::InvalidRouter { reason }
    }

    fn type_name(v: &Value) -> &'static str {
        match v {
            Value::Null => "null",
            Value::Bool(_) => "bool",
            Value::Number(_) => "number",
            Value::String(_) => "string",
            Value::Array(_) => "array",
            Value::Object(_) => "object",
        }
    }

    fn number(v: &Value, label: &str) -> Result<f64> {
        match v.as_f64() {
            Some(f) if v.is_number() && f.is_finite() => Ok(f),
            Some(f) if v.is_number() => Err(invalid(format!("'{label}' must be finite, got {f}"))),
            _ => Err(invalid(format!("'{label}' must be a number, got {v}"))),
        }
    }

    fn required_number(obj: &Map<String, Value>, key: &str) -> Result<f64> {
        match obj.get(key) {
            None => Err(invalid(format!("router file is missing '{key}'"))),
            Some(v) => number(v, key),
        }
    }

    fn optional_number(obj: &Map<String, Value>, key: &str) -> Result<Option<f64>> {
        match obj.get(key) {
            None | Some(Value::Null) => Ok(None),
            Some(v) => number(v, key).map(Some),
        }
    }

    fn number_list(v: &Value, label: &str) -> Result<Vec<f64>> {
        let arr = v.as_array().ok_or_else(|| {
            invalid(format!(
                "'{label}' must be a list of numbers, got {}",
                type_name(v)
            ))
        })?;
        arr.iter()
            .enumerate()
            .map(|(i, x)| number(x, &format!("{label}[{i}]")))
            .collect()
    }

    impl Router {
        /// Builds a router from a decoded `ucci-router` document, applying
        /// every rule of the format (see the [module documentation](crate::router)).
        ///
        /// # Errors
        ///
        /// [`UcciError::InvalidRouter`] naming the first problem found.
        pub fn from_json_value(data: &Value) -> Result<Self> {
            let obj = data.as_object().ok_or_else(|| {
                invalid(format!(
                    "router file must hold a JSON object, got {}",
                    type_name(data)
                ))
            })?;
            match obj.get("format") {
                Some(Value::String(s)) if s == FORMAT_NAME => {}
                other => {
                    return Err(invalid(format!(
                        "not a UCCI router file: 'format' is {}, expected \"{FORMAT_NAME}\"",
                        other.map_or("missing".to_string(), Value::to_string)
                    )))
                }
            }
            let version = obj.get("version").unwrap_or(&Value::Null);
            if !(version.is_u64() || version.is_i64()) {
                return Err(invalid(format!(
                    "'version' must be an integer, got {version}"
                )));
            }
            if version.as_u64() != Some(FORMAT_VERSION) {
                return Err(invalid(format!(
                    "unsupported router format version {version}; this crate reads version \
                     {FORMAT_VERSION} (a newer file needs a newer ucci)"
                )));
            }

            let cal = obj
                .get("calibrator")
                .and_then(Value::as_object)
                .ok_or_else(|| {
                    invalid("'calibrator' must be an object with lists 'x' and 'y'".to_string())
                })?;
            for key in ["x", "y"] {
                if !cal.contains_key(key) {
                    return Err(invalid(format!(
                        "router file is missing 'calibrator.{key}'"
                    )));
                }
            }
            let x = number_list(&cal["x"], "calibrator.x")?;
            let y = number_list(&cal["y"], "calibrator.y")?;
            if x.len() != y.len() {
                return Err(invalid(format!(
                    "'calibrator.x' has {} values but 'calibrator.y' has {}",
                    x.len(),
                    y.len()
                )));
            }
            if x.is_empty() {
                return Err(invalid(
                    "'calibrator.x' is empty; at least one knot is required".to_string(),
                ));
            }
            for i in 1..x.len() {
                if x[i] <= x[i - 1] {
                    return Err(invalid(format!(
                        "'calibrator.x' must be strictly increasing; x[{}] = {} >= x[{}] = {}",
                        i - 1,
                        x[i - 1],
                        i,
                        x[i]
                    )));
                }
                if y[i] < y[i - 1] {
                    return Err(invalid(format!(
                        "'calibrator.y' must be non-decreasing; y[{}] = {} > y[{}] = {}",
                        i - 1,
                        y[i - 1],
                        i,
                        y[i]
                    )));
                }
            }
            if let Some(i) = y.iter().position(|v| !(0.0..=1.0).contains(v)) {
                return Err(invalid(format!(
                    "'calibrator.y[{i}]' = {} is not a probability in [0, 1]",
                    y[i]
                )));
            }

            let theta = required_number(obj, "theta")?;
            let c_small = required_number(obj, "c_small")?;
            let c_large = required_number(obj, "c_large")?;
            for (label, c) in [("c_small", c_small), ("c_large", c_large)] {
                if c <= 0.0 {
                    return Err(invalid(format!("'{label}' must be positive, got {c}")));
                }
            }
            let model = match obj.get("cost_model") {
                Some(Value::String(s)) if s == "routing" => CostModel::Routing,
                Some(Value::String(s)) if s == "sequential" => CostModel::Sequential,
                other => {
                    return Err(invalid(format!(
                        "'cost_model' must be 'routing' or 'sequential', got {}",
                        other.map_or("missing".to_string(), Value::to_string)
                    )))
                }
            };
            let tau = optional_number(obj, "tau")?;
            let grid_step = optional_number(obj, "grid_step")?;
            if let Some(step) = grid_step {
                if !(step > 0.0 && step <= 1.0) {
                    return Err(invalid(format!(
                        "'grid_step' must lie in (0, 1], got {step}"
                    )));
                }
            }
            let grid_step = check_grid_step(grid_step.unwrap_or(DEFAULT_GRID_STEP))
                .map_err(|e| invalid(e.to_string()))?;
            let created_by = obj
                .get("created_by")
                .and_then(Value::as_str)
                .map(str::to_string);

            let calibrator =
                IsotonicCalibrator::from_knots(x, y).map_err(|e| invalid(e.to_string()))?;
            let costs = Costs {
                c_small,
                c_large,
                model,
            };
            let mut router = Router::new(calibrator, theta, costs)?;
            router.grid_step = grid_step;
            router.tau = tau;
            router.created_by = created_by;
            Ok(router)
        }

        /// Parses a `ucci-router` JSON document (as written by the Python
        /// package's `UCCIRouter.save` or by [`Router::to_json`]).
        ///
        /// # Errors
        ///
        /// [`UcciError::InvalidRouter`] for invalid JSON or a document that
        /// breaks a rule of the format.
        pub fn from_json(text: &str) -> Result<Self> {
            let value: Value =
                serde_json::from_str(text).map_err(|e| invalid(format!("invalid JSON: {e}")))?;
            Self::from_json_value(&value)
        }

        /// Reads a router file.
        ///
        /// # Errors
        ///
        /// [`UcciError::InvalidRouter`] naming the path, for an unreadable
        /// file, invalid JSON or an invalid document.
        pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
            let path = path.as_ref();
            let text = std::fs::read_to_string(path)
                .map_err(|e| invalid(format!("{}: {e}", path.display())))?;
            Self::from_json(&text).map_err(|e| match e {
                UcciError::InvalidRouter { reason } => {
                    invalid(format!("{}: {reason}", path.display()))
                }
                other => other,
            })
        }

        fn wire(&self) -> Wire<'_> {
            Wire {
                format: FORMAT_NAME,
                version: FORMAT_VERSION,
                calibrator: WireCalibrator {
                    x: self.calibrator.x(),
                    y: self.calibrator.y(),
                },
                theta: self.theta,
                c_small: self.costs.c_small,
                c_large: self.costs.c_large,
                cost_model: self.costs.model.as_str(),
                tau: self.tau,
                grid_step: self.grid_step,
                created_by: CREATED_BY,
            }
        }

        /// The router as a `ucci-router` version 1 document. `tau` is written
        /// as null when unknown and `created_by` is [`CREATED_BY`].
        pub fn to_json_value(&self) -> Value {
            serde_json::to_value(self.wire())
                .expect("a document of finite numbers and strings always serializes")
        }

        /// The router as a `ucci-router` JSON document, laid out like the
        /// Python package writes it: one top-level key per line, lists on one
        /// line (a calibrator with hundreds of knots stays a short,
        /// diff-friendly file), trailing newline. Floats are written in
        /// shortest round-trip form, so the file loads back bit for bit in
        /// Rust and in Python.
        pub fn to_json(&self) -> String {
            let w = self.wire();
            let lines = [
                format!("  \"format\": {}", spaced(w.format)),
                format!("  \"version\": {}", spaced(&w.version)),
                format!("  \"calibrator\": {}", spaced(&w.calibrator)),
                format!("  \"theta\": {}", spaced(&w.theta)),
                format!("  \"c_small\": {}", spaced(&w.c_small)),
                format!("  \"c_large\": {}", spaced(&w.c_large)),
                format!("  \"cost_model\": {}", spaced(w.cost_model)),
                format!("  \"tau\": {}", spaced(&w.tau)),
                format!("  \"grid_step\": {}", spaced(&w.grid_step)),
                format!("  \"created_by\": {}", spaced(w.created_by)),
            ];
            format!("{{\n{}\n}}\n", lines.join(",\n"))
        }

        /// Writes the router to `path` atomically: the JSON goes to a
        /// temporary file in the same directory, which is then renamed over
        /// `path`, so readers never see a half-written file.
        ///
        /// # Errors
        ///
        /// [`UcciError::InvalidRouter`] naming the path if the file cannot be
        /// written.
        pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
            let path = path.as_ref();
            let fail = |e: std::io::Error| invalid(format!("{}: {e}", path.display()));
            let file_name = path
                .file_name()
                .ok_or_else(|| invalid(format!("{}: not a file path", path.display())))?;
            // Unique per process and per call, so concurrent saves never share
            // a temporary file.
            static SAVES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let n = SAVES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let mut tmp_name = std::ffi::OsString::from(".");
            tmp_name.push(file_name);
            tmp_name.push(format!(".{}.{n}.tmp", std::process::id()));
            let tmp = path.with_file_name(tmp_name);
            if let Err(e) = std::fs::write(&tmp, self.to_json()) {
                let _ = std::fs::remove_file(&tmp);
                return Err(fail(e));
            }
            std::fs::rename(&tmp, path).map_err(|e| {
                let _ = std::fs::remove_file(&tmp);
                fail(e)
            })
        }
    }

    /// Serializes as a `ucci-router` version 1 document, keys in the order
    /// the Python package writes them.
    impl Serialize for Router {
        fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
            self.wire().serialize(serializer)
        }
    }

    /// Deserializes from a `ucci-router` document with the same validation
    /// as [`Router::from_json`]. Needs a self-describing format.
    impl<'de> Deserialize<'de> for Router {
        fn deserialize<D: Deserializer<'de>>(
            deserializer: D,
        ) -> std::result::Result<Self, D::Error> {
            let value = Value::deserialize(deserializer)?;
            Router::from_json_value(&value).map_err(serde::de::Error::custom)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn calibrator() -> IsotonicCalibrator {
        IsotonicCalibrator::from_knots(vec![0.1, 0.4, 0.8], vec![0.05, 0.2, 0.7]).unwrap()
    }

    #[test]
    fn routes_by_calibrated_probability() {
        let r = Router::new(calibrator(), 0.2, Costs::paper()).unwrap();
        assert!(!r.escalate(0.4).unwrap()); // p_hat == theta stays small
        assert!(r.escalate(0.41).unwrap());
        assert!(!r.escalate(-1.0).unwrap());
        assert!(r.escalate(5.0).unwrap());
        assert!(r.escalate(f64::NAN).is_err());
        let route = r.route(0.8).unwrap();
        assert_eq!(
            route,
            Route {
                escalate: true,
                p_hat: 0.7
            }
        );
        assert_eq!(r.error_probability(0.1).unwrap(), 0.05);
        let batch = r.route_many(&[0.1, 0.8]).unwrap();
        assert_eq!(
            batch.iter().map(|d| d.escalate).collect::<Vec<_>>(),
            vec![false, true]
        );
        assert_eq!(r.escalate_many(&[0.1, 0.8]).unwrap(), vec![false, true]);
        assert!(r.route_many(&[0.1, f64::INFINITY]).is_err());
    }

    #[test]
    fn new_validates() {
        assert!(Router::new(calibrator(), f64::NAN, Costs::paper()).is_err());
        let bad = Costs {
            c_small: 0.0,
            ..Costs::paper()
        };
        assert!(Router::new(calibrator(), 0.2, bad).is_err());
        let r = Router::new(calibrator(), 0.2, Costs::paper()).unwrap();
        assert_eq!(r.grid_step(), 0.005);
        assert_eq!(r.tau(), None);
        assert_eq!(r.budget(), None);
        assert!(r.choice().is_none());
        assert!(r.created_by().is_none());
        assert!(r.clone().with_grid_step(0.3).is_err());
        assert_eq!(r.with_grid_step(0.01).unwrap().grid_step(), 0.01);
    }

    #[test]
    fn pipeline() {
        let u_cal = [0.05, 0.10, 0.20, 0.30, 0.40, 0.55, 0.70, 0.90];
        let e_cal = [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0];
        let u_val = [0.05, 0.25, 0.35, 0.60, 0.80, 0.95];
        let small = [1.0, 1.0, 0.0, 0.0, 1.0, 0.0];
        let large = [1.0, 1.0, 1.0, 1.0, 1.0, 0.0];
        let cal = RouterBuilder::new(Costs::paper())
            .unwrap()
            .calibrate(&u_cal, &e_cal)
            .unwrap();
        let r = cal.choose_threshold(&u_val, &small, &large, 0.8).unwrap();
        let choice = *r.choice().unwrap();
        assert_eq!(r.theta(), choice.theta);
        assert_eq!(r.tau(), Some(0.8));
        // Evaluating on the validation split reproduces the selection.
        assert_eq!(r.evaluate(&u_val, &small, &large).unwrap(), choice);
        assert!(matches!(
            cal.choose_threshold(&u_val, &small, &large, 0.99),
            Err(UcciError::Infeasible { .. })
        ));
        let b = cal
            .choose_threshold_for_budget(&u_val, &small, &large, 2.0)
            .unwrap();
        assert_eq!(b.budget(), Some(2.0));
        assert!(b.choice().unwrap().cost <= 2.0);
        let h = cal.with_theta(0.5).unwrap();
        assert_eq!(h.theta(), 0.5);
        assert!(cal.error_probability(0.3).unwrap() >= 0.0);
        let weighted = RouterBuilder::new(Costs::paper())
            .unwrap()
            .grid_step(0.01)
            .unwrap()
            .calibrate_weighted(&u_cal, &e_cal, &[1.0; 8])
            .unwrap();
        assert_eq!(weighted.calibrator(), cal.calibrator());
        let r2 = weighted
            .choose_threshold(&u_val, &small, &large, 0.8)
            .unwrap();
        assert_eq!(r2.grid_step(), 0.01);
        assert!(RouterBuilder::new(Costs::paper())
            .unwrap()
            .grid_step(0.0)
            .is_err());
    }

    #[cfg(feature = "json")]
    mod json {
        use super::*;
        use serde_json::json;

        fn doc() -> serde_json::Value {
            json!({
                "format": "ucci-router", "version": 1,
                "calibrator": {"x": [0.1, 0.4, 0.8], "y": [0.05, 0.2, 0.7]},
                "theta": 0.3, "c_small": 1.0, "c_large": 3.02, "cost_model": "routing",
                "tau": 0.91, "grid_step": 0.005, "created_by": "ucci-python 0.1.0",
                "some_future_field": {"nested": [1, 2, 3]}
            })
        }

        fn with(key: &str, value: serde_json::Value) -> String {
            let mut d = doc();
            d[key] = value;
            d.to_string()
        }

        fn without(key: &str) -> String {
            let mut d = doc();
            d.as_object_mut().unwrap().remove(key);
            d.to_string()
        }

        #[test]
        fn reads_valid_documents() {
            let r = Router::from_json(&doc().to_string()).unwrap();
            assert_eq!(r.theta(), 0.3);
            assert_eq!(r.tau(), Some(0.91));
            assert_eq!(r.created_by(), Some("ucci-python 0.1.0"));
            assert_eq!(r.costs(), &Costs::paper());
            // Optional fields may be missing or null.
            for key in ["tau", "grid_step", "created_by"] {
                assert!(Router::from_json(&without(key)).is_ok(), "{key}");
                assert!(Router::from_json(&with(key, json!(null))).is_ok(), "{key}");
            }
            assert_eq!(
                Router::from_json(&without("grid_step"))
                    .unwrap()
                    .grid_step(),
                0.005
            );
            // A non-string created_by is ignored, as in Python.
            assert!(Router::from_json(&with("created_by", json!(5)))
                .unwrap()
                .created_by()
                .is_none());
            // Integers are numbers.
            assert_eq!(
                Router::from_json(&with("theta", json!(1))).unwrap().theta(),
                1.0
            );
            assert!(Router::from_json(&with("cost_model", json!("sequential"))).is_ok());
        }

        #[test]
        fn rejects_invalid_documents() {
            let cases: Vec<String> = vec![
                "[]".to_string(),
                "not json".to_string(),
                with("format", json!("ucci")),
                without("format"),
                with("version", json!(2)),
                with("version", json!(1.0)),
                with("version", json!("1")),
                with("version", json!(true)),
                without("version"),
                with("calibrator", json!([1, 2])),
                with("calibrator", json!({"x": [0.1]})),
                with("calibrator", json!({"x": [0.1, 0.2], "y": [0.1]})),
                with("calibrator", json!({"x": [], "y": []})),
                with("calibrator", json!({"x": [0.2, 0.1], "y": [0.1, 0.2]})),
                with("calibrator", json!({"x": [0.1, 0.1], "y": [0.1, 0.2]})),
                with("calibrator", json!({"x": [0.1, 0.2], "y": [0.3, 0.2]})),
                with("calibrator", json!({"x": [0.1, 0.2], "y": [0.3, 1.2]})),
                with("calibrator", json!({"x": [0.1, 0.2], "y": [-0.1, 0.2]})),
                with("calibrator", json!({"x": [0.1, true], "y": [0.1, 0.2]})),
                with("calibrator", json!({"x": "0.1", "y": [0.1]})),
                without("theta"),
                with("theta", json!("0.3")),
                with("theta", json!(null)),
                without("c_small"),
                with("c_small", json!(0.0)),
                with("c_large", json!(-1.0)),
                with("cost_model", json!("latency")),
                without("cost_model"),
                with("tau", json!("high")),
                with("grid_step", json!(0.0)),
                with("grid_step", json!(1.5)),
                with("grid_step", json!(0.3)),
                r#"{"format": "ucci-router", "version": 1, "theta": NaN}"#.to_string(),
            ];
            for text in cases {
                match Router::from_json(&text) {
                    Err(UcciError::InvalidRouter { .. }) => {}
                    other => panic!("accepted or wrong error for {text}: {other:?}"),
                }
            }
        }

        #[test]
        fn round_trip_is_exact() {
            let r = Router::from_json(&doc().to_string()).unwrap();
            let text = r.to_json();
            assert!(text.ends_with('\n'));
            let back = Router::from_json(&text).unwrap();
            assert_eq!(back.calibrator(), r.calibrator());
            assert_eq!(back.theta(), r.theta());
            assert_eq!(back.costs(), r.costs());
            assert_eq!(back.tau(), r.tau());
            assert_eq!(back.grid_step(), r.grid_step());
            assert_eq!(back.created_by(), Some(CREATED_BY));
            let keys = [
                "format",
                "version",
                "calibrator",
                "theta",
                "c_small",
                "c_large",
                "cost_model",
                "tau",
                "grid_step",
                "created_by",
            ];
            let positions: Vec<usize> = keys
                .iter()
                .map(|k| text.find(&format!("\"{k}\"")).unwrap())
                .collect();
            assert!(positions.windows(2).all(|w| w[0] < w[1]), "{text}");
            assert_eq!(r.to_json_value()["version"], 1);
            let no_tau = Router::new(calibrator(), 0.2, Costs::paper()).unwrap();
            assert_eq!(no_tau.to_json_value()["tau"], serde_json::Value::Null);
        }

        #[test]
        fn serde_impls_validate() {
            let r: Router = serde_json::from_value(doc()).unwrap();
            assert_eq!(r.theta(), 0.3);
            let v = serde_json::to_value(&r).unwrap();
            assert_eq!(v["format"], "ucci-router");
            let mut bad = doc();
            bad["calibrator"]["y"] = json!([0.5, 0.2, 0.7]);
            assert!(serde_json::from_value::<Router>(bad).is_err());
        }

        #[test]
        fn save_and_load() {
            let dir = std::env::temp_dir().join(format!("ucci-router-test-{}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            let path = dir.join("router.json");
            let r = Router::from_json(&doc().to_string()).unwrap();
            r.save(&path).unwrap();
            let back = Router::load(&path).unwrap();
            assert_eq!(back.calibrator(), r.calibrator());
            assert_eq!(back.theta(), r.theta());
            let missing = Router::load(dir.join("missing.json")).unwrap_err();
            assert!(missing.to_string().contains("missing.json"));
            std::fs::write(dir.join("bad.json"), "{}").unwrap();
            let bad = Router::load(dir.join("bad.json")).unwrap_err();
            assert!(bad.to_string().contains("bad.json"));
            std::fs::remove_dir_all(&dir).unwrap();
        }
    }
}
