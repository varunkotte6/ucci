//! Replays the golden vectors in `../tests/golden/*.json` against this crate.
//!
//! The vectors are written by `tools/make_golden.py` from the Python
//! reference package: each case holds the inputs of one call and the outputs
//! (or the error) the Python core produced. Every output must match to the
//! file's absolute tolerance (1e-12); in practice nearly all values match bit
//! for bit, and the test prints how many did.
//!
//! Set `UCCI_GOLDEN_DIR` to read the vectors from another directory. Outside
//! the repository (for example from the published crate) the test is skipped
//! when the directory is missing.
//!
//! With the `json` feature, the test also writes every Python router it reads
//! back out with [`Router::to_json`] and compares the text with
//! `tests/golden/rust_written_routers.json`, which `tests/test_golden.py`
//! loads with the Python package. Regenerate that file after an intended
//! change of the writer with `UCCI_BLESS=1 cargo test --test golden`.

use std::fs;
use std::path::PathBuf;

use serde_json::{json, Map, Value};
use ucci::calibration::{pav, IsotonicCalibrator};
use ucci::metrics::{self, BinStrategy, RoutedMicroF1};
use ucci::policy::{self, CostModel, Costs};
use ucci::router::RouterBuilder;
use ucci::signal;
use ucci::UcciError;

// ---------------------------------------------------------------- loading

fn golden_dir() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("UCCI_GOLDEN_DIR") {
        return Some(PathBuf::from(dir));
    }
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let dir = manifest.join("../tests/golden");
    if dir.is_dir() {
        return Some(dir);
    }
    // Inside the repository the vectors must exist; elsewhere, skip.
    assert!(
        !manifest.join("../src/ucci").is_dir(),
        "golden vectors missing: run `python tools/make_golden.py` in the repository root"
    );
    eprintln!("golden vectors not found next to the crate; skipping (set UCCI_GOLDEN_DIR)");
    None
}

const FILES: [&str; 6] = [
    "signal.json",
    "calibration.json",
    "policy.json",
    "metrics.json",
    "router.json",
    "pipeline.json",
];

fn read_json(path: &PathBuf) -> Value {
    let text = fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// Case input with its shared dataset merged in (see `resolve` in make_golden.py).
fn resolve(input: &Value, data: &Value) -> Map<String, Value> {
    let own = input.as_object().expect("input is an object");
    let mut merged = Map::new();
    if let Some(name) = own.get("$data").and_then(Value::as_str) {
        merged = data[name]
            .as_object()
            .expect("dataset is an object")
            .clone();
    }
    for (k, v) in own {
        if k != "$data" {
            merged.insert(k.clone(), v.clone());
        }
    }
    merged
}

// ---------------------------------------------------------------- decoding

fn num(v: &Value) -> f64 {
    match v {
        Value::Number(n) => n.as_f64().expect("number"),
        Value::String(s) if s == "nan" => f64::NAN,
        Value::String(s) if s == "inf" => f64::INFINITY,
        Value::String(s) if s == "-inf" => f64::NEG_INFINITY,
        other => panic!("not a number: {other}"),
    }
}

fn nums(v: &Value) -> Vec<f64> {
    v.as_array().expect("array").iter().map(num).collect()
}

fn opt_nums(v: Option<&Value>) -> Option<Vec<f64>> {
    match v {
        None | Some(Value::Null) => None,
        Some(v) => Some(nums(v)),
    }
}

fn bools(v: &Value) -> Vec<bool> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|b| b.as_bool().expect("bool"))
        .collect()
}

fn nested(v: &Value) -> Vec<Vec<f64>> {
    v.as_array().expect("array").iter().map(nums).collect()
}

fn counts(v: &Value) -> Vec<[f64; 3]> {
    nested(v).into_iter().map(|r| [r[0], r[1], r[2]]).collect()
}

fn costs(inp: &Map<String, Value>) -> Result<Costs, UcciError> {
    let model: CostModel = inp["cost_model"].as_str().expect("cost_model").parse()?;
    Costs::new(num(&inp["c_small"]), num(&inp["c_large"]), model)
}

fn grid(inp: &Map<String, Value>) -> Vec<f64> {
    opt_nums(inp.get("grid")).unwrap_or_else(policy::default_grid)
}

fn strategy(inp: &Map<String, Value>) -> BinStrategy {
    inp["strategy"]
        .as_str()
        .expect("strategy")
        .parse()
        .expect("strategy")
}

fn f(x: f64) -> Value {
    // Non-finite values never appear in expected outputs; keep them visible.
    if x.is_finite() {
        json!(x)
    } else {
        json!(x.to_string())
    }
}

fn fs_(xs: &[f64]) -> Value {
    Value::Array(xs.iter().map(|&x| f(x)).collect())
}

fn choice(c: &policy::ThresholdChoice) -> Value {
    json!({"theta": f(c.theta), "cost": f(c.cost), "accuracy": f(c.accuracy),
           "escalation_rate": f(c.escalation_rate)})
}

fn frontier(p: &policy::ParetoFrontier) -> Value {
    json!({"theta": fs_(&p.theta), "cost": fs_(&p.cost), "accuracy": fs_(&p.accuracy),
           "escalation_rate": fs_(&p.escalation_rate), "efficient": p.efficient})
}

fn micro_f1_metric(inp: &Map<String, Value>) -> Result<Option<RoutedMicroF1>, UcciError> {
    if !inp.contains_key("small_counts") {
        return Ok(None);
    }
    let zd = inp.get("zero_division").map_or(0.0, num);
    RoutedMicroF1::new(
        &counts(&inp["small_counts"]),
        &counts(&inp["large_counts"]),
        zd,
    )
    .map(Some)
}

// ---------------------------------------------------------------- dispatch

/// Rust counterpart of `_compute` in tools/make_golden.py. `None` means the
/// function needs a feature that is disabled in this build.
fn run(func: &str, inp: &Map<String, Value>) -> Option<Result<Value, UcciError>> {
    let pairs =
        |v: &Value| -> Vec<(f64, f64)> { nested(v).into_iter().map(|p| (p[0], p[1])).collect() };
    let out: Result<Value, UcciError> = match func {
        // signal
        "margins_from_top2" => {
            signal::margins_from_top2(&pairs(&inp["pairs"])).map(|m| json!({"margins": fs_(&m)}))
        }
        "token_margin_uncertainty" => {
            signal::token_margin_uncertainty(&pairs(&inp["pairs"])).map(|u| json!({"u": f(u)}))
        }
        "uncertainty_from_margins" => {
            signal::uncertainty_from_margins(&nums(&inp["margins"])).map(|u| json!({"u": f(u)}))
        }
        "uncertainty_from_probs" => {
            signal::uncertainty_from_probs(&nums(&inp["p1"]), &nums(&inp["p2"]))
                .map(|u| json!({"u": f(u)}))
        }
        "uncertainty_from_logprobs" => {
            signal::uncertainty_from_logprobs(&nums(&inp["lp1"]), &nums(&inp["lp2"]))
                .map(|u| json!({"u": f(u)}))
        }
        "top2_from_logprobs" => signal::top2_from_logprobs(&nested(&inp["per_token"])).map(
            |ps| json!({"pairs": ps.iter().map(|&(a, b)| json!([f(a), f(b)])).collect::<Vec<_>>()}),
        ),
        "uncertainty_from_top_logprobs" => {
            signal::uncertainty_from_top_logprobs(&nested(&inp["per_token"]))
                .map(|u| json!({"u": f(u)}))
        }
        "from_openai_logprobs" => {
            #[cfg(feature = "json")]
            {
                signal::from_openai_logprobs(&inp["content"]).map(|u| json!({"u": f(u)}))
            }
            #[cfg(not(feature = "json"))]
            {
                return None;
            }
        }
        // calibration
        "pav" => pav(&nums(&inp["y"]), opt_nums(inp.get("w")).as_deref())
            .map(|fit| json!({"fit": fs_(&fit)})),
        "calibrator_fit" => (|| {
            let u = nums(&inp["u"]);
            let e = nums(&inp["e"]);
            let g = match opt_nums(inp.get("sample_weight")) {
                Some(w) => IsotonicCalibrator::fit_weighted(&u, &e, &w)?,
                None => IsotonicCalibrator::fit(&u, &e)?,
            };
            let pred = g.predict_many(&nums(&inp["query"]))?;
            Ok(
                json!({"x": fs_(g.x()), "y": fs_(g.y()), "n_samples": g.n_samples(),
                      "predict": fs_(&pred)}),
            )
        })(),
        "calibrator_from_knots" => (|| {
            let g = IsotonicCalibrator::from_knots(nums(&inp["x"]), nums(&inp["y"]))?;
            Ok(json!({"predict": fs_(&g.predict_many(&nums(&inp["query"]))?)}))
        })(),
        "calibrator_predict" => (|| {
            let g = IsotonicCalibrator::from_knots(nums(&inp["x"]), nums(&inp["y"]))?;
            Ok(json!({"p_hat": f(g.predict(num(&inp["u"]))?)}))
        })(),
        // policy
        "default_grid" => Ok(json!({"grid": fs_(&policy::default_grid())})),
        "make_grid" => policy::make_grid(num(&inp["step"])).map(|g| json!({"grid": fs_(&g)})),
        "escalate" => policy::escalate_many(&nums(&inp["p_hat"]), num(&inp["theta"]))
            .map(|m| json!({"mask": m})),
        "escalate_scalar" => {
            policy::escalate(num(&inp["p_hat"]), num(&inp["theta"])).map(|e| json!({"escalate": e}))
        }
        "policy_cost" => costs(inp)
            .and_then(|c| policy::policy_cost(&bools(&inp["esc"]), &c))
            .map(|c| json!({"cost": f(c)})),
        "policy_accuracy" => policy::policy_accuracy(
            &bools(&inp["esc"]),
            &nums(&inp["small_score"]),
            &nums(&inp["large_score"]),
        )
        .map(|a| json!({"accuracy": f(a)})),
        "select_threshold" => (|| {
            let c = costs(inp)?;
            let p = nums(&inp["p_hat"]);
            let tau = num(&inp["tau"]);
            let ch = match micro_f1_metric(inp)? {
                Some(m) => {
                    policy::select_threshold_with_metric(&p, tau, &c, &grid(inp), m.metric())?
                }
                None => policy::select_threshold(
                    &p,
                    &nums(&inp["small_score"]),
                    &nums(&inp["large_score"]),
                    tau,
                    &c,
                    &grid(inp),
                )?,
            };
            Ok(choice(&ch))
        })(),
        "select_threshold_for_budget" => (|| {
            let c = costs(inp)?;
            let p = nums(&inp["p_hat"]);
            let budget = num(&inp["budget"]);
            let ch = match micro_f1_metric(inp)? {
                Some(m) => policy::select_threshold_for_budget_with_metric(
                    &p,
                    budget,
                    &c,
                    &grid(inp),
                    m.metric(),
                )?,
                None => policy::select_threshold_for_budget(
                    &p,
                    &nums(&inp["small_score"]),
                    &nums(&inp["large_score"]),
                    budget,
                    &c,
                    &grid(inp),
                )?,
            };
            Ok(choice(&ch))
        })(),
        "pareto_frontier" => (|| {
            let c = costs(inp)?;
            let p = nums(&inp["p_hat"]);
            let fr = match micro_f1_metric(inp)? {
                Some(m) => policy::pareto_frontier_with_metric(&p, &c, &grid(inp), m.metric())?,
                None => policy::pareto_frontier(
                    &p,
                    &nums(&inp["small_score"]),
                    &nums(&inp["large_score"]),
                    &c,
                    &grid(inp),
                )?,
            };
            Ok(frontier(&fr))
        })(),
        "evaluate" => (|| {
            let c = costs(inp)?;
            let p = nums(&inp["p_hat"]);
            let theta = num(&inp["theta"]);
            let ch = match micro_f1_metric(inp)? {
                Some(m) => policy::evaluate_with_metric(&p, theta, &c, m.metric())?,
                None => policy::evaluate(
                    &p,
                    &nums(&inp["small_score"]),
                    &nums(&inp["large_score"]),
                    theta,
                    &c,
                )?,
            };
            Ok(choice(&ch))
        })(),
        // metrics
        "ece" => metrics::ece(
            &nums(&inp["p"]),
            &nums(&inp["y"]),
            inp["n_bins"].as_u64().expect("n_bins") as usize,
            strategy(inp),
            opt_nums(inp.get("sample_weight")).as_deref(),
        )
        .map(|e| json!({"ece": f(e)})),
        "reliability_table" => metrics::reliability_table(
            &nums(&inp["p"]),
            &nums(&inp["y"]),
            inp["n_bins"].as_u64().expect("n_bins") as usize,
            strategy(inp),
            opt_nums(inp.get("sample_weight")).as_deref(),
        )
        .map(|rows| {
            let rows: Vec<Value> = rows
                .iter()
                .map(|r| {
                    json!({"bin_lower": f(r.bin_lower), "bin_upper": f(r.bin_upper),
                           "count": r.count, "mean_forecast": f(r.mean_forecast),
                           "observed_frequency": f(r.observed_frequency)})
                })
                .collect();
            json!({ "rows": rows })
        }),
        "brier_score" => metrics::brier_score(
            &nums(&inp["p"]),
            &nums(&inp["y"]),
            opt_nums(inp.get("sample_weight")).as_deref(),
        )
        .map(|b| json!({"brier": f(b)})),
        "micro_f1" => metrics::micro_f1(
            num(&inp["tp"]),
            num(&inp["fp"]),
            num(&inp["fn"]),
            num(&inp["zero_division"]),
        )
        .map(|v| json!({"f1": f(v)})),
        "routed_micro_f1" => (|| {
            let m = RoutedMicroF1::new(
                &counts(&inp["small_counts"]),
                &counts(&inp["large_counts"]),
                num(&inp["zero_division"]),
            )?;
            let scores = inp["masks"]
                .as_array()
                .expect("masks")
                .iter()
                .map(|mask| m.score(&bools(mask)).map(f))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(json!({ "scores": scores }))
        })(),
        // router
        "router_load" => {
            #[cfg(feature = "json")]
            {
                router_load(inp)
            }
            #[cfg(not(feature = "json"))]
            {
                return None;
            }
        }
        "router_fit" => router_fit(inp),
        other => panic!("unknown golden function {other:?}"),
    };
    Some(out)
}

fn router_fields(r: &ucci::Router) -> Map<String, Value> {
    let mut m = Map::new();
    m.insert("x".into(), fs_(r.calibrator().x()));
    m.insert("y".into(), fs_(r.calibrator().y()));
    m.insert("theta".into(), f(r.theta()));
    m.insert("c_small".into(), f(r.costs().c_small));
    m.insert("c_large".into(), f(r.costs().c_large));
    m.insert("cost_model".into(), json!(r.costs().model.as_str()));
    m.insert("tau".into(), r.tau().map_or(Value::Null, f));
    m.insert("grid_step".into(), f(r.grid_step()));
    m
}

#[cfg(feature = "json")]
fn router_load(inp: &Map<String, Value>) -> Result<Value, UcciError> {
    let text = inp["json_text"].as_str().expect("json_text");
    let r = ucci::Router::from_json(text)?;
    let queries = nums(&inp["queries"]);
    let mut m = router_fields(&r);
    m.insert(
        "created_by".into(),
        r.created_by().map_or(Value::Null, |s| json!(s)),
    );
    let routes = r.route_many(&queries)?;
    m.insert(
        "p_hat".into(),
        Value::Array(routes.iter().map(|d| f(d.p_hat)).collect()),
    );
    m.insert(
        "escalate".into(),
        Value::Array(routes.iter().map(|d| json!(d.escalate)).collect()),
    );
    // Scalar API agrees with the batch API.
    for (q, d) in queries.iter().zip(&routes) {
        assert_eq!(r.escalate(*q)?, d.escalate);
        assert_eq!(r.error_probability(*q)?.to_bits(), d.p_hat.to_bits());
    }
    // Writing and reading back is exact.
    let again = ucci::Router::from_json(&r.to_json())?;
    assert_eq!(
        router_fields(&again),
        router_fields(&r),
        "round trip changed the router"
    );
    Ok(Value::Object(m))
}

fn router_fit(inp: &Map<String, Value>) -> Result<Value, UcciError> {
    let c = costs(inp)?;
    let builder = RouterBuilder::new(c)?.grid_step(num(&inp["grid_step"]))?;
    let (u_cal, e_cal) = (nums(&inp["u_cal"]), nums(&inp["e_cal"]));
    let calibrated = match opt_nums(inp.get("w_cal")) {
        Some(w) => builder.calibrate_weighted(&u_cal, &e_cal, &w)?,
        None => builder.calibrate(&u_cal, &e_cal)?,
    };
    let (u_val, s_val, l_val) = (
        nums(&inp["u_val"]),
        nums(&inp["small_val"]),
        nums(&inp["large_val"]),
    );
    let router = match inp.get("tau") {
        Some(Value::Null) | None => {
            calibrated.choose_threshold_for_budget(&u_val, &s_val, &l_val, num(&inp["budget"]))?
        }
        Some(tau) => calibrated.choose_threshold(&u_val, &s_val, &l_val, num(tau))?,
    };
    let test = router.evaluate(
        &nums(&inp["u_test"]),
        &nums(&inp["small_test"]),
        &nums(&inp["large_test"]),
    )?;
    let mut m = router_fields(&router);
    m.insert("choice".into(), choice(router.choice().expect("chosen")));
    m.insert("test".into(), choice(&test));
    m.insert("n_samples".into(), json!(router.calibrator().n_samples()));
    Ok(Value::Object(m))
}

// ---------------------------------------------------------------- comparison

#[derive(Default)]
struct Stats {
    floats: usize,
    exact: usize,
    max_diff: f64,
    cases: usize,
    skipped: usize,
    failures: Vec<String>,
}

fn compare(got: &Value, want: &Value, tol: f64, path: &str, st: &mut Stats) {
    match (got, want) {
        (Value::Object(g), Value::Object(w)) => {
            let mut gk: Vec<&String> = g.keys().collect();
            let mut wk: Vec<&String> = w.keys().collect();
            gk.sort();
            wk.sort();
            if gk != wk {
                st.failures.push(format!("{path}: keys {gk:?} != {wk:?}"));
                return;
            }
            for k in wk {
                compare(&g[k], &w[k], tol, &format!("{path}.{k}"), st);
            }
        }
        (Value::Array(g), Value::Array(w)) => {
            if g.len() != w.len() {
                st.failures
                    .push(format!("{path}: length {} != {}", g.len(), w.len()));
                return;
            }
            for (i, (a, b)) in g.iter().zip(w).enumerate() {
                compare(a, b, tol, &format!("{path}[{i}]"), st);
            }
        }
        (Value::Number(_), Value::Number(_)) if want.is_f64() || got.is_f64() => {
            let (a, b) = (num(got), num(want));
            st.floats += 1;
            let d = (a - b).abs();
            if a.to_bits() == b.to_bits() {
                st.exact += 1;
            } else if std::env::var_os("UCCI_GOLDEN_VERBOSE").is_some() {
                eprintln!("not bit-identical: {path}: rust {a:?} python {b:?}");
            }
            st.max_diff = st.max_diff.max(d);
            if d.is_nan() || d > tol {
                st.failures.push(format!(
                    "{path}: {a:?} != {b:?} (diff {d:e}, tolerance {tol:e})"
                ));
            }
        }
        _ => {
            if got != want {
                st.failures.push(format!("{path}: {got} != {want}"));
            }
        }
    }
}

fn check_error(err: &UcciError, want: &Value, tol: f64, path: &str, st: &mut Stats) {
    match want.get("kind").and_then(Value::as_str) {
        Some("infeasible") => match err {
            UcciError::Infeasible {
                best_accuracy,
                best_theta,
                ..
            } => {
                if let Some(v) = want.get("best_accuracy") {
                    compare(
                        &f(*best_accuracy),
                        v,
                        tol,
                        &format!("{path}.best_accuracy"),
                        st,
                    );
                    compare(
                        &f(*best_theta),
                        &want["best_theta"],
                        tol,
                        &format!("{path}.best_theta"),
                        st,
                    );
                }
            }
            other => st
                .failures
                .push(format!("{path}: expected Infeasible, got {other:?}")),
        },
        Some("over_budget") => match err {
            UcciError::OverBudget {
                min_cost,
                cheapest_theta,
                ..
            } => {
                if let Some(v) = want.get("min_cost") {
                    compare(&f(*min_cost), v, tol, &format!("{path}.min_cost"), st);
                    compare(
                        &f(*cheapest_theta),
                        &want["cheapest_theta"],
                        tol,
                        &format!("{path}.cheapest_theta"),
                        st,
                    );
                }
            }
            other => st
                .failures
                .push(format!("{path}: expected OverBudget, got {other:?}")),
        },
        _ => {
            if matches!(
                err,
                UcciError::Infeasible { .. } | UcciError::OverBudget { .. }
            ) {
                st.failures.push(format!(
                    "{path}: Python raised {} but Rust reports {err:?}",
                    want["type"]
                ));
            }
        }
    }
}

// ---------------------------------------------------------------- the test

#[test]
fn golden_vectors() {
    let Some(dir) = golden_dir() else { return };
    let mut total = Stats::default();
    #[cfg(feature = "json")]
    let mut written: Vec<(String, String)> = Vec::new();

    for name in FILES {
        let doc = read_json(&dir.join(name));
        assert_eq!(doc["schema"], "ucci-golden/1", "{name}: unknown schema");
        let tol = num(&doc["tolerance"]);
        let data = doc.get("data").cloned().unwrap_or(Value::Null);
        let mut st = Stats::default();
        for case in doc["cases"].as_array().expect("cases") {
            let id = case["id"].as_str().expect("id");
            let func = case["fn"].as_str().expect("fn");
            let path = format!("{name}::{id}");
            let inp = resolve(&case["input"], &data);
            let Some(result) = run(func, &inp) else {
                st.skipped += 1;
                continue;
            };
            st.cases += 1;
            match (result, case.get("expected"), case.get("error")) {
                (Ok(got), Some(want), None) => {
                    let tol = if func == "default_grid" { 0.0 } else { tol };
                    compare(&got, want, tol, &path, &mut st);
                    #[cfg(feature = "json")]
                    if func == "router_load" {
                        let python_text = inp["json_text"].as_str().unwrap();
                        let r = ucci::Router::from_json(python_text).expect("parsed above");
                        let rust_text = r.to_json();
                        // Files saved by the Python package and by this crate are
                        // byte-identical apart from the created_by value.
                        if id.starts_with("saved_") {
                            let created = format!(
                                "\"created_by\": {}",
                                serde_json::to_string(r.created_by().unwrap()).unwrap()
                            );
                            let ours = format!("\"created_by\": \"{}\"", ucci::router::CREATED_BY);
                            if rust_text.replace(&ours, &created) != python_text {
                                st.failures.push(format!(
                                    "{path}: Rust writes a different layout than Python:\n{rust_text}"
                                ));
                            }
                        }
                        written.push((id.to_string(), rust_text));
                    }
                }
                (Err(e), Some(_), None) => st
                    .failures
                    .push(format!("{path}: Rust returned an error: {e}")),
                (Ok(got), None, Some(err)) => st.failures.push(format!(
                    "{path}: Python raised {} ({}) but Rust returned {got}",
                    err["type"], err["message"]
                )),
                (Err(e), None, Some(err)) => check_error(&e, err, tol, &path, &mut st),
                _ => panic!("{path}: a case needs exactly one of expected and error"),
            }
        }
        eprintln!(
            "{name}: {} cases ({} skipped for disabled features), {} floats compared, {} bit-identical, max abs diff {:e}",
            st.cases, st.skipped, st.floats, st.exact, st.max_diff
        );
        total.cases += st.cases;
        total.floats += st.floats;
        total.exact += st.exact;
        total.max_diff = total.max_diff.max(st.max_diff);
        total.failures.extend(st.failures);
    }
    eprintln!(
        "total: {} cases, {} floats, {} bit-identical, max abs diff {:e}",
        total.cases, total.floats, total.exact, total.max_diff
    );
    if !total.failures.is_empty() {
        let shown: Vec<&String> = total.failures.iter().take(40).collect();
        panic!(
            "{} golden mismatches (first {}):\n{}",
            total.failures.len(),
            shown.len(),
            shown
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join("\n")
        );
    }
    assert!(
        total.cases > 250,
        "too few golden cases ran: {}",
        total.cases
    );

    #[cfg(feature = "json")]
    check_rust_written(&dir, &written);
}

/// Routers written by this crate, read back by `tests/test_golden.py`.
#[cfg(feature = "json")]
fn check_rust_written(dir: &std::path::Path, written: &[(String, String)]) {
    let path = dir.join("rust_written_routers.json");
    let cases: Vec<Value> = written
        .iter()
        .map(|(id, text)| json!({"id": id, "json_text": text}))
        .collect();
    let doc = json!({
        "schema": "ucci-golden/1",
        "generator": "rust/tests/golden.rs (UCCI_BLESS=1 cargo test --test golden)",
        "note": "Routers from router.json written back out by the Rust crate; \
                 tests/test_golden.py reads each one with the Python package.",
        "cases": cases,
    });
    let text = serde_json::to_string_pretty(&doc).expect("serializes") + "\n";
    if std::env::var("UCCI_BLESS").as_deref() == Ok("1") {
        fs::write(&path, &text).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        eprintln!("wrote {} ({} routers)", path.display(), written.len());
        return;
    }
    match fs::read_to_string(&path) {
        Ok(stored) => assert!(
            stored == text,
            "{} is out of date with the Rust writer; rerun with UCCI_BLESS=1 after an intended change",
            path.display()
        ),
        Err(_) => panic!(
            "{} is missing; create it with UCCI_BLESS=1 cargo test --test golden",
            path.display()
        ),
    }
}
