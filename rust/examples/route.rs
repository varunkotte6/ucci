//! Load a router fitted by the Python package and route a few queries.
//!
//! ```text
//! cargo run --example route -- path/to/router.json 0.05 0.3 0.8
//! ```
//!
//! With no arguments the example reads `examples/router.json`, a router
//! written by `UCCIRouter.save` in the Python package, and routes a small grid
//! of uncertainty values. Each output line shows `u(x)`, the calibrated error
//! probability `p_hat = g(u)` (Section 4.2) and the decision of Eq. 6.

use std::env;
use std::process::ExitCode;

use ucci::Router;

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let path = args.next().unwrap_or_else(|| {
        concat!(env!("CARGO_MANIFEST_DIR"), "/examples/router.json").to_string()
    });
    let mut us: Vec<f64> = Vec::new();
    for a in args {
        match a.parse::<f64>() {
            Ok(u) => us.push(u),
            Err(_) => {
                eprintln!("not a number: {a}");
                return ExitCode::FAILURE;
            }
        }
    }
    if us.is_empty() {
        us = vec![0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0];
    }

    let router = match Router::load(&path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    println!(
        "router from {path}: theta = {}, {} knots, costs {} / {} ({}), created by {}",
        router.theta(),
        router.calibrator().n_knots(),
        router.costs().c_small,
        router.costs().c_large,
        router.costs().model,
        router.created_by().unwrap_or("unknown"),
    );
    println!("{:>8}  {:>8}  decision", "u(x)", "p_hat");
    for u in us {
        match router.route(u) {
            Ok(r) => println!(
                "{u:>8.4}  {:>8.4}  {}",
                r.p_hat,
                if r.escalate { "large" } else { "small" }
            ),
            Err(e) => println!("{u:>8}  error: {e}"),
        }
    }
    ExitCode::SUCCESS
}
