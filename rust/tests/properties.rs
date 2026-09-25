//! Property tests: invariants that must hold for every input, checked on
//! random inputs with proptest.

use proptest::collection::vec;
use proptest::prelude::*;
use ucci::calibration::{pav, IsotonicCalibrator};
use ucci::policy::{default_grid, evaluate, select_threshold, CostModel, Costs};
use ucci::signal::{token_margin_uncertainty, MarginAccumulator};

/// Weighted mean of `y[j..=k]`.
fn wmean(y: &[f64], w: &[f64], j: usize, k: usize) -> f64 {
    let (mut s, mut t) = (0.0, 0.0);
    for i in j..=k {
        s += w[i] * y[i];
        t += w[i];
    }
    s / t
}

/// Deterministic Fisher-Yates shuffle driven by `seed`.
fn shuffled<T: Clone>(items: &[T], seed: u64) -> Vec<T> {
    let mut out = items.to_vec();
    let mut state = seed | 1;
    for i in (1..out.len()).rev() {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let j = ((state >> 33) as usize) % (i + 1);
        out.swap(i, j);
    }
    out
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 400, ..ProptestConfig::default() })]

    /// PAV returns a non-decreasing sequence with the same weighted sum.
    #[test]
    fn pav_is_monotone_and_preserves_the_weighted_sum(
        data in vec((-10.0f64..10.0, 0.01f64..5.0), 1..80)
    ) {
        let (y, w): (Vec<f64>, Vec<f64>) = data.into_iter().unzip();
        let fit = pav(&y, Some(&w)).unwrap();
        prop_assert_eq!(fit.len(), y.len());
        prop_assert!(fit.windows(2).all(|p| p[0] <= p[1]));
        let s_in: f64 = y.iter().zip(&w).map(|(a, b)| a * b).sum();
        let s_out: f64 = fit.iter().zip(&w).map(|(a, b)| a * b).sum();
        prop_assert!((s_in - s_out).abs() <= 1e-9 * (1.0 + s_in.abs()));
    }

    /// PAV equals the max-min characterization of weighted isotonic
    /// regression, f_i = max_{j <= i} min_{k >= i} mean_w(y[j..=k])
    /// (Barlow et al., 1972), an independent O(n^3) computation.
    #[test]
    fn pav_matches_the_max_min_formula(data in vec((0.0f64..1.0, 0.1f64..3.0), 1..12)) {
        let (y, w): (Vec<f64>, Vec<f64>) = data.into_iter().unzip();
        let fit = pav(&y, Some(&w)).unwrap();
        let n = y.len();
        for (i, &fi) in fit.iter().enumerate() {
            let best = (0..=i)
                .map(|j| (i..n).map(|k| wmean(&y, &w, j, k)).fold(f64::INFINITY, f64::min))
                .fold(f64::NEG_INFINITY, f64::max);
            prop_assert!((fi - best).abs() <= 1e-12, "i={} pav={} max-min={}", i, fi, best);
        }
    }

    /// Fitting a fitted sequence changes nothing, and sorted input passes
    /// through unchanged, bit for bit.
    #[test]
    fn pav_is_idempotent(y in vec(-5.0f64..5.0, 1..60)) {
        let once = pav(&y, None).unwrap();
        prop_assert_eq!(pav(&once, None).unwrap(), once);
        let mut sorted = y.clone();
        sorted.sort_by(f64::total_cmp);
        prop_assert_eq!(pav(&sorted, None).unwrap(), sorted);
    }

    /// With unit weights and 0/1 labels every sum in the fit is exact, so the
    /// knots do not depend on the order of the calibration set at all.
    #[test]
    fn calibrator_is_invariant_to_input_order(
        pts in vec((0u8..25, any::<bool>()), 1..120),
        seed in any::<u64>(),
    ) {
        let u: Vec<f64> = pts.iter().map(|p| f64::from(p.0) / 24.0).collect();
        let e: Vec<f64> = pts.iter().map(|p| if p.1 { 1.0 } else { 0.0 }).collect();
        let a = IsotonicCalibrator::fit(&u, &e).unwrap();
        let perm = shuffled(&(0..u.len()).collect::<Vec<_>>(), seed);
        let u2: Vec<f64> = perm.iter().map(|&i| u[i]).collect();
        let e2: Vec<f64> = perm.iter().map(|&i| e[i]).collect();
        let b = IsotonicCalibrator::fit(&u2, &e2).unwrap();
        prop_assert_eq!(a.x(), b.x());
        prop_assert_eq!(a.y(), b.y());
    }

    /// With real-valued weights, reordering only changes summation order
    /// inside tied groups, so the fits agree to round-off.
    #[test]
    fn weighted_calibrator_is_invariant_to_input_order(
        pts in vec((0u8..15, any::<bool>(), 0.0f64..3.0), 1..100),
        seed in any::<u64>(),
    ) {
        prop_assume!(pts.iter().any(|p| p.2 > 0.0));
        let u: Vec<f64> = pts.iter().map(|p| f64::from(p.0) / 14.0).collect();
        let e: Vec<f64> = pts.iter().map(|p| if p.1 { 1.0 } else { 0.0 }).collect();
        let w: Vec<f64> = pts.iter().map(|p| p.2).collect();
        let a = IsotonicCalibrator::fit_weighted(&u, &e, &w).unwrap();
        let perm = shuffled(&(0..u.len()).collect::<Vec<_>>(), seed);
        let pick = |v: &[f64]| -> Vec<f64> { perm.iter().map(|&i| v[i]).collect() };
        let b = IsotonicCalibrator::fit_weighted(&pick(&u), &pick(&e), &pick(&w)).unwrap();
        for q in 0..=40 {
            let q = f64::from(q) / 40.0;
            prop_assert!((a.predict(q).unwrap() - b.predict(q).unwrap()).abs() <= 1e-12);
        }
    }

    /// An integer weight is the same as repeating the point.
    #[test]
    fn integer_weights_equal_repetition(pts in vec((0u8..10, any::<bool>(), 1u8..4), 1..40)) {
        let u: Vec<f64> = pts.iter().map(|p| f64::from(p.0) / 9.0).collect();
        let e: Vec<f64> = pts.iter().map(|p| if p.1 { 1.0 } else { 0.0 }).collect();
        let w: Vec<f64> = pts.iter().map(|p| f64::from(p.2)).collect();
        let a = IsotonicCalibrator::fit_weighted(&u, &e, &w).unwrap();
        let (mut ur, mut er) = (Vec::new(), Vec::new());
        for p in &pts {
            for _ in 0..p.2 {
                ur.push(f64::from(p.0) / 9.0);
                er.push(if p.1 { 1.0 } else { 0.0 });
            }
        }
        let b = IsotonicCalibrator::fit(&ur, &er).unwrap();
        prop_assert_eq!(a.x(), b.x());
        prop_assert_eq!(a.y(), b.y());
    }

    /// g is non-decreasing, stays within its knot values and clips outside
    /// the calibration range.
    #[test]
    fn calibrated_probability_is_monotone_and_bounded(
        pts in vec((0.0f64..1.0, any::<bool>()), 1..150),
        queries in vec(-0.5f64..1.5, 1..50),
    ) {
        let u: Vec<f64> = pts.iter().map(|p| p.0).collect();
        let e: Vec<f64> = pts.iter().map(|p| if p.1 { 1.0 } else { 0.0 }).collect();
        let g = IsotonicCalibrator::fit(&u, &e).unwrap();
        prop_assert!(g.x().windows(2).all(|p| p[0] < p[1]));
        prop_assert!(g.y().windows(2).all(|p| p[0] <= p[1]));
        let (lo, hi) = (g.y()[0], g.y()[g.n_knots() - 1]);
        let mut q = queries;
        q.sort_by(f64::total_cmp);
        let p = g.predict_many(&q).unwrap();
        prop_assert!(p.windows(2).all(|w| w[0] <= w[1]));
        prop_assert!(p.iter().all(|&v| (lo..=hi).contains(&v)));
        prop_assert_eq!(g.predict(-10.0).unwrap(), lo);
        prop_assert_eq!(g.predict(10.0).unwrap(), hi);
    }

    /// The fitted map is the isotonic regression of the pooled labels, so its
    /// weighted mean over the calibration points equals the error rate.
    #[test]
    fn calibration_preserves_the_error_rate(pts in vec((0u8..30, any::<bool>()), 1..200)) {
        let u: Vec<f64> = pts.iter().map(|p| f64::from(p.0) / 29.0).collect();
        let e: Vec<f64> = pts.iter().map(|p| if p.1 { 1.0 } else { 0.0 }).collect();
        let g = IsotonicCalibrator::fit(&u, &e).unwrap();
        let mean_fit: f64 = g.predict_many(&u).unwrap().iter().sum::<f64>() / u.len() as f64;
        let mean_e: f64 = e.iter().sum::<f64>() / e.len() as f64;
        prop_assert!((mean_fit - mean_e).abs() <= 1e-12);
    }

    /// u(x) lies in [0, 1], does not depend on the order inside each pair,
    /// and the streaming accumulator agrees bit for bit.
    #[test]
    fn uncertainty_is_a_bounded_order_free_average(pairs in vec((0.0f64..=1.0, 0.0f64..=1.0), 1..300)) {
        let u = token_margin_uncertainty(&pairs).unwrap();
        prop_assert!((0.0..=1.0).contains(&u));
        let swapped: Vec<(f64, f64)> = pairs.iter().map(|&(a, b)| (b, a)).collect();
        prop_assert_eq!(token_margin_uncertainty(&swapped).unwrap(), u);
        let mut acc = MarginAccumulator::new();
        for &(a, b) in &pairs {
            acc.push_top2(a, b).unwrap();
        }
        prop_assert_eq!(acc.uncertainty().unwrap(), u);
    }

    /// Eq. 7 on the grid: the selected threshold is feasible, no feasible grid
    /// threshold is cheaper, and ties go to the largest theta.
    #[test]
    fn selected_threshold_is_the_cheapest_feasible_one(
        rows in vec((0.0f64..1.0, any::<bool>(), any::<bool>()), 1..120),
        tau in 0.0f64..1.0,
        sequential in any::<bool>(),
    ) {
        let p: Vec<f64> = rows.iter().map(|r| r.0).collect();
        let small: Vec<f64> = rows.iter().map(|r| if r.1 { 1.0 } else { 0.0 }).collect();
        let large: Vec<f64> = rows.iter().map(|r| if r.2 { 1.0 } else { 0.0 }).collect();
        let model = if sequential { CostModel::Sequential } else { CostModel::Routing };
        let costs = Costs::new(1.0, 3.02, model).unwrap();
        let grid = default_grid();
        let all: Vec<_> = grid.iter().map(|&t| evaluate(&p, &small, &large, t, &costs).unwrap()).collect();
        let feasible: Vec<_> = all.iter().filter(|c| c.accuracy >= tau).collect();
        match select_threshold(&p, &small, &large, tau, &costs, &grid) {
            Ok(c) => {
                prop_assert!(c.accuracy >= tau);
                let min_cost = feasible.iter().map(|c| c.cost).fold(f64::INFINITY, f64::min);
                prop_assert_eq!(c.cost, min_cost);
                let largest = feasible
                    .iter()
                    .filter(|f| f.cost == min_cost)
                    .map(|f| f.theta)
                    .fold(f64::NEG_INFINITY, f64::max);
                prop_assert_eq!(c.theta, largest);
            }
            Err(_) => prop_assert!(feasible.is_empty()),
        }
    }
}

#[cfg(feature = "json")]
mod json {
    use super::*;
    use ucci::Router;

    proptest! {
        #![proptest_config(ProptestConfig { cases: 200, ..ProptestConfig::default() })]

        /// Any valid router survives a JSON round trip bit for bit.
        #[test]
        fn router_json_round_trip_is_exact(
            xs in vec(-1e6f64..1e6, 1..40),
            ys in vec(0.0f64..=1.0, 1..40),
            theta in -2.0f64..2.0,
            c_large in 0.001f64..1e6,
            sequential in any::<bool>(),
            queries in vec(-1e6f64..1e6, 0..20),
        ) {
            let mut x = xs;
            x.sort_by(f64::total_cmp);
            x.dedup();
            let mut y = ys;
            y.sort_by(f64::total_cmp);
            let n = x.len().min(y.len());
            x.truncate(n);
            y.truncate(n);
            let model = if sequential { CostModel::Sequential } else { CostModel::Routing };
            let g = IsotonicCalibrator::from_knots(x, y).unwrap();
            let r = Router::new(g, theta, Costs::new(1.0, c_large, model).unwrap()).unwrap();
            let back = Router::from_json(&r.to_json()).unwrap();
            prop_assert_eq!(back.calibrator(), r.calibrator());
            prop_assert_eq!(back.theta().to_bits(), r.theta().to_bits());
            prop_assert_eq!(back.costs(), r.costs());
            for q in queries {
                prop_assert_eq!(back.route(q).unwrap(), r.route(q).unwrap());
            }
        }
    }
}
