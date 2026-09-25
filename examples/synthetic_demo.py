"""UCCI end to end on SYNTHETIC data: every number this prints is simulated.

The data below is generated, not measured. It shows how the pieces fit
together (paper Section 4 and the evaluation protocol of Section 6.1); it
does not reproduce or approximate any result of the paper. For a real
replication on public data, see ``benchmarks/conll2003``.

Simulated cascade: u(x) comes from a Beta(2, 5) draw, the small model is wrong
with a probability that rises with u(x) but not one for one (so raw u is a
miscalibrated error probability), and the large model is right 93% of the
time regardless of the query (Theorem 1, assumption (ii), holds by
construction).

    python examples/synthetic_demo.py
    python examples/synthetic_demo.py --plot figures/   # also writes two PNGs (needs matplotlib)
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np

import ucci
from ucci import UCCIRouter, bootstrap_ci, ece, policy_cost

C_SMALL, C_LARGE = 1.0, 3.02  # the paper's normalized costs (Section 6.1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n", type=int, default=50_000, help="number of simulated queries")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot", type=Path, default=None, help="directory for the two figures")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    n = args.n
    u = rng.beta(2.0, 5.0, n)
    p_true = 1.0 / (1.0 + np.exp(-10.0 * (u - 0.45)))
    small_ok = (rng.random(n) >= p_true).astype(float)
    large_ok = (rng.random(n) < 0.93).astype(float)

    # Disjoint splits: calibration 30%, validation 20%, test 50% (Section 6.1).
    cal, val, test = np.split(rng.permutation(n), [int(0.3 * n), int(0.5 * n)])
    print("SYNTHETIC DATA: the numbers below illustrate the API; they are not results.\n")

    # Step 1: fit g on the calibration split (Section 4.2), e = 1 when the small model is wrong.
    router = UCCIRouter(c_small=C_SMALL, c_large=C_LARGE)
    router.calibrate(u[cal], 1.0 - small_ok[cal])
    e_test = 1.0 - small_ok[test]
    p_hat_test = router.error_probability(u[test])
    # ece() defaults to 10 equal-width bins; the reliability figure (--plot)
    # bins by deciles of the forecast, so its legend values differ slightly.
    print("ECE on test (10 equal-width bins)")
    print(f"  raw u as an error probability : {ece(u[test], e_test):.3f}")
    print(f"  isotonic p_hat                : {ece(p_hat_test, e_test):.3f}")

    # Step 2: choose theta on the validation split (Eq. 7), target = small + 3/4 of the gap.
    acc_small, acc_large = small_ok[val].mean(), large_ok[val].mean()
    tau = acc_small + 0.75 * (acc_large - acc_small)
    choice = router.choose_threshold(u[val], small_ok[val], large_ok[val], tau)
    print(
        f"\nvalidation: small-only {acc_small:.3f}, large-only {acc_large:.3f}, target tau {tau:.3f}"
    )
    print(
        f"theta* = {choice.theta:.3f}: escalates {choice.escalation_rate:.1%}, "
        f"accuracy {choice.accuracy:.3f}, cost {choice.cost:.2f}"
    )

    # Step 3: route every test query end to end with the chosen threshold.
    res = router.evaluate(u[test], small_ok[test], large_ok[test])
    print(
        f"\ntest: accuracy {res.accuracy:.3f}, cost {res.cost:.2f} vs {C_LARGE:.2f} large-only "
        f"(saving {1 - res.cost / C_LARGE:.1%})"
    )
    esc = router.route(u[test]).escalate
    lo, hi = bootstrap_ci(
        lambda ix: 1.0 - policy_cost(esc[ix], C_SMALL, C_LARGE) / C_LARGE, len(test), n_boot=1000
    )
    print(f"saving, 95% bootstrap CI over test queries (1000 resamples): [{lo:.1%}, {hi:.1%}]")

    # The budget form (Table 2, bottom block): best accuracy at mean cost <= 2.0.
    budget_router = UCCIRouter(c_small=C_SMALL, c_large=C_LARGE).calibrate(
        u[cal], 1.0 - small_ok[cal]
    )
    b = budget_router.choose_threshold_for_budget(u[val], small_ok[val], large_ok[val], budget=2.0)
    print(
        f"\nbudget 2.0: theta {b.theta:.3f}, validation accuracy {b.accuracy:.3f}, cost {b.cost:.2f}"
    )

    # The fitted router is a small JSON file, readable by the CLI and the Rust crate.
    with tempfile.TemporaryDirectory() as tmp:
        path = router.save(Path(tmp) / "router.json")
        again = UCCIRouter.load(path)
        assert np.array_equal(again.route(u[test]).escalate, esc)
        print(f"\nsaved and reloaded {path.name}: identical routing on {len(test)} test queries")

    if args.plot is not None:
        from ucci import plotting

        args.plot.mkdir(parents=True, exist_ok=True)
        fig = plotting.reliability_diagram(
            {"raw u(x)": u[test], "isotonic p_hat": p_hat_test},
            e_test,
            title="Synthetic data, test split",
        )
        fig.savefig(args.plot / "synthetic_reliability.png", dpi=150)
        front = ucci.pareto_frontier(p_hat_test, small_ok[test], large_ok[test], C_SMALL, C_LARGE)
        fig = plotting.pareto_plot(
            {"UCCI threshold sweep": front},
            {
                "UCCI (theta*)": (res.cost, res.accuracy),
                "Small-only": (C_SMALL, small_ok[test].mean()),
                "Large-only": (C_LARGE, large_ok[test].mean()),
            },
            target=tau,
            title="Synthetic data, test split",
            ylabel="Accuracy",
        )
        fig.savefig(args.plot / "synthetic_pareto.png", dpi=150)
        print(f"wrote synthetic_reliability.png and synthetic_pareto.png to {args.plot}")


if __name__ == "__main__":
    main()
