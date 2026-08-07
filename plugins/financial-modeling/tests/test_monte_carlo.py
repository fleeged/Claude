"""Correctness tests for the Monte Carlo engine.

Checks the numerical approximations against known values, the distributions
against their analytic moments, the copula against its target correlation, and
the DCF integration against hand-computable results.
"""

from __future__ import annotations

import numpy as np

from _harness import Checker, build_model, spearman
from monte_carlo import (
    Custom,
    DCFMonteCarlo,
    Fixed,
    LogNormal,
    MonteCarloSimulator,
    Normal,
    Triangular,
    Uniform,
    _norm_cdf,
    _norm_ppf,
)


def run(c: Checker) -> None:
    c.section("Numerical approximations")
    known_ppf = {
        0.5: 0.0,
        0.975: 1.959963985,
        0.025: -1.959963985,
        0.99: 2.326347874,
        0.001: -3.090232306,
        0.999: 3.090232306,
    }
    err = max(abs(float(_norm_ppf(np.array([p]))[0]) - v) for p, v in known_ppf.items())
    c.check("_norm_ppf matches known quantiles", err < 1e-6, f"max err {err:.2e}")

    known_cdf = {0.0: 0.5, 1.0: 0.841344746, -1.96: 0.024997895, 2.5: 0.993790335}
    err = max(abs(float(_norm_cdf(np.array([z]))[0]) - v) for z, v in known_cdf.items())
    c.check("_norm_cdf matches known values", err < 1e-6, f"max err {err:.2e}")

    u = np.linspace(0.0001, 0.9999, 5000)
    err = np.abs(_norm_cdf(_norm_ppf(u)) - u).max()
    c.check("cdf(ppf(u)) round-trips", err < 1e-6, f"max err {err:.2e}")

    c.section("Marginal distributions")
    s = MonteCarloSimulator(
        {
            "norm": Normal(0.10, 0.02),
            "uni": Uniform(2.0, 6.0),
            "tri": Triangular(1.0, 2.0, 6.0),
            "logn": LogNormal(10.0, 0.3),
            "fix": Fixed(7.0),
        },
        seed=1,
    ).sample(200_000)

    c.check(
        "Normal mean/std",
        abs(s.norm.mean() - 0.10) < 5e-4 and abs(s.norm.std() - 0.02) < 5e-4,
        f"mean {s.norm.mean():.5f} std {s.norm.std():.5f}",
    )
    c.check(
        "Uniform bounds and mean",
        s.uni.min() > 2.0 and s.uni.max() < 6.0 and abs(s.uni.mean() - 4.0) < 0.01,
        f"mean {s.uni.mean():.4f}",
    )
    c.check("Triangular mean = (a+m+b)/3", abs(s.tri.mean() - 3.0) < 0.01, f"mean {s.tri.mean():.4f}")
    c.check("Triangular within bounds", s.tri.min() >= 1.0 and s.tri.max() <= 6.0)
    triangular_std = s.tri.std()
    c.check("LogNormal median", abs(s.logn.median() - 10.0) < 0.05, f"median {s.logn.median():.4f}")
    c.check("LogNormal strictly positive", (s.logn > 0).all())
    c.check("Fixed is constant", s.fix.nunique() == 1)

    res = MonteCarloSimulator({"c": Custom(lambda u: np.asarray(u) ** 2)}, seed=1).sample(50_000)
    c.check("Custom inverse-CDF works", abs(res.c.mean() - 1 / 3) < 0.01, f"mean {res.c.mean():.4f}")

    try:
        from monte_carlo import PERT

        p = MonteCarloSimulator({"p": PERT(1.0, 2.0, 6.0)}, seed=1).sample(200_000)
        c.check("PERT mean = (a+4m+b)/6", abs(p.p.mean() - 15 / 6) < 0.01, f"mean {p.p.mean():.4f}")
        c.check("PERT within bounds", p.p.min() >= 1.0 and p.p.max() <= 6.0)
        c.check(
            "PERT tighter than Triangular",
            p.p.std() < triangular_std,
            f"PERT sd {p.p.std():.4f} vs Tri sd {triangular_std:.4f}",
        )
    except ImportError:
        c.skip("PERT distribution", "scipy not installed")

    c.section("Correlation")
    s = MonteCarloSimulator(
        {"a": Normal(0, 1), "b": Normal(0, 1), "c": Uniform(0, 1)},
        correlations={("a", "b"): 0.7, ("a", "c"): -0.4},
        seed=7,
    ).sample(200_000)
    # A Gaussian copula attenuates: rho_spearman = (6/pi) * arcsin(rho/2).
    target_ab = 6 / np.pi * np.arcsin(0.7 / 2)
    target_ac = 6 / np.pi * np.arcsin(-0.4 / 2)
    ab, ac, bc = spearman(s.a, s.b), spearman(s.a, s.c), spearman(s.b, s.c)
    c.check(
        "rank corr(a,b) matches copula theory",
        abs(ab - target_ab) < 0.01,
        f"got {ab:.4f}, theory {target_ab:.4f}",
    )
    c.check(
        "rank corr(a,c) matches copula theory",
        abs(ac - target_ac) < 0.01,
        f"got {ac:.4f}, theory {target_ac:.4f}",
    )
    c.check("no spurious correlation induced", abs(bc) < 0.01, f"corr(b,c) {bc:.4f}")

    s2 = MonteCarloSimulator({"x": Normal(0, 1), "y": Normal(0, 1)}, seed=3).sample(100_000)
    c.check("independent by default", abs(s2.x.corr(s2.y)) < 0.02, f"got {s2.x.corr(s2.y):.4f}")

    try:
        # Mutually inconsistent pairwise correlations -> non-positive-semidefinite.
        bad = MonteCarloSimulator(
            {"a": Normal(0, 1), "b": Normal(0, 1), "d": Normal(0, 1)},
            correlations={("a", "b"): 0.9, ("b", "d"): 0.9, ("a", "d"): -0.9},
            seed=5,
        ).sample(20_000)
        c.check("non-PSD matrix repaired", bad.notna().all().all() and len(bad) == 20_000)
    except Exception as exc:  # noqa: BLE001
        c.check("non-PSD matrix repaired", False, repr(exc))

    c.section("Reproducibility")
    r1 = MonteCarloSimulator({"x": Normal(0, 1)}, seed=99).sample(1000)
    r2 = MonteCarloSimulator({"x": Normal(0, 1)}, seed=99).sample(1000)
    r3 = MonteCarloSimulator({"x": Normal(0, 1)}, seed=100).sample(1000)
    c.check("same seed reproduces", np.allclose(r1.x, r2.x))
    c.check("different seed differs", not np.allclose(r1.x, r3.x))

    c.section("Statistics against closed-form results")
    res = MonteCarloSimulator({"g": Normal(0.05, 0.01)}, seed=11).run(
        lambda d: 100 * (1 + d["g"]), 100_000
    )
    c.check("mean matches analytic", abs(res.summary()["mean"] - 105.0) < 0.05, f"{res.summary()['mean']:.4f}")
    lo, hi = res.confidence_interval(0.90)
    c.check(
        "90% CI matches analytic +/-1.645 sd",
        abs(lo - (105 - 1.6449)) < 0.05 and abs(hi - (105 + 1.6449)) < 0.05,
        f"[{lo:.3f}, {hi:.3f}]",
    )
    c.check("VaR(95%) ~ 1.645 sd", abs(res.value_at_risk(0.95) - 1.6449) < 0.05, f"{res.value_at_risk(0.95):.4f}")
    c.check("CVaR exceeds VaR", res.conditional_value_at_risk(0.95) > res.value_at_risk(0.95))
    c.check("probability_below(mean) ~ 0.5", abs(res.probability_below(105) - 0.5) < 0.01, f"{res.probability_below(105):.4f}")
    c.check("below + above = 1", abs(res.probability_below(104) + res.probability_above(104) - 1) < 1e-9)

    h = MonteCarloSimulator({"x": Normal(0, 1)}, seed=1).run(lambda d: d["x"], 10_000).histogram(bins=10)
    c.check("histogram bins and frequencies", len(h) == 10 and h["count"].sum() == 10_000 and abs(h.frequency.sum() - 1) < 1e-9)

    c.section("Failure handling and multi-output")

    def flaky(d):
        if d["x"] > 0:
            raise ValueError("infeasible")
        return d["x"]

    res = MonteCarloSimulator({"x": Normal(0, 1)}, seed=4).run(flaky, 10_000)
    c.check("failures counted, run completes", 4700 < res.failures < 5300, f"{res.failures} failures")
    c.check("valid rows exclude failures", int(res.summary()["valid"]) == 10_000 - res.failures)

    res = MonteCarloSimulator({"up": Normal(0, 1), "down": Normal(0, 1), "flat": Fixed(3.0)}, seed=6).run(
        lambda d: {"m1": 2 * d["up"] - 5 * d["down"], "m2": d["up"]}, 20_000
    )
    c.check("multi-output columns", list(res.outputs.columns) == ["m1", "m2"])
    c.check("primary is first column", res.primary == "m1")
    drivers = res.driver_ranking("m1")
    c.check("driver ranking excludes constants", "flat" not in set(drivers.variable))
    c.check("strongest driver ranked first", drivers.iloc[0].variable == "down", f"got {drivers.iloc[0].variable}")
    indexed = drivers.set_index("variable")
    c.check(
        "driver signs correct",
        indexed.loc["down", "rank_correlation"] < 0 < indexed.loc["up", "rank_correlation"],
    )
    c.check("contributions sum to 100%", abs(drivers.contribution_pct.sum() - 100) < 1e-6)

    sim = MonteCarloSimulator({"x": Normal(0, 1)}, seed=8)
    draws = sim.sample(5_000)
    a = sim.run(lambda d: d["x"] * 2, draws=draws)
    b = sim.run(lambda d: d["x"] * 3, draws=draws)
    c.check("shared draws reused across models", np.allclose(a.draws.x, b.draws.x) and len(a.outputs) == 5_000)

    c.section("DCF integration")
    base = build_model()
    base.project_cash_flows()
    base_ev = base.calculate_enterprise_value()["enterprise_value"]

    degenerate = DCFMonteCarlo(
        build_model(),
        {"wacc": Fixed(base.wacc_components["wacc"]), "terminal_growth": Fixed(0.03)},
        seed=1,
    ).run(200)
    c.check(
        "constant inputs reproduce base EV",
        abs(degenerate.outputs.enterprise_value.iloc[0] - base_ev) < 1e-6,
        f"{degenerate.outputs.enterprise_value.iloc[0]:,.2f} vs {base_ev:,.2f}",
    )
    c.check("constant inputs give zero variance", degenerate.outputs.enterprise_value.std() < 1e-9)

    res = DCFMonteCarlo(
        build_model(),
        {"wacc": Normal(0.095, 0.01), "revenue_growth": Normal(0.10, 0.03)},
        net_debt=200,
        shares_outstanding=50,
        seed=2,
    ).run(3_000)
    c.check(
        "reports all four metrics",
        set(res.outputs.columns) == {"enterprise_value", "terminal_percent", "equity_value", "value_per_share"},
    )
    c.check("equity value = EV - net debt", np.allclose(res.outputs.equity_value, res.outputs.enterprise_value - 200))
    c.check("per share = equity / shares", np.allclose(res.outputs.value_per_share, res.outputs.equity_value / 50))
    c.check(
        "WACC drives EV negatively",
        res.driver_ranking("enterprise_value").set_index("variable").loc["wacc", "rank_correlation"] < 0,
    )

    # Gordon growth diverges as WACC approaches terminal growth; those draws
    # must be rejected rather than emitting absurd valuations.
    res = DCFMonteCarlo(
        build_model(),
        {"wacc": Normal(0.05, 0.02), "terminal_growth": Normal(0.04, 0.01)},
        min_wacc_spread=0.005,
        seed=3,
    ).run(3_000)
    c.check("infeasible WACC/growth scenarios rejected", res.failures > 0, f"{res.failures} rejected")
    c.check(
        "no divergent valuations survive",
        res.outputs.enterprise_value.dropna().max() < 1e7,
        f"max EV {res.outputs.enterprise_value.dropna().max():,.0f}",
    )
    c.check("surviving valuations all finite", np.isfinite(res.outputs.enterprise_value.dropna()).all())

    model = build_model()
    before = (list(model.assumptions["revenue_growth"]), model.wacc_components["wacc"])
    DCFMonteCarlo(model, {"revenue_growth": Normal(0.5, 0.1), "wacc": Normal(0.2, 0.01)}, seed=4).run(200)
    after = (list(model.assumptions["revenue_growth"]), model.wacc_components["wacc"])
    c.check("caller's model never mutated", before == after)

    res = DCFMonteCarlo(
        build_model(), {"exit_multiple": Triangular(8, 10, 14)}, terminal_method="multiple", seed=5
    ).run(1_000)
    c.check("exit multiple path runs", res.failures == 0 and res.outputs.enterprise_value.std() > 0)
    c.check(
        "EV monotonic in exit multiple",
        res.driver_ranking("enterprise_value").set_index("variable").loc["exit_multiple", "rank_correlation"] > 0.99,
    )

    c.section("Input validation")
    c.raises("rejects unknown DCF variable", lambda: DCFMonteCarlo(build_model(), {"bogus": Normal(0, 1)}))
    c.raises("rejects empty variables", lambda: MonteCarloSimulator({}))
    c.raises(
        "rejects unknown correlation key",
        lambda: MonteCarloSimulator({"a": Normal(0, 1)}, correlations={("a", "zz"): 0.5}).sample(10),
    )
    c.raises(
        "rejects correlation outside [-1,1]",
        lambda: MonteCarloSimulator(
            {"a": Normal(0, 1), "b": Normal(0, 1)}, correlations={("a", "b"): 1.5}
        ).sample(10),
    )
    c.raises("rejects mode outside triangular bounds", lambda: Triangular(5, 1, 9))
    c.raises("rejects negative standard deviation", lambda: Normal(0, -1))
    c.raises(
        "rejects invalid terminal_method",
        lambda: DCFMonteCarlo(build_model(), {"wacc": Normal(0.1, 0.01)}, terminal_method="bogus"),
    )
