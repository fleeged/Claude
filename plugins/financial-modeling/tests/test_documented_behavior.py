"""Tests for the behavior SKILL.md documents, including its two known traps.

Every code example in SKILL.md is executed here as written. The two traps are
asserted to still exist: if an upstream change fixes either one, these tests
fail and the documentation needs updating.
"""

from __future__ import annotations

import numpy as np

from _harness import Checker, build_model
from monte_carlo import DCFMonteCarlo, MonteCarloSimulator, Normal, Triangular
from sensitivity_analysis import SensitivityAnalyzer


def run(c: Checker) -> None:
    c.section("Trap 1: stale projections silently return the old valuation")
    model = build_model()
    model.project_cash_flows()
    base = model.calculate_enterprise_value()["enterprise_value"]
    model.assumptions["ebitda_margin"] = [0.50] * 5  # double the margin
    stale = model.calculate_enterprise_value()["enterprise_value"]
    model.project_cash_flows()
    fresh = model.calculate_enterprise_value()["enterprise_value"]
    c.check(
        "doubling margin without re-projecting changes nothing",
        stale == base,
        f"base {base:,.0f} -> stale {stale:,.0f}",
    )
    c.check("re-projecting produces the correct, higher value", fresh > base, f"fresh {fresh:,.0f}")
    c.check(
        "SKILL.md figures still accurate",
        abs(base - 3070.58) < 0.01 and abs(fresh - 7138) < 1,
        f"documented 3,071 / 7,138; got {base:,.0f} / {fresh:,.0f}",
    )

    # WACC and terminal growth are applied at discounting time, so they appear
    # to work without re-projecting. That asymmetry is what makes trap 1 subtle.
    model = build_model()
    model.project_cash_flows()
    before = model.calculate_enterprise_value()["enterprise_value"]
    model.wacc_components["wacc"] = 0.12
    after = model.calculate_enterprise_value()["enterprise_value"]
    c.check("WACC bypasses the trap (applied at discounting)", after != before)

    c.section("Trap 2: breakeven bisection assumes a rising output")
    model = build_model()

    def revalue():
        model.project_cash_flows()
        return model.calculate_enterprise_value()["enterprise_value"]

    analyzer = SensitivityAnalyzer(model)
    naive = analyzer.breakeven_analysis(
        variable_name="wacc",
        variable_update=lambda v: model.wacc_components.__setitem__("wacc", v),
        output_func=revalue,
        target_value=3000,
        min_search=0.05,
        max_search=0.20,
    )
    model.wacc_components["wacc"] = naive
    naive_ev = revalue()
    c.check(
        "searching WACC directly fails silently at the endpoint",
        abs(naive_ev - 3000) > 100 and naive > 0.19,
        f"returned {naive:.4f} -> EV {naive_ev:,.0f}, target 3,000",
    )

    model = build_model()
    analyzer = SensitivityAnalyzer(model)
    negated = analyzer.breakeven_analysis(
        variable_name="negative wacc",
        variable_update=lambda x: model.wacc_components.__setitem__("wacc", -x),
        output_func=revalue,
        target_value=3000,
        min_search=-0.20,
        max_search=-0.05,
        tolerance=1e-5,
    )
    breakeven_wacc = -negated
    model.wacc_components["wacc"] = breakeven_wacc
    converged_ev = revalue()
    c.check(
        "documented sign inversion converges",
        abs(converged_ev - 3000) < 5,
        f"WACC {breakeven_wacc:.4%} -> EV {converged_ev:,.0f}",
    )
    c.check(
        "SKILL.md breakeven figure still accurate",
        abs(breakeven_wacc - 0.096651) < 1e-4,
        f"documented 9.6651%, got {breakeven_wacc:.4%}",
    )

    c.section("SKILL.md example: DCF valuation")
    model = build_model()
    model.project_cash_flows()
    valuation = model.calculate_enterprise_value(terminal_method="growth")
    equity = model.calculate_equity_value(net_debt=200, shares_outstanding=50)
    c.check("enterprise value", abs(valuation["enterprise_value"] - 3070.58) < 0.01, f"{valuation['enterprise_value']:,.2f}")
    c.check("value per share", abs(equity["value_per_share"] - 57.41) < 0.01, f"${equity['value_per_share']:.2f}")
    c.check(
        "terminal share of value",
        abs(valuation["terminal_percent"] - 76.5) < 0.1,
        f"{valuation['terminal_percent']:.1f}% (documented as above the 75% caution line)",
    )
    c.check("summary renders", "DCF Valuation Summary" in model.generate_summary())

    # Documented simplifications.
    proj = model.projections
    c.check(
        "FCF reduces to NOPAT - change in NWC (depreciation cancels capex)",
        abs(proj["fcf"][0] - (proj["nopat"][0] - proj["nwc_change"][0])) < 1e-9,
    )
    c.check(
        "opening NWC hardcoded at 10% of base revenue",
        abs(proj["nwc_change"][0] - 15.0) < 1e-9,
        f"year-1 change {proj['nwc_change'][0]:.2f}",
    )

    c.section("SKILL.md example: tornado analysis")
    model = build_model()
    analyzer = SensitivityAnalyzer(model)
    tornado = analyzer.tornado_analysis(
        variables={
            "WACC": {
                "base": 0.095,
                "low": 0.08,
                "high": 0.11,
                "update_func": lambda v: model.wacc_components.__setitem__("wacc", v),
            },
            "EBITDA margin": {
                "base": 0.25,
                "low": 0.20,
                "high": 0.28,
                "update_func": lambda v: model.assumptions.__setitem__("ebitda_margin", [v] * 5),
            },
        },
        output_func=revalue,
    )
    indexed = tornado.set_index("variable")
    c.check("WACC ranked first", tornado.iloc[0].variable == "WACC")
    c.check(
        "documented tornado figures",
        abs(indexed.loc["WACC", "impact"] - 1554.30) < 0.5
        and abs(indexed.loc["EBITDA margin", "impact"] - 1297.48) < 0.5,
        f"WACC {indexed.loc['WACC', 'impact']:,.2f}, margin {indexed.loc['EBITDA margin', 'impact']:,.2f}",
    )
    c.check("margin registers impact when re-projected", indexed.loc["EBITDA margin", "impact"] > 1.0)

    c.section("SKILL.md example: scenario planning")
    model = build_model()
    analyzer = SensitivityAnalyzer(model)
    scenarios = analyzer.scenario_analysis(
        scenarios={
            "Bull": {"growth": 0.14, "margin": 0.28},
            "Base": {"growth": 0.10, "margin": 0.25},
            "Bear": {"growth": 0.04, "margin": 0.20},
        },
        variable_updates={
            "growth": lambda v: model.assumptions.__setitem__("revenue_growth", [v] * 5),
            "margin": lambda v: model.assumptions.__setitem__("ebitda_margin", [v] * 5),
        },
        output_func=revalue,
        probability_weights={"Bull": 0.25, "Base": 0.50, "Bear": 0.25},
    )
    cases = scenarios[scenarios.scenario != "Expected Value"]
    c.check("three distinct scenario values", cases.output.nunique() == 3)
    c.check(
        "ordered Bull > Base > Bear",
        cases.output.iloc[0] > cases.output.iloc[1] > cases.output.iloc[2],
    )
    expected = scenarios[scenarios.scenario == "Expected Value"].output.iloc[0]
    c.check(
        "expected value is the probability weighting",
        abs(expected - (cases.output * cases.probability).sum()) < 1e-9,
        f"{expected:,.2f}",
    )
    c.check(
        "documented scenario figures",
        abs(cases.output.iloc[0] - 3999.94) < 0.5 and abs(expected - 2943.50) < 0.5,
        f"Bull {cases.output.iloc[0]:,.2f}, expected {expected:,.2f}",
    )

    c.section("SKILL.md example: Monte Carlo")
    mc = DCFMonteCarlo(
        build_model(),
        variables={
            "revenue_growth": Normal(0.10, 0.03),
            "ebitda_margin": Triangular(0.20, 0.25, 0.28),
            "terminal_growth": Normal(0.03, 0.005),
            "wacc": Normal(0.095, 0.010),
        },
        correlations={("revenue_growth", "ebitda_margin"): 0.35},
        net_debt=200,
        shares_outstanding=50,
        seed=42,
    )
    result = mc.run(iterations=10_000)
    lo, hi = result.confidence_interval(0.90, "value_per_share")
    c.check("documented 90% CI", abs(lo - 35.83) < 0.01 and abs(hi - 84.40) < 0.01, f"${lo:.2f} to ${hi:.2f}")
    c.check(
        "documented probability below $40",
        abs(result.probability_below(40, "value_per_share") - 0.116) < 0.001,
        f"{result.probability_below(40, 'value_per_share'):.1%}",
    )
    drivers = result.driver_ranking("value_per_share")
    c.check("WACC is the top driver", drivers.iloc[0].variable == "wacc", f"got {drivers.iloc[0].variable}")
    c.check("report renders", "Monte Carlo Simulation" in result.to_text("value_per_share"))

    sim = MonteCarloSimulator(
        {"growth": Normal(0.10, 0.03), "multiple": Triangular(8, 10, 14)},
        correlations={("growth", "multiple"): 0.4},
        seed=42,
    )
    generic = sim.run(lambda d: 1000 * (1 + d["growth"]) * d["multiple"], 10_000)
    c.check(
        "generic simulator example",
        abs(generic.summary()["mean"] - 11766.43) < 0.01 and generic.failures == 0,
        f"mean {generic.summary()['mean']:,.2f}",
    )

    c.section("SKILL.md claim: DCFMonteCarlo handles the traps internally")
    mc = DCFMonteCarlo(build_model(), {"ebitda_margin": Normal(0.25, 0.05)}, seed=9)
    res = mc.run(500)
    c.check(
        "margin actually moves EV inside DCFMonteCarlo",
        res.outputs.enterprise_value.std() > 1.0,
        f"EV sd {res.outputs.enterprise_value.std():,.2f}",
    )
    c.check(
        "margin correlates positively with EV",
        res.driver_ranking("enterprise_value").set_index("variable").loc["ebitda_margin", "rank_correlation"] > 0.9,
    )
    c.check("no NaN leakage in valid outputs", np.isfinite(res.outputs.enterprise_value.dropna()).all())
