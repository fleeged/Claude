---
name: financial-modeling
description: "Build DCF valuations and stress them with sensitivity analysis, scenario planning, and Monte Carlo simulation. Use when the user asks to value a company or project, build or run a DCF, compute WACC or terminal value, test how sensitive a valuation is to its assumptions, build best/base/worst cases, or run a simulation over uncertain inputs. Also trigger on 'tornado chart', 'data table', 'two-way sensitivity', 'breakeven', 'probability-weighted', 'confidence interval on valuation', 'value at risk', 'what are the key value drivers', or 'how confident are we in this number'."
---

# Financial Modeling

Executable models for valuation under uncertainty. Three scripts in `scripts/`:

| Script | Provides |
|---|---|
| `dcf_model.py` | `DCFModel` — projections, WACC, terminal value, EV and equity value |
| `sensitivity_analysis.py` | `SensitivityAnalyzer` — one-way, two-way, tornado, scenario, breakeven |
| `monte_carlo.py` | `MonteCarloSimulator`, `DCFMonteCarlo`, distributions, risk metrics |

Requires `numpy` and `pandas`. `scipy` is optional and needed only for the
`PERT` distribution.

## Choosing the right tool

These four techniques answer genuinely different questions. Pick deliberately:

- **DCF** — what is it worth, under one specific set of assumptions?
- **Sensitivity** — how much does the answer move when one input moves? Isolates
  a single driver holding everything else fixed.
- **Scenario** — what happens under a coherent *combination* of assumptions?
  Use when inputs move together (a recession hits growth *and* margins *and*
  the discount rate at once) and a one-at-a-time sweep would understate the
  swing.
- **Monte Carlo** — given uncertainty in every input at once, what is the
  distribution of outcomes, and how likely is any particular result? Use when
  you need a probability, a confidence interval, or a downside quantile rather
  than a point estimate.

A tornado chart and a Monte Carlo driver ranking look similar but are not the
same. The tornado shows response to a *fixed, imposed* swing in each input. The
driver ranking shows contribution to the spread *actually observed* given each
input's own uncertainty — a wildly influential input that is known precisely
will top the tornado and barely register as a driver.

## DCF valuation

```python
from dcf_model import DCFModel

model = DCFModel("TechCorp")
model.set_historical_financials(
    revenue=[800, 900, 1000], ebitda=[160, 189, 220],
    capex=[40, 45, 50], nwc=[80, 90, 100], years=[2022, 2023, 2024],
)
model.set_assumptions(
    projection_years=5,
    revenue_growth=[0.15, 0.12, 0.10, 0.08, 0.06],
    ebitda_margin=[0.23, 0.24, 0.25, 0.25, 0.25],
    tax_rate=0.25, terminal_growth=0.03,
)
model.calculate_wacc(risk_free_rate=0.04, beta=1.2, market_premium=0.07,
                     cost_of_debt=0.05, debt_to_equity=0.5)
model.project_cash_flows()
model.calculate_enterprise_value(terminal_method="growth")   # or "multiple"
model.calculate_equity_value(net_debt=200, shares_outstanding=50)
print(model.generate_summary())
```

WACC uses CAPM for cost of equity. Terminal value supports both perpetuity
growth (Gordon) and exit multiple. `calculate_enterprise_value()` reports
`terminal_percent` — the share of value sitting in the terminal value. Above
roughly 75%, the DCF is mostly an assertion about the terminal assumption and
should be triangulated against multiples.

### Always re-project before re-valuing

`calculate_enterprise_value()` only builds projections `if not self.projections`.
Once they exist it reuses them, so **changing an assumption that feeds the
projections and then re-valuing silently returns the old number** — no error, no
warning:

```python
model.project_cash_flows()
model.calculate_enterprise_value()["enterprise_value"]   # 3,071
model.assumptions["ebitda_margin"] = [0.50] * 5          # double the margin
model.calculate_enterprise_value()["enterprise_value"]   # 3,071 — WRONG, stale
model.project_cash_flows()
model.calculate_enterprise_value()["enterprise_value"]   # 7,138 — correct
```

This bites `revenue_growth`, `ebitda_margin`, `capex_percent`, `nwc_percent`,
and `tax_rate`. It does *not* bite `wacc` or `terminal_growth`, which are
applied at discounting time — which is precisely what makes it dangerous, since
a sensitivity run over WACC looks fine and the same code over margin quietly
reports zero impact.

Whenever you drive this model from `SensitivityAnalyzer`, pass an output
function that re-projects first:

```python
def revalue():
    model.project_cash_flows()
    return model.calculate_enterprise_value()["enterprise_value"]
```

Every example below uses it. `DCFMonteCarlo` and `DCFModel.sensitivity_analysis()`
already handle this internally.

**Other model behavior worth knowing before you rely on it:**

- `project_cash_flows()` sets depreciation equal to capex, so FCF reduces to
  NOPAT less the change in working capital. Reasonable for a steady-state
  business; wrong for one with a large capex cycle or a big existing asset base
  running off. Override the projections directly if that matters.
- Opening working capital is hardcoded at 10% of base revenue, independent of
  the `nwc` history you supply, so year 1's working capital change can be off if
  actual NWC intensity differs. Check `projections["nwc_change"][0]`.
- Gordon growth requires WACC > terminal growth. As the spread narrows the value
  diverges toward infinity; the model does not guard this, so a careless
  sensitivity range can produce nonsense. `DCFMonteCarlo` does guard it.

## Sensitivity analysis

`SensitivityAnalyzer` is model-agnostic — it drives any model through callables,
so it is not limited to `DCFModel`.

```python
from sensitivity_analysis import SensitivityAnalyzer

def revalue():
    """Re-project before valuing, so assumption changes take effect."""
    model.project_cash_flows()
    return model.calculate_enterprise_value()["enterprise_value"]

analyzer = SensitivityAnalyzer(model)

# Rank drivers by impact
tornado = analyzer.tornado_analysis(
    variables={
        "WACC": {"base": 0.095, "low": 0.08, "high": 0.11,
                 "update_func": lambda v: model.wacc_components.__setitem__("wacc", v)},
        "EBITDA margin": {"base": 0.25, "low": 0.20, "high": 0.28,
                          "update_func": lambda v: model.assumptions.__setitem__("ebitda_margin", [v] * 5)},
    },
    output_func=revalue,        # re-projects; see the warning above
)
#      variable  low_output  high_output   impact  impact_pct
#          WACC     4035.83      2481.53  1554.30       50.62
# EBITDA margin     2291.62      3589.10  1297.48       42.26
```

`breakeven_analysis()` bisects for the assumption that hits a target value, but
it assumes the output *rises* with the variable. Enterprise value falls as WACC
rises, so search on the negated variable and flip the result:

```python
neg = analyzer.breakeven_analysis(
    variable_name="negative wacc",
    variable_update=lambda x: model.wacc_components.__setitem__("wacc", -x),
    output_func=revalue,
    target_value=3000, min_search=-0.20, max_search=-0.05, tolerance=1e-5,
)
breakeven_wacc = -neg        # 9.6651% -> EV of exactly 3,000
```

Search directly on WACC instead and the bisection walks the wrong way, returning
the range endpoint with no indication anything went wrong.

Also available: `one_way_sensitivity()` (sweep ±% around a base),
`two_way_sensitivity()` (labeled grid), and `create_data_table()` (Excel-style
two-variable table). `DCFModel.sensitivity_analysis()` is a convenience
shortcut for a two-way grid over `wacc`, `growth`, and `margin`.

`tornado_analysis()` divides by the base output for `impact_pct`, so a base
value at or near zero gives meaningless percentages.

## Scenario planning

```python
results = analyzer.scenario_analysis(
    scenarios={
        "Bull": {"growth": 0.14, "margin": 0.28},
        "Base": {"growth": 0.10, "margin": 0.25},
        "Bear": {"growth": 0.04, "margin": 0.20},
    },
    variable_updates={
        "growth": lambda v: model.assumptions.__setitem__("revenue_growth", [v] * 5),
        "margin": lambda v: model.assumptions.__setitem__("ebitda_margin", [v] * 5),
    },
    output_func=revalue,        # re-projects; see the warning above
    probability_weights={"Bull": 0.25, "Base": 0.50, "Bear": 0.25},
)
#       scenario  probability   output  growth  margin  weighted_output
#           Bull         0.25  3999.94    0.14    0.28           999.99
#           Base         0.50  2990.67    0.10    0.25          1495.33
#           Bear         0.25  1792.72    0.04    0.20           448.18
# Expected Value         1.00  2943.50     NaN     NaN          2943.50
```

Returns one row per scenario plus a probability-weighted **Expected Value** row.

Three things to watch:

- Without `revalue`, all three scenarios return the identical base value. Growth
  and margin only reach the valuation through the projections.
- Weights should sum to 1.0. This is not enforced, and unnormalized weights
  silently produce a wrong expected value.
- `scenario_analysis()` does not restore the model to its base state afterward.
  Every scenario must set every variable it cares about, or it inherits values
  from the previous one. Rebuild the model before reusing it.

## Monte Carlo simulation

Where sensitivity and scenarios ask "what if", simulation asks "how likely".

```python
from monte_carlo import DCFMonteCarlo, Normal, Triangular, LogNormal

mc = DCFMonteCarlo(
    model,
    variables={
        "revenue_growth":  Normal(0.10, 0.03),
        "ebitda_margin":   Triangular(0.20, 0.25, 0.28),
        "terminal_growth": Normal(0.03, 0.005),
        "wacc":            Normal(0.095, 0.010),
    },
    correlations={("revenue_growth", "ebitda_margin"): 0.35},
    net_debt=200, shares_outstanding=50, seed=42,
)
result = mc.run(iterations=10_000)

print(result.to_text("value_per_share"))
result.confidence_interval(0.90, "value_per_share")
result.probability_below(40, "value_per_share")
result.driver_ranking("value_per_share")
```

`DCFMonteCarlo` accepts these variable names: `revenue_growth`,
`ebitda_margin`, `capex_percent`, `nwc_percent`, `tax_rate`, `terminal_growth`,
`wacc`, `exit_multiple`. Each drawn rate is applied flat across all projection
years. It runs every iteration against a deep copy, so the caller's model is
never mutated, and it rejects scenarios where WACC minus terminal growth falls
below `min_wacc_spread` (default 0.5pp) instead of emitting a divergent
valuation. Those show up in `result.failures`. Outputs: `enterprise_value`,
`terminal_percent`, and — when `net_debt` is given — `equity_value` and
`value_per_share`.

For non-DCF models, use `MonteCarloSimulator` directly with any callable:

```python
from monte_carlo import MonteCarloSimulator, Normal, Triangular

sim = MonteCarloSimulator(
    {"growth": Normal(0.10, 0.03), "multiple": Triangular(8, 10, 14)},
    correlations={("growth", "multiple"): 0.4}, seed=42,
)
result = sim.run(lambda d: 1000 * (1 + d["growth"]) * d["multiple"], 10_000)
```

The output function may return a single number or a dict of named metrics. If
it raises, that iteration is recorded as a failure rather than aborting the run
— which is how infeasible parameter combinations are meant to be handled.

### Distributions

| Distribution | Use for |
|---|---|
| `Normal(mean, std)` | Symmetric quantities: margins, rates, growth |
| `LogNormal(median, sigma)` | Strictly positive, right-skewed: revenue, multiples |
| `Triangular(low, mode, high)` | Three-point expert estimates |
| `PERT(low, mode, high, lambd=4)` | Three-point, weighted toward the mode (needs scipy) |
| `Uniform(low, high)` | Bounds known, shape unknown |
| `Fixed(value)` | Pin one input while others vary |
| `Custom(ppf_func)` | Any inverse-CDF callable, including scipy frozen distributions |

Prefer `LogNormal` over `Normal` for anything that cannot go negative. A
`Normal(0.10, 0.06)` growth rate puts real mass below -5%, which may be exactly
what you want, or may be an artifact you did not intend.

### Correlation

Correlations are imposed with a Gaussian copula, which preserves each marginal
exactly while inducing dependence across mixed distribution families. The value
you pass is the copula's linear correlation, and the realized rank correlation
comes out slightly lower — `rho_spearman = (6/pi) * arcsin(rho/2)`, so 0.70
measures as about 0.68. Inconsistent pairwise inputs that produce a non-positive
-semidefinite matrix are repaired to the nearest valid one rather than raising.

Correlation matters more than it looks. Growth and margin are usually
positively correlated, and modeling them independently understates the spread —
the tails are exactly where the decision usually sits.

### Reading the results

`SimulationResult` provides `summary()`, `percentile()`,
`confidence_interval()`, `value_at_risk()`, `conditional_value_at_risk()`,
`probability_below()`, `probability_above()`, `driver_ranking()`,
`histogram()`, and `to_text()`.

`value_at_risk(0.95)` is returned as a positive magnitude: the gap between the
reference (the mean, by default) and the 5th percentile. CVaR averages across
the whole tail beyond that point, so it reflects how bad things get rather than
just where "bad" begins. Pass `reference=` to measure against a base-case value
instead of the mean.

Use 10,000 iterations as a default. 1,000 is enough for central estimates but
too noisy for tail quantiles; go to 50,000+ if a decision turns on P1 or P99.
Always set `seed` for anything reproducible.

## Interpreting output honestly

- **Report ranges, not points.** A single number implies a precision the model
  does not have. Lead with the interval.
- **Distinguish the two kinds of uncertainty.** Simulation captures uncertainty
  *within* a model structure. It says nothing about whether the structure is
  right — a Monte Carlo over a wrong model produces confident nonsense.
- **A narrow confidence interval is not accuracy.** It reflects the input
  distributions you chose. Garbage assumptions produce a tight interval around
  the wrong answer.
- **Check `terminal_percent` and `failures` before quoting anything.** High
  terminal share means the DCF is really a terminal-value assertion; a high
  failure count means much of your input space was infeasible and the surviving
  distribution is truncated.
- Never present these outputs as investment advice. They are analytical tools;
  the assumptions are the analysis, and they belong in any writeup alongside the
  result.
