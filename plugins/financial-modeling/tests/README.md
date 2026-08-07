# Tests

```bash
python tests/run_tests.py
```

Requires `numpy` and `pandas`. `scipy` is optional — without it the three PERT
checks skip and the remaining 81 still run. Exits non-zero on any failure.

No pytest dependency: the suite uses a small harness in `_harness.py` so it runs
anywhere the plugin itself runs.

## What is covered

**`test_monte_carlo.py`** — correctness of the simulation engine.

- The embedded numerical approximations against published values: the inverse
  normal CDF at known quantiles (max error 1.3e-9), the normal CDF, and their
  round trip
- Each distribution against its analytic moments, with `Triangular` and `PERT`
  checked against their textbook means, `(a+m+b)/3` and `(a+4m+b)/6`
- Correlation against Gaussian copula theory, `rho_spearman = (6/pi) *
  arcsin(rho/2)` — a requested 0.70 must realize as 0.683, not 0.70, and mixed
  distribution families must not induce correlation where none was asked for
- Repair of a non-positive-semidefinite matrix built from inconsistent pairwise
  correlations
- Statistics against closed-form results: on `Normal(0.05, 0.01)` scaled to 100,
  the mean, the 90% interval, and VaR must hit 105, ±1.645σ, and 1.645σ
- Seeded reproducibility, failure counting, multi-output runs, driver-ranking
  signs, and shared draws across models
- The DCF integration: constant inputs must reproduce the base valuation
  exactly, the equity bridge must hold, the caller's model must never be
  mutated, and scenarios where WACC approaches terminal growth must be rejected
  rather than emitting a divergent value
- Input validation for unknown variables, out-of-range correlations, and
  malformed distribution parameters

**`test_documented_behavior.py`** — every SKILL.md example, executed as written,
asserted against the exact figures the documentation quotes.

It also pins the two traps SKILL.md warns about:

1. `calculate_enterprise_value()` reuses existing projections, so doubling
   EBITDA margin without re-projecting leaves enterprise value unchanged at
   3,071 instead of 7,138 — silently, with no error. The test also confirms the
   asymmetry that makes this subtle: WACC and terminal growth bypass it, so a
   sensitivity sweep over those looks correct while the same code over margin
   reports zero impact.
2. `breakeven_analysis()` bisects assuming the output rises with the variable,
   so searching WACC directly returns the range endpoint (19.53%, EV 1,154)
   against a 3,000 target. The documented sign inversion converges exactly.

**These two tests are written to fail if the traps are ever fixed.** They assert
the broken behavior on purpose, because SKILL.md documents workarounds for it.
If an upstream change fixes either one, the failure is the signal to delete the
warning and the workaround from the docs.
