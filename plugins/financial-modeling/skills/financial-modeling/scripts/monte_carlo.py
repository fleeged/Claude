"""
Monte Carlo simulation engine for financial models.

Propagates uncertainty in model inputs through to a distribution of outputs,
optionally with correlation between inputs, and reports confidence intervals,
downside risk metrics, and a ranking of which inputs actually drive the spread.

Sampling uses a Gaussian copula: correlated standard normals are mapped to
uniforms, then through each variable's inverse CDF. This keeps the marginal
distributions exactly as specified while inducing dependence between them, and
it works across mixed distribution families.

Note that the copula correlation you supply is the linear correlation of the
underlying normals, not the realized rank correlation of the outputs. The two
are related by rho_spearman = (6 / pi) * arcsin(rho / 2), so a requested 0.70
shows up as roughly 0.68 in the samples. The gap is under 2 points across the
usual range and in the same direction, so it rarely matters in practice — but
do not be surprised when a measured correlation reads slightly low.

Depends only on numpy and pandas. The normal CDF and its inverse are
implemented locally so scipy is not required.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Normal CDF / inverse CDF (no scipy dependency)
# --------------------------------------------------------------------------

# Acklam's rational approximation to the inverse normal CDF (|error| < 1.15e-9).
_A = (
    -3.969683028665376e01,
    2.209460984245205e02,
    -2.759285104469687e02,
    1.383577518672690e02,
    -3.066479806614716e01,
    2.506628277459239e00,
)
_B = (
    -5.447609879822406e01,
    1.615858368580409e02,
    -1.556989798598866e02,
    6.680131188771972e01,
    -1.328068155288572e01,
)
_C = (
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e00,
    -2.549732539343734e00,
    4.374664141464968e00,
    2.938163982698783e00,
)
_D = (
    7.784695709041462e-03,
    3.224671290700398e-01,
    2.445134137142996e00,
    3.754408661907416e00,
)
_P_LOW = 0.02425


def _norm_ppf(u: np.ndarray) -> np.ndarray:
    """Inverse standard normal CDF, vectorized."""
    u = np.clip(np.asarray(u, dtype=float), 1e-15, 1 - 1e-15)
    out = np.empty_like(u)

    lo = u < _P_LOW
    hi = u > 1 - _P_LOW
    mid = ~(lo | hi)

    if lo.any():
        q = np.sqrt(-2 * np.log(u[lo]))
        out[lo] = (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / (
            (((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1
        )

    if hi.any():
        q = np.sqrt(-2 * np.log(1 - u[hi]))
        out[hi] = -(((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / (
            (((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1
        )

    if mid.any():
        q = u[mid] - 0.5
        r = q * q
        out[mid] = (
            (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * q
        ) / (((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1)

    return out


def _erf(x: np.ndarray) -> np.ndarray:
    """Abramowitz & Stegun 7.1.26 approximation to erf (|error| < 1.5e-7)."""
    x = np.asarray(x, dtype=float)
    sign = np.sign(x)
    z = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * z)
    poly = (
        (((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592
    ) * t
    return sign * (1.0 - poly * np.exp(-z * z))


def _norm_cdf(z: np.ndarray) -> np.ndarray:
    """Standard normal CDF, vectorized."""
    return 0.5 * (1.0 + _erf(np.asarray(z, dtype=float) / np.sqrt(2.0)))


# --------------------------------------------------------------------------
# Distributions
# --------------------------------------------------------------------------


class Distribution(ABC):
    """A marginal distribution, defined by its inverse CDF.

    Any object exposing ``ppf(u)`` for uniforms ``u`` in (0, 1) works with the
    simulator, so scipy.stats frozen distributions can be dropped in directly.
    """

    @abstractmethod
    def ppf(self, u: np.ndarray) -> np.ndarray:
        """Map uniforms in (0, 1) to values of this distribution."""


@dataclass(frozen=True)
class Normal(Distribution):
    """Normal distribution. Symmetric — appropriate for margins and rates."""

    mean: float
    std: float

    def __post_init__(self) -> None:
        if self.std < 0:
            raise ValueError("std must be non-negative")

    def ppf(self, u: np.ndarray) -> np.ndarray:
        return self.mean + self.std * _norm_ppf(u)


@dataclass(frozen=True)
class LogNormal(Distribution):
    """Log-normal distribution, parameterized by median and log-scale sigma.

    Strictly positive and right-skewed — the natural choice for quantities that
    cannot go negative, such as revenue, exit multiples, or terminal values.
    """

    median: float
    sigma: float

    def __post_init__(self) -> None:
        if self.median <= 0:
            raise ValueError("median must be positive")
        if self.sigma < 0:
            raise ValueError("sigma must be non-negative")

    def ppf(self, u: np.ndarray) -> np.ndarray:
        return self.median * np.exp(self.sigma * _norm_ppf(u))


@dataclass(frozen=True)
class Uniform(Distribution):
    """Uniform distribution — use when only bounds are known, nothing more."""

    low: float
    high: float

    def __post_init__(self) -> None:
        if self.high < self.low:
            raise ValueError("high must be >= low")

    def ppf(self, u: np.ndarray) -> np.ndarray:
        return self.low + np.asarray(u, dtype=float) * (self.high - self.low)


@dataclass(frozen=True)
class Triangular(Distribution):
    """Triangular distribution from a three-point (low / mode / high) estimate.

    The standard choice for expert-elicited assumptions where a most-likely
    value is known but the shape is not.
    """

    low: float
    mode: float
    high: float

    def __post_init__(self) -> None:
        if not self.low <= self.mode <= self.high:
            raise ValueError("require low <= mode <= high")

    def ppf(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        span = self.high - self.low
        if span == 0:
            return np.full_like(u, self.low)
        c = (self.mode - self.low) / span
        left = u <= c
        out = np.empty_like(u)
        out[left] = self.low + np.sqrt(u[left] * span * (self.mode - self.low))
        out[~left] = self.high - np.sqrt((1 - u[~left]) * span * (self.high - self.mode))
        return out


@dataclass(frozen=True)
class PERT(Distribution):
    """PERT (beta-PERT) distribution from a three-point estimate.

    Like Triangular but smooth, and it concentrates more weight near the mode —
    generally a better model of expert judgement. ``lambd`` controls confidence
    in the mode; 4.0 is the classic PERT value, higher is tighter.

    Requires scipy for the beta inverse CDF. Use Triangular if scipy is not
    available.
    """

    low: float
    mode: float
    high: float
    lambd: float = 4.0

    def __post_init__(self) -> None:
        if not self.low <= self.mode <= self.high:
            raise ValueError("require low <= mode <= high")

    def ppf(self, u: np.ndarray) -> np.ndarray:
        try:
            from scipy.stats import beta as _beta
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "PERT requires scipy (pip install scipy). Use Triangular for a "
                "scipy-free three-point distribution."
            ) from exc

        span = self.high - self.low
        if span == 0:
            return np.full_like(np.asarray(u, dtype=float), self.low)
        mean = (self.low + self.lambd * self.mode + self.high) / (self.lambd + 2)
        if np.isclose(mean, self.mode):
            alpha = 1 + self.lambd / 2
        else:
            alpha = ((mean - self.low) * (2 * self.mode - self.low - self.high)) / (
                (self.mode - mean) * span
            )
        beta_param = alpha * (self.high - mean) / (mean - self.low)
        alpha = max(alpha, 1e-6)
        beta_param = max(beta_param, 1e-6)
        return self.low + span * _beta.ppf(u, alpha, beta_param)


@dataclass(frozen=True)
class Fixed(Distribution):
    """A constant. Useful for pinning one input while others vary."""

    value: float

    def ppf(self, u: np.ndarray) -> np.ndarray:
        return np.full_like(np.asarray(u, dtype=float), self.value)


@dataclass(frozen=True)
class Custom(Distribution):
    """Wrap an arbitrary inverse-CDF callable."""

    ppf_func: Callable[[np.ndarray], np.ndarray]

    def ppf(self, u: np.ndarray) -> np.ndarray:
        return np.asarray(self.ppf_func(u), dtype=float)


# --------------------------------------------------------------------------
# Correlation handling
# --------------------------------------------------------------------------


def _build_correlation_matrix(
    names: Sequence[str],
    correlations: Mapping[tuple[str, str], float] | np.ndarray | pd.DataFrame | None,
) -> np.ndarray | None:
    """Assemble a full correlation matrix from pairwise entries."""
    if correlations is None:
        return None

    n = len(names)
    index = {name: i for i, name in enumerate(names)}

    if isinstance(correlations, pd.DataFrame):
        return correlations.reindex(index=list(names), columns=list(names)).to_numpy(dtype=float)

    if isinstance(correlations, np.ndarray):
        if correlations.shape != (n, n):
            raise ValueError(f"correlation matrix must be {n}x{n}, got {correlations.shape}")
        return correlations.astype(float)

    corr = np.eye(n)
    for (a, b), rho in correlations.items():
        if a not in index or b not in index:
            raise KeyError(f"correlation references unknown variable: {(a, b)}")
        if not -1.0 <= rho <= 1.0:
            raise ValueError(f"correlation for {(a, b)} must be in [-1, 1], got {rho}")
        corr[index[a], index[b]] = rho
        corr[index[b], index[a]] = rho
    return corr


def _cholesky_psd(corr: np.ndarray) -> np.ndarray:
    """Cholesky factor, repairing a non-positive-semidefinite matrix if needed.

    Inconsistent pairwise correlations (a common hazard when they are elicited
    one pair at a time) produce a matrix with negative eigenvalues. Rather than
    fail, clip the eigenvalues and renormalize to the nearest valid correlation
    matrix.
    """
    try:
        return np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh(corr)
        vals = np.clip(vals, 1e-10, None)
        repaired = vecs @ np.diag(vals) @ vecs.T
        scale = np.sqrt(np.diag(repaired))
        repaired = repaired / np.outer(scale, scale)
        return np.linalg.cholesky(repaired + np.eye(len(corr)) * 1e-12)


def _rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation, ignoring pairs with NaN."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    rx = pd.Series(x[mask]).rank().to_numpy()
    ry = pd.Series(y[mask]).rank().to_numpy()
    if rx.std() == 0 or ry.std() == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------


@dataclass
class SimulationResult:
    """Outcome of a Monte Carlo run.

    Attributes:
        outputs: One column per output metric, one row per iteration.
        draws: The sampled input values, one column per variable.
        failures: Iterations whose output function raised or returned NaN.
        iterations: Total iterations attempted.
    """

    outputs: pd.DataFrame
    draws: pd.DataFrame
    failures: int
    iterations: int

    @property
    def primary(self) -> str:
        """Name of the first output column."""
        return str(self.outputs.columns[0])

    def _series(self, column: str | None = None) -> pd.Series:
        return self.outputs[column or self.primary].dropna()

    def summary(self, column: str | None = None) -> pd.Series:
        """Central tendency and spread statistics for one output."""
        values = self._series(column)
        return pd.Series(
            {
                "iterations": float(self.iterations),
                "valid": float(len(values)),
                "mean": values.mean(),
                "median": values.median(),
                "std": values.std(ddof=1),
                "min": values.min(),
                "p5": values.quantile(0.05),
                "p25": values.quantile(0.25),
                "p75": values.quantile(0.75),
                "p95": values.quantile(0.95),
                "max": values.max(),
                "coef_of_variation": (
                    values.std(ddof=1) / values.mean() if values.mean() else float("nan")
                ),
            }
        )

    def percentile(self, q: float, column: str | None = None) -> float:
        """Value at percentile ``q`` (given as a fraction, e.g. 0.05)."""
        return float(self._series(column).quantile(q))

    def confidence_interval(
        self, level: float = 0.90, column: str | None = None
    ) -> tuple[float, float]:
        """Central confidence interval containing ``level`` of the outcomes."""
        if not 0 < level < 1:
            raise ValueError("level must be in (0, 1)")
        tail = (1 - level) / 2
        values = self._series(column)
        return float(values.quantile(tail)), float(values.quantile(1 - tail))

    def value_at_risk(
        self, level: float = 0.95, column: str | None = None, reference: float | None = None
    ) -> float:
        """Downside shortfall at ``level`` confidence.

        Returned as a positive magnitude: the gap between ``reference`` (the
        mean outcome by default) and the (1 - level) percentile. A 95% VaR of
        320 means that in the worst 5% of runs the outcome is at least 320
        below the reference.
        """
        values = self._series(column)
        ref = values.mean() if reference is None else reference
        return float(ref - values.quantile(1 - level))

    def conditional_value_at_risk(
        self, level: float = 0.95, column: str | None = None, reference: float | None = None
    ) -> float:
        """Expected shortfall: mean gap versus ``reference`` in the worst tail.

        Averages over the outcomes beyond the VaR threshold, so unlike VaR it
        reflects how bad the tail actually gets, not just where it starts.
        """
        values = self._series(column)
        ref = values.mean() if reference is None else reference
        cutoff = values.quantile(1 - level)
        tail = values[values <= cutoff]
        if tail.empty:
            return float("nan")
        return float(ref - tail.mean())

    def probability_below(self, threshold: float, column: str | None = None) -> float:
        """Share of outcomes strictly below ``threshold``."""
        values = self._series(column)
        return float((values < threshold).mean()) if len(values) else float("nan")

    def probability_above(self, threshold: float, column: str | None = None) -> float:
        """Share of outcomes strictly above ``threshold``."""
        values = self._series(column)
        return float((values > threshold).mean()) if len(values) else float("nan")

    def driver_ranking(self, column: str | None = None) -> pd.DataFrame:
        """Rank inputs by how strongly they move the output.

        Uses Spearman rank correlation between each sampled input and the
        output, so it captures monotonic relationships without assuming
        linearity. This is the simulation counterpart to a tornado chart: the
        tornado shows response to a fixed swing, this shows contribution to the
        spread actually observed given each input's own uncertainty.
        """
        target = self.outputs[column or self.primary].to_numpy(dtype=float)
        rows = []
        for name in self.draws.columns:
            values = self.draws[name].to_numpy(dtype=float)
            if np.nanstd(values) == 0:
                continue
            rho = _rank_correlation(values, target)
            rows.append(
                {"variable": name, "rank_correlation": rho, "abs_correlation": abs(rho)}
            )
        if not rows:
            return pd.DataFrame(columns=["variable", "rank_correlation", "abs_correlation"])
        frame = pd.DataFrame(rows).sort_values("abs_correlation", ascending=False)
        total = frame["abs_correlation"].sum()
        frame["contribution_pct"] = (
            frame["abs_correlation"] / total * 100 if total else float("nan")
        )
        return frame.reset_index(drop=True)

    def histogram(self, bins: int = 20, column: str | None = None) -> pd.DataFrame:
        """Bin the output distribution into a frequency table."""
        values = self._series(column)
        counts, edges = np.histogram(values, bins=bins)
        return pd.DataFrame(
            {
                "lower": edges[:-1],
                "upper": edges[1:],
                "count": counts,
                "frequency": counts / counts.sum() if counts.sum() else counts,
            }
        )

    def to_text(self, column: str | None = None, currency: str = "$") -> str:
        """Formatted text report for one output metric."""
        name = column or self.primary
        stats = self.summary(name)
        lo, hi = self.confidence_interval(0.90, name)
        lines = [
            f"Monte Carlo Simulation - {name}",
            "=" * 50,
            "",
            f"  Iterations:        {self.iterations:,}",
            f"  Valid results:     {int(stats['valid']):,}",
        ]
        if self.failures:
            lines.append(
                f"  Discarded:         {self.failures:,} "
                f"({self.failures / self.iterations * 100:.1f}% - see failures)"
            )
        lines += [
            "",
            "Distribution:",
            f"  Mean:              {currency}{stats['mean']:,.0f}",
            f"  Median:            {currency}{stats['median']:,.0f}",
            f"  Std deviation:     {currency}{stats['std']:,.0f}",
            f"  Coef. of variation:{stats['coef_of_variation']:.2f}",
            "",
            "Percentiles:",
            f"  P5:                {currency}{stats['p5']:,.0f}",
            f"  P25:               {currency}{stats['p25']:,.0f}",
            f"  P75:               {currency}{stats['p75']:,.0f}",
            f"  P95:               {currency}{stats['p95']:,.0f}",
            "",
            f"  90% confidence:    {currency}{lo:,.0f} to {currency}{hi:,.0f}",
            "",
            "Downside risk (vs mean):",
            f"  VaR (95%):         {currency}{self.value_at_risk(0.95, name):,.0f}",
            f"  CVaR (95%):        {currency}{self.conditional_value_at_risk(0.95, name):,.0f}",
            "",
        ]

        drivers = self.driver_ranking(name)
        if not drivers.empty:
            lines.append("Key drivers (rank correlation with output):")
            for _, row in drivers.iterrows():
                lines.append(
                    f"  {row['variable']:<22} {row['rank_correlation']:+.3f}"
                    f"   ({row['contribution_pct']:.1f}% of total)"
                )
            lines.append("")

        return "\n".join(lines)


# --------------------------------------------------------------------------
# Simulator
# --------------------------------------------------------------------------


class MonteCarloSimulator:
    """Run a model repeatedly over sampled inputs to get an output distribution.

    Args:
        variables: Input name to its Distribution.
        correlations: Optional pairwise correlations, given either as a mapping
            of ``(name_a, name_b) -> rho`` or as a full matrix. Applied through
            a Gaussian copula, so the realized rank correlation lands slightly
            below the requested value (see the module docstring).
        seed: Seed for reproducible runs.

    Example:
        >>> sim = MonteCarloSimulator(
        ...     {"growth": Normal(0.10, 0.03), "multiple": Triangular(8, 10, 14)},
        ...     correlations={("growth", "multiple"): 0.4},
        ...     seed=42,
        ... )
        >>> result = sim.run(lambda d: 1000 * (1 + d["growth"]) * d["multiple"], 10_000)
        >>> lo, hi = result.confidence_interval(0.90)
    """

    def __init__(
        self,
        variables: Mapping[str, Distribution],
        correlations: Mapping[tuple[str, str], float] | np.ndarray | pd.DataFrame | None = None,
        seed: int | None = None,
    ):
        if not variables:
            raise ValueError("at least one variable is required")
        self.variables = dict(variables)
        self.names = list(self.variables)
        self.correlation = _build_correlation_matrix(self.names, correlations)
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def sample(self, iterations: int = 10_000) -> pd.DataFrame:
        """Draw ``iterations`` correlated samples of every input variable."""
        if iterations < 1:
            raise ValueError("iterations must be >= 1")

        n = len(self.names)
        if self.correlation is None:
            uniforms = self._rng.random((iterations, n))
        else:
            factor = _cholesky_psd(self.correlation)
            normals = self._rng.standard_normal((iterations, n)) @ factor.T
            uniforms = _norm_cdf(normals)

        return pd.DataFrame(
            {name: self.variables[name].ppf(uniforms[:, i]) for i, name in enumerate(self.names)}
        )

    def run(
        self,
        output_func: Callable[[dict[str, float]], float | Mapping[str, float]],
        iterations: int = 10_000,
        draws: pd.DataFrame | None = None,
    ) -> SimulationResult:
        """Evaluate ``output_func`` once per sampled scenario.

        Args:
            output_func: Receives a dict of input name to value for one
                iteration. Returns a single number, or a mapping of metric name
                to number for multiple outputs. Raising is treated as an
                infeasible scenario and recorded as a failure rather than
                aborting the run.
            iterations: Number of scenarios to evaluate.
            draws: Optional pre-sampled inputs, e.g. from ``sample()``. Useful
                for evaluating several models against identical scenarios.

        Returns:
            SimulationResult holding outputs, draws, and failure count.
        """
        if draws is None:
            draws = self.sample(iterations)
        else:
            iterations = len(draws)

        records: list[dict[str, float]] = []
        failures = 0

        for scenario in draws.to_dict("records"):
            try:
                value = output_func(scenario)
            except (ArithmeticError, ValueError, KeyError):
                records.append({})
                failures += 1
                continue

            if isinstance(value, Mapping):
                row = {str(k): float(v) for k, v in value.items()}
            else:
                row = {"output": float(value)}

            if not row or any(not np.isfinite(v) for v in row.values()):
                failures += 1
            records.append(row)

        outputs = pd.DataFrame(records)
        if outputs.empty or outputs.columns.empty:
            outputs = pd.DataFrame({"output": [np.nan] * iterations})

        return SimulationResult(
            outputs=outputs,
            draws=draws.reset_index(drop=True),
            failures=failures,
            iterations=iterations,
        )


# --------------------------------------------------------------------------
# DCF integration
# --------------------------------------------------------------------------

#: Assumption names DCFMonteCarlo knows how to substitute into a DCFModel.
DCF_VARIABLES = frozenset(
    {
        "revenue_growth",
        "ebitda_margin",
        "capex_percent",
        "nwc_percent",
        "tax_rate",
        "terminal_growth",
        "wacc",
        "exit_multiple",
    }
)


class DCFMonteCarlo:
    """Drive a :class:`dcf_model.DCFModel` from sampled assumptions.

    Handles the two things that make a naive DCF simulation wrong. First, the
    Gordon growth terminal value diverges as WACC approaches the terminal
    growth rate, so scenarios where the spread is too small are rejected rather
    than allowed to emit absurd valuations. Second, each iteration runs against
    a fresh copy of the base model, so a drawn assumption cannot leak into the
    next scenario.

    Args:
        model: A configured DCFModel. Assumptions and WACC must already be set;
            drawn variables override them per iteration.
        variables: Distributions keyed by names in :data:`DCF_VARIABLES`.
        correlations: Optional pairwise correlations between those variables.
        terminal_method: ``"growth"`` or ``"multiple"``.
        net_debt: If given, equity value and value per share are also reported.
        shares_outstanding: Share count for per-share output.
        min_wacc_spread: Minimum required WACC minus terminal growth. Scenarios
            below this are discarded as infeasible.
        seed: Seed for reproducible runs.

    Example:
        >>> mc = DCFMonteCarlo(
        ...     model,
        ...     {
        ...         "revenue_growth": Normal(0.10, 0.03),
        ...         "ebitda_margin": Triangular(0.20, 0.25, 0.28),
        ...         "wacc": Normal(0.095, 0.01),
        ...     },
        ...     correlations={("revenue_growth", "ebitda_margin"): 0.3},
        ...     net_debt=200,
        ...     shares_outstanding=50,
        ...     seed=42,
        ... )
        >>> result = mc.run(10_000)
        >>> print(result.to_text("value_per_share"))
    """

    def __init__(
        self,
        model: Any,
        variables: Mapping[str, Distribution],
        correlations: Mapping[tuple[str, str], float] | None = None,
        terminal_method: str = "growth",
        net_debt: float | None = None,
        shares_outstanding: float = 100,
        min_wacc_spread: float = 0.005,
        seed: int | None = None,
    ):
        unknown = set(variables) - DCF_VARIABLES
        if unknown:
            raise KeyError(
                f"unsupported DCF variables: {sorted(unknown)}. "
                f"Supported: {sorted(DCF_VARIABLES)}"
            )
        if terminal_method not in ("growth", "multiple"):
            raise ValueError("terminal_method must be 'growth' or 'multiple'")

        self.base_model = copy.deepcopy(model)
        self.terminal_method = terminal_method
        self.net_debt = net_debt
        self.shares_outstanding = shares_outstanding
        self.min_wacc_spread = min_wacc_spread
        self.simulator = MonteCarloSimulator(variables, correlations, seed)

    def _evaluate(self, scenario: dict[str, float]) -> dict[str, float]:
        model = copy.deepcopy(self.base_model)
        years = model.assumptions["projection_years"]

        for name in ("revenue_growth", "ebitda_margin", "capex_percent", "nwc_percent"):
            if name in scenario:
                model.assumptions[name] = [scenario[name]] * years

        for name in ("tax_rate", "terminal_growth"):
            if name in scenario:
                model.assumptions[name] = scenario[name]

        if "wacc" in scenario:
            model.wacc_components["wacc"] = scenario["wacc"]

        wacc = model.wacc_components["wacc"]
        if self.terminal_method == "growth":
            spread = wacc - model.assumptions["terminal_growth"]
            if spread < self.min_wacc_spread:
                raise ValueError("WACC too close to terminal growth; scenario infeasible")

        model.project_cash_flows()
        valuation = model.calculate_enterprise_value(
            terminal_method=self.terminal_method,
            exit_multiple=scenario.get("exit_multiple"),
        )

        results = {
            "enterprise_value": valuation["enterprise_value"],
            "terminal_percent": valuation["terminal_percent"],
        }

        if self.net_debt is not None:
            equity = model.calculate_equity_value(
                net_debt=self.net_debt, shares_outstanding=self.shares_outstanding
            )
            results["equity_value"] = equity["equity_value"]
            results["value_per_share"] = equity["value_per_share"]

        return results

    def run(self, iterations: int = 10_000) -> SimulationResult:
        """Run the simulation and return the distribution of valuations."""
        return self.simulator.run(self._evaluate, iterations)


# --------------------------------------------------------------------------
# Example usage
# --------------------------------------------------------------------------

if __name__ == "__main__":
    from dcf_model import DCFModel

    model = DCFModel("TechCorp")
    model.set_historical_financials(
        revenue=[800, 900, 1000],
        ebitda=[160, 189, 220],
        capex=[40, 45, 50],
        nwc=[80, 90, 100],
        years=[2022, 2023, 2024],
    )
    model.set_assumptions(
        projection_years=5,
        revenue_growth=[0.15, 0.12, 0.10, 0.08, 0.06],
        ebitda_margin=[0.23, 0.24, 0.25, 0.25, 0.25],
        tax_rate=0.25,
        terminal_growth=0.03,
    )
    model.calculate_wacc(
        risk_free_rate=0.04,
        beta=1.2,
        market_premium=0.07,
        cost_of_debt=0.05,
        debt_to_equity=0.5,
    )

    mc = DCFMonteCarlo(
        model,
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
    print(result.to_text("value_per_share"))
    print(f"P(share price < $40): {result.probability_below(40, 'value_per_share'):.1%}")
