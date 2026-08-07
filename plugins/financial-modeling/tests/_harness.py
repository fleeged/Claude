"""Minimal test harness — no pytest, so the suite runs with numpy and pandas alone."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "skills" / "financial-modeling" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


class Checker:
    """Collects pass/fail/skip results and prints them as they run."""

    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.failures: list[str] = []

    def section(self, name: str) -> None:
        print(f"\n--- {name} ---")

    def check(self, label: str, condition: object, detail: str = "") -> bool:
        suffix = f"  {detail}" if detail else ""
        if condition:
            self.passed += 1
            print(f"PASS  {label}{suffix}")
            return True
        self.failed += 1
        self.failures.append(label)
        print(f"FAIL  {label}{suffix}")
        return False

    def skip(self, label: str, reason: str) -> None:
        self.skipped += 1
        print(f"SKIP  {label}  ({reason})")

    def raises(self, label: str, func, expected=(KeyError, ValueError)) -> bool:
        """Assert that ``func()`` raises one of ``expected``."""
        try:
            func()
        except expected as exc:
            return self.check(label, True, type(exc).__name__)
        except Exception as exc:  # noqa: BLE001 - wrong exception type is a failure
            return self.check(label, False, f"raised {type(exc).__name__}, expected {expected}")
        return self.check(label, False, "no error raised")


def spearman(a, b) -> float:
    """Rank correlation without scipy: Pearson on ranks."""
    return a.rank().corr(b.rank())


def build_model():
    """The reference DCF model used across the suite and in SKILL.md."""
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
    return model
