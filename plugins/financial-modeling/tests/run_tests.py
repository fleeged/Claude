#!/usr/bin/env python3
"""Run the financial-modeling test suite.

    python tests/run_tests.py

Requires numpy and pandas. scipy is optional; without it the PERT tests are
skipped. Exits non-zero if anything fails.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _harness import Checker  # noqa: E402

import test_documented_behavior  # noqa: E402
import test_monte_carlo  # noqa: E402

MODULES = [
    ("Monte Carlo correctness", test_monte_carlo),
    ("Documented behavior", test_documented_behavior),
]


def main() -> int:
    checker = Checker()
    started = time.time()

    for title, module in MODULES:
        print(f"\n{'=' * 62}\n{title}\n{'=' * 62}")
        module.run(checker)

    elapsed = time.time() - started
    print(f"\n{'=' * 62}")
    print(
        f"{checker.passed} passed, {checker.failed} failed, "
        f"{checker.skipped} skipped in {elapsed:.1f}s"
    )
    if checker.failures:
        print("\nFailures:")
        for name in checker.failures:
            print(f"  - {name}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
