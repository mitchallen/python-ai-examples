"""Generate (or verify) the test-count badge data in .github/badges/tests.json.

A hand-typed count in the README goes stale the moment someone adds a test and
nobody notices. So the number is generated from pytest itself, and `--check`
re-derives it and fails if the committed file disagrees -- which is what CI
runs, so the badge cannot drift without turning the build red.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BADGE_PATH = REPO_ROOT / ".github" / "badges" / "tests.json"
COLLECTED = re.compile(r"(\d+) tests? collected")


def count_tests() -> int:
    """Ask pytest how many tests exist, without running them."""
    result = subprocess.run(
        # `-o addopts=` clears the repo's own -q: combined with the -q below it
        # becomes double-quiet, which drops the "N tests collected" summary.
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-o", "addopts="],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stdout + result.stderr)
        raise SystemExit("pytest collection failed")

    match = COLLECTED.search(result.stdout)
    if not match:
        sys.stderr.write(result.stdout)
        raise SystemExit("could not find a test count in pytest's output")
    return int(match.group(1))


def badge(count: int) -> dict[str, object]:
    """Shields.io endpoint schema: https://shields.io/badges/endpoint-badge"""
    return {
        "schemaVersion": 1,
        "label": "tests",
        "message": f"{count} passing",
        "color": "brightgreen",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if the committed badge does not match the current count",
    )
    args = parser.parse_args()

    current = badge(count_tests())

    if args.check:
        if not BADGE_PATH.exists():
            print(f"{BADGE_PATH.name} is missing; run `make badge`.", file=sys.stderr)
            return 1
        committed = json.loads(BADGE_PATH.read_text())
        if committed != current:
            print(
                f"Test badge is stale: says {committed.get('message')!r}, "
                f"tests say {current['message']!r}. Run `make badge`.",
                file=sys.stderr,
            )
            return 1
        print(f"Test badge is current: {current['message']}.")
        return 0

    BADGE_PATH.write_text(json.dumps(current, indent=2) + "\n")
    print(f"Wrote {BADGE_PATH.relative_to(REPO_ROOT)}: {current['message']}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
