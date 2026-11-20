#!/usr/bin/env python3
"""Run the C++ binary and hold its output to the Python golden CSV.

Usage: parity_check.py <binary> <golden_csv>

Stdlib only, so the CI check needs no extra install.
"""

from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
from pathlib import Path

TOLERANCE = 1e-9


def read_csv(path: Path) -> tuple[list[str], list[list[float]]]:
    with path.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [[float(v) for v in row] for row in reader]
    return header, rows


def main() -> int:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <binary> <golden_csv>", file=sys.stderr)
        return 2

    binary, golden_path = Path(sys.argv[1]), Path(sys.argv[2])
    if not golden_path.is_file():
        print(f"golden CSV not found: {golden_path}", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "cpp_through.csv"
        result = subprocess.run(
            [str(binary), str(out_path)], capture_output=True, text=True, check=False
        )
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr, file=sys.stderr)
            print(f"{binary} exited with {result.returncode}", file=sys.stderr)
            return 1

        got_header, got_rows = read_csv(out_path)

    want_header, want_rows = read_csv(golden_path)

    if got_header != want_header:
        print(f"header mismatch:\n  got  {got_header}\n  want {want_header}", file=sys.stderr)
        return 1
    if len(got_rows) != len(want_rows):
        print(
            f"row count mismatch: got {len(got_rows)}, want {len(want_rows)}", file=sys.stderr
        )
        return 1

    max_abs_diff = 0.0
    worst = None
    for r, (got_row, want_row) in enumerate(zip(got_rows, want_rows)):
        for c, (g, w) in enumerate(zip(got_row, want_row)):
            diff = abs(g - w)
            if diff > max_abs_diff:
                max_abs_diff = diff
                worst = (r, got_header[c], g, w)

    print(f"max abs diff = {max_abs_diff:.3e} (tolerance {TOLERANCE:.0e})")
    if max_abs_diff > TOLERANCE:
        r, col, g, w = worst
        print(f"worst at row {r}, column {col!r}: got {g!r}, want {w!r}", file=sys.stderr)
        return 1

    print(f"parity OK: {len(got_rows)} rows within {TOLERANCE:.0e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
