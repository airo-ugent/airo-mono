#!/usr/bin/env python3
"""Calculate the next CalVer (YYYY.MM.N) given the current version.

Usage: python scripts/next_version.py 2026.5.0
Output: 2026.7.0
"""

import datetime
import re
import sys


def next_version(current: str) -> str:
    parts = re.match(r"(\d{4})\.(\d+)\.(\d+)$", current)
    if not parts:
        raise ValueError(f"Invalid version: {current}")

    cur_year, cur_month, cur_micro = int(parts.group(1)), int(parts.group(2)), int(parts.group(3))
    now = datetime.date.today()

    if now.year > cur_year or (now.year == cur_year and now.month > cur_month):
        # New month → reset micro to 0
        return f"{now.year}.{now.month}.0"
    else:
        # Same month → increment micro
        return f"{cur_year}.{cur_month}.{cur_micro + 1}"


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <current-version>", file=sys.stderr)
        sys.exit(1)
    print(next_version(sys.argv[1]))
