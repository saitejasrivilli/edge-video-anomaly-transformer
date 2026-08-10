#!/usr/bin/env python3
"""DAVIS 2017 preparation — BLOCKED pending license verification.

See docs/datasets.md: DAVIS 2017 licensing information found so far is
contradictory (a BSD statement scoped to the evaluation toolkit vs. an
unverified, internally-inconsistent third-party "CC BY 4.0 non-commercial"
claim). No dataset ingestion happens until this is resolved from an
official source.
"""

from __future__ import annotations

import sys


def main() -> int:
    print(
        "DAVIS 2017 preparation is intentionally blocked.\n"
        "Reason: dataset license is not verified from an official source.\n"
        "See docs/datasets.md ('Dataset: DAVIS 2017') for the full record.\n"
        "Do not proceed until an authoritative license statement is found "
        "and docs/datasets.md is updated.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
