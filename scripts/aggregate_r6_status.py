#!/usr/bin/env python
"""Aggregate MaTBridge R6 + flagship status."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "r6_experiment_status.json"


def _load(name: str) -> dict | None:
    path = RESULTS / name
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    pilot = _load("pilot_r6_summary.json") or {}
    flagship = _load("flagship_9_summary.json") or {}
    linkage = _load("coupling_gap_linkage.json") or {}

    r6 = pilot.get("r6_global", {})
    synth_pass = pilot.get("pilot_status") == "PASS"
    flagship_pass = flagship.get("pilot_status") == "PASS"
    linkage_pass = linkage.get("status") == "PASS"

    pilot_pass = synth_pass and flagship_pass

    status = {
        "schema": "matbridge_experiment_status/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pilot_status": "PASS" if pilot_pass else "FAIL",
        "synthetic_r6": {
            "pilot_status": pilot.get("pilot_status", "MISSING"),
            "r6_global": r6,
        },
        "flagship_9": {
            "pilot_status": flagship.get("pilot_status", "MISSING"),
            "r6_global": flagship.get("r6_global", {}),
        },
        "clause_3_linkage": {
            "status": linkage.get("status", "PENDING"),
            "note": "Analyzed separately; not gated in pilot_status",
        },
        "overall_ready_for_manuscript": pilot_pass,
        "sources": {
            "synthetic": "pilot_r6_summary.json",
            "flagship": "flagship_9_summary.json",
            "linkage": "coupling_gap_linkage.json",
        },
    }
    OUT.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))
    return 0 if pilot_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
