#!/usr/bin/env python3
"""Utility-weight sensitivity of the composition analysis.

Reads results/paper2_expectation_prevalence.json (exact-expectation outcome rates per producer and
prevalence) and recomputes U = SAR - lambda*Adverse - mu*Wrongful for the XAIR gate over a grid of
lambda and mu, so that the comparison of the VLM producers with the perception-free
always-nominal producer and with revoke-all does not rest on the single default (5, 1).

Output: results/paper2_weight_sensitivity.json
"""
from __future__ import annotations

import json
from pathlib import Path

RES = Path(__file__).resolve().parent / "results"
LAMS = (1, 2, 5, 10, 20)
MUS = (0.2, 1.0)


def main() -> None:
    d = json.loads((RES / "paper2_expectation_prevalence.json").read_text())
    out = {"lambdas": LAMS, "mus": MUS, "cells": {}}
    for prod in ("qwen7b_blind", "always_nominal_7b_ages"):
        pd = d["producers"][prod]
        cells = {"balanced": pd["balanced"], **{f"prev={k}": v for k, v in pd["prevalence"].items()}}
        for prev, gates in cells.items():
            x = gates["xair"]
            for mu in MUS:
                row = {}
                for lam in LAMS:
                    u = x["success"] - lam * x["adverse"] - mu * x["wrongful"]
                    ra = -mu * x["valid_correct"]
                    row[str(lam)] = {"U_xair": u, "U_revoke_all": ra, "beats_revoke_all": u > ra}
                out["cells"][f"{prod}|{prev}|mu={mu}"] = row
    (RES / "paper2_weight_sensitivity.json").write_text(json.dumps(out, indent=1))
    for k, row in out["cells"].items():
        if "balanced" in k or "prev=0.1|" in k:
            print(k, {l: (round(v["U_xair"], 2), round(v["U_revoke_all"], 2)) for l, v in row.items()})


if __name__ == "__main__":
    main()
