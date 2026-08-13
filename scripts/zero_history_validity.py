"""Does the zero-history shortlist beat a coin flip?

The headline (+18.4pp) was measured selecting 38,909 units across the whole universe. The
application shows the top B of the *zero-history stratum* only, at B <= 500. Those are not
the same claim, and the second one has never been measured. This script measures it.

Within the stratum every unit has `casualties_36mo == 0`, so a count-based ranking has no
information at all and the honest baseline is a random draw of B units from the same
stratum. The question is whether the model's top B captures more subsequent casualties
than that draw does.

It also checks the ordering defect the engineering review found. `empirical_bayes` blends
`w = 1/(1+k*P)` with `EB = w*P + (1-w)*observed`. With `observed = 0` the whole stratum
collapses to `EB = P/(1+kP)`, which is bounded above by `1/k` -- a different ceiling per
unit type, because corridors and intersections are fitted separately and so have different
dispersions. If the ceilings straddle, one unit type is structurally excluded from the top
of a merged list no matter what its geometry says. The fix is to rank on the SPF prediction
percentile *within* unit type before merging, and this script scores both orderings so the
choice is made on evidence.

    .venv/bin/python scripts/zero_history_validity.py

Writes data/processed/zero-history-validity.json and .md.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR
from src.pipeline import PREDICTORS, build_scored_units, latest_snapshot
from src.spf import fit_and_blend

log = logging.getLogger("zero_history_validity")

BUDGETS = (100, 500, 2000)
RANDOM_DRAWS = 2000
SEED = 20260813


@dataclass
class BudgetResult:
    budget: int
    ordering: str
    captured: float
    capture_pp: float
    n_corridors: int
    n_intersections: int
    boundary_tie_block: int


def _capture(stratum: pd.DataFrame, order_col: str, budget: int, ascending: bool = False):
    """Top-`budget` rows by `order_col`, with a deterministic tiebreak on unit_id."""
    ranked = stratum.sort_values(
        [order_col, "unit_id"], ascending=[ascending, True], kind="mergesort"
    )
    top = ranked.head(budget)
    captured = float(top["holdout_casualties"].sum())

    # How many units share the score sitting exactly on the cut line? Truncating inside a
    # tie block means the last places were decided by the tiebreak, not by the model, and
    # the interface has to say so rather than present them as ranked.
    tie_block = 0
    if len(ranked) > budget:
        boundary = ranked.iloc[budget - 1][order_col]
        tie_block = int((ranked[order_col] == boundary).sum())

    return top, captured, tie_block


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")

    units, _ = build_scored_units(latest_snapshot(), use_cache=True)
    scored, spf_results = fit_and_blend(
        units, target="casualties_36mo", predictors=PREDICTORS, observed_col="casualties_36mo"
    )

    dispersion = {k: v.dispersion for k, v in spf_results.items()}
    ceilings = {k: 1.0 / v for k, v in dispersion.items()}
    log.info("dispersion: %s", {k: round(v, 4) for k, v in dispersion.items()})
    log.info("EB ceiling 1/k per type: %s", {k: round(v, 4) for k, v in ceilings.items()})

    stratum = scored[scored["casualties_36mo"] == 0].copy()
    n_stratum = len(stratum)
    total = float(stratum["holdout_casualties"].sum())
    log.info("zero-history stratum: %d units, %.0f holdout casualties", n_stratum, total)

    if total <= 0:
        log.error("stratum has zero holdout casualties: capture rate is undefined, not 0%%")
        return 1

    # Proposed ordering: percentile of the SPF prediction within unit type, so the two
    # fitted scales are made comparable before they are merged into one list.
    stratum["p_pct_within_type"] = stratum.groupby("unit_type")["spf_prediction"].rank(pct=True)

    orderings = {"eb_estimate": "eb_estimate", "p_pct_within_type": "p_pct_within_type"}

    rng = np.random.default_rng(SEED)
    casualties = stratum["holdout_casualties"].to_numpy(dtype=float)

    results: list[BudgetResult] = []
    random_stats: dict[int, dict[str, float]] = {}

    for budget in BUDGETS:
        if budget > n_stratum:
            log.warning("budget %d exceeds stratum size %d, skipping", budget, n_stratum)
            continue

        draws = np.array(
            [casualties[rng.choice(n_stratum, budget, replace=False)].sum()
             for _ in range(RANDOM_DRAWS)]
        )
        random_stats[budget] = {
            "mean_captured": float(draws.mean()),
            "mean_pp": float(100.0 * draws.mean() / total),
            "p95_captured": float(np.percentile(draws, 95)),
            "p95_pp": float(100.0 * np.percentile(draws, 95) / total),
            "max_captured": float(draws.max()),
        }

        for label, col in orderings.items():
            top, captured, tie_block = _capture(stratum, col, budget)
            results.append(
                BudgetResult(
                    budget=budget,
                    ordering=label,
                    captured=captured,
                    capture_pp=100.0 * captured / total,
                    n_corridors=int((top["unit_type"] == "corridor").sum()),
                    n_intersections=int((top["unit_type"] == "intersection").sum()),
                    boundary_tie_block=tie_block,
                )
            )

    payload = {
        "stratum_units": n_stratum,
        "stratum_holdout_casualties": total,
        "dispersion": dispersion,
        "eb_ceiling_1_over_k": ceilings,
        "random_draws": RANDOM_DRAWS,
        "seed": SEED,
        "random_baseline": random_stats,
        "results": [asdict(r) for r in results],
    }

    out_json = PROCESSED_DIR / "zero-history-validity.json"
    out_json.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# Zero-history shortlist: is the ordering worth anything?",
        "",
        f"Stratum: **{n_stratum:,} units** with no crash history, carrying "
        f"**{total:,.0f} holdout casualties**.",
        "",
        "EB ceiling `1/k` per unit type: "
        + ", ".join(f"`{k}` {v:.4f}" for k, v in sorted(ceilings.items())),
        "",
        "| B | ordering | captured | of stratum | random mean | random p95 | "
        "corridors | intersections | tie block at cut |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        rb = random_stats[r.budget]
        lines.append(
            f"| {r.budget} | `{r.ordering}` | {r.captured:.0f} | {r.capture_pp:.2f}% | "
            f"{rb['mean_captured']:.1f} ({rb['mean_pp']:.2f}%) | {rb['p95_captured']:.0f} | "
            f"{r.n_corridors} | {r.n_intersections} | {r.boundary_tie_block} |"
        )
    out_md = PROCESSED_DIR / "zero-history-validity.md"
    out_md.write_text("\n".join(lines) + "\n")

    print()
    print("=" * 78)
    print("ZERO-HISTORY VALIDITY")
    print("=" * 78)
    print(f"  stratum: {n_stratum:,} units, {total:,.0f} holdout casualties")
    for k, v in sorted(ceilings.items()):
        print(f"  EB ceiling 1/k [{k}]: {v:.4f}")
    print()
    for r in results:
        rb = random_stats[r.budget]
        beats = r.captured > rb["p95_captured"]
        verdict = "BEATS random p95" if beats else "does NOT beat random p95"
        print(
            f"  B={r.budget:<5} {r.ordering:<20} captured {r.captured:>6.0f} "
            f"({r.capture_pp:5.2f}%)  random mean {rb['mean_captured']:6.1f}  -> {verdict}"
        )
        print(
            f"        composition: {r.n_corridors} corridors / {r.n_intersections} "
            f"intersections   tie block at cut: {r.boundary_tie_block}"
        )
    print("=" * 78)
    print(f"  wrote {out_json}")
    print(f"  wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
