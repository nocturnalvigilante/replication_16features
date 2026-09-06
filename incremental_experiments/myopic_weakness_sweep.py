"""Where does myopic actually break? A 2-axis sweep on Kanix's own machinery.

Motivation
----------
Two candidate weak axes came out of earlier work, each verified on only one
half of the data:

  * COST     -- confirmed on Kanix's own 102-row suite: myopic ratio2 climbs
                0.0377 -> 0.0855 -> 0.1753 as fixed_cost goes 0.005 -> 0.010
                -> 0.015. His grid stops at 0.015, so we do not know whether it
                keeps degrading.
  * RARITY   -- suggested by OUR LowLow regime, which is NOT in his suite at
                all (he only has LowHigh and MediumEven), so it has never been
                tested on his settings.

This sweeps both together, on his families and his solver, so the two axes can
be compared on one footing and their interaction seen.

Everything substantive is Kanix's code -- his `_build_config`, his belief
snapshot, his exact DP, and `genetic_dp.policy.baselines.myopic_greedy` reached
through his `_evaluate_myopic_root`. No Gurobi: exact DP here is backward
induction (`solve_exact_dp_primal`), which needs no LP solver. This file only
loops over settings.

READ THE RIGHT COLUMN
---------------------
ratio2 = (V* - L) / (V* - V_stop) is a regret normalised by the optimal value,
so it inflates wherever V* is near zero -- and rare-variant regimes have the
smallest V* by construction. We proved this on the 2-gene side: LowLow scored
ratio2 0.99 while having the LOWEST absolute regret of any regime. So a rarity
sweep scored on ratio2 alone will manufacture exactly the answer it is looking
for. Three columns are therefore recorded and the analysis reports all three:

  ratio2          normalised regret  -- comparable to Kanix's published numbers
  abs_regret      |V* - L|           -- denominator-free, the honest magnitude
  root_match      myopic's first action == DP's -- a pure decision-quality
                  signal with no normalisation in it at all

A region only counts as genuinely weak if abs_regret and/or root_match agree
with ratio2. Where they disagree, ratio2 is measuring its own denominator.

Grid: 7 allele pairs x 5 fixed_cost x 3 variable_cost x 2 families x 2 presets
      = 420 rows. Costs extend past Kanix's 0.015 ceiling to test whether the
      cost effect saturates. Allele pairs extend below his 0.02 floor.

Resume: one row per task, own JSON, skip guard -- safe to resubmit.

Usage:
    python myopic_weakness_sweep.py --row 0
    python myopic_weakness_sweep.py --summarize
"""
from __future__ import annotations
import argparse
import json
import sys
from itertools import product
from pathlib import Path

ROOT = Path("/net/projects/ranalab/rajhansini/replication_16features")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from genetic_dp.exact_dp.utils import GENOTYPE_STATES
from genetic_dp.exact_dp.solver import solve_exact_dp_primal
from genetic_dp.experiments.core import _build_factorized_multigene_belief_snapshot
from genetic_dp.utils.pedigree_generator import generate_deterministic_pedigree
from scripts.search_multigene_myopic_vs_stop import (
    GENES, FAMILY_CASES, _build_config, _build_child_cpds,
    _evaluate_myopic_root, _stop_value_from_state,
)

OUT = ROOT / "incremental_experiments" / "results" / "myopic_weakness_sweep"

# (label, GeneA, GeneB). UltraRare/VeryRare sit BELOW Kanix's 0.02 floor.
ALLELES = [
    ("UltraRare",  0.005, 0.005),
    ("VeryRare",   0.01,  0.01),
    ("LowLow",     0.02,  0.02),
    ("LowHigh",    0.02,  0.15),   # Kanix's
    ("MixedA",     0.02,  0.10),
    ("MediumEven", 0.08,  0.08),   # Kanix's
    ("HighHigh",   0.15,  0.15),
]
FIXED    = [0.005, 0.010, 0.015, 0.020, 0.030]   # his max is 0.015
VARIABLE = [0.010, 0.020, 0.030]                 # his full range
FAMILIES = ["ThreeGeneration", "Extended"]
PRESETS  = ["Base", "Aggressive"]


def grid():
    rows = []
    for fam, pre, (alab, ga, gb), fc, vc in product(FAMILIES, PRESETS, ALLELES, FIXED, VARIABLE):
        rows.append(dict(family=fam, preset=pre, allele_label=alab,
                         allele_freqs={"GeneA": ga, "GeneB": gb},
                         fixed_cost=fc, variable_cost=vc))
    return rows


def run_row(r):
    pedigree = generate_deterministic_pedigree(FAMILY_CASES[r["family"]])
    config = _build_config(
        pedigree, allele_freqs=r["allele_freqs"], preset_label=r["preset"],
        a_scale=1.0, b_scale=1.0, delta_shift=0.0,
        fixed_cost=r["fixed_cost"], variable_cost=r["variable_cost"],
    )
    belief = _build_factorized_multigene_belief_snapshot(
        pedigree=pedigree, config=config, genes=GENES,
        child_cpds=_build_child_cpds(pedigree), belief_parallelism=1,
        progress_label=None,
    )
    gen_states_full = list(product(GENOTYPE_STATES, repeat=len(GENES)))
    V, policy_dp = solve_exact_dp_primal(
        pedigree.to_list(), gen_states_full, {frozenset(): 1.0}, belief,
        config.a, config.b, config.c, config.delta,
        config.fixed_cost, config.variable_cost,
        genes=GENES, a_gene=config.a_gene, b_gene=config.b_gene,
        c_gene=config.c_gene, delta_gene=config.delta_gene,
    )
    V_root = float(V[frozenset()])
    V_stop = _stop_value_from_state(frozenset(), pedigree=pedigree, config=config,
                                    belief=belief, belief_gene={})
    L, myo_pol = _evaluate_myopic_root(pedigree=pedigree, config=config, belief=belief)
    denom = V_root - V_stop
    rm, rd = myo_pol.get(frozenset()), policy_dp.get(frozenset())
    out = dict(r)
    out.update({
        "V_root": V_root, "V_stop_root": V_stop, "headroom": denom,
        "L_myopic": L,
        "ratio2_myopic": (V_root - L) / denom if abs(denom) > 1e-12 else 0.0,
        "abs_regret_myopic": abs(V_root - L),
        "root_action_myopic": list(rm[:2]) if rm else None,
        "root_action_dp": list(rd[:2]) if rd else None,
        "root_match": bool(rm and rd and tuple(rm[:2]) == tuple(rd[:2])),
        "n_states": len(belief),
    })
    return out


def key(r):
    return (f"{r['family']}_{r['preset']}_{r['allele_label']}"
            f"_f{r['fixed_cost']:.3f}_v{r['variable_cost']:.3f}".replace(".", "p"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--row", type=int)
    p.add_argument("--summarize", action="store_true")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    rows = grid()
    if a.summarize:
        got = [json.loads(f.read_text()) for f in sorted(OUT.glob("*.json")) if f.name != "SUMMARY.json"]
        (OUT / "SUMMARY.json").write_text(json.dumps(got, indent=2))
        print(f"{len(got)}/{len(rows)} rows -> {OUT/'SUMMARY.json'}")
        return
    assert a.row is not None and 0 <= a.row < len(rows), f"--row 0..{len(rows)-1}"
    r = rows[a.row]
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{a.row:04d}_{key(r)}.json"
    if path.exists() and not a.force:
        print(f"[skip] {path.name}")
        return
    res = run_row(r)
    path.write_text(json.dumps(res, indent=2))
    print(f"{key(r)}  ratio2={res['ratio2_myopic']:.4f}  abs_regret={res['abs_regret_myopic']:.5f}  "
          f"headroom={res['headroom']:.5f}  root_match={res['root_match']}")


if __name__ == "__main__":
    main()
