"""Myopic baseline on Kanix's OWN 102-family benchmark suite.

Kanix's `full102_expected.json` reports ADP (ABCD-16) ratio2/ratio3 against
exact DP for all 102 families -- it contains no myopic numbers, because myopic
was never his baseline. This script fills that gap: it runs Kanix's canonical
zero-lookahead `myopic_greedy` and his exact DP over the SAME 102 settings, so
the myopic column sits on exactly his footing.

Nothing here is reimplemented. Every piece is his:
  * settings          scripts/load_suite_cases.py::load_cases (original8 /
                      local40 / phase6_54, read from documentation/ + artifacts/)
  * Setting -> config scripts/search_multigene_myopic_vs_stop.py::_build_config
                      (this is what applies a_scale / b_scale / delta_shift; do
                      NOT hand-roll it -- delta_shift is clamped to [0, 0.99])
  * belief            genetic_dp/experiments/core.py
                      ::_build_factorized_multigene_belief_snapshot
  * exact DP          genetic_dp/exact_dp/solver.py::solve_exact_dp_primal
  * myopic            scripts/..._myopic_vs_stop.py::_evaluate_myopic_root,
                      which calls genetic_dp.policy.baselines.myopic_greedy

ratio2 is computed with the identical formula used everywhere else in this
repo: (V_root - L) / (V_root - V_stop_root), so these numbers are directly
comparable to the myopic_TRUE files and to Kanix's own ADP ratio2.

The suite is 2-gene (GENES = ("GeneA", "GeneB")). It covers only the LowHigh
and MediumEven regimes -- LowLow/HighHigh/MixedA/MixedB are ours, not his.

Resume: one row per task, each writing its own JSON with a skip guard, so the
whole array is safe to resubmit and only redoes what is missing. --force redoes
a row from scratch.

Usage:
    python myopic_on_kanix_suite.py --row 0
    python myopic_on_kanix_suite.py --all          # every row, one process
    python myopic_on_kanix_suite.py --summarize    # build the summary table
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

from scripts.load_suite_cases import load_cases, SUITE_ORDER
from scripts.search_multigene_myopic_vs_stop import (
    GENES, FAMILY_CASES, _build_config, _build_child_cpds,
    _evaluate_myopic_root, _stop_value_from_state,
)

OUT = ROOT / "incremental_experiments" / "results" / "myopic_kanix102"


def all_cases():
    """The 102 SuiteCases in a fixed, reproducible order."""
    cases = []
    for suite in SUITE_ORDER:
        cases.extend(load_cases(suite))
    return cases


def kanix_adp():
    """{row_id: his published ADP ratio2/ratio3}, for side-by-side reporting."""
    samples = json.loads((ROOT / "full102_expected.json").read_text())["samples"]
    return {s["row_id"]: {"adp_ratio2": s["ratio2"], "adp_ratio3": s["ratio3"],
                          "suite": s["suite"]} for s in samples}


def run_case(case):
    s = case.setting
    pedigree = generate_deterministic_pedigree(FAMILY_CASES[s.family])
    config = _build_config(
        pedigree,
        allele_freqs=s.allele_freqs,
        preset_label=s.preset,
        a_scale=s.a_scale,
        b_scale=s.b_scale,
        delta_shift=s.delta_shift,
        fixed_cost=s.fixed_cost,
        variable_cost=s.variable_cost,
    )

    belief = _build_factorized_multigene_belief_snapshot(
        pedigree=pedigree, config=config, genes=GENES,
        child_cpds=_build_child_cpds(pedigree), belief_parallelism=1,
        progress_label=None,
    )

    mu0 = {frozenset(): 1.0}
    gen_states_full = list(product(GENOTYPE_STATES, repeat=len(GENES)))
    V, policy_dp = solve_exact_dp_primal(
        pedigree.to_list(), gen_states_full, mu0, belief,
        config.a, config.b, config.c, config.delta,
        config.fixed_cost, config.variable_cost,
        genes=GENES, a_gene=config.a_gene, b_gene=config.b_gene,
        c_gene=config.c_gene, delta_gene=config.delta_gene,
    )
    V_root = float(V[frozenset()])
    V_stop_root = _stop_value_from_state(
        frozenset(), pedigree=pedigree, config=config, belief=belief, belief_gene={},
    )
    L_myopic, myopic_policy = _evaluate_myopic_root(
        pedigree=pedigree, config=config, belief=belief,
    )

    denom = V_root - V_stop_root
    ratio2 = (V_root - L_myopic) / denom if abs(denom) > 1e-12 else 0.0
    root_myo = myopic_policy.get(frozenset())
    root_dp = policy_dp.get(frozenset())
    return {
        "row_id": case.row_id, "suite": case.suite, "row_index": case.row_index,
        "family": s.family, "preset": s.preset, "allele_freqs": dict(s.allele_freqs),
        "a_scale": s.a_scale, "b_scale": s.b_scale, "delta_shift": s.delta_shift,
        "fixed_cost": s.fixed_cost, "variable_cost": s.variable_cost,
        "V_root": V_root, "V_stop_root": V_stop_root,
        "L_myopic": L_myopic,
        "ratio2_myopic": ratio2,
        "abs_regret_myopic": abs(V_root - L_myopic),
        "root_action_myopic": list(root_myo[:2]) if root_myo else None,
        "root_action_dp": list(root_dp[:2]) if root_dp else None,
        "root_match": bool(root_myo and root_dp and tuple(root_myo[:2]) == tuple(root_dp[:2])),
        "n_states": len(belief),
    }


def do_row(case, force=False):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{case.row_index:03d}_{case.row_id}.json"
    if p.exists() and not force:
        print(f"[skip] {case.row_id} already done -> {p.name}")
        return json.loads(p.read_text())
    res = run_case(case)
    p.write_text(json.dumps(res, indent=2))
    print(f"[{case.suite}] {case.row_id}  ratio2_myopic={res['ratio2_myopic']:.4f}  "
          f"abs_regret={res['abs_regret_myopic']:.4f}  root_match={res['root_match']}  "
          f"n_states={res['n_states']:,}")
    return res


def summarize():
    adp = kanix_adp()
    rows = []
    for f in sorted(OUT.glob("*.json")):
        if f.name == "SUMMARY.json":
            continue
        rows.append(json.loads(f.read_text()))
    rows.sort(key=lambda r: r["row_index"])
    for r in rows:
        r.update(adp.get(r["row_id"], {}))
    (OUT / "SUMMARY.json").write_text(json.dumps(rows, indent=2))
    print(f"{len(rows)}/102 rows present -> {OUT/'SUMMARY.json'}")
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--row", type=int, help="row index 0-101")
    p.add_argument("--all", action="store_true")
    p.add_argument("--summarize", action="store_true")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    if a.summarize:
        summarize(); return
    cases = all_cases()
    assert len(cases) == 102, f"expected 102 cases, got {len(cases)}"
    if a.all:
        for c in cases:
            do_row(c, a.force)
        summarize()
    else:
        assert a.row is not None, "pass --row N, --all, or --summarize"
        do_row(cases[a.row], a.force)


if __name__ == "__main__":
    main()
