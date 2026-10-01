"""3-gene version of myopic_weakness_sweep.py: where does myopic break at 3 genes?

Kanix's code only
-----------------
Config, belief, exact DP and myopic all come from Kanix's code, through
scripts/search_multigene_myopic_vs_stop_3gene.py. That file is his
search_multigene_myopic_vs_stop.py with GeneC added to GENES and COEF_PRESETS
and nothing else -- `diff` against his original is exactly those 7 lines.
Exact DP is his solve_exact_dp_primal (backward induction), the same solver the
2-gene sweep used, so the two sweeps sit on one footing. This file only loops.

GeneC
-----
Kanix never defined a third gene. GeneC continues his own GeneA -> GeneB step
(a and b x0.75, delta +0.10):
    Base        a -0.045   b -0.0225   delta 0.80
    Aggressive  a -0.0675  b -0.03375  delta 0.90
Allele frequency: even regimes use the shared value; the mixed regimes
(LowHigh, MixedA) copy GeneB. GeneA/GeneB are identical to the 2-gene sweep.

Variants
--------
  pattern  7 alleles x 5 fixed_cost x 3 variable_cost x 2 presets = 210 rows,
           ThreeGeneration only (1,054,528 states at 3 genes; Extended is 9.2M).
  cloneA   GeneC = exact copy of GeneA (coefficients AND allele frequency),
           fixed_cost >= 0.020 only -> 84 rows. Robustness to the GeneC choice.
  cloneB   same, GeneC = exact copy of GeneB.
  sanity   GeneC allele frequency ~0 on 4 ThreeGeneration rows of the 2-gene
           sweep. A gene nobody carries changes nothing, so V*, V_stop, myopic
           and both root actions must reproduce the 2-gene numbers. Row 4 uses
           exactly 0.0 to see whether his code accepts it.

Metrics are the 2-gene sweep's: abs_regret |V* - L| and root_match are the
ones to read; ratio2 is recorded for comparability only (it inflates wherever
headroom V* - V_stop is near zero).

Resume: one row per task, own JSON, skip guard -- safe to resubmit.

Usage:
    python myopic_weakness_sweep_3gene.py --variant sanity  --row 0
    python myopic_weakness_sweep_3gene.py --variant pattern --row 0
    python myopic_weakness_sweep_3gene.py --variant pattern --summarize
"""
from __future__ import annotations
import argparse
import copy
import json
import resource
import sys
import time
from itertools import product
from pathlib import Path

ROOT = Path("/net/projects/ranalab/rajhansini/replication_16features")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from genetic_dp.exact_dp.utils import GENOTYPE_STATES
from genetic_dp.exact_dp.solver import solve_exact_dp_primal
from genetic_dp.experiments.core import _build_factorized_multigene_belief_snapshot
from genetic_dp.utils.pedigree_generator import generate_deterministic_pedigree
import scripts.search_multigene_myopic_vs_stop_3gene as k3

OUT = ROOT / "incremental_experiments" / "results" / "myopic_weakness_sweep_3gene"
OUT_2GENE = ROOT / "incremental_experiments" / "results" / "myopic_weakness_sweep"

# (label, GeneA, GeneB, GeneC). GeneA/GeneB exactly as in the 2-gene sweep.
ALLELES = [
    ("UltraRare",  0.005, 0.005, 0.005),
    ("VeryRare",   0.01,  0.01,  0.01),
    ("LowLow",     0.02,  0.02,  0.02),
    ("LowHigh",    0.02,  0.15,  0.15),   # mixed -> GeneC copies GeneB
    ("MixedA",     0.02,  0.10,  0.10),   # mixed -> GeneC copies GeneB
    ("MediumEven", 0.08,  0.08,  0.08),
    ("HighHigh",   0.15,  0.15,  0.15),
]
FIXED    = [0.005, 0.010, 0.015, 0.020, 0.030]
VARIABLE = [0.010, 0.020, 0.030]
FAMILIES = ["ThreeGeneration", "Extended"]   # Extended = 9,190,992 states, ~256G/task
PRESETS  = ["Base", "Aggressive"]

# 2-gene sweep rows re-run with GeneC ~absent: (2-gene row index, GeneC freq).
# Extended uses the same four cells (2-gene rows +210), at exactly 0.0 only:
# 1e-9 would cost a full 9.2M-state solve each, 0.0 is pruned to 2-gene size.
SANITY = {
    "ThreeGeneration": [(45, 1e-9), (74, 1e-9), (110, 1e-9), (205, 1e-9), (45, 0.0)],
    "Extended":        [(255, 0.0), (284, 0.0), (320, 0.0), (415, 0.0)],
}


def out_dir(variant, family):
    """ThreeGeneration keeps its original paths; other families get a suffix."""
    return OUT / (variant if family == "ThreeGeneration" else f"{variant}_{family}")


def grid(variant, family="ThreeGeneration"):
    if variant == "sanity":
        rows = []
        for idx, eps in SANITY[family]:
            src = json.loads(next(OUT_2GENE.glob(f"{idx:04d}_*.json")).read_text())
            rows.append(dict(family=src["family"], preset=src["preset"],
                             allele_label=src["allele_label"],
                             allele_freqs={**src["allele_freqs"], "GeneC": eps},
                             fixed_cost=src["fixed_cost"], variable_cost=src["variable_cost"],
                             genec="pattern", source_2gene_row=idx))
        return rows
    fixed = FIXED if variant == "pattern" else [f for f in FIXED if f >= 0.020]
    rows = []
    for pre, (alab, ga, gb, gc), fc, vc in product(PRESETS, ALLELES, fixed, VARIABLE):
        if variant == "cloneA":
            gc = ga
        elif variant == "cloneB":
            gc = gb
        rows.append(dict(family=family, preset=pre, allele_label=alab,
                         allele_freqs={"GeneA": ga, "GeneB": gb, "GeneC": gc},
                         fixed_cost=fc, variable_cost=vc, genec=variant))
    return rows


_K3_PRESETS = copy.deepcopy(k3.COEF_PRESETS)


def presets_for(genec):
    """Kanix's COEF_PRESETS from the 3-gene copy; clone variants overwrite GeneC."""
    presets = copy.deepcopy(_K3_PRESETS)
    if genec in ("cloneA", "cloneB"):
        src = "GeneA" if genec == "cloneA" else "GeneB"
        for p in presets.values():
            for coef in ("a_gene", "b_gene", "delta_gene"):
                p[coef]["GeneC"] = p[coef][src]
    return presets


def run_row(r, belief=None):
    """belief: reuse a snapshot built for the same family + allele_freqs. His belief
    builder reads only config.allele_freqs (core.py::_build_factorized_multigene_
    belief_snapshot), and his own script caches it on exactly that key (_BELIEF_CACHE)."""
    t0 = time.time()
    k3.COEF_PRESETS = presets_for(r["genec"])   # _build_config reads this global
    pedigree = generate_deterministic_pedigree(k3.FAMILY_CASES[r["family"]])
    config = k3._build_config(
        pedigree, allele_freqs=r["allele_freqs"], preset_label=r["preset"],
        a_scale=1.0, b_scale=1.0, delta_shift=0.0,
        fixed_cost=r["fixed_cost"], variable_cost=r["variable_cost"],
    )
    belief_shared = belief is not None
    if belief is None:
        belief = _build_factorized_multigene_belief_snapshot(
            pedigree=pedigree, config=config, genes=k3.GENES,
            child_cpds=k3._build_child_cpds(pedigree), belief_parallelism=1,
            progress_label=None,
        )
    t_belief = time.time()
    gen_states_full = list(product(GENOTYPE_STATES, repeat=len(k3.GENES)))
    V, policy_dp = solve_exact_dp_primal(
        pedigree.to_list(), gen_states_full, {frozenset(): 1.0}, belief,
        config.a, config.b, config.c, config.delta,
        config.fixed_cost, config.variable_cost,
        genes=k3.GENES, a_gene=config.a_gene, b_gene=config.b_gene,
        c_gene=config.c_gene, delta_gene=config.delta_gene,
    )
    t_dp = time.time()
    V_root = float(V[frozenset()])
    V_stop = k3._stop_value_from_state(frozenset(), pedigree=pedigree, config=config,
                                       belief=belief, belief_gene={})
    L, myo_pol = k3._evaluate_myopic_root(pedigree=pedigree, config=config, belief=belief)
    t_myo = time.time()
    denom = V_root - V_stop
    rm, rd = myo_pol.get(frozenset()), policy_dp.get(frozenset())
    used = k3.COEF_PRESETS[r["preset"]]
    out = dict(r)
    out.update({
        "genes": list(k3.GENES),
        "coefs_used": {c: dict(used[c]) for c in ("a_gene", "b_gene", "delta_gene")},
        "V_root": V_root, "V_stop_root": V_stop, "headroom": denom,
        "L_myopic": L,
        "ratio2_myopic": (V_root - L) / denom if abs(denom) > 1e-12 else 0.0,
        "abs_regret_myopic": abs(V_root - L),
        "root_action_myopic": list(rm[:2]) if rm else None,
        "root_action_dp": list(rd[:2]) if rd else None,
        "root_match": bool(rm and rd and tuple(rm[:2]) == tuple(rd[:2])),
        "n_states": len(belief), "belief_shared": belief_shared,
        "sec_belief": t_belief - t0, "sec_dp": t_dp - t_belief, "sec_myopic": t_myo - t_dp,
        "peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
    })
    if r["genec"] == "pattern" and "source_2gene_row" in r:
        src = json.loads(next(OUT_2GENE.glob(f"{r['source_2gene_row']:04d}_*.json")).read_text())
        diffs = {k: abs(out[k] - src[k]) for k in ("V_root", "V_stop_root", "L_myopic")}
        out["vs_2gene"] = {
            "abs_diff": diffs,
            "root_action_myopic_same": out["root_action_myopic"] == src["root_action_myopic"],
            "root_action_dp_same": out["root_action_dp"] == src["root_action_dp"],
            "pass": max(diffs.values()) < 1e-6
                    and out["root_action_myopic"] == src["root_action_myopic"]
                    and out["root_action_dp"] == src["root_action_dp"],
        }
    return out


def key(r):
    if "source_2gene_row" in r:
        return f"sanity_from2gene{r['source_2gene_row']:04d}_geneC{r['allele_freqs']['GeneC']:g}"
    return (f"{r['family']}_{r['preset']}_{r['allele_label']}"
            f"_f{r['fixed_cost']:.3f}_v{r['variable_cost']:.3f}".replace(".", "p"))


def _load(variant, family):
    return [json.loads(f.read_text()) for f in sorted(out_dir(variant, family).glob("*.json"))
            if f.name != "SUMMARY.json"]


def _cell(rs):
    n = len(rs)
    return (f"{sum(r['abs_regret_myopic'] for r in rs)/n:.4f}  "
            f"{sum(not r['root_match'] for r in rs):>3}/{n:<3}  "
            f"{sum(r['L_myopic'] < r['V_stop_root'] for r in rs):>3}/{n:<3}")


def report(family):
    """Step 8: 3-gene tables, 3-gene next to 2-gene (same family's rows only), robustness."""
    AL = [a[0] for a in ALLELES]
    ck = lambda r: (r["preset"], r["allele_label"], round(r["fixed_cost"], 3), round(r["variable_cost"], 3))
    g3 = _load("pattern", family)
    g2 = [r for r in json.loads((OUT_2GENE / "SUMMARY.json").read_text()) if r["family"] == family]
    two = {ck(r): r for r in g2}
    paired = [(r, two[ck(r)]) for r in g3 if ck(r) in two]
    print(f"{family} 3-gene pattern rows: {len(g3)}/210   paired with 2-gene {family} rows: {len(paired)}")
    print("cell = mean abs_regret | root wrong | myopic below stop\n")
    for title, keyf, order in [
        ("fixed_cost", lambda r: r["fixed_cost"], FIXED),
        ("allele", lambda r: r["allele_label"], AL),
        ("preset", lambda r: r["preset"], PRESETS),
        ("variable_cost", lambda r: r["variable_cost"], VARIABLE),
    ]:
        print(f"{title:<14}{'3-gene':<30}{'2-gene (same cells)':<30}")
        for k in order:
            s3 = [a for a, b in paired if keyf(a) == k]
            s2 = [b for a, b in paired if keyf(a) == k]
            if s3:
                print(f"{str(k):<14}{_cell(s3):<30}{_cell(s2):<30}")
        print()
    print("3-gene allele x fixed_cost (mean abs_regret, root wrong /n)")
    print(f"{'':<11}" + "".join(f"{f:>14}" for f in FIXED))
    for al in AL:
        cells = []
        for f in FIXED:
            rs = [r for r in g3 if r["allele_label"] == al and r["fixed_cost"] == f]
            cells.append(f"{sum(r['abs_regret_myopic'] for r in rs)/len(rs):.4f} {sum(not r['root_match'] for r in rs):>2}/{len(rs)}"
                         if rs else "-")
        print(f"{al:<11}" + "".join(f"{c:>14}" for c in cells))
    print("\nworst 10 (3-gene)")
    for r in sorted(g3, key=lambda r: -r["abs_regret_myopic"])[:10]:
        print(f"  {r['preset']:<11}{r['allele_label']:<11}f={r['fixed_cost']:.3f} v={r['variable_cost']:.3f}  "
              f"abs={r['abs_regret_myopic']:.4f} head={r['headroom']:.4f} root_ok={r['root_match']}")
    if not any(out_dir(v, family).exists() for v in ("cloneA", "cloneB")):
        print(f"\nrobustness: no clone runs for {family}")
        return
    print("\nrobustness, fixed_cost >= 0.02 (cell = mean abs_regret | root wrong | below stop)")
    hi = {v: {ck(r): r for r in (_load(v, family) if v != "pattern" else g3) if r["fixed_cost"] >= 0.020}
          for v in ("pattern", "cloneA", "cloneB")}
    common = set.intersection(*(set(d) for d in hi.values()))
    print(f"rows present in all three: {len(common)}/84")
    for al in AL + ["ALL"]:
        ks = [k for k in common if al == "ALL" or k[1] == al]
        if ks:
            print(f"{al:<11}" + "".join(f"{v}: {_cell([hi[v][k] for k in ks]):<28}" for v in hi))


def run_group(g, rows, od):
    """All rows of one (preset, allele) pair on ONE belief snapshot.

    Before writing anything, re-runs one already-finished row of the group on the
    shared belief and requires bit-identical V*, V_stop, myopic and root actions;
    otherwise exits without writing. Rows are skip-guarded one at a time, so rows
    finished meanwhile by another job are not redone.
    """
    groups = [(pre, al[0]) for pre in PRESETS for al in ALLELES]
    pre, al = groups[g]
    idx = [i for i, r in enumerate(rows) if r["preset"] == pre and r["allele_label"] == al]
    path = lambda i: od / f"{i:04d}_{key(rows[i])}.json"
    print(f"group {g}: {pre}/{al}  rows {idx[0]}..{idx[-1]}  done already: {sum(path(i).exists() for i in idx)}/{len(idx)}")
    od.mkdir(parents=True, exist_ok=True)

    r0 = rows[idx[0]]
    k3.COEF_PRESETS = presets_for(r0["genec"])
    pedigree = generate_deterministic_pedigree(k3.FAMILY_CASES[r0["family"]])
    config = k3._build_config(
        pedigree, allele_freqs=r0["allele_freqs"], preset_label=r0["preset"],
        a_scale=1.0, b_scale=1.0, delta_shift=0.0,
        fixed_cost=r0["fixed_cost"], variable_cost=r0["variable_cost"],
    )
    t0 = time.time()
    belief = _build_factorized_multigene_belief_snapshot(
        pedigree=pedigree, config=config, genes=k3.GENES,
        child_cpds=k3._build_child_cpds(pedigree), belief_parallelism=1,
        progress_label=None,
    )
    print(f"belief built once: {len(belief)} states, {time.time()-t0:.0f}s")

    done = [i for i in idx if path(i).exists()]
    if done:
        ref = json.loads(path(done[-1]).read_text())
        chk = run_row(rows[done[-1]], belief=belief)
        same = (all(chk[k] == ref[k] for k in ("V_root", "V_stop_root", "L_myopic"))
                and chk["root_action_dp"] == ref["root_action_dp"]
                and chk["root_action_myopic"] == ref["root_action_myopic"])
        print(f"reuse check vs row {done[-1]}: {'IDENTICAL' if same else 'MISMATCH'}")
        if not same:
            print({k: (chk[k], ref[k]) for k in ("V_root", "V_stop_root", "L_myopic", "root_action_dp", "root_action_myopic")})
            sys.exit(2)
    else:
        print("reuse check: no finished row in this group yet (other groups cover it)")

    for i in idx:
        if path(i).exists():
            continue
        res = run_row(rows[i], belief=belief)
        path(i).write_text(json.dumps(res, indent=2))
        print(f"{key(rows[i])}  abs_regret={res['abs_regret_myopic']:.5f}  root_match={res['root_match']}  "
              f"sec={res['sec_dp']:.0f}/{res['sec_myopic']:.0f}  rss={res['peak_rss_gb']:.1f}GB")
    print(f"group {g} complete")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["pattern", "cloneA", "cloneB", "sanity"])
    p.add_argument("--family", default="ThreeGeneration", choices=FAMILIES)
    p.add_argument("--row", type=int)
    p.add_argument("--group", type=int, help="0..13: one (preset, allele) pair, one shared belief")
    p.add_argument("--summarize", action="store_true")
    p.add_argument("--report", action="store_true")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    if a.report:
        report(a.family)
        return
    assert a.variant, "--variant is required unless --report"
    rows = grid(a.variant, a.family)
    od = out_dir(a.variant, a.family)
    if a.summarize:
        got = _load(a.variant, a.family)
        (od / "SUMMARY.json").write_text(json.dumps(got, indent=2))
        print(f"{len(got)}/{len(rows)} rows -> {od/'SUMMARY.json'}")
        return
    if a.group is not None:
        run_group(a.group, rows, od)
        return
    assert a.row is not None and 0 <= a.row < len(rows), f"--row 0..{len(rows)-1}"
    r = rows[a.row]
    od.mkdir(parents=True, exist_ok=True)
    path = od / f"{a.row:04d}_{key(r)}.json"
    if path.exists() and not a.force:
        print(f"[skip] {path.name}")
        return
    res = run_row(r)
    path.write_text(json.dumps(res, indent=2))
    print(f"{key(r)}  ratio2={res['ratio2_myopic']:.4f}  abs_regret={res['abs_regret_myopic']:.5f}  "
          f"headroom={res['headroom']:.5f}  root_match={res['root_match']}  "
          f"n_states={res['n_states']}  sec={res['sec_belief']:.0f}/{res['sec_dp']:.0f}/{res['sec_myopic']:.0f}  "
          f"rss={res['peak_rss_gb']:.1f}GB")
    if "vs_2gene" in res:
        print(f"vs 2-gene: {res['vs_2gene']}")


if __name__ == "__main__":
    main()
