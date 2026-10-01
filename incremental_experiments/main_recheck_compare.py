"""Compare the myopic weakness tables with Kanix's own main() output.

submit_main_recheck.sh runs his main() (2-gene: unmodified script, all 420
configs; 3-gene: his script + GeneC, the 60 MixedA configs, where his default
GeneC 0.1 equals ours). This only reads JSON and compares; nothing is computed.

Per config: exact_root_value vs V_root, stop_value vs V_stop_root,
myopic_root_value vs L_myopic, and the first two elements of his
exact_root_action / myopic_root_action vs ours.
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path("/net/projects/ranalab/rajhansini/replication_16features/incremental_experiments/results")
CHK = ROOT / "main_recheck"
SWEEP = {2: [ROOT / "myopic_weakness_sweep"],
         3: [ROOT / "myopic_weakness_sweep_3gene" / "pattern", ROOT / "myopic_weakness_sweep_3gene" / "pattern_Extended"]}


def k(g, fam, pre, afa, afb, fc, vc):
    return (g, fam, pre, round(afa, 6), round(afb, 6), round(fc, 6), round(vc, 6))


ours = {}
for g, dirs in SWEEP.items():
    for d in dirs:
        for f in d.glob("*.json"):
            if f.name == "SUMMARY.json":
                continue
            j = json.loads(f.read_text())
            af = j["allele_freqs"]
            ours[k(g, j["family"], j["preset"], af["GeneA"], af["GeneB"], j["fixed_cost"], j["variable_cost"])] = j

for g in (2, 3):
    n = errors = act_bad = 0
    worst = {"V*": 0.0, "stop": 0.0, "myopic": 0.0}
    missing = []
    for f in sorted(CHK.glob(f"g{g}_*.json")):
        for row in json.loads(f.read_text()):
            if "error" in row:
                errors += 1
                print(f"  ERROR in his run: {f.name} {row['error']}")
                continue
            key = k(g, row["family"], row["preset"], row["af_a"], row["af_b"], row["fixed_cost"], row["variable_cost"])
            o = ours.get(key)
            if o is None:
                missing.append(key)
                continue
            n += 1
            worst["V*"] = max(worst["V*"], abs(row["exact_root_value"] - o["V_root"]))
            worst["stop"] = max(worst["stop"], abs(row["stop_value"] - o["V_stop_root"]))
            worst["myopic"] = max(worst["myopic"], abs(row["myopic_root_value"] - o["L_myopic"]))
            if row["exact_root_action"][:2] != o["root_action_dp"] or row["myopic_root_action"][:2] != o["root_action_myopic"]:
                act_bad += 1
                print(f"  ACTION DIFF {key}: his dp {row['exact_root_action'][:2]} ours {o['root_action_dp']} | "
                      f"his myopic {row['myopic_root_action'][:2]} ours {o['root_action_myopic']}")
    print(f"{g}-gene: {n} configs compared, his-run errors {errors}, unmatched {len(missing)}")
    print(f"  max |diff|  V* {worst['V*']:.3g}   stop {worst['stop']:.3g}   myopic {worst['myopic']:.3g}")
    print(f"  root actions (DP and myopic) differing: {act_bad}/{n}")
