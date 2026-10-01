"""E6 (2-gene) on the high-cost region where myopic is weak.

Why
---
myopic_weakness_sweep.py showed myopic's gap to exact DP is driven by TESTING
COST, not allele rarity: mean |V* - L| goes 0.0067 -> 0.0334 as fixed_cost goes
0.005 -> 0.030, and myopic picks the wrong first person 64/84 at the top. The
winnable slice of that (cells where DP actually still wants to test -- headroom
>= 0.03) is 83 configs at fixed_cost >= 0.020.

Every E6 checkpoint on disk was trained with fixed_cost=0.01, variable_cost=0.02
and NOTHING ELSE. That matters more than it looks: the model's cost vector is
COST_DIM = 3*len(GENES)+2 = 8 -- per-gene a/b/delta (6) plus fixed and variable
cost (2) -- so cost IS a model input, but those last two features were literally
constant across every training config. The model cannot have learned any
cost dependence, because it never saw cost vary.

Two modes, both scored on the same 83 target cells:

  --mode eval   take the existing seed checkpoint as-is and score it. Answers
                the prerequisite question: does what E6 learned at f=0.01
                transfer to f=0.02-0.03? If yes, no retraining is needed.

  --mode train  retrain E6 from scratch with cost VARYING in TRAIN (three
                fixed_cost levels), so the two constant features stop being
                constant, then score. Family split is untouched -- TRAIN is
                still Trio+Nuclear, so this isolates the cost change and
                nothing else.

Reuses E6's model and training code UNCHANGED -- GNNQBidirSumPool,
train_one_epoch, precompute_qhat, ds_to_tensors, q_rollout, all imported from
e6_train_two_gene_gnn_q.py. Only the config tuples grow a cost pair.
build_two_gene_dataset already accepts fixed_cost/variable_cost, so no shared
code is touched.

Comparability: the myopic sweep used Kanix's COEF_PRESETS and this path uses
shared/data_gen PRESETS -- verified identical for Base and Aggressive, so the
ratio2 / abs_regret numbers here sit directly beside the myopic ones.

Usage:
    python e6_cost_experiment.py --mode eval  --seed 0
    python e6_cost_experiment.py --mode train --seed 0 --epochs 500
"""
from __future__ import annotations
import argparse, importlib.util, json, random, sys, time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ground-up-experiments"))


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m


e6 = _load_module("e6_cost_gnn", HERE / "e6_train_two_gene_gnn_q.py")
tg = e6.tg
q_rollout = e6.q_rollout          # e6 imports it from exputils.eval

OUT = HERE / "results" / "e6_cost"
SWEEP = HERE / "results" / "myopic_weakness_sweep" / "SUMMARY.json"

# Cost levels seen during training. The point is that fixed_cost VARIES; without
# that, the two cost features stay constant and the model cannot condition on them.
TRAIN_COSTS = [(0.010, 0.020), (0.020, 0.020), (0.030, 0.020)]


def target_cells():
    """The 83 winnable high-cost cells, read straight from the myopic sweep so
    both analyses are scored on exactly the same configs."""
    rows = json.loads(SWEEP.read_text())
    sel = [r for r in rows
           if r["fixed_cost"] >= 0.020 and r["headroom"] >= 0.03
           and r["allele_label"] in ("LowLow", "MixedA", "LowHigh", "MediumEven", "HighHigh")]
    return sorted(sel, key=lambda r: (r["family"], r["preset"], r["allele_label"],
                                      r["fixed_cost"], r["variable_cost"]))


def build_ds(fam, reg, pre, fc, vc):
    return tg.build_two_gene_dataset(
        family_label=fam, allele_freqs=tg.ALLELE_FREQ_REGIMES[reg],
        preset_label=pre, genes=e6.GENES, fixed_cost=fc, variable_cost=vc,
    )


def score(model, cells, dev, log):
    """ratio2 + absolute regret per cell, beside myopic's own numbers."""
    struct_cache, edge_cache, out = {}, {}, {}
    for c in cells:
        fam = c["family"]
        if fam not in struct_cache:
            ds0 = build_ds(fam, c["allele_label"], c["preset"], c["fixed_cost"], c["variable_cost"])
            struct_cache[fam] = tg.compute_structural_features(ds0["pedigree"], ds0["individuals"])
            edge_cache[fam] = tg.build_edge_index(ds0["pedigree"], ds0["individuals"])
        ds = build_ds(fam, c["allele_label"], c["preset"], c["fixed_cost"], c["variable_cost"])
        base = tg.ds_to_tensors(ds, struct_cache[fam], dev)
        ds["_nf"], ds["_gf"] = base["nf"], base["gf"]
        eit = torch.tensor(edge_cache[fam], device=dev)
        key = (f"{fam}_{c['allele_label']}_{c['preset']}"
               f"_f{c['fixed_cost']:.3f}_v{c['variable_cost']:.3f}")
        q_hat = e6.precompute_qhat(model, ds, key, eit, dev)
        ratio2, L = q_rollout(q_hat, ds, log=log, trace=False)
        out[key] = {
            "family": fam, "allele": c["allele_label"], "preset": c["preset"],
            "fixed_cost": c["fixed_cost"], "variable_cost": c["variable_cost"],
            "ratio2_gnn": ratio2, "L_gnn": L,
            "abs_regret_gnn": abs(ds["V_root"] - L),
            "V_root": ds["V_root"], "headroom": c["headroom"],
            "ratio2_myopic": c["ratio2_myopic"],
            "abs_regret_myopic": c["abs_regret_myopic"],
        }
        log(f"  [{key}] gnn ratio2={ratio2:.4f} absreg={out[key]['abs_regret_gnn']:.5f}"
            f"  | myopic absreg={c['abs_regret_myopic']:.5f}")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["eval", "train"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--groups_per_batch", type=int, default=512)
    p.add_argument("--lambda_ce", type=float, default=1.0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = torch.device(a.device)
    rd = OUT / a.mode / f"seed{a.seed}"; rd.mkdir(parents=True, exist_ok=True)
    res_path = rd / "results.json"
    if res_path.exists() and not a.force:
        print(f"[skip] {res_path} exists"); return
    log_f = open(rd / "run.log", "a")

    def log(m=""):
        print(m, flush=True); log_f.write(m + "\n"); log_f.flush()

    cells = target_cells()
    log(f"[e6_cost] {datetime.now().isoformat()} mode={a.mode} seed={a.seed} "
        f"cells={len(cells)} device={a.device}")

    model = e6.GNNQBidirSumPool().to(dev)

    if a.mode == "eval":
        ck = (HERE / "results" / "e6_gnn_sumpool_2gene" / "seed_runs"
              / f"seed{a.seed}" / "gnn_q_e6_2gene.pt")
        assert ck.exists(), f"missing checkpoint {ck}"
        model.load_state_dict(torch.load(ck, map_location=dev))
        log(f"loaded {ck}  (trained at fixed_cost=0.01, variable_cost=0.02 only)")
    else:
        train_cfgs = [(fam, reg, pre, fc, vc)
                      for fam in tg.TRAIN_FAMILIES
                      for reg in tg.ALLELE_FREQ_REGIMES
                      for pre in tg.PRESETS_LIST
                      for (fc, vc) in TRAIN_COSTS]
        log(f"[1] {len(train_cfgs)} train configs "
            f"(TRAIN families {tg.TRAIN_FAMILIES}, costs {TRAIN_COSTS})")
        struct_cache, edge_cache, per_config = {}, {}, []
        for fam, reg, pre, fc, vc in train_cfgs:
            ds = build_ds(fam, reg, pre, fc, vc)
            if fam not in struct_cache:
                struct_cache[fam] = tg.compute_structural_features(ds["pedigree"], ds["individuals"])
                edge_cache[fam] = tg.build_edge_index(ds["pedigree"], ds["individuals"])
            base = tg.ds_to_tensors(ds, struct_cache[fam], dev)
            key = f"{fam}_{reg}_{pre}_f{fc:.3f}_v{vc:.3f}"
            s_idx, a_idx, y = e6.build_qsa_index(ds, device=dev, cache_key=key)
            k_max = len(ds["individuals"])
            gr, gm, gt = e6.build_state_groups(s_idx, y, k_max)
            per_config.append((fam, {
                "nf": base["nf"], "gf": base["gf"],
                "state_idx": s_idx, "action_idx": a_idx, "y": y,
                "group_rows": gr, "group_mask": gm, "group_target": gt,
            }))
            log(f"    [{key}] {len(ds['states']):,} states -> {len(y):,} rows -> {gr.shape[0]:,} groups")
        edge_index_t = {f: torch.tensor(edge_cache[f], device=dev) for f in edge_cache}
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        # Per-epoch checkpoint + resume, same as e6_train_two_gene_gnn_q.py.
        # 72 train configs is 3x E6's baseline, so this can outrun the 4h wall;
        # without resume a killed task would produce nothing at all.
        ckpt_path = rd / "checkpoint.pt"
        start_epoch = 1
        if ckpt_path.exists() and not a.force:
            ck = torch.load(ckpt_path, map_location=dev)
            model.load_state_dict(ck["model_state"]); opt.load_state_dict(ck["optimizer_state"])
            start_epoch = ck["epoch"] + 1
            log(f"[RESUME] checkpoint at epoch {ck['epoch']}, resuming from {start_epoch}")
        log(f"[2] training epochs {start_epoch}-{a.epochs} over {len(per_config)} configs")
        t0 = time.time()
        for ep in range(start_epoch, a.epochs + 1):
            tot = 0.0
            for fam, d in per_config:
                loss, _, _ = e6.train_one_epoch(model, edge_index_t[fam], d, opt,
                                                a.groups_per_batch, a.lambda_ce, dev)
                tot += loss
            if ep % 50 == 0 or ep == 1 or ep == a.epochs:
                log(f"    epoch {ep:4d}  loss={tot/len(per_config):.5f}  ({time.time()-t0:.0f}s)")
            torch.save({"epoch": ep, "model_state": model.state_dict(),
                        "optimizer_state": opt.state_dict()}, ckpt_path)
        torch.save(model.state_dict(), rd / "gnn_q_e6_cost.pt")
        log(f"    saved -> {rd/'gnn_q_e6_cost.pt'}")

    model.eval()
    log(f"\n[3] scoring on {len(cells)} high-cost cells")
    res = score(model, cells, dev, log)
    g = float(np.mean([r["abs_regret_gnn"] for r in res.values()]))
    m = float(np.mean([r["abs_regret_myopic"] for r in res.values()]))
    wins = sum(1 for r in res.values() if r["abs_regret_gnn"] < r["abs_regret_myopic"])
    log(f"\nMEAN abs_regret  gnn={g:.5f}  myopic={m:.5f}   gnn beats myopic in {wins}/{len(res)}")
    res_path.write_text(json.dumps(
        {"mode": a.mode, "seed": a.seed, "n_cells": len(res),
         "mean_abs_regret_gnn": g, "mean_abs_regret_myopic": m,
         "gnn_beats_myopic": wins, "per_cell": res}, indent=2))
    log(f"saved -> {res_path}")
    log_f.close()


if __name__ == "__main__":
    main()
