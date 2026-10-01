"""E5 GNN-Q and MLP-Q trained on the myopic-weakness grid, scored against it.

Split (unchanged from every E-series run -- the models never see a test family):
  TRAIN  Trio + Nuclear, on the sweep's own settings grid:
         2 presets x 7 allele regimes x 5 fixed_cost x 3 variable_cost
         = 210 configs per family, 420 per gene count.
  TEST   ThreeGeneration + Extended = the 420 configs per gene count of
         myopic_weakness_sweep{,_3gene}, scored against that sweep's exact-DP V*
         and Kanix's myopic L, config for config.
The grid (ALLELES / FIXED / VARIABLE, GeneC frequencies) is imported from the
two sweep drivers, so train and test settings are the sweep's by construction.

Kanix's code for everything that is not the model:
  config  his _build_config (2-gene: his unmodified script; 3-gene: his script
          + GeneC, pattern coefficients)
  belief  _build_factorized_multigene_belief_snapshot
  DP      solve_exact_dp_primal (backward induction)
  stop    his _stop_value_from_state      myopic  his _evaluate_myopic_root
Q*(s,a) targets come from build_qsa_index (the existing pipeline: his
r_reward_test on his belief and his V*). GUARD: max(stop(s), max_a Q*(s,a))
must equal his V*(s) at every state, or the config is refused.

Models and training loop reused unchanged: GNNQBidir + train_one_epoch from
e5_train_{two_gene_,}gnn_q.py at n_rounds=3 (the existing E5); MLPQ from
q_learning/{two_gene_,}mlp_q.py + train_one_epoch from
fixing_gnn_q/train_{two_gene_,}mlp_q_ce.py (the MLP paired with E5). CE+MSE,
state-grouped batching, Adam lr 1e-3, 512 groups/batch, 500 epochs.

Evaluation reuses exputils.eval.q_rollout unchanged. GUARDS per test config:
stop value and Kanix's myopic L recomputed here must equal the sweep's, and
q_rollout driven by his myopic policy must reproduce his myopic L -- so the
evaluator that scores the models is proven to agree with his on that config.
Q-hat is computed lazily, only at states the rollout visits (Extended 3-gene
has 9,190,992 states).

Stages:
  --stage data      --genes G --family {Trio,Nuclear}   build + save train tensors
  --stage train     --genes G --model {gnn,mlp} --seed S
  --stage eval      --genes G --unit U   U = (test family, allele, preset), 0-27
  --stage summarize --genes G
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
import time
from array import array
from itertools import product
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (ROOT, ROOT / "scripts", ROOT / "ground-up-experiments", ROOT / "experiments_after_understanding",
          ROOT / "experiments_after_understanding" / "q_learning", ROOT / "fixing_gnn_q"):
    sys.path.insert(0, str(p))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m


# Kanix
from genetic_dp.exact_dp.utils import GENOTYPE_STATES  # noqa: E402
from genetic_dp.exact_dp.solver import solve_exact_dp_primal  # noqa: E402
from genetic_dp.experiments.core import _build_factorized_multigene_belief_snapshot  # noqa: E402
from genetic_dp.models.belief import InferenceResult  # noqa: E402
from genetic_dp.policy.baselines import myopic_greedy  # noqa: E402
from genetic_dp.utils.pedigree_generator import generate_deterministic_pedigree  # noqa: E402
import scripts.search_multigene_myopic_vs_stop as k2  # noqa: E402
import scripts.search_multigene_myopic_vs_stop_3gene as k3  # noqa: E402

# ours: train-family pedigrees, features, Q targets, evaluator, loss grouping
from shared.data_gen import FAMILY_CASES as OUR_FAMILY_CASES, state_to_vector_k_genes  # noqa: E402
from qsa_data import build_qsa_index  # noqa: E402
from exputils.eval import q_rollout, _get_entry, _stop_val  # noqa: E402
from losses import build_state_groups  # noqa: E402

sw2 = _load("e5weak_sweep2", HERE / "myopic_weakness_sweep.py")
sw3 = _load("e5weak_sweep3", HERE / "myopic_weakness_sweep_3gene.py")

TRAIN_FAMILIES = ["Trio", "Nuclear"]
TEST_FAMILIES = ["ThreeGeneration", "Extended"]
PRESETS = ["Base", "Aggressive"]
FIXED, VARIABLE = sw2.FIXED, sw2.VARIABLE
assert (sw3.FIXED, sw3.VARIABLE, sw3.PRESETS, sw2.PRESETS) == (FIXED, VARIABLE, PRESETS, PRESETS)
N_ROUNDS = 3                      # the existing E5
SEEDS = [0, 1, 2]
PATTERN_GENEC = {"Base": (-0.045, -0.0225, 0.80), "Aggressive": (-0.0675, -0.03375, 0.90)}
OUT = HERE / "results" / "e5_weak"
SWEEP_DIRS = {2: [HERE / "results" / "myopic_weakness_sweep"],
              3: [HERE / "results" / "myopic_weakness_sweep_3gene" / "pattern",
                  HERE / "results" / "myopic_weakness_sweep_3gene" / "pattern_Extended"]}


def kanix(G):
    return k2 if G == 2 else k3


def alleles(G):
    """[(label, {gene: freq})] in the sweep's order, straight from its driver."""
    if G == 2:
        return [(lab, {"GeneA": ga, "GeneB": gb}) for lab, ga, gb in sw2.ALLELES]
    return [(lab, {"GeneA": ga, "GeneB": gb, "GeneC": gc}) for lab, ga, gb, gc in sw3.ALLELES]


def settings(G):
    return [dict(preset=pre, allele=lab, freqs=fr, fixed=fc, var=vc)
            for pre, (lab, fr), fc, vc in product(PRESETS, alleles(G), FIXED, VARIABLE)]


def key(fam, s):
    return f"{fam}_{s['preset']}_{s['allele']}_f{s['fixed']:.3f}_v{s['var']:.3f}".replace(".", "p")


def check_kanix_presets(G):
    if G == 3:
        for pre, (a, b, d) in PATTERN_GENEC.items():
            c = k3.COEF_PRESETS[pre]
            assert (c["a_gene"]["GeneC"], c["b_gene"]["GeneC"], c["delta_gene"]["GeneC"]) == (a, b, d), \
                f"k3.COEF_PRESETS[{pre}] GeneC is not the pattern values"


def pedigree(fam):
    cases = k2.FAMILY_CASES.get(fam) or OUR_FAMILY_CASES[fam]
    return generate_deterministic_pedigree(cases)


def build_config(G, ped, s):
    return kanix(G)._build_config(ped, allele_freqs=s["freqs"], preset_label=s["preset"],
                                  a_scale=1.0, b_scale=1.0, delta_shift=0.0,
                                  fixed_cost=s["fixed"], variable_cost=s["var"])


def build_belief(G, ped, config):
    K = kanix(G)
    b = _build_factorized_multigene_belief_snapshot(
        pedigree=ped, config=config, genes=K.GENES, child_cpds=K._build_child_cpds(ped),
        belief_parallelism=1, progress_label=None)
    assert isinstance(b[frozenset()], InferenceResult), "unexpected belief entry type"
    return b


# ── model side (ours) ─────────────────────────────────────────────────────────

class ModelKit:
    """The existing E5 GNN and the MLP paired with it, per gene count, unchanged."""

    def __init__(self, G):
        if G == 2:
            g = _load("e5weak_gnn2", HERE / "e5_train_two_gene_gnn_q.py")
            m = _load("e5weak_mlp2", ROOT / "fixing_gnn_q" / "train_two_gene_mlp_q_ce.py")
            feat = g.tg
            self.MLP = m.two_mlp.MLPQ
        else:
            g = _load("e5weak_gnn3", HERE / "e5_train_gnn_q.py")
            m = _load("e5weak_mlp3", ROOT / "fixing_gnn_q" / "train_mlp_q_ce.py")
            feat = g.gnn_mod
            self.MLP = m.mlp_q.MLPQ
        self.GNN, self.gnn_epoch, self.mlp_epoch = g.GNNQBidir, g.train_one_epoch, m.train_one_epoch
        self.struct, self.edges, self.cost_vec = (feat.compute_structural_features,
                                                  feat.build_edge_index, feat.config_to_cost_vec)
        self.node_feat, self.genes = g.NODE_FEAT, g.GENES
        assert tuple(self.genes) == tuple(kanix(G).GENES)

    def new(self, model):
        return self.GNN(n_rounds=N_ROUNDS) if model == "gnn" else self.MLP()


def node_feats(states, belief, individuals, genes, struct):
    """(N, n_people, 3k+4): per-gene beliefs | is_tested | n_parents, n_children, depth.
    Same layout as two_gene/run.py::ds_to_tensors and gnn/run.py::load_dataset."""
    n, N = len(individuals), len(states)
    idx = {p: i for i, p in enumerate(individuals)}
    X = np.stack([state_to_vector_k_genes(s, belief, individuals, genes) for s in states])
    X = X.reshape(N, n, 3 * len(genes)).astype(np.float32)
    tested = np.zeros((N, n, 1), dtype=np.float32)
    for i, s in enumerate(states):
        for person, _ in s:
            tested[i, idx[person], 0] = 1.0
    st = np.tile(struct[np.newaxis], (N, 1, 1))
    return np.concatenate([X, tested, st], axis=-1).astype(np.float32)


# ── stage: data ───────────────────────────────────────────────────────────────

def stage_data(G, fam, limit=None, out=OUT):
    check_kanix_presets(G)
    K, kit = kanix(G), ModelKit(G)
    od = out / f"g{G}" / "data"; od.mkdir(parents=True, exist_ok=True)
    ped = pedigree(fam); ind = ped.to_list()
    struct = kit.struct(ped, ind)
    todo = settings(G)[:limit] if limit else settings(G)
    by_allele = {}
    for s in todo:
        by_allele.setdefault(s["allele"], []).append(s)
    worst = 0.0
    for lab, group in by_allele.items():
        pending = [s for s in group if not (od / f"{key(fam, s)}.pt").exists()]
        if not pending:
            continue
        t0 = time.time()
        belief = build_belief(G, ped, build_config(G, ped, pending[0]))   # depends on family + freqs only
        nf_all, states_ref = None, None
        for s in pending:
            config = build_config(G, ped, s)
            V, _ = solve_exact_dp_primal(
                ind, list(product(GENOTYPE_STATES, repeat=G)), {frozenset(): 1.0}, belief,
                config.a, config.b, config.c, config.delta, config.fixed_cost, config.variable_cost,
                genes=K.GENES, a_gene=config.a_gene, b_gene=config.b_gene,
                c_gene=config.c_gene, delta_gene=config.delta_gene)
            states = [st for st in V if st in belief]
            ds = {"states": states, "individuals": ind, "belief": belief, "config": config,
                  "genes": K.GENES, "V_star": V}
            s_idx, a_idx, y = build_qsa_index(ds, device="cpu", cache_key=None)

            # GUARD: Q* must be consistent with his V*: V*(s) = max(stop(s), max_a Q*(s,a))
            best = np.full(len(states), -np.inf)
            np.maximum.at(best, s_idx.numpy(), y.numpy().astype(np.float64))
            bg, diff = {}, 0.0
            for i in np.unique(s_idx.numpy()):
                stp = K._stop_value_from_state(states[i], pedigree=ped, config=config, belief=belief, belief_gene=bg)
                diff = max(diff, abs(max(stp, best[i]) - V[states[i]]))
            assert diff < 1e-6, f"{key(fam, s)}: Q* inconsistent with Kanix V* (max diff {diff:.3g})"
            worst = max(worst, float(diff))

            if states_ref != states:                  # features depend on the belief only
                nf_all, states_ref = node_feats(states, belief, ind, K.GENES, struct), states
            assert nf_all.shape[-1] == kit.node_feat
            cv = kit.cost_vec(config)
            torch.save({"family": fam, "key": key(fam, s), "setting": s,
                        "nf": torch.tensor(nf_all), "gf": torch.tensor(np.tile(cv[None], (len(states), 1))),
                        "state_idx": s_idx, "action_idx": a_idx, "y": y,
                        "V_root": float(V[frozenset()]), "n_states": len(states), "guard_max_diff": float(diff)},
                       od / f"{key(fam, s)}.pt")
            print(f"  {key(fam, s)}  {len(states):,} states  {len(y):,} rows  V*={V[frozenset()]:.6f}  "
                  f"guard {diff:.2e}", flush=True)
        print(f"[{fam} {lab}] {len(pending)} configs in {time.time()-t0:.0f}s", flush=True)
    print(f"done {fam} G={G}: Q*-vs-V* guard max diff {worst:.2e}")


# ── stage: train ──────────────────────────────────────────────────────────────

def stage_train(G, model_name, seed, epochs=500, gpb=512, lambda_ce=1.0, limit=None, out=OUT, device="cpu"):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    dev = torch.device(device)
    kit = ModelKit(G)
    files = sorted((out / f"g{G}" / "data").glob("*.pt"))
    expect = len(TRAIN_FAMILIES) * len(settings(G))
    if limit is None:
        assert len(files) == expect, f"{len(files)} data files, expected {expect}: run --stage data first"
    rd = out / f"g{G}" / model_name / f"seed{seed}"; rd.mkdir(parents=True, exist_ok=True)
    final = rd / "model.pt"
    if final.exists():
        print(f"[skip] {final} exists"); return
    log_f = open(rd / "run.log", "a")

    def log(m):
        print(m, flush=True); log_f.write(m + "\n"); log_f.flush()

    edges, per_config = {}, []
    for f in files:
        d = torch.load(f, map_location=dev, weights_only=False)   # our own data files
        fam = d["family"]
        if fam not in edges:
            ped = pedigree(fam)
            edges[fam] = torch.tensor(kit.edges(ped, ped.to_list()), device=dev)
        k_max = d["nf"].shape[1]
        gr, gm, gt = build_state_groups(d["state_idx"], d["y"], k_max)
        per_config.append((fam, {"nf": d["nf"], "gf": d["gf"], "state_idx": d["state_idx"],
                                 "action_idx": d["action_idx"], "y": d["y"],
                                 "group_rows": gr, "group_mask": gm, "group_target": gt}))
    log(f"[e5_weak] G={G} model={model_name} seed={seed} n_rounds={N_ROUNDS if model_name == 'gnn' else '-'} "
        f"configs={len(per_config)} groups={sum(d['group_rows'].shape[0] for _, d in per_config):,}")

    model = kit.new(model_name).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ck, prev = rd / "checkpoint.pt", rd / "checkpoint_prev.pt"; start = 1
    for cand in (ck, prev):          # a kill mid-write can only hit the newer file
        if not cand.exists():
            continue
        try:
            c = torch.load(cand, map_location=dev)
            model.load_state_dict(c["model_state"]); opt.load_state_dict(c["optimizer_state"])
            start = c["epoch"] + 1
            log(f"[RESUME] {cand.name} epoch {c['epoch']} -> {start}")
            break
        except Exception as exc:
            log(f"[CHECKPOINT CORRUPT] {cand.name}: {exc!r}")
    else:
        if ck.exists() or prev.exists():
            log("[CHECKPOINT CORRUPT] no loadable checkpoint -- training restarts from epoch 1")
    t0 = time.time()
    for ep in range(start, epochs + 1):
        tot = 0.0
        for fam, d in per_config:
            if model_name == "gnn":
                loss, _, _ = kit.gnn_epoch(model, edges[fam], d, opt, gpb, lambda_ce, dev)
            else:
                loss, _, _ = kit.mlp_epoch(model, d, opt, gpb, lambda_ce, dev)
            tot += loss
        if ep % 20 == 0 or ep == 1 or ep == epochs:
            log(f"  epoch {ep:4d}  loss={tot/len(per_config):.5f}  ({time.time()-t0:.0f}s)")
        tmp = rd / "checkpoint.tmp"
        torch.save({"epoch": ep, "model_state": model.state_dict(), "optimizer_state": opt.state_dict()}, tmp)
        if ck.exists():
            os.replace(ck, prev)
        os.replace(tmp, ck)          # atomic: checkpoint.pt is always a complete file
    torch.save(model.state_dict(), rd / "model.tmp")
    os.replace(rd / "model.tmp", final)
    log(f"saved {final}")


# ── stage: eval ───────────────────────────────────────────────────────────────

class PolicyQ(dict):
    """Kanix's myopic policy as a Q table q_rollout can follow: the chosen person
    scores +1e9, everyone else -1e9, and a stop state is all -1e9 (<= stop value)."""

    def __init__(self, policy, individuals):
        super().__init__(); self.policy, self.ind = policy, individuals

    def __missing__(self, state):
        a = self.policy[state]
        chosen = None if a[0] == "stop" else a[1]
        self[state] = d = {p: (1e9 if p == chosen else -1e9) for p in self.ind}
        return d


class LazyQ(dict):
    """Model Q-hat computed only at states the rollout actually visits."""

    def __init__(self, model, is_gnn, belief, individuals, genes, struct, cost_vec, ei):
        super().__init__()
        self.m, self.gnn, self.b, self.ind, self.genes, self.struct, self.ei = \
            model, is_gnn, belief, individuals, genes, struct, ei
        self.gf = torch.tensor(cost_vec[None], dtype=torch.float32)

    def __missing__(self, state):
        tested = {p for p, _ in state}
        cand = [i for i, p in enumerate(self.ind) if p not in tested]
        nf = torch.tensor(node_feats([state], self.b, self.ind, self.genes, self.struct)).expand(len(cand), -1, -1)
        gf, a = self.gf.expand(len(cand), -1), torch.tensor(cand)
        with torch.no_grad():
            q = self.m(nf, self.ei, gf, a) if self.gnn else self.m(nf, gf, a)
        self[state] = d = {self.ind[i]: float(q[j]) for j, i in enumerate(cand)}
        return d


class ArrayQ(dict):
    """Model Q-hat for every (state, untested person), computed in large batches
    once per config; q_rollout reads it through this dict view exactly as it
    reads LazyQ. Nothing is cached per state (q_rollout memoizes values itself)."""

    def __init__(self, q, index, starts, counts, row_person, individuals):
        super().__init__()
        self.q, self.index, self.starts, self.counts, self.rp, self.ind = q, index, starts, counts, row_person, individuals
        self.visited = 0

    def __missing__(self, state):
        i = self.index[state]; a = int(self.starts[i]); self.visited += 1
        return {self.ind[int(self.rp[a + j])]: float(self.q[a + j]) for j in range(int(self.counts[i]))}


def all_state_rows(belief, individuals, genes, struct, chunk=200_000):
    """Node features for every belief state, plus one row per (state, untested person)."""
    states = list(belief.keys())
    index = {s: i for i, s in enumerate(states)}
    nf = np.empty((len(states), len(individuals), 3 * len(genes) + 4), dtype=np.float32)
    for a in range(0, len(states), chunk):
        nf[a:a + chunk] = node_feats(states[a:a + chunk], belief, individuals, genes, struct)
    pidx = {p: i for i, p in enumerate(individuals)}
    counts = np.empty(len(states), dtype=np.int64); rp = array("q")
    for i, s in enumerate(states):
        tested = {p for p, _ in s}
        un = [pidx[p] for p in individuals if p not in tested]
        counts[i] = len(un); rp.extend(un)
    starts = np.zeros(len(states), dtype=np.int64); starts[1:] = np.cumsum(counts)[:-1]
    row_state = torch.from_numpy(np.repeat(np.arange(len(states), dtype=np.int64), counts))
    return states, index, torch.from_numpy(nf), starts, counts, np.array(rp, dtype=np.int64), row_state


def batched_q(net, is_gnn, nf_all, row_state, row_person_t, cost_vec, ei, batch=32768):
    gf1 = torch.tensor(cost_vec[None], dtype=torch.float32)
    q = torch.empty(len(row_state))
    with torch.no_grad():
        for a in range(0, len(row_state), batch):
            s_b, p_b = row_state[a:a + batch], row_person_t[a:a + batch]
            nf, gf = nf_all[s_b], gf1.expand(len(s_b), -1)
            q[a:a + batch] = net(nf, ei, gf, p_b) if is_gnn else net(nf, gf, p_b)
    return q.numpy()


def sweep_rows(G):
    rows = {}
    for d in SWEEP_DIRS[G]:
        for f in d.glob("*.json"):
            if f.name != "SUMMARY.json":
                r = json.loads(f.read_text())
                rows[(r["family"], r["preset"], r["allele_label"], round(r["fixed_cost"], 6),
                      round(r["variable_cost"], 6))] = r
    assert len(rows) == len(TEST_FAMILIES) * len(settings(G)), f"sweep rows: {len(rows)}"
    return rows


def units(G):
    return [(fam, lab, pre) for fam in TEST_FAMILIES for lab, _ in alleles(G) for pre in PRESETS]


def stage_eval(G, unit, limit=None, out=OUT, seeds=SEEDS, model_types=("gnn",), qmode="array"):
    check_kanix_presets(G)
    torch.set_num_threads(max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))))
    K, kit = kanix(G), ModelKit(G)
    fam, lab, pre = units(G)[unit]
    todo = [s for s in settings(G) if s["allele"] == lab and s["preset"] == pre][:limit]
    od = out / f"g{G}" / "eval"; od.mkdir(parents=True, exist_ok=True)
    todo = [s for s in todo if not (od / f"{key(fam, s)}.json").exists()]
    if not todo:
        print("[skip] unit done"); return
    rows = sweep_rows(G)
    models = {}
    for m in model_types:
        for sd in seeds:
            w = out / f"g{G}" / m / f"seed{sd}" / "model.pt"
            assert w.exists(), f"missing {w}: run --stage train first"
            net = kit.new(m); net.load_state_dict(torch.load(w, map_location="cpu")); net.eval()
            models[f"{m}_seed{sd}"] = (net, m == "gnn")
    ped = pedigree(fam); ind = ped.to_list()
    struct = kit.struct(ped, ind)
    ei = torch.tensor(kit.edges(ped, ind))
    t0 = time.time()
    belief = build_belief(G, ped, build_config(G, ped, todo[0]))
    print(f"[{fam} {lab} {pre}] belief {len(belief):,} states in {time.time()-t0:.0f}s", flush=True)
    root = frozenset()
    if qmode == "array":            # features depend on the belief only: build once for the unit
        t0 = time.time()
        states_all, index, nf_all, starts, counts, row_person, row_state = all_state_rows(belief, ind, K.GENES, struct)
        row_person_t = torch.from_numpy(row_person)
        rng = np.random.default_rng(0)
        sample = [root] + [states_all[i] for i in rng.choice(len(states_all), 200, replace=False) if counts[i] > 0]
        print(f"  features: {len(states_all):,} states, {len(row_state):,} (state, person) rows "
              f"in {time.time()-t0:.0f}s", flush=True)
    for s in todo:
        t1 = time.time()
        r = rows[(fam, pre, lab, round(s["fixed"], 6), round(s["var"], 6))]
        assert r["allele_freqs"] == s["freqs"], f"allele freqs differ from sweep row"
        if G == 3:
            assert r["coefs_used"] == K.COEF_PRESETS[pre], "sweep row used other GeneC coefficients"
        config = build_config(G, ped, s)

        # GUARDS: same config/belief as the sweep, and q_rollout agrees with Kanix's evaluator
        v_stop = K._stop_value_from_state(root, pedigree=ped, config=config, belief=belief, belief_gene={})
        L_myo, myo_pol = K._evaluate_myopic_root(pedigree=ped, config=config, belief=belief)
        assert abs(v_stop - r["V_stop_root"]) < 1e-12, f"stop {v_stop} != sweep {r['V_stop_root']}"
        assert abs(L_myo - r["L_myopic"]) < 1e-12, f"myopic {L_myo} != sweep {r['L_myopic']}"
        ds = {"belief": belief, "individuals": ind, "config": config, "genes": K.GENES,
              "V_root": r["V_root"], "V_stop_root": r["V_stop_root"]}
        _, L_chk = q_rollout(PolicyQ(myo_pol, ind), ds, trace=False)
        assert abs(L_chk - L_myo) < 1e-9, f"q_rollout on myopic's policy {L_chk} != Kanix {L_myo}"

        res = {"family": fam, "preset": pre, "allele_label": lab, "allele_freqs": s["freqs"],
               "fixed_cost": s["fixed"], "variable_cost": s["var"],
               "V_root": r["V_root"], "V_stop_root": r["V_stop_root"],
               "L_myopic": r["L_myopic"], "abs_regret_myopic": r["abs_regret_myopic"],
               "root_action_dp": r["root_action_dp"], "root_action_myopic": r["root_action_myopic"],
               "evaluator_check_diff": abs(L_chk - L_myo), "models": {}}
        cv = kit.cost_vec(config)
        for name, (net, is_gnn) in models.items():
            qdiff = None
            if qmode == "array":
                q = ArrayQ(batched_q(net, is_gnn, nf_all, row_state, row_person_t, cv, ei),
                           index, starts, counts, row_person, ind)
                # GUARD: the batched Q-hat must equal the one-state-at-a-time Q-hat
                lz = LazyQ(net, is_gnn, belief, ind, K.GENES, struct, cv, ei)
                qdiff = max(abs(q[st_][p] - v) for st_ in sample for p, v in lz[st_].items())
                assert qdiff < 1e-4, f"{name}: batched Q-hat differs from per-state Q-hat by {qdiff:.3g}"
            else:
                q = LazyQ(net, is_gnn, belief, ind, K.GENES, struct, cv, ei)
            ratio2, L = q_rollout(q, ds, trace=False)
            qr = q[root]
            best = max(qr, key=qr.get)
            act = ["test", best] if qr[best] > v_stop else ["stop", None]
            res["models"][name] = {"L": L, "abs_regret": r["V_root"] - L, "ratio2": ratio2,
                                   "root_action": act, "root_match_dp": act == r["root_action_dp"],
                                   "states_visited": q.visited if qmode == "array" else len(q),
                                   "q_batch_check_diff": qdiff}
        tmp = od / f"{key(fam, s)}.json.tmp"
        tmp.write_text(json.dumps(res, indent=2))
        os.replace(tmp, od / f"{key(fam, s)}.json")     # atomic: a present JSON is complete
        per = "  ".join(f"{mt} {np.mean([v['abs_regret'] for n, v in res['models'].items() if n.startswith(mt)]):.5f}"
                        for mt in model_types)
        print(f"  {key(fam, s)}  regret {per}  myopic {r['abs_regret_myopic']:.5f}  "
              f"(evaluator check {res['evaluator_check_diff']:.1e}, {time.time()-t1:.0f}s)", flush=True)


# ── stage: traj ───────────────────────────────────────────────────────────────
# Whole-trajectory agreement, same definition as log_e5_fourway.py: walk MYOPIC's
# path (its action at each state, the most likely test result), and at every
# visited state compare DP's, myopic's and each GNN seed's pick. Needs Kanix's
# DP policy per config, so the DP is re-solved here (guarded against the sweep).

def _pick(kind_person):
    k, p = kind_person
    return "stop" if k == "stop" else f"test {p}"


def stage_traj(G, unit, limit=None, out=OUT, seeds=SEEDS, max_steps=20):
    check_kanix_presets(G)
    K, kit = kanix(G), ModelKit(G)
    fam, lab, pre = units(G)[unit]
    todo = [s for s in settings(G) if s["allele"] == lab and s["preset"] == pre][:limit]
    od = out / f"g{G}" / "traj"; od.mkdir(parents=True, exist_ok=True)
    todo = [s for s in todo if not (od / f"{key(fam, s)}.json").exists()]
    if not todo:
        print("[skip] unit done"); return
    rows = sweep_rows(G)
    models = {}
    for sd in seeds:
        w = out / f"g{G}" / "gnn" / f"seed{sd}" / "model.pt"
        net = kit.new("gnn"); net.load_state_dict(torch.load(w, map_location="cpu")); net.eval()
        models[f"gnn_seed{sd}"] = net
    ped = pedigree(fam); ind = ped.to_list()
    struct = kit.struct(ped, ind)
    ei = torch.tensor(kit.edges(ped, ind))
    t0 = time.time()
    belief = build_belief(G, ped, build_config(G, ped, todo[0]))
    print(f"[{fam} {lab} {pre}] belief {len(belief):,} states in {time.time()-t0:.0f}s", flush=True)
    root = frozenset()
    for s in todo:
        t1 = time.time()
        r = rows[(fam, pre, lab, round(s["fixed"], 6), round(s["var"], 6))]
        config = build_config(G, ped, s)
        V, pol = solve_exact_dp_primal(
            ind, list(product(GENOTYPE_STATES, repeat=G)), {root: 1.0}, belief,
            config.a, config.b, config.c, config.delta, config.fixed_cost, config.variable_cost,
            genes=K.GENES, a_gene=config.a_gene, b_gene=config.b_gene,
            c_gene=config.c_gene, delta_gene=config.delta_gene)
        # GUARD: this is the sweep's DP
        assert abs(float(V[root]) - r["V_root"]) < 1e-12, f"V* {V[root]} != sweep {r['V_root']}"
        assert list(pol[root][:2]) == r["root_action_dp"], f"DP root {pol[root][:2]} != sweep {r['root_action_dp']}"
        del V
        cv = kit.cost_vec(config)
        lazy = {n: LazyQ(net, True, belief, ind, K.GENES, struct, cv, ei) for n, net in models.items()}
        mkw = dict(belief=belief, individuals=ind, gen_states=GENOTYPE_STATES, infer=None,
                   a=config.a, b=config.b, c=config.c, delta=config.delta,
                   fixed_cost=config.fixed_cost, variable_cost=config.variable_cost, genes=K.GENES,
                   a_gene=config.a_gene or None, b_gene=config.b_gene or None,
                   c_gene=config.c_gene or None, delta_gene=config.delta_gene or None, tuple_mode=True)
        state, steps = root, []
        for _ in range(max_steps):
            tested = {p for p, _ in state}
            untested = [p for p in ind if p not in tested]
            ya, yw, _ = myopic_greedy(state, **mkw)
            ypk = ("stop", None) if ya == "stop" else ("test", yw)
            dpk = ("stop", None) if pol[state][0] == "stop" else ("test", pol[state][1])
            per_gene, tuple_pmfs = _get_entry(belief, state, K.GENES)
            v_stop = _stop_val(per_gene, ind, tested, config)
            gpk = {}
            for n, q in lazy.items():
                if not untested:
                    gpk[n] = ("stop", None); continue
                qv = q[state]; best = max(untested, key=lambda p: qv[p])
                gpk[n] = ("stop", None) if qv[best] <= v_stop else ("test", best)
            steps.append({"dp": _pick(dpk), "myopic": _pick(ypk), **{n: _pick(v) for n, v in gpk.items()}})
            if ypk[0] == "stop":
                break
            pmf = tuple_pmfs.get(ypk[1], {})
            outcome = max(pmf.items(), key=lambda kv: kv[1])[0]
            nxt = frozenset(state | {(ypk[1], outcome)})
            if nxt not in belief:
                break
            state = nxt
        res = {"family": fam, "preset": pre, "allele_label": lab, "fixed_cost": s["fixed"], "variable_cost": s["var"],
               "n_steps": len(steps), "steps": steps,
               "match_myopic": sum(st_["dp"] == st_["myopic"] for st_ in steps),
               "match_gnn": {n: sum(st_["dp"] == st_[n] for st_ in steps) for n in models}}
        tmp = od / f"{key(fam, s)}.json.tmp"
        tmp.write_text(json.dumps(res, indent=2))
        os.replace(tmp, od / f"{key(fam, s)}.json")
        print(f"  {key(fam, s)}  steps {len(steps)}  DP-match myopic {res['match_myopic']}  gnn "
              + " ".join(str(v) for v in res["match_gnn"].values()) + f"  ({time.time()-t1:.0f}s)", flush=True)


# ── stage: summarize ──────────────────────────────────────────────────────────

def stage_summarize(G, out=OUT):
    files = sorted((out / f"g{G}" / "eval").glob("*.json"))
    R = [json.loads(f.read_text()) for f in files]
    print(f"{G}-gene: {len(R)}/{len(TEST_FAMILIES) * len(settings(G))} test configs evaluated; "
          f"evaluator check max diff {max((r['evaluator_check_diff'] for r in R), default=0):.1e}")
    if not R:
        return

    def mean_reg(rs, m):
        return np.mean([np.mean([v["abs_regret"] for n, v in r["models"].items() if n.startswith(m)]) for r in rs])

    def root_ok(rs, m):
        return sum(np.mean([v["root_match_dp"] for n, v in r["models"].items() if n.startswith(m)]) for r in rs)

    mts = [mt for mt in ("gnn", "mlp") if any(n.startswith(mt) for n in R[0]["models"])]
    print(f"{'':16}{'n':>4}{'myopic':>9}" + "".join(f"{mt:>9}" for mt in mts)
          + "   first action = DP: myopic / " + " / ".join(mts))
    groups = [("all", R)] + [(f"fixed {fc:.3f}", [r for r in R if abs(r["fixed_cost"] - fc) < 1e-9]) for fc in FIXED] \
        + [(fam, [r for r in R if r["family"] == fam]) for fam in TEST_FAMILIES]
    for lab, rs in groups:
        if rs:
            myo_ok = sum(r["root_action_myopic"] == r["root_action_dp"] for r in rs)
            print(f"{lab:16}{len(rs):>4}{np.mean([r['abs_regret_myopic'] for r in rs]):>9.4f}"
                  + "".join(f"{mean_reg(rs, mt):>9.4f}" for mt in mts)
                  + f"   {myo_ok}/{len(rs)} / " + " / ".join(f"{root_ok(rs, mt):.1f}" for mt in mts))
    (out / f"g{G}" / "SUMMARY.json").write_text(json.dumps(R, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True, choices=["data", "train", "eval", "traj", "summarize"])
    p.add_argument("--genes", type=int, required=True, choices=[2, 3])
    p.add_argument("--family", choices=TRAIN_FAMILIES)
    p.add_argument("--model", choices=["gnn", "mlp"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--unit", type=int)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--limit", type=int, default=None, help="smoke tests only")
    p.add_argument("--out", type=Path, default=OUT, help="smoke tests only")
    p.add_argument("--device", default="cpu", help="train stage: cpu or cuda")
    p.add_argument("--qmode", default="array", choices=["array", "lazy"], help="eval: batched or per-state Q-hat")
    a = p.parse_args()
    if a.stage == "data":
        stage_data(a.genes, a.family, a.limit, a.out)
    elif a.stage == "train":
        stage_train(a.genes, a.model, a.seed, a.epochs, limit=a.limit, out=a.out, device=a.device)
    elif a.stage == "eval":
        stage_eval(a.genes, a.unit, a.limit, a.out, qmode=a.qmode)
    elif a.stage == "traj":
        stage_traj(a.genes, a.unit, a.limit, a.out)
    else:
        stage_summarize(a.genes, a.out)


if __name__ == "__main__":
    main()
