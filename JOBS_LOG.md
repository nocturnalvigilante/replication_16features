# SLURM job log

Last updated: 2026-08-24 (updated every time a new job is submitted or a
tracked job's status changes materially). Only jobs from this investigation
are listed -- `squeue -u $(whoami)` also shows unrelated jobs (nlp_rq*,
rarm7_*, tex_*, etc.) from other projects, not touched or tracked here.

To check status yourself: `squeue -u $(whoami) | grep <jobname>` or
`sacct -j <jobid> --format=JobID,State,Elapsed -n`

## Currently running / pending

### 2026-09-10: E5 on Extended (OOD), 2-gene + 3-gene

The 2-gene vs 3-gene comparison is now E5 only (two-way MP, CE+MSE, per-state
batching, mean pooling, 3 rounds, existing checkpoints -- no retrain). E5 had
never been run on Extended; E4/E6/E7 had. Added `--variant e5` to
`log_extended_fourway.py` (GNN = `e5_gnn_rounds3{,_2gene}`, MLP = E2 ce, as
E4/E7). Smoke on 2-gene seed0 LowHigh_Base: ran clean in ~9 min, V* and myopic
ratio2 bit-identical to the committed E4 Extended run (0.142944), so DP/myopic
are untouched; smoke output deleted.

| Job ID | Name | Purpose | Status |
|---|---|---|---|
| 2275616 | ext_e5 | `submit_extended_fourway_e5.sh`: tasks 0-2 = 3-gene seeds 0-2, 3-5 = 2-gene seeds 0-2; outputs `results/extended/{2,3}gene/seed*/extended_fourway_e5.json` | 3-gene: **COMPLETED** 09-11 02:53, 3/3 seeds, 12/12 configs each; requeued at the wall and resumed from partials as designed. Tasks 3-5 (2-gene) CANCELLED while still pending -- stuck on the 256G request. |
| 2276134 | ext_e5_2g | Same script, `--array=3-5 --mem=8G --time=00:45:00` -- the 2-gene seeds only. 2-gene Extended peaked at 2.5 GB RSS and 3.5-4.7 min per seed in 2206261. | COMPLETED 18:08, 3/3 seeds, 12/12 configs each. |

2-gene E5 Extended (3-seed avg, 12 configs): GNN ratio2 0.102 (0.073 without
LowLow_Base), root 3.7/12, whole-traj 0.74. Myopic 0.221 (0.124 w/o LowLow_Base),
MLP 0.216 -- MLP and myopic identical to the E4 Extended run, as they must be.
vs E4 2-gene Extended: ratio2 0.113 -> 0.102, w/o LowLow_Base 0.072 -> 0.073,
root 6.0 -> 3.7/12, whole-traj 0.72 -> 0.74.

3-gene E5 Extended (3-seed avg, 12 configs): GNN ratio2 0.091 (0.080 w/o
LowLow_Base), root 0.7/12 (seeds 1/0/1), whole-traj 46.7/81 = 0.58. Myopic
ratio2 0.115 (0.068 w/o LowLow_Base), root 8/12, whole-traj 32/81 = 0.40 --
myopic ratio2 identical to the E4 Extended run.

Share of stake lost = sum_c(ratio2_c * stake_c) / sum_c(stake_c), stake =
V* - V_stop (Extended: from the per-config log line; ThreeGeneration:
`myopic_TRUE_summary.json`). Not an average of ratio2, so the tiny LowLow_Base
denominator can't dominate:

| E5 | ThreeGeneration 2g | 3g | Extended 2g | 3g |
|---|---|---|---|---|
| mean stake | 0.109 | 0.169 | 0.149 | 0.226 |
| GNN share lost | 6.2% | 6.1% | 5.9% | 6.0% |
| myopic share lost | 8.0% | 4.4% | 9.4% | 5.3% |

GNN is flat from 2 to 3 genes on both families; myopic roughly halves its
share lost. Both E5 checkpoint sets are current (3-gene retrained 08-22 on the
corrected allele frequencies; 2-gene 07-25, never affected by that bug).

### 2026-09-10: E5 GNN trained on the myopic-weakness grid

`incremental_experiments/e5_weak.py`, `submit_e5_weak.sh`. E5 GNN-Q (n_rounds=3,
CE+MSE, state-grouped batching, 500 epochs, 3 seeds), GNN only (no MLP, user
decision). TRAIN Trio+Nuclear on the sweep's full grid (7 alleles x 5 fixed x 3
variable x 2 presets = 420 configs per gene count); TEST the sweep's 420
ThreeGeneration+Extended configs per gene count, scored against its V* and
Kanix's myopic L. Config/belief/DP/stop/myopic all Kanix's; Q* targets guarded
against his V* at every state; evaluator guarded per config (q_rollout on his
myopic policy must equal his myopic L). 3-gene GeneC = pattern (matches the sweep),
NOT the old data_gen placeholders. Smoke test (scratch, 2-gene + 3-gene data/train,
2-gene eval): guards 7e-9, evaluator diff 0.0. OMP_NUM_THREADS=1 (4x faster than 4 threads).

| Job ID | Stage | Purpose | Status |
|---|---|---|---|
| 2277136 | data 2 (0-1) | Trio / Nuclear train tensors, 2-gene | **COMPLETE**, 420/420, Q*-vs-V* guard 1.5e-8. |
| 2277137 | data 3 (0-1) | Trio / Nuclear train tensors, 3-gene | **COMPLETE**, 420/420, Q*-vs-V* guard 1.5e-8. |
| 2277138 | train 2 (0-2) | GNN 2-gene seeds 0-2, after 2277136 | **COMPLETE, 3/3** (CPU, 9.4 s/epoch, ~78 min). |
| 2277139 | train 3 (0-2) | GNN 3-gene seeds 0-2 on CPU, 106 s/epoch (~15h) | **CANCELLED at epoch 64/62/62** to move to GPU; checkpoints verified loadable. |
| 2277140 | eval 2 (0-27) | 2-gene test, after 2277138 | **COMPLETE**, 420/420, evaluator check 0.0. |
| 2277141 / 2277142 | eval 3 | dependents of 2277139 | CANCELLED (resubmitted as 2277579 / 2277580). |
| 2277578 | train 3 GPU (0-2) | Same 3-gene training resumed from the CPU checkpoints, `--device cuda`, 1 GPU each (L40S, g002) | **COMPLETE** 09-11 ~05:25, 3/3 at 500 epochs; USR1 self-requeue at 03:06 resumed from atomic checkpoints (epochs 339/348/344). |
| 2277579 / 2277580 | eval 3 | dependency-chained 3-gene evals | CANCELLED -- the monitor submits 3-gene eval itself once all 3 seeds exist (output-driven, no dependencies). |
| 2277599 (+ successors) | e5w_monitor | Watchdog, `submit_e5_weak_monitor.sh` -> `incremental_experiments/e5_weak_monitor.py` every 5 min. Done-ness from disk (`results/e5_weak/manifest.json`); resubmits timed-out/failed/OOM'd units (OOM -> 2x memory; train GPU -> CPU after 2 GPU failures; max 4 attempts); never resubmits a unit whose log has an AssertionError (guard failure -> alert); deletes unparseable eval JSONs; releases held jobs; summarizes each gene count when its 420 files exist; writes `DONE`. Keeps one successor queued (afterany). Status: `results/e5_weak/STATUS.txt`, events: `monitor.log`. | **DONE** 09-11 10:38: submitted the 28 3-gene eval units (2278838-2278865, all COMPLETED first try, longest 2h09m); 0 resubmits, 0 alerts; wrote FINAL_SUMMARY_g2/g3.txt and DONE; chain stopped. |

3-gene eval tests with stand-in checkpoints (epoch ~77), outputs in `results/e5_weak_evaltest{,2}/` (tests only, not results):
2277651/2277652 per-state Q-hat -- ThreeGeneration 160 s/config 19.6 GB; Extended 21 min/config, 198 GB of 210 GB (too slow for a 4h wall, too close to OOM).
2277823/2277824 batched Q-hat (`--qmode array`, now the default; guard: batched vs per-state Q-hat on root + 200 states < 1e-4) --
ThreeGeneration 30 s/config 20 GB; Extended 5-9 min/config, 201 GB (monitor now asks 300 GB, 8 CPUs). Batched vs per-state:
Q-hat within 1.7e-6; L identical on 2-gene (5/5, diff 0.0); on 3-gene 3 of 4 L within 1e-4, one Extended seed flips a near-tie
root choice (L diff 1.5e-3) -- float noise between two equally valid evaluations of the same model, far below the seed spread.
2-gene eval (2277140) was per-state; 3-gene eval uses batched.

RESULT (`results/e5_weak/FINAL_SUMMARY_g{2,3}.txt`; artifact "Where E5 Beats Myopic" https://claude.ai/code/artifact/d8676e37-8023-4f67-8516-7200211fd527), mean regret myopic / GNN (3-seed mean):
2-gene all 420: 0.0184 / 0.0217; testing pays (f>=0.02, V*-stop>=0.03, n=83): 0.0306 / 0.0254 (GNN better 62/83; ThreeGen 0.0245/0.0213, Extended 0.0360/0.0289); DP stops at root (n=80): 0.0143 / 0.0379.
3-gene all 420: 0.0175 / 0.0208; testing pays (n=100): 0.0262 / 0.0260 (41/100; ThreeGen 0.0211/0.0228, Extended 0.0310/0.0291); DP stops at root (n=68): 0.0183 / 0.0324.
GNN almost never stops at the root; seeds vary widely (2-gene 0.0226/0.0164/0.0262, 3-gene 0.0277/0.0161/0.0185).

Whole-trajectory stage (09-11, `--stage traj`, same definition as log_e5_fourway.py: walk myopic's most-likely path, compare DP / myopic / GNN picks per state; DP re-solved per config, guarded: V* and root action == sweep). Submitted by the watchdog (2280145) as 56 units e5w_traj{2,3}_* (2280147-2280202). 2-gene 420/420 in minutes; 3-gene 405/420 by 13:46. Unit traj3_25 (2280200_25) crawled on overloaded p008 (load 221/256; no belief after 2h) -> duplicate 2280500_25 on g002 (--exclude=p008); identical atomic outputs, cancel the loser. -> 2280500_25 COMPLETED in 60 min on g002; 2280200_25 cancelled; ALL DONE 09-11 14:51, 0 alerts, 840/840 trajectories.
Whole trajectory = DP (myopic / E5 3-seed mean): 2-gene all 38% / 45%, testing pays 31% / 50%; 3-gene all 41% / 42%, testing pays 37% / 39% (ThreeGen 42/43%, Extended 33/36%); DP tests nobody 2-gene 48% / 3%, 3-gene 42% / 18%. Artifact v3 updated.

Resume/safety in `e5_weak.py` (09-10 23:14): checkpoint written to a temp file then renamed, previous epoch kept as `checkpoint_prev.pt`, resume falls back to it; `model.pt` and every eval JSON written atomically. GPU training: 51 s/epoch vs 106 s on CPU.

### 2026-09-10: 3-gene myopic weakness sweep (Kanix's code + GeneC)

`incremental_experiments/myopic_weakness_sweep_3gene.py`, driven by
`submit_myopic_weakness_3gene.sh <variant>`. DP/myopic/belief/config are all
Kanix's, via `scripts/search_multigene_myopic_vs_stop_3gene.py` = his script +
GeneC in `GENES`/`COEF_PRESETS`, nothing else. GeneC continues his GeneA->GeneB
step (a,b x0.75, delta +0.10). ThreeGeneration only. No training.

| Job ID | Variant | Purpose | Status |
|---|---|---|---|
| 2273910 | sanity (0-4) | GeneC freq ~0 on 2-gene rows 45/74/110/205 (+45 at exactly 0.0) must reproduce 2-gene V*, V_stop, myopic, root actions; also the time/memory pilot | **COMPLETE, 5/5 PASS.** freq 1e-9: max diff 1.4e-9, all root actions identical. freq 0.0: bit-identical (diff 0.0) -- his belief code prunes the zero-prob GeneC states, 20,816 states = the 2-gene count. Pilot: ~2.5 min, 17.7 GB per config at 1,054,528 states. |
| 2273937 | pattern (0-209) | Main grid, 32G / 1h per task | **COMPLETE, 210/210, 0 errors.** |
| 2273938 | cloneA (0-83) | GeneC = copy of GeneA, fixed_cost >= 0.02 | **COMPLETE, 84/84, 0 errors.** |
| 2273939 | cloneB (0-83) | GeneC = copy of GeneB, fixed_cost >= 0.02 | **COMPLETE, 84/84, 0 errors.** |

| 2274876 | sanity Extended (0-3) | GeneC freq exactly 0.0 on 2-gene Extended rows 255/284/320/415 must be bit-identical to 2-gene | **COMPLETE, 4/4 PASS, all diffs 0.0**, root actions identical. |
| 2274877 | pattern Extended (0-209) | Step 9: full grid on Extended (9,190,992 states), 256G / 4h per task. Outputs `pattern_Extended/` | **53 COMPLETED; 157 CANCELLED while still PENDING** (never started) -- each task rebuilt the same ~15-min belief; replaced by 2275475. |
| 2275475 | pattern Extended group (0-13) | Same rows, one task per (preset, allele) group = one belief per 15 rows; 210G / 1:30, threedle-own/threedle-contrib/general/peanut-cpu | **COMPLETE, 14/14, 210/210 rows, 0 errors.** Tasks 10/11 requeued at the wall and resumed via skip guard. Shared-belief reuse checks vs rows 14/29/44/46/162/177: bit-IDENTICAL. |

| 2275003 | lp_xcheck | Kanix's UNMODIFIED script with PuLP on PYTHONPATH (isolated dir `/net/projects/ranalab/rajhansini/pylibs_pulp`), so exact DP runs his Gurobi LP instead of the silent primal fallback; 2-gene rows 45/74/205/284/320 | **COMPLETE.** LP V* is HIGHER than backward induction on 5/5 (+4e-4 to +1.1e-2); stop + myopic identical (0.0); DP root action differs on row 74. Cause: multigene LP forces Phi = sum of per-gene Phi (`per_gene_phi_active = bool(gene_list)`), a restricted min-LP -> upper bound, not exact. Backward induction is the true optimum; all sweeps used it. `incremental_experiments/lp_crosscheck_compare.py`. |

ThreeGeneration result (`--report`): same weak region as 2-gene. Cost is the main axis -- 3-gene
mean abs_regret 0.0053 -> 0.0265 as fixed_cost 0.005 -> 0.030, root wrong 12/42 ->
31/42. Worst: rare alleles at high cost (LowLow f=0.03 0.0439, 6/6 root wrong).
Robust to GeneC: at fixed_cost >= 0.02, abs_regret 0.0219 / 0.0254 / 0.0220 for
pattern / cloneA / cloneB, root wrong 56 / 60 / 52 of 84.

| 2276374 / 2276375 / 2276376 | main_recheck (0-27 / 28-29 / 30-39) | Recheck through Kanix's own `main()` CLI, not our drivers (`submit_main_recheck.sh`). 2-gene: his UNMODIFIED script, all 420 configs. 3-gene: his main() has no GeneC flag, get_config defaults GeneC to 0.1 = our MixedA GeneC, so the 60 MixedA configs run through it untouched. pulp not importable -> primal fallback; DP root action from his `extract_exact_policy`. | **COMPLETE, 40/40.** `incremental_experiments/main_recheck_compare.py`: 2-gene 420/420 and 3-gene 60/60 with max diff 0 on V*, stop, myopic; DP and myopic root actions identical 480/480. |

Extended result (`--report --family Extended`): same weak region. 3-gene mean
abs_regret 0.0081 -> 0.0358 as fixed_cost 0.005 -> 0.030 (2-gene same cells
0.0082 -> 0.0384), root wrong 14/42 -> 34/42. Worst: Aggressive LowLow f=0.03
(0.0615-0.0643, root wrong). 9,190,992 states, ~19 min/config un-shared, 178.6 GB peak.

Full re-derivation of every number in the E0-E9 PI briefing artifact, so that
no figure on that page predates the 3-gene allele-frequency fix (2026-08-15/16)
or the E0-E9 retrain (2026-08-22).

| Job ID | Name | Purpose | Status |
|---|---|---|---|
| 2206182 | fourway_rerun | Four-way action logs (DP vs myopic vs GNN-Q vs MLP-Q; root action + whole trajectory) for all 8 rungs x {3-gene, 2-gene} x 3 seeds, 48 tasks | **COMPLETE 08-24, 48/48, 0 errors.** Was RUNNING as of 08-24 03:00. Sat in `JobHeldUser` from submission (08-23 17:59) until released on 08-24 -- it had never started, so until it lands every root-agreement / whole-trajectory figure is still the stale 07-25 -> 08-10 data. 24/48 done (all 2-gene, ~55s each); the 24 3-gene tasks are the slow half but historical runs top out at ~28 min, comfortably inside the 4h wall, so this array needs no resume logic. |

### 2026-08-24: the queue was holding, not running

`2206182` never ran because it was `JobHeldUser` with `Priority=0` -- along with
22 other jobs of this user's (`mnca_*`, `tex_*`, `canary_mem12`). All 23 were
released on 08-24; `JobHeldUser` is now zero. The 3 `nlp_rq7_*` jobs are
`JobHeldAdmin` and cannot be released from a user account.

**Check for this first when a submitted job shows no progress.** `squeue` shows
it as PENDING, which reads like "waiting for nodes" -- the distinguishing signal
is `Reason=JobHeldUser` and `Priority=0` in `scontrol show job <id>`.


### Verified: the 3-gene dataset cache IS the corrected data

Checked before submitting, because regenerating logs against stale inputs would
just produce new wrong numbers. `ground-up-experiments/step9_gnn_3gene/results/cache`
holds 36 pickles, and the mtimes line up exactly with the bug:

| Regime | Files | Dated |
|---|---|---|
| HighHigh, MixedA, MixedB | 6 each | **08-15** (regenerated -- these are the three regimes whose GeneA/GeneB frequencies were wrong) |
| LowHigh, MediumEven, LowLow | 6 each | 06-30 / 07-01 (untouched -- these were always correct) |

So the fourway rerun reads corrected 3-gene data. `log_*_fourway.py` also calls
`genetic_dp.policy.baselines.myopic_greedy` directly, so its myopic rows are
canonical by construction.

### Resume logic (added 2026-08-23)

Every partition on this cluster (`general`, `peanut-cpu`, `threedle-*`) has a
hard **4h MaxTime** -- there is no longer limit to ask for. The leave-one-out
GNN folds do not fit in 4h, so they can only finish by checkpointing and
requeueing across several 4h slots. `submit_e6_split_experiment.sh` +
`incremental_experiments/e6_split_experiment.py` do that, and
`submit_overfit_probes.sh` + `incremental_experiments/overfit_probes_2gene.py`
use the same pattern:

- `#SBATCH --signal=B:USR1@300` fires 5 min before the wall clock; the trap
  calls `scontrol requeue` so the task restarts instead of dying.
- Training resumes from `checkpoint.pt` (written every epoch, already existed).
- TRAIN-side eval resumes from `eval_train_partial.json` (flushed per config).
- A finished `(exp,kind,seed)` exits in <1s via a `results_e6.json` skip guard,
  so **the whole 0-29 array can be resubmitted at any time and only does the
  work that is actually missing**. `--force` redoes one from scratch.

### 2206182 result: every 3-gene action-agreement figure moved (2026-08-24)

Standard families (TRAIN = Trio + Nuclear, TEST = ThreeGeneration), all 12
configs (6 regimes x 2 presets), averaged over seeds 0/1/2. Old = the committed
07-25 -> 08-10 files, New = regenerated against corrected data + retrained
checkpoints.

**2-gene: bit-identical at all 8 rungs**, root and whole-trajectory alike. That
is now the *fourth* independent confirmation that 2-gene was untouched by the
3-gene allele-frequency fix (the others: myopic ratio2, the 2-gene Extended
rerun, and the E0-E9 retrain).

**3-gene: changed at every rung.** Whole-trajectory agreement vs DP:

| Rung | GNN old -> new | MLP old -> new |
|---|---|---|
| E0 | 0.486 -> **0.314** | 0.443 -> 0.517 |
| E1 | 0.462 -> 0.527 | 0.557 -> 0.502 |
| E2 | 0.681 -> **0.729** | 0.533 -> 0.725 |
| E4 | 0.633 -> 0.705 | 0.533 -> 0.725 |
| E5 | 0.590 -> 0.623 | 0.533 -> 0.725 |
| E6 | 0.567 -> 0.628 | 0.710 -> **0.565** |
| E7 | 0.581 -> 0.623 | 0.533 -> 0.725 |
| E9 | 0.395 -> 0.522 | 0.533 -> 0.725 |

Myopic moved 0.557 -> 0.565, consistent with its ratio2 going 0.1052 -> 0.1090.
E0's GNN is the big loser (0.486 -> 0.314); E6's MLP drops hard (0.710 -> 0.565,
and root 2.0/12 -> 0.3/12). E2/E4/E5/E7/E9 share one MLP number because they all
reuse the E2 CE MLP checkpoint -- expected, and true in the old data too.

**Every root-agreement and whole-trajectory figure in the 3-gene half of the PI
briefing is therefore stale and needs replacing from these files.**

### 3-gene Extended does not fit in one slot (2026-08-24)

All nine 3-gene Extended tasks in jobs 2206213 / 2206261 hit the 4h wall and
produced **nothing**. Two compounding causes:

1. Each of the 12 configs solves exact DP over **9,190,992 states from
   scratch** -- there is no cached pickle for Extended at 3 genes, only for
   Trio / Nuclear / ThreeGeneration. The logs show each task finished only
   4-6 of 12 configs before being killed.
2. `log_extended_fourway.py` wrote its JSON **only after all 12 configs
   finished**. So a task that solved 5 configs and then hit the wall threw all
   five away. That is the part that made the timeout total rather than partial.

Fixed by giving the whole-seed path the same resume machinery the split
experiments already use:

- Every config is flushed to `extended/{genes}gene/seed{N}/partial{suffix}/
  {regime}_{preset}.json` the instant it is solved, and a resumed task skips
  what it already has. The log is appended to, not truncated.
- `submit_extended_fourway.sh` and `submit_extended_fourway_e6e7_rerun.sh` now
  carry `--signal=B:USR1@300` + a `scontrol requeue` trap, same as
  `submit_e6_split_experiment.sh`.
- **Partials are keyed to the SLURM job id.** `scontrol requeue` preserves the
  job id, a fresh `sbatch` does not -- so a requeue resumes, but a new
  submission *wipes* the partials rather than inheriting them. Without that
  guard a resume could silently fold a config solved against superseded data or
  superseded checkpoints into a fresh aggregate, which is exactly the failure
  this entire regeneration pass exists to undo. `--force` re-solves from
  scratch.

**Verified end-to-end on a compute node (jobs 2207508 / 2207509, 2-gene so it
is cheap, 2026-08-24).** The test kills a run mid-flight, restarts it under the
same job id, then simulates a fresh submission:

| Property | Result |
|---|---|
| Killed at the simulated wall, partials survive | **3/12 configs** checkpointed (LowHigh_Base, LowHigh_Aggressive, MediumEven_Base); final JSON correctly still the old 08-23 one. Under the old code this was **0**. |
| Requeue resumes rather than restarts | slot 2 logged `[resume] 3/12 configs already solved in an earlier slot of this job`, skipped those three, solved the remaining nine, wrote the final JSON |
| A new submission does not inherit partials | 12 partials on disk, **0 inherited** -- the job-id key held |
| **Resume does not change the answer** | the resumed seed1 output is **identical** to the known-good pre-test copy (whole-trajectory, root-only, per-config and ratio2 blocks all equal). seed0, run clean, likewise identical. |

One expected side effect: a resumed run's `.log` legitimately contains the
earlier slot's config sections plus a header per slot, so a resumed log has
more than 12 config blocks. The `.json` is unaffected and is what every
analysis script reads -- but do not count config blocks in a resumed `.log`.

Note the per-config mode (`--regime X --preset Y`, driven by
`submit_extended_3gene_perconfig{,_e6,_e7}.sh`, 36 tasks each) was always
immune to this -- one config per task, own JSON written immediately. It remains
the alternative if the resume path is ever in doubt.

**Resubmitted and COMPLETE** as 2207525 (e4) / 2207526 (e6+e7), 9 tasks,
**9/9 tasks, 108/108 configs, 0 errors**, finished 2026-08-24 12:33.

The resume machinery fired in production exactly as designed: at 08:05 all nine
tasks hit the 4h wall, the USR1 trap requeued them, and each restart logged
`[resume] N/12 configs already solved` and skipped that work. Three tasks needed
two requeues (`Restarts=2`). Under the old code all nine would have produced
nothing, a second time.

Result -- 3-gene Extended (OOD, never trained on), 12 configs, seeds 0/1/2,
ratio2 (lower = better):

| Variant | GNN old -> new | MLP old -> new | myopic old -> new |
|---|---|---|---|
| E4 | 0.0656 -> 0.0771 | 0.1406 -> 0.2366 | 0.1107 -> 0.1146 |
| E6 | 0.0500 -> 0.0730 | 0.2273 -> **0.8956** | 0.1107 -> 0.1146 |
| E7 | 0.0740 -> 0.0689 | 0.1406 -> 0.2366 | 0.1107 -> 0.1146 |

Two findings. (1) **GNN-Q beats myopic out-of-distribution at all three
variants** (0.069-0.077 vs 0.115). (2) **E6's MLP collapsed to 0.896**, i.e.
barely better than testing nobody. That is corroborated by two independent
measurements on the corrected data -- E6 MLP standard-family ratio2 went
0.214 -> 0.821, and its whole-trajectory DP agreement dropped 0.710 -> 0.565.
E6 remains the strongest rung on the GNN side; its MLP is now the worst of any
variant.


### E0-E9 briefing artifact fully updated (2026-08-24)

`claude.ai/code/artifact/c1d8a29e-ab37-4d88-aa83-9493714f72f3` -- every 3-gene
figure replaced from the regenerated logs. 2-gene tables were left byte-identical
on purpose: the regeneration re-derived them and they came back the same, so
there was nothing to change and no reason to disturb the hand-placed
best/worst highlighting.

Replaced: all 7 rung sections' 3-gene root + whole-trajectory tables, the master
table, the per-config breakdown, and the whole Extended/OOD section. Denominators
moved with the data -- 3-gene whole-traj **210 -> 207**, Extended **246 -> 243** --
because those trajectories are walked along myopic's own path and myopic's values
shifted with the corrected data.

**Trap worth remembering:** the per-rung "vs." delta column is measured against
the PREVIOUS rung, not against E0 -- E1 vs E0, E2 vs E1, E4 vs E2, and E6/E7/E9
all vs E4. A first pass recomputed every delta against E0 and silently corrupted
the 2-gene columns (which are otherwise correct); caught by diffing the 2-gene
tables against the original and requiring zero changes.

Five directional claims were inverted by the new data and were rewritten in place:

| Rung | Claim that no longer held |
|---|---|
| E4 | 3-gene root read as "flat, tied with E2 at 23/36" -- it is a collapse, 80.6% -> 30.6% |
| E6 | "MLP root more than doubles in both gene counts" -- true only at 2-gene; at 3-gene it collapses 83.3% -> 11.1% |
| E7 | "the single worst root number anywhere on this page" -- no longer; E0's own 11.1% is |
| E9 | "a comprehensive loss to every simpler alternative" -- now beats E4 and E7 on 3-gene root and ratio2; still decisively worst on whole-trajectory |
| Extended | "MLP-Q beats every GNN-Q variant on both root and whole-trajectory" -- still wins root (52.8%), now loses whole-trajectory to all three (61.3% vs E4 67.9%) |

Note the artifact is shared by link and **viewers see a pinned earlier version**
until the share pin is moved -- updating the page does not update what a
link-holder sees.

## Finished

| Job ID | Name | Purpose | Result |
|---|---|---|---|
| 2206212 | verify_myopic_TRUE | Myopic ratio2 reference, 24 tasks | **COMPLETE, 24/24.** 3-gene 0.1052 -> 0.1090; 2-gene 0.2303 bit-identical. Already folded into the artifact and commit 07f979d. |
| 2206213 | extended_fourway (e4) | Extended/OOD four-way, variant e4, {2,3}-gene x 3 seeds, 6 tasks | **PARTIAL: 2-gene 3/3 COMPLETE, 3-gene 3/3 TIMEOUT** at the 4h wall. See "3-gene Extended does not fit in one slot" below. The 2-gene results came back bit-identical to the committed ones apart from a cosmetic `variant` label -- a third independent confirmation that 2-gene was untouched by the 3-gene fix, alongside the myopic 2-gene result. |
| 2206261 | ext_e6e7_rerun | Extended/OOD four-way, variants E6+E7, {2,3}-gene x 3 seeds, 12 tasks | **PARTIAL: 2-gene 6/6 COMPLETE, 3-gene 6/6 TIMEOUT** at the 4h wall. Same cause. So all 2-gene Extended rows are now post-correction; all **3-gene** Extended rows (e4/e6/e7) are still the stale 07-26 / 08-09 data. |
| 2204396 (artifact) | — | 3-gene ratio2 figures in the PI briefing artifact replaced from the corrected retrain | DONE 2026-08-23. Per-rung tables, master table, all 12 per-config rows, and every vs-previous-rung delta. Four directional claims flipped (E1 batching no longer a uniform win; E4 bidirectional MP now regresses at 3-gene; E7/E9 now better than E4; E6 MLP 0.214 -> 0.821) and the prose was rewritten to match. 2-gene verified unchanged and correct. |
| 2204396 | ovfit_probe | Overfit probes: curve / randlabel / configholdout x GNN+MLP x 3 seeds, 500 epochs, on the FAIR rotate split (TRAIN=Trio+ThreeGeneration, TEST=Nuclear) | **COMPLETE, 18/18, 0 errors.** Three results: (1) real but shallow overfitting -- GNN val loss bottoms at ep150-250 then drifts up only ~2-3%; MLP shows none. (2) Neither model can memorize shuffled targets -- shuffled-target R2 sits at mean-predictor level (GNN -0.16 to -0.44), and total loss is HIGHER under shuffling. (3) Config-level and family-level generalization gaps are the same size (~1.7x vs ~1.6x), so neither config memorization nor a specific topology-transfer failure is supported. Analysis: `analyze_overfit_probes.py`. |
| 2204343 | ovfit_smoke | 2-epoch smoke of all three overfit probes on the real partition | COMPLETE, 4/4, exit 0. All probes emitted their expected fields; `configholdout` produced all three ratio2 numbers. seed99 output dirs deleted so they do not pollute the results tree. |
| 2202164 | e6_split | E6 2-gene split experiments: rotate/add/loo_trio/loo_nuclear/loo_threegen x GNN+MLP x 3 seeds (30 tasks) | **COMPLETE, 30/30, 0 errors.** The last LOO GNN fold beat the 4h wall by minutes, so the resume machinery never had to fire. Headline: the standard-split "overfitting" is mostly a split artifact -- swapping Nuclear -> ThreeGeneration in TRAIN (same 2 families, same 24 configs) raises GNN TRAIN error ~6x and collapses the gap from 8.3x/3.1x (E0/E6) to 1.3x/1.1x. Confirmed on both E0 and E6, so not rung-specific. |
| 2204292 | e6_split (resume) | Resubmit of the 0-29 array with resume logic, `--dependency=afterany:2202164` | CANCELLED (scancel) -- 2202164 finished all 30 tasks on its own, so every task would have been a <1s no-op. Cancelled rather than let it hold queue slots under `QOSMaxJobsPerUserLimit`. The resume machinery it added stays in the scripts. |
| 2204260 | overfit2g_eval | Standard-split (TRAIN=Trio+Nuclear, TEST=ThreeGeneration) TRAIN-vs-TEST overfit check, 2-gene, E0/E1/E2/E4/E6, 3 seeds | COMPLETE, 3/3, 24s. Result: the overfit gap is a **GNN** phenomenon, not an MLP one -- every GNN rung degrades 3-16x from TRAIN to TEST, every MLP rung is ~1x (E1/E2 MLP even slightly negative). E6 has the SMALLEST GNN gap (3.1x vs E0's 8.3x). |
| 2199861-2199871 | gnn_q_seeds / mlp_q_seeds / e1_gnn_train / e1_mlp_train / gnn_q_ce_train / mlp_q_ce_train / e4_gnn_bidir_train / e5_gnn_rounds3_train / e6_sumpool_train / e7_neighborpool_train / e9_combopool_train | Retrain E0-E9 (3-gene) on corrected allele-frequency data | ALL COMPLETE, 108/108 sub-tasks, 0 errors. E6 is the best rung (avg TEST ratio2 = 0.056 vs E0's 0.058, 3-seed avg). |
| 2201157 | (unnamed) | E0 2-gene split experiments: rotate + add x GNN+MLP x 3 seeds (12 tasks) | COMPLETE, 12/12, 0 errors. Not used further -- E0 isn't the rung that matters, superseded by the E6 version (2202164). |
| 2202143 | e6_smoke | Diagnostic smoke test for e6_split_experiment.py on the real peanut-cpu partition (after the same script looked pathologically slow on the shared login node) | COMPLETE, 38s total. Confirmed the "slowness" was login-node CPU throttling, not a bug -- unblocked the real 2202164 submission. |

## ADP ground truth (separate track, supplementary comparison only -- not part of the E0-E9 GNN/MLP investigation)

| Job ID | Name | Purpose | Result |
|---|---|---|---|
| (original 96-config array, ~job 2182xxx) | adp_ground_truth | Kanix's real ADP solver (dual_dp) on all 96 configs | 72/96 done: all 2-gene (48/48), 3-gene Trio+Nuclear (24/24). |
| 2199700 | adp_gt_fix_3g | Retry 3-gene ThreeGeneration (12 configs) after raising memory | **ALL 12 FAILED** -- not the OOM/pickle bug, a different real error: `RuntimeError: Row generation did not converge ... state_truncation` from Kanix's own `dual_dp.py` (only scanned 110k/1,054,528 states before giving up). This is an algorithmic limit of his ADP solver at this state-space size, not an infra bug -- have not touched his convergence logic, flagged not fixed. |
| 2201156 | smoke_adp_patch | Smoke test of the belief-snapshot pickle-fix on 3-gene Extended_LowHigh_Base | **TIMED OUT** (not completed) -- Extended (9.19M states) needs a longer time limit / bigger job than a smoke test allows. 3-gene Extended ADP remains 0/12, unresolved. |

### Bottom line on ADP: 3-gene ThreeGeneration and Extended (24/96 configs) are NOT done and need a real decision (raise convergence tolerance? accept partial coverage? different approach?) before pursuing further -- not silently retried.
