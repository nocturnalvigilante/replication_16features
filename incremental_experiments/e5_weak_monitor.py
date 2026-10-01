"""Watchdog for the E5-weak pipeline. submit_e5_weak_monitor.sh runs one pass
every 5 minutes, all night, inside its own SLURM job that keeps a successor queued.

Output-driven: what counts as done comes from disk (results/e5_weak/manifest.json),
never from SLURM dependencies, so a timed-out, killed, preempted or OOM'd job
cannot strand anything. Each pass:
  1. check outputs: an eval JSON that does not parse is deleted (it gets redone)
  2. read squeue: which units already have a queued/running job; release held jobs
  3. for every unit not done and not covered by a job:
       - its last log has an AssertionError -> a correctness guard failed:
         ALERT and never resubmit (that needs a human)
       - it has been submitted MAX_ATTEMPTS times -> ALERT, stop
       - else submit it. train: GPU, resumes from its checkpoint (CPU after 2
         GPU failures). eval: only once all 3 seeds of that gene count exist;
         memory doubles after OUT_OF_MEMORY.
  4. when a gene count's 420 eval files all exist: summarize it -> FINAL_SUMMARY_g{G}.txt
     when both do: write DONE (the monitor chain then stops)
Writes STATUS.txt (one screen), appends monitor.log, state in monitor_state.json.

    python e5_weak_monitor.py --once [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path("/net/projects/ranalab/rajhansini/replication_16features")
R = ROOT / "incremental_experiments" / "results" / "e5_weak"
PY = "/net/projects/ranalab/rajhansini/conda_envs/genetic-rl/bin/python"
MAN = json.loads((R / "manifest.json").read_text())
STATE_F = R / "monitor_state.json"
MAX_ATTEMPTS = 4
# arrays submitted before the monitor existed: base job id -> (stage, genes)
LEGACY = {"2277140": ("eval", "2"), "2277578": ("train", "3")}
CPU_PARTS = "threedle-own,threedle-contrib,general,peanut-cpu"
GPU_PARTS = "threedle-contrib,threedle-own,general"


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=ROOT)


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg):
    line = f"[{now()}] {msg}"
    print(line, flush=True)
    with open(R / "monitor.log", "a") as f:
        f.write(line + "\n")


def units():
    for G, seeds in MAN["train"].items():
        for s, path in seeds.items():
            yield ("train", G, s), [path]
    for stage in ("eval", "traj"):
        for G, us in MAN.get(stage, {}).items():
            for u, spec in us.items():
                yield (stage, G, u), spec["files"]


def check_outputs(st):
    """A present eval JSON must parse; a broken one is deleted so it gets redone."""
    ok = set(st.setdefault("valid", []))
    for (stage, _, _), files in units():
        if stage == "train":
            continue
        for f in files:
            p = R / f
            if f in ok or not p.exists():
                continue
            try:
                json.loads(p.read_text())
                ok.add(f)
            except Exception as exc:
                log(f"CORRUPT {f} ({exc!r}) -- deleted, will be redone")
                p.unlink()
    st["valid"] = sorted(ok)


def done(files):
    return all((R / f).exists() for f in files)


def covered():
    """{(stage, G, idx): (jobid, state, reason)} for every queued/running job; None if squeue failed."""
    out = sh('squeue -h -u "$USER" -r -o "%i|%j|%T|%r"')
    if out.returncode != 0:
        log(f"squeue failed ({out.stderr.strip()}) -- skipping this pass")
        return None
    cov = {}
    for line in out.stdout.splitlines():
        jid, name, state, reason = line.split("|", 3)
        m = re.fullmatch(r"e5w_(train|eval|traj)(\d)_(\d+)", name)
        if m:
            cov[(m.group(1), m.group(2), m.group(3))] = (jid, state, reason)
        elif name == "e5_weak" and "_" in jid:
            base, idx = jid.split("_", 1)
            if base in LEGACY and idx.isdigit():
                cov[(*LEGACY[base], idx)] = (jid, state, reason)
    return cov


def last_end(jid):
    """Final state of a finished array task, e.g. ('OUT_OF_MEMORY', log path)."""
    out = sh(f"sacct -j {jid} -X -n -P -o JobID,State")
    state = "UNKNOWN"
    for line in out.stdout.splitlines():
        j, s = line.split("|", 1)
        if j == jid:
            state = s.split()[0] if s else "UNKNOWN"
    base, idx = jid.split("_", 1)
    return state, R / f"slurm_{base}_task{idx}.log"


def use_gpu(unit, info):
    return unit[0] == "train" and info.get("gpu_failures", 0) < 2


def mem_mb(unit, info):
    stage, G, idx = unit
    if info.get("mem"):
        return info["mem"]
    if stage == "train":
        return 49152 if use_gpu(unit, info) else 32768
    return 32768 if G == "2" else (65536 if int(idx) < 14 else 307200)   # Extended 3-gene: 9.2M states, 198 GB measured


def resources(unit, info):
    stage, G, idx = unit
    mem = mem_mb(unit, info)
    if use_gpu(unit, info):
        return f"--gres=gpu:1 --partition={GPU_PARTS} --mem={mem} --time=04:00:00", "--device cuda"
    wall = "02:00:00" if (stage != "train" and G == "2") else "04:00:00"
    cpus = "--cpus-per-task=8 " if (stage == "eval" and G == "3") else ""     # batched Q-hat uses threads
    return f"{cpus}--partition={CPU_PARTS} --mem={mem} --time={wall}", ""


def submit(unit, info, dry):
    stage, G, idx = unit
    res, extra = resources(unit, info)
    cmd = (f"sbatch --parsable --job-name=e5w_{stage}{G}_{idx} --array={idx} {res} "
           f"submit_e5_weak.sh {stage} {G} {extra}").strip()
    if dry:
        log(f"DRY-RUN would submit: {cmd}")
        return None
    out = sh(cmd)
    if out.returncode != 0:
        log(f"SUBMIT FAILED {unit}: {out.stderr.strip()}")
        return None
    jid = f"{out.stdout.strip()}_{idx}"
    log(f"submitted {stage} g{G} #{idx} -> {jid}  ({res} {extra})")
    return jid


def train_epoch(G, s):
    f = R / f"g{G}" / "gnn" / f"seed{s}" / "run.log"
    if not f.exists():
        return "-"
    ep = re.findall(r"epoch\s+(\d+)", f.read_text())
    return ep[-1] if ep else "loading"


def one_pass(dry=False):
    st = json.loads(STATE_F.read_text()) if STATE_F.exists() else {}
    st.setdefault("units", {}); st.setdefault("alerts", [])
    check_outputs(st)
    cov = covered()
    if cov is None:
        return
    for unit, (jid, state, reason) in cov.items():
        if state == "PENDING" and re.search(r"held|Held|launch failed", reason):
            log(f"releasing held job {jid} ({reason})"); dry or sh(f"scontrol release {jid}")
        if "DependencyNeverSatisfied" in reason:
            log(f"cancelling {jid}: dependency can never be satisfied; will resubmit"); dry or sh(f"scancel {jid}")

    models_ready = {G: all((R / p).exists() for p in seeds.values()) for G, seeds in MAN["train"].items()}
    for unit, files in units():
        stage, G, idx = unit
        k = "/".join(unit)
        if k not in st["units"]:
            legacy = [f"{b}_{idx}" for b, sg in LEGACY.items() if sg == (stage, G)]
            st["units"][k] = {"jobs": legacy, "attempts": 0}
        info = st["units"][k]
        if unit in cov:
            jid = cov[unit][0]
            if jid not in info["jobs"]:
                info["jobs"].append(jid)
            continue
        if done(files) or info.get("blocked"):
            continue
        if stage in ("eval", "traj") and not models_ready[G]:
            continue
        if info["jobs"]:                                   # it ran before and did not finish
            state, logf = last_end(info["jobs"][-1])
            text = logf.read_text(errors="ignore") if logf.exists() else ""
            if "AssertionError" in text:
                info["blocked"] = True
                msg = f"GUARD FAILED {k} (job {info['jobs'][-1]}): see {logf.name} -- not resubmitting"
                st["alerts"].append(f"{now()} {msg}"); log(msg)
                continue
            if state == "OUT_OF_MEMORY":
                info["mem"] = min(mem_mb(unit, info) * 2, 900000)
            if use_gpu(unit, info) and state in ("FAILED", "NODE_FAIL"):
                info["gpu_failures"] = info.get("gpu_failures", 0) + 1
            log(f"{k}: last job {info['jobs'][-1]} ended {state}, output incomplete -> resubmitting")
        if info["attempts"] >= MAX_ATTEMPTS:
            if not info.get("gave_up"):
                info["gave_up"] = True
                msg = f"GAVE UP {k} after {MAX_ATTEMPTS} submissions -- needs a human"
                st["alerts"].append(f"{now()} {msg}"); log(msg)
            continue
        jid = submit(unit, info, dry)
        if jid:
            info["jobs"].append(jid); info["attempts"] += 1

    # summaries
    status_eval = {}
    for G, us in MAN["eval"].items():
        n = sum((R / f).exists() for spec in us.values() for f in spec["files"])
        status_eval[G] = n
        summ = R / f"FINAL_SUMMARY_g{G}.txt"
        if n == 420 and not summ.exists() and not dry:
            out = sh(f"OMP_NUM_THREADS=1 {PY} incremental_experiments/e5_weak.py --stage summarize --genes {G}")
            summ.write_text(out.stdout + ("\n" + out.stderr[-2000:] if out.returncode else ""))
            log(f"g{G}: all 420 test configs scored -> {summ.name}")
    status_traj = {G: sum((R / f).exists() for spec in us.values() for f in spec["files"]) for G, us in MAN.get("traj", {}).items()}
    if all(v == 420 for v in status_eval.values()) and all(v == 420 for v in status_traj.values()) and not dry:
        (R / "DONE").write_text(now() + "\n"); log("ALL DONE")

    active = sum(1 for v in cov.values() if v[1] == "RUNNING"), sum(1 for v in cov.values() if v[1] == "PENDING")
    lines = [f"E5-weak status  {now()}",
             "train epochs (of 500): " + "  ".join(
                 f"g{G} s{s}: {'done' if (R / p).exists() else train_epoch(G, s)}"
                 for G, seeds in MAN["train"].items() for s, p in seeds.items()),
             "eval configs scored:   " + "  ".join(f"g{G} {n}/420" for G, n in status_eval.items()),
             "trajectories done:     " + "  ".join(f"g{G} {n}/420" for G, n in status_traj.items()),
             f"jobs: {active[0]} running, {active[1]} pending",
             "alerts: " + ("none" if not st["alerts"] else ""), *[f"  {a}" for a in st["alerts"]]]
    if (R / "DONE").exists():
        lines.insert(1, "ALL DONE")
    (R / "STATUS.txt").write_text("\n".join(lines) + "\n")
    if not dry:
        STATE_F.write_text(json.dumps(st, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    one_pass(dry=a.dry_run)
