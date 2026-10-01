#!/bin/bash
#SBATCH --job-name=myo_weak3
#SBATCH --output=incremental_experiments/results/slurm_myoweak3_%A_task%a.log
#SBATCH --time=04:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --partition=general
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@300

# 3-gene myopic weakness sweep, ThreeGeneration (1,054,528 states at 3 genes).
# Kanix's code only, via scripts/search_multigene_myopic_vs_stop_3gene.py (his
# script + GeneC). Variant is the first argument; set --array to match:
#
#   sbatch --array=0-4   submit_myopic_weakness_3gene.sh sanity    # GeneC ~0 must reproduce 2-gene
#   sbatch --array=0-209 submit_myopic_weakness_3gene.sh pattern   # main grid
#   sbatch --array=0-83  submit_myopic_weakness_3gene.sh cloneA    # robustness, fixed_cost >= 0.02
#   sbatch --array=0-83  submit_myopic_weakness_3gene.sh cloneB
#
# One row per task, own JSON, skip guard: safe to resubmit. USR1 300s before
# the wall requeues the task (the row restarts from scratch).

VARIANT="$1"
FAMILY="${2:-ThreeGeneration}"   # Extended: pass --mem=256G (9,190,992 states)
cd /net/projects/ranalab/rajhansini/replication_16features
echo "task=$SLURM_ARRAY_TASK_ID variant=$VARIANT family=$FAMILY host=$(hostname) start=$(date -Is)"

requeue_handler() {
    echo "[SIGNAL] $(date -Is) USR1 (300s to wall) -- requeueing task $SLURM_ARRAY_TASK_ID"
    scontrol requeue "${SLURM_JOB_ID}"
}
trap requeue_handler USR1

# Third arg "group": task id = (preset, allele) group 0-13, one belief per task
# (sbatch --array=0-13 ... pattern Extended group). Default: task id = row.
UNIT=$([ "$3" = "group" ] && echo "--group" || echo "--row")
/net/projects/ranalab/rajhansini/conda_envs/genetic-rl/bin/python -u \
  incremental_experiments/myopic_weakness_sweep_3gene.py --variant "$VARIANT" --family "$FAMILY" $UNIT "$SLURM_ARRAY_TASK_ID" &
CHILD=$!
wait "$CHILD"
echo "end=$(date -Is)"
